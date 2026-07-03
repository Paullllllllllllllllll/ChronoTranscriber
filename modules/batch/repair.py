"""Repair operations for failed or placeholder transcriptions.

This module encapsulates the interactive repair workflow previously embedded
in ``main/repair_transcriptions.py``. It keeps a clean separation of
orchestration logic from the CLI entry point so it can be imported and
tested.

Merged from the former ``modules/operations/repair/utils.py`` and
``modules/operations/repair/run.py`` during the deep-module refactor so
that all batch repair code lives in a single location.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from modules.config.service import get_config_service
from modules.infra.concurrency import run_concurrent_transcription_tasks
from modules.infra.logger import setup_logger
from modules.infra.token_budget import (
    check_and_wait_for_token_limit,
    get_token_tracker,
)
from modules.llm import open_transcriber
from modules.llm.openai_sdk_utils import coerce_file_id, sdk_to_dict
from modules.llm.response_parsing import (
    detect_transcription_cause,
    extract_transcribed_text,
    format_page_line,
)
from modules.ui import (
    NavigationAction,
    PromptStyle,
    print_error,
    print_header,
    print_info,
    print_separator,
    print_success,
    print_warning,
    prompt_select,
    prompt_text,
    prompt_yes_no,
    ui_print,
)

logger = setup_logger(__name__)


def _parse_req_index(custom_id: str) -> int | None:
    """Parse the 0-based index from a ``req-<n>`` custom_id, or None.

    The batch request builder assigns ``req-<n>`` where ``n`` is the 1-based
    position in the submitted image list, which equals the position in the
    repair ``targets`` list. Parsing it lets repair correlate results to
    targets by identity rather than by fragile positional order (B7).
    """
    if custom_id.startswith("req-"):
        try:
            idx = int(custom_id.split("-", 1)[1]) - 1
        except (ValueError, IndexError):
            return None
        return idx if idx >= 0 else None
    return None


def _correlate_repair_targets(
    targets: list[Any],
    metadata_records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Map each submitted custom_id to its target by the custom_id's index.

    Correlation is by the index encoded in ``req-<n>`` (position in the
    submitted image list == position in ``targets``), NOT by positional zip: a
    zip shifts every pair after any image whose metadata record was skipped
    because it failed to encode, patching repaired text onto the wrong page
    (B7). Returns (order_index, line_index, image_name) maps keyed by custom_id.
    """
    order_by_custom: dict[str, int] = {}
    line_index_by_custom: dict[str, int] = {}
    image_name_by_custom: dict[str, str] = {}

    if len(metadata_records) != len(targets):
        # Tripwire: some images did not produce a request (encode failure).
        logger.warning(
            "Repair correlation: %d metadata record(s) for %d target(s); "
            "correlating by custom_id index.",
            len(metadata_records),
            len(targets),
        )

    for rec in metadata_records:
        br = rec.get("batch_request", {})
        cid = br.get("custom_id")
        if not isinstance(cid, str):
            continue
        idx = _parse_req_index(cid)
        if idx is None or not (0 <= idx < len(targets)):
            logger.warning(
                "Repair correlation: custom_id %r has no matching target; "
                "skipping to avoid mis-patching a page.",
                cid,
            )
            continue
        t = targets[idx]
        order_by_custom[cid] = t.order_index
        line_index_by_custom[cid] = t.line_index
        image_name_by_custom[cid] = t.image_name

    return order_by_custom, line_index_by_custom, image_name_by_custom


# ---------------------------------------------------------------------------
# Utilities (formerly modules/operations/repair/utils.py)
# ---------------------------------------------------------------------------


# Regex patterns used to detect failure placeholders in final text
FAILURE_PATTERNS = [
    # Allow optional leading "Page N:" or "Image name:" prefixes before the bracket
    re.compile(r"^(?:[^\[]+?:\s*)?\[\s*transcription\s+error.*\]$", re.IGNORECASE),
    re.compile(
        r"^(?:[^\[]+?:\s*)?\[\s*Transcription\s+not\s+possible.*\]$", re.IGNORECASE
    ),
]
NO_TEXT_PATTERN = re.compile(
    r"^(?:[^\[]+?:\s*)?\[\s*No\s+transcribable\s+text.*\]$", re.IGNORECASE
)


@dataclass
class ImageEntry:
    order_index: int
    image_name: str
    pre_processed_image: str | None
    custom_id: str | None
    page_number: int | None = None
    source_file: str | None = None
    page_index: int | None = None


@dataclass
class Job:
    parent_folder: Path
    identifier: str
    final_txt_path: Path
    temp_jsonl_path: Path | None
    kind: str  # "PDF" or "Images"


def extract_image_name_from_failure_line(line: str) -> str | None:
    """
    Extract the image file name from a placeholder line such as:
      - "0089_pre_processed.jpg: [transcription error]"       (prefix format)
      - "page_12.jpg: [Transcription not possible]"           (prefix format)
      - "[Transcription not possible: IMG_0001.png]"          (inline format)
      - "[No transcribable text: page_12.jpg]"                (inline format)
      - "[transcription error: scan_03.png; status 400; code invalid_image]"
    Returns the extracted image name if found, else None.
    """
    stripped = line.strip()

    # Prefix format: "image_name.ext: [placeholder]"
    prefix_m = re.match(
        r"^(.+?\.(?:jpg|jpeg|png|tif|tiff|jp2|bmp|webp))\s*:\s*\[",
        stripped,
        re.IGNORECASE,
    )
    if prefix_m:
        return prefix_m.group(1).strip()

    # Inline format: "[placeholder: image_name; ...]"
    core = re.sub(r"^[^\[]*?:\s*", "", stripped)
    pattern = re.compile(
        r"^\[(?:transcription error|Transcription not possible"
        r"|No transcribable text):\s*([^;]+)",
        re.IGNORECASE,
    )
    m = pattern.match(core)
    if m:
        name = m.group(1).strip()
        name = name.rstrip("]\")' ")
        name = name.lstrip("'\" ")
        return name
    return None


def is_failure_line(line: str) -> bool:
    return any(pat.match(line.strip()) for pat in FAILURE_PATTERNS)


def collect_image_entries_from_jsonl(temp_jsonl_path: Path | None) -> list[ImageEntry]:
    """
    Parse local transcription JSONL to reconstruct page ordering and metadata.
    Returns a list of ImageEntry sorted by order_index.
    """
    entries: dict[int, ImageEntry] = {}
    run_source_file: str | None = None
    if temp_jsonl_path is None or not temp_jsonl_path.exists():
        return []

    try:
        with temp_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # Run-level provenance (streaming pipeline): remember the
                # source file as a fallback for entries without one.
                if "file_provenance" in obj and isinstance(
                    obj["file_provenance"], dict
                ):
                    src = obj["file_provenance"].get("source_file")
                    if src:
                        run_source_file = str(src)
                    continue

                # Preferred: image_metadata
                if "image_metadata" in obj and isinstance(obj["image_metadata"], dict):
                    meta = obj["image_metadata"]
                    oi = meta.get("order_index")
                    if isinstance(oi, int):
                        entries[oi] = ImageEntry(
                            order_index=oi,
                            image_name=str(meta.get("image_name") or "").strip(),
                            pre_processed_image=str(
                                meta.get("pre_processed_image") or ""
                            ).strip()
                            or None,
                            custom_id=str(meta.get("custom_id") or "").strip() or None,
                            page_number=meta.get("page_number"),
                            source_file=str(meta.get("source_file") or "").strip()
                            or None,
                            page_index=meta.get("page_index")
                            if isinstance(meta.get("page_index"), int)
                            else None,
                        )
                    continue

                # Fallback: batch_request lines
                if "batch_request" in obj and isinstance(obj["batch_request"], dict):
                    br = obj["batch_request"]
                    custom_id = str(br.get("custom_id") or "").strip() or None
                    ii = br.get("image_info") or {}
                    oi = ii.get("order_index")
                    if isinstance(oi, int):
                        entries[oi] = ImageEntry(
                            order_index=oi,
                            image_name=str(ii.get("image_name") or "").strip(),
                            pre_processed_image=None,
                            custom_id=custom_id,
                            page_number=ii.get("page_number"),
                        )
                    continue

                # Synchronous GPT JSONL records
                if (
                    "method" in obj
                    and obj.get("method") == "gpt"
                    and "order_index" in obj
                ):
                    try:
                        oi = int(obj.get("order_index"))
                        entries.setdefault(
                            oi,
                            ImageEntry(
                                order_index=oi,
                                image_name=str(obj.get("image_name") or "").strip(),
                                pre_processed_image=str(
                                    obj.get("pre_processed_image") or ""
                                ).strip()
                                or None,
                                custom_id=None,
                                page_number=None,
                                source_file=str(obj.get("source_file") or "").strip()
                                or None,
                                page_index=obj.get("page_index")
                                if isinstance(obj.get("page_index"), int)
                                else None,
                            ),
                        )
                    except Exception:
                        pass

    except Exception as e:
        logger.error("Error reading JSONL %s: %s", temp_jsonl_path, e)

    # Fall back to the run-level source file for entries lacking one
    if run_source_file:
        for entry in entries.values():
            if entry.source_file is None:
                entry.source_file = run_source_file

    return [entries[k] for k in sorted(entries.keys())]


def find_failure_indices(lines: list[str], include_no_text: bool) -> list[int]:
    idxs: list[int] = []
    for i, line in enumerate(lines):
        if (
            is_failure_line(line)
            or include_no_text
            and NO_TEXT_PATTERN.match(line.strip())
        ):
            idxs.append(i)
    return idxs


def resolve_image_path(
    parent_folder: Path,
    entry: ImageEntry,
    identifier: str | None = None,
) -> Path | None:
    if entry.pre_processed_image:
        p = Path(entry.pre_processed_image)
        if p.exists():
            return p
        rel = parent_folder / Path(entry.pre_processed_image).name
        if rel.exists():
            return rel

    # Fallback search by name in common subfolders
    for sub in ("preprocessed_images", "preprocessed_images_tesseract"):
        d = parent_folder / sub
        if d.exists():
            cand = d / entry.image_name
            if cand.exists():
                return cand

    # Search entry's own image directory (raw source images)
    if identifier:
        entry_dir = parent_folder / identifier
        if entry_dir.is_dir():
            cand = entry_dir / entry.image_name
            if cand.exists():
                return cand
            # Strip _pre_processed suffix and try common extensions
            raw_stem = Path(entry.image_name).stem
            for suffix in ("_pre_processed", "_preprocessed"):
                raw_stem = raw_stem.replace(suffix, "")
            for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"):
                cand = entry_dir / f"{raw_stem}{ext}"
                if cand.exists():
                    return cand

    return None


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".bak.{ts}{path.suffix}")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def write_repair_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def discover_jobs(paths_config: dict[str, Any]) -> list[Job]:
    """
    Discover repairable jobs by scanning the configured output folders for
    transcription .txt and .md files and locating their corresponding temporary
    JSONL.

    Supports both legacy (*_transcription.txt) and new (*.txt / *.md) naming
    conventions.

    Returns a list of Job records sorted by parent folder path.
    """
    jobs: list[Job] = []

    def find_companion_jsonl(txt_path: Path, identifier: str) -> Path | None:
        """Find the companion JSONL file for a transcription txt file."""
        parent = txt_path.parent
        # Try new format first (same base name)
        new_format = parent / f"{identifier}.jsonl"
        if new_format.exists():
            return new_format
        # Try legacy format
        legacy_format = parent / f"{identifier}_transcription.jsonl"
        if legacy_format.exists():
            return legacy_format
        # Search well-known subdirectories (e.g. after sync_manifest consolidation)
        for subdir_name in ("transcription_jsonl", "temp_jsonl"):
            subdir = parent / subdir_name
            candidate = subdir / f"{identifier}.jsonl"
            if candidate.exists():
                return candidate
            legacy_candidate = subdir / f"{identifier}_transcription.jsonl"
            if legacy_candidate.exists():
                return legacy_candidate
        return None

    def scan_root(root: str | None, kind: str) -> None:
        if not root:
            return
        root_path = Path(root)
        if not root_path.exists():
            return
        # Scan for all .txt and .md files in output folders
        for ext in ("*.txt", "*.md"):
            for p in root_path.rglob(ext):
                parent = p.parent
                # Determine identifier (strip _transcription suffix if present)
                identifier = p.stem.replace("_transcription", "").strip()
                temp = find_companion_jsonl(p, identifier)
                jobs.append(
                    Job(
                        parent_folder=parent,
                        identifier=identifier,
                        final_txt_path=p,
                        temp_jsonl_path=temp,
                        kind=kind,
                    )
                )

    file_paths = paths_config.get("file_paths", {})
    pdf_out = file_paths.get("PDFs", {}).get("output", None)
    img_out = file_paths.get("Images", {}).get("output", None)
    scan_root(pdf_out, "PDF")
    scan_root(img_out, "Images")
    jobs.sort(key=lambda j: str(j.parent_folder))
    return jobs


def read_final_lines(final_txt_path: Path) -> list[str]:
    """
    Read a final transcription file and return its lines without trailing newlines.
    """
    content = final_txt_path.read_text(encoding="utf-8")
    return content.splitlines()


# ---------------------------------------------------------------------------
# Orchestration (formerly modules/operations/repair/run.py)
# ---------------------------------------------------------------------------


@dataclass
class RepairTarget:
    order_index: int
    image_name: str
    image_path: Path | None
    custom_id: str | None
    line_index: int
    page_number: int | None = None
    image_base64: str | None = None
    mime_type: str | None = None


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_service = get_config_service()
    paths = config_service.get_paths_config()
    model = config_service.get_model_config()
    image_proc = config_service.get_image_processing_config()
    return paths, model, image_proc


def _rerender_payload_for_entry(
    entry: ImageEntry,
    model_config: dict[str, Any],
) -> Any | None:
    """Re-render a page/image in memory from its recorded source file.

    Used when no preprocessed image exists on disk (the streaming pipeline
    never writes one). Returns a PagePayload or None.
    """
    from modules.images.page_stream import (
        load_image_payload,
        parse_pdf_page_index,
        render_single_pdf_page_payload,
        resolve_image_settings,
    )

    if not entry.source_file:
        return None
    source = Path(entry.source_file)
    if not source.exists():
        logger.warning("Recorded source file missing: %s", source)
        return None

    tm = model_config.get("transcription_model", {})
    img_cfg, model_type, target_dpi, max_pixels = resolve_image_settings(
        tm.get("provider", "openai"), tm.get("name", "")
    )

    try:
        if source.suffix.lower() == ".pdf":
            page_index = entry.page_index
            if page_index is None:
                page_index = parse_pdf_page_index(entry.image_name)
            if page_index is None and entry.order_index >= 0:
                page_index = entry.order_index
            if page_index is None:
                return None
            return render_single_pdf_page_payload(
                source,
                page_index,
                target_dpi=target_dpi,
                img_cfg=img_cfg,
                model_type=model_type,
                max_pixels=max_pixels,
            )
        return load_image_payload(
            source,
            max(0, entry.order_index),
            img_cfg=img_cfg,
            model_type=model_type,
        )
    except Exception as e:
        logger.error(
            "In-memory re-render failed for %s (%s): %s",
            entry.image_name,
            source.name,
            e,
        )
        return None


def _resolve_repair_targets(
    job: Job,
    image_entries: list[ImageEntry],
    failure_indices: list[int],
    final_lines: list[str],
    model_config: dict[str, Any] | None = None,
) -> list[RepairTarget]:
    """Resolve failure line indices to concrete RepairTarget objects.

    Shared between sync and batch repair modes. Resolution order: existing
    preprocessed/source image file on disk, then in-memory re-render from
    the source recorded by the streaming pipeline.
    """
    name_to_entry: dict[str, ImageEntry] = {e.image_name: e for e in image_entries}
    targets: list[RepairTarget] = []

    for idx in failure_indices:
        image_name = extract_image_name_from_failure_line(final_lines[idx])
        entry = name_to_entry.get(image_name) if image_name else None

        resolved_path: Path | None = None
        resolved_order_index: int = -1
        rerendered: Any | None = None

        if entry:
            resolved_path = resolve_image_path(job.parent_folder, entry, job.identifier)
            resolved_order_index = entry.order_index
            if resolved_path is None and model_config is not None:
                rerendered = _rerender_payload_for_entry(entry, model_config)
        else:
            if image_name:
                for sub in ("preprocessed_images", "preprocessed_images_tesseract"):
                    cand = job.parent_folder / sub / image_name
                    if cand.exists():
                        resolved_path = cand
                        break
                # Search entry's own image directory with suffix stripping
                if resolved_path is None and job.identifier:
                    entry_dir = job.parent_folder / job.identifier
                    if entry_dir.is_dir():
                        raw_stem = Path(image_name).stem
                        for sfx in ("_pre_processed", "_preprocessed"):
                            raw_stem = raw_stem.replace(sfx, "")
                        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"):
                            cand = entry_dir / f"{raw_stem}{ext}"
                            if cand.exists():
                                resolved_path = cand
                                break

        if rerendered is None and (resolved_path is None or not resolved_path.exists()):
            logger.warning(
                "Could not resolve image for failure line %s (%s); skipping.",
                idx,
                image_name or "[unknown]",
            )
            continue

        pn: int | None = None
        if entry and isinstance(entry.page_number, int):
            pn = entry.page_number
        elif resolved_order_index is not None and resolved_order_index >= 0:
            pn = resolved_order_index + 1

        targets.append(
            RepairTarget(
                order_index=resolved_order_index,
                image_name=image_name
                or (resolved_path.name if resolved_path else "[unknown]"),
                image_path=resolved_path,
                custom_id=None,
                line_index=idx,
                page_number=pn,
                image_base64=rerendered.base64 if rerendered else None,
                mime_type=rerendered.mime_type if rerendered else None,
            )
        )

    return targets


def _persist_repaired_file(
    job: Job,
    final_lines: list[str],
) -> Path:
    """Back up the original file and write updated lines. Returns backup path."""
    backup = backup_file(job.final_txt_path)
    job.final_txt_path.write_text("\n".join(final_lines), encoding="utf-8")
    return backup


async def _repair_sync_mode(
    job: Job,
    model_config: dict[str, Any],
    image_entries: list[ImageEntry],
    failure_indices: list[int],
    final_lines: list[str],
    repair_jsonl_path: Path,
    schema_path: Path | None = None,
    additional_context_path: Path | None = None,
) -> tuple[int, int]:
    """Synchronous repair pass. Returns ``(repaired, failed)`` counts (CT-4)."""
    from modules.llm import transcribe_image_with_llm

    targets = _resolve_repair_targets(
        job, image_entries, failure_indices, final_lines, model_config
    )

    # Always record a session entry, even with zero targets
    write_repair_jsonl_line(
        repair_jsonl_path,
        {
            "repair_session": {
                "identifier": job.identifier,
                "mode": "synchronous",
                "timestamp": datetime.now(UTC).isoformat(),
                "targets": len(targets),
            }
        },
    )

    if not targets:
        print_info("[INFO] No targets resolved for synchronous repair.")
        return (0, 0)

    print_info(
        f"[INFO] Synchronous repair of {len(targets)} page(s) for '{job.identifier}'."
    )

    async def worker(
        target: RepairTarget, transcriber: Any
    ) -> tuple[int, str, dict[str, Any]]:
        try:
            if target.image_base64:
                raw = await transcriber.transcribe_image_from_base64(
                    target.image_base64, target.mime_type or "image/jpeg"
                )
            elif target.image_path is not None:
                raw = await transcribe_image_with_llm(target.image_path, transcriber)
            else:
                raise ValueError(f"No image data for repair target {target.image_name}")
            text = extract_transcribed_text(raw, target.image_name)
            return target.line_index, text, raw
        except Exception as e:
            logger.error("Sync repair failed for %s: %s", target.image_name, e)
            return target.line_index, "[transcription error]", {}

    from modules.llm.providers.factory import (
        ProviderType,
        resolve_api_key_env_var,
    )

    env_var = resolve_api_key_env_var(ProviderType.OPENAI) or "OPENAI_API_KEY"
    api_key = os.getenv(env_var)
    if not api_key:
        print_error(f"[ERROR] {env_var} is required for GPT repair. Aborting.")
        return (0, len(targets))

    model_name = model_config.get("transcription_model", {}).get(
        "name", "gpt-4o-2024-08-06"
    )

    conc = get_config_service().get_concurrency_config()
    trans_cfg = conc.get("concurrency", {}).get("transcription", {})
    concurrency_limit = int(trans_cfg.get("concurrency_limit", 8))
    delay_between = float(trans_cfg.get("delay_between_tasks", 0))

    repairs_dir = job.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    # Do not write a duplicate repair_session marker here; already recorded above.

    # Repair issues real synchronous GPT calls; gate them on the daily token
    # budget so a large repair pauses at the cap and resumes after the reset
    # instead of blowing through it (mirrors the streaming manager's drain/wait).
    tracker = get_token_tracker()

    async with open_transcriber(
        api_key=api_key,
        model=model_name,
        schema_path=schema_path,
        additional_context_path=additional_context_path,
    ) as trans:
        write_lock = asyncio.Lock()

        async def on_result(res: Any) -> None:
            if not res:
                return
            line_index, text, raw = res
            img_name = next(
                (t.image_name for t in targets if t.line_index == line_index), None
            )
            # Keep placeholders minimal; page/image context added during formatting
            normalized_text = text
            record = {
                "repair_response": {
                    "line_index": line_index,
                    "order_index": next(
                        (t.order_index for t in targets if t.line_index == line_index),
                        None,
                    ),
                    "image_name": img_name,
                    "page_number": next(
                        (t.page_number for t in targets if t.line_index == line_index),
                        None,
                    ),
                    "raw_response": raw,
                    "text": normalized_text,
                    "raw_text": text,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            }
            async with write_lock:
                write_repair_jsonl_line(repair_jsonl_path, record)

        # Token-budget-gated multi-pass loop: each pass admits pages until the
        # budget is exhausted, then drains and waits for the daily reset before
        # re-passing over the deferred pages (their worker returned None and left
        # no JSONL record, so they remain repairable on the next pass/run).
        remaining = list(targets)
        collected: list[tuple[int, str, dict[str, Any]]] = []
        stalled_resets = 0
        while remaining:
            exhausted = asyncio.Event()
            args_list = [(t, trans) for t in remaining]
            results = await run_concurrent_transcription_tasks(
                worker,
                args_list,
                concurrency_limit=concurrency_limit,
                delay=delay_between,
                on_result=on_result,
                tracker=tracker,
                exhausted=exhausted,
            )

            deferred: list[RepairTarget] = []
            for target, res in zip(remaining, results, strict=True):
                if res is None:
                    deferred.append(target)
                else:
                    collected.append(res)

            if not exhausted.is_set() or not deferred:
                break

            made_progress = len(deferred) < len(remaining)
            print_warning(
                f"[WARN] Daily token budget reached; {len(deferred)} repair "
                f"page(s) deferred. Waiting for daily reset..."
            )
            if not await check_and_wait_for_token_limit(conc):
                print_info("[INFO] Wait cancelled; remaining pages left unrepaired.")
                break

            # Safeguard: if a full day's reset yields no progress twice running, a
            # single page exceeds the entire daily budget; stop.
            if not made_progress:
                stalled_resets += 1
                if stalled_resets >= 2:
                    print_warning(
                        "[WARN] A single page appears to exceed the entire daily "
                        "token budget; stopping. Raise daily_tokens to repair the "
                        "remaining pages."
                    )
                    break
            else:
                stalled_resets = 0
            remaining = deferred

    results = collected

    # Apply edits to final_lines with unified page-aware formatting
    target_by_line: dict[int, RepairTarget] = {t.line_index: t for t in targets}

    for line_index, text, _raw in results:
        if 0 <= line_index < len(final_lines):
            t = target_by_line.get(line_index)
            if t:
                pn = (
                    t.page_number
                    if isinstance(t.page_number, int)
                    else (
                        t.order_index + 1
                        if isinstance(t.order_index, int) and t.order_index >= 0
                        else None
                    )
                )
                final_lines[line_index] = format_page_line(text, pn, t.image_name)

    backup = _persist_repaired_file(job, final_lines)
    print_success(
        f"[SUCCESS] Synchronous repair complete for '{job.identifier}'. "
        f"Backup written to: {backup.name}"
    )

    # Outcome counts for the --json summary (CT-4): a result whose text is
    # still an error placeholder counts as failed, as do pages left
    # unrepaired (deferred/aborted before completion).
    error_count = sum(
        1
        for _, text, _ in results
        if detect_transcription_cause((text or "").strip()) == "api_error"
    )
    repaired = len(results) - error_count
    failed = error_count + (len(targets) - len(results))
    return (repaired, failed)


def _await_batches_blocking(
    client: OpenAI,
    batch_ids: list[str],
    poll_seconds: float = 10.0,
    timeout_seconds: float = 7200.0,
) -> tuple[bool, dict[str, str]]:
    start = time.time()
    status_map: dict[str, str] = {}
    terminal = {"completed", "failed", "expired", "cancelled"}

    while True:
        all_terminal = True
        for bid in batch_ids:
            try:
                b = sdk_to_dict(client.batches.retrieve(bid))
                status = str(b.get("status", "")).lower()
                status_map[bid] = status
                if status not in terminal:
                    all_terminal = False
            except Exception as e:
                status_map[bid] = f"error: {e}"
                all_terminal = False

        if all_terminal:
            return True, status_map

        if time.time() - start > timeout_seconds:
            return False, status_map

        time.sleep(max(1.0, poll_seconds))


def _parse_batch_outputs_for_repairs(
    batches: list[dict[str, Any]],
    client: OpenAI,
) -> list[dict[str, Any]]:
    parsed_lines: list[dict[str, Any]] = []
    for b in batches:
        try:
            if str(b.get("status", "")).lower() != "completed":
                continue
            # Robustly resolve output file id across possible shapes
            output_file_id = coerce_file_id(b.get("output_file_id"))
            if not output_file_id:
                for key in (
                    "output_file_id",
                    "output_file",
                    "output_file_ids",
                    "response_file_id",
                    "result_file_id",
                    "results_file_id",
                    "result_file_ids",
                ):
                    output_file_id = coerce_file_id(b.get(key))
                    if output_file_id:
                        break
            if not output_file_id:
                continue
            resp = client.files.content(output_file_id)
            content = resp.read()
            text = (
                content.decode("utf-8") if isinstance(content, bytes) else str(content)
            )
            text = text.strip()
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                with contextlib.suppress(Exception):
                    parsed_lines.append(json.loads(ln))
        except Exception as e:
            logger.warning("Error reading batch output: %s", e)
    return parsed_lines


async def _repair_batch_mode(
    job: Job,
    model_config: dict[str, Any],
    image_entries: list[ImageEntry],
    failure_indices: list[int],
    final_lines: list[str],
    repair_jsonl_path: Path,
    schema_path: Path | None = None,
    additional_context_path: Path | None = None,
) -> tuple[int, int]:
    """Batch repair pass. Returns ``(repaired, failed)`` counts (CT-4)."""
    targets = _resolve_repair_targets(
        job, image_entries, failure_indices, final_lines, model_config
    )
    # Items are file paths when the image exists on disk, otherwise the
    # re-rendered in-memory target (carrying base64 + mime type).
    images_for_batch: list[Any] = [
        t.image_path if t.image_path is not None else t for t in targets
    ]

    # Always write a session record (even if zero targets)
    repairs_dir = job.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)

    write_repair_jsonl_line(
        repair_jsonl_path,
        {
            "repair_session": {
                "identifier": job.identifier,
                "mode": "batch",
                "timestamp": datetime.now(UTC).isoformat(),
                "targets": len(targets),
            }
        },
    )

    if not targets:
        print_info("[INFO] No targets resolved for batch repair.")
        return (0, 0)

    print_info(f"[INFO] Batch repair of {len(targets)} page(s) for '{job.identifier}'.")

    from modules.batch import requests as batching

    try:
        batch_responses, metadata_records = await asyncio.to_thread(
            batching.process_batch_transcription,
            images_for_batch,
            "",  # prompt_text placeholder (unused)
            model_config.get("transcription_model", {}),
            schema_path=schema_path,
            additional_context_path=additional_context_path,
        )
    except Exception as e:
        logger.exception("Error submitting repair batch: %s", e)
        print_error("[ERROR] Failed to submit repair batch.")
        return (0, len(targets))

    # Persist metadata and batch tracking in repair JSONL
    for rec in metadata_records:
        write_repair_jsonl_line(repair_jsonl_path, rec)

    batch_ids: list[str] = []
    for resp in batch_responses:
        try:
            bid = resp.id
            batch_ids.append(bid)
            write_repair_jsonl_line(
                repair_jsonl_path,
                {
                    "batch_tracking": {
                        "batch_id": bid,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "batch_file": str(bid),
                        "repair": True,
                    }
                },
            )
        except Exception:
            pass

    if not batch_ids:
        print_error("[ERROR] No batch IDs returned for repair submission.")
        return (0, len(targets))

    client = OpenAI()
    print_info("[INFO] Waiting for repair batches to complete...")
    _ = await asyncio.to_thread(
        _await_batches_blocking, client, batch_ids, 10.0, 7200.0
    )

    batches = []
    for bid in batch_ids:
        try:
            b_dict = sdk_to_dict(client.batches.retrieve(bid))
            batches.append(b_dict)
        except Exception as e:
            logger.warning("Could not retrieve batch %s: %s", bid, e)

    parsed_lines = _parse_batch_outputs_for_repairs(batches, client)

    fixed_text_by_custom: dict[str, str] = {}

    for obj in parsed_lines:
        custom_id = obj.get("custom_id")
        resp = obj.get("response")
        transcription_text = None

        if isinstance(resp, dict):
            status_code = resp.get("status_code")
            body = resp.get("body")
            if isinstance(status_code, int) and status_code != 200:
                transcription_text = "[transcription error: [repair item]]"
            elif isinstance(body, dict):
                error_obj = body.get("error")
                if (
                    isinstance(status_code, int)
                    and status_code != 200
                    or isinstance(error_obj, dict)
                ):
                    transcription_text = "[transcription error: [repair item]]"
                else:
                    transcription_text = extract_transcribed_text(body, "")

        if transcription_text is None and isinstance(resp, dict):
            body = resp.get("body")
            if isinstance(body, dict) and "choices" in body:
                transcription_text = extract_transcribed_text(body, "")

        if isinstance(custom_id, str) and transcription_text is not None:
            fixed_text_by_custom[custom_id] = transcription_text

    order_by_custom, line_index_by_custom, image_name_by_custom = (
        _correlate_repair_targets(targets, metadata_records)
    )

    # Write parsed repair responses and patch final lines
    for cid, text in fixed_text_by_custom.items():
        oi = order_by_custom.get(cid)
        li = line_index_by_custom.get(cid)
        img_name = image_name_by_custom.get(cid, "")
        write_repair_jsonl_line(
            repair_jsonl_path,
            {
                "repair_response": {
                    "custom_id": cid,
                    "order_index": oi,
                    "line_index": li,
                    "image_name": img_name,
                    "text": text,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            },
        )
        if isinstance(li, int) and 0 <= li < len(final_lines):
            # Compute best-effort page number
            pn = None
            if isinstance(oi, int) and oi >= 0:
                pn = oi + 1
            final_lines[li] = format_page_line(text, pn, img_name)

    backup = _persist_repaired_file(job, final_lines)
    print_success(
        f"[SUCCESS] Batch repair complete for '{job.identifier}'. "
        f"Backup written to: {backup.name}"
    )

    # Outcome counts for the --json summary (CT-4): a repaired line whose text
    # is still an error placeholder counts as failed, as do targets that never
    # received a parsed batch result.
    error_count = sum(
        1
        for text in fixed_text_by_custom.values()
        if detect_transcription_cause((text or "").strip()) == "api_error"
    )
    repaired = len(fixed_text_by_custom) - error_count
    failed = error_count + max(0, len(targets) - len(fixed_text_by_custom))
    return (repaired, failed)


async def main() -> None:
    """Interactive repair workflow entrypoint.

    - Discovers completed jobs by scanning configured output folders.
    - Lets the user select which transcription to repair.
    - Identifies failed lines (and optionally "no text" lines).
    - Performs either synchronous or batched repair and updates the final file.
    """
    print_header("REPAIR TRANSCRIPTIONS", "Fix failed or incomplete transcriptions")

    try:
        paths_cfg, model_cfg, _img_cfg = _load_configs()
    except Exception as e:
        print_error(f"Failed to load configs: {e}")
        logger.critical(f"Failed to load configs: {e}")
        return

    jobs = discover_jobs(paths_cfg)
    if not jobs:
        print_info("No completed transcription jobs found.")
        return

    # Build options for job selection
    job_options: list[tuple[str, str]] = []
    for idx, j in enumerate(jobs, 1):
        desc = f"[{j.kind}] {j.parent_folder.name} → {j.final_txt_path.name}"
        job_options.append((str(idx), desc))

    result = prompt_select(
        "Select a transcription to repair:", job_options, allow_back=False
    )

    if result.action != NavigationAction.CONTINUE:
        return

    job_sel = jobs[int(result.value) - 1]
    logger.info(f"User selected job: {job_sel.identifier}")

    image_entries = collect_image_entries_from_jsonl(job_sel.temp_jsonl_path)
    if not image_entries:
        print_warning(
            "Could not reconstruct page order from the job's JSONL. "
            "Will attempt to map by filename where possible."
        )

    final_lines = read_final_lines(job_sel.final_txt_path)

    # Configure which failure classes to repair
    print_separator()
    ui_print("\n  Configure which failure types to repair:\n", PromptStyle.INFO)

    include_api_errors_result = prompt_yes_no(
        "Include '[transcription error]' lines?", default=True, allow_back=False
    )
    include_api_errors = include_api_errors_result.value

    include_not_possible_result = prompt_yes_no(
        "Include '[Transcription not possible]' lines?", default=False, allow_back=False
    )
    include_not_possible = include_not_possible_result.value

    include_no_text_result = prompt_yes_no(
        "Include '[No transcribable text]' lines?", default=False, allow_back=False
    )
    include_no_text = include_no_text_result.value

    selected_causes = set()
    if include_api_errors:
        selected_causes.add("api_error")
    if include_not_possible:
        selected_causes.add("not_possible")
    if include_no_text:
        selected_causes.add("no_text")

    if not selected_causes:
        print_info("No failure classes selected; nothing to repair.")
        return

    # Detect causes on the already-formatted final lines
    failure_indices = [
        i
        for i, ln in enumerate(final_lines)
        if detect_transcription_cause((ln or "").strip()) in selected_causes
    ]

    if not failure_indices:
        print_info("No failed lines detected matching your criteria.")
        return

    print_success(f"Found {len(failure_indices)} line(s) to repair.")
    if len(failure_indices) <= 10:
        ui_print(
            f"  Line indices: {', '.join(map(str, failure_indices))}", PromptStyle.DIM
        )
    else:
        ui_print(
            f"  First 10 indices: {', '.join(map(str, failure_indices[:10]))}",
            PromptStyle.DIM,
        )

    # Ask whether to repair all or subset
    scope_result = prompt_select(
        "Repair all detected lines, or select a subset?",
        [
            ("all", "Repair all detected failures"),
            ("subset", "Select specific line indices to repair"),
        ],
        allow_back=False,
    )

    if scope_result.value == "subset":
        ui_print(
            "\n  Enter comma-separated line indices to repair (e.g., 0,5,12)",
            PromptStyle.INFO,
        )

        while True:
            indices_result = prompt_text(
                "Line indices:", allow_empty=False, allow_back=False
            )

            try:
                chosen = sorted(
                    set(
                        int(x.strip())
                        for x in indices_result.value.split(",")
                        if x.strip()
                    )
                )
                failure_indices = [i for i in chosen if 0 <= i < len(final_lines)]
                if failure_indices:
                    print_success(
                        f"Selected {len(failure_indices)} line(s) for repair."
                    )
                    break
                print_error("No valid indices provided.")
            except Exception:
                print_error("Invalid format. Please use comma-separated numbers.")

    # Choose repair mode
    mode_result = prompt_select(
        "Choose repair mode:",
        [
            ("sync", "Synchronous Repair — Direct API calls with immediate results"),
            ("batch", "Batch Repair — Submit as batch job (will wait for completion)"),
        ],
        allow_back=False,
    )

    mode = mode_result.value
    logger.info(f"User selected repair mode: {mode}")

    repairs_dir = job_sel.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    repair_jsonl_path = repairs_dir / f"{job_sel.identifier}_temporary_repair.jsonl"
    if not repair_jsonl_path.exists():
        repair_jsonl_path.touch()

    # Resolve schema and context for consistent repair with original transcription
    from modules.config.config_loader import PROJECT_ROOT
    from modules.config.context import resolve_context_for_folder

    # Use default schema
    default_schema = (
        PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
    ).resolve()
    schema_path = default_schema if default_schema.exists() else None

    # Resolve context using folder-based hierarchy
    context_content, context_path = resolve_context_for_folder(job_sel.parent_folder)
    additional_context_path = context_path

    if additional_context_path:
        print_info(f"Using context: {additional_context_path.name}")
    if schema_path:
        print_info(f"Using schema: {schema_path.name}")

    print_header("PROCESSING REPAIR", f"Repairing {len(failure_indices)} line(s)...")

    if mode == "sync":
        await _repair_sync_mode(
            job=job_sel,
            model_config=model_cfg,
            image_entries=image_entries,
            failure_indices=failure_indices,
            final_lines=final_lines,
            repair_jsonl_path=repair_jsonl_path,
            schema_path=schema_path,
            additional_context_path=additional_context_path,
        )
    else:
        await _repair_batch_mode(
            job=job_sel,
            model_config=model_cfg,
            image_entries=image_entries,
            failure_indices=failure_indices,
            final_lines=final_lines,
            repair_jsonl_path=repair_jsonl_path,
            schema_path=schema_path,
            additional_context_path=additional_context_path,
        )

    print_header("REPAIR COMPLETE", "")
    print_success(f"Repair session completed for '{job_sel.identifier}'")
    print_info(f"Repair log: {repair_jsonl_path.relative_to(job_sel.parent_folder)}")


async def main_cli(args: Any, paths_config: dict[str, Any]) -> dict[str, int]:
    """CLI mode repair workflow entrypoint.

    Args:
        args: Parsed command-line arguments
        paths_config: Paths configuration dictionary

    Returns:
        Summary counts ``{"repaired": N, "failed": N}`` for the ``--json``
        summary line (CT-4); zeros when the run ends before any repair.
    """
    from modules.config.config_loader import PROJECT_ROOT
    from modules.core.cli_args import parse_indices, resolve_path, validate_input_path

    empty_summary = {"repaired": 0, "failed": 0}
    print_header("REPAIR TRANSCRIPTIONS (CLI MODE)", "")

    try:
        _, model_cfg, _img_cfg = _load_configs()
    except Exception as e:
        print_error(f"Failed to load configs: {e}")
        logger.critical(f"Failed to load configs: {e}")
        return empty_summary

    # Resolve transcription file path
    transcription_path = resolve_path(args.transcription, PROJECT_ROOT)
    validate_input_path(transcription_path)

    if transcription_path.suffix not in (".txt", ".md"):
        print_error(
            f"Expected .txt or .md transcription file, got: {transcription_path.name}"
        )
        return empty_summary

    # Find corresponding temp JSONL file
    # Support both legacy (*_transcription.txt) and new (*.txt) naming
    identifier = transcription_path.stem.replace("_transcription", "")
    parent_folder = transcription_path.parent

    # Try new format first, then legacy
    temp_jsonl_path = parent_folder / f"{identifier}.jsonl"
    if not temp_jsonl_path.exists():
        temp_jsonl_path = parent_folder / f"{identifier}_transcription.jsonl"
    # Search well-known subdirectories (e.g. after sync_manifest consolidation)
    if not temp_jsonl_path.exists():
        for subdir_name in ("transcription_jsonl", "temp_jsonl"):
            candidate = parent_folder / subdir_name / f"{identifier}.jsonl"
            if candidate.exists():
                temp_jsonl_path = candidate
                break
            legacy_candidate = (
                parent_folder / subdir_name / f"{identifier}_transcription.jsonl"
            )
            if legacy_candidate.exists():
                temp_jsonl_path = legacy_candidate
                break

    # Create job object
    job = Job(
        identifier=identifier,
        parent_folder=parent_folder,
        final_txt_path=transcription_path,
        temp_jsonl_path=temp_jsonl_path if temp_jsonl_path.exists() else None,
        kind="manual_cli",
    )

    print_info(f"Repairing: {transcription_path.name}")

    # Collect image entries
    image_entries = collect_image_entries_from_jsonl(job.temp_jsonl_path)
    if not image_entries:
        print_warning(
            "Could not reconstruct page order from JSONL."
            " Will attempt to map by filename."
        )

    # Read final lines
    final_lines = read_final_lines(job.final_txt_path)

    # Determine which failure classes to include
    selected_causes = set()

    if args.all_failures:
        selected_causes = {"api_error", "not_possible", "no_text"}
    else:
        if args.errors_only or (not args.not_possible and not args.no_text):
            # Default to errors only if nothing else specified
            selected_causes.add("api_error")
        if args.not_possible:
            selected_causes.add("not_possible")
        if args.no_text:
            selected_causes.add("no_text")

    if not selected_causes:
        print_error(
            "No failure types selected."
            " Use --errors-only, --not-possible, --no-text, or --all-failures"
        )
        return empty_summary

    print_info(f"Targeting failure types: {', '.join(selected_causes)}")

    # Detect failures
    failure_indices = [
        i
        for i, ln in enumerate(final_lines)
        if detect_transcription_cause((ln or "").strip()) in selected_causes
    ]

    if not failure_indices:
        print_info("No failed lines detected matching your criteria.")
        return empty_summary

    print_success(f"Found {len(failure_indices)} line(s) to repair.")

    # Handle index filtering
    if args.indices:
        try:
            specified_indices = parse_indices(args.indices)
            failure_indices = [
                i
                for i in specified_indices
                if i in failure_indices and 0 <= i < len(final_lines)
            ]
            if not failure_indices:
                print_error("None of the specified indices matched detected failures.")
                return empty_summary
            print_info(f"Filtered to {len(failure_indices)} specified line(s).")
        except ValueError as e:
            print_error(f"Invalid indices format: {e}")
            return empty_summary

    # Determine mode
    mode = "batch" if args.batch else "sync"
    print_info(f"Repair mode: {mode}")

    # Setup repair directory
    repairs_dir = job.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    repair_jsonl_path = repairs_dir / f"{job.identifier}_temporary_repair.jsonl"
    if not repair_jsonl_path.exists():
        repair_jsonl_path.touch()

    # Resolve schema and context for consistent repair with original transcription
    from modules.config.context import resolve_context_for_folder

    # Use default schema or CLI-specified schema
    schema_path = None
    if hasattr(args, "schema") and args.schema:
        schema_path = resolve_path(args.schema, PROJECT_ROOT)
    else:
        default_schema = (
            PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
        ).resolve()
        if default_schema.exists():
            schema_path = default_schema

    # Resolve context using folder-based hierarchy or CLI-specified context
    additional_context_path = None
    if hasattr(args, "context") and args.context:
        additional_context_path = resolve_path(args.context, PROJECT_ROOT)
    else:
        context_content, context_path = resolve_context_for_folder(job.parent_folder)
        additional_context_path = context_path

    if additional_context_path:
        print_info(f"Using context: {additional_context_path.name}")
    if schema_path:
        print_info(f"Using schema: {schema_path.name}")

    print_header("PROCESSING REPAIR", f"Repairing {len(failure_indices)} line(s)...")

    # Execute repair
    if mode == "sync":
        repaired, failed = await _repair_sync_mode(
            job=job,
            model_config=model_cfg,
            image_entries=image_entries,
            failure_indices=failure_indices,
            final_lines=final_lines,
            repair_jsonl_path=repair_jsonl_path,
            schema_path=schema_path,
            additional_context_path=additional_context_path,
        )
    else:
        repaired, failed = await _repair_batch_mode(
            job=job,
            model_config=model_cfg,
            image_entries=image_entries,
            failure_indices=failure_indices,
            final_lines=final_lines,
            repair_jsonl_path=repair_jsonl_path,
            schema_path=schema_path,
            additional_context_path=additional_context_path,
        )

    print_header("REPAIR COMPLETE", "")
    print_success(f"Repair session completed for '{job.identifier}'")
    print_info(f"Repair log: {repair_jsonl_path.relative_to(job.parent_folder)}")
    return {"repaired": repaired, "failed": failed}
