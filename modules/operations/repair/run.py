"""Repair operations for failed or placeholder transcriptions.

This module encapsulates the interactive repair workflow previously embedded in
`main/repair_transcriptions.py`. It keeps a clean separation of orchestration
logic from the CLI entry point so it can be imported and tested.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from modules.infra.logger import setup_logger
from modules.config.config_loader import ConfigLoader
from modules.core.utils import console_print, safe_input, check_exit
from modules.processing.text_processing import extract_transcribed_text
from modules.llm.openai_utils import open_transcriber
from modules.infra.concurrency import run_concurrent_transcription_tasks
from modules.llm.openai_sdk_utils import sdk_to_dict, coerce_file_id
from modules.ui.core import UserPrompt

# Centralized repair helpers
from modules.operations.repair.utils import (
    Job,
    ImageEntry,
    extract_image_name_from_failure_line as ru_extract_image_name_from_failure_line,
    is_failure_line as ru_is_failure_line,
    collect_image_entries_from_jsonl as ru_collect_image_entries_from_jsonl,
    find_failure_indices as ru_find_failure_indices,
    resolve_image_path as ru_resolve_image_path,
    backup_file as ru_backup_file,
    write_repair_jsonl_line as ru_write_repair_jsonl_line,
    discover_jobs as ru_discover_jobs,
    read_final_lines as ru_read_final_lines,
)

logger = setup_logger(__name__)


# Using Job and ImageEntry from modules.repair_utils


@dataclass
class RepairTarget:
    order_index: int
    image_name: str
    image_path: Path
    custom_id: Optional[str]
    line_index: int


# Failure patterns centralized in modules.repair_utils


def _extract_image_name_from_failure_line(line: str) -> Optional[str]:
    return ru_extract_image_name_from_failure_line(line)


def _load_configs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = ConfigLoader()
    cfg.load_configs()
    paths = cfg.get_paths_config()
    model = cfg.get_model_config()
    image_proc = cfg.get_image_processing_config()
    return paths, model, image_proc


def _discover_jobs(paths_config: Dict[str, Any]) -> List[Job]:
    return ru_discover_jobs(paths_config)


def _read_final_lines(final_txt_path: Path) -> List[str]:
    return ru_read_final_lines(final_txt_path)


def _is_failure_line(line: str) -> bool:
    return ru_is_failure_line(line)


def _collect_image_entries_from_jsonl(
    temp_jsonl_path: Optional[Path],
) -> List[ImageEntry]:
    # Delegate to centralized utility; returned items are attribute-compatible
    return ru_collect_image_entries_from_jsonl(temp_jsonl_path)


def _find_failure_indices(
    lines: List[str],
    include_no_text: bool,
) -> List[int]:
    return ru_find_failure_indices(lines, include_no_text)


def _resolve_image_path(parent_folder: Path, entry: ImageEntry) -> Optional[Path]:
    return ru_resolve_image_path(parent_folder, entry)



def _backup_file(path: Path) -> Path:
    return ru_backup_file(path)


def _write_repair_jsonl_line(path: Path, record: Dict[str, Any]) -> None:
    return ru_write_repair_jsonl_line(path, record)


async def _repair_sync_mode(
    job: Job,
    model_config: Dict[str, Any],
    image_entries: List[ImageEntry],
    failure_indices: List[int],
    final_lines: List[str],
    repair_jsonl_path: Path,
) -> None:
    from modules.llm.openai_utils import transcribe_image_with_openai

    # Build mapping by image_name for robust lookup
    name_to_entry: Dict[str, ImageEntry] = {e.image_name: e for e in image_entries}

    # Attempt to resolve each failure line by extracting its image name
    targets: List[RepairTarget] = []
    for idx in failure_indices:
        image_name = _extract_image_name_from_failure_line(final_lines[idx])
        entry = name_to_entry.get(image_name) if image_name else None

        resolved_path: Optional[Path] = None
        resolved_order_index: int = -1

        if entry:
            resolved_path = _resolve_image_path(job.parent_folder, entry)
            resolved_order_index = entry.order_index
        else:
            # Fallback: try to locate by image name in known folders
            if image_name:
                for sub in ("preprocessed_images", "preprocessed_images_tesseract"):
                    cand = job.parent_folder / sub / image_name
                    if cand.exists():
                        resolved_path = cand
                        break

        if resolved_path is None or not resolved_path.exists():
            logger.warning(
                "Could not resolve image for failure line %s (%s); skipping.",
                idx,
                image_name or "[unknown]",
            )
            continue

        targets.append(
            RepairTarget(
                order_index=resolved_order_index,
                image_name=image_name or resolved_path.name,
                image_path=resolved_path,
                custom_id=None,
                line_index=idx,
            )
        )

    # Always record a session entry, even with zero targets
    _write_repair_jsonl_line(
        repair_jsonl_path,
        {
            "repair_session": {
                "identifier": job.identifier,
                "mode": "synchronous",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "targets": len(targets),
            }
        },
    )

    if not targets:
        console_print("[INFO] No targets resolved for synchronous repair.")
        return

    console_print(
        f"[INFO] Synchronous repair of {len(targets)} page(s) for "
        f"'{job.identifier}'."
    )

    async def worker(
        img_path: Path, line_index: int, image_name: str, transcriber: Any
    ) -> Tuple[int, str, Dict[str, Any]]:
        try:
            raw = await transcribe_image_with_openai(img_path, transcriber)
            text = extract_transcribed_text(raw, image_name)
            return line_index, text, raw
        except Exception as e:
            logger.error("Sync repair failed for %s: %s", image_name, e)
            return line_index, f"[transcription error: {image_name}]", {}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console_print("[ERROR] OPENAI_API_KEY is required for GPT repair. Aborting.")
        return

    model_name = model_config.get("transcription_model", {}).get(
        "name", "gpt-4o-2024-08-06"
    )

    cl = ConfigLoader()
    cl.load_configs()
    conc = cl.get_concurrency_config()
    trans_cfg = conc.get("concurrency", {}).get("transcription", {})
    concurrency_limit = int(trans_cfg.get("concurrency_limit", 8))
    delay_between = float(trans_cfg.get("delay_between_tasks", 0))

    repairs_dir = job.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    # Do not write a duplicate repair_session marker here; already recorded above.

    async with open_transcriber(api_key=api_key, model=model_name) as trans:
        write_lock = asyncio.Lock()

        async def on_result(res: Any) -> None:
            if not res:
                return
            line_index, text, raw = res
            img_name = next(
                (t.image_name for t in targets if t.line_index == line_index), None
            )
            # Normalize placeholders to include image name for traceability
            normalized_text = (
                f"[Transcription not possible: {img_name}]"
                if text == "[Transcription not possible]" and img_name
                else (
                    f"[No transcribable text: {img_name}]"
                    if text == "[No transcribable text]" and img_name
                    else text
                )
            )
            record = {
                "repair_response": {
                    "line_index": line_index,
                    "order_index": next(
                        (t.order_index for t in targets if t.line_index == line_index),
                        None,
                    ),
                    "image_name": img_name,
                    "raw_response": raw,
                    "text": normalized_text,
                    "raw_text": text,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }
            async with write_lock:
                _write_repair_jsonl_line(repair_jsonl_path, record)

        args_list = [(t.image_path, t.line_index, t.image_name, trans) for t in targets]

        results = await run_concurrent_transcription_tasks(
            worker,
            args_list,
            concurrency_limit=concurrency_limit,
            delay=delay_between,
            on_result=on_result,
        )

    # Apply edits to final_lines by order_index (sync path preserves 1:1 mapping)
    # Quick lookup for normalization
    target_by_line: Dict[int, RepairTarget] = {t.line_index: t for t in targets}

    for line_index, text, _raw in results:
        if 0 <= line_index < len(final_lines):
            t = target_by_line.get(line_index)
            if t and text == "[Transcription not possible]":
                final_lines[line_index] = f"[Transcription not possible: {t.image_name}]"
            elif t and text == "[No transcribable text]":
                final_lines[line_index] = f"[No transcribable text: {t.image_name}]"
            else:
                final_lines[line_index] = text

    backup = _backup_file(job.final_txt_path)
    job.final_txt_path.write_text("\n".join(final_lines), encoding="utf-8")
    console_print(
        f"[SUCCESS] Synchronous repair complete for '{job.identifier}'. "
        f"Backup written to: {backup.name}"
    )


def _await_batches_blocking(
    client: OpenAI,
    batch_ids: List[str],
    poll_seconds: float = 10.0,
    timeout_seconds: float = 7200.0,
) -> Tuple[bool, Dict[str, str]]:
    start = time.time()
    status_map: Dict[str, str] = {}
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
    batches: List[Dict[str, Any]],
    client: OpenAI,
) -> List[Dict[str, Any]]:
    parsed_lines: List[Dict[str, Any]] = []
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
            text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
            text = text.strip()
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    parsed_lines.append(json.loads(ln))
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Error reading batch output: %s", e)
    return parsed_lines


async def _repair_batch_mode(
    job: Job,
    model_config: Dict[str, Any],
    image_entries: List[ImageEntry],
    failure_indices: List[int],
    final_lines: List[str],
    repair_jsonl_path: Path,
) -> None:
    targets: List[RepairTarget] = []
    images_for_batch: List[Path] = []

    # Map by image name from failure lines (same strategy as sync mode)
    name_to_entry: Dict[str, ImageEntry] = {e.image_name: e for e in image_entries}

    for idx in failure_indices:
        image_name = _extract_image_name_from_failure_line(final_lines[idx])
        entry = name_to_entry.get(image_name) if image_name else None

        resolved_path: Optional[Path] = None
        resolved_order_index: int = -1

        if entry:
            resolved_path = _resolve_image_path(job.parent_folder, entry)
            resolved_order_index = entry.order_index
        else:
            if image_name:
                for sub in ("preprocessed_images", "preprocessed_images_tesseract"):
                    cand = job.parent_folder / sub / image_name
                    if cand.exists():
                        resolved_path = cand
                        break

        if resolved_path is None or not resolved_path.exists():
            logger.warning(
                "Could not resolve image for failure line %s (%s); skipping.",
                idx,
                image_name or "[unknown]",
            )
            continue

        targets.append(
            RepairTarget(
                order_index=resolved_order_index,
                image_name=image_name or resolved_path.name,
                image_path=resolved_path,
                custom_id=None,
                line_index=idx,
            )
        )
        images_for_batch.append(resolved_path)

    # Always write a session record (even if zero targets)
    repairs_dir = job.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)

    _write_repair_jsonl_line(
        repair_jsonl_path,
        {
            "repair_session": {
                "identifier": job.identifier,
                "mode": "batch",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "targets": len(targets),
            }
        },
    )

    if not targets:
        console_print("[INFO] No targets resolved for batch repair.")
        return

    console_print(f"[INFO] Batch repair of {len(targets)} page(s) for '{job.identifier}'.")

    from modules.llm.batch import batching

    try:
        batch_responses, metadata_records = await asyncio.to_thread(
            batching.process_batch_transcription,
            images_for_batch,
            "",  # prompt_text placeholder (unused)
            model_config.get("transcription_model", {}),
        )
    except Exception as e:
        logger.exception("Error submitting repair batch: %s", e)
        console_print("[ERROR] Failed to submit repair batch.")
        return

    # Persist metadata and batch tracking in repair JSONL
    for rec in metadata_records:
        _write_repair_jsonl_line(repair_jsonl_path, rec)

    batch_ids: List[str] = []
    for resp in batch_responses:
        try:
            bid = resp.id
            batch_ids.append(bid)
            _write_repair_jsonl_line(
                repair_jsonl_path,
                {
                    "batch_tracking": {
                        "batch_id": bid,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "batch_file": str(bid),
                        "repair": True,
                    }
                },
            )
        except Exception:
            pass

    if not batch_ids:
        console_print("[ERROR] No batch IDs returned for repair submission.")
        return

    client = OpenAI()
    console_print("[INFO] Waiting for repair batches to complete...")
    _ = await asyncio.to_thread(_await_batches_blocking, client, batch_ids, 10.0, 7200.0)

    batches = []
    for bid in batch_ids:
        try:
            b_dict = sdk_to_dict(client.batches.retrieve(bid))
            batches.append(b_dict)
        except Exception as e:
            logger.warning("Could not retrieve batch %s: %s", bid, e)

    parsed_lines = _parse_batch_outputs_for_repairs(batches, client)

    fixed_text_by_custom: Dict[str, str] = {}

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
                if isinstance(status_code, int) and status_code != 200:
                    transcription_text = "[transcription error: [repair item]]"
                elif isinstance(error_obj, dict):
                    transcription_text = "[transcription error: [repair item]]"
                else:
                    transcription_text = extract_transcribed_text(body, "")

        if transcription_text is None and isinstance(resp, dict):
            body = resp.get("body")
            if isinstance(body, dict) and "choices" in body:
                transcription_text = extract_transcribed_text(body, "")

        if isinstance(custom_id, str) and transcription_text is not None:
            fixed_text_by_custom[custom_id] = transcription_text

    # IMPORTANT: Map custom_id back to the selected target, including
    # both original order_index and the selected line_index.
    order_by_custom: Dict[str, int] = {}
    line_index_by_custom: Dict[str, int] = {}
    image_name_by_custom: Dict[str, str] = {}

    for t, rec in zip(targets, metadata_records):
        br = rec.get("batch_request", {})
        cid = br.get("custom_id")
        if isinstance(cid, str):
            order_by_custom[cid] = t.order_index
            line_index_by_custom[cid] = t.line_index
            image_name_by_custom[cid] = t.image_name

    # Write parsed repair responses and patch final lines
    for cid, text in fixed_text_by_custom.items():
        oi = order_by_custom.get(cid)
        li = line_index_by_custom.get(cid)
        img_name = image_name_by_custom.get(cid, "")
        _write_repair_jsonl_line(
            repair_jsonl_path,
            {
                "repair_response": {
                    "custom_id": cid,
                    "order_index": oi,
                    "line_index": li,
                    "image_name": img_name,
                    "text": (
                        f"[Transcription not possible: {img_name}]"
                        if text == "[Transcription not possible]"
                        else (
                            f"[No transcribable text: {img_name}]"
                            if text == "[No transcribable text]"
                            else text
                        )
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        if isinstance(li, int) and 0 <= li < len(final_lines):
            if text == "[Transcription not possible]":
                final_lines[li] = f"[Transcription not possible: {img_name}]"
            elif text == "[No transcribable text]":
                final_lines[li] = f"[No transcribable text: {img_name}]"
            else:
                final_lines[li] = text

    backup = _backup_file(job.final_txt_path)
    job.final_txt_path.write_text("\n".join(final_lines), encoding="utf-8")

    console_print(
        f"[SUCCESS] Batch repair complete for '{job.identifier}'. "
        f"Backup written to: {backup.name}"
    )


async def main() -> None:
    """Interactive repair workflow entrypoint.

    - Discovers completed jobs by scanning configured output folders.
    - Lets the user select which transcription to repair.
    - Identifies failed lines (and optionally "no text" lines).
    - Performs either synchronous or batched repair and updates the final file.
    """
    console_print("\n" + "=" * 80)
    console_print("  REPAIR TRANSCRIPTIONS")
    console_print("=" * 80)

    try:
        paths_cfg, model_cfg, _img_cfg = _load_configs()
    except Exception as e:
        console_print(f"[CRITICAL] Failed to load configs: {e}")
        return

    jobs = _discover_jobs(paths_cfg)
    if not jobs:
        console_print("[INFO] No completed transcription jobs found.")
        return

    # Build options and delegate selection to centralized UI helper
    job_options: List[Tuple[str, str]] = []
    for idx, j in enumerate(jobs, 1):
        desc = f"[{j.kind}] {j.parent_folder.name} -> {j.final_txt_path.name}"
        job_options.append((str(idx), desc))
    selected_idx = UserPrompt.prompt_choice("Select a transcription to repair:", job_options)
    job_sel = jobs[int(selected_idx) - 1]

    image_entries = _collect_image_entries_from_jsonl(job_sel.temp_jsonl_path)
    if not image_entries:
        console_print(
            "[WARN] Could not reconstruct page order from the job's JSONL. "
            "Will attempt to map by filename where possible."
        )

    final_lines = _read_final_lines(job_sel.final_txt_path)

    console_print("\nWhich failure classes should be included for repair?")
    include_no_text = (
        UserPrompt.prompt_choice(
            "Include '[No transcribable text]' lines in repair set?",
            [("y", "Yes"), ("n", "No")],
        )
        == "y"
    )

    failure_indices = _find_failure_indices(final_lines, include_no_text)
    if not failure_indices:
        console_print("[INFO] No failed lines detected; nothing to repair.")
        return

    console_print(f"[INFO] Found {len(failure_indices)} line(s) to repair.")
    console_print("First 10 indices: " + ", ".join(map(str, failure_indices[:10])))

    use_all = (
        UserPrompt.prompt_choice(
            "Repair all detected lines, or select a subset?",
            [("all", "Repair all detected failures"), ("subset", "Select a subset by indices")],
        )
        == "all"
    )

    if not use_all:
        console_print(
            "Enter comma-separated 0-based line indices to repair "
            "(e.g., 0,5,12). Type 'q' to exit."
        )
        while True:
            s = safe_input("Indices: ").strip()
            check_exit(s)
            try:
                chosen = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
                failure_indices = [i for i in chosen if 0 <= i < len(final_lines)]
                if failure_indices:
                    break
                console_print("[ERROR] No valid indices provided.")
            except Exception:
                console_print("[ERROR] Invalid format. Try again.")

    mode = UserPrompt.prompt_choice(
        "Choose repair mode",
        [
            ("sync", "Non-batched (synchronous) repair via OpenAI Responses API"),
            ("batch", "Batched repair via OpenAI Batch API (will wait for completion)"),
        ],
    )

    repairs_dir = job_sel.parent_folder / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    repair_jsonl_path = repairs_dir / f"{job_sel.identifier}_temporary_repair.jsonl"
    if not repair_jsonl_path.exists():
        repair_jsonl_path.touch()

    if mode == "sync":
        await _repair_sync_mode(
            job=job_sel,
            model_config=model_cfg,
            image_entries=image_entries,
            failure_indices=failure_indices,
            final_lines=final_lines,
            repair_jsonl_path=repair_jsonl_path,
        )
    else:
        await _repair_batch_mode(
            job=job_sel,
            model_config=model_cfg,
            image_entries=image_entries,
            failure_indices=failure_indices,
            final_lines=final_lines,
            repair_jsonl_path=repair_jsonl_path,
        )

    console_print(f"\n[INFO] Repair session completed for '{job_sel.identifier}'.")
    console_print(
        f"[INFO] Repair log: {repair_jsonl_path.relative_to(job_sel.parent_folder)}"
    )
