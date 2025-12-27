from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.llm.prompt_utils import prepare_prompt_with_context
from modules.llm.schema_utils import find_schema_path_by_name
from modules.llm.providers.base import BaseProvider


_OPENAI_VISION_FT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_OPENAI_VISION_FT_MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _iter_jsonl_dicts(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_no}")
            yield obj


def _collect_jsonl_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    raise ValueError(f"Path does not exist or is not a file/directory: {path}")


def _load_schema_and_prompt(
    *,
    system_prompt_path: Path,
    schema_arg: Optional[str],
    additional_context_path: Optional[Path],
) -> Tuple[str, Dict[str, Any]]:
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt missing: {system_prompt_path}")

    raw_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    if schema_arg:
        schema_path = Path(schema_arg)
        if schema_path.exists():
            resolved_schema_path = schema_path
        else:
            by_name = find_schema_path_by_name(schema_arg)
            if by_name is None:
                raise FileNotFoundError(f"Schema not found by path or name: {schema_arg}")
            resolved_schema_path = by_name
    else:
        resolved_schema_path = PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"

    if not resolved_schema_path.exists():
        raise FileNotFoundError(f"Schema file missing: {resolved_schema_path}")

    with resolved_schema_path.open("r", encoding="utf-8") as sf:
        schema_obj = json.load(sf)

    # Load additional context if provided
    context = None
    if additional_context_path and additional_context_path.exists():
        context = additional_context_path.read_text(encoding="utf-8").strip()
    
    system_prompt = prepare_prompt_with_context(raw_prompt, schema_obj, context)

    return system_prompt, schema_obj


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _get_first_str(obj: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _resolve_image_path(manifest_path: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        p = (manifest_path.parent / p).resolve()
    if p.exists():
        return p

    fallback = (manifest_path.parent / Path(raw_path).name).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Image path not found: {raw_path}")


def _validate_openai_vision_image_constraints(image_path: Path) -> None:
    ext = image_path.suffix.lower()
    if ext not in _OPENAI_VISION_FT_IMAGE_EXTS:
        raise ValueError(
            f"Unsupported image extension for OpenAI vision fine-tuning: {ext} ({image_path})"
        )
    try:
        size = int(image_path.stat().st_size)
    except Exception:
        size = 0
    if size > _OPENAI_VISION_FT_MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is too large for OpenAI vision fine-tuning: {size} bytes ({image_path})"
        )


def _parse_manifest_entry(
    entry: Dict[str, Any],
    *,
    default_source_name: Optional[str],
    manifest_path: Path,
) -> Tuple[str, int, Dict[str, Any]]:
    if "image_metadata" in entry and isinstance(entry["image_metadata"], dict):
        merged = dict(entry["image_metadata"])
        for k in ("source_name", "source", "document", "doc", "folder_name", "file_name"):
            if k in entry and k not in merged:
                merged[k] = entry[k]
        entry = merged

    source_name = (
        _get_first_str(
            entry,
            ("source_name", "folder_name", "file_name", "source", "document", "doc"),
        )
        or default_source_name
    )
    if not source_name:
        raise ValueError("Manifest entry missing source_name")

    page_index = _safe_int(entry.get("page_index"))
    if page_index is None:
        page_index = _safe_int(entry.get("order_index"))
    if page_index is None and entry.get("page_number") is not None:
        pn = _safe_int(entry.get("page_number"))
        page_index = pn - 1 if pn is not None else None
    if page_index is None or page_index < 0:
        raise ValueError(f"Manifest entry missing valid page_index/order_index: {entry}")

    image_url = _get_first_str(entry, ("image_url", "url"))
    if image_url:
        image_payload: Dict[str, Any] = {"url": image_url}
        return source_name, page_index, {"type": "image_url", "image_url": image_payload}

    image_path_str = _get_first_str(entry, ("image_path", "pre_processed_image", "image_file", "path"))
    if not image_path_str:
        image_name = _get_first_str(entry, ("file_name", "image_name"))
        if image_name:
            image_path_str = image_name

    if not image_path_str:
        raise ValueError("Manifest entry missing image_path/pre_processed_image/image_url")

    image_path = _resolve_image_path(manifest_path, image_path_str)
    _validate_openai_vision_image_constraints(image_path)
    b64, mime = BaseProvider.encode_image_to_base64(image_path)
    data_url = BaseProvider.create_data_url(b64, mime)

    image_payload = {"url": data_url}
    return source_name, page_index, {"type": "image_url", "image_url": image_payload}


def _load_ground_truth(ground_truth_path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    files = _collect_jsonl_files(ground_truth_path)
    if not files:
        raise ValueError(f"No JSONL files found in: {ground_truth_path}")

    sources: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for p in files:
        source_name = p.stem
        if source_name in sources:
            raise ValueError(f"Duplicate ground truth source name: {source_name} ({p})")

        pages: Dict[int, Dict[str, Any]] = {}
        for obj in _iter_jsonl_dicts(p):
            idx = _safe_int(obj.get("page_index"))
            if idx is None:
                continue
            pages[idx] = obj

        sources[source_name] = pages

    return sources


def _get_gt_record(
    gt: Dict[str, Dict[int, Dict[str, Any]]],
    source_name: str,
    page_index: int,
) -> Optional[Dict[str, Any]]:
    pages = gt.get(source_name)
    if pages is None:
        return None
    return pages.get(page_index)


def build_openai_vision_sft_jsonl(
    *,
    ground_truth_path: Path,
    manifest_path: Path,
    output_path: Path,
    system_prompt_path: Path,
    schema_arg: Optional[str],
    additional_context_path: Optional[Path],
    image_detail: str,
    strict: bool,
) -> int:
    gt = _load_ground_truth(ground_truth_path)

    system_prompt, _schema_obj = _load_schema_and_prompt(
        system_prompt_path=system_prompt_path,
        schema_arg=schema_arg,
        additional_context_path=additional_context_path,
    )

    manifest_entries = list(_iter_jsonl_dicts(manifest_path))
    if not manifest_entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    default_source_name: Optional[str] = manifest_path.stem
    if default_source_name not in gt and len(gt) == 1:
        default_source_name = next(iter(gt.keys()))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out:
        for i, entry in enumerate(manifest_entries, 1):
            try:
                source_name, page_index, image_content = _parse_manifest_entry(
                    entry,
                    default_source_name=default_source_name,
                    manifest_path=manifest_path,
                )

                if image_detail in ("low", "high"):
                    image_content["image_url"]["detail"] = image_detail

                gt_record = _get_gt_record(gt, source_name, page_index)
                if gt_record is None:
                    raise ValueError(
                        f"No ground truth found for source={source_name} page_index={page_index}"
                    )

                no_text = bool(gt_record.get("no_transcribable_text", False))
                not_possible = bool(gt_record.get("transcription_not_possible", False))
                transcription = gt_record.get("transcription")

                if no_text or not_possible:
                    transcription = None

                if transcription is not None and not isinstance(transcription, str):
                    transcription = str(transcription)

                if transcription is None and not (no_text or not_possible):
                    raise ValueError(
                        f"Ground truth has null transcription but flags are false for source={source_name} page_index={page_index}"
                    )

                assistant_obj = {
                    "image_analysis": "",
                    "transcription": transcription,
                    "no_transcribable_text": no_text,
                    "transcription_not_possible": not_possible,
                }

                example = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "The image:"},
                                image_content,
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(assistant_obj, ensure_ascii=False),
                        },
                    ]
                }

                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1

            except Exception as e:
                skipped += 1
                msg = f"[SKIP] Manifest line {i}: {e}"
                print(msg, file=sys.stderr)
                if strict:
                    raise

    print(f"[INFO] Ground truth sources loaded: {len(gt)}")
    print(f"[INFO] Manifest entries: {len(manifest_entries)}")
    print(f"[INFO] Examples written: {written}")
    if skipped:
        print(f"[WARN] Entries skipped: {skipped}")

    if written == 0:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build OpenAI vision SFT JSONL from ChronoTranscriber ground truth + manifest (one example per page)."
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to a ground truth JSONL file or a directory containing ground truth JSONL files.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a JSONL manifest (can be a filtered transcriber output JSONL).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file for OpenAI fine-tuning.",
    )

    parser.add_argument(
        "--system-prompt",
        default=str(PROJECT_ROOT / "system_prompt" / "system_prompt.txt"),
        help="System prompt file path.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Schema name (from schemas/) or path to a schema JSON file.",
    )
    parser.add_argument(
        "--additional-context",
        default=None,
        help="Optional additional context file.",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="high",
        help="Image detail level. 'auto' omits the detail field.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on the first skipped manifest entry.",
    )

    args = parser.parse_args()

    try:
        image_detail = args.image_detail
        if image_detail == "auto":
            image_detail = ""

        return build_openai_vision_sft_jsonl(
            ground_truth_path=Path(args.ground_truth),
            manifest_path=Path(args.manifest),
            output_path=Path(args.output),
            system_prompt_path=Path(args.system_prompt),
            schema_arg=args.schema,
            additional_context_path=Path(args.additional_context) if args.additional_context else None,
            image_detail=image_detail,
            strict=bool(args.strict),
        )

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
