from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.annotation_txt import PageAnnotation, read_annotations_txt, write_annotations_txt
from fine_tuning.annotations_jsonl import build_annotation_records, write_annotations_jsonl
from fine_tuning.jsonl_io import iter_jsonl
from fine_tuning.paths import annotations_root, datasets_root, editable_root
from fine_tuning.sft_dataset import build_sft_dataset
from fine_tuning.validation import validate_transcription_output


def _load_schema(schema_arg: Optional[str]) -> Dict[str, Any]:
    """Load schema from path or name."""
    if not schema_arg:
        default_schema = PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
        if default_schema.exists():
            with default_schema.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    schema_path = Path(schema_arg)
    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    # Try as schema name
    from modules.llm.schema_utils import find_schema_path_by_name
    by_name = find_schema_path_by_name(schema_arg)
    if by_name and by_name.exists():
        with by_name.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Schema not found: {schema_arg}")


def _load_system_prompt(prompt_path: Optional[str]) -> str:
    """Load system prompt from file."""
    if prompt_path:
        p = Path(prompt_path)
    else:
        p = PROJECT_ROOT / "system_prompt" / "system_prompt.txt"
    
    if not p.exists():
        raise FileNotFoundError(f"System prompt not found: {p}")
    
    return p.read_text(encoding="utf-8").strip()


def _blank_output() -> Dict[str, Any]:
    """Return a blank transcription output template."""
    return {
        "image_analysis": "",
        "transcription": "",
        "no_transcribable_text": False,
        "transcription_not_possible": False,
    }


def _parse_manifest_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Parse a manifest or transcription JSONL file."""
    records = []
    for rec in iter_jsonl(path):
        records.append(rec)
    return records


def cmd_create_editable(args: argparse.Namespace) -> None:
    """
    Create an editable text file from a manifest or transcription output.
    
    This allows a research assistant to review and correct transcriptions
    with clear identification of which image each transcription belongs to.
    """
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    records = _parse_manifest_or_jsonl(manifest_path)
    if not records:
        raise ValueError(f"No records found in: {manifest_path}")
    
    annotations: List[PageAnnotation] = []
    
    for i, rec in enumerate(records):
        # Extract image info
        image_name = rec.get("image_name") or rec.get("file_name", "")
        if not image_name:
            meta = rec.get("image_metadata", {})
            if isinstance(meta, dict):
                image_name = meta.get("image_name") or meta.get("file_name", "")
        if not image_name:
            image_name = f"page_{i + 1:04d}"
        
        # Extract image path
        image_path = rec.get("image_path") or rec.get("pre_processed_image", "")
        if not image_path:
            meta = rec.get("image_metadata", {})
            if isinstance(meta, dict):
                image_path = meta.get("image_path") or meta.get("pre_processed_image", "")
        
        # Extract page index
        page_index = rec.get("page_index") or rec.get("order_index", i)
        if page_index is None:
            page_index = i
        
        # Build output from existing transcription or blank
        if args.blank:
            output = _blank_output()
        else:
            # Try to extract existing transcription
            transcription = rec.get("transcription")
            no_text = rec.get("no_transcribable_text", False)
            not_possible = rec.get("transcription_not_possible", False)
            image_analysis = rec.get("image_analysis", "")
            
            output = {
                "image_analysis": image_analysis or "",
                "transcription": transcription if transcription else "",
                "no_transcribable_text": bool(no_text),
                "transcription_not_possible": bool(not_possible),
            }
        
        annotations.append(
            PageAnnotation(
                page_index=int(page_index),
                image_name=str(image_name),
                image_path=str(image_path) if image_path else None,
                output=output,
            )
        )
    
    # Sort by page index
    annotations.sort(key=lambda a: a.page_index)
    
    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = editable_root() / f"{manifest_path.stem}_editable.txt"
    
    write_annotations_txt(annotations, out_path)
    print(f"[OK] Wrote editable file: {out_path}")
    print(f"     Pages: {len(annotations)}")
    print(f"     Open in Notepad++ to review and correct transcriptions.")


def cmd_import_annotations(args: argparse.Namespace) -> None:
    """
    Import corrected annotations from an editable text file.
    
    Validates the corrections and saves to JSONL format.
    """
    editable_path = Path(args.editable)
    if not editable_path.exists():
        raise FileNotFoundError(f"Editable file not found: {editable_path}")
    
    annotations = read_annotations_txt(editable_path)
    
    # Load schema for validation if provided
    schema = None
    if args.schema:
        schema = _load_schema(args.schema)
    
    # Validate each annotation
    for ann in annotations:
        if ann.output is None:
            raise ValueError(f"Page {ann.image_name}: missing output JSON")
        if schema:
            validate_transcription_output(ann.output, schema)
    
    # Build annotation records with metadata
    records = build_annotation_records(
        schema_name=args.schema,
        annotations=annotations,
        source_id=args.source_id,
        annotator_id=args.annotator_id,
    )
    
    # Determine output path
    default_out_name = editable_path.stem.replace("_editable", "") + ".jsonl"
    out_path = Path(args.out) if args.out else annotations_root() / default_out_name
    
    write_annotations_jsonl(out_path, records)
    print(f"[OK] Wrote annotations JSONL: {out_path}")
    print(f"     Records: {len(records)}")


def cmd_build_sft(args: argparse.Namespace) -> None:
    """
    Build an SFT dataset from annotation JSONL files.
    
    Creates train.jsonl and val.jsonl files ready for OpenAI fine-tuning.
    """
    annotations_paths = [Path(p) for p in args.annotations]
    
    for p in annotations_paths:
        if not p.exists():
            raise FileNotFoundError(f"Annotations file not found: {p}")
    
    # Load and render system prompt
    system_prompt = _load_system_prompt(args.prompt_path)
    
    if args.schema:
        schema = _load_schema(args.schema)
        from modules.llm.prompt_utils import render_prompt_with_schema
        system_prompt = render_prompt_with_schema(system_prompt, schema)
    
    # Build dataset
    out_dir = Path(args.out_dir) if args.out_dir else (datasets_root() / args.dataset_id)
    
    train_path, val_path = build_sft_dataset(
        annotations_paths=annotations_paths,
        out_dir=out_dir,
        system_prompt=system_prompt,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    
    print(f"[OK] Wrote SFT train JSONL: {train_path}")
    if val_path is not None:
        print(f"[OK] Wrote SFT val JSONL: {val_path}")


def cmd_build_vision_sft(args: argparse.Namespace) -> None:
    """
    Build a vision SFT dataset from ground truth and manifest.
    
    This wraps the existing build_openai_vision_sft_jsonl functionality.
    """
    from fine_tuning.build_openai_vision_sft_jsonl import build_openai_vision_sft_jsonl
    
    additional_context_path = Path(args.additional_context) if args.additional_context else None
    image_detail = args.image_detail
    if image_detail == "auto":
        image_detail = ""
    
    result = build_openai_vision_sft_jsonl(
        ground_truth_path=Path(args.ground_truth),
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        system_prompt_path=Path(args.system_prompt),
        schema_arg=args.schema,
        additional_context_path=additional_context_path,
        image_detail=image_detail,
        strict=bool(args.strict),
    )
    
    if result != 0:
        raise RuntimeError("Vision SFT build failed")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="fine_tuning",
        description="ChronoTranscriber Fine-Tuning CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create-editable command
    p_editable = sub.add_parser(
        "create-editable",
        help="Create editable text file from manifest/transcription output",
    )
    p_editable.add_argument(
        "--manifest", required=True,
        help="Path to manifest or transcription JSONL file",
    )
    p_editable.add_argument(
        "--out",
        help="Output path for editable file (default: artifacts/editable_txt/)",
    )
    p_editable.add_argument(
        "--blank", action="store_true",
        help="Create blank output templates instead of using existing transcriptions",
    )
    p_editable.set_defaults(func=cmd_create_editable)

    # import-annotations command
    p_import = sub.add_parser(
        "import-annotations",
        help="Import corrected annotations from editable text file",
    )
    p_import.add_argument(
        "--editable", required=True,
        help="Path to editable text file",
    )
    p_import.add_argument(
        "--schema",
        help="Schema name or path for validation",
    )
    p_import.add_argument(
        "--out",
        help="Output path for annotations JSONL",
    )
    p_import.add_argument(
        "--source-id",
        help="Source document identifier",
    )
    p_import.add_argument(
        "--annotator-id",
        help="Annotator identifier",
    )
    p_import.set_defaults(func=cmd_import_annotations)

    # build-sft command (text-based)
    p_sft = sub.add_parser(
        "build-sft",
        help="Build text-based SFT dataset from annotation JSONL files",
    )
    p_sft.add_argument(
        "--annotations", required=True, nargs="+",
        help="Paths to annotation JSONL files",
    )
    p_sft.add_argument(
        "--dataset-id", required=True,
        help="Dataset identifier for output directory",
    )
    p_sft.add_argument(
        "--out-dir",
        help="Output directory (default: artifacts/datasets/{dataset-id})",
    )
    p_sft.add_argument(
        "--schema",
        help="Schema name or path for prompt rendering",
    )
    p_sft.add_argument(
        "--prompt-path",
        help="Path to system prompt file",
    )
    p_sft.add_argument(
        "--val-ratio", default="0.1",
        help="Validation set ratio (default: 0.1)",
    )
    p_sft.add_argument(
        "--seed", default="0",
        help="Random seed for train/val split (default: 0)",
    )
    p_sft.set_defaults(func=cmd_build_sft)

    # build-vision-sft command
    p_vision = sub.add_parser(
        "build-vision-sft",
        help="Build vision SFT dataset from ground truth and manifest",
    )
    p_vision.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth JSONL file or directory",
    )
    p_vision.add_argument(
        "--manifest", required=True,
        help="Path to manifest JSONL file",
    )
    p_vision.add_argument(
        "--output", required=True,
        help="Output JSONL file for OpenAI fine-tuning",
    )
    p_vision.add_argument(
        "--system-prompt",
        default=str(PROJECT_ROOT / "system_prompt" / "system_prompt.txt"),
        help="System prompt file path",
    )
    p_vision.add_argument(
        "--schema",
        help="Schema name or path",
    )
    p_vision.add_argument(
        "--additional-context",
        help="Optional additional context file",
    )
    p_vision.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="high",
        help="Image detail level (default: high)",
    )
    p_vision.add_argument(
        "--strict", action="store_true",
        help="Fail immediately on first error",
    )
    p_vision.set_defaults(func=cmd_build_vision_sft)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
