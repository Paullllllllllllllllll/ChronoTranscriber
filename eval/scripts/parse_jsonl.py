"""Parse ChronoTranscriber JSONL outputs into tabular form.

The script ingests one or more JSONL files (either directly or via a directory
containing them) and emits a CSV/Parquet file with flattened fields that are
helpful for downstream analysis.

Usage (see --help for details):
    python eval/scripts/parse_jsonl.py --input eval/data/gpt/run --output parsed.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def find_jsonl_files(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    files = sorted(p for p in target.rglob("*.jsonl") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {target}")
    return files


def parse_structured_output(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to extract structured JSON payload from raw_response output."""
    output = raw_response.get("output")
    if not isinstance(output, list):
        return {}

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            text = content.get("text") if isinstance(content, dict) else None
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            if not stripped.startswith("{"):
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                # Try salvaging the last JSON object in the string.
                closing = stripped.rfind("}")
                if closing == -1:
                    continue
                for start in range(closing, -1, -1):
                    if stripped[start] == "{":
                        fragment = stripped[start : closing + 1]
                        try:
                            return json.loads(fragment)
                        except json.JSONDecodeError:
                            continue
    return {}


def flatten_record(record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "source_file": source_file,
        "file_name": record.get("file_name"),
        "image_name": record.get("image_name"),
        "order_index": record.get("order_index"),
        "timestamp": record.get("timestamp"),
        "method": record.get("method"),
        "text_chunk": record.get("text_chunk"),
        "pre_processed_image": record.get("pre_processed_image"),
    }

    request_context = record.get("request_context") or {}
    if isinstance(request_context, dict):
        for key, value in request_context.items():
            base[f"request_{key}"] = value

    raw_response = record.get("raw_response") or {}
    if isinstance(raw_response, dict):
        base["response_id"] = raw_response.get("id")
        base["response_model"] = raw_response.get("model")
        usage = raw_response.get("usage")
        if isinstance(usage, dict):
            base["usage_input_tokens"] = usage.get("input_tokens")
            base["usage_output_tokens"] = usage.get("output_tokens")
            base["usage_total_tokens"] = usage.get("total_tokens")
        parsed_structured = parse_structured_output(raw_response)
        if parsed_structured:
            base.setdefault("no_transcribable_text", parsed_structured.get("no_transcribable_text"))
            base.setdefault(
                "transcription_not_possible", parsed_structured.get("transcription_not_possible")
            )

    # For convenience: mark simple flag when text chunk indicates no text.
    text_chunk = base.get("text_chunk")
    if isinstance(text_chunk, str) and text_chunk.strip().lower() in {
        "[no transcribable text]",
        "no transcribable text",
    }:
        base.setdefault("no_transcribable_text", True)

    return base


def load_records(files: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {file_path}:{line_number}: {exc}") from exc
                rows.append(flatten_record(data, source_file=str(file_path)))
    return rows


def write_output(df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        df.to_csv(output_path, index=False)
    elif output_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {output_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flatten ChronoTranscriber JSONL files")
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL file or directory")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination file (CSV or Parquet based on --format)",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Output file format",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    files = find_jsonl_files(args.input)
    records = load_records(files)
    if not records:
        print("No data rows found.", file=sys.stderr)
        return 1

    df = pd.DataFrame(records)
    # Sort by file then order_index for deterministic outputs.
    if "order_index" in df.columns:
        df.sort_values(by=["file_name", "order_index"], inplace=True)

    write_output(df, args.output, args.format)
    print(f"Parsed {len(df)} rows from {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
