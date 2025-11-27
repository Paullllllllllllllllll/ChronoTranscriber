"""Compute accuracy metrics (CER/WER) for ChronoTranscriber outputs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from Levenshtein import distance as levenshtein_distance
from jiwer import wer


@dataclass
class PageMetrics:
    file_name: Optional[str]
    image_name: str
    cer: Optional[float]
    wer: Optional[float]
    hypothesis_length_chars: int
    reference_length_chars: int
    reference_length_words: int
    missing_reference: bool = False


def load_parsed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parsed predictions file not found: {path}")
    df = pd.read_csv(path)
    if "image_name" not in df.columns or "text_chunk" not in df.columns:
        raise ValueError("Parsed CSV must contain 'image_name' and 'text_chunk' columns.")
    return df


def load_reference_text(truth_dir: Path, image_name: str) -> Optional[str]:
    candidates = [
        truth_dir / f"{image_name}",
        truth_dir / f"{Path(image_name).stem}.txt",
        truth_dir / f"{image_name}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def compute_page_metrics(
    row: pd.Series, truth_dir: Path, normalize_whitespace: bool
) -> PageMetrics:
    hypothesis = str(row.get("text_chunk") or "")
    image_name = str(row.get("image_name") or "")
    file_name = row.get("file_name")

    reference = load_reference_text(truth_dir, image_name)
    if reference is None:
        return PageMetrics(
            file_name=file_name,
            image_name=image_name,
            cer=None,
            wer=None,
            hypothesis_length_chars=len(hypothesis),
            reference_length_chars=0,
            reference_length_words=0,
            missing_reference=True,
        )

    ref_text = reference
    hyp_text = hypothesis
    if normalize_whitespace:
        ref_text = " ".join(ref_text.split())
        hyp_text = " ".join(hyp_text.split())

    ref_chars = len(ref_text)
    hyp_chars = len(hyp_text)

    cer_value: Optional[float]
    if ref_chars == 0:
        cer_value = None
    else:
        cer_value = levenshtein_distance(ref_text, hyp_text) / ref_chars

    ref_words = len(ref_text.split())
    wer_value: Optional[float]
    if ref_words == 0:
        wer_value = None
    else:
        wer_value = wer(ref_text, hyp_text)

    return PageMetrics(
        file_name=file_name,
        image_name=image_name,
        cer=cer_value,
        wer=wer_value,
        hypothesis_length_chars=hyp_chars,
        reference_length_chars=ref_chars,
        reference_length_words=ref_words,
        missing_reference=False,
    )


def aggregate_metrics(pages: List[PageMetrics]) -> Dict[str, any]:
    summary: Dict[str, any] = {}
    valid_pages = [p for p in pages if not p.missing_reference and p.cer is not None]
    valid_wer_pages = [p for p in pages if not p.missing_reference and p.wer is not None]

    summary["page_count"] = len(pages)
    summary["with_reference"] = len([p for p in pages if not p.missing_reference])
    summary["missing_reference"] = len([p for p in pages if p.missing_reference])

    if valid_pages:
        summary["avg_cer"] = sum(p.cer for p in valid_pages if p.cer is not None) / len(valid_pages)
    else:
        summary["avg_cer"] = None

    if valid_wer_pages:
        summary["avg_wer"] = sum(p.wer for p in valid_wer_pages if p.wer is not None) / len(valid_wer_pages)
    else:
        summary["avg_wer"] = None

    per_doc: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, List[PageMetrics]] = {}
    for metric in pages:
        key = str(metric.file_name) if metric.file_name is not None else "<unknown>"
        grouped.setdefault(key, []).append(metric)

    for doc, doc_pages in grouped.items():
        doc_valid_cer = [p.cer for p in doc_pages if p.cer is not None and not p.missing_reference]
        doc_valid_wer = [p.wer for p in doc_pages if p.wer is not None and not p.missing_reference]
        per_doc[doc] = {
            "pages": len(doc_pages),
            "avg_cer": sum(doc_valid_cer) / len(doc_valid_cer) if doc_valid_cer else None,
            "avg_wer": sum(doc_valid_wer) / len(doc_valid_wer) if doc_valid_wer else None,
        }

    summary["per_document"] = per_doc
    summary["pages"] = [asdict(p) for p in pages]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute CER/WER for ChronoTranscriber outputs")
    parser.add_argument("--pred", required=True, type=Path, help="CSV produced by parse_jsonl.py")
    parser.add_argument("--truth", required=True, type=Path, help="Directory containing reference texts")
    parser.add_argument("--output", required=True, type=Path, help="Destination metrics JSON file")
    parser.add_argument(
        "--normalize-whitespace",
        action="store_true",
        help="Collapse whitespace before computing metrics",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_parsed_csv(args.pred)
    if not args.truth.exists():
        raise FileNotFoundError(f"Truth directory not found: {args.truth}")

    metrics: List[PageMetrics] = [
        compute_page_metrics(row, args.truth, args.normalize_whitespace)
        for _, row in df.iterrows()
    ]

    summary = aggregate_metrics(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote metrics for {len(metrics)} pages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
