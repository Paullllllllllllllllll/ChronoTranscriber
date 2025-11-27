"""Generate Markdown reports summarizing ChronoTranscriber evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> Dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_comparison(comparison_path: Optional[Path]) -> Optional[Dict]:
    if comparison_path is None:
        return None
    if not comparison_path.exists():
        return None
    with comparison_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_markdown(
    run_path: Path,
    metrics: Dict,
    comparison: Optional[Dict],
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# Evaluation Report – {run_path.name}")
    lines.append("")

    summary_table = {
        "Total Pages": metrics.get("page_count"),
        "Pages with Reference": metrics.get("with_reference"),
        "Missing Reference": metrics.get("missing_reference"),
        "Average CER": metrics.get("avg_cer"),
        "Average WER": metrics.get("avg_wer"),
    }

    lines.append("## Summary")
    for key, value in summary_table.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    per_doc = metrics.get("per_document", {})
    if per_doc:
        lines.append("## Per-Document Metrics")
        lines.append("")
        lines.append("| Document | Pages | Avg CER | Avg WER |")
        lines.append("|---|---:|---:|---:|")
        for doc, values in per_doc.items():
            lines.append(
                f"| {doc} | {values.get('pages')} | {values.get('avg_cer')} | {values.get('avg_wer')} |"
            )
        lines.append("")

    if comparison and comparison.get("runs"):
        lines.append("## Backend Comparison")
        lines.append("")
        lines.append("| Label | Backend | Avg CER | Avg WER | Total Cost (USD) |")
        lines.append("|---|---|---:|---:|---:|")
        for record in comparison["runs"]:
            lines.append(
                "| {label} | {backend} | {avg_cer} | {avg_wer} | {total_cost_usd} |".format(
                    label=record.get("label"),
                    backend=record.get("backend"),
                    avg_cer=record.get("avg_cer"),
                    avg_wer=record.get("avg_wer"),
                    total_cost_usd=record.get("total_cost_usd"),
                )
            )
        lines.append("")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_distributions(metrics: Dict, figures_dir: Path) -> None:
    pages = metrics.get("pages") or []
    if not pages:
        return

    df = pd.DataFrame(pages)
    if df.empty:
        return

    ensure_dir(figures_dir)

    if "cer" in df.columns and not df["cer"].dropna().empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["cer"].dropna(), bins=20)
        plt.title("CER Distribution")
        plt.xlabel("Character Error Rate")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "cer_distribution.png")
        plt.close()

    if "wer" in df.columns and not df["wer"].dropna().empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(df["wer"].dropna(), bins=20)
        plt.title("WER Distribution")
        plt.xlabel("Word Error Rate")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "wer_distribution.png")
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Markdown report for evaluation run")
    parser.add_argument("--run", required=True, type=Path, help="Path to evaluation run directory")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
        help="Optional comparison JSON produced by compare_backends.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for Markdown output (defaults to run/report.md)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional directory to store generated figures",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_path = args.run
    metrics = load_metrics(run_path / "metrics.json")
    comparison = load_comparison(args.comparison)

    output_path = args.output or (run_path / "report.md")
    figures_dir = args.figures_dir or (run_path / "figures")

    render_markdown(run_path, metrics, comparison, output_path)
    plot_distributions(metrics, figures_dir)

    print(f"Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
