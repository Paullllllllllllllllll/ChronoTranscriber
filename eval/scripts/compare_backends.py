"""Compare multiple ChronoTranscriber evaluation runs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


@dataclass
class RunSummary:
    run_path: Path
    label: str
    backend: Optional[str]
    pages: int
    with_reference: int
    avg_cer: Optional[float]
    avg_wer: Optional[float]
    total_cost: Optional[float]


DEFAULT_METRICS_FILE = "metrics.json"
DEFAULT_COST_FILE_CANDIDATES = ("costs.json", "cost_summary.json", "cost.json")
DEFAULT_CONFIG_FILE = "run_config.yaml"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metrics(path: Path) -> Dict:
    data = load_json(path)
    if "page_count" not in data:
        raise ValueError(f"Metrics file missing 'page_count': {path}")
    return data


def load_cost(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_backend(config_path: Path) -> Optional[str]:
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError:
            return None
    if isinstance(data, dict):
        return str(data.get("backend") or data.get("method") or data.get("model") or None)
    return None


def discover_cost_file(run_path: Path) -> Optional[Path]:
    for candidate in DEFAULT_COST_FILE_CANDIDATES:
        path = run_path / candidate
        if path.exists():
            return path
    return None


def summarize_run(run_path: Path, label: Optional[str]) -> RunSummary:
    metrics_path = run_path / DEFAULT_METRICS_FILE
    metrics = load_metrics(metrics_path)

    backend = load_backend(run_path / DEFAULT_CONFIG_FILE)

    cost_path = discover_cost_file(run_path)
    total_cost: Optional[float] = None
    if cost_path is not None:
        cost_data = load_cost(cost_path)
        if isinstance(cost_data, dict):
            total_cost = float(cost_data.get("total_cost_usd")) if cost_data.get("total_cost_usd") is not None else None

    label_to_use = label or run_path.name

    return RunSummary(
        run_path=run_path,
        label=label_to_use,
        backend=backend,
        pages=int(metrics.get("page_count", 0)),
        with_reference=int(metrics.get("with_reference", 0)),
        avg_cer=metrics.get("avg_cer"),
        avg_wer=metrics.get("avg_wer"),
        total_cost=total_cost,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple evaluation runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to evaluation run directories",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels corresponding to --runs order",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON summary file",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional CSV table output",
    )
    return parser


def create_dataframe(summaries: List[RunSummary]) -> pd.DataFrame:
    records = []
    for summary in summaries:
        records.append(
            {
                "label": summary.label,
                "backend": summary.backend,
                "pages": summary.pages,
                "with_reference": summary.with_reference,
                "avg_cer": summary.avg_cer,
                "avg_wer": summary.avg_wer,
                "total_cost_usd": summary.total_cost,
            }
        )
    df = pd.DataFrame(records)
    return df


def compute_rankings(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    results: Dict[str, List[Dict[str, float]]] = {}
    if "avg_cer" in df.columns and not df["avg_cer"].isna().all():
        ranked = df.sort_values("avg_cer", na_position="last")
        results["cer"] = ranked[["label", "backend", "avg_cer"]].to_dict(orient="records")
    if "avg_wer" in df.columns and not df["avg_wer"].isna().all():
        ranked = df.sort_values("avg_wer", na_position="last")
        results["wer"] = ranked[["label", "backend", "avg_wer"]].to_dict(orient="records")
    if "total_cost_usd" in df.columns and not df["total_cost_usd"].isna().all():
        ranked = df.sort_values("total_cost_usd", na_position="last")
        results["cost"] = ranked[["label", "backend", "total_cost_usd"]].to_dict(orient="records")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    labels = args.labels or []
    if labels and len(labels) != len(args.runs):
        raise ValueError("Number of labels must match number of runs or be omitted.")

    summaries = [
        summarize_run(run, labels[idx] if labels else None)
        for idx, run in enumerate(args.runs)
    ]

    df = create_dataframe(summaries)
    rankings = compute_rankings(df)

    output_data = {
        "runs": df.to_dict(orient="records"),
        "rankings": rankings,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    if args.export_csv is not None:
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.export_csv, index=False)

    print(f"Wrote comparison summary for {len(summaries)} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
