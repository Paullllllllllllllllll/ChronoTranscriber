from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def package_root() -> Path:
    """Return the fine_tuning package root directory."""
    return project_root() / "fine_tuning"


def artifacts_root() -> Path:
    """Return the artifacts directory for fine-tuning data."""
    return package_root() / "artifacts"


def editable_root() -> Path:
    """Return the directory for human-editable annotation files."""
    return artifacts_root() / "editable_txt"


def annotations_root() -> Path:
    """Return the directory for validated annotation JSONL files."""
    return artifacts_root() / "annotations_jsonl"


def datasets_root() -> Path:
    """Return the directory for final SFT datasets."""
    return artifacts_root() / "datasets"
