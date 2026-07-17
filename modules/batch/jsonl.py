"""JSONL artifact utilities for batch and repair operations.

Provides shared functions for reading, writing, and parsing JSONL files
used in batch processing and repair workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


# Resume-artifact format version. Temp JSONLs written by the transcription
# pipeline carry a marker record so a later resume can refuse artifacts written
# by an incompatible version instead of silently mis-ordering or dropping pages.
RESUME_FORMAT_VERSION = 1
_RESUME_MARKER_KEY = "resume_format"

# Keys that identify non-transcription (metadata/marker) JSONL records.
METADATA_KEYS = (
    "batch_session",
    "image_metadata",
    "batch_tracking",
    "file_provenance",
    _RESUME_MARKER_KEY,
)


class ResumeFormatError(RuntimeError):
    """Raised when a resume artifact lacks a compatible format-version marker."""


def resume_marker_record() -> dict[str, Any]:
    """Return the resume-format marker record for the current version."""
    return {_RESUME_MARKER_KEY: {"version": RESUME_FORMAT_VERSION}}


def _record_is_transcription(record: dict[str, Any]) -> bool:
    """True when a JSONL record carries a page transcription (not metadata)."""
    if any(k in record for k in METADATA_KEYS):
        return False
    return bool(record.get("image_name")) and record.get("text_chunk") is not None


def read_resume_version(records: list[dict[str, Any]]) -> int | None:
    """Return the resume-format version marked in *records*, or None."""
    for record in records:
        marker = record.get(_RESUME_MARKER_KEY)
        if isinstance(marker, dict):
            version = marker.get("version")
            if isinstance(version, int):
                return version
    return None


def ensure_resume_marker(
    jsonl_path: Path, *, records: list[dict[str, Any]] | None = None
) -> None:
    """Append the current resume-format marker if the file lacks one.

    Written lazily before the first result is streamed so fresh artifacts are
    always versioned; a no-op when a marker is already present.

    Args:
        jsonl_path: Path to the JSONL file.
        records: Pre-parsed records for the file, to skip a redundant read. Must
            reflect the file's *current* on-disk state (no writes since it was
            parsed); pass None to read fresh. The read-then-maybe-append
            behavior is unchanged: the append still targets ``jsonl_path``.
    """
    if records is None:
        records = read_jsonl_records(jsonl_path)
    if read_resume_version(records) is not None:
        return
    write_jsonl_record(jsonl_path, resume_marker_record())


def verify_resume_compatible(
    jsonl_path: Path, *, records: list[dict[str, Any]] | None = None
) -> None:
    """Refuse to resume an artifact written by an incompatible version.

    Only enforced when the artifact already contains page transcriptions to
    resume: an unversioned artifact carrying prior results would otherwise be
    re-ordered/merged under assumptions that no longer hold (see decision 1).
    Fresh or marker-only files pass through untouched.

    Args:
        jsonl_path: Path to the JSONL file (used only for the error message).
        records: Pre-parsed records for the file, to skip a redundant read;
            pass None to read fresh.
    """
    if records is None:
        records = read_jsonl_records(jsonl_path)
    if not any(_record_is_transcription(r) for r in records):
        return
    version = read_resume_version(records)
    if version != RESUME_FORMAT_VERSION:
        found = "none" if version is None else str(version)
        raise ResumeFormatError(
            f"Resume artifact {jsonl_path.name} has format version {found}, but "
            f"this version of ChronoTranscriber writes version "
            f"{RESUME_FORMAT_VERSION}. Re-run from scratch (use --force/overwrite) "
            f"or finish the job with the version that created it."
        )


@dataclass
class ImageMetadata:
    """Metadata for an image in a batch or repair operation."""

    image_name: str
    order_index: int
    page_number: int | None = None
    custom_id: str | None = None


def read_jsonl_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file.

    Returns:
        List of parsed JSON records.
    """
    records: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return records

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON at line {line_num} in {jsonl_path.name}: {e}"
                )

    return records


def write_jsonl_record(jsonl_path: Path, record: dict[str, Any]) -> None:
    """Append a single record to a JSONL file.

    Args:
        jsonl_path: Path to JSONL file.
        record: Dictionary to write as JSON line.
    """
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_image_metadata(records: list[dict[str, Any]]) -> list[ImageMetadata]:
    """Extract image metadata from JSONL records.

    Args:
        records: List of JSONL records.

    Returns:
        List of ImageMetadata objects.
    """
    metadata_list = []

    for record in records:
        if "image_metadata" in record:
            meta = record["image_metadata"]
            if isinstance(meta, dict):
                metadata_list.append(
                    ImageMetadata(
                        image_name=meta.get("image_name", ""),
                        order_index=meta.get("order_index", -1),
                        page_number=meta.get("page_number"),
                        custom_id=meta.get("custom_id"),
                    )
                )

    return metadata_list


def extract_batch_ids(records: list[dict[str, Any]]) -> list[str]:
    """Extract batch IDs from JSONL records.

    Args:
        records: List of JSONL records.

    Returns:
        List of batch ID strings.
    """
    batch_ids = []

    for record in records:
        if "batch_tracking" in record:
            batch_id = record["batch_tracking"].get("batch_id")
            if batch_id:
                batch_ids.append(batch_id)

    return batch_ids


def extract_transcription_records(
    records: list[dict[str, Any]],
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """Extract transcription records from JSONL, filtering out metadata.

    Args:
        records: List of JSONL records.
        deduplicate: If True, keep only the latest record per image_name.

    Returns:
        List of transcription records with image_name and text_chunk.
    """
    # Filter out metadata records (METADATA_KEYS defined at module level).
    transcription_records = []

    for record in records:
        # Skip metadata records
        if any(k in record for k in METADATA_KEYS):
            continue
        # Only include records with valid transcription
        if record.get("image_name") and record.get("text_chunk") is not None:
            transcription_records.append(record)

    if deduplicate and transcription_records:
        # Keep latest record per image_name (last occurrence wins)
        by_image: dict[str, dict[str, Any]] = {}
        for record in transcription_records:
            by_image[record["image_name"]] = record
        transcription_records = list(by_image.values())

    return transcription_records


def _is_error_placeholder(text: Any) -> bool:
    """True when a transcription text is a ``[transcription error ...]`` marker."""
    return isinstance(text, str) and text.lstrip().lower().startswith(
        "[transcription error"
    )


def get_processed_image_names(
    jsonl_path: Path,
    *,
    exclude_errors: bool = False,
    records: list[dict[str, Any]] | None = None,
) -> set[str]:
    """Get set of image names that have been successfully processed.

    Args:
        jsonl_path: Path to JSONL file.
        exclude_errors: When True, image names whose latest transcription is a
            ``[transcription error]`` placeholder are omitted, so ``--retry-errors``
            re-processes them (decision 13).
        records: Pre-parsed records for the file, to skip a redundant read;
            pass None to read fresh.

    Returns:
        Set of image names with valid transcriptions.
    """
    if records is None:
        if not jsonl_path.exists():
            return set()
        records = read_jsonl_records(jsonl_path)
    # Deduplicate so the *latest* record per image decides its error status.
    transcriptions = extract_transcription_records(records, deduplicate=exclude_errors)
    if exclude_errors:
        return {
            r["image_name"]
            for r in transcriptions
            if not _is_error_placeholder(r.get("text_chunk"))
        }
    return {r["image_name"] for r in transcriptions}


def is_batch_jsonl(jsonl_path: Path) -> bool:
    """Check if a JSONL file contains batch processing markers.

    Args:
        jsonl_path: Path to JSONL file.

    Returns:
        True if file contains batch markers, False otherwise.
    """
    if not jsonl_path.exists():
        return False

    records = read_jsonl_records(jsonl_path)

    has_batch_session = any("batch_session" in r for r in records)
    has_batch_tracking = any("batch_tracking" in r for r in records)
    has_batch_metadata = any(
        "image_metadata" in r
        and isinstance(r["image_metadata"], dict)
        and r["image_metadata"].get("custom_id")
        for r in records
    )

    return has_batch_session and (has_batch_tracking or has_batch_metadata)


def backup_file(file_path: Path, suffix: str = "_backup") -> Path:
    """Create a backup copy of a file.

    Args:
        file_path: Path to file to backup.
        suffix: Suffix to add to backup filename.

    Returns:
        Path to backup file.
    """
    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        file_path.parent / f"{file_path.stem}{suffix}_{timestamp}{file_path.suffix}"
    )
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")

    return backup_path


def find_companion_files(base_path: Path) -> dict[str, Path | None]:
    """Find companion files for a transcription (JSONL, debug, etc.).

    Supports both legacy (*_transcription.*) and new (*.*) naming conventions.

    Args:
        base_path: Base path (e.g., *.txt, *.jsonl, *_transcription.txt).

    Returns:
        Dictionary with keys: 'jsonl', 'debug', 'txt' mapping to paths or None.
    """
    parent = base_path.parent
    stem = base_path.stem.replace("_transcription", "")

    def find_file(stem: str, extensions: list[str]) -> Path | None:
        """Find file with any of the given extensions, trying new format first."""
        for ext in extensions:
            # Try new format first
            new_path = parent / f"{stem}{ext}"
            if new_path.exists():
                return new_path
            # Try legacy format
            legacy_path = parent / f"{stem}_transcription{ext}"
            if legacy_path.exists():
                return legacy_path
        return None

    return {
        "jsonl": find_file(stem, [".jsonl"]),
        "debug": parent / f"{stem}_batch_submission_debug.json"
        if (parent / f"{stem}_batch_submission_debug.json").exists()
        else None,
        "txt": find_file(stem, [".txt"]),
    }
