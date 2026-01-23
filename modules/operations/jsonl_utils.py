"""JSONL artifact utilities for batch and repair operations.

Provides shared functions for reading, writing, and parsing JSONL files
used in batch processing and repair workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for an image in a batch or repair operation."""
    
    image_name: str
    order_index: int
    page_number: Optional[int] = None
    custom_id: Optional[str] = None


def read_jsonl_records(jsonl_path: Path) -> List[Dict[str, Any]]:
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
                logger.warning(f"Invalid JSON at line {line_num} in {jsonl_path.name}: {e}")
    
    return records


def write_jsonl_record(jsonl_path: Path, record: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file.
        record: Dictionary to write as JSON line.
    """
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def extract_image_metadata(records: List[Dict[str, Any]]) -> List[ImageMetadata]:
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
                metadata_list.append(ImageMetadata(
                    image_name=meta.get("image_name", ""),
                    order_index=meta.get("order_index", -1),
                    page_number=meta.get("page_number"),
                    custom_id=meta.get("custom_id"),
                ))
    
    return metadata_list


def extract_batch_ids(records: List[Dict[str, Any]]) -> List[str]:
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
    records: List[Dict[str, Any]],
    deduplicate: bool = True,
) -> List[Dict[str, Any]]:
    """Extract transcription records from JSONL, filtering out metadata.
    
    Args:
        records: List of JSONL records.
        deduplicate: If True, keep only the latest record per image_name.
        
    Returns:
        List of transcription records with image_name and text_chunk.
    """
    # Filter out metadata records
    METADATA_KEYS = ("batch_session", "image_metadata", "batch_tracking")
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
        by_image: Dict[str, Dict[str, Any]] = {}
        for record in transcription_records:
            by_image[record["image_name"]] = record
        transcription_records = list(by_image.values())
    
    return transcription_records


def get_processed_image_names(jsonl_path: Path) -> set[str]:
    """Get set of image names that have been successfully processed.
    
    Args:
        jsonl_path: Path to JSONL file.
        
    Returns:
        Set of image names with valid transcriptions.
    """
    if not jsonl_path.exists():
        return set()
    
    records = read_jsonl_records(jsonl_path)
    transcriptions = extract_transcription_records(records, deduplicate=False)
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
        "image_metadata" in r and isinstance(r["image_metadata"], dict) 
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
    backup_path = file_path.parent / f"{file_path.stem}{suffix}_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    return backup_path


def find_companion_files(base_path: Path) -> Dict[str, Optional[Path]]:
    """Find companion files for a transcription (JSONL, debug, etc.).
    
    Supports both legacy (*_transcription.*) and new (*.*) naming conventions.
    
    Args:
        base_path: Base path (e.g., *.txt, *.jsonl, *_transcription.txt).
        
    Returns:
        Dictionary with keys: 'jsonl', 'debug', 'txt' mapping to paths or None.
    """
    parent = base_path.parent
    stem = base_path.stem.replace("_transcription", "")
    
    def find_file(stem: str, extensions: list) -> Optional[Path]:
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
        "debug": parent / f"{stem}_batch_submission_debug.json" if (parent / f"{stem}_batch_submission_debug.json").exists() else None,
        "txt": find_file(stem, [".txt"]),
    }
