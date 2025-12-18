from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fine_tuning.annotation_txt import PageAnnotation
from fine_tuning.jsonl_io import write_jsonl


def build_annotation_records(
    *,
    schema_name: Optional[str] = None,
    annotations: List[PageAnnotation],
    source_id: Optional[str] = None,
    annotator_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build annotation records from page annotations.
    
    Args:
        schema_name: Name of the schema used (e.g., 'markdown_transcription_schema').
        annotations: List of PageAnnotation objects.
        source_id: Identifier for the source document/folder.
        annotator_id: Identifier for the annotator.
        
    Returns:
        List of annotation records ready for JSONL serialization.
    """
    records: List[Dict[str, Any]] = []
    
    for ann in sorted(annotations, key=lambda a: a.page_index):
        if ann.output is None:
            raise ValueError(f"Page {ann.image_name}: missing output JSON")

        record: Dict[str, Any] = {
            "page_index": ann.page_index,
            "image_name": ann.image_name,
            "image_path": ann.image_path,
            "output": ann.output,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        if schema_name is not None:
            record["schema_name"] = schema_name
        if source_id is not None:
            record["source_id"] = source_id
        if annotator_id is not None:
            record["annotator_id"] = annotator_id

        records.append(record)

    return records


def write_annotations_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write annotation records to a JSONL file."""
    write_jsonl(path, records)
