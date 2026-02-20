"""
JSONL utilities for page-level evaluation.

This module provides functions to parse temporary JSONL files produced by
the transcriber, extract per-page transcriptions, and align them with
ground truth for CER/WER computation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PageTranscription:
    """Container for a single page's transcription data."""

    page_index: int
    image_name: str
    transcription: Optional[str]
    no_transcribable_text: bool = False
    transcription_not_possible: bool = False
    custom_id: Optional[str] = None

    def has_text(self) -> bool:
        """Check if page has valid transcription text."""
        return (
            self.transcription is not None
            and not self.no_transcribable_text
            and not self.transcription_not_possible
        )


@dataclass
class DocumentTranscriptions:
    """Container for all page transcriptions from a document."""

    source_name: str
    pages: List[PageTranscription] = field(default_factory=list)
    method: Optional[str] = None

    def get_page(self, index: int) -> Optional[PageTranscription]:
        """Get page by index."""
        for page in self.pages:
            if page.page_index == index:
                return page
        return None

    def page_count(self) -> int:
        """Return number of pages."""
        return len(self.pages)

    def to_text(self, separator: str = "\n\n") -> str:
        """Concatenate all page transcriptions into single text."""
        texts = []
        for page in sorted(self.pages, key=lambda p: p.page_index):
            if page.has_text() and page.transcription:
                texts.append(page.transcription)
        return separator.join(texts)


def parse_transcription_jsonl(jsonl_path: Path) -> DocumentTranscriptions:
    """
    Parse a temporary JSONL file and extract per-page transcriptions.

    Handles both synchronous and batch processing output formats:
    - Synchronous: Records with 'transcription' or 'text_chunk' fields
    - Batch: Records with 'image_metadata' for ordering and 'transcription' for text

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        DocumentTranscriptions object with all extracted pages
    """
    result = DocumentTranscriptions(source_name=jsonl_path.stem)

    if not jsonl_path.exists():
        return result

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Build image metadata index for ordering
    image_metadata: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if "image_metadata" in record:
            meta = record["image_metadata"]
            if isinstance(meta, dict):
                custom_id = meta.get("custom_id")
                if custom_id:
                    image_metadata[custom_id] = meta

    # Extract method if present
    for record in records:
        if "method" in record:
            result.method = record["method"]
            break

    # Process transcription records
    page_index = 0
    for record in records:
        # Skip metadata-only records
        if "batch_session" in record or "batch_tracking" in record:
            continue
        if "image_metadata" in record and "transcription" not in record and "text_chunk" not in record:
            continue

        # Extract transcription text
        transcription = record.get("transcription") or record.get("text_chunk")

        # Handle nested response structure (from batch API)
        if transcription is None and "response" in record:
            response = record["response"]
            if isinstance(response, dict):
                body = response.get("body", {})
                if "transcription" in body:
                    transcription = body["transcription"]

        # Skip if no transcription found
        if (
            transcription is None
            and "no_transcribable_text" not in record
            and "transcription_not_possible" not in record
        ):
            # Check if this is a content record without transcription
            if "file_name" not in record and "pre_processed_image" not in record:
                continue

        # Determine page index and image name
        image_name = ""
        custom_id = record.get("custom_id")

        if custom_id and custom_id in image_metadata:
            meta = image_metadata[custom_id]
            image_name = meta.get("image_name", "")
            order_index = meta.get("order_index", page_index)
        else:
            # Prefer image_name field, then pre_processed_image path, then file_name
            image_name = record.get("image_name", "")
            if not image_name:
                image_name = record.get("pre_processed_image", "")
                if isinstance(image_name, str) and ("/" in image_name or "\\" in image_name):
                    image_name = Path(image_name).name
            if not image_name:
                image_name = record.get("file_name", "")
            order_index = record.get("order_index", page_index)

        # Extract boolean flags
        no_text = record.get("no_transcribable_text", False)
        not_possible = record.get("transcription_not_possible", False)

        page = PageTranscription(
            page_index=order_index,
            image_name=image_name,
            transcription=transcription,
            no_transcribable_text=no_text,
            transcription_not_possible=not_possible,
            custom_id=custom_id,
        )
        result.pages.append(page)
        page_index += 1

    # Sort pages by index
    result.pages.sort(key=lambda p: p.page_index)

    return result


def find_jsonl_file(base_dir: Path, source_name: str) -> Optional[Path]:
    """
    Find the JSONL file for a given source in a directory.

    Searches for various naming patterns used by the transcriber.

    Args:
        base_dir: Directory to search in
        source_name: Name of the source (PDF name or folder name)

    Returns:
        Path to JSONL file or None if not found
    """
    if not base_dir.exists():
        return None

    base_name = Path(source_name).stem

    # Direct JSONL file
    direct = base_dir / f"{base_name}.jsonl"
    if direct.exists():
        return direct

    # JSONL in subfolder (standard transcriber output)
    subfolder = base_dir / base_name / f"{base_name}.jsonl"
    if subfolder.exists():
        return subfolder

    # Legacy naming with _transcription suffix
    legacy = base_dir / f"{base_name}_transcription.jsonl"
    if legacy.exists():
        return legacy

    # Search exact subfolder for any JSONL
    subfolder_dir = base_dir / base_name
    if subfolder_dir.exists() and subfolder_dir.is_dir():
        jsonl_files = list(subfolder_dir.glob("*.jsonl"))
        if jsonl_files:
            return sorted(jsonl_files)[0]

    # Search hash-suffixed subfolders (transcriber output: e.g. address_books-f12f1e9c/)
    for hashed_dir in sorted(base_dir.glob(f"{base_name}-*")):
        if hashed_dir.is_dir():
            jsonl_files = list(hashed_dir.glob("*.jsonl"))
            if jsonl_files:
                return sorted(jsonl_files)[0]

    return None


def load_page_transcriptions(
    output_dir: Path,
    category: str,
    model_name: str,
    source_name: str,
) -> Optional[DocumentTranscriptions]:
    """
    Load page-level transcriptions from model output.

    Args:
        output_dir: Base output directory
        category: Dataset category
        model_name: Model identifier
        source_name: Source file/folder name

    Returns:
        DocumentTranscriptions or None if not found
    """
    model_output_dir = output_dir / category / model_name
    jsonl_path = find_jsonl_file(model_output_dir, source_name)

    if jsonl_path is None:
        return None

    return parse_transcription_jsonl(jsonl_path)


def load_ground_truth_pages(
    ground_truth_dir: Path,
    category: str,
    source_name: str,
) -> Optional[DocumentTranscriptions]:
    """
    Load page-level ground truth transcriptions.

    Ground truth JSONL files follow the same format as model output.

    Args:
        ground_truth_dir: Base ground truth directory
        category: Dataset category
        source_name: Source file/folder name

    Returns:
        DocumentTranscriptions or None if not found
    """
    gt_category_dir = ground_truth_dir / category
    jsonl_path = find_jsonl_file(gt_category_dir, source_name)

    if jsonl_path is None:
        return None

    return parse_transcription_jsonl(jsonl_path)


def align_pages(
    hypothesis: DocumentTranscriptions,
    reference: DocumentTranscriptions,
) -> List[Tuple[Optional[PageTranscription], Optional[PageTranscription]]]:
    """
    Align hypothesis pages with reference pages by index.

    Returns pairs of (hypothesis_page, reference_page) where either may be None
    if the page exists only in one document.

    Args:
        hypothesis: Model output transcriptions
        reference: Ground truth transcriptions

    Returns:
        List of aligned page pairs
    """
    all_indices = set()
    for page in hypothesis.pages:
        all_indices.add(page.page_index)
    for page in reference.pages:
        all_indices.add(page.page_index)

    aligned = []
    for idx in sorted(all_indices):
        hyp_page = hypothesis.get_page(idx)
        ref_page = reference.get_page(idx)
        aligned.append((hyp_page, ref_page))

    return aligned


# =============================================================================
# Ground Truth Editing Utilities
# =============================================================================

PAGE_MARKER_PATTERN = re.compile(r"^=== (.+?) ===\s*$", re.MULTILINE)


def export_pages_to_editable_txt(
    doc: DocumentTranscriptions,
    output_path: Path,
) -> None:
    """
    Export page transcriptions to an editable text file with page markers.

    Format:
    === page_0001_pre_processed.jpg ===
    {transcription text}

    === page_0002_pre_processed.jpg ===
    {transcription text}
    ...

    Args:
        doc: DocumentTranscriptions to export
        output_path: Path for output text file
    """
    lines = []

    for page in sorted(doc.pages, key=lambda p: p.page_index):
        # Use image_name if available, otherwise fall back to page number
        if page.image_name:
            marker = f"=== {page.image_name} ==="
        else:
            marker = f"=== page_{page.page_index + 1:04d} ==="
        lines.append(marker)

        if page.has_text() and page.transcription:
            lines.append(page.transcription)
        elif page.no_transcribable_text:
            lines.append("[NO TRANSCRIBABLE TEXT]")
        elif page.transcription_not_possible:
            lines.append("[TRANSCRIPTION NOT POSSIBLE]")
        else:
            lines.append("")

        lines.append("")  # Blank line between pages

    output_path.write_text("\n".join(lines), encoding="utf-8")


def import_pages_from_editable_txt(
    txt_path: Path,
    original_doc: Optional[DocumentTranscriptions] = None,
) -> DocumentTranscriptions:
    """
    Import page transcriptions from an edited text file.

    Parses the text file format with === image_name === markers.

    Args:
        txt_path: Path to edited text file
        original_doc: Optional original document to preserve metadata

    Returns:
        DocumentTranscriptions with updated transcriptions
    """
    content = txt_path.read_text(encoding="utf-8")

    # Split by page markers
    parts = PAGE_MARKER_PATTERN.split(content)

    # parts[0] is content before first marker (should be empty/whitespace)
    # parts[1], parts[3], parts[5], ... are image names (or legacy page numbers)
    # parts[2], parts[4], parts[6], ... are page contents

    source_name = txt_path.stem.replace("_editable", "")
    result = DocumentTranscriptions(source_name=source_name)

    if original_doc:
        result.method = original_doc.method

    # Build lookup from image_name to original page
    image_name_to_page: Dict[str, PageTranscription] = {}
    if original_doc:
        for p in original_doc.pages:
            if p.image_name:
                image_name_to_page[p.image_name] = p

    i = 1
    page_index_counter = 0
    while i < len(parts):
        marker_text = parts[i].strip()

        if i + 1 < len(parts):
            text = parts[i + 1].strip()
        else:
            text = ""

        # Handle special markers
        no_text = text == "[NO TRANSCRIBABLE TEXT]"
        not_possible = text == "[TRANSCRIPTION NOT POSSIBLE]"

        if no_text or not_possible:
            transcription = None
        else:
            transcription = text if text else None

        # Try to find matching page from original doc by image name
        image_name = marker_text
        custom_id = None
        page_index = page_index_counter

        if marker_text in image_name_to_page:
            orig_page = image_name_to_page[marker_text]
            page_index = orig_page.page_index
            custom_id = orig_page.custom_id
        elif original_doc:
            # Legacy format: try parsing as page number
            if marker_text.startswith("page_"):
                try:
                    page_index = int(marker_text.replace("page_", "")) - 1
                except ValueError:
                    pass
            elif marker_text.isdigit():
                page_index = int(marker_text) - 1
            # Try to get metadata from original by index
            orig_page = original_doc.get_page(page_index)
            if orig_page:
                image_name = orig_page.image_name or marker_text
                custom_id = orig_page.custom_id

        page = PageTranscription(
            page_index=page_index,
            image_name=image_name,
            transcription=transcription,
            no_transcribable_text=no_text,
            transcription_not_possible=not_possible,
            custom_id=custom_id,
        )
        result.pages.append(page)

        page_index_counter += 1
        i += 2

    result.pages.sort(key=lambda p: p.page_index)
    return result


def write_ground_truth_jsonl(
    doc: DocumentTranscriptions,
    output_path: Path,
) -> None:
    """
    Write document transcriptions to a ground truth JSONL file.

    Args:
        doc: DocumentTranscriptions to write
        output_path: Path for output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for page in sorted(doc.pages, key=lambda p: p.page_index):
            record = {
                "page_index": page.page_index,
                "image_name": page.image_name,
                "transcription": page.transcription,
                "no_transcribable_text": page.no_transcribable_text,
                "transcription_not_possible": page.transcription_not_possible,
            }
            if page.custom_id:
                record["custom_id"] = page.custom_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
