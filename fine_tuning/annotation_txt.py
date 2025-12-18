from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# Markers for the editable text format
# Format mirrors ChronoTranscriber eval format for consistency
PAGE_HEADER_RE = re.compile(r"^===\s*(.+?)\s*===\s*$", re.IGNORECASE)
SOURCE_IMAGE = "--- SOURCE_IMAGE ---"
OUTPUT_BEGIN = "--- OUTPUT_JSON_BEGIN ---"
OUTPUT_END = "--- OUTPUT_JSON_END ---"


@dataclass
class PageAnnotation:
    """Represents a single page annotation for fine-tuning."""
    page_index: int
    image_name: str
    image_path: Optional[str] = None
    output: Optional[Dict[str, Any]] = None


def read_annotations_txt(path: Path) -> List[PageAnnotation]:
    """
    Read annotations from an editable text file.
    
    Format:
    === page_0001_pre_processed.jpg ===
    --- SOURCE_IMAGE ---
    C:/path/to/images/page_0001_pre_processed.jpg
    --- OUTPUT_JSON_BEGIN ---
    {
      "transcription": "...",
      "no_transcribable_text": false,
      "transcription_not_possible": false
    }
    --- OUTPUT_JSON_END ---
    
    Args:
        path: Path to the editable text file.
        
    Returns:
        List of PageAnnotation objects.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    annotations: List[PageAnnotation] = []

    current_image_name: Optional[str] = None
    current_image_path: Optional[str] = None
    output_lines: List[str] = []
    mode: Optional[str] = None
    page_index = 0

    def _flush() -> None:
        nonlocal current_image_name, current_image_path, output_lines, mode, page_index
        if current_image_name is None:
            return

        raw_output = "\n".join(output_lines).strip()
        output: Optional[Dict[str, Any]]
        if not raw_output:
            output = None
        else:
            parsed = json.loads(raw_output)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Page {current_image_name}: output JSON must be an object (got {type(parsed).__name__})"
                )
            output = parsed

        annotations.append(
            PageAnnotation(
                page_index=page_index,
                image_name=current_image_name,
                image_path=current_image_path,
                output=output,
            )
        )

        page_index += 1
        current_image_name = None
        current_image_path = None
        output_lines = []
        mode = None

    for line_no, line in enumerate(lines, 1):
        header_match = PAGE_HEADER_RE.match(line.strip())
        if header_match:
            _flush()
            current_image_name = header_match.group(1).strip()
            continue

        if current_image_name is None:
            if not line.strip():
                continue
            raise ValueError(
                f"Line {line_no}: content found before first page header. "
                f"Expected '=== image_name ==='"
            )

        if line.strip() == SOURCE_IMAGE:
            mode = "source"
            continue
        if line.strip() == OUTPUT_BEGIN:
            mode = "output"
            continue
        if line.strip() == OUTPUT_END:
            mode = None
            continue

        if mode == "source":
            # The source line contains the image path
            if line.strip():
                current_image_path = line.strip()
        elif mode == "output":
            output_lines.append(line)

    _flush()

    if not annotations:
        raise ValueError("No pages found")

    return annotations


def write_annotations_txt(
    annotations: List[PageAnnotation],
    path: Path,
) -> None:
    """
    Write annotations to an editable text file.
    
    The format is designed for easy editing in Notepad++:
    - Clear page markers with image filename
    - Image path for reference (so annotator can open the original)
    - Pretty-printed JSON output for easy editing
    
    Args:
        annotations: List of PageAnnotation objects.
        path: Output path for the editable text file.
    """
    out_lines: List[str] = []
    
    for ann in sorted(annotations, key=lambda a: a.page_index):
        # Clear page header with image name
        out_lines.append(f"=== {ann.image_name} ===")
        
        # Source image section - path for annotator to locate the original
        out_lines.append(SOURCE_IMAGE)
        if ann.image_path:
            out_lines.append(ann.image_path)
        else:
            out_lines.append(f"[Image path not available - see: {ann.image_name}]")
        
        # Output section with pretty-printed JSON
        out_lines.append(OUTPUT_BEGIN)
        if ann.output is not None:
            # Pretty-print with 2-space indent for easy reading in Notepad++
            out_lines.append(json.dumps(ann.output, ensure_ascii=False, indent=2, sort_keys=True))
        out_lines.append(OUTPUT_END)
        
        # Blank line between pages for readability
        out_lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
