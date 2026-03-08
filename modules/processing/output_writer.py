# modules/processing/output_writer.py
# Python 3.11+ • PEP8-compliant

"""Central output writer for transcription results.

Provides a single entry point for writing assembled transcription pages to
disk in multiple formats (txt, md, json).  All write locations in the
pipeline delegate to :func:`write_transcription_output` instead of
inlining their own file-writing logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from modules.processing.postprocess import postprocess_transcription
from modules.processing.response_parsing import format_page_line

logger = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS = {"txt", "md", "json"}

_FORMAT_TO_EXT = {
    "txt": ".txt",
    "md": ".md",
    "json": ".json",
}


def resolve_output_path(base_path: Path, output_format: str) -> Path:
    """Replace extension on *base_path* to match *output_format*.

    Args:
        base_path: Original output path (any extension).
        output_format: One of ``"txt"``, ``"md"``, ``"json"``.

    Returns:
        Path with the correct extension for the requested format.
    """
    ext = _FORMAT_TO_EXT.get(output_format, ".txt")
    return base_path.with_suffix(ext)


def write_transcription_output(
    pages: List[Dict],
    output_path: Path,
    output_format: str = "txt",
    postprocess: bool = True,
    postprocessing_config: Optional[Dict] = None,
) -> Path:
    """Write assembled transcription to disk in the requested format.

    Args:
        pages: List of page dicts, each containing ``"text"`` (str),
            ``"page_number"`` (int | None), and ``"image_name"`` (str | None).
        output_path: Base path (extension will be replaced to match format).
        output_format: ``"txt"``, ``"md"``, or ``"json"``.
        postprocess: Whether to apply post-processing (ignored for JSON).
        postprocessing_config: Configuration dict forwarded to
            :func:`postprocess_transcription`.

    Returns:
        The actual path written (with correct extension).
    """
    if output_format not in VALID_OUTPUT_FORMATS:
        logger.warning(
            "Unknown output format '%s'; falling back to 'txt'.", output_format
        )
        output_format = "txt"

    actual_path = resolve_output_path(output_path, output_format)
    actual_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        _write_json(pages, actual_path)
    elif output_format == "md":
        _write_md(pages, actual_path, postprocess, postprocessing_config)
    else:
        _write_txt(pages, actual_path, postprocess, postprocessing_config)

    return actual_path


def _write_txt(
    pages: List[Dict],
    path: Path,
    postprocess: bool,
    config: Optional[Dict],
) -> None:
    """Plain text — pages joined by newline, post-processing applied."""
    lines = []
    for page in pages:
        lines.append(
            format_page_line(
                page.get("text", ""),
                page.get("page_number"),
                page.get("image_name"),
            )
        )
    combined = "\n".join(lines)
    if postprocess:
        combined = postprocess_transcription(combined, config or {})
    path.write_text(combined, encoding="utf-8")


def _write_md(
    pages: List[Dict],
    path: Path,
    postprocess: bool,
    config: Optional[Dict],
) -> None:
    """Markdown with ``## <header>`` before each page block."""
    blocks = []
    for page in pages:
        image_name = (page.get("image_name") or "").strip()
        page_number = page.get("page_number")
        if image_name:
            header = f"## {image_name}"
        elif isinstance(page_number, int) and page_number > 0:
            header = f"## Page {page_number}"
        else:
            header = "## [unknown page]"

        text = (page.get("text") or "").strip()
        blocks.append(f"{header}\n\n{text}")

    combined = "\n\n".join(blocks)
    if postprocess:
        combined = postprocess_transcription(combined, config or {})
    path.write_text(combined, encoding="utf-8")


def _write_json(pages: List[Dict], path: Path) -> None:
    """Structured JSON array — no post-processing (preserves raw text)."""
    records = []
    for page in pages:
        records.append({
            "page_number": page.get("page_number"),
            "image_name": page.get("image_name"),
            "transcription": page.get("text", ""),
        })
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
