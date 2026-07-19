# modules/postprocess/writer.py
# Python 3.11+ • PEP8-compliant

"""Central output writer for transcription results.

Provides a single entry point for writing assembled transcription pages to
disk in multiple formats (txt, md, json).  All write locations in the
pipeline delegate to :func:`write_transcription_output` instead of
inlining their own file-writing logic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from modules.infra.logger import setup_logger
from modules.llm.response_parsing import format_page_line
from modules.postprocess.text import postprocess_transcription

logger = setup_logger(__name__)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically (temp file + ``os.replace``).

    A crash mid-write must never leave a truncated final output on disk: an
    item-level resume treats any existing, non-empty output as COMPLETE and
    would skip re-transcription, silently losing the tail of the document. By
    writing to a sibling temp file first and atomically renaming it into place,
    the target file either holds the previous content or the complete new
    content, never a partial write. UTF-8 encoding and ``\\n`` newlines are
    preserved so the on-disk bytes match the previous ``Path.write_text``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        # Never leave the temp artifact behind on a failed write/replace.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


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
    pages: list[dict[str, Any]],
    output_path: Path,
    output_format: str = "txt",
    postprocess: bool = True,
    postprocessing_config: dict[str, Any] | None = None,
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
    pages: list[dict[str, Any]],
    path: Path,
    postprocess: bool,
    config: dict[str, Any] | None,
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
    _atomic_write_text(path, combined)


def _write_md(
    pages: list[dict[str, Any]],
    path: Path,
    postprocess: bool,
    config: dict[str, Any] | None,
) -> None:
    """Markdown — pages joined by double newline, post-processing applied.

    Uses :func:`format_page_line` so that only failure placeholders receive
    a header (image name or page number); normal transcription text is
    emitted as-is, preserving LLM-produced ``<page_number>`` tags as the
    authoritative page identification.
    """
    blocks = []
    for page in pages:
        blocks.append(
            format_page_line(
                page.get("text", ""),
                page.get("page_number"),
                page.get("image_name"),
            )
        )
    combined = "\n\n".join(blocks)
    if postprocess:
        combined = postprocess_transcription(combined, config or {})
    _atomic_write_text(path, combined)


def _write_json(pages: list[dict[str, Any]], path: Path) -> None:
    """Structured JSON array — no post-processing (preserves raw text)."""
    records = []
    for page in pages:
        records.append(
            {
                "page_number": page.get("page_number"),
                "image_name": page.get("image_name"),
                "transcription": page.get("text", ""),
            }
        )
    _atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2))
