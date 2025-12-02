from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import ebooklib
from ebooklib import epub
from lxml import html

from modules.core.safe_paths import create_safe_directory_name, create_safe_filename

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EPUBTextExtraction:
    """Represents the result of extracting plain text from an EPUB file."""

    title: Optional[str]
    authors: List[str]
    sections: List[str]

    def to_plain_text(self) -> str:
        """Render the extracted content (with metadata) as plain text."""
        lines: List[str] = []

        if self.title:
            lines.append(f"# Title: {self.title.strip()}")
        if self.authors:
            author_line = ", ".join(author.strip() for author in self.authors if author.strip())
            if author_line:
                lines.append(f"# Author(s): {author_line}")

        if lines:
            lines.append("")  # Add a blank line between metadata and content

        for section in self.sections:
            normalized = _normalize_text(section)
            if not normalized:
                continue
            lines.append(normalized)
            lines.append("")

        # Remove trailing blank lines while keeping a terminating newline for POSIX compliance
        while lines and lines[-1] == "":
            lines.pop()

        rendered = "\n".join(lines)
        return rendered + ("\n" if rendered and not rendered.endswith("\n") else "")


class EPUBProcessor:
    """Utility for extracting text content from EPUB files."""

    def __init__(self, epub_path: Path) -> None:
        self.epub_path = epub_path

    def extract_text(self) -> EPUBTextExtraction:
        """Extract the EPUB text content and metadata using EbookLib."""
        try:
            book = epub.read_epub(str(self.epub_path))
        except Exception as exc:  # pragma: no cover - surface errors upstream
            logger.exception("Failed to read EPUB: %s", self.epub_path)
            raise

        title = _first_metadata_value(book.get_metadata("DC", "title"))
        author_entries = book.get_metadata("DC", "creator")
        authors = [entry[0].strip() for entry in author_entries if entry and entry[0].strip()]

        sections: List[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                content_bytes = item.get_content()
                text = _html_bytes_to_text(content_bytes)
                if text:
                    sections.append(text)
            except Exception as exc:
                logger.warning("Skipping EPUB item %s due to parsing error: %s", getattr(item, "file_name", "<unknown>"), exc)

        return EPUBTextExtraction(title=title, authors=authors, sections=sections)

    def prepare_output_folder(self, epub_output_dir: Path) -> tuple[Path, Path]:
        """Prepare a deterministic output folder and text file path for this EPUB."""
        safe_dir_name = create_safe_directory_name(self.epub_path.stem)
        parent_folder = epub_output_dir / safe_dir_name
        parent_folder.mkdir(parents=True, exist_ok=True)

        # Create safe filename (truncated with hash if needed, considering full path length)
        output_txt_name = create_safe_filename(self.epub_path.stem, ".txt", parent_folder)
        output_txt_path = parent_folder / output_txt_name
        return parent_folder, output_txt_path


def _html_bytes_to_text(content: bytes) -> str:
    """Convert XHTML bytes from the EPUB into normalized plain text."""
    if not content:
        return ""

    # lxml.html.fromstring expects bytes/str; wrap in BytesIO for clarity
    document = html.fromstring(BytesIO(content).getvalue())

    # Remove scripts and styles explicitly if present
    html.etree.strip_elements(document, "script", "style", with_tail=False)

    text_content = document.text_content()
    return _normalize_text(text_content)


def _normalize_text(value: str) -> str:
    """Collapse extraneous whitespace while preserving single blank lines."""
    if not value:
        return ""

    lines = [line.strip() for line in value.splitlines()]
    normalized_lines: List[str] = []

    for line in lines:
        if line:
            normalized_lines.append(line)
        elif normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")

    # Remove trailing blanks
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def _first_metadata_value(entries: List[tuple[str, dict]]) -> Optional[str]:
    """Return the first metadata value from an EbookLib metadata list."""
    for value, _attrs in entries or []:  # type: ignore[assignment]
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None
