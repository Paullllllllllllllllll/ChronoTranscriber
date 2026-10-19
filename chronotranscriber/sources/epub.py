from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ebooklib
from ebooklib import epub
from lxml import html

from chronotranscriber.sources._text import ebook_plain_text, normalize_text

logger = logging.getLogger(__name__)

# Navigation, cover and toc documents to drop from the reading order. The match
# is anchored on the file stem, so chapters whose names merely contain these
# tokens (discovery.xhtml, navarra.xhtml, coverage.xhtml) stay.
_NAV_DOC_RE = re.compile(r"^(nav|toc|cover)([._-]|$)")


@dataclass(slots=True)
class EPUBTextExtraction:
    """Represents the result of extracting plain text from an EPUB file."""

    title: str | None
    authors: list[str]
    sections: list[str]

    def to_plain_text(self) -> str:
        """Render the extracted content (with metadata) as plain text."""
        return ebook_plain_text(self.title, self.authors, self.sections)


class EPUBProcessor:
    """Utility for extracting text content from EPUB files."""

    def __init__(self, epub_path: Path) -> None:
        self.epub_path = epub_path

    def extract_text(
        self, section_indices: list[int] | None = None
    ) -> EPUBTextExtraction:
        """Extract the EPUB text content and metadata using EbookLib.

        Args:
            section_indices: Optional list of 0-based section indices to extract.
                If None, all sections are extracted.
        """
        try:
            book = epub.read_epub(str(self.epub_path))
        except Exception:  # pragma: no cover - surface errors upstream
            logger.exception("Failed to read EPUB: %s", self.epub_path)
            raise

        title = _first_metadata_value(book.get_metadata("DC", "title"))
        author_entries = book.get_metadata("DC", "creator")
        authors = [
            entry[0].strip() for entry in author_entries if entry and entry[0].strip()
        ]

        all_items = self._ordered_documents(book)

        if section_indices is not None:
            all_items = [
                all_items[i] for i in section_indices if 0 <= i < len(all_items)
            ]

        sections: list[str] = []
        for item in all_items:
            try:
                content_bytes = item.get_content()
                text = _html_bytes_to_text(content_bytes)
                if text:
                    sections.append(text)
            except Exception as exc:  # noqa: BLE001 - a broken item is skipped
                logger.warning(
                    "Skipping EPUB item %s due to parsing error: %s",
                    getattr(item, "file_name", "<unknown>"),
                    exc,
                )

        return EPUBTextExtraction(title=title, authors=authors, sections=sections)

    @staticmethod
    def _ordered_documents(book: Any) -> list[Any]:
        """Return content documents in spine (reading) order.

        The manifest order (``get_items_of_type(ITEM_DOCUMENT)``) is not the
        reading order and includes nav/cover documents; iterate the spine
        instead and drop navigation, cover and toc items. Falls back to
        manifest order when the book exposes no usable spine.
        """
        spine = getattr(book, "spine", None)
        get_by_id = getattr(book, "get_item_with_id", None)
        ordered: list[Any] = []
        if spine and callable(get_by_id):
            for entry in spine:
                idref = entry[0] if isinstance(entry, (tuple, list)) else entry
                item = get_by_id(idref)
                if item is None:
                    continue
                get_type = getattr(item, "get_type", None)
                if callable(get_type) and get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                name = ""
                get_name = getattr(item, "get_name", None)
                if callable(get_name):
                    name = (get_name() or "").lower()
                if _NAV_DOC_RE.match(Path(name).stem):
                    continue
                ordered.append(item)
        if ordered:
            return ordered
        return list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))


def _html_bytes_to_text(content: bytes) -> str:
    """Convert XHTML bytes from the EPUB into normalized plain text."""
    if not content:
        return ""

    document = html.fromstring(content)
    html.etree.strip_elements(document, "script", "style", with_tail=False)

    text_content = document.text_content()
    return normalize_text(text_content)


def _first_metadata_value(entries: list[tuple[str, dict[str, str]]]) -> str | None:
    """Return the first metadata value from an EbookLib metadata list."""
    for value, _attrs in entries or []:
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None
