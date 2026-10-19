"""MOBI file processing utilities.

This module provides functionality to extract text content from MOBI (Kindle) files
by unpacking them to intermediate formats (EPUB/HTML) and extracting text.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import mobi
from lxml import html

from chronotranscriber.sources._text import ebook_plain_text
from chronotranscriber.sources._text import normalize_text as normalize_text
from chronotranscriber.sources.epub import EPUBProcessor, EPUBTextExtraction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MOBITextExtraction:
    """Represents the result of extracting plain text from a MOBI file."""

    title: str | None
    authors: list[str]
    sections: list[str]
    source_format: str  # epub, html, pdf or text

    def to_plain_text(self) -> str:
        """Render the extracted content (with metadata) as plain text."""
        return ebook_plain_text(self.title, self.authors, self.sections)


class MOBIProcessor:
    """Utility for extracting text content from MOBI files."""

    def __init__(self, mobi_path: Path) -> None:
        self.mobi_path = mobi_path
        self._tempdir: Path | None = None

    def extract_text(
        self, section_indices: list[int] | None = None
    ) -> MOBITextExtraction:
        """Extract the MOBI text content by unpacking to intermediate format.

        The mobi library extracts to EPUB, HTML, or PDF depending on the MOBI type.
        We handle each case appropriately.

        Args:
            section_indices: Optional list of 0-based section/page indices.
                If None, all sections are extracted.
        """
        try:
            tempdir, filepath = mobi.extract(str(self.mobi_path))
            self._tempdir = Path(tempdir)
            extracted_path = Path(filepath)

            logger.info(
                "Extracted MOBI '%s' to '%s' (format: %s)",
                self.mobi_path.name,
                extracted_path.name,
                extracted_path.suffix.lower(),
            )

            suffix = extracted_path.suffix.lower()

            if suffix == ".epub":
                return self._extract_from_epub(extracted_path, section_indices)
            elif suffix in (".html", ".htm", ".xhtml"):
                return self._extract_from_html(extracted_path, section_indices)
            elif suffix == ".pdf":
                return self._extract_from_pdf(extracted_path, section_indices)
            else:
                logger.warning(
                    "Unknown extracted format '%s' for MOBI '%s', attempting text read",
                    suffix,
                    self.mobi_path.name,
                )
                return self._extract_as_text(extracted_path)

        except Exception:
            logger.exception("Failed to extract MOBI: %s", self.mobi_path)
            raise
        finally:
            self._cleanup()

    def _extract_from_epub(
        self, epub_path: Path, section_indices: list[int] | None = None
    ) -> MOBITextExtraction:
        """Extract text from the unpacked EPUB."""
        processor = EPUBProcessor(epub_path)
        extraction: EPUBTextExtraction = processor.extract_text(
            section_indices=section_indices
        )

        return MOBITextExtraction(
            title=extraction.title,
            authors=extraction.authors,
            sections=extraction.sections,
            source_format="epub",
        )

    def _extract_from_html(
        self, html_path: Path, section_indices: list[int] | None = None
    ) -> MOBITextExtraction:
        """Extract text from the unpacked HTML."""
        try:
            content = html_path.read_bytes()
            document = html.fromstring(content)

            title = None
            title_elements = document.xpath("//title/text()")
            if title_elements:
                title = str(title_elements[0]).strip()

            html.etree.strip_elements(document, "script", "style", with_tail=False)

            text_content = document.text_content()
            sections = [text_content] if text_content else []

            # An HTML file is a single section.
            if section_indices is not None:
                sections = [
                    sections[i] for i in section_indices if 0 <= i < len(sections)
                ]

            return MOBITextExtraction(
                title=title, authors=[], sections=sections, source_format="html"
            )
        except Exception as exc:  # noqa: BLE001 - any parse failure reads raw text
            logger.warning("Failed to parse HTML from MOBI: %s", exc)
            return self._extract_as_text(html_path)

    def _extract_from_pdf(
        self, pdf_path: Path, section_indices: list[int] | None = None
    ) -> MOBITextExtraction:
        """Extract text from the unpacked PDF using PyMuPDF."""
        try:
            import fitz

            sections: list[str] = []
            title = None
            authors: list[str] = []

            with fitz.open(str(pdf_path)) as doc:
                metadata = doc.metadata
                if metadata:
                    title = metadata.get("title", "").strip() or None
                    author = metadata.get("author", "").strip()
                    if author:
                        authors = [author]

                pages = (
                    section_indices
                    if section_indices is not None
                    else list(range(len(doc)))
                )
                for page_num in pages:
                    if 0 <= page_num < len(doc):
                        page = doc[page_num]
                        text = page.get_text()
                        if text.strip():
                            sections.append(text)

            return MOBITextExtraction(
                title=title, authors=authors, sections=sections, source_format="pdf"
            )
        except Exception as exc:
            logger.exception("Failed to extract PDF from MOBI: %s", exc)
            raise

    def _extract_as_text(self, file_path: Path) -> MOBITextExtraction:
        """Fallback: read file as plain text."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return MOBITextExtraction(
                title=None,
                authors=[],
                sections=[content] if content.strip() else [],
                source_format="text",
            )
        except Exception as exc:
            logger.exception("Failed to read extracted file as text: %s", exc)
            raise

    def _cleanup(self) -> None:
        """Clean up temporary extraction directory."""
        if self._tempdir and self._tempdir.exists():
            try:
                shutil.rmtree(self._tempdir)
                logger.debug("Cleaned up temp directory: %s", self._tempdir)
            except OSError as exc:
                logger.warning(
                    "Failed to clean up temp directory %s: %s", self._tempdir, exc
                )
