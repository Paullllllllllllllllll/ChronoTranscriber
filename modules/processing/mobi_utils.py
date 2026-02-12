"""MOBI file processing utilities.

This module provides functionality to extract text content from MOBI (Kindle) files
by unpacking them to intermediate formats (EPUB/HTML) and extracting text.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mobi
from lxml import html

from modules.core.safe_paths import create_safe_directory_name, create_safe_filename
from modules.processing.epub_utils import EPUBProcessor, EPUBTextExtraction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MOBITextExtraction:
    """Represents the result of extracting plain text from a MOBI file."""

    title: Optional[str]
    authors: List[str]
    sections: List[str]
    source_format: str  # 'epub', 'html', or 'pdf'

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


class MOBIProcessor:
    """Utility for extracting text content from MOBI files."""

    def __init__(self, mobi_path: Path) -> None:
        self.mobi_path = mobi_path
        self._tempdir: Optional[Path] = None

    def extract_text(self, section_indices: Optional[List[int]] = None) -> MOBITextExtraction:
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
                extracted_path.suffix.lower()
            )
            
            suffix = extracted_path.suffix.lower()
            
            if suffix == ".epub":
                return self._extract_from_epub(extracted_path, section_indices)
            elif suffix in (".html", ".htm", ".xhtml"):
                return self._extract_from_html(extracted_path, section_indices)
            elif suffix == ".pdf":
                return self._extract_from_pdf(extracted_path, section_indices)
            else:
                # Fallback: try to read as text
                logger.warning(
                    "Unknown extracted format '%s' for MOBI '%s', attempting text read",
                    suffix,
                    self.mobi_path.name
                )
                return self._extract_as_text(extracted_path)
                
        except Exception as exc:
            logger.exception("Failed to extract MOBI: %s", self.mobi_path)
            raise
        finally:
            self._cleanup()

    def _extract_from_epub(self, epub_path: Path,
                            section_indices: Optional[List[int]] = None) -> MOBITextExtraction:
        """Extract text from the unpacked EPUB."""
        processor = EPUBProcessor(epub_path)
        extraction: EPUBTextExtraction = processor.extract_text(section_indices=section_indices)
        
        return MOBITextExtraction(
            title=extraction.title,
            authors=extraction.authors,
            sections=extraction.sections,
            source_format="epub"
        )

    def _extract_from_html(self, html_path: Path,
                            section_indices: Optional[List[int]] = None) -> MOBITextExtraction:
        """Extract text from the unpacked HTML."""
        try:
            content = html_path.read_bytes()
            document = html.fromstring(content)
            
            # Try to extract title
            title = None
            title_elements = document.xpath("//title/text()")
            if title_elements:
                title = str(title_elements[0]).strip()
            
            # Remove scripts and styles
            html.etree.strip_elements(document, "script", "style", with_tail=False)
            
            # Extract text content
            text_content = document.text_content()
            sections = [text_content] if text_content else []

            # Apply section filter (HTML is typically a single section)
            if section_indices is not None:
                sections = [sections[i] for i in section_indices if 0 <= i < len(sections)]
            
            return MOBITextExtraction(
                title=title,
                authors=[],
                sections=sections,
                source_format="html"
            )
        except Exception as exc:
            logger.warning("Failed to parse HTML from MOBI: %s", exc)
            # Fallback to raw text
            return self._extract_as_text(html_path)

    def _extract_from_pdf(self, pdf_path: Path,
                           section_indices: Optional[List[int]] = None) -> MOBITextExtraction:
        """Extract text from the unpacked PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            sections: List[str] = []
            title = None
            authors: List[str] = []
            
            with fitz.open(str(pdf_path)) as doc:
                # Try to get metadata
                metadata = doc.metadata
                if metadata:
                    title = metadata.get("title", "").strip() or None
                    author = metadata.get("author", "").strip()
                    if author:
                        authors = [author]
                
                # Determine which pages to extract
                pages = section_indices if section_indices is not None else list(range(len(doc)))
                for page_num in pages:
                    if 0 <= page_num < len(doc):
                        page = doc[page_num]
                        text = page.get_text()
                        if text.strip():
                            sections.append(text)
            
            return MOBITextExtraction(
                title=title,
                authors=authors,
                sections=sections,
                source_format="pdf"
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
                source_format="text"
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
            except Exception as exc:
                logger.warning("Failed to clean up temp directory %s: %s", self._tempdir, exc)

    def prepare_output_folder(self, mobi_output_dir: Path) -> tuple[Path, Path]:
        """Prepare a deterministic output folder and text file path for this MOBI."""
        safe_dir_name = create_safe_directory_name(self.mobi_path.stem)
        parent_folder = mobi_output_dir / safe_dir_name
        parent_folder.mkdir(parents=True, exist_ok=True)

        # Create safe filename (truncated with hash if needed, considering full path length)
        output_txt_name = create_safe_filename(self.mobi_path.stem, ".txt", parent_folder)
        output_txt_path = parent_folder / output_txt_name
        return parent_folder, output_txt_path


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
