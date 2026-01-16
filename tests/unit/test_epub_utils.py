from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.processing.epub_utils import (
    EPUBProcessor,
    EPUBTextExtraction,
    _first_metadata_value,
    _html_bytes_to_text,
    _normalize_text,
)


@pytest.mark.unit
def test_first_metadata_value_returns_first_non_empty() -> None:
    entries = [("  ", {}), ("Title", {}), ("Other", {})]
    assert _first_metadata_value(entries) == "Title"


@pytest.mark.unit
def test_first_metadata_value_returns_none_when_empty() -> None:
    assert _first_metadata_value([]) is None
    assert _first_metadata_value([("	 ", {})]) is None


@pytest.mark.unit
def test_normalize_text_collapses_whitespace_and_preserves_single_blank_lines() -> None:
    raw = "\n  A  \n\n\n  B\n\n\n\nC  \n\n"
    assert _normalize_text(raw) == "A\n\nB\n\nC"


@pytest.mark.unit
def test_html_bytes_to_text_strips_script_style_and_normalizes() -> None:
    content = (
        b"<html><head><style>.x{}</style><script>bad()</script></head>"
        b"<body>  Hello <b>World</b>\n\n\nNext</body></html>"
    )
    assert _html_bytes_to_text(content) == "Hello World\n\nNext"


@pytest.mark.unit
def test_epub_text_extraction_to_plain_text_renders_metadata_and_sections() -> None:
    ext = EPUBTextExtraction(
        title=" My Title ",
        authors=["Alice", "  ", "Bob"],
        sections=["\n\nSection 1\n\n", "", "Section 2"],
    )
    rendered = ext.to_plain_text()
    assert rendered.startswith("# Title: My Title\n# Author(s): Alice, Bob\n\n")
    assert "Section 1" in rendered
    assert "Section 2" in rendered
    assert rendered.endswith("\n")


@pytest.mark.unit
def test_prepare_output_folder_creates_folder_and_returns_paths(temp_dir: Path) -> None:
    epub_path = temp_dir / "Some Book Title.epub"
    epub_path.write_bytes(b"not a real epub")

    out_dir = temp_dir / "epub_out"
    out_dir.mkdir()

    processor = EPUBProcessor(epub_path)
    parent, out_txt = processor.prepare_output_folder(out_dir)

    assert parent.exists() and parent.is_dir()
    assert out_txt.parent == parent
    assert out_txt.suffix == ".txt"


@pytest.mark.unit
def test_extract_text_uses_metadata_and_collects_sections(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    epub_path = temp_dir / "test.epub"
    epub_path.write_bytes(b"not a real epub")

    class FakeItem:
        def __init__(self, content: bytes, file_name: str = "ch1.xhtml"):
            self._content = content
            self.file_name = file_name

        def get_content(self) -> bytes:
            return self._content

    class FakeBook:
        def get_metadata(self, _ns: str, key: str):
            if key == "title":
                return [("Example EPUB", {})]
            if key == "creator":
                return [("Alice", {}), ("Bob", {})]
            return []

        def get_items_of_type(self, _type):
            return [
                FakeItem(b"<html><body>Chapter 1</body></html>", "c1.xhtml"),
                FakeItem(b"", "empty.xhtml"),
                FakeItem(b"<html><body>Chapter 2</body></html>", "c2.xhtml"),
            ]

    monkeypatch.setattr(
        "modules.processing.epub_utils.epub.read_epub",
        MagicMock(return_value=FakeBook()),
    )

    processor = EPUBProcessor(epub_path)
    extraction = processor.extract_text()

    assert extraction.title == "Example EPUB"
    assert extraction.authors == ["Alice", "Bob"]
    assert extraction.sections == ["Chapter 1", "Chapter 2"]
