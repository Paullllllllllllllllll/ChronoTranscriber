"""EPUB source: metadata lookup, HTML-to-text conversion, section extraction and
navigation-file filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chronotranscriber.sources._text import normalize_text
from chronotranscriber.sources.epub import (
    EPUBProcessor,
    EPUBTextExtraction,
    _first_metadata_value,
    _html_bytes_to_text,
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
    assert normalize_text(raw) == "A\n\nB\n\nC"


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
def test_extract_text_uses_metadata_and_collects_sections(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
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
        "chronotranscriber.sources.epub.epub.read_epub",
        MagicMock(return_value=FakeBook()),
    )

    processor = EPUBProcessor(epub_path)
    extraction = processor.extract_text()

    assert extraction.title == "Example EPUB"
    assert extraction.authors == ["Alice", "Bob"]
    assert extraction.sections == ["Chapter 1", "Chapter 2"]


class _FakeEpubItem:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_type(self) -> int:
        import ebooklib

        return int(ebooklib.ITEM_DOCUMENT)

    def get_name(self) -> str:
        return self._name


class _FakeEpubBook:
    def __init__(self, names: list[str]) -> None:
        self._items = {f"id{i}": _FakeEpubItem(n) for i, n in enumerate(names)}
        self.spine = [(f"id{i}", "yes") for i in range(len(names))]

    def get_item_with_id(self, idref: str) -> Any:
        return self._items.get(idref)

    def get_items_of_type(self, item_type: int) -> list[Any]:
        return list(self._items.values())


@pytest.mark.unit
class TestNavFiltering:
    def test_only_whole_nav_toc_and_cover_names_are_dropped(self) -> None:
        book = _FakeEpubBook(
            [
                "nav.xhtml",  # dropped
                "toc.ncx",  # dropped
                "cover.xhtml",  # dropped
                "discovery.xhtml",  # kept: no anchored token
                "navarra.xhtml",  # kept: starts with 'nav' but is not the token
                "chapter1.xhtml",  # kept
            ]
        )
        ordered = EPUBProcessor._ordered_documents(book)
        names = [i.get_name() for i in ordered]
        assert names == ["discovery.xhtml", "navarra.xhtml", "chapter1.xhtml"]

    def test_hyphen_and_underscore_variants_dropped(self) -> None:
        book = _FakeEpubBook(["nav-1.xhtml", "toc_2.xhtml", "body.xhtml"])
        ordered = EPUBProcessor._ordered_documents(book)
        assert [i.get_name() for i in ordered] == ["body.xhtml"]
