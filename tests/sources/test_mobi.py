"""MOBI source: plain-text rendering, routing of the unpacked book to the EPUB or
HTML reader, and extraction errors.

The ``mobi.extract`` call is patched, so no MOBI file is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chronotranscriber.sources import mobi as mobi_mod
from chronotranscriber.sources.epub import EPUBTextExtraction
from chronotranscriber.sources.mobi import MOBIProcessor, MOBITextExtraction


class TestMOBITextExtraction:
    @pytest.mark.unit
    def test_to_plain_text_renders_metadata_and_sections(self) -> None:
        ext = MOBITextExtraction(
            title=" My Title ",
            authors=["Alice", "  ", "Bob"],
            sections=["\n\nSection 1\n\n", "", "Section 2"],
            source_format="epub",
        )
        rendered = ext.to_plain_text()
        assert rendered.startswith("# Title: My Title\n# Author(s): Alice, Bob\n\n")
        assert "Section 1" in rendered
        assert "Section 2" in rendered
        assert rendered.endswith("\n")

    @pytest.mark.unit
    def test_to_plain_text_without_metadata(self) -> None:
        ext = MOBITextExtraction(
            title=None,
            authors=[],
            sections=["Just content"],
            source_format="html",
        )
        rendered = ext.to_plain_text()
        assert "Just content" in rendered
        assert "# Title:" not in rendered
        assert "# Author(s):" not in rendered


class TestMOBIProcessorExtractText:
    @pytest.mark.unit
    def test_routes_epub_output_to_epub_processor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When mobi.extract yields .epub, extract_text delegates to EPUBProcessor."""
        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"fake mobi bytes")

        # Prepare a fake extraction dir with an .epub filepath
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        epub_path = extract_dir / "book.epub"
        epub_path.write_bytes(b"fake epub bytes")

        def fake_extract(path: str) -> tuple[str, str]:
            assert Path(path) == mobi_path
            return str(extract_dir), str(epub_path)

        monkeypatch.setattr(mobi_mod.mobi, "extract", fake_extract)

        # Replace EPUBProcessor with one returning a predictable result
        captured: dict = {}

        class FakeEPUBProcessor:
            def __init__(self, path: Path) -> None:
                captured["epub_path"] = path

            def extract_text(self, section_indices=None) -> EPUBTextExtraction:
                captured["section_indices"] = section_indices
                return EPUBTextExtraction(
                    title="From EPUB",
                    authors=["Anon"],
                    sections=["Content body"],
                )

        monkeypatch.setattr(mobi_mod, "EPUBProcessor", FakeEPUBProcessor)

        proc = MOBIProcessor(mobi_path)
        result = proc.extract_text(section_indices=[0])

        assert isinstance(result, MOBITextExtraction)
        assert result.title == "From EPUB"
        assert result.authors == ["Anon"]
        assert result.sections == ["Content body"]
        assert result.source_format == "epub"
        assert captured["epub_path"] == epub_path
        assert captured["section_indices"] == [0]

    @pytest.mark.unit
    def test_routes_html_output_to_html_extractor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When mobi.extract yields an .html, extract_text uses the HTML path."""
        mobi_path = tmp_path / "book.mobi"
        mobi_path.write_bytes(b"fake mobi bytes")

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        html_path = extract_dir / "book.html"
        html_path.write_bytes(
            b"<html><head><title>HTML Title</title></head>"
            b"<body><p>Hello HTML</p></body></html>"
        )

        def fake_extract(path: str) -> tuple[str, str]:
            return str(extract_dir), str(html_path)

        monkeypatch.setattr(mobi_mod.mobi, "extract", fake_extract)

        proc = MOBIProcessor(mobi_path)
        result = proc.extract_text()

        assert result.source_format == "html"
        assert result.title == "HTML Title"
        assert any("Hello HTML" in s for s in result.sections)

    @pytest.mark.unit
    def test_extract_text_raises_when_underlying_extract_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mobi_path = tmp_path / "bad.mobi"
        mobi_path.write_bytes(b"garbage")

        def fake_extract(path: str) -> tuple[str, str]:
            raise RuntimeError("mobi library failed")

        monkeypatch.setattr(mobi_mod.mobi, "extract", fake_extract)

        proc = MOBIProcessor(mobi_path)
        with pytest.raises(RuntimeError, match="mobi library failed"):
            proc.extract_text()


class TestNormalizeText:
    @pytest.mark.unit
    def test_empty_input_returns_empty(self) -> None:
        assert mobi_mod.normalize_text("") == ""
