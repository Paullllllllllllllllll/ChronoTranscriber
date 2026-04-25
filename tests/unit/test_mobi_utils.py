"""Unit tests for modules.documents.mobi.

Exercises the public surface of the MOBI processor: the
``MOBITextExtraction`` dataclass, the plain-text rendering helper, the
suffix-based routing inside ``MOBIProcessor.extract_text`` (which delegates
to ``EPUBProcessor`` when the mobi library yields an EPUB), and the output
folder preparation helper.

The underlying ``mobi.extract`` call is patched via ``monkeypatch`` so that
the tests never need a real MOBI binary on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

# Priming import to avoid a circular-import chain when this module is the
# first to touch modules.documents.
import modules.transcribe.dual_mode  # noqa: F401

from modules.config.constants import SUPPORTED_MOBI_EXTENSIONS
from modules.documents import mobi as mobi_mod
from modules.documents.epub import EPUBTextExtraction
from modules.documents.mobi import MOBIProcessor, MOBITextExtraction


# ---------------------------------------------------------------------------
# MOBITextExtraction dataclass
# ---------------------------------------------------------------------------

class TestMOBITextExtraction:

    @pytest.mark.unit
    def test_construction_and_field_access(self) -> None:
        ext = MOBITextExtraction(
            title="Sample Title",
            authors=["Alice", "Bob"],
            sections=["Section A", "Section B"],
            source_format="epub",
        )
        assert ext.title == "Sample Title"
        assert ext.authors == ["Alice", "Bob"]
        assert ext.sections == ["Section A", "Section B"]
        assert ext.source_format == "epub"

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


# ---------------------------------------------------------------------------
# Suffix-based recognition (via SUPPORTED_MOBI_EXTENSIONS constant)
# ---------------------------------------------------------------------------

class TestMOBIExtensionRecognition:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "suffix",
        [".mobi", ".azw", ".azw3", ".kfx"],
    )
    def test_recognised_extensions(self, suffix: str) -> None:
        assert suffix in SUPPORTED_MOBI_EXTENSIONS

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "suffix",
        [".txt", ".pdf", ".epub", ".html", ".jpg"],
    )
    def test_unrecognised_extensions(self, suffix: str) -> None:
        assert suffix not in SUPPORTED_MOBI_EXTENSIONS


# ---------------------------------------------------------------------------
# MOBIProcessor.extract_text -- routes through EPUBProcessor
# ---------------------------------------------------------------------------

class TestMOBIProcessorExtractText:

    @pytest.mark.unit
    def test_routes_epub_output_to_epub_processor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When mobi.extract yields an .epub, extract_text delegates to EPUBProcessor."""
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


# ---------------------------------------------------------------------------
# MOBIProcessor.prepare_output_folder
# ---------------------------------------------------------------------------

class TestMOBIProcessorPrepareOutputFolder:

    @pytest.mark.unit
    def test_creates_parent_and_returns_txt_path(self, tmp_path: Path) -> None:
        mobi_path = tmp_path / "Some Book.mobi"
        mobi_path.write_bytes(b"stub")

        out_dir = tmp_path / "mobi_out"
        out_dir.mkdir()

        proc = MOBIProcessor(mobi_path)
        parent, out_txt = proc.prepare_output_folder(out_dir)

        assert parent.exists() and parent.is_dir()
        assert parent.parent == out_dir
        assert out_txt.parent == parent
        assert out_txt.suffix == ".txt"


# ---------------------------------------------------------------------------
# Module-level helper: _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:

    @pytest.mark.unit
    def test_collapses_blank_lines(self) -> None:
        raw = "\n  A  \n\n\n  B\n\n\n\nC  \n\n"
        assert mobi_mod._normalize_text(raw) == "A\n\nB\n\nC"

    @pytest.mark.unit
    def test_empty_input_returns_empty(self) -> None:
        assert mobi_mod._normalize_text("") == ""
