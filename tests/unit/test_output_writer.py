"""Unit tests for modules/processing/output_writer.py.

Tests output format writers (txt, md, json) and the unified entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.processing.output_writer import (
    resolve_output_path,
    write_transcription_output,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_normal_page(
    text: str = "Normal transcription text.",
    page_number: int = 1,
    image_name: str = "page_001.jpg",
) -> dict:
    return {"text": text, "page_number": page_number, "image_name": image_name}


def _make_failure_page(
    placeholder: str = "[transcription error]",
    page_number: int = 1,
    image_name: str = "page_001.jpg",
) -> dict:
    return {"text": placeholder, "page_number": page_number, "image_name": image_name}


# =============================================================================
# TestResolveOutputPath
# =============================================================================

class TestResolveOutputPath:
    """Tests for resolve_output_path function."""

    @pytest.mark.unit
    def test_txt_extension(self, tmp_path):
        assert resolve_output_path(tmp_path / "out.xyz", "txt").suffix == ".txt"

    @pytest.mark.unit
    def test_md_extension(self, tmp_path):
        assert resolve_output_path(tmp_path / "out.xyz", "md").suffix == ".md"

    @pytest.mark.unit
    def test_json_extension(self, tmp_path):
        assert resolve_output_path(tmp_path / "out.xyz", "json").suffix == ".json"

    @pytest.mark.unit
    def test_unknown_format_defaults_to_txt(self, tmp_path):
        assert resolve_output_path(tmp_path / "out.xyz", "csv").suffix == ".txt"


# =============================================================================
# TestWriteTxt
# =============================================================================

class TestWriteTxt:
    """Tests for the txt output format."""

    @pytest.mark.unit
    def test_normal_pages_no_headers(self, tmp_path):
        """Normal transcription text should have no image/page headers."""
        pages = [_make_normal_page("First page text."), _make_normal_page("Second page text.", 2, "page_002.jpg")]
        path = write_transcription_output(pages, tmp_path / "out.txt", "txt", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "page_001" not in content
        assert "page_002" not in content
        assert "Page 1" not in content
        assert "First page text." in content
        assert "Second page text." in content

    @pytest.mark.unit
    def test_failure_page_gets_header(self, tmp_path):
        """Failure placeholders should be prefixed with image name."""
        pages = [_make_failure_page("[transcription error]", 5, "scan_005.jpg")]
        path = write_transcription_output(pages, tmp_path / "out.txt", "txt", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "scan_005.jpg:" in content
        assert "[transcription error]" in content

    @pytest.mark.unit
    def test_mixed_normal_and_failure_pages(self, tmp_path):
        """Only failure pages get headers; normal pages do not."""
        pages = [
            _make_normal_page("Good text.", 1, "p1.jpg"),
            _make_failure_page("[transcription error]", 2, "p2.jpg"),
            _make_normal_page("More good text.", 3, "p3.jpg"),
        ]
        path = write_transcription_output(pages, tmp_path / "out.txt", "txt", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "p1.jpg" not in content
        assert "p2.jpg:" in content
        assert "p3.jpg" not in content

    @pytest.mark.unit
    def test_empty_pages_list(self, tmp_path):
        """Empty pages list produces an empty file."""
        path = write_transcription_output([], tmp_path / "out.txt", "txt", postprocess=False)
        assert path.read_text(encoding="utf-8") == ""


# =============================================================================
# TestWriteMd
# =============================================================================

class TestWriteMd:
    """Tests for the md output format."""

    @pytest.mark.unit
    def test_normal_pages_no_programmatic_headers(self, tmp_path):
        """Core bug-fix test: normal pages must NOT get ## headers."""
        pages = [
            {
                "text": "<page_number>64</page_number>\nChapter 3\n\nSome transcribed text.",
                "page_number": 65,
                "image_name": "page_0065_pre_processed.jpg",
            },
        ]
        path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "## page_0065" not in content
        assert "## Page 65" not in content
        assert "## [unknown page]" not in content
        assert "<page_number>64</page_number>" in content
        assert "Chapter 3" in content

    @pytest.mark.unit
    def test_page_number_tags_preserved(self, tmp_path):
        """LLM-produced <page_number> tags must pass through unchanged."""
        pages = [_make_normal_page("<page_number>12</page_number>\nText here.", 13, "img_013.jpg")]
        path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "<page_number>12</page_number>" in content

    @pytest.mark.unit
    def test_failure_page_gets_header_in_md(self, tmp_path):
        """Failure placeholders should still have identifying headers in md."""
        pages = [_make_failure_page("[Transcription not possible]", 3, "scan_003.png")]
        path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "scan_003.png:" in content
        assert "[Transcription not possible]" in content

    @pytest.mark.unit
    def test_pages_separated_by_double_newline(self, tmp_path):
        """Pages in md format should be separated by double newlines."""
        pages = [
            _make_normal_page("Page one text.", 1, "p1.jpg"),
            _make_normal_page("Page two text.", 2, "p2.jpg"),
        ]
        path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "Page one text.\n\nPage two text." in content

    @pytest.mark.unit
    def test_mixed_normal_and_failure_pages(self, tmp_path):
        """Only failure pages get headers; normal pages emit raw text."""
        pages = [
            _make_normal_page("Good text.", 1, "p1.jpg"),
            _make_failure_page("[transcription error]", 2, "p2.jpg"),
            _make_normal_page("More good text.", 3, "p3.jpg"),
        ]
        path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "## " not in content
        assert "p1.jpg" not in content
        assert "p2.jpg:" in content
        assert "p3.jpg" not in content

    @pytest.mark.unit
    def test_no_header_when_llm_omits_page_number_tag(self, tmp_path):
        """Even if the LLM omits <page_number>, no programmatic header should appear."""
        pages = [_make_normal_page("Just plain text without any page tag.", 10, "img_010.jpg")]
        path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        content = path.read_text(encoding="utf-8")
        assert "## " not in content
        assert "img_010" not in content
        assert "Page 10" not in content
        assert "Just plain text without any page tag." in content

    @pytest.mark.unit
    def test_empty_pages_list(self, tmp_path):
        """Empty pages list produces an empty file."""
        path = write_transcription_output([], tmp_path / "out.md", "md", postprocess=False)
        assert path.read_text(encoding="utf-8") == ""


# =============================================================================
# TestWriteJson
# =============================================================================

class TestWriteJson:
    """Tests for the json output format."""

    @pytest.mark.unit
    def test_writes_structured_records(self, tmp_path):
        """JSON output should be a list of structured records."""
        pages = [_make_normal_page("Text.", 1, "img.jpg")]
        path = write_transcription_output(pages, tmp_path / "out.json", "json")
        records = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(records, list)
        assert len(records) == 1
        assert records[0]["transcription"] == "Text."

    @pytest.mark.unit
    def test_preserves_page_metadata(self, tmp_path):
        """JSON records should include page_number and image_name metadata."""
        pages = [_make_normal_page("Text.", 42, "scan_042.png")]
        path = write_transcription_output(pages, tmp_path / "out.json", "json")
        records = json.loads(path.read_text(encoding="utf-8"))
        assert records[0]["page_number"] == 42
        assert records[0]["image_name"] == "scan_042.png"

    @pytest.mark.unit
    def test_no_postprocessing_on_json(self, tmp_path):
        """JSON output should preserve raw text without postprocessing."""
        pages = [_make_normal_page("Text   with   extra   spaces.", 1, "img.jpg")]
        path = write_transcription_output(pages, tmp_path / "out.json", "json", postprocess=True)
        records = json.loads(path.read_text(encoding="utf-8"))
        assert "   " in records[0]["transcription"]

    @pytest.mark.unit
    def test_empty_pages_list(self, tmp_path):
        """Empty pages list produces an empty JSON array."""
        path = write_transcription_output([], tmp_path / "out.json", "json")
        records = json.loads(path.read_text(encoding="utf-8"))
        assert records == []


# =============================================================================
# TestWriteTranscriptionOutput
# =============================================================================

class TestWriteTranscriptionOutput:
    """Tests for the unified write_transcription_output entry point."""

    @pytest.mark.unit
    def test_dispatches_to_txt(self, tmp_path):
        pages = [_make_normal_page()]
        path = write_transcription_output(pages, tmp_path / "out.xyz", "txt", postprocess=False)
        assert path.suffix == ".txt"
        assert path.exists()

    @pytest.mark.unit
    def test_dispatches_to_md(self, tmp_path):
        pages = [_make_normal_page()]
        path = write_transcription_output(pages, tmp_path / "out.xyz", "md", postprocess=False)
        assert path.suffix == ".md"
        assert path.exists()

    @pytest.mark.unit
    def test_dispatches_to_json(self, tmp_path):
        pages = [_make_normal_page()]
        path = write_transcription_output(pages, tmp_path / "out.xyz", "json")
        assert path.suffix == ".json"
        assert path.exists()

    @pytest.mark.unit
    def test_unknown_format_falls_back_to_txt(self, tmp_path):
        pages = [_make_normal_page()]
        path = write_transcription_output(pages, tmp_path / "out.xyz", "csv", postprocess=False)
        assert path.suffix == ".txt"
        assert path.exists()

    @pytest.mark.unit
    def test_creates_parent_directory(self, tmp_path):
        pages = [_make_normal_page()]
        nested = tmp_path / "deep" / "nested" / "out.txt"
        path = write_transcription_output(pages, nested, "txt", postprocess=False)
        assert path.exists()

    @pytest.mark.unit
    def test_all_formats_consistent_for_normal_text(self, tmp_path):
        """Normal text should appear without programmatic headers in txt and md."""
        pages = [
            _make_normal_page("<page_number>1</page_number>\nHello world.", 1, "p1.jpg"),
            _make_normal_page("<page_number>2</page_number>\nGoodbye world.", 2, "p2.jpg"),
        ]
        txt_path = write_transcription_output(pages, tmp_path / "out.txt", "txt", postprocess=False)
        md_path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        txt_content = txt_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")

        for content in (txt_content, md_content):
            assert "## " not in content
            assert "p1.jpg" not in content
            assert "p2.jpg" not in content
            assert "<page_number>1</page_number>" in content
            assert "<page_number>2</page_number>" in content

    @pytest.mark.unit
    def test_all_formats_consistent_for_failures(self, tmp_path):
        """Failure placeholders should have identifying headers in both txt and md."""
        pages = [_make_failure_page("[transcription error]", 5, "fail.jpg")]
        txt_path = write_transcription_output(pages, tmp_path / "out.txt", "txt", postprocess=False)
        md_path = write_transcription_output(pages, tmp_path / "out.md", "md", postprocess=False)
        txt_content = txt_path.read_text(encoding="utf-8")
        md_content = md_path.read_text(encoding="utf-8")

        for content in (txt_content, md_content):
            assert "fail.jpg:" in content
            assert "[transcription error]" in content
