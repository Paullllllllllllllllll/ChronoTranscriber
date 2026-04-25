"""Unit tests for modules/core/resume.py ResumeChecker.

Tests file-level resume/skip logic for all processing types (PDFs, images,
EPUBs, MOBIs) and both resume modes (skip, overwrite).
"""

from __future__ import annotations

import pytest
from pathlib import Path

from modules.transcribe.resume import ResumeChecker, ResumeResult, ProcessingState
from modules.infra.paths import create_safe_directory_name, create_safe_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paths_config(output_dir: Path) -> dict:
    """Build a minimal paths_config pointing all outputs at *output_dir*."""
    s = str(output_dir)
    return {
        "general": {"input_paths_is_output_path": False},
        "file_paths": {
            "PDFs": {"input": s, "output": s},
            "Images": {"input": s, "output": s},
            "EPUBs": {"input": s, "output": s},
            "MOBIs": {"input": s, "output": s},
        },
    }


def _create_output_txt(output_dir: Path, stem: str, content: str = "text") -> Path:
    """Create the deterministic output folder + .txt file for *stem*."""
    safe_dir = create_safe_directory_name(stem)
    parent = output_dir / safe_dir
    parent.mkdir(parents=True, exist_ok=True)
    txt_name = create_safe_filename(stem, ".txt", parent)
    txt_path = parent / txt_name
    txt_path.write_text(content, encoding="utf-8")
    return txt_path


def _create_output_jsonl(output_dir: Path, stem: str, content: str = '{"x":1}\n') -> Path:
    """Create the deterministic output folder + .jsonl file for *stem*."""
    safe_dir = create_safe_directory_name(stem)
    parent = output_dir / safe_dir
    parent.mkdir(parents=True, exist_ok=True)
    jsonl_name = create_safe_filename(stem, ".jsonl", parent)
    jsonl_path = parent / jsonl_name
    jsonl_path.write_text(content, encoding="utf-8")
    return jsonl_path


# ===========================================================================
# ResumeChecker — skip mode
# ===========================================================================

class TestResumeCheckerSkipMode:
    """Tests for ResumeChecker with resume_mode='skip'."""

    # --- PDF ---

    @pytest.mark.unit
    def test_pdf_no_output_returns_none(self, temp_dir: Path) -> None:
        """PDF with no output folder -> NONE state."""
        pdf = temp_dir / "input" / "doc.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")

        checker = ResumeChecker("skip", _make_paths_config(temp_dir / "out"))
        result = checker.should_skip(pdf, "pdfs")
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_pdf_complete_output_returns_complete(self, temp_dir: Path) -> None:
        """PDF with non-empty .txt output -> COMPLETE state."""
        out = temp_dir / "out"
        out.mkdir()
        pdf = temp_dir / "input" / "my_document.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")

        _create_output_txt(out, "my_document", "some transcription")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(pdf, "pdfs")
        assert result.state == ProcessingState.COMPLETE
        assert result.output_path is not None

    @pytest.mark.unit
    def test_pdf_empty_txt_returns_none(self, temp_dir: Path) -> None:
        """PDF with empty .txt output -> treated as not processed (NONE)."""
        out = temp_dir / "out"
        out.mkdir()
        pdf = temp_dir / "input" / "empty_doc.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")

        _create_output_txt(out, "empty_doc", "")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(pdf, "pdfs")
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_pdf_partial_jsonl_returns_partial(self, temp_dir: Path) -> None:
        """PDF with JSONL but no .txt -> PARTIAL state."""
        out = temp_dir / "out"
        out.mkdir()
        pdf = temp_dir / "input" / "partial_doc.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")

        _create_output_jsonl(out, "partial_doc")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(pdf, "pdfs")
        assert result.state == ProcessingState.PARTIAL

    # --- EPUB ---

    @pytest.mark.unit
    def test_epub_no_output_returns_none(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        epub = temp_dir / "input" / "book.epub"
        epub.parent.mkdir(parents=True, exist_ok=True)
        epub.write_bytes(b"PK")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(epub, "epubs")
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_epub_complete_output_returns_complete(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        epub = temp_dir / "input" / "book.epub"
        epub.parent.mkdir(parents=True, exist_ok=True)
        epub.write_bytes(b"PK")

        _create_output_txt(out, "book", "chapter one")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(epub, "epubs")
        assert result.state == ProcessingState.COMPLETE

    # --- MOBI ---

    @pytest.mark.unit
    def test_mobi_no_output_returns_none(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        mobi = temp_dir / "input" / "kindle.mobi"
        mobi.parent.mkdir(parents=True, exist_ok=True)
        mobi.write_bytes(b"\x00")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(mobi, "mobis")
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_mobi_complete_output_returns_complete(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        mobi = temp_dir / "input" / "kindle.mobi"
        mobi.parent.mkdir(parents=True, exist_ok=True)
        mobi.write_bytes(b"\x00")

        _create_output_txt(out, "kindle", "chapter one")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(mobi, "mobis")
        assert result.state == ProcessingState.COMPLETE

    # --- Images ---

    @pytest.mark.unit
    def test_image_folder_no_output_returns_none(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        folder = temp_dir / "input" / "scan_pages"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "p1.png").write_bytes(b"")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(folder, "images")
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_image_folder_complete_output_returns_complete(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        folder = temp_dir / "input" / "scan_pages"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "p1.png").write_bytes(b"")

        _create_output_txt(out, "scan_pages", "page 1 text")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(folder, "images")
        assert result.state == ProcessingState.COMPLETE

    @pytest.mark.unit
    def test_image_folder_partial_jsonl_returns_partial(self, temp_dir: Path) -> None:
        out = temp_dir / "out"
        out.mkdir()
        folder = temp_dir / "input" / "scan_pages"
        folder.mkdir(parents=True, exist_ok=True)

        _create_output_jsonl(out, "scan_pages")

        checker = ResumeChecker("skip", _make_paths_config(out))
        result = checker.should_skip(folder, "images")
        assert result.state == ProcessingState.PARTIAL

    # --- Unknown type ---

    @pytest.mark.unit
    def test_unknown_type_returns_none(self, temp_dir: Path) -> None:
        checker = ResumeChecker("skip", _make_paths_config(temp_dir))
        result = checker.should_skip(temp_dir / "foo.xyz", "unknown_type")
        assert result.state == ProcessingState.NONE


# ===========================================================================
# ResumeChecker — overwrite mode
# ===========================================================================

class TestResumeCheckerOverwriteMode:
    """Tests for ResumeChecker with resume_mode='overwrite'."""

    @pytest.mark.unit
    def test_overwrite_always_returns_none(self, temp_dir: Path) -> None:
        """Even with complete output, overwrite mode returns NONE."""
        out = temp_dir / "out"
        out.mkdir()
        pdf = temp_dir / "input" / "doc.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")

        _create_output_txt(out, "doc", "existing")

        checker = ResumeChecker("overwrite", _make_paths_config(out))
        result = checker.should_skip(pdf, "pdfs")
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_overwrite_filter_keeps_all(self, temp_dir: Path) -> None:
        """filter_items in overwrite mode returns all items."""
        out = temp_dir / "out"
        out.mkdir()
        items = []
        for name in ["a", "b", "c"]:
            p = temp_dir / "input" / f"{name}.pdf"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"%PDF")
            items.append(p)
            _create_output_txt(out, name, "done")

        checker = ResumeChecker("overwrite", _make_paths_config(out))
        to_process, skipped = checker.filter_items(items, "pdfs")
        assert len(to_process) == 3
        assert len(skipped) == 0


# ===========================================================================
# ResumeChecker.filter_items
# ===========================================================================

class TestFilterItems:
    """Tests for filter_items method."""

    @pytest.mark.unit
    def test_mixed_items_filtered_correctly(self, temp_dir: Path) -> None:
        """Mix of processed and new items is partitioned correctly."""
        out = temp_dir / "out"
        out.mkdir()
        inp = temp_dir / "input"
        inp.mkdir()

        processed_pdf = inp / "done.pdf"
        processed_pdf.write_bytes(b"%PDF")
        _create_output_txt(out, "done", "text")

        new_pdf = inp / "new.pdf"
        new_pdf.write_bytes(b"%PDF")

        checker = ResumeChecker("skip", _make_paths_config(out))
        to_process, skipped = checker.filter_items(
            [processed_pdf, new_pdf], "pdfs"
        )

        assert len(to_process) == 1
        assert to_process[0] == new_pdf
        assert len(skipped) == 1
        assert skipped[0].item == processed_pdf
        assert skipped[0].state == ProcessingState.COMPLETE

    @pytest.mark.unit
    def test_partial_items_not_skipped(self, temp_dir: Path) -> None:
        """Partial items (JSONL only) should NOT be skipped."""
        out = temp_dir / "out"
        out.mkdir()
        pdf = temp_dir / "input" / "partial.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF")
        _create_output_jsonl(out, "partial")

        checker = ResumeChecker("skip", _make_paths_config(out))
        to_process, skipped = checker.filter_items([pdf], "pdfs")

        assert len(to_process) == 1
        assert len(skipped) == 0

    @pytest.mark.unit
    def test_empty_list(self, temp_dir: Path) -> None:
        checker = ResumeChecker("skip", _make_paths_config(temp_dir))
        to_process, skipped = checker.filter_items([], "pdfs")
        assert to_process == []
        assert skipped == []

    @pytest.mark.unit
    def test_all_complete(self, temp_dir: Path) -> None:
        """When all items are complete, to_process is empty."""
        out = temp_dir / "out"
        out.mkdir()
        inp = temp_dir / "input"
        inp.mkdir()

        items = []
        for name in ["a", "b"]:
            p = inp / f"{name}.epub"
            p.write_bytes(b"PK")
            items.append(p)
            _create_output_txt(out, name, "done")

        checker = ResumeChecker("skip", _make_paths_config(out))
        to_process, skipped = checker.filter_items(items, "epubs")
        assert len(to_process) == 0
        assert len(skipped) == 2


# ===========================================================================
# Input-as-output mode
# ===========================================================================

class TestInputAsOutput:
    """Tests for use_input_as_output=True behavior."""

    @pytest.mark.unit
    def test_pdf_detected_when_output_colocated(self, temp_dir: Path) -> None:
        """When input_paths_is_output_path, output is in same dir as PDF."""
        inp = temp_dir / "docs"
        inp.mkdir()
        pdf = inp / "scan.pdf"
        pdf.write_bytes(b"%PDF")

        # Create output in the same directory as the PDF
        safe_dir = create_safe_directory_name("scan")
        parent = inp / safe_dir
        parent.mkdir(parents=True, exist_ok=True)
        txt_name = create_safe_filename("scan", ".txt", parent)
        (parent / txt_name).write_text("done", encoding="utf-8")

        cfg = _make_paths_config(temp_dir / "unused")
        cfg["general"]["input_paths_is_output_path"] = True
        checker = ResumeChecker("skip", cfg, use_input_as_output=True)
        result = checker.should_skip(pdf, "pdfs")
        assert result.state == ProcessingState.COMPLETE

    @pytest.mark.unit
    def test_epub_detected_when_output_colocated(self, temp_dir: Path) -> None:
        inp = temp_dir / "books"
        inp.mkdir()
        epub = inp / "novel.epub"
        epub.write_bytes(b"PK")

        safe_dir = create_safe_directory_name("novel")
        parent = inp / safe_dir
        parent.mkdir(parents=True, exist_ok=True)
        txt_name = create_safe_filename("novel", ".txt", parent)
        (parent / txt_name).write_text("chapter 1", encoding="utf-8")

        cfg = _make_paths_config(temp_dir / "unused")
        cfg["general"]["input_paths_is_output_path"] = True
        checker = ResumeChecker("skip", cfg, use_input_as_output=True)
        result = checker.should_skip(epub, "epubs")
        assert result.state == ProcessingState.COMPLETE


# ===========================================================================
# CLI argument parsing for --resume / --force
# ===========================================================================

class TestCLIResumeArgs:
    """Tests for --resume and --force/--overwrite CLI flags."""

    @pytest.mark.unit
    def test_resume_flag(self) -> None:
        from modules.core.cli_args import create_transcriber_parser
        parser = create_transcriber_parser()
        args = parser.parse_args(["--auto", "--resume"])
        assert args.resume is True
        assert args.force is None

    @pytest.mark.unit
    def test_force_flag(self) -> None:
        from modules.core.cli_args import create_transcriber_parser
        parser = create_transcriber_parser()
        args = parser.parse_args(["--auto", "--force"])
        assert args.force is True
        assert args.resume is None

    @pytest.mark.unit
    def test_overwrite_alias(self) -> None:
        from modules.core.cli_args import create_transcriber_parser
        parser = create_transcriber_parser()
        args = parser.parse_args(["--auto", "--overwrite"])
        assert args.force is True

    @pytest.mark.unit
    def test_resume_and_force_mutually_exclusive(self) -> None:
        from modules.core.cli_args import create_transcriber_parser
        parser = create_transcriber_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--auto", "--resume", "--force"])

    @pytest.mark.unit
    def test_default_neither_set(self) -> None:
        from modules.core.cli_args import create_transcriber_parser
        parser = create_transcriber_parser()
        args = parser.parse_args(["--auto"])
        assert args.resume is None
        assert args.force is None


# ===========================================================================
# UserConfiguration resume_mode field
# ===========================================================================

class TestUserConfigResumeMode:
    """Tests for the resume_mode field on UserConfiguration."""

    @pytest.mark.unit
    def test_default_resume_mode(self) -> None:
        from modules.transcribe.user_config import UserConfiguration
        config = UserConfiguration()
        assert config.resume_mode == "skip"

    @pytest.mark.unit
    def test_custom_resume_mode(self) -> None:
        from modules.transcribe.user_config import UserConfiguration
        config = UserConfiguration(resume_mode="overwrite")
        assert config.resume_mode == "overwrite"


# ===========================================================================
# ResumeResult / ProcessingState
# ===========================================================================

class TestResumeResultDataclass:
    """Basic tests for the ResumeResult dataclass."""

    @pytest.mark.unit
    def test_default_fields(self, temp_dir: Path) -> None:
        r = ResumeResult(item=temp_dir / "x.pdf", state=ProcessingState.NONE)
        assert r.output_path is None
        assert r.reason == ""

    @pytest.mark.unit
    def test_complete_with_output(self, temp_dir: Path) -> None:
        p = temp_dir / "out.txt"
        r = ResumeResult(
            item=temp_dir / "x.pdf",
            state=ProcessingState.COMPLETE,
            output_path=p,
            reason="output exists",
        )
        assert r.state == ProcessingState.COMPLETE
        assert r.output_path == p


# ===========================================================================
# _check_output_exists generic method
# ===========================================================================

class TestCheckOutputExists:
    """Tests for ResumeChecker._check_output_exists generic method."""

    @pytest.mark.unit
    def test_returns_none_in_overwrite_mode(self, tmp_path: Path) -> None:
        """Overwrite mode bypasses existence checks (handled by should_skip)."""
        checker = ResumeChecker(
            resume_mode="skip",
            paths_config={},
            output_format="txt",
        )
        result = checker._check_output_exists(
            tmp_path / "test.pdf", "test", tmp_path,
        )
        assert result.state == ProcessingState.NONE

    @pytest.mark.unit
    def test_finds_colocated_output(self, tmp_path: Path) -> None:
        """Detects output file next to input when use_input_as_output."""
        checker = ResumeChecker(
            resume_mode="skip",
            paths_config={},
            use_input_as_output=True,
            output_format="txt",
        )
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.touch()
        out = tmp_path / "doc.txt"
        out.write_text("content")

        result = checker._check_output_exists(pdf_path, "doc", tmp_path)
        assert result.state == ProcessingState.COMPLETE

    @pytest.mark.unit
    def test_finds_output_in_working_dir(self, tmp_path: Path) -> None:
        """Detects output inside hash-suffixed working directory."""
        checker = ResumeChecker(
            resume_mode="skip",
            paths_config={},
            output_format="txt",
        )

        safe_dir = create_safe_directory_name("test_doc")
        working = tmp_path / safe_dir
        working.mkdir()
        out_name = create_safe_filename("test_doc", ".txt", working)
        (working / out_name).write_text("content")

        item = tmp_path / "test_doc.pdf"
        result = checker._check_output_exists(item, "test_doc", tmp_path)
        assert result.state == ProcessingState.COMPLETE

    @pytest.mark.unit
    def test_partial_jsonl_detected(self, tmp_path: Path) -> None:
        """Partial JSONL file produces PARTIAL state."""
        checker = ResumeChecker(
            resume_mode="skip",
            paths_config={},
            output_format="txt",
        )

        safe_dir = create_safe_directory_name("test_doc")
        working = tmp_path / safe_dir
        working.mkdir()
        jsonl_name = create_safe_filename("test_doc", ".jsonl", working)
        (working / jsonl_name).write_text('{"data": true}\n')

        item = tmp_path / "test_doc.pdf"
        result = checker._check_output_exists(item, "test_doc", tmp_path)
        assert result.state == ProcessingState.PARTIAL

    @pytest.mark.unit
    def test_no_partial_jsonl_when_disabled(self, tmp_path: Path) -> None:
        """JSONL ignored when supports_partial_jsonl=False."""
        checker = ResumeChecker(
            resume_mode="skip",
            paths_config={},
            output_format="txt",
        )

        safe_dir = create_safe_directory_name("test_doc")
        working = tmp_path / safe_dir
        working.mkdir()
        jsonl_name = create_safe_filename("test_doc", ".jsonl", working)
        (working / jsonl_name).write_text('{"data": true}\n')

        item = tmp_path / "test_doc.epub"
        result = checker._check_output_exists(
            item, "test_doc", tmp_path, supports_partial_jsonl=False
        )
        assert result.state == ProcessingState.NONE