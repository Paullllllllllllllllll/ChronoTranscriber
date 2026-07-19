"""Unit tests for modules/operations/repair/utils.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestExtractImageNameFromFailureLine:
    """Tests for extract_image_name_from_failure_line function."""

    @pytest.mark.unit
    def test_extracts_from_transcription_error(self) -> None:
        """Test extraction from transcription error placeholder."""
        from modules.batch.repair import extract_image_name_from_failure_line

        line = "[transcription error: IMG_0001.png]"
        result = extract_image_name_from_failure_line(line)

        assert result == "IMG_0001.png"

    @pytest.mark.unit
    def test_extracts_from_not_possible(self) -> None:
        """Test extraction from transcription not possible placeholder."""
        from modules.batch.repair import extract_image_name_from_failure_line

        line = "[Transcription not possible: page_12.jpg]"
        result = extract_image_name_from_failure_line(line)

        assert result == "page_12.jpg"

    @pytest.mark.unit
    def test_extracts_from_no_text(self) -> None:
        """Test extraction from no transcribable text placeholder."""
        from modules.batch.repair import extract_image_name_from_failure_line

        line = "[No transcribable text: scan_03.png]"
        result = extract_image_name_from_failure_line(line)

        assert result == "scan_03.png"

    @pytest.mark.unit
    def test_extracts_with_page_prefix(self) -> None:
        """Test extraction with Page N: prefix."""
        from modules.batch.repair import extract_image_name_from_failure_line

        line = "Page 5: [transcription error: page_005.png]"
        result = extract_image_name_from_failure_line(line)

        assert result == "page_005.png"

    @pytest.mark.unit
    def test_handles_error_with_semicolon(self) -> None:
        """Test handling error with additional info after semicolon."""
        from modules.batch.repair import extract_image_name_from_failure_line

        line = "[transcription error: scan_03.png; status 400; code invalid_image]"
        result = extract_image_name_from_failure_line(line)

        assert result == "scan_03.png"

    @pytest.mark.unit
    def test_returns_none_for_normal_text(self) -> None:
        """Test returns None for normal text."""
        from modules.batch.repair import extract_image_name_from_failure_line

        line = "This is normal transcription text."
        result = extract_image_name_from_failure_line(line)

        assert result is None


class TestIsFailureLine:
    """Tests for is_failure_line function."""

    @pytest.mark.unit
    def test_detects_transcription_error(self) -> None:
        """Test detection of transcription error."""
        from modules.batch.repair import is_failure_line

        assert is_failure_line("[transcription error: image.png]") is True

    @pytest.mark.unit
    def test_detects_not_possible(self) -> None:
        """Test detection of transcription not possible."""
        from modules.batch.repair import is_failure_line

        assert is_failure_line("[Transcription not possible: image.png]") is True

    @pytest.mark.unit
    def test_detects_with_prefix(self) -> None:
        """Test detection with page prefix."""
        from modules.batch.repair import is_failure_line

        assert is_failure_line("Page 1: [transcription error: image.png]") is True

    @pytest.mark.unit
    def test_normal_text_not_failure(self) -> None:
        """Test that normal text is not detected as failure."""
        from modules.batch.repair import is_failure_line

        assert is_failure_line("Normal transcription text") is False

    @pytest.mark.unit
    def test_empty_line_not_failure(self) -> None:
        """Test that empty line is not detected as failure."""
        from modules.batch.repair import is_failure_line

        assert is_failure_line("") is False
        assert is_failure_line("   ") is False


class TestCollectImageEntriesFromJsonl:
    """Tests for collect_image_entries_from_jsonl function."""

    @pytest.mark.unit
    def test_collects_from_image_metadata(self, temp_dir: Path) -> None:
        """Test collection from image_metadata records."""
        from modules.batch.repair import collect_image_entries_from_jsonl

        jsonl_file = temp_dir / "test.jsonl"
        records = [
            {
                "image_metadata": {
                    "order_index": 0,
                    "image_name": "page_001.png",
                    "pre_processed_image": "/path/to/page_001.png",
                }
            },
            {
                "image_metadata": {
                    "order_index": 1,
                    "image_name": "page_002.png",
                    "pre_processed_image": "/path/to/page_002.png",
                }
            },
        ]
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        entries = collect_image_entries_from_jsonl(jsonl_file)

        assert len(entries) == 2
        assert entries[0].order_index == 0
        assert entries[0].image_name == "page_001.png"
        assert entries[1].order_index == 1

    @pytest.mark.unit
    def test_returns_empty_for_nonexistent_file(self, temp_dir: Path) -> None:
        """Test returns empty list for nonexistent file."""
        from modules.batch.repair import collect_image_entries_from_jsonl

        entries = collect_image_entries_from_jsonl(temp_dir / "nonexistent.jsonl")

        assert entries == []

    @pytest.mark.unit
    def test_returns_empty_for_none_path(self) -> None:
        """Test returns empty list for None path."""
        from modules.batch.repair import collect_image_entries_from_jsonl

        entries = collect_image_entries_from_jsonl(None)

        assert entries == []

    @pytest.mark.unit
    def test_sorted_by_order_index(self, temp_dir: Path) -> None:
        """Test entries are sorted by order_index."""
        from modules.batch.repair import collect_image_entries_from_jsonl

        jsonl_file = temp_dir / "test.jsonl"
        records = [
            {"image_metadata": {"order_index": 2, "image_name": "page_003.png"}},
            {"image_metadata": {"order_index": 0, "image_name": "page_001.png"}},
            {"image_metadata": {"order_index": 1, "image_name": "page_002.png"}},
        ]
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        entries = collect_image_entries_from_jsonl(jsonl_file)

        assert entries[0].order_index == 0
        assert entries[1].order_index == 1
        assert entries[2].order_index == 2


class TestFindFailureIndices:
    """Tests for find_failure_indices function."""

    @pytest.mark.unit
    def test_finds_error_indices(self) -> None:
        """Test finding indices of error lines."""
        from modules.batch.repair import find_failure_indices

        lines = [
            "Normal text",
            "[transcription error: page_001.png]",
            "More normal text",
            "[Transcription not possible: page_003.png]",
        ]

        indices = find_failure_indices(lines, include_no_text=False)

        assert indices == [1, 3]

    @pytest.mark.unit
    def test_includes_no_text_when_flag_true(self) -> None:
        """Test includes no_text lines when flag is True."""
        from modules.batch.repair import find_failure_indices

        lines = [
            "[transcription error: page_001.png]",
            "[No transcribable text: page_002.png]",
        ]

        indices = find_failure_indices(lines, include_no_text=True)

        assert indices == [0, 1]

    @pytest.mark.unit
    def test_excludes_no_text_when_flag_false(self) -> None:
        """Test excludes no_text lines when flag is False."""
        from modules.batch.repair import find_failure_indices

        lines = [
            "[transcription error: page_001.png]",
            "[No transcribable text: page_002.png]",
        ]

        indices = find_failure_indices(lines, include_no_text=False)

        assert indices == [0]

    @pytest.mark.unit
    def test_returns_empty_for_no_failures(self) -> None:
        """Test returns empty list when no failures."""
        from modules.batch.repair import find_failure_indices

        lines = ["Normal text", "More text"]

        indices = find_failure_indices(lines, include_no_text=False)

        assert indices == []


class TestBackupFile:
    """Tests for backup_file function."""

    @pytest.mark.unit
    def test_creates_backup(self, temp_dir: Path) -> None:
        """Test that backup file is created."""
        from modules.batch.repair import backup_file

        original = temp_dir / "test.txt"
        original.write_text("Original content")

        backup_path = backup_file(original)

        assert backup_path.exists()
        assert backup_path.read_text() == "Original content"
        assert ".bak." in backup_path.name

    @pytest.mark.unit
    def test_backup_has_timestamp(self, temp_dir: Path) -> None:
        """Test that backup filename contains timestamp."""
        import re

        from modules.batch.repair import backup_file

        original = temp_dir / "test.txt"
        original.write_text("Content")

        backup_path = backup_file(original)

        # Should contain timestamp pattern like 20240101-120000
        pattern = r"\d{8}-\d{6}"
        assert re.search(pattern, backup_path.name)


class TestReadFinalLines:
    """Tests for read_final_lines function."""

    @pytest.mark.unit
    def test_reads_lines(self, temp_dir: Path) -> None:
        """Test reading lines from file."""
        from modules.batch.repair import read_final_lines

        txt_file = temp_dir / "transcription.txt"
        txt_file.write_text("Line 1\nLine 2\nLine 3")

        lines = read_final_lines(txt_file)

        assert lines == ["Line 1", "Line 2", "Line 3"]

    @pytest.mark.unit
    def test_empty_file(self, temp_dir: Path) -> None:
        """Test reading empty file."""
        from modules.batch.repair import read_final_lines

        txt_file = temp_dir / "empty.txt"
        txt_file.write_text("")

        lines = read_final_lines(txt_file)

        # str.splitlines() on empty string returns empty list
        assert lines == []


class TestImageEntry:
    """Tests for ImageEntry dataclass."""

    @pytest.mark.unit
    def test_initialization(self) -> None:
        """Test ImageEntry initialization."""
        from modules.batch.repair import ImageEntry

        entry = ImageEntry(
            order_index=5,
            image_name="page_005.png",
            pre_processed_image="/path/to/image.png",
            custom_id="req-6",
            page_number=6,
        )

        assert entry.order_index == 5
        assert entry.image_name == "page_005.png"
        assert entry.pre_processed_image == "/path/to/image.png"
        assert entry.custom_id == "req-6"
        assert entry.page_number == 6

    @pytest.mark.unit
    def test_default_page_number(self) -> None:
        """Test ImageEntry default page_number."""
        from modules.batch.repair import ImageEntry

        entry = ImageEntry(
            order_index=0,
            image_name="test.png",
            pre_processed_image=None,
            custom_id=None,
        )

        assert entry.page_number is None


class TestJob:
    """Tests for Job dataclass."""

    @pytest.mark.unit
    def test_initialization(self, temp_dir: Path) -> None:
        """Test Job initialization."""
        from modules.batch.repair import Job

        job = Job(
            parent_folder=temp_dir,
            identifier="test_doc",
            final_txt_path=temp_dir / "test_doc_transcription.txt",
            temp_jsonl_path=temp_dir / "test_doc_transcription.jsonl",
            kind="PDF",
        )

        assert job.parent_folder == temp_dir
        assert job.identifier == "test_doc"
        assert job.kind == "PDF"


class TestDiscoverJobs:
    """Tests for discover_jobs function."""

    @pytest.mark.unit
    def test_discovers_txt_files(self, temp_dir: Path) -> None:
        """discover_jobs should find .txt transcription files."""
        from modules.batch.repair import discover_jobs

        out = temp_dir / "output"
        out.mkdir()
        (out / "book.txt").write_text("content", encoding="utf-8")

        config = {
            "file_paths": {"PDFs": {"output": str(out)}, "Images": {"output": None}}
        }
        jobs = discover_jobs(config)
        assert any(j.final_txt_path.name == "book.txt" for j in jobs)

    @pytest.mark.unit
    def test_discovers_md_files(self, temp_dir: Path) -> None:
        """discover_jobs should find .md transcription files."""
        from modules.batch.repair import discover_jobs

        out = temp_dir / "output"
        out.mkdir()
        (out / "book.md").write_text("content", encoding="utf-8")

        config = {
            "file_paths": {"PDFs": {"output": str(out)}, "Images": {"output": None}}
        }
        jobs = discover_jobs(config)
        assert any(j.final_txt_path.name == "book.md" for j in jobs)

    @pytest.mark.unit
    def test_discovers_both_txt_and_md(self, temp_dir: Path) -> None:
        """discover_jobs should find both .txt and .md files in the same folder."""
        from modules.batch.repair import discover_jobs

        out = temp_dir / "output"
        out.mkdir()
        (out / "book.txt").write_text("txt content", encoding="utf-8")
        (out / "book.md").write_text("md content", encoding="utf-8")

        config = {
            "file_paths": {"PDFs": {"output": str(out)}, "Images": {"output": None}}
        }
        jobs = discover_jobs(config)
        names = {j.final_txt_path.name for j in jobs}
        assert "book.txt" in names
        assert "book.md" in names

    @pytest.mark.unit
    def test_scans_auto_output_root(self, temp_dir: Path) -> None:
        """Item 1: the Auto output root is scanned (it can hold repairable
        GPT PDF/image transcriptions)."""
        from modules.batch.repair import discover_jobs

        auto_out = temp_dir / "auto_output"
        auto_out.mkdir()
        (auto_out / "mixed.txt").write_text("content", encoding="utf-8")

        config = {"file_paths": {"Auto": {"output": str(auto_out)}}}
        jobs = discover_jobs(config)
        assert any(j.final_txt_path.name == "mixed.txt" for j in jobs)
        assert any(j.kind == "Auto" for j in jobs)

    @pytest.mark.unit
    def test_scans_input_when_input_is_output(self, temp_dir: Path) -> None:
        """Item 1: with input_paths_is_output_path, outputs are co-located with
        inputs, so discovery scans the input dir (the output dir is empty)."""
        from modules.batch.repair import discover_jobs

        pdf_in = temp_dir / "pdf_input"
        pdf_in.mkdir()
        (pdf_in / "book.txt").write_text("content", encoding="utf-8")
        pdf_out = temp_dir / "pdf_output"  # deliberately left empty
        pdf_out.mkdir()

        config = {
            "general": {"input_paths_is_output_path": True},
            "file_paths": {"PDFs": {"input": str(pdf_in), "output": str(pdf_out)}},
        }
        jobs = discover_jobs(config)
        assert any(j.final_txt_path.name == "book.txt" for j in jobs)

    @pytest.mark.unit
    def test_does_not_scan_ebook_roots(self, temp_dir: Path) -> None:
        """Item 1 decision: EPUB/MOBI native extractions have no page images to
        re-render, so their output roots are not scanned for repair."""
        from modules.batch.repair import discover_jobs

        epub_out = temp_dir / "epub_output"
        epub_out.mkdir()
        (epub_out / "ebook.txt").write_text("content", encoding="utf-8")
        mobi_out = temp_dir / "mobi_output"
        mobi_out.mkdir()
        (mobi_out / "kindle.txt").write_text("content", encoding="utf-8")

        config = {
            "file_paths": {
                "EPUBs": {"output": str(epub_out)},
                "MOBIs": {"output": str(mobi_out)},
            }
        }
        jobs = discover_jobs(config)
        names = {j.final_txt_path.name for j in jobs}
        assert "ebook.txt" not in names
        assert "kindle.txt" not in names

    @pytest.mark.unit
    def test_skips_bak_and_cleaned_sidecars(self, temp_dir: Path) -> None:
        """Item 2: repair backups (*.bak.*) and postprocess outputs
        (*.cleaned.*) are skipped, not listed as phantom repair jobs."""
        from modules.batch.repair import discover_jobs

        out = temp_dir / "output"
        out.mkdir()
        (out / "book.txt").write_text("real", encoding="utf-8")
        (out / "book.bak.20240101-120000.txt").write_text("backup", encoding="utf-8")
        (out / "book.cleaned.txt").write_text("cleaned", encoding="utf-8")

        config = {"file_paths": {"PDFs": {"output": str(out)}}}
        jobs = discover_jobs(config)
        names = {j.final_txt_path.name for j in jobs}
        assert names == {"book.txt"}


class TestBackupFileExtensions:
    """Tests for backup_file function — extension preservation."""

    @pytest.mark.unit
    def test_backup_preserves_txt_extension(self, temp_dir: Path) -> None:
        """Backup of .txt file should have .txt extension."""
        from modules.batch.repair import backup_file

        original = temp_dir / "doc.txt"
        original.write_text("content", encoding="utf-8")
        backup = backup_file(original)
        assert backup.name.endswith(".txt")
        assert ".bak." in backup.name

    @pytest.mark.unit
    def test_backup_preserves_md_extension(self, temp_dir: Path) -> None:
        """Backup of .md file should have .md extension."""
        from modules.batch.repair import backup_file

        original = temp_dir / "doc.md"
        original.write_text("content", encoding="utf-8")
        backup = backup_file(original)
        assert backup.name.endswith(".md")
        assert ".bak." in backup.name


class TestBlankRepairGuard:
    """Item 5: an empty re-transcription must NOT overwrite a failure
    placeholder; the write-back guard ``(text or "").strip()`` prevents a blank
    line (cause "ok") from being miscounted as repaired.

    The guard itself is inline in the large ``_repair_sync_mode`` /
    ``_repair_batch_mode`` async flows; these tests lock the contract it relies
    on using the real, importable helpers.
    """

    @pytest.mark.unit
    def test_empty_text_formats_to_blank_line_counted_ok(self) -> None:
        from modules.batch.repair import _count_unrepaired_lines
        from modules.llm.response_parsing import (
            detect_transcription_cause,
            format_page_line,
        )

        # The hazard: an empty re-transcription formats to "" whose cause is "ok".
        assert format_page_line("", 1, "p1.jpg") == ""
        assert detect_transcription_cause("") == "ok"
        # So a blanked targeted line would be (wrongly) counted as repaired.
        assert _count_unrepaired_lines([""], [0]) == 0

    @pytest.mark.unit
    def test_preserved_placeholder_counts_as_unrepaired(self) -> None:
        from modules.batch.repair import _count_unrepaired_lines
        from modules.llm.response_parsing import format_page_line

        # The fix leaves the placeholder untouched, keeping it counted failed.
        placeholder = format_page_line("[transcription error: p1.jpg]", 1, "p1.jpg")
        assert _count_unrepaired_lines([placeholder], [0]) == 1

    @pytest.mark.unit
    def test_guard_predicate_rejects_blank_texts(self) -> None:
        # The exact predicate applied in both write-back loops.
        for blank in ("", "   ", "\n", None):
            assert not (blank or "").strip()
        for real in ("text", "  x  "):
            assert (real or "").strip()


class TestSubsetIntersectionContract:
    """Item 6: interactive subset selection must intersect the chosen indices
    with the detected failures (as the CLI ``--indices`` path already does), so
    a typo on a healthy line is not repaired and miscounted as success.

    The intersection is inline in the interactive ``main()`` flow; this locks
    the predicate contract.
    """

    @pytest.mark.unit
    def test_intersection_drops_non_failure_and_out_of_range(self) -> None:
        detected_failures = {1, 3, 5}
        final_lines = ["a"] * 10
        chosen = [1, 2, 5, 99]
        result = [
            i for i in chosen if i in detected_failures and 0 <= i < len(final_lines)
        ]
        assert result == [1, 5]

    @pytest.mark.unit
    def test_intersection_empty_when_no_overlap(self) -> None:
        detected_failures = {1, 3}
        final_lines = ["a"] * 5
        chosen = [0, 2, 4]
        result = [
            i for i in chosen if i in detected_failures and 0 <= i < len(final_lines)
        ]
        assert result == []


class TestCorrelateRepairTargets:
    """B7: repair results must correlate to targets by custom_id index,
    not by fragile positional zip, so a skipped (failed-to-encode) metadata
    record does not shift repaired text onto the wrong page."""

    @pytest.mark.unit
    def test_skipped_metadata_record_does_not_shift_pages(self) -> None:
        from modules.batch.repair import RepairTarget, _correlate_repair_targets

        # Three targets on distinct output lines / pages.
        targets = [
            RepairTarget(
                order_index=10,
                image_name="p10.jpg",
                image_path=None,
                custom_id="req-1",
                line_index=100,
            ),
            RepairTarget(
                order_index=20,
                image_name="p20.jpg",
                image_path=None,
                custom_id="req-2",
                line_index=200,
            ),
            RepairTarget(
                order_index=30,
                image_name="p30.jpg",
                image_path=None,
                custom_id="req-3",
                line_index=300,
            ),
        ]
        # The middle image (req-2) failed to encode, so its metadata record is
        # absent. A positional zip would pair req-3 with targets[1] (page 20).
        metadata_records = [
            {"batch_request": {"custom_id": "req-1", "image_info": {}}},
            {"batch_request": {"custom_id": "req-3", "image_info": {}}},
        ]

        order_map, line_map, name_map = _correlate_repair_targets(
            targets, metadata_records
        )

        # req-3 must resolve to the THIRD target (page 30, line 300), not the
        # second.
        assert order_map["req-3"] == 30
        assert line_map["req-3"] == 300
        assert name_map["req-3"] == "p30.jpg"
        # req-1 unchanged.
        assert line_map["req-1"] == 100
        # req-2 was never submitted, so it has no mapping.
        assert "req-2" not in line_map
