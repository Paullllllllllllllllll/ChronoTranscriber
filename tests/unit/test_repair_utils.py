"""Unit tests for modules/operations/repair/utils.py."""

from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestExtractImageNameFromFailureLine:
    """Tests for extract_image_name_from_failure_line function."""

    @pytest.mark.unit
    def test_extracts_from_transcription_error(self):
        """Test extraction from transcription error placeholder."""
        from modules.operations.repair.utils import extract_image_name_from_failure_line
        
        line = "[transcription error: IMG_0001.png]"
        result = extract_image_name_from_failure_line(line)
        
        assert result == "IMG_0001.png"

    @pytest.mark.unit
    def test_extracts_from_not_possible(self):
        """Test extraction from transcription not possible placeholder."""
        from modules.operations.repair.utils import extract_image_name_from_failure_line
        
        line = "[Transcription not possible: page_12.jpg]"
        result = extract_image_name_from_failure_line(line)
        
        assert result == "page_12.jpg"

    @pytest.mark.unit
    def test_extracts_from_no_text(self):
        """Test extraction from no transcribable text placeholder."""
        from modules.operations.repair.utils import extract_image_name_from_failure_line
        
        line = "[No transcribable text: scan_03.png]"
        result = extract_image_name_from_failure_line(line)
        
        assert result == "scan_03.png"

    @pytest.mark.unit
    def test_extracts_with_page_prefix(self):
        """Test extraction with Page N: prefix."""
        from modules.operations.repair.utils import extract_image_name_from_failure_line
        
        line = "Page 5: [transcription error: page_005.png]"
        result = extract_image_name_from_failure_line(line)
        
        assert result == "page_005.png"

    @pytest.mark.unit
    def test_handles_error_with_semicolon(self):
        """Test handling error with additional info after semicolon."""
        from modules.operations.repair.utils import extract_image_name_from_failure_line
        
        line = "[transcription error: scan_03.png; status 400; code invalid_image]"
        result = extract_image_name_from_failure_line(line)
        
        assert result == "scan_03.png"

    @pytest.mark.unit
    def test_returns_none_for_normal_text(self):
        """Test returns None for normal text."""
        from modules.operations.repair.utils import extract_image_name_from_failure_line
        
        line = "This is normal transcription text."
        result = extract_image_name_from_failure_line(line)
        
        assert result is None


class TestIsFailureLine:
    """Tests for is_failure_line function."""

    @pytest.mark.unit
    def test_detects_transcription_error(self):
        """Test detection of transcription error."""
        from modules.operations.repair.utils import is_failure_line
        
        assert is_failure_line("[transcription error: image.png]") is True

    @pytest.mark.unit
    def test_detects_not_possible(self):
        """Test detection of transcription not possible."""
        from modules.operations.repair.utils import is_failure_line
        
        assert is_failure_line("[Transcription not possible: image.png]") is True

    @pytest.mark.unit
    def test_detects_with_prefix(self):
        """Test detection with page prefix."""
        from modules.operations.repair.utils import is_failure_line
        
        assert is_failure_line("Page 1: [transcription error: image.png]") is True

    @pytest.mark.unit
    def test_normal_text_not_failure(self):
        """Test that normal text is not detected as failure."""
        from modules.operations.repair.utils import is_failure_line
        
        assert is_failure_line("Normal transcription text") is False

    @pytest.mark.unit
    def test_empty_line_not_failure(self):
        """Test that empty line is not detected as failure."""
        from modules.operations.repair.utils import is_failure_line
        
        assert is_failure_line("") is False
        assert is_failure_line("   ") is False


class TestCollectImageEntriesFromJsonl:
    """Tests for collect_image_entries_from_jsonl function."""

    @pytest.mark.unit
    def test_collects_from_image_metadata(self, temp_dir):
        """Test collection from image_metadata records."""
        from modules.operations.repair.utils import collect_image_entries_from_jsonl
        
        jsonl_file = temp_dir / "test.jsonl"
        records = [
            {"image_metadata": {"order_index": 0, "image_name": "page_001.png", "pre_processed_image": "/path/to/page_001.png"}},
            {"image_metadata": {"order_index": 1, "image_name": "page_002.png", "pre_processed_image": "/path/to/page_002.png"}},
        ]
        with open(jsonl_file, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        
        entries = collect_image_entries_from_jsonl(jsonl_file)
        
        assert len(entries) == 2
        assert entries[0].order_index == 0
        assert entries[0].image_name == "page_001.png"
        assert entries[1].order_index == 1

    @pytest.mark.unit
    def test_returns_empty_for_nonexistent_file(self, temp_dir):
        """Test returns empty list for nonexistent file."""
        from modules.operations.repair.utils import collect_image_entries_from_jsonl
        
        entries = collect_image_entries_from_jsonl(temp_dir / "nonexistent.jsonl")
        
        assert entries == []

    @pytest.mark.unit
    def test_returns_empty_for_none_path(self):
        """Test returns empty list for None path."""
        from modules.operations.repair.utils import collect_image_entries_from_jsonl
        
        entries = collect_image_entries_from_jsonl(None)
        
        assert entries == []

    @pytest.mark.unit
    def test_sorted_by_order_index(self, temp_dir):
        """Test entries are sorted by order_index."""
        from modules.operations.repair.utils import collect_image_entries_from_jsonl
        
        jsonl_file = temp_dir / "test.jsonl"
        records = [
            {"image_metadata": {"order_index": 2, "image_name": "page_003.png"}},
            {"image_metadata": {"order_index": 0, "image_name": "page_001.png"}},
            {"image_metadata": {"order_index": 1, "image_name": "page_002.png"}},
        ]
        with open(jsonl_file, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        
        entries = collect_image_entries_from_jsonl(jsonl_file)
        
        assert entries[0].order_index == 0
        assert entries[1].order_index == 1
        assert entries[2].order_index == 2


class TestFindFailureIndices:
    """Tests for find_failure_indices function."""

    @pytest.mark.unit
    def test_finds_error_indices(self):
        """Test finding indices of error lines."""
        from modules.operations.repair.utils import find_failure_indices
        
        lines = [
            "Normal text",
            "[transcription error: page_001.png]",
            "More normal text",
            "[Transcription not possible: page_003.png]",
        ]
        
        indices = find_failure_indices(lines, include_no_text=False)
        
        assert indices == [1, 3]

    @pytest.mark.unit
    def test_includes_no_text_when_flag_true(self):
        """Test includes no_text lines when flag is True."""
        from modules.operations.repair.utils import find_failure_indices
        
        lines = [
            "[transcription error: page_001.png]",
            "[No transcribable text: page_002.png]",
        ]
        
        indices = find_failure_indices(lines, include_no_text=True)
        
        assert indices == [0, 1]

    @pytest.mark.unit
    def test_excludes_no_text_when_flag_false(self):
        """Test excludes no_text lines when flag is False."""
        from modules.operations.repair.utils import find_failure_indices
        
        lines = [
            "[transcription error: page_001.png]",
            "[No transcribable text: page_002.png]",
        ]
        
        indices = find_failure_indices(lines, include_no_text=False)
        
        assert indices == [0]

    @pytest.mark.unit
    def test_returns_empty_for_no_failures(self):
        """Test returns empty list when no failures."""
        from modules.operations.repair.utils import find_failure_indices
        
        lines = ["Normal text", "More text"]
        
        indices = find_failure_indices(lines, include_no_text=False)
        
        assert indices == []


class TestBackupFile:
    """Tests for backup_file function."""

    @pytest.mark.unit
    def test_creates_backup(self, temp_dir):
        """Test that backup file is created."""
        from modules.operations.repair.utils import backup_file
        
        original = temp_dir / "test.txt"
        original.write_text("Original content")
        
        backup_path = backup_file(original)
        
        assert backup_path.exists()
        assert backup_path.read_text() == "Original content"
        assert ".bak." in backup_path.name

    @pytest.mark.unit
    def test_backup_has_timestamp(self, temp_dir):
        """Test that backup filename contains timestamp."""
        from modules.operations.repair.utils import backup_file
        import re
        
        original = temp_dir / "test.txt"
        original.write_text("Content")
        
        backup_path = backup_file(original)
        
        # Should contain timestamp pattern like 20240101-120000
        pattern = r"\d{8}-\d{6}"
        assert re.search(pattern, backup_path.name)


class TestReadFinalLines:
    """Tests for read_final_lines function."""

    @pytest.mark.unit
    def test_reads_lines(self, temp_dir):
        """Test reading lines from file."""
        from modules.operations.repair.utils import read_final_lines
        
        txt_file = temp_dir / "transcription.txt"
        txt_file.write_text("Line 1\nLine 2\nLine 3")
        
        lines = read_final_lines(txt_file)
        
        assert lines == ["Line 1", "Line 2", "Line 3"]

    @pytest.mark.unit
    def test_empty_file(self, temp_dir):
        """Test reading empty file."""
        from modules.operations.repair.utils import read_final_lines
        
        txt_file = temp_dir / "empty.txt"
        txt_file.write_text("")
        
        lines = read_final_lines(txt_file)
        
        # str.splitlines() on empty string returns empty list
        assert lines == []


class TestImageEntry:
    """Tests for ImageEntry dataclass."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test ImageEntry initialization."""
        from modules.operations.repair.utils import ImageEntry
        
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
    def test_default_page_number(self):
        """Test ImageEntry default page_number."""
        from modules.operations.repair.utils import ImageEntry
        
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
    def test_initialization(self, temp_dir):
        """Test Job initialization."""
        from modules.operations.repair.utils import Job
        
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
