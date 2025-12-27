"""Unit tests for modules/core/safe_paths.py.

Tests Windows path length limitation handling with safe directory/file names.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from modules.core.safe_paths import (
    create_safe_directory_name,
    create_safe_filename,
    create_safe_log_filename,
    ensure_path_safe,
    WINDOWS_MAX_PATH,
    MAX_SAFE_NAME_LENGTH,
    HASH_LENGTH,
    MIN_READABLE_LENGTH,
)


class TestCreateSafeDirectoryName:
    """Tests for create_safe_directory_name function."""
    
    @pytest.mark.unit
    def test_short_name_unchanged_with_hash(self):
        """Test that short names get hash appended."""
        result = create_safe_directory_name("short_name")
        assert result.startswith("short_name-")
        assert len(result) == len("short_name") + 1 + HASH_LENGTH  # name + dash + hash
    
    @pytest.mark.unit
    def test_long_name_truncated(self):
        """Test that long names are truncated."""
        long_name = "A" * 200
        result = create_safe_directory_name(long_name)
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert "-" in result  # Should have hash separator
    
    @pytest.mark.unit
    def test_suffix_included(self):
        """Test that suffix is appended correctly."""
        result = create_safe_directory_name("document", suffix="_working_files")
        assert result.endswith("_working_files")
    
    @pytest.mark.unit
    def test_long_name_with_suffix(self):
        """Test long name with suffix stays within limits."""
        long_name = "Very long document name that exceeds normal limits" * 3
        suffix = "_working_files"
        result = create_safe_directory_name(long_name, suffix=suffix)
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert result.endswith(suffix)
    
    @pytest.mark.unit
    def test_hash_provides_uniqueness(self):
        """Test that different names produce different hashes."""
        result1 = create_safe_directory_name("document_a")
        result2 = create_safe_directory_name("document_b")
        # Extract hashes (after last dash, before suffix)
        hash1 = result1.split("-")[-1]
        hash2 = result2.split("-")[-1]
        assert hash1 != hash2
    
    @pytest.mark.unit
    def test_same_name_produces_same_hash(self):
        """Test that same name always produces same hash."""
        result1 = create_safe_directory_name("consistent_name")
        result2 = create_safe_directory_name("consistent_name")
        assert result1 == result2
    
    @pytest.mark.unit
    def test_trailing_punctuation_stripped(self):
        """Test that trailing punctuation is stripped from truncated names."""
        # Create a name that will be truncated at a punctuation point
        long_name = "Document name with trailing... " + "x" * 100
        result = create_safe_directory_name(long_name)
        # Should not end with punctuation before the hash
        name_part = result.rsplit("-", 1)[0]
        assert not name_part.endswith((".", "-", "_", " "))


class TestCreateSafeFilename:
    """Tests for create_safe_filename function."""
    
    @pytest.mark.unit
    def test_short_name_no_truncation(self):
        """Test that short names are not truncated."""
        result = create_safe_filename("short", ".txt")
        assert result == "short.txt"  # No hash needed for short names
    
    @pytest.mark.unit
    def test_long_name_truncated_with_hash(self):
        """Test that long names are truncated and get hash."""
        long_name = "A" * 200
        result = create_safe_filename(long_name, ".txt")
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert result.endswith(".txt")
        assert "-" in result  # Should have hash
    
    @pytest.mark.unit
    def test_extension_preserved(self):
        """Test that file extension is always preserved."""
        result = create_safe_filename("A" * 200, ".jsonl")
        assert result.endswith(".jsonl")
    
    @pytest.mark.unit
    def test_parent_path_considered(self, temp_dir):
        """Test that parent path length is considered for truncation."""
        # Create a deep directory structure
        deep_dir = temp_dir / ("a" * 50) / ("b" * 50) / ("c" * 50)
        deep_dir.mkdir(parents=True, exist_ok=True)
        
        result = create_safe_filename("document", ".txt", parent_path=deep_dir)
        full_path_len = len(str(deep_dir / result))
        # Should stay under MAX_PATH with some margin
        assert full_path_len < WINDOWS_MAX_PATH
    
    @pytest.mark.unit
    def test_various_extensions(self):
        """Test with various file extensions."""
        extensions = [".txt", ".md", ".jsonl", ".pdf", ".png"]
        for ext in extensions:
            result = create_safe_filename("test_file", ext)
            assert result.endswith(ext)


class TestCreateSafeLogFilename:
    """Tests for create_safe_log_filename function."""
    
    @pytest.mark.unit
    def test_log_suffix_format(self):
        """Test that log filename has correct format."""
        result = create_safe_log_filename("document", "transcription")
        assert "_transcription_log.json" in result
    
    @pytest.mark.unit
    def test_short_name_format(self):
        """Test format with short base name."""
        result = create_safe_log_filename("doc", "summary")
        assert result.endswith("_summary_log.json")
    
    @pytest.mark.unit
    def test_long_name_truncated(self):
        """Test that long names are truncated."""
        long_name = "Very long document name" * 10
        result = create_safe_log_filename(long_name, "transcription")
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert result.endswith("_transcription_log.json")
    
    @pytest.mark.unit
    def test_hash_included(self):
        """Test that hash is always included for uniqueness."""
        result = create_safe_log_filename("document", "log")
        # Should have format: [name]-[hash]_[type]_log.json
        assert "-" in result


class TestEnsurePathSafe:
    """Tests for ensure_path_safe function."""
    
    @pytest.mark.unit
    def test_returns_resolved_path(self, temp_dir):
        """Test that path is resolved to absolute."""
        path = temp_dir / "subdir" / ".." / "file.txt"
        result = ensure_path_safe(path)
        assert result.is_absolute()
    
    @pytest.mark.unit
    def test_existing_path(self, temp_dir):
        """Test with existing directory."""
        result = ensure_path_safe(temp_dir)
        assert result == temp_dir.resolve()
    
    @pytest.mark.unit
    def test_nonexistent_path(self, temp_dir):
        """Test with non-existent path (should not raise)."""
        path = temp_dir / "nonexistent"
        result = ensure_path_safe(path)
        # Should return resolved path even if it doesn't exist
        assert result is not None


class TestConstants:
    """Tests for module constants."""
    
    @pytest.mark.unit
    def test_windows_max_path_value(self):
        """Test WINDOWS_MAX_PATH constant."""
        assert WINDOWS_MAX_PATH == 260
    
    @pytest.mark.unit
    def test_max_safe_name_length(self):
        """Test MAX_SAFE_NAME_LENGTH is reasonable."""
        assert MAX_SAFE_NAME_LENGTH < 255  # NTFS limit
        assert MAX_SAFE_NAME_LENGTH > 50  # Usable length
    
    @pytest.mark.unit
    def test_hash_length(self):
        """Test HASH_LENGTH provides enough uniqueness."""
        assert HASH_LENGTH >= 8  # At least 4 billion combinations
    
    @pytest.mark.unit
    def test_min_readable_length(self):
        """Test MIN_READABLE_LENGTH is reasonable."""
        assert MIN_READABLE_LENGTH >= 10  # Should be readable
        assert MIN_READABLE_LENGTH < MAX_SAFE_NAME_LENGTH / 2
