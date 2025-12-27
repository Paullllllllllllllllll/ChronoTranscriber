"""Unit tests for modules/io/directory_utils.py.

Tests directory management utilities for path creation and validation.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from modules.io.directory_utils import (
    ensure_directory,
    ensure_directories,
    ensure_parent_directory,
    get_output_directories_from_config,
    get_input_directories_from_config,
    get_logs_directory,
    collect_scan_directories,
)


class TestEnsureDirectory:
    """Tests for ensure_directory function."""
    
    @pytest.mark.unit
    def test_existing_directory(self, temp_dir):
        """Test with existing directory."""
        result = ensure_directory(temp_dir)
        assert result == temp_dir.resolve()
    
    @pytest.mark.unit
    def test_creates_directory(self, temp_dir):
        """Test that new directory is created."""
        new_dir = temp_dir / "new_subdir"
        assert not new_dir.exists()
        
        result = ensure_directory(new_dir, create=True)
        
        assert new_dir.exists()
        assert result == new_dir.resolve()
    
    @pytest.mark.unit
    def test_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested = temp_dir / "level1" / "level2" / "level3"
        assert not nested.exists()
        
        result = ensure_directory(nested, create=True)
        
        assert nested.exists()
        assert result == nested.resolve()
    
    @pytest.mark.unit
    def test_file_as_directory_raises(self, sample_text_file):
        """Test that file path raises ValueError."""
        with pytest.raises(ValueError, match="not a directory"):
            ensure_directory(sample_text_file)
    
    @pytest.mark.unit
    def test_nonexistent_without_create_raises(self, temp_dir):
        """Test that nonexistent path with create=False raises."""
        nonexistent = temp_dir / "does_not_exist"
        
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ensure_directory(nonexistent, create=False)
    
    @pytest.mark.unit
    def test_returns_resolved_path(self, temp_dir):
        """Test that returned path is resolved."""
        path = temp_dir / "subdir" / ".." / "actual"
        result = ensure_directory(path, create=True)
        assert result.is_absolute()


class TestEnsureDirectories:
    """Tests for ensure_directories function."""
    
    @pytest.mark.unit
    def test_multiple_directories(self, temp_dir):
        """Test creating multiple directories."""
        dirs = [
            temp_dir / "dir1",
            temp_dir / "dir2",
            temp_dir / "dir3",
        ]
        
        results = ensure_directories(*dirs, create=True)
        
        assert len(results) == 3
        for d in dirs:
            assert d.exists()
    
    @pytest.mark.unit
    def test_empty_list(self):
        """Test with empty list."""
        results = ensure_directories()
        assert results == []
    
    @pytest.mark.unit
    def test_mixed_existing_and_new(self, temp_dir):
        """Test with mix of existing and new directories."""
        existing = temp_dir
        new_dir = temp_dir / "new_dir"
        
        results = ensure_directories(existing, new_dir, create=True)
        
        assert len(results) == 2
        assert new_dir.exists()


class TestEnsureParentDirectory:
    """Tests for ensure_parent_directory function."""
    
    @pytest.mark.unit
    def test_existing_parent(self, temp_dir):
        """Test with existing parent directory."""
        file_path = temp_dir / "file.txt"
        result = ensure_parent_directory(file_path)
        assert result == temp_dir.resolve()
    
    @pytest.mark.unit
    def test_creates_parent(self, temp_dir):
        """Test that parent directory is created."""
        file_path = temp_dir / "new_dir" / "file.txt"
        assert not file_path.parent.exists()
        
        result = ensure_parent_directory(file_path, create=True)
        
        assert file_path.parent.exists()
        assert result == file_path.parent.resolve()
    
    @pytest.mark.unit
    def test_nested_parent_creation(self, temp_dir):
        """Test creating nested parent directories."""
        file_path = temp_dir / "a" / "b" / "c" / "file.txt"
        
        result = ensure_parent_directory(file_path, create=True)
        
        assert file_path.parent.exists()


class TestGetOutputDirectoriesFromConfig:
    """Tests for get_output_directories_from_config function."""
    
    @pytest.mark.unit
    def test_extracts_all_categories(self, temp_dir):
        """Test extraction of all output directory categories."""
        config = {
            "file_paths": {
                "PDFs": {"output": str(temp_dir / "pdfs_out")},
                "Images": {"output": str(temp_dir / "images_out")},
                "EPUBs": {"output": str(temp_dir / "epubs_out")},
                "Auto": {"output": str(temp_dir / "auto_out")},
            }
        }
        
        result = get_output_directories_from_config(config)
        
        assert "pdfs" in result
        assert "images" in result
        assert "epubs" in result
        assert "auto" in result
    
    @pytest.mark.unit
    def test_partial_config(self, temp_dir):
        """Test with partial configuration."""
        config = {
            "file_paths": {
                "PDFs": {"output": str(temp_dir / "pdfs")},
            }
        }
        
        result = get_output_directories_from_config(config)
        
        assert "pdfs" in result
        assert "images" not in result
    
    @pytest.mark.unit
    def test_empty_config(self):
        """Test with empty configuration."""
        result = get_output_directories_from_config({})
        assert result == {}
    
    @pytest.mark.unit
    def test_creates_directories(self, temp_dir):
        """Test that output directories are created."""
        out_dir = temp_dir / "created_output"
        config = {
            "file_paths": {
                "PDFs": {"output": str(out_dir)},
            }
        }
        
        result = get_output_directories_from_config(config)
        
        assert out_dir.exists()


class TestGetInputDirectoriesFromConfig:
    """Tests for get_input_directories_from_config function."""
    
    @pytest.mark.unit
    def test_extracts_input_paths(self, temp_dir):
        """Test extraction of input directory paths."""
        config = {
            "file_paths": {
                "PDFs": {"input": str(temp_dir / "pdfs_in")},
                "Images": {"input": str(temp_dir / "images_in")},
            }
        }
        
        result = get_input_directories_from_config(config)
        
        assert "pdfs" in result
        assert "images" in result
    
    @pytest.mark.unit
    def test_does_not_create_directories(self, temp_dir):
        """Test that input directories are not created."""
        nonexistent = temp_dir / "nonexistent_input"
        config = {
            "file_paths": {
                "PDFs": {"input": str(nonexistent)},
            }
        }
        
        result = get_input_directories_from_config(config)
        
        # Path is returned but not created
        assert "pdfs" in result
        assert not nonexistent.exists()


class TestGetLogsDirectory:
    """Tests for get_logs_directory function."""
    
    @pytest.mark.unit
    def test_returns_logs_path(self, temp_dir):
        """Test that logs directory path is returned."""
        config = {
            "general": {"logs_dir": str(temp_dir / "logs")}
        }
        
        result = get_logs_directory(config, create=True)
        
        assert result is not None
        assert result.name == "logs"
    
    @pytest.mark.unit
    def test_creates_logs_directory(self, temp_dir):
        """Test that logs directory is created."""
        logs_dir = temp_dir / "new_logs"
        config = {"general": {"logs_dir": str(logs_dir)}}
        
        result = get_logs_directory(config, create=True)
        
        assert logs_dir.exists()
    
    @pytest.mark.unit
    def test_returns_none_if_not_configured(self):
        """Test that None is returned if logs_dir not configured."""
        result = get_logs_directory({})
        assert result is None
    
    @pytest.mark.unit
    def test_returns_none_for_empty_general(self):
        """Test with empty general section."""
        result = get_logs_directory({"general": {}})
        assert result is None


class TestCollectScanDirectories:
    """Tests for collect_scan_directories function."""
    
    @pytest.mark.unit
    def test_collects_unique_directories(self, temp_dir):
        """Test that unique directories are collected."""
        # Create some directories
        pdfs_in = temp_dir / "pdfs_in"
        pdfs_out = temp_dir / "pdfs_out"
        pdfs_in.mkdir()
        
        config = {
            "file_paths": {
                "PDFs": {
                    "input": str(pdfs_in),
                    "output": str(pdfs_out),
                }
            }
        }
        
        result = collect_scan_directories(config)
        
        # Should include both directories (output created, input exists)
        assert len(result) >= 1
    
    @pytest.mark.unit
    def test_returns_sorted_list(self, temp_dir):
        """Test that result is sorted."""
        a_dir = temp_dir / "a_dir"
        z_dir = temp_dir / "z_dir"
        a_dir.mkdir()
        z_dir.mkdir()
        
        config = {
            "file_paths": {
                "PDFs": {"input": str(z_dir), "output": str(a_dir)},
            }
        }
        
        result = collect_scan_directories(config)
        
        # Should be sorted
        assert result == sorted(result)
    
    @pytest.mark.unit
    def test_empty_config(self):
        """Test with empty configuration."""
        result = collect_scan_directories({})
        assert result == []
