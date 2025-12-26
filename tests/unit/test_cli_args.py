"""Unit tests for modules/core/cli_args.py.

Tests CLI argument parsing, path resolution, and validation utilities.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from modules.core.cli_args import (
    create_transcriber_parser,
    create_repair_parser,
    create_check_batches_parser,
    create_cancel_batches_parser,
    create_postprocess_parser,
    resolve_path,
    validate_input_path,
    validate_output_path,
    parse_indices,
)


class TestCreateTranscriberParser:
    """Tests for create_transcriber_parser function."""
    
    @pytest.mark.unit
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_transcriber_parser()
        assert parser is not None
        assert parser.description is not None
    
    @pytest.mark.unit
    def test_parse_basic_args(self):
        """Test parsing basic required arguments."""
        parser = create_transcriber_parser()
        args = parser.parse_args([
            "--input", "test_input",
            "--output", "test_output",
            "--type", "images",
            "--method", "tesseract",
        ])
        assert args.input == "test_input"
        assert args.output == "test_output"
        assert args.type == "images"
        assert args.method == "tesseract"
    
    @pytest.mark.unit
    def test_parse_gpt_args(self):
        """Test parsing GPT-specific arguments."""
        parser = create_transcriber_parser()
        args = parser.parse_args([
            "--input", "input",
            "--output", "output",
            "--type", "pdfs",
            "--method", "gpt",
            "--batch",
            "--schema", "custom_schema",
            "--context", "context.txt",
        ])
        assert args.method == "gpt"
        assert args.batch is True
        assert args.schema == "custom_schema"
        assert args.context == "context.txt"
    
    @pytest.mark.unit
    def test_parse_auto_mode(self):
        """Test parsing auto mode argument."""
        parser = create_transcriber_parser()
        args = parser.parse_args(["--auto"])
        assert args.auto is True
    
    @pytest.mark.unit
    def test_parse_files_and_recursive(self):
        """Test parsing files and recursive arguments."""
        parser = create_transcriber_parser()
        args = parser.parse_args([
            "--input", "input",
            "--output", "output",
            "--type", "images",
            "--method", "tesseract",
            "--files", "file1.pdf", "file2.pdf",
            "--recursive",
        ])
        assert args.files == ["file1.pdf", "file2.pdf"]
        assert args.recursive is True
    
    @pytest.mark.unit
    def test_invalid_type_choice(self):
        """Test that invalid type choice raises error."""
        parser = create_transcriber_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--input", "input",
                "--output", "output",
                "--type", "invalid",
                "--method", "tesseract",
            ])
    
    @pytest.mark.unit
    def test_invalid_method_choice(self):
        """Test that invalid method choice raises error."""
        parser = create_transcriber_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--input", "input",
                "--output", "output",
                "--type", "images",
                "--method", "invalid",
            ])


class TestCreateRepairParser:
    """Tests for create_repair_parser function."""
    
    @pytest.mark.unit
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_repair_parser()
        assert parser is not None
    
    @pytest.mark.unit
    def test_parse_basic_args(self):
        """Test parsing basic repair arguments."""
        parser = create_repair_parser()
        args = parser.parse_args(["--transcription", "test.txt"])
        assert args.transcription == "test.txt"
    
    @pytest.mark.unit
    def test_parse_failure_flags(self):
        """Test parsing failure type flags."""
        parser = create_repair_parser()
        args = parser.parse_args([
            "--transcription", "test.txt",
            "--errors-only",
            "--not-possible",
            "--no-text",
        ])
        assert args.errors_only is True
        assert args.not_possible is True
        assert args.no_text is True
    
    @pytest.mark.unit
    def test_parse_all_failures(self):
        """Test parsing all-failures flag."""
        parser = create_repair_parser()
        args = parser.parse_args([
            "--transcription", "test.txt",
            "--all-failures",
        ])
        assert args.all_failures is True
    
    @pytest.mark.unit
    def test_parse_indices(self):
        """Test parsing indices argument."""
        parser = create_repair_parser()
        args = parser.parse_args([
            "--transcription", "test.txt",
            "--indices", "0,5,12",
        ])
        assert args.indices == "0,5,12"


class TestCreateCheckBatchesParser:
    """Tests for create_check_batches_parser function."""
    
    @pytest.mark.unit
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_check_batches_parser()
        assert parser is not None
    
    @pytest.mark.unit
    def test_parse_directory(self):
        """Test parsing directory argument."""
        parser = create_check_batches_parser()
        args = parser.parse_args(["--directory", "results"])
        assert args.directory == "results"
    
    @pytest.mark.unit
    def test_parse_no_diagnostics(self):
        """Test parsing no-diagnostics flag."""
        parser = create_check_batches_parser()
        args = parser.parse_args(["--no-diagnostics"])
        assert args.no_diagnostics is True


class TestCreateCancelBatchesParser:
    """Tests for create_cancel_batches_parser function."""
    
    @pytest.mark.unit
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_cancel_batches_parser()
        assert parser is not None
    
    @pytest.mark.unit
    def test_parse_batch_ids(self):
        """Test parsing batch-ids argument."""
        parser = create_cancel_batches_parser()
        args = parser.parse_args(["--batch-ids", "batch_123", "batch_456"])
        assert args.batch_ids == ["batch_123", "batch_456"]
    
    @pytest.mark.unit
    def test_parse_force(self):
        """Test parsing force flag."""
        parser = create_cancel_batches_parser()
        args = parser.parse_args(["--force"])
        assert args.force is True


class TestResolvePath:
    """Tests for resolve_path function."""
    
    @pytest.mark.unit
    def test_absolute_path(self, temp_dir):
        """Test that absolute paths are returned unchanged."""
        abs_path = temp_dir / "test"
        result = resolve_path(str(abs_path))
        assert result == abs_path
    
    @pytest.mark.unit
    def test_relative_path_with_base(self, temp_dir):
        """Test resolving relative path with base directory."""
        result = resolve_path("subdir/file.txt", temp_dir)
        assert result == (temp_dir / "subdir" / "file.txt").resolve()
    
    @pytest.mark.unit
    def test_none_path_with_base(self, temp_dir):
        """Test that None path returns base path."""
        result = resolve_path(None, temp_dir)
        assert result == temp_dir.resolve()
    
    @pytest.mark.unit
    def test_none_path_without_base(self):
        """Test that None path without base returns cwd."""
        result = resolve_path(None, None)
        assert result == Path.cwd()


class TestValidateInputPath:
    """Tests for validate_input_path function."""
    
    @pytest.mark.unit
    def test_valid_existing_file(self, sample_text_file):
        """Test validation passes for existing file."""
        validate_input_path(sample_text_file)  # Should not raise
    
    @pytest.mark.unit
    def test_valid_existing_directory(self, temp_input_dir):
        """Test validation passes for existing directory."""
        validate_input_path(temp_input_dir)  # Should not raise
    
    @pytest.mark.unit
    def test_invalid_nonexistent_path(self, temp_dir):
        """Test validation fails for nonexistent path."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_input_path(temp_dir / "nonexistent")
    
    @pytest.mark.unit
    def test_skip_existence_check(self, temp_dir):
        """Test validation with must_exist=False."""
        validate_input_path(temp_dir / "nonexistent", must_exist=False)


class TestValidateOutputPath:
    """Tests for validate_output_path function."""
    
    @pytest.mark.unit
    def test_valid_existing_directory(self, temp_output_dir):
        """Test validation passes for existing directory."""
        validate_output_path(temp_output_dir)  # Should not raise
    
    @pytest.mark.unit
    def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        new_path = temp_dir / "new" / "nested" / "dir"
        validate_output_path(new_path, create_parents=True)
        assert new_path.exists()
    
    @pytest.mark.unit
    def test_invalid_file_as_directory(self, sample_text_file):
        """Test validation fails when output is a file not directory."""
        with pytest.raises(ValueError, match="not a directory"):
            validate_output_path(sample_text_file)


class TestParseIndices:
    """Tests for parse_indices function."""
    
    @pytest.mark.unit
    def test_single_indices(self):
        """Test parsing comma-separated single indices."""
        result = parse_indices("0,5,12")
        assert result == [0, 5, 12]
    
    @pytest.mark.unit
    def test_range_indices(self):
        """Test parsing range notation."""
        result = parse_indices("1-5")
        assert result == [1, 2, 3, 4, 5]
    
    @pytest.mark.unit
    def test_mixed_indices(self):
        """Test parsing mixed single and range indices."""
        result = parse_indices("0,3-5,10")
        assert result == [0, 3, 4, 5, 10]
    
    @pytest.mark.unit
    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result = parse_indices("0, 5, 12")
        assert result == [0, 5, 12]
    
    @pytest.mark.unit
    def test_duplicate_removal(self):
        """Test that duplicates are removed."""
        result = parse_indices("1,1,2,2,3")
        assert result == [1, 2, 3]
    
    @pytest.mark.unit
    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError):
            parse_indices("invalid")
    
    @pytest.mark.unit
    def test_empty_parts_ignored(self):
        """Test that empty parts are ignored."""
        result = parse_indices("1,,2,")
        assert result == [1, 2]
