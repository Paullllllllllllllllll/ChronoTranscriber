"""Unit tests for modules/operations/jsonl_utils.py.

Tests JSONL file parsing and manipulation utilities.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from modules.operations.jsonl_utils import (
    read_jsonl_records,
    write_jsonl_record,
    extract_image_metadata,
    extract_batch_ids,
    is_batch_jsonl,
    ImageMetadata,
)


class TestReadJsonlRecords:
    """Tests for read_jsonl_records function."""
    
    @pytest.mark.unit
    def test_reads_valid_jsonl(self, temp_dir):
        """Test reading valid JSONL file."""
        jsonl_path = temp_dir / "test.jsonl"
        records = [
            {"id": 1, "text": "First"},
            {"id": 2, "text": "Second"},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        
        result = read_jsonl_records(jsonl_path)
        
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
    
    @pytest.mark.unit
    def test_skips_empty_lines(self, temp_dir):
        """Test that empty lines are skipped."""
        jsonl_path = temp_dir / "test.jsonl"
        content = '{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n'
        jsonl_path.write_text(content, encoding="utf-8")
        
        result = read_jsonl_records(jsonl_path)
        
        assert len(result) == 3
    
    @pytest.mark.unit
    def test_handles_invalid_json_lines(self, temp_dir):
        """Test handling of invalid JSON lines."""
        jsonl_path = temp_dir / "test.jsonl"
        content = '{"id": 1}\nnot valid json\n{"id": 3}\n'
        jsonl_path.write_text(content, encoding="utf-8")
        
        result = read_jsonl_records(jsonl_path)
        
        # Should skip invalid line and return valid ones
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_nonexistent_file_returns_empty(self, temp_dir):
        """Test that nonexistent file returns empty list."""
        result = read_jsonl_records(temp_dir / "nonexistent.jsonl")
        assert result == []
    
    @pytest.mark.unit
    def test_empty_file(self, temp_dir):
        """Test reading empty file."""
        jsonl_path = temp_dir / "empty.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        
        result = read_jsonl_records(jsonl_path)
        
        assert result == []


class TestWriteJsonlRecord:
    """Tests for write_jsonl_record function."""
    
    @pytest.mark.unit
    def test_writes_record(self, temp_dir):
        """Test writing a single record."""
        jsonl_path = temp_dir / "output.jsonl"
        record = {"key": "value", "number": 42}
        
        write_jsonl_record(jsonl_path, record)
        
        # Verify written content
        content = jsonl_path.read_text(encoding="utf-8")
        parsed = json.loads(content.strip())
        assert parsed == record
    
    @pytest.mark.unit
    def test_appends_records(self, temp_dir):
        """Test appending multiple records."""
        jsonl_path = temp_dir / "output.jsonl"
        
        write_jsonl_record(jsonl_path, {"id": 1})
        write_jsonl_record(jsonl_path, {"id": 2})
        write_jsonl_record(jsonl_path, {"id": 3})
        
        # Verify all records
        records = read_jsonl_records(jsonl_path)
        assert len(records) == 3
        assert records[0]["id"] == 1
        assert records[2]["id"] == 3


class TestExtractBatchIds:
    """Tests for extract_batch_ids function."""
    
    @pytest.mark.unit
    def test_extracts_batch_ids(self, temp_dir):
        """Test extracting batch IDs from records."""
        records = [
            {"image_metadata": {"name": "page1.png"}},
            {"batch_tracking": {"batch_id": "batch_123", "provider": "openai"}},
            {"text_chunk": "Some text"},
        ]
        
        result = extract_batch_ids(records)
        
        assert len(result) == 1
        assert result[0] == "batch_123"
    
    @pytest.mark.unit
    def test_returns_empty_if_no_batch_tracking(self, temp_dir):
        """Test returning empty list when no batch tracking."""
        records = [
            {"image_metadata": {"name": "page1.png"}},
            {"text_chunk": "Some text"},
        ]
        
        result = extract_batch_ids(records)
        
        assert result == []


class TestExtractImageMetadata:
    """Tests for extract_image_metadata function."""
    
    @pytest.mark.unit
    def test_extracts_all_image_metadata(self, temp_dir):
        """Test extracting all image metadata records."""
        records = [
            {"image_metadata": {"image_name": "page1.png", "order_index": 0}},
            {"batch_tracking": {"batch_id": "batch_123"}},
            {"image_metadata": {"image_name": "page2.png", "order_index": 1}},
        ]
        
        result = extract_image_metadata(records)
        
        assert len(result) == 2
        assert result[0].image_name == "page1.png"
        assert result[1].image_name == "page2.png"
    
    @pytest.mark.unit
    def test_returns_empty_if_none(self, temp_dir):
        """Test returning empty list when no image metadata."""
        records = [
            {"batch_tracking": {"batch_id": "batch_123"}},
            {"text_chunk": "Some text"},
        ]
        
        result = extract_image_metadata(records)
        
        assert result == []


class TestIsBatchJsonl:
    """Tests for is_batch_jsonl function."""
    
    @pytest.mark.unit
    def test_detects_batch_jsonl(self, temp_dir):
        """Test detection of batch JSONL file."""
        jsonl_path = temp_dir / "test.jsonl"
        records = [
            {"batch_session": {"status": "submitted"}},
            {"image_metadata": {"image_name": "page1.png", "order_index": 0, "custom_id": "req-1"}},
            {"batch_tracking": {"batch_id": "batch_123"}},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        
        result = is_batch_jsonl(jsonl_path)
        
        assert result is True
    
    @pytest.mark.unit
    def test_non_batch_jsonl(self, temp_dir):
        """Test detection of non-batch JSONL file."""
        jsonl_path = temp_dir / "test.jsonl"
        records = [
            {"text_chunk": "Some text", "order_index": 0},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        
        result = is_batch_jsonl(jsonl_path)
        
        assert result is False
    
    @pytest.mark.unit
    def test_nonexistent_file(self, temp_dir):
        """Test handling of nonexistent file."""
        result = is_batch_jsonl(temp_dir / "nonexistent.jsonl")
        assert result is False
