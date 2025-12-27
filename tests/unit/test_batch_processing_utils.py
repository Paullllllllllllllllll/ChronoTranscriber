"""Unit tests for modules/llm/batch/batch_utils.py.

Tests batch processing utilities for diagnostics and metadata extraction.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.llm.batch.batch_utils import (
    diagnose_batch_failure,
    extract_custom_id_mapping,
)


class TestDiagnoseBatchFailure:
    """Tests for diagnose_batch_failure function."""
    
    @pytest.mark.unit
    def test_failed_batch(self):
        """Test diagnosis of failed batch."""
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.status = "failed"
        mock_client.batches.retrieve.return_value = mock_batch
        
        with patch('modules.llm.batch.batch_utils.sdk_to_dict', return_value={"status": "failed"}):
            result = diagnose_batch_failure("batch_123", mock_client)
        
        assert "failed" in result.lower()
        assert "batch_123" in result
    
    @pytest.mark.unit
    def test_cancelled_batch(self):
        """Test diagnosis of cancelled batch."""
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.status = "cancelled"
        mock_client.batches.retrieve.return_value = mock_batch
        
        with patch('modules.llm.batch.batch_utils.sdk_to_dict', return_value={"status": "cancelled"}):
            result = diagnose_batch_failure("batch_456", mock_client)
        
        assert "cancelled" in result.lower()
    
    @pytest.mark.unit
    def test_expired_batch(self):
        """Test diagnosis of expired batch."""
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.status = "expired"
        mock_client.batches.retrieve.return_value = mock_batch
        
        with patch('modules.llm.batch.batch_utils.sdk_to_dict', return_value={"status": "expired"}):
            result = diagnose_batch_failure("batch_789", mock_client)
        
        assert "expired" in result.lower()
        assert "24 hours" in result
    
    @pytest.mark.unit
    def test_in_progress_batch(self):
        """Test diagnosis of in-progress batch."""
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.status = "in_progress"
        mock_client.batches.retrieve.return_value = mock_batch
        
        with patch('modules.llm.batch.batch_utils.sdk_to_dict', return_value={"status": "in_progress"}):
            result = diagnose_batch_failure("batch_abc", mock_client)
        
        assert "in_progress" in result
    
    @pytest.mark.unit
    def test_batch_not_found(self):
        """Test diagnosis when batch not found."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.side_effect = Exception("Batch not found in the system")
        
        result = diagnose_batch_failure("nonexistent", mock_client)
        
        assert "not found" in result.lower()
    
    @pytest.mark.unit
    def test_unauthorized_error(self):
        """Test diagnosis of unauthorized error."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.side_effect = Exception("Unauthorized access")
        
        result = diagnose_batch_failure("batch_xyz", mock_client)
        
        assert "unauthorized" in result.lower()
    
    @pytest.mark.unit
    def test_quota_error(self):
        """Test diagnosis of quota error."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.side_effect = Exception("Quota exceeded")
        
        result = diagnose_batch_failure("batch_quota", mock_client)
        
        assert "quota" in result.lower()
    
    @pytest.mark.unit
    def test_unknown_error(self):
        """Test diagnosis of unknown error."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.side_effect = Exception("Something unexpected happened")
        
        result = diagnose_batch_failure("batch_err", mock_client)
        
        assert "error" in result.lower()


class TestExtractCustomIdMapping:
    """Tests for extract_custom_id_mapping function."""
    
    @pytest.mark.unit
    def test_extracts_batch_request_records(self, temp_dir):
        """Test extraction from batch_request records."""
        jsonl_file = temp_dir / "batch.jsonl"
        records = [
            {
                "batch_request": {
                    "custom_id": "req_001",
                    "image_info": {
                        "image_name": "page_001.png",
                        "order_index": 0,
                    }
                }
            },
            {
                "batch_request": {
                    "custom_id": "req_002",
                    "image_info": {
                        "image_name": "page_002.png",
                        "order_index": 1,
                    }
                }
            },
        ]
        
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert len(custom_id_map) == 2
        assert "req_001" in custom_id_map
        assert "req_002" in custom_id_map
        assert custom_id_map["req_001"]["image_name"] == "page_001.png"
        assert batch_order["req_001"] == 0
        assert batch_order["req_002"] == 1
    
    @pytest.mark.unit
    def test_extracts_image_metadata_records(self, temp_dir):
        """Test extraction from image_metadata records."""
        jsonl_file = temp_dir / "metadata.jsonl"
        records = [
            {
                "image_metadata": {
                    "custom_id": "img_001",
                    "image_name": "scan_001.jpg",
                    "order_index": 0,
                }
            },
        ]
        
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert "img_001" in custom_id_map
        assert custom_id_map["img_001"]["image_name"] == "scan_001.jpg"
    
    @pytest.mark.unit
    def test_mixed_record_types(self, temp_dir):
        """Test extraction from mixed record types."""
        jsonl_file = temp_dir / "mixed.jsonl"
        records = [
            {
                "batch_request": {
                    "custom_id": "req_001",
                    "image_info": {"image_name": "a.png", "order_index": 0}
                }
            },
            {
                "image_metadata": {
                    "custom_id": "img_002",
                    "image_name": "b.png",
                    "order_index": 1,
                }
            },
            {"some_other_record": "ignored"},
        ]
        
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert len(custom_id_map) == 2
        assert "req_001" in custom_id_map
        assert "img_002" in custom_id_map
    
    @pytest.mark.unit
    def test_empty_file(self, temp_dir):
        """Test extraction from empty file."""
        jsonl_file = temp_dir / "empty.jsonl"
        jsonl_file.write_text("", encoding="utf-8")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert custom_id_map == {}
        assert batch_order == {}
    
    @pytest.mark.unit
    def test_invalid_json_lines_skipped(self, temp_dir):
        """Test that invalid JSON lines are skipped."""
        jsonl_file = temp_dir / "invalid.jsonl"
        content = 'invalid json line\n{"batch_request": {"custom_id": "valid", "image_info": {"name": "x"}}}\n'
        jsonl_file.write_text(content, encoding="utf-8")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert "valid" in custom_id_map
    
    @pytest.mark.unit
    def test_nonexistent_file(self, temp_dir):
        """Test extraction from nonexistent file."""
        nonexistent = temp_dir / "nonexistent.jsonl"
        
        custom_id_map, batch_order = extract_custom_id_mapping(nonexistent)
        
        assert custom_id_map == {}
        assert batch_order == {}
    
    @pytest.mark.unit
    def test_missing_custom_id(self, temp_dir):
        """Test records without custom_id are skipped."""
        jsonl_file = temp_dir / "no_id.jsonl"
        records = [
            {"batch_request": {"image_info": {"name": "x"}}},  # No custom_id
            {"batch_request": {"custom_id": "has_id", "image_info": {"name": "y"}}},
        ]
        
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert len(custom_id_map) == 1
        assert "has_id" in custom_id_map
    
    @pytest.mark.unit
    def test_missing_image_info(self, temp_dir):
        """Test records without image_info are skipped."""
        jsonl_file = temp_dir / "no_info.jsonl"
        records = [
            {"batch_request": {"custom_id": "no_info"}},  # No image_info
            {"batch_request": {"custom_id": "has_info", "image_info": {"name": "y"}}},
        ]
        
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        custom_id_map, batch_order = extract_custom_id_mapping(jsonl_file)
        
        assert len(custom_id_map) == 1
        assert "has_info" in custom_id_map
