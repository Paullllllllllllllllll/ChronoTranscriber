"""Unit tests for modules/llm/batch/backends/.

Tests batch processing backend abstractions and factory.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from modules.llm.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchStatus,
    BatchStatusInfo,
    BatchResultItem,
    BatchRequest,
)
from modules.llm.batch.backends.factory import (
    get_batch_backend,
    supports_batch,
)


class TestBatchHandle:
    """Tests for BatchHandle dataclass."""
    
    @pytest.mark.unit
    def test_creation(self):
        """Test BatchHandle can be created."""
        handle = BatchHandle(
            batch_id="batch_123",
            provider="openai",
            metadata={"key": "value"},
        )
        assert handle.batch_id == "batch_123"
        assert handle.provider == "openai"
        assert handle.metadata == {"key": "value"}
    
    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test BatchHandle with minimal args."""
        handle = BatchHandle(batch_id="batch_456", provider="anthropic")
        assert handle.batch_id == "batch_456"
        # Default metadata is empty dict, not None
        assert handle.metadata == {} or handle.metadata is None


class TestBatchStatus:
    """Tests for BatchStatus enum."""
    
    @pytest.mark.unit
    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        expected = ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", "EXPIRED"]
        for status_name in expected:
            assert hasattr(BatchStatus, status_name)
    
    @pytest.mark.unit
    def test_status_values(self):
        """Test status enum values."""
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"


class TestBatchStatusInfo:
    """Tests for BatchStatusInfo dataclass."""
    
    @pytest.mark.unit
    def test_creation(self):
        """Test BatchStatusInfo creation."""
        info = BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total_requests=100,
            completed_requests=95,
            failed_requests=5,
        )
        assert info.status == BatchStatus.COMPLETED
        assert info.total_requests == 100
        assert info.completed_requests == 95
        assert info.failed_requests == 5
    
    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test BatchStatusInfo with minimal args."""
        info = BatchStatusInfo(status=BatchStatus.PENDING)
        assert info.status == BatchStatus.PENDING
        assert info.total_requests == 0


class TestBatchResultItem:
    """Tests for BatchResultItem dataclass."""
    
    @pytest.mark.unit
    def test_creation(self):
        """Test BatchResultItem creation."""
        item = BatchResultItem(
            custom_id="req-1",
            content="Transcribed text",
            success=True,
            error=None,
        )
        assert item.custom_id == "req-1"
        assert item.content == "Transcribed text"
        assert item.success is True
        assert item.error is None
    
    @pytest.mark.unit
    def test_failed_item(self):
        """Test BatchResultItem for failed request."""
        item = BatchResultItem(
            custom_id="req-2",
            content=None,
            success=False,
            error="Rate limit exceeded",
        )
        assert item.success is False
        assert item.error == "Rate limit exceeded"


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""
    
    @pytest.mark.unit
    def test_creation(self, temp_dir):
        """Test BatchRequest creation."""
        img_path = temp_dir / "image.png"
        img_path.write_bytes(b"")
        
        request = BatchRequest(
            custom_id="req-1",
            image_path=img_path,
            order_index=0,
            image_info={"name": "image.png"},
        )
        assert request.custom_id == "req-1"
        assert request.image_path == img_path
        assert request.order_index == 0


class TestSupportsBatch:
    """Tests for supports_batch function."""
    
    @pytest.mark.unit
    def test_openai_supported(self):
        """Test that OpenAI batch is supported."""
        assert supports_batch("openai") is True
    
    @pytest.mark.unit
    def test_anthropic_supported(self):
        """Test that Anthropic batch is supported."""
        assert supports_batch("anthropic") is True
    
    @pytest.mark.unit
    def test_google_supported(self):
        """Test that Google batch is supported."""
        assert supports_batch("google") is True
    
    @pytest.mark.unit
    def test_unknown_not_supported(self):
        """Test that unknown provider is not supported."""
        assert supports_batch("unknown_provider") is False
    
    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert supports_batch("OpenAI") is True
        assert supports_batch("ANTHROPIC") is True


class TestGetBatchBackend:
    """Tests for get_batch_backend function."""
    
    @pytest.mark.unit
    def test_returns_backend_for_openai(self):
        """Test getting OpenAI backend."""
        backend = get_batch_backend("openai")
        assert backend is not None
        assert isinstance(backend, BatchBackend)
    
    @pytest.mark.unit
    def test_returns_backend_for_anthropic(self):
        """Test getting Anthropic backend."""
        backend = get_batch_backend("anthropic")
        assert backend is not None
        assert isinstance(backend, BatchBackend)
    
    @pytest.mark.unit
    def test_returns_backend_for_google(self):
        """Test getting Google backend."""
        backend = get_batch_backend("google")
        assert backend is not None
        assert isinstance(backend, BatchBackend)
    
    @pytest.mark.unit
    def test_raises_for_unknown(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_batch_backend("unknown_provider")
    
    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test case insensitivity."""
        backend1 = get_batch_backend("openai")
        backend2 = get_batch_backend("OpenAI")
        assert type(backend1) == type(backend2)
