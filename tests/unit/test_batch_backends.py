"""Unit tests for modules/llm/batch/backends/.

Tests batch processing backend abstractions and factory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from modules.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
)
from modules.batch.backends.factory import (
    get_batch_backend,
    supports_batch,
)


class TestBatchHandle:
    """Tests for BatchHandle dataclass."""

    @pytest.mark.unit
    def test_creation(self) -> None:
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
    def test_minimal_creation(self) -> None:
        """Test BatchHandle with minimal args."""
        handle = BatchHandle(batch_id="batch_456", provider="anthropic")
        assert handle.batch_id == "batch_456"
        # Default metadata is empty dict, not None
        assert handle.metadata == {} or handle.metadata is None


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    @pytest.mark.unit
    def test_all_statuses_exist(self) -> None:
        """Test that all expected statuses exist."""
        expected = [
            "PENDING",
            "IN_PROGRESS",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "EXPIRED",
        ]
        for status_name in expected:
            assert hasattr(BatchStatus, status_name)

    @pytest.mark.unit
    def test_status_values(self) -> None:
        """Test status enum values."""
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"


class TestBatchStatusInfo:
    """Tests for BatchStatusInfo dataclass."""

    @pytest.mark.unit
    def test_creation(self) -> None:
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
    def test_minimal_creation(self) -> None:
        """Test BatchStatusInfo with minimal args."""
        info = BatchStatusInfo(status=BatchStatus.PENDING)
        assert info.status == BatchStatus.PENDING
        assert info.total_requests == 0


class TestBatchResultItem:
    """Tests for BatchResultItem dataclass."""

    @pytest.mark.unit
    def test_creation(self) -> None:
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
    def test_failed_item(self) -> None:
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
    def test_creation(self, temp_dir: Path) -> None:
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
    def test_openai_supported(self) -> None:
        """Test that OpenAI batch is supported."""
        assert supports_batch("openai") is True

    @pytest.mark.unit
    def test_anthropic_supported(self) -> None:
        """Test that Anthropic batch is supported."""
        assert supports_batch("anthropic") is True

    @pytest.mark.unit
    def test_google_supported(self) -> None:
        """Test that Google batch is supported."""
        assert supports_batch("google") is True

    @pytest.mark.unit
    def test_unknown_not_supported(self) -> None:
        """Test that unknown provider is not supported."""
        assert supports_batch("unknown_provider") is False

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert supports_batch("OpenAI") is True
        assert supports_batch("ANTHROPIC") is True


class TestGetBatchBackend:
    """Tests for get_batch_backend function."""

    @pytest.mark.unit
    def test_returns_backend_for_openai(self) -> None:
        """Test getting OpenAI backend."""
        backend = get_batch_backend("openai")
        assert backend is not None
        assert isinstance(backend, BatchBackend)

    @pytest.mark.unit
    def test_returns_backend_for_anthropic(self) -> None:
        """Test getting Anthropic backend."""
        backend = get_batch_backend("anthropic")
        assert backend is not None
        assert isinstance(backend, BatchBackend)

    @pytest.mark.unit
    def test_returns_backend_for_google(self) -> None:
        """Test getting Google backend."""
        backend = get_batch_backend("google")
        assert backend is not None
        assert isinstance(backend, BatchBackend)

    @pytest.mark.unit
    def test_raises_for_unknown(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_batch_backend("unknown_provider")

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        backend1 = get_batch_backend("openai")
        backend2 = get_batch_backend("OpenAI")
        assert type(backend1) is type(backend2)


# =============================================================================
# CT-3: text.verbosity forwarding in OpenAI batch backend (fix for bug where
#        text.verbosity was never included in batch request bodies)
# =============================================================================


class TestOpenAIBuildResponsesBodyTextVerbosity:
    """Verify _build_responses_body includes text.verbosity for reasoning models.

    CT-3 fix: LangChainTranscriber now extracts text_cfg from model config and
    _build_responses_body reads it to add verbosity to the Responses API body.
    """

    def _model_cfg(self, model_name: str, verbosity: str) -> dict:
        return {
            "name": model_name,
            "max_output_tokens": 4096,
            "text": {"verbosity": verbosity},
        }

    @pytest.mark.unit
    def test_verbosity_included_for_gpt5_model(self) -> None:
        """text.verbosity is added to the body for GPT-5 (reasoning) models."""
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config=self._model_cfg("gpt-5-mini", "concise"),
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
            )

        assert "text" in body
        assert body["text"].get("verbosity") == "concise"

    @pytest.mark.unit
    def test_verbosity_not_included_for_non_reasoning_model(self) -> None:
        """text.verbosity is NOT added for non-reasoning models (e.g. gpt-4o)."""
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config=self._model_cfg("gpt-4o", "verbose"),
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
            )

        assert "verbosity" not in body.get("text", {})

    @pytest.mark.unit
    def test_verbosity_skipped_when_text_config_absent(self) -> None:
        """No text.verbosity in body when model_config has no text key."""
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={"name": "gpt-5-mini", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
            )

        assert "verbosity" not in body.get("text", {})

    @pytest.mark.unit
    def test_verbosity_coexists_with_structured_output_format(self) -> None:
        """text.verbosity and text.format can coexist in the request body."""
        from modules.batch.backends.openai_backend import _build_responses_body

        schema = {
            "type": "object",
            "properties": {"transcribed_text": {"type": "string"}},
            "required": ["transcribed_text"],
            "additionalProperties": False,
        }

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config=self._model_cfg("gpt-5-mini", "medium"),
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
                transcription_schema=schema,
            )

        assert "text" in body
        assert body["text"].get("verbosity") == "medium"


class TestOpenAIBuildResponsesBodyContextImage:
    """Verify _build_responses_body handles context_image_url."""

    @pytest.mark.unit
    def test_context_image_prepended(self) -> None:
        """Context image blocks appear before the page image."""
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={"name": "gpt-4o", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
                context_image_url="data:image/jpeg;base64,CTX",
            )

        user_msg = body["input"][1]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        # 4 blocks: ctx label, ctx image, page label, page image
        assert len(content) == 4
        assert content[0] == {"type": "input_text", "text": "Context image:"}
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == "data:image/jpeg;base64,CTX"
        assert content[2] == {"type": "input_text", "text": "The image:"}
        assert content[3]["type"] == "input_image"
        assert content[3]["image_url"] == "data:image/png;base64,PAGE"

    @pytest.mark.unit
    def test_no_context_image_two_blocks(self) -> None:
        """Without context image, only page label + page image present."""
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={"name": "gpt-4o", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
            )

        user_msg = body["input"][1]
        content = user_msg["content"]
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "The image:"}
        assert content[1]["type"] == "input_image"

    @pytest.mark.unit
    def test_context_image_uses_same_detail(self) -> None:
        """Context image gets the same detail kwarg as the page image."""
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={"name": "gpt-4o", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
                context_image_url="data:image/jpeg;base64,CTX",
                llm_detail="high",
            )

        user_msg = body["input"][1]
        content = user_msg["content"]
        ctx_img = content[1]
        page_img = content[3]
        assert ctx_img.get("detail") == "high"
        assert page_img.get("detail") == "high"

    @pytest.mark.unit
    def test_empty_user_instruction_omits_text_block(self) -> None:
        """Empty user_instruction produces image-only content."""
        from modules.batch.backends.openai_backend import (
            _build_responses_body,
        )

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                    "user_instruction": "",
                },
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
            )

        content = body["input"][1]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "input_image"

    @pytest.mark.unit
    def test_empty_context_image_instruction_omits_label(
        self,
    ) -> None:
        """Empty context_image_instruction omits label but keeps image."""
        from modules.batch.backends.openai_backend import (
            _build_responses_body,
        )

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                    "context_image_instruction": "",
                },
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
                context_image_url="data:image/jpeg;base64,CTX",
            )

        content = body["input"][1]["content"]
        # 3 blocks: ctx image (no label), page label, page image
        assert len(content) == 3
        assert content[0]["type"] == "input_image"
        assert content[1] == {
            "type": "input_text",
            "text": "The image:",
        }

    @pytest.mark.unit
    def test_custom_user_instruction_string(self) -> None:
        """Custom user_instruction string appears in content."""
        from modules.batch.backends.openai_backend import (
            _build_responses_body,
        )

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            body = _build_responses_body(
                model_config={
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                    "user_instruction": "OCR this page.",
                },
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
            )

        content = body["input"][1]["content"]
        assert len(content) == 2
        assert content[0] == {
            "type": "input_text",
            "text": "OCR this page.",
        }


class TestOpenAIBuildResponsesBodyOriginalDetail:
    """Regression: batch bodies must honor llm_detail='original' where supported.

    The backend previously accepted only 'low'/'high' and silently dropped
    'original' — the shipped v2.0.0 default — so batch submissions fell back
    to the API's default detail while the sync path sent full resolution.
    """

    def _build(self, model_name: str, llm_detail: str) -> dict:
        from modules.batch.backends.openai_backend import _build_responses_body

        with patch(
            "modules.batch.backends.openai_backend.get_config_service"
        ) as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            return _build_responses_body(
                model_config={"name": model_name, "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,PAGE",
                llm_detail=llm_detail,
            )

    @pytest.mark.unit
    def test_original_detail_included_for_capable_model(self) -> None:
        """detail='original' reaches the body for models that support it."""
        body = self._build("gpt-5.6-luna", "original")
        page_img = body["input"][1]["content"][-1]
        assert page_img["type"] == "input_image"
        assert page_img.get("detail") == "original"

    @pytest.mark.unit
    def test_original_detail_dropped_for_incapable_model(self) -> None:
        """detail='original' is omitted for models without original support."""
        body = self._build("gpt-4o", "original")
        page_img = body["input"][1]["content"][-1]
        assert page_img["type"] == "input_image"
        assert "detail" not in page_img

    @pytest.mark.unit
    def test_high_detail_still_included(self) -> None:
        """Existing 'high' behavior is unchanged."""
        body = self._build("gpt-4o", "high")
        page_img = body["input"][1]["content"][-1]
        assert page_img.get("detail") == "high"
