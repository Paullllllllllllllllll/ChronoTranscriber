"""Unit tests for LLM prompt caching support.

Tests cache token fields, cache_control annotations, config loading,
and session-level cache accumulation across all affected modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Step 1: Data model defaults
# ---------------------------------------------------------------------------


class TestTranscriptionResultCacheFields:
    """TranscriptionResult has cache fields with correct defaults."""

    @pytest.mark.unit
    def test_defaults_are_zero(self) -> None:
        from modules.llm.providers.base import TranscriptionResult

        result = TranscriptionResult(content="hello")
        assert result.cached_input_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_hit is False

    @pytest.mark.unit
    def test_explicit_values_stored(self) -> None:
        from modules.llm.providers.base import TranscriptionResult

        result = TranscriptionResult(
            content="hello",
            cached_input_tokens=500,
            cache_creation_tokens=1200,
            cache_hit=True,
        )
        assert result.cached_input_tokens == 500
        assert result.cache_creation_tokens == 1200
        assert result.cache_hit is True


class TestBatchResultItemCacheFields:
    """BatchResultItem has cache fields with correct defaults."""

    @pytest.mark.unit
    def test_defaults_are_zero(self) -> None:
        from modules.batch.backends.base import BatchResultItem

        item = BatchResultItem(custom_id="req-1")
        assert item.cached_input_tokens == 0
        assert item.cache_creation_tokens == 0

    @pytest.mark.unit
    def test_explicit_values_stored(self) -> None:
        from modules.batch.backends.base import BatchResultItem

        item = BatchResultItem(
            custom_id="req-1",
            cached_input_tokens=300,
            cache_creation_tokens=800,
        )
        assert item.cached_input_tokens == 300
        assert item.cache_creation_tokens == 800


# ---------------------------------------------------------------------------
# Step 2: Cache token extraction in _process_llm_response
# ---------------------------------------------------------------------------


class TestCacheTokenExtraction:
    """Cache tokens are extracted from mock API responses."""

    def _make_ai_message(
        self,
        content: str = "text",
        response_metadata: Optional[Dict[str, Any]] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        msg.response_metadata = response_metadata or {}
        msg.usage_metadata = usage_metadata
        return msg

    @staticmethod
    def _make_provider_mock() -> MagicMock:
        """Create a BaseProvider mock with real helper methods bound."""
        from modules.llm.providers.base import BaseProvider

        provider = MagicMock(spec=BaseProvider)
        provider._normalize_list_content = BaseProvider._normalize_list_content
        provider._extract_content = lambda resp: BaseProvider._extract_content(provider, resp)
        provider._extract_token_usage = BaseProvider._extract_token_usage
        provider._track_token_usage = lambda *a: BaseProvider._track_token_usage(provider, *a)
        return provider

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_anthropic_cache_tokens_from_response_metadata(self) -> None:
        """Anthropic cache tokens are extracted from usage dict."""
        from modules.llm.providers.base import (
            ANTHROPIC_TOKEN_MAPPING,
            BaseProvider,
        )

        msg = self._make_ai_message(
            response_metadata={
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 0,
                }
            }
        )

        # Use a concrete subclass to call _process_llm_response
        provider = self._make_provider_mock()
        result = await BaseProvider._process_llm_response(provider, msg, ANTHROPIC_TOKEN_MAPPING)

        assert result.cached_input_tokens == 800
        assert result.cache_creation_tokens == 0
        assert result.cache_hit is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_anthropic_cache_creation_tokens(self) -> None:
        """Anthropic cache_creation_input_tokens are extracted."""
        from modules.llm.providers.base import (
            ANTHROPIC_TOKEN_MAPPING,
            BaseProvider,
        )

        msg = self._make_ai_message(
            response_metadata={
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 300,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 1200,
                }
            }
        )

        provider = self._make_provider_mock()
        result = await BaseProvider._process_llm_response(provider, msg, ANTHROPIC_TOKEN_MAPPING)

        assert result.cache_creation_tokens == 1200
        assert result.cache_hit is True  # creation counts as a cache event

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_cache_tokens_from_prompt_details(self) -> None:
        """OpenAI cached_tokens are extracted from prompt_tokens_details."""
        from modules.llm.providers.base import (
            OPENAI_TOKEN_MAPPING,
            BaseProvider,
        )

        msg = self._make_ai_message(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 2000,
                    "completion_tokens": 500,
                    "total_tokens": 2500,
                    "prompt_tokens_details": {"cached_tokens": 1500},
                }
            }
        )

        provider = self._make_provider_mock()
        result = await BaseProvider._process_llm_response(provider, msg, OPENAI_TOKEN_MAPPING)

        assert result.cached_input_tokens == 1500
        assert result.cache_hit is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_cache_tokens_when_absent(self) -> None:
        """Cache fields are zero when API returns no cache info."""
        from modules.llm.providers.base import (
            OPENAI_TOKEN_MAPPING,
            BaseProvider,
        )

        msg = self._make_ai_message(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                }
            }
        )

        provider = self._make_provider_mock()
        result = await BaseProvider._process_llm_response(provider, msg, OPENAI_TOKEN_MAPPING)

        assert result.cached_input_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_hit is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_tokens_from_usage_metadata_fallback(self) -> None:
        """Cache tokens extracted from usage_metadata.input_token_details."""
        from modules.llm.providers.base import (
            OPENAI_TOKEN_MAPPING,
            BaseProvider,
        )

        msg = self._make_ai_message(
            response_metadata={},
            usage_metadata={
                "input_tokens": 2000,
                "output_tokens": 400,
                "total_tokens": 2400,
                "input_token_details": {
                    "cache_read": 1800,
                    "cache_creation": 0,
                },
            },
        )

        provider = self._make_provider_mock()
        result = await BaseProvider._process_llm_response(provider, msg, OPENAI_TOKEN_MAPPING)

        assert result.cached_input_tokens == 1800
        assert result.cache_hit is True


# ---------------------------------------------------------------------------
# Step 3: Configuration
# ---------------------------------------------------------------------------


class TestPromptCachingConfig:
    """Config service returns correct prompt caching config."""

    @pytest.mark.unit
    def test_returns_config_when_present(self) -> None:
        from modules.config.service import ConfigService

        mock_svc = MagicMock(spec=ConfigService)
        mock_svc.get_concurrency_config.return_value = {
            "prompt_caching": {
                "enabled": True,
                "anthropic": {"ttl": "5m"},
            }
        }
        # Call the real method with mock self
        result = ConfigService.get_prompt_caching_config(mock_svc)
        assert result["enabled"] is True
        assert result["anthropic"]["ttl"] == "5m"

    @pytest.mark.unit
    def test_returns_default_when_absent(self) -> None:
        from modules.config.service import ConfigService

        mock_svc = MagicMock(spec=ConfigService)
        mock_svc.get_concurrency_config.return_value = {}
        result = ConfigService.get_prompt_caching_config(mock_svc)
        assert result == {"enabled": False}

    @pytest.mark.unit
    def test_convenience_function(self) -> None:
        with patch("modules.config.service.get_config_service") as mock_cs:
            mock_cs.return_value.get_prompt_caching_config.return_value = {"enabled": True}
            from modules.config.service import get_prompt_caching_config

            result = get_prompt_caching_config()
            assert result["enabled"] is True


# ---------------------------------------------------------------------------
# Step 4: Anthropic provider — cache_control on SystemMessage
# ---------------------------------------------------------------------------


class TestAnthropicProviderCacheControl:
    """Anthropic provider adds cache_control to SystemMessage when enabled."""

    def _make_provider(self, caching_cfg: Dict[str, Any]) -> Any:
        """Create an AnthropicProvider with mocked dependencies."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic"):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries", return_value=3):
                with patch("modules.llm.providers.base.get_config_service") as mock_cs:
                    mock_cs.return_value.get_prompt_caching_config.return_value = caching_cfg
                    provider = AnthropicProvider(
                        api_key="sk-ant-test",
                        model="claude-3-5-sonnet-20241022",
                    )
        return provider

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_control_added_when_enabled(self) -> None:
        """SystemMessage uses content-block form with cache_control when caching enabled."""
        provider = self._make_provider({"enabled": True, "anthropic": {"ttl": "5m"}})

        captured_messages = []

        async def fake_invoke(llm, messages, **kw):
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {"usage": {"input_tokens": 100, "output_tokens": 50}}
            msg.usage_metadata = None
            return msg

        provider._ainvoke_with_retry = fake_invoke

        await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe this.",
        )

        system_msg = captured_messages[0]
        assert isinstance(system_msg.content, list)
        block = system_msg.content[0]
        assert block["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_control_1h_ttl(self) -> None:
        """cache_control includes ttl='1h' when configured."""
        provider = self._make_provider({"enabled": True, "anthropic": {"ttl": "1h"}})

        captured_messages = []

        async def fake_invoke(llm, messages, **kw):
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {"usage": {"input_tokens": 100, "output_tokens": 50}}
            msg.usage_metadata = None
            return msg

        provider._ainvoke_with_retry = fake_invoke

        await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe this.",
        )

        block = captured_messages[0].content[0]
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_cache_control_when_disabled(self) -> None:
        """SystemMessage uses plain string content when caching disabled."""
        provider = self._make_provider({"enabled": False})

        captured_messages = []

        async def fake_invoke(llm, messages, **kw):
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {"usage": {"input_tokens": 100, "output_tokens": 50}}
            msg.usage_metadata = None
            return msg

        provider._ainvoke_with_retry = fake_invoke

        await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe this.",
        )

        system_msg = captured_messages[0]
        assert isinstance(system_msg.content, str)


# ---------------------------------------------------------------------------
# Step 5: OpenAI provider — prompt_cache_retention
# ---------------------------------------------------------------------------


class TestOpenAIProviderCacheRetention:
    """OpenAI provider passes prompt_cache_retention when configured."""

    @pytest.mark.unit
    def test_retention_passed_to_chat_openai(self) -> None:
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch(
            "modules.llm.providers.openai_provider.ChatOpenAI",
            side_effect=lambda **kw: captured.update(kw) or MagicMock(),
        ):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                with patch("modules.llm.providers.base.get_config_service") as mock_cs:
                    mock_cs.return_value.get_prompt_caching_config.return_value = {
                        "enabled": True,
                        "openai": {"prompt_cache_retention": "24h"},
                    }
                    OpenAIProvider(
                        api_key="sk-test",
                        model="gpt-4o",
                    )

        assert captured.get("model_kwargs", {}).get("prompt_cache_retention") == "24h"

    @pytest.mark.unit
    def test_no_retention_when_null(self) -> None:
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch(
            "modules.llm.providers.openai_provider.ChatOpenAI",
            side_effect=lambda **kw: captured.update(kw) or MagicMock(),
        ):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                with patch("modules.llm.providers.base.get_config_service") as mock_cs:
                    mock_cs.return_value.get_prompt_caching_config.return_value = {
                        "enabled": True,
                        "openai": {"prompt_cache_retention": None},
                    }
                    OpenAIProvider(
                        api_key="sk-test",
                        model="gpt-4o",
                    )

        assert "prompt_cache_retention" not in captured

    @pytest.mark.unit
    def test_no_retention_when_disabled(self) -> None:
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch(
            "modules.llm.providers.openai_provider.ChatOpenAI",
            side_effect=lambda **kw: captured.update(kw) or MagicMock(),
        ):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                with patch("modules.llm.providers.base.get_config_service") as mock_cs:
                    mock_cs.return_value.get_prompt_caching_config.return_value = {
                        "enabled": False,
                    }
                    OpenAIProvider(
                        api_key="sk-test",
                        model="gpt-4o",
                    )

        assert "prompt_cache_retention" not in captured


# ---------------------------------------------------------------------------
# Step 6: OpenRouter provider — Anthropic pass-through
# ---------------------------------------------------------------------------


class TestOpenRouterProviderCacheControl:
    """OpenRouter provider adds cache_control for Anthropic models."""

    def _make_provider(
        self,
        model: str,
        caching_cfg: Dict[str, Any],
    ) -> Any:
        from modules.llm.providers.openrouter_provider import OpenRouterProvider

        with patch(
            "modules.llm.providers.openrouter_provider.ChatOpenAI",
            return_value=MagicMock(),
        ):
            with patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3):
                with patch("modules.llm.providers.base.get_config_service") as mock_cs:
                    mock_cs.return_value.get_prompt_caching_config.return_value = caching_cfg
                    provider = OpenRouterProvider(
                        api_key="sk-or-test",
                        model=model,
                    )
        return provider

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_control_for_claude_model(self) -> None:
        """cache_control added for Claude model when enabled."""
        provider = self._make_provider(
            "anthropic/claude-3-5-sonnet",
            {"enabled": True, "openrouter": {"anthropic_cache_control": True}},
        )

        captured_messages = []

        async def fake_invoke(llm, messages, **kw):
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
            msg.usage_metadata = None
            return msg

        provider._ainvoke_with_retry = fake_invoke

        await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe.",
        )

        system_msg = captured_messages[0]
        assert isinstance(system_msg.content, list)
        assert system_msg.content[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_cache_control_for_non_claude_model(self) -> None:
        """cache_control NOT added for non-Claude models via OpenRouter."""
        provider = self._make_provider(
            "openai/gpt-4o",
            {"enabled": True, "openrouter": {"anthropic_cache_control": True}},
        )

        captured_messages = []

        async def fake_invoke(llm, messages, **kw):
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
            msg.usage_metadata = None
            return msg

        provider._ainvoke_with_retry = fake_invoke

        await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe.",
        )

        system_msg = captured_messages[0]
        assert isinstance(system_msg.content, str)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_cache_control_when_disabled(self) -> None:
        """cache_control NOT added when caching is disabled."""
        provider = self._make_provider(
            "anthropic/claude-3-5-sonnet",
            {"enabled": False},
        )

        captured_messages = []

        async def fake_invoke(llm, messages, **kw):
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
            msg.usage_metadata = None
            return msg

        provider._ainvoke_with_retry = fake_invoke

        await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe.",
        )

        system_msg = captured_messages[0]
        assert isinstance(system_msg.content, str)


# ---------------------------------------------------------------------------
# Step 7: Transcriber — _result_to_dict includes cache fields
# ---------------------------------------------------------------------------


class TestTranscriberResultToDict:
    """LangChainTranscriber._result_to_dict includes cache token fields."""

    @pytest.mark.unit
    def test_cache_tokens_included_in_usage(self) -> None:
        from modules.llm.providers.base import TranscriptionResult

        result = TranscriptionResult(
            content="hello",
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            cached_input_tokens=800,
            cache_creation_tokens=0,
            cache_hit=True,
        )

        # Create a minimal transcriber-like object to call _result_to_dict
        transcriber = MagicMock()
        transcriber._total_request_count = 0
        transcriber._total_input_tokens = 0
        transcriber._total_cached_tokens = 0
        transcriber._cache_request_count = 0

        from modules.llm.transcriber import LangChainTranscriber

        d = LangChainTranscriber._result_to_dict(transcriber, result)

        assert d["usage"]["cached_input_tokens"] == 800
        assert "cache_creation_tokens" not in d["usage"]  # 0 is excluded

    @pytest.mark.unit
    def test_no_cache_fields_when_zero(self) -> None:
        from modules.llm.providers.base import TranscriptionResult

        result = TranscriptionResult(
            content="hello",
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
        )

        transcriber = MagicMock()
        transcriber._total_request_count = 0
        transcriber._total_input_tokens = 0
        transcriber._total_cached_tokens = 0
        transcriber._cache_request_count = 0

        from modules.llm.transcriber import LangChainTranscriber

        d = LangChainTranscriber._result_to_dict(transcriber, result)

        assert "cached_input_tokens" not in d["usage"]
        assert "cache_creation_tokens" not in d["usage"]


# ---------------------------------------------------------------------------
# Step 7b: Transcriber — session accumulators
# ---------------------------------------------------------------------------


class TestTranscriberSessionAccumulators:
    """Session-level cache accumulators are updated correctly."""

    @pytest.mark.unit
    def test_accumulators_increment(self) -> None:
        from modules.llm.providers.base import TranscriptionResult
        from modules.llm.transcriber import LangChainTranscriber

        transcriber = MagicMock()
        transcriber._total_request_count = 0
        transcriber._total_input_tokens = 0
        transcriber._total_cached_tokens = 0
        transcriber._cache_request_count = 0

        # First result — cache creation (write)
        r1 = TranscriptionResult(
            content="a",
            input_tokens=1000,
            output_tokens=100,
            total_tokens=1100,
            cache_creation_tokens=1000,
            cache_hit=True,
        )
        LangChainTranscriber._result_to_dict(transcriber, r1)

        # Second result — cache read (hit)
        r2 = TranscriptionResult(
            content="b",
            input_tokens=1000,
            output_tokens=150,
            total_tokens=1150,
            cached_input_tokens=900,
            cache_hit=True,
        )
        LangChainTranscriber._result_to_dict(transcriber, r2)

        # Third result — no cache
        r3 = TranscriptionResult(
            content="c",
            input_tokens=500,
            output_tokens=80,
            total_tokens=580,
        )
        LangChainTranscriber._result_to_dict(transcriber, r3)

        assert transcriber._total_request_count == 3
        assert transcriber._cache_request_count == 2
        assert transcriber._total_input_tokens == 2500
        assert transcriber._total_cached_tokens == 900


# ---------------------------------------------------------------------------
# Anthropic batch backend — cache_control on system param
# ---------------------------------------------------------------------------


class TestAnthropicBatchBackendCacheControl:
    """Anthropic batch backend adds cache_control to system parameter."""

    @pytest.mark.unit
    def test_system_param_has_cache_control_when_enabled(self) -> None:
        from modules.batch.backends.anthropic_backend import AnthropicBatchBackend
        from modules.batch.backends.base import BatchRequest

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_123"
        mock_client.messages.batches.create.return_value = mock_batch_response

        backend = AnthropicBatchBackend()
        backend._client = mock_client

        req = BatchRequest(
            custom_id="req-1",
            image_base64="abc",
            mime_type="image/png",
        )

        caching_cfg = {"enabled": True, "anthropic": {"ttl": "5m"}}
        with patch("modules.batch.backends.anthropic_backend.get_config_service") as mock_cs:
            mock_cs.return_value.get_prompt_caching_config.return_value = caching_cfg
            backend.submit_batch(
                [req],
                {"name": "claude-3-5-sonnet-20241022", "max_tokens": 4096},
                system_prompt="Transcribe.",
            )

        call_kwargs = mock_client.messages.batches.create.call_args
        batch_requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        system_param = batch_requests[0]["params"]["system"]

        # Should be a list with cache_control
        assert isinstance(system_param, list)
        assert system_param[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.unit
    def test_system_param_is_string_when_disabled(self) -> None:
        from modules.batch.backends.anthropic_backend import AnthropicBatchBackend
        from modules.batch.backends.base import BatchRequest

        mock_client = MagicMock()
        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch_456"
        mock_client.messages.batches.create.return_value = mock_batch_response

        backend = AnthropicBatchBackend()
        backend._client = mock_client

        req = BatchRequest(
            custom_id="req-1",
            image_base64="abc",
            mime_type="image/png",
        )

        with patch("modules.batch.backends.anthropic_backend.get_config_service") as mock_cs:
            mock_cs.return_value.get_prompt_caching_config.return_value = {"enabled": False}
            backend.submit_batch(
                [req],
                {"name": "claude-3-5-sonnet-20241022", "max_tokens": 4096},
                system_prompt="Transcribe.",
            )

        call_kwargs = mock_client.messages.batches.create.call_args
        batch_requests = call_kwargs.kwargs.get("requests") or call_kwargs[1].get("requests")
        system_param = batch_requests[0]["params"]["system"]

        assert isinstance(system_param, str)


# ---------------------------------------------------------------------------
# OpenAI batch backend — cached token extraction
# ---------------------------------------------------------------------------


class TestOpenAIBatchBackendCacheExtraction:
    """OpenAI batch backend extracts cached_tokens from results."""

    @pytest.mark.unit
    def test_cached_tokens_extracted_from_prompt_details(self) -> None:
        from modules.batch.backends.openai_backend import OpenAIBatchBackend

        # Build a mock batch result line
        result_line = json.dumps({
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "hello"},
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 2000,
                        "output_tokens": 300,
                        "prompt_tokens_details": {"cached_tokens": 1500},
                    },
                },
            },
        })

        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.output_file_id = "file-abc"
        mock_client.batches.retrieve.return_value = mock_batch
        mock_client.files.content.return_value.read.return_value = result_line.encode()

        backend = OpenAIBatchBackend()
        backend._client = mock_client

        from modules.batch.backends.base import BatchHandle

        handle = BatchHandle(provider="openai", batch_id="batch_test")
        items = list(backend.download_results(handle))

        assert len(items) == 1
        assert items[0].cached_input_tokens == 1500


# ---------------------------------------------------------------------------
# Batching module — prompt_cache_retention in Responses API body
# ---------------------------------------------------------------------------


class TestBatchingPromptCacheRetention:
    """_build_responses_body_for_image includes prompt_cache_retention."""

    @pytest.mark.unit
    def test_retention_added_when_configured(self) -> None:
        from modules.batch.requests import _build_responses_body_for_image

        with patch("modules.batch.requests.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            mock_cs.return_value.get_prompt_caching_config.return_value = {
                "enabled": True,
                "openai": {"prompt_cache_retention": "24h"},
            }
            body = _build_responses_body_for_image(
                model_config={"name": "gpt-4o", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
                transcription_schema={"type": "object", "properties": {}},
            )

        assert body.get("prompt_cache_retention") == "24h"

    @pytest.mark.unit
    def test_retention_not_added_when_null(self) -> None:
        from modules.batch.requests import _build_responses_body_for_image

        with patch("modules.batch.requests.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            mock_cs.return_value.get_prompt_caching_config.return_value = {
                "enabled": True,
                "openai": {"prompt_cache_retention": None},
            }
            body = _build_responses_body_for_image(
                model_config={"name": "gpt-4o", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
                transcription_schema={"type": "object", "properties": {}},
            )

        assert "prompt_cache_retention" not in body

    @pytest.mark.unit
    def test_retention_not_added_when_disabled(self) -> None:
        from modules.batch.requests import _build_responses_body_for_image

        with patch("modules.batch.requests.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            mock_cs.return_value.get_prompt_caching_config.return_value = {
                "enabled": False,
            }
            body = _build_responses_body_for_image(
                model_config={"name": "gpt-4o", "max_output_tokens": 4096},
                system_prompt="prompt",
                image_url="data:image/png;base64,abc",
                transcription_schema={"type": "object", "properties": {}},
            )

        assert "prompt_cache_retention" not in body
