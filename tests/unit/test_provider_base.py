"""Unit tests for modules/llm/providers/base.py.

Tests base provider abstraction and common utilities.
"""

from __future__ import annotations

import base64
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.llm.providers.base import (
    ProviderCapabilities,
    TranscriptionResult,
    BaseProvider,
)
from modules.llm.model_capabilities import Capabilities


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""
    
    @pytest.mark.unit
    def test_default_values(self):
        """Test default capability values."""
        caps = Capabilities(
            model="test-model",
            family="test",
        )
        
        assert caps.supports_image_input is False
        assert caps.supports_structured_outputs is True
        assert caps.is_reasoning_model is False
        assert caps.supports_sampler_controls is True
    
    @pytest.mark.unit
    def test_vision_capabilities(self):
        """Test vision-related capabilities."""
        caps = Capabilities(
            model="gpt-4o",
            family="gpt-4o",
            provider="openai",
            supports_image_input=True,
            supports_image_detail=True,
            default_image_detail="high",
        )
        
        assert caps.supports_image_input is True
        assert caps.supports_image_detail is True
        assert caps.default_image_detail == "high"
    
    @pytest.mark.unit
    def test_reasoning_model_capabilities(self):
        """Test reasoning model capabilities."""
        caps = Capabilities(
            model="o1",
            family="o1",
            provider="openai",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_sampler_controls=False,
            supports_top_p=False,
        )
        
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_sampler_controls is False
    
    @pytest.mark.unit
    def test_frozen_dataclass(self):
        """Test that capabilities are immutable."""
        caps = Capabilities(
            model="test-model",
            family="test",
        )
        
        with pytest.raises(AttributeError):
            caps.model = "modified"
    
    @pytest.mark.unit
    def test_token_limits(self):
        """Test token limit attributes."""
        caps = Capabilities(
            model="test-model",
            family="test",
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
        
        assert caps.max_context_tokens == 200000
        assert caps.max_output_tokens == 8192


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""
    
    @pytest.mark.unit
    def test_basic_result(self):
        """Test basic transcription result."""
        result = TranscriptionResult(content="Transcribed text here")
        
        assert result.content == "Transcribed text here"
        assert result.error is None
        assert result.no_transcribable_text is False
        assert result.transcription_not_possible is False
    
    @pytest.mark.unit
    def test_json_content_parsing(self):
        """Test automatic parsing of JSON content."""
        json_content = json.dumps({
            "transcription": "Parsed text",
            "no_transcribable_text": True,
            "transcription_not_possible": False,
        })
        
        result = TranscriptionResult(content=json_content)
        
        assert result.parsed_output is not None
        assert result.no_transcribable_text is True
        assert result.transcription_not_possible is False
    
    @pytest.mark.unit
    def test_non_json_content(self):
        """Test handling of non-JSON content."""
        result = TranscriptionResult(content="Plain text transcription")
        
        assert result.parsed_output is None
        assert result.no_transcribable_text is False
    
    @pytest.mark.unit
    def test_token_usage(self):
        """Test token usage tracking."""
        result = TranscriptionResult(
            content="Text",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
    
    @pytest.mark.unit
    def test_error_result(self):
        """Test error in result."""
        result = TranscriptionResult(
            content="",
            error="API rate limit exceeded",
        )
        
        assert result.error == "API rate limit exceeded"
    
    @pytest.mark.unit
    def test_raw_response_stored(self):
        """Test raw response storage."""
        raw = {"id": "resp_123", "model": "gpt-4o"}
        result = TranscriptionResult(content="Text", raw_response=raw)
        
        assert result.raw_response == raw
    
    @pytest.mark.unit
    def test_transcription_not_possible_flag(self):
        """Test transcription_not_possible flag parsing."""
        json_content = json.dumps({
            "transcription": "",
            "no_transcribable_text": False,
            "transcription_not_possible": True,
        })
        
        result = TranscriptionResult(content=json_content)
        
        assert result.transcription_not_possible is True
    
    @pytest.mark.unit
    def test_malformed_json_handled(self):
        """Test that malformed JSON doesn't crash."""
        result = TranscriptionResult(content="{invalid json")
        
        assert result.parsed_output is None
        assert result.no_transcribable_text is False
    
    @pytest.mark.unit
    def test_json_with_extra_whitespace(self):
        """Test JSON parsing with leading/trailing whitespace."""
        json_content = "  \n" + json.dumps({
            "transcription": "Text",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }) + "  \n"
        
        result = TranscriptionResult(content=json_content)
        
        assert result.parsed_output is not None


class TestBaseProviderStatics:
    """Tests for BaseProvider static methods."""
    
    @pytest.mark.unit
    def test_encode_image_to_base64(self, temp_dir):
        """Test image encoding to base64."""
        # Create a simple test image file
        image_path = temp_dir / "test.png"
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        image_path.write_bytes(image_data)
        
        b64_data, mime_type = BaseProvider.encode_image_to_base64(image_path)
        
        assert mime_type == "image/png"
        assert b64_data == base64.b64encode(image_data).decode("utf-8")
    
    @pytest.mark.unit
    def test_encode_jpeg_image(self, temp_dir):
        """Test JPEG image encoding."""
        image_path = temp_dir / "test.jpg"
        image_data = b"\xff\xd8\xff" + b"\x00" * 100
        image_path.write_bytes(image_data)
        
        b64_data, mime_type = BaseProvider.encode_image_to_base64(image_path)
        
        assert mime_type == "image/jpeg"
    
    @pytest.mark.unit
    def test_unsupported_format_raises(self, temp_dir):
        """Test that unsupported format raises ValueError."""
        image_path = temp_dir / "test.xyz"
        image_path.write_bytes(b"data")
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            BaseProvider.encode_image_to_base64(image_path)
    
    @pytest.mark.unit
    def test_create_data_url(self):
        """Test data URL creation."""
        result = BaseProvider.create_data_url("abc123", "image/png")
        
        assert result == "data:image/png;base64,abc123"
    
    @pytest.mark.unit
    def test_create_data_url_jpeg(self):
        """Test data URL for JPEG."""
        result = BaseProvider.create_data_url("xyz789", "image/jpeg")
        
        assert result == "data:image/jpeg;base64,xyz789"


class TestBaseProviderInit:
    """Tests for BaseProvider initialization."""
    
    @pytest.mark.unit
    def test_cannot_instantiate_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseProvider(api_key="key", model="model")
    
    @pytest.mark.unit
    def test_concrete_implementation_attributes(self):
        """Test that concrete implementation has correct attributes."""
        # Create a minimal concrete implementation for testing
        class ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"
            
            def get_capabilities(self):
                return Capabilities(
                    model=self.model,
                    family="test",
                )
            
            async def transcribe_image(self, *args, **kwargs):
                pass
            
            async def transcribe_image_from_base64(self, *args, **kwargs):
                pass
            
            async def close(self):
                pass
        
        provider = ConcreteProvider(
            api_key="test-key",
            model="test-model",
            temperature=0.5,
            max_tokens=2048,
            timeout=30.0,
        )
        
        assert provider.api_key == "test-key"
        assert provider.model == "test-model"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048
        assert provider.timeout == 30.0


class TestProviderCapabilitiesEdgeCases:
    """Edge case tests for ProviderCapabilities."""
    
    @pytest.mark.unit
    def test_google_media_resolution(self):
        """Test Google-specific media resolution settings."""
        caps = Capabilities(
            model="gemini-pro",
            family="gemini",
            provider="google",
            supports_media_resolution=True,
            default_media_resolution="high",
        )
        
        assert caps.supports_media_resolution is True
        assert caps.default_media_resolution == "high"
    
    @pytest.mark.unit
    def test_structured_output_capability(self):
        """Test structured output capability."""
        caps = Capabilities(
            model="gpt-4o",
            family="gpt-4o",
            provider="openai",
            supports_structured_outputs=True,
            supports_json_mode=True,
        )
        
        assert caps.supports_structured_outputs is True
        assert caps.supports_json_mode is True


class TestLoadMaxRetries:
    """Tests for load_max_retries() config loader."""

    @pytest.mark.unit
    def test_reads_from_concurrency_config(self):
        """load_max_retries reads attempts from concurrency.transcription.retry."""
        from modules.llm.providers.base import load_max_retries

        cfg = {"concurrency": {"transcription": {"retry": {"attempts": 7}}}}
        with patch("modules.llm.providers.base.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = cfg
            result = load_max_retries()

        assert result == 7

    @pytest.mark.unit
    def test_defaults_to_5_when_missing(self):
        """load_max_retries returns 5 when config key is absent."""
        from modules.llm.providers.base import load_max_retries

        with patch("modules.llm.providers.base.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = {}
            result = load_max_retries()

        assert result == 5

    @pytest.mark.unit
    def test_returns_5_on_exception(self):
        """load_max_retries returns 5 when config service raises an exception."""
        from modules.llm.providers.base import load_max_retries

        with patch("modules.llm.providers.base.get_config_service",
                   side_effect=RuntimeError("config unavailable")):
            result = load_max_retries()

        assert result == 5

    @pytest.mark.unit
    def test_minimum_is_1(self):
        """load_max_retries returns at least 1 even when configured as 0."""
        from modules.llm.providers.base import load_max_retries

        cfg = {"concurrency": {"transcription": {"retry": {"attempts": 0}}}}
        with patch("modules.llm.providers.base.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = cfg
            result = load_max_retries()

        assert result >= 1


class TestBuildDisabledParams:
    """Tests for BaseProvider._build_disabled_params()."""

    def _make_provider(self, caps):
        """Create a minimal concrete provider with given capabilities."""
        class _ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"
            def get_capabilities(self):
                return caps
            async def transcribe_image_from_base64(self, *a, **kw):
                pass
            async def close(self):
                pass

        p = _ConcreteProvider.__new__(_ConcreteProvider)
        p._capabilities = caps
        return p

    @pytest.mark.unit
    def test_returns_none_when_all_supported(self):
        """Returns None when all sampler params are supported."""
        caps = Capabilities(
            model="gpt-4o",
            family="gpt-4o",
            provider="openai",
            supports_sampler_controls=True,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
        )
        provider = self._make_provider(caps)
        assert provider._build_disabled_params() is None

    @pytest.mark.unit
    def test_disables_temperature_when_not_supported(self):
        """Adds temperature=None when supports_sampler_controls is False."""
        caps = Capabilities(
            model="o3",
            family="o3",
            provider="openai",
            supports_sampler_controls=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        )
        provider = self._make_provider(caps)
        result = provider._build_disabled_params()
        assert result is not None
        assert "temperature" in result

    @pytest.mark.unit
    def test_only_unsupported_params_disabled(self):
        """Only params with supports_X=False appear in disabled dict."""
        caps = Capabilities(
            model="claude-sonnet",
            family="claude",
            provider="anthropic",
            supports_sampler_controls=True,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        )
        provider = self._make_provider(caps)
        result = provider._build_disabled_params()
        assert result is not None
        assert "temperature" not in result
        assert "top_p" in result
        assert "frequency_penalty" in result

    @pytest.mark.unit
    def test_returns_none_when_capabilities_not_set(self):
        """Returns None when provider has no _capabilities attribute."""
        class _BareProvider(BaseProvider):
            @property
            def provider_name(self):
                return "bare"
            def get_capabilities(self):
                return None
            async def transcribe_image_from_base64(self, *a, **kw):
                pass
            async def close(self):
                pass

        p = _BareProvider.__new__(_BareProvider)
        # Deliberately don't set _capabilities
        assert p._build_disabled_params() is None


class TestProcessLLMResponse:
    """Tests for BaseProvider._process_llm_response()."""

    def _make_provider(self):
        """Create a minimal concrete provider for testing _process_llm_response."""
        class _ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"
            def get_capabilities(self):
                return Capabilities(model="m", family="test")
            async def transcribe_image_from_base64(self, *a, **kw):
                pass
            async def close(self):
                pass

        p = _ConcreteProvider.__new__(_ConcreteProvider)
        return p

    @pytest.mark.unit
    def test_plain_ai_message_response(self):
        """Plain AIMessage with string content is handled correctly."""
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "Hello transcribed text"
        mock_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(mock_msg, OPENAI_TOKEN_MAPPING)
            )

        assert result.content == "Hello transcribed text"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15

    @pytest.mark.unit
    def test_structured_output_dict_response(self):
        """with_structured_output(include_raw=True) dict response is parsed correctly."""
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        parsed_data = {"transcription": "Structured text", "no_transcribable_text": False}
        raw_msg = MagicMock()
        raw_msg.response_metadata = {}
        response = {"raw": raw_msg, "parsed": parsed_data}

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(response, OPENAI_TOKEN_MAPPING)
            )

        assert result.parsed_output == parsed_data
        assert "Structured text" in result.content or result.content != ""

    @pytest.mark.unit
    def test_plain_dict_response_serialized(self):
        """A plain dict response is JSON-serialized."""
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        response = {"transcription": "dict content"}

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(response, OPENAI_TOKEN_MAPPING)
            )

        assert "dict content" in result.content

    @pytest.mark.unit
    def test_anthropic_token_mapping(self):
        """Anthropic token mapping (different key names) extracts tokens correctly."""
        import asyncio
        from modules.llm.providers.base import ANTHROPIC_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "anthropic text"
        mock_msg.response_metadata = {
            "usage": {"input_tokens": 20, "output_tokens": 8}
        }

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(mock_msg, ANTHROPIC_TOKEN_MAPPING)
            )

        assert result.input_tokens == 20
        assert result.output_tokens == 8
        assert result.total_tokens == 28  # computed from in+out

    @pytest.mark.unit
    def test_zero_tokens_no_tracker_call(self):
        """Token tracker is not called when total_tokens is 0."""
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "text"
        mock_msg.response_metadata = {}  # no usage key
        mock_msg.usage_metadata = None   # no fallback either

        with patch("modules.infra.token_tracker.get_token_tracker") as mock_tracker:
            asyncio.run(
                provider._process_llm_response(mock_msg, OPENAI_TOKEN_MAPPING)
            )

        mock_tracker.assert_not_called()

    @pytest.mark.unit
    def test_usage_metadata_fallback_when_response_metadata_empty(self):
        """usage_metadata fallback fires when response_metadata has no token_usage.

        Regression test for LangChain 1.x / Responses API: token usage is stored
        in AIMessage.usage_metadata (a TypedDict, i.e. plain dict), not
        response_metadata['token_usage'].
        """
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "fallback text"
        mock_msg.response_metadata = {}  # no token_usage key

        # Use a real dict to match UsageMetadata TypedDict behaviour
        mock_msg.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
        }

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(mock_msg, OPENAI_TOKEN_MAPPING)
            )

        assert result.input_tokens == 200
        assert result.output_tokens == 80
        assert result.total_tokens == 280

    @pytest.mark.unit
    def test_usage_metadata_fallback_responses_api_shape(self):
        """Responses API response_metadata shape yields tokens via usage_metadata.

        When LangChain routes through the Responses API (e.g. due to text.verbosity
        or reasoning parameters), response_metadata contains id/model/object/service_tier
        but no token_usage.  Token counts must be read from AIMessage.usage_metadata
        which is a TypedDict (plain dict).
        """
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "transcribed page"
        mock_msg.response_metadata = {
            "id": "resp_abc123",
            "object": "response",
            "model": "gpt-5-mini-2025-08-07",
            "service_tier": "flex",
            "status": "completed",
            "model_provider": "openai",
            "model_name": "gpt-5-mini-2025-08-07",
        }

        # Use a real dict to match UsageMetadata TypedDict behaviour
        mock_msg.usage_metadata = {
            "input_tokens": 1500,
            "output_tokens": 350,
            "total_tokens": 1850,
        }

        with patch("modules.infra.token_tracker.get_token_tracker") as mock_tracker:
            result = asyncio.run(
                provider._process_llm_response(mock_msg, OPENAI_TOKEN_MAPPING)
            )

        assert result.input_tokens == 1500
        assert result.output_tokens == 350
        assert result.total_tokens == 1850
        mock_tracker.return_value.add_tokens.assert_called_once_with(1850)

    @pytest.mark.unit
    def test_response_metadata_takes_priority_over_usage_metadata(self):
        """response_metadata token_usage takes priority; usage_metadata is not read.

        When response_metadata already contains token_usage with non-zero totals,
        the usage_metadata fallback must not overwrite those values.
        """
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "some text"
        mock_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        }

        # Use a real dict to match UsageMetadata TypedDict behaviour
        mock_msg.usage_metadata = {
            "input_tokens": 999,
            "output_tokens": 999,
            "total_tokens": 1998,
        }

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(mock_msg, OPENAI_TOKEN_MAPPING)
            )

        assert result.input_tokens == 50
        assert result.output_tokens == 20
        assert result.total_tokens == 70

    @pytest.mark.unit
    def test_usage_metadata_fallback_non_dict_object(self):
        """Defensive fallback: usage_metadata as object (not dict) still works.

        Covers the getattr branch for non-dict usage_metadata implementations.
        """
        import asyncio
        from modules.llm.providers.base import OPENAI_TOKEN_MAPPING

        provider = self._make_provider()
        mock_msg = MagicMock()
        mock_msg.content = "object-style text"
        mock_msg.response_metadata = {}  # no token_usage key

        # Use MagicMock (object-attribute style) for the defensive branch
        usage_meta = MagicMock()
        usage_meta.input_tokens = 400
        usage_meta.output_tokens = 100
        usage_meta.total_tokens = 500
        mock_msg.usage_metadata = usage_meta

        with patch("modules.infra.token_tracker.get_token_tracker"):
            result = asyncio.run(
                provider._process_llm_response(mock_msg, OPENAI_TOKEN_MAPPING)
            )

        assert result.input_tokens == 400
        assert result.output_tokens == 100
        assert result.total_tokens == 500


class TestAInvokeWithRetry:
    """Tests for BaseProvider._ainvoke_with_retry() retry logic."""

    def _make_provider(self):
        """Create a minimal concrete provider for testing _ainvoke_with_retry."""
        class _ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"

            def get_capabilities(self):
                return Capabilities(model="m", family="test")

            async def transcribe_image_from_base64(self, *a, **kw):
                pass

            async def close(self):
                pass

        return _ConcreteProvider.__new__(_ConcreteProvider)

    @pytest.mark.unit
    def test_retries_on_connect_error(self):
        """Retries on httpx.ConnectError and returns the result on eventual success."""
        import asyncio
        import httpx
        import tenacity
        from unittest.mock import AsyncMock

        provider = self._make_provider()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            "success_response",
        ])

        async def _run():
            with patch("modules.llm.providers.base.load_max_retries", return_value=5):
                with patch(
                    "tenacity.wait_exponential_jitter",
                    return_value=tenacity.wait_none(),
                ):
                    return await provider._ainvoke_with_retry(mock_llm, ["msg"])

        result = asyncio.run(_run())
        assert result == "success_response"
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.unit
    def test_reraises_after_exhausting_attempts(self):
        """Raises httpx.ConnectError after all retry attempts are exhausted."""
        import asyncio
        import httpx
        import tenacity
        from unittest.mock import AsyncMock

        provider = self._make_provider()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=httpx.ConnectError("always fails")
        )

        async def _run():
            with patch("modules.llm.providers.base.load_max_retries", return_value=2):
                with patch(
                    "tenacity.wait_exponential_jitter",
                    return_value=tenacity.wait_none(),
                ):
                    return await provider._ainvoke_with_retry(mock_llm, ["msg"])

        with pytest.raises(httpx.ConnectError):
            asyncio.run(_run())
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.unit
    def test_does_not_retry_non_transient_error(self):
        """Non-transient errors (e.g. ValueError) propagate immediately without retry."""
        import asyncio
        import tenacity
        from unittest.mock import AsyncMock

        provider = self._make_provider()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ValueError("bad value"))

        async def _run():
            with patch("modules.llm.providers.base.load_max_retries", return_value=5):
                with patch(
                    "tenacity.wait_exponential_jitter",
                    return_value=tenacity.wait_none(),
                ):
                    return await provider._ainvoke_with_retry(mock_llm, ["msg"])

        with pytest.raises(ValueError, match="bad value"):
            asyncio.run(_run())
        assert mock_llm.ainvoke.call_count == 1


class TestBaseProviderAsyncContextManager:
    """Tests for BaseProvider async context manager (__aenter__, __aexit__)."""

    @pytest.mark.unit
    def test_async_context_manager_returns_self(self):
        """__aenter__ returns the provider instance."""
        import asyncio

        class _ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"
            def get_capabilities(self):
                return Capabilities(model="m", family="test")
            async def transcribe_image_from_base64(self, *a, **kw):
                pass
            async def close(self):
                pass

        provider = _ConcreteProvider(api_key="k", model="m")

        async def _check():
            async with provider as p:
                assert p is provider

        asyncio.run(_check())
