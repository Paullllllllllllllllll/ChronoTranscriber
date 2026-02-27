"""Unit tests for modules/llm/providers/openrouter_provider.py.

Tests capability detection for OpenRouter models (200+ models via unified API),
provider initialization, reasoning configuration, and transcription flow.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.llm.providers.openrouter_provider import (
    _effort_to_ratio,
    _compute_openrouter_reasoning_max_tokens,
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
)


class TestOpenRouterGetModelCapabilities:
    """Tests for _get_model_capabilities() — pure capability detection logic."""

    @pytest.mark.unit
    def test_deepseek_r1_is_reasoning_model(self):
        """DeepSeek R1 is classified as a reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("deepseek/deepseek-r1")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.provider == "openrouter"

    @pytest.mark.unit
    def test_deepseek_v3_is_not_reasoning_model(self):
        """DeepSeek V3 (non-R1) is not classified as a reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("deepseek/deepseek-v3")
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_openai_gpt5_via_openrouter_is_reasoning(self):
        """GPT-5 via OpenRouter is classified as reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("openai/gpt-5")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_openai_gpt4o_via_openrouter_is_not_reasoning(self):
        """GPT-4o via OpenRouter is not a reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("openai/gpt-4o")
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_anthropic_claude_45_via_openrouter_is_reasoning(self):
        """Claude 4.5 via OpenRouter is a reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("anthropic/claude-sonnet-4-5")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_anthropic_claude_35_via_openrouter_returns_caps(self):
        """Claude 3.5 via OpenRouter returns OpenRouter capabilities."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("anthropic/claude-3-5-sonnet")
        assert caps.provider == "openrouter"
        assert caps.supports_image_input is True

    @pytest.mark.unit
    def test_google_gemini_25_via_openrouter_is_reasoning(self):
        """Gemini 2.5 via OpenRouter is a reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("google/gemini-2.5-pro")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_google_gemini_20_via_openrouter_is_not_reasoning(self):
        """Gemini 2.0 via OpenRouter is not a reasoning model."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("google/gemini-2.0-flash")
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_meta_llama_via_openrouter(self):
        """Meta Llama models are handled via OpenRouter."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("meta/llama-3.2-90b-vision")
        assert caps.provider == "openrouter"

    @pytest.mark.unit
    def test_mistral_pixtral_via_openrouter(self):
        """Mistral Pixtral models are handled via OpenRouter."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("mistral/pixtral-large")
        assert caps.provider == "openrouter"

    @pytest.mark.unit
    def test_unknown_model_returns_default_caps(self):
        """Unknown model names return sensible defaults."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("unknown-provider/future-model-9000")
        assert caps.provider == "openrouter"

    @pytest.mark.unit
    def test_gpt_oss_is_reasoning_model(self):
        """GPT-OSS models support reasoning."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("openai/gpt-oss-120b")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_all_vision_models_support_vision(self):
        """Key vision-capable models correctly report supports_image_input=True."""
        from modules.llm.model_capabilities import detect_capabilities

        vision_models = [
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4-5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-r1",
        ]
        for model in vision_models:
            caps = detect_capabilities(model)
            assert caps.supports_image_input is True, f"{model} should support vision"


# ---------------------------------------------------------------------------
# _effort_to_ratio
# ---------------------------------------------------------------------------

class TestEffortToRatio:
    def test_xhigh(self):
        assert _effort_to_ratio("xhigh") == 0.95

    def test_high(self):
        assert _effort_to_ratio("high") == 0.80

    def test_medium(self):
        assert _effort_to_ratio("medium") == 0.50

    def test_low(self):
        assert _effort_to_ratio("low") == 0.20

    def test_minimal(self):
        assert _effort_to_ratio("minimal") == 0.10

    def test_none_effort(self):
        assert _effort_to_ratio("none") == 0.0

    def test_unknown_defaults_to_medium(self):
        assert _effort_to_ratio("unknown") == 0.50

    def test_case_insensitive(self):
        assert _effort_to_ratio("HIGH") == 0.80
        assert _effort_to_ratio("  Medium  ") == 0.50

    def test_empty_string(self):
        assert _effort_to_ratio("") == 0.50

    def test_none_value(self):
        assert _effort_to_ratio(None) == 0.50


# ---------------------------------------------------------------------------
# _compute_openrouter_reasoning_max_tokens
# ---------------------------------------------------------------------------

class TestComputeOpenrouterReasoningMaxTokens:
    def test_medium_effort(self):
        result = _compute_openrouter_reasoning_max_tokens(
            max_tokens=16384, effort="medium"
        )
        assert 1024 <= result <= 32000

    def test_high_effort_greater_than_low(self):
        high = _compute_openrouter_reasoning_max_tokens(
            max_tokens=16384, effort="high"
        )
        low = _compute_openrouter_reasoning_max_tokens(
            max_tokens=16384, effort="low"
        )
        assert high > low

    def test_none_effort_returns_zero(self):
        result = _compute_openrouter_reasoning_max_tokens(
            max_tokens=16384, effort="none"
        )
        assert result == 0

    def test_minimum_budget_is_1024(self):
        result = _compute_openrouter_reasoning_max_tokens(
            max_tokens=2000, effort="minimal"
        )
        assert result >= 1024

    def test_budget_capped_at_32000(self):
        result = _compute_openrouter_reasoning_max_tokens(
            max_tokens=100000, effort="xhigh"
        )
        assert result <= 32000


# ---------------------------------------------------------------------------
# OpenRouterProvider Initialization
# ---------------------------------------------------------------------------

class TestOpenRouterProviderInit:
    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    def test_basic_initialization(self, mock_chat, mock_retries):
        provider = OpenRouterProvider(
            api_key="test-key",
            model="openai/gpt-4o",
            temperature=0.1,
            max_tokens=4096,
        )
        assert provider.provider_name == "openrouter"
        assert provider.model == "openai/gpt-4o"
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["base_url"] == OPENROUTER_BASE_URL

    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    def test_default_app_name(self, mock_chat, mock_retries):
        provider = OpenRouterProvider(api_key="k", model="m")
        assert provider.app_name == "ChronoTranscriber"

    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    def test_custom_app_name(self, mock_chat, mock_retries):
        provider = OpenRouterProvider(api_key="k", model="m", app_name="MyApp")
        assert provider.app_name == "MyApp"

    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    def test_get_capabilities_returns_object(self, mock_chat, mock_retries):
        provider = OpenRouterProvider(api_key="k", model="openai/gpt-4o")
        caps = provider.get_capabilities()
        assert caps is not None

    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    def test_reasoning_config_stored(self, mock_chat, mock_retries):
        rc = {"effort": "high"}
        provider = OpenRouterProvider(api_key="k", model="m", reasoning_config=rc)
        assert provider.reasoning_config == rc

    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    def test_headers_include_referer_and_title(self, mock_chat, mock_retries):
        provider = OpenRouterProvider(
            api_key="k",
            model="m",
            site_url="https://example.com",
            app_name="TestApp",
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert call_kwargs["default_headers"]["X-Title"] == "TestApp"


# ---------------------------------------------------------------------------
# OpenRouterProvider Reasoning Config
# ---------------------------------------------------------------------------

class TestOpenRouterProviderReasoningConfig:
    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    @patch("modules.llm.providers.openrouter_provider.detect_capabilities")
    def test_anthropic_model_maps_effort_to_max_tokens(
        self, mock_caps, mock_chat, mock_retries
    ):
        caps = MagicMock()
        caps.supports_reasoning_effort = True
        caps.supports_image_input = True
        caps.supports_image_detail = False
        caps.supports_structured_outputs = True
        mock_caps.return_value = caps

        provider = OpenRouterProvider(
            api_key="k",
            model="anthropic/claude-opus-4",
            reasoning_config={"effort": "high"},
            max_tokens=16384,
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        assert "reasoning" in extra_body
        assert "max_tokens" in extra_body["reasoning"]
        assert "effort" not in extra_body["reasoning"]

    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    @patch("modules.llm.providers.openrouter_provider.detect_capabilities")
    def test_deepseek_model_maps_effort_to_enabled_flag(
        self, mock_caps, mock_chat, mock_retries
    ):
        caps = MagicMock()
        caps.supports_reasoning_effort = True
        caps.supports_image_input = False
        mock_caps.return_value = caps

        provider = OpenRouterProvider(
            api_key="k",
            model="deepseek/deepseek-r1",
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        assert "reasoning" in extra_body
        assert extra_body["reasoning"].get("enabled") is True
        assert "effort" not in extra_body["reasoning"]


# ---------------------------------------------------------------------------
# OpenRouterProvider.transcribe_image_from_base64
# ---------------------------------------------------------------------------

class TestOpenRouterProviderTranscribe:
    @pytest.mark.asyncio
    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    @patch("modules.llm.providers.openrouter_provider.detect_capabilities")
    async def test_returns_error_for_non_vision_model(
        self, mock_caps, mock_chat, mock_retries
    ):
        caps = MagicMock()
        caps.supports_image_input = False
        mock_caps.return_value = caps

        provider = OpenRouterProvider(api_key="k", model="text-only-model")
        result = await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe.",
        )
        assert result.transcription_not_possible is True
        assert "does not support vision" in result.error

    @pytest.mark.asyncio
    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    @patch("modules.llm.providers.openrouter_provider.detect_capabilities")
    async def test_invokes_llm_for_vision_model(
        self, mock_caps, mock_chat, mock_retries
    ):
        caps = MagicMock()
        caps.supports_image_input = True
        caps.supports_image_detail = False
        caps.supports_structured_outputs = False
        caps.default_image_detail = None
        mock_caps.return_value = caps

        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Transcribed text"
        mock_response.response_metadata = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)
        mock_chat.return_value = mock_llm_instance

        provider = OpenRouterProvider(api_key="k", model="openai/gpt-4o")
        result = await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe this.",
        )
        mock_llm_instance.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    @patch("modules.llm.providers.openrouter_provider.detect_capabilities")
    async def test_invoke_exception_returns_error_result(
        self, mock_caps, mock_chat, mock_retries
    ):
        caps = MagicMock()
        caps.supports_image_input = True
        caps.supports_image_detail = False
        caps.supports_structured_outputs = False
        caps.default_image_detail = None
        mock_caps.return_value = caps

        mock_llm_instance = MagicMock()
        mock_llm_instance.ainvoke = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        mock_chat.return_value = mock_llm_instance

        provider = OpenRouterProvider(api_key="k", model="openai/gpt-4o")
        result = await provider.transcribe_image_from_base64(
            image_base64="abc123",
            mime_type="image/png",
            system_prompt="Transcribe.",
        )
        assert result.transcription_not_possible is True
        assert "API error" in result.error


class TestOpenRouterProviderClose:
    @pytest.mark.asyncio
    @patch("modules.llm.providers.openrouter_provider.load_max_retries", return_value=3)
    @patch("modules.llm.providers.openrouter_provider.ChatOpenAI")
    async def test_close_does_not_raise(self, mock_chat, mock_retries):
        provider = OpenRouterProvider(api_key="k", model="m")
        await provider.close()
