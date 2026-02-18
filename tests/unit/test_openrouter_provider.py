"""Unit tests for modules/llm/providers/openrouter_provider.py.

Tests capability detection for OpenRouter models (200+ models via unified API).
"""

from __future__ import annotations

import pytest


class TestOpenRouterGetModelCapabilities:
    """Tests for _get_model_capabilities() — pure capability detection logic."""

    @pytest.mark.unit
    def test_deepseek_r1_is_reasoning_model(self):
        """DeepSeek R1 is classified as a reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("deepseek/deepseek-r1")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.provider_name == "openrouter"

    @pytest.mark.unit
    def test_deepseek_v3_is_not_reasoning_model(self):
        """DeepSeek V3 (non-R1) is not classified as a reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("deepseek/deepseek-v3")
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_openai_gpt5_via_openrouter_is_reasoning(self):
        """GPT-5 via OpenRouter is classified as reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("openai/gpt-5")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_openai_gpt4o_via_openrouter_is_not_reasoning(self):
        """GPT-4o via OpenRouter is not a reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("openai/gpt-4o")
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_anthropic_claude_45_via_openrouter_is_reasoning(self):
        """Claude 4.5 via OpenRouter is a reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("anthropic/claude-sonnet-4-5")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_anthropic_claude_35_via_openrouter_returns_caps(self):
        """Claude 3.5 via OpenRouter returns OpenRouter capabilities."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("anthropic/claude-3-5-sonnet")
        assert caps.provider_name == "openrouter"
        assert caps.supports_vision is True

    @pytest.mark.unit
    def test_google_gemini_25_via_openrouter_is_reasoning(self):
        """Gemini 2.5 via OpenRouter is a reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("google/gemini-2.5-pro")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_google_gemini_20_via_openrouter_is_not_reasoning(self):
        """Gemini 2.0 via OpenRouter is not a reasoning model."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("google/gemini-2.0-flash")
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_meta_llama_via_openrouter(self):
        """Meta Llama models are handled via OpenRouter."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("meta/llama-3.2-90b-vision")
        assert caps.provider_name == "openrouter"

    @pytest.mark.unit
    def test_mistral_pixtral_via_openrouter(self):
        """Mistral Pixtral models are handled via OpenRouter."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("mistral/pixtral-large")
        assert caps.provider_name == "openrouter"

    @pytest.mark.unit
    def test_unknown_model_returns_default_caps(self):
        """Unknown model names return sensible defaults."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("unknown-provider/future-model-9000")
        assert caps.provider_name == "openrouter"

    @pytest.mark.unit
    def test_gpt_oss_is_reasoning_model(self):
        """GPT-OSS models support reasoning."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        caps = _get_model_capabilities("openai/gpt-oss-120b")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_all_vision_models_support_vision(self):
        """Key vision-capable models correctly report supports_vision=True."""
        from modules.llm.providers.openrouter_provider import _get_model_capabilities

        vision_models = [
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4-5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-r1",
        ]
        for model in vision_models:
            caps = _get_model_capabilities(model)
            assert caps.supports_vision is True, f"{model} should support vision"
