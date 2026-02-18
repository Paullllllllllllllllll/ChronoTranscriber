"""Unit tests for modules/llm/providers/google_provider.py.

Tests capability detection, provider initialization, and thinking-level
parameter translation for Google Gemini models.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestGoogleGetModelCapabilities:
    """Tests for _get_model_capabilities() — pure capability detection logic."""

    @pytest.mark.unit
    def test_gemini_3_flash_is_reasoning_model(self):
        """Gemini 3 Flash supports thinking mode."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-3-flash-preview-05-20")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.provider == "google"

    @pytest.mark.unit
    def test_gemini_3_pro_is_reasoning_model(self):
        """Gemini 3 Pro supports thinking mode."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-3-pro")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gemini_25_pro_is_reasoning_model(self):
        """Gemini 2.5 Pro supports adaptive thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-2.5-pro")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gemini_25_flash_is_reasoning_model(self):
        """Gemini 2.5 Flash supports adaptive thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-2.5-flash")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gemini_20_flash_is_not_reasoning_model(self):
        """Gemini 2.0 Flash does NOT support thinking mode."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-2.0-flash")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_gemini_15_pro_is_not_reasoning_model(self):
        """Gemini 1.5 Pro does NOT support thinking mode."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-1.5-pro")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_gemini_15_flash_is_not_reasoning_model(self):
        """Gemini 1.5 Flash does NOT support thinking mode."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-1.5-flash")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_all_gemini_models_support_vision(self):
        """All supported Gemini models support vision."""
        from modules.llm.model_capabilities import detect_capabilities

        for model in (
            "gemini-3-pro",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ):
            caps = detect_capabilities(model)
            assert caps.supports_image_input is True, f"{model} should support vision"

    @pytest.mark.unit
    def test_gemini_models_support_media_resolution(self):
        """Gemini models support the media_resolution parameter."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-2.5-pro")
        assert caps.supports_media_resolution is True

    @pytest.mark.unit
    def test_gemini_models_do_not_support_image_detail(self):
        """Gemini models do not use the OpenAI-style image_detail parameter."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-2.5-pro")
        assert caps.supports_image_detail is False

    @pytest.mark.unit
    def test_unknown_gemini_model_returns_default_caps(self):
        """Unknown Gemini model names return sensible defaults."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-future-model")
        assert caps.provider == "google"
        assert caps.supports_image_input is True

    @pytest.mark.unit
    def test_gemini_25_pro_has_large_context(self):
        """Gemini 2.5 Pro has 2M token context window."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("gemini-2.5-pro")
        assert caps.max_context_tokens >= 1_000_000


class TestGoogleProviderInit:
    """Tests for GoogleProvider.__init__ and parameter handling."""

    @pytest.mark.unit
    def test_provider_name_is_google(self):
        """provider_name property returns 'google'."""
        from modules.llm.providers.google_provider import GoogleProvider

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI"):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                provider = GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-1.5-pro",
                )

        assert provider.provider_name == "google"

    @pytest.mark.unit
    def test_reasoning_config_stored(self):
        """reasoning_config is stored when provided."""
        from modules.llm.providers.google_provider import GoogleProvider

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI"):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                provider = GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-2.5-pro",
                    reasoning_config={"effort": "high"},
                )

        assert provider.reasoning_config == {"effort": "high"}

    @pytest.mark.unit
    def test_thinking_level_high_for_medium_effort(self):
        """effort='medium' maps to thinking_level='high'."""
        from modules.llm.providers.google_provider import GoogleProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-2.5-pro",
                    reasoning_config={"effort": "medium"},
                )

        assert captured.get("thinking_level") == "high"

    @pytest.mark.unit
    def test_thinking_level_high_for_high_effort(self):
        """effort='high' maps to thinking_level='high'."""
        from modules.llm.providers.google_provider import GoogleProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-2.5-pro",
                    reasoning_config={"effort": "high"},
                )

        assert captured.get("thinking_level") == "high"

    @pytest.mark.unit
    def test_thinking_level_low_for_low_effort(self):
        """effort='low' maps to thinking_level='low'."""
        from modules.llm.providers.google_provider import GoogleProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-2.5-pro",
                    reasoning_config={"effort": "low"},
                )

        assert captured.get("thinking_level") == "low"

    @pytest.mark.unit
    def test_thinking_level_not_set_for_non_reasoning_model(self):
        """No thinking_level for non-reasoning models (e.g. Gemini 1.5 Pro)."""
        from modules.llm.providers.google_provider import GoogleProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-1.5-pro",
                    reasoning_config={"effort": "high"},
                )

        assert "thinking_level" not in captured

    @pytest.mark.unit
    def test_thinking_level_not_set_without_reasoning_config(self):
        """No thinking_level when reasoning_config is None."""
        from modules.llm.providers.google_provider import GoogleProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-2.5-pro",
                )

        assert "thinking_level" not in captured

    @pytest.mark.unit
    def test_max_tokens_capped_at_model_limit(self):
        """max_tokens is capped at model's max_output_tokens."""
        from modules.llm.providers.google_provider import GoogleProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-1.5-flash",  # max 8192
                    max_tokens=100000,
                )

        assert captured.get("max_tokens") <= 8192

    @pytest.mark.unit
    def test_get_capabilities_returns_provider_capabilities(self):
        """get_capabilities() returns a ProviderCapabilities instance."""
        from modules.llm.providers.google_provider import GoogleProvider
        from modules.llm.model_capabilities import Capabilities

        with patch("modules.llm.providers.google_provider.ChatGoogleGenerativeAI"):
            with patch("modules.llm.providers.google_provider.load_max_retries",
                       return_value=3):
                provider = GoogleProvider(
                    api_key="AIza-test",
                    model="gemini-1.5-pro",
                )

        caps = provider.get_capabilities()
        assert isinstance(caps, Capabilities)
        assert caps.provider == "google"
