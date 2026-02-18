"""Unit tests for modules/llm/providers/openai_provider.py.

Tests OpenAI provider initialization, capabilities, and parameter forwarding.
Includes CT-3 regression tests for text.verbosity forwarding.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestOpenAIProviderCapabilities:
    """Tests for _get_model_capabilities helper."""

    @pytest.mark.unit
    def test_gpt5_is_reasoning_model(self):
        """GPT-5 family is classified as a reasoning model."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        caps = _get_model_capabilities("gpt-5-mini")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gpt5_1_is_reasoning_model(self):
        """GPT-5.1 family is classified as a reasoning model."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        caps = _get_model_capabilities("gpt-5.1")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gpt4o_is_not_reasoning_model(self):
        """GPT-4o is NOT classified as a reasoning model."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        caps = _get_model_capabilities("gpt-4o")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_o3_is_reasoning_model(self):
        """o3 is classified as a reasoning model."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        caps = _get_model_capabilities("o3")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gpt4o_supports_temperature(self):
        """GPT-4o supports sampler controls."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        caps = _get_model_capabilities("gpt-4o")
        assert caps.supports_temperature is True

    @pytest.mark.unit
    def test_gpt5_does_not_support_temperature(self):
        """GPT-5 reasoning models do not support temperature."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        caps = _get_model_capabilities("gpt-5")
        assert caps.supports_temperature is False

    @pytest.mark.unit
    def test_gpt4_1_family_capabilities(self):
        """GPT-4.1 family has correct capability flags."""
        from modules.llm.providers.openai_provider import _get_model_capabilities

        for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            caps = _get_model_capabilities(model)
            assert caps.is_reasoning_model is False
            assert caps.supports_vision is True
            assert caps.supports_structured_output is True


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider.__init__ parameter handling."""

    @pytest.mark.unit
    def test_service_tier_stored(self):
        """service_tier is stored on the provider instance."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with patch("modules.llm.providers.openai_provider.ChatOpenAI"):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                provider = OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-4o",
                    service_tier="auto",
                )

        assert provider.service_tier == "auto"

    @pytest.mark.unit
    def test_reasoning_config_stored(self):
        """reasoning_config is stored on the provider instance."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with patch("modules.llm.providers.openai_provider.ChatOpenAI"):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                provider = OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    reasoning_config={"effort": "high"},
                )

        assert provider.reasoning_config == {"effort": "high"}

    @pytest.mark.unit
    def test_reasoning_dict_forwarded_to_chat_openai(self):
        """reasoning dict is passed to ChatOpenAI for reasoning models (Responses API)."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    reasoning_config={"effort": "high"},
                )

        assert captured.get("reasoning") == {"effort": "high"}
        assert "reasoning_effort" not in captured

    @pytest.mark.unit
    def test_service_tier_forwarded_to_chat_openai(self):
        """service_tier is passed to ChatOpenAI when set."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-4o",
                    service_tier="flex",
                )

        assert captured.get("service_tier") == "flex"

    @pytest.mark.unit
    def test_provider_name_is_openai(self):
        """provider_name property returns 'openai'."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with patch("modules.llm.providers.openai_provider.ChatOpenAI"):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

        assert provider.provider_name == "openai"

    @pytest.mark.unit
    def test_max_completion_tokens_used_for_reasoning_model(self):
        """Reasoning models use max_completion_tokens instead of max_tokens."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    max_tokens=8192,
                )

        assert captured.get("max_completion_tokens") == 8192
        assert "max_tokens" not in captured

    @pytest.mark.unit
    def test_max_tokens_used_for_non_reasoning_model(self):
        """Non-reasoning models use max_tokens."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-4o",
                    max_tokens=4096,
                )

        assert captured.get("max_tokens") == 4096
        assert "max_completion_tokens" not in captured

    @pytest.mark.unit
    def test_use_responses_api_always_set(self):
        """use_responses_api=True is always passed to ChatOpenAI."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(api_key="sk-test", model="gpt-4o")

        assert captured.get("use_responses_api") is True


# =============================================================================
# CT-3: text.verbosity forwarding through OpenAIProvider (Responses API)
# =============================================================================

class TestOpenAIProviderTextVerbosity:
    """CT-3 regression tests: text_config stored and verbosity forwarded.

    Verbosity is forwarded via the LangChain `verbosity` parameter
    (Responses API native) rather than `model_kwargs["text"]`.
    """

    @pytest.mark.unit
    def test_text_config_stored_on_provider(self):
        """OpenAIProvider stores text_config attribute when provided."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with patch("modules.llm.providers.openai_provider.ChatOpenAI"):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                provider = OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    text_config={"verbosity": "concise"},
                )

        assert provider.text_config == {"verbosity": "concise"}

    @pytest.mark.unit
    def test_text_config_defaults_to_none(self):
        """text_config defaults to None when not provided."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with patch("modules.llm.providers.openai_provider.ChatOpenAI"):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

        assert provider.text_config is None

    @pytest.mark.unit
    def test_verbosity_passed_as_parameter_for_reasoning_model(self):
        """For GPT-5 reasoning models, text verbosity is passed as direct parameter."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    text_config={"verbosity": "verbose"},
                )

        assert captured.get("verbosity") == "verbose"

    @pytest.mark.unit
    def test_verbosity_not_set_for_non_reasoning_model(self):
        """For non-reasoning models (gpt-4o), verbosity is NOT passed."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-4o",
                    text_config={"verbosity": "concise"},
                )

        assert "verbosity" not in captured

    @pytest.mark.unit
    def test_verbosity_skipped_when_text_config_empty(self):
        """Empty text_config dict does not add verbosity parameter."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.openai_provider.load_max_retries", return_value=3):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    text_config={},
                )

        assert "verbosity" not in captured

    @pytest.mark.unit
    def test_all_verbosity_levels_forwarded(self):
        """All valid verbosity levels (concise/medium/verbose) are forwarded correctly."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        for level in ("concise", "medium", "verbose"):
            captured: Dict[str, Any] = {}

            with patch("modules.llm.providers.openai_provider.ChatOpenAI",
                       side_effect=lambda **kw: captured.update(kw) or MagicMock()):
                with patch("modules.llm.providers.openai_provider.load_max_retries",
                           return_value=3):
                    OpenAIProvider(
                        api_key="sk-test",
                        model="gpt-5-mini",
                        text_config={"verbosity": level},
                    )

            assert captured.get("verbosity") == level
