"""Unit tests for modules/llm/providers/anthropic_provider.py.

Tests capability detection, provider initialization, and reasoning parameter
translation for Anthropic Claude models.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestAnthropicGetModelCapabilities:
    """Tests for _get_model_capabilities() — pure capability detection logic."""

    @pytest.mark.unit
    def test_claude_opus_45_is_reasoning_model(self):
        """Claude Opus 4.5 supports extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-opus-4-5-20250929")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.provider == "anthropic"

    @pytest.mark.unit
    def test_claude_sonnet_45_is_reasoning_model(self):
        """Claude Sonnet 4.5 supports extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-sonnet-4-5-20250929")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_claude_haiku_45_is_reasoning_model(self):
        """Claude Haiku 4.5 supports extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-haiku-4-5-20251001")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_claude_opus_41_is_reasoning_model(self):
        """Claude Opus 4.1 supports extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-opus-4-1-20250805")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_claude_opus_4_is_reasoning_model(self):
        """Claude Opus 4 supports extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-opus-4-20250601")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_claude_sonnet_4_is_not_reasoning_model(self):
        """Claude Sonnet 4 (non-4.5) does not support extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-sonnet-4-20250514")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_claude_35_sonnet_is_not_reasoning_model(self):
        """Claude 3.5 Sonnet does not support extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-3-5-sonnet-20241022")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_claude_3_opus_is_not_reasoning_model(self):
        """Claude 3 Opus does not support extended thinking."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-3-opus-20240229")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_all_claude_models_support_vision(self):
        """All supported Claude models support vision."""
        from modules.llm.model_capabilities import detect_capabilities

        for model in (
            "claude-opus-4-5-20250929",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ):
            caps = detect_capabilities(model)
            assert caps.supports_image_input is True, f"{model} should support vision"

    @pytest.mark.unit
    def test_claude_models_do_not_support_image_detail(self):
        """Claude models do not use the OpenAI-style image_detail parameter."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-sonnet-4-5-20250929")
        assert caps.supports_image_detail is False

    @pytest.mark.unit
    def test_claude_35_haiku_no_structured_output(self):
        """Claude 3.5 Haiku does not support structured outputs."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-3-5-haiku-20241022")
        assert caps.supports_structured_outputs is False

    @pytest.mark.unit
    def test_claude_45_sonnet_supports_structured_output(self):
        """Claude 4.5 Sonnet supports native structured outputs."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-sonnet-4-5-20250929")
        assert caps.supports_structured_outputs is True

    @pytest.mark.unit
    def test_unknown_claude_model_returns_default_caps(self):
        """Unknown Claude model names return sensible defaults."""
        from modules.llm.model_capabilities import detect_capabilities

        caps = detect_capabilities("claude-future-model-9000")
        assert caps.provider == "anthropic"
        assert caps.supports_image_input is True


class TestAnthropicProviderInit:
    """Tests for AnthropicProvider.__init__ and parameter handling."""

    @pytest.mark.unit
    def test_provider_name_is_anthropic(self):
        """provider_name property returns 'anthropic'."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic"):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                provider = AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-3-5-sonnet-20241022",
                )

        assert provider.provider_name == "anthropic"

    @pytest.mark.unit
    def test_reasoning_config_stored(self):
        """reasoning_config is stored when provided."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic"):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                provider = AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-sonnet-4-5-20250929",
                    reasoning_config={"effort": "high"},
                )

        assert provider.reasoning_config == {"effort": "high"}

    @pytest.mark.unit
    def test_thinking_param_set_for_reasoning_model(self):
        """Extended thinking parameter is passed to ChatAnthropic for reasoning models."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-sonnet-4-5-20250929",
                    reasoning_config={"effort": "medium"},
                )

        assert "thinking" in captured, "thinking param must be set for reasoning models"
        assert captured["thinking"]["type"] == "enabled"
        assert captured["thinking"]["budget_tokens"] == 4096  # medium maps to 4096

    @pytest.mark.unit
    def test_thinking_low_effort_maps_to_1024_budget(self):
        """effort='low' maps to budget_tokens=1024."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-sonnet-4-5-20250929",
                    reasoning_config={"effort": "low"},
                )

        assert captured["thinking"]["budget_tokens"] == 1024

    @pytest.mark.unit
    def test_thinking_high_effort_maps_to_16384_budget(self):
        """effort='high' maps to budget_tokens=16384."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-sonnet-4-5-20250929",
                    reasoning_config={"effort": "high"},
                )

        assert captured["thinking"]["budget_tokens"] == 16384

    @pytest.mark.unit
    def test_thinking_not_set_for_non_reasoning_model(self):
        """No thinking parameter for non-reasoning models (e.g. Claude 3.5 Sonnet)."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-3-5-sonnet-20241022",
                    reasoning_config={"effort": "medium"},
                )

        assert "thinking" not in captured

    @pytest.mark.unit
    def test_thinking_not_set_without_reasoning_config(self):
        """No thinking parameter when reasoning_config is None."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-sonnet-4-5-20250929",
                )

        assert "thinking" not in captured

    @pytest.mark.unit
    def test_max_tokens_capped_at_model_limit(self):
        """max_tokens is capped at model's max_output_tokens."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider

        captured: Dict[str, Any] = {}

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic",
                   side_effect=lambda **kw: captured.update(kw) or MagicMock()):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-3-haiku-20240307",  # max 4096
                    max_tokens=100000,  # far exceeds model limit
                )

        assert captured.get("max_tokens") <= 4096

    @pytest.mark.unit
    def test_get_capabilities_returns_provider_capabilities(self):
        """get_capabilities() returns a ProviderCapabilities instance."""
        from modules.llm.providers.anthropic_provider import AnthropicProvider
        from modules.llm.model_capabilities import Capabilities

        with patch("modules.llm.providers.anthropic_provider.ChatAnthropic"):
            with patch("modules.llm.providers.anthropic_provider.load_max_retries",
                       return_value=3):
                provider = AnthropicProvider(
                    api_key="sk-ant-test",
                    model="claude-3-5-sonnet-20241022",
                )

        caps = provider.get_capabilities()
        assert isinstance(caps, Capabilities)
        assert caps.provider == "anthropic"


class TestTransformSchemaForAnthropic:
    """Tests for _transform_schema_for_anthropic() utility."""

    @pytest.mark.unit
    def test_union_type_collapsed_to_first_non_null(self):
        """Union types like ['string', 'null'] are collapsed to 'string'."""
        from modules.llm.providers.anthropic_provider import _transform_schema_for_anthropic

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
            },
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["properties"]["name"]["type"] == "string"

    @pytest.mark.unit
    def test_title_added_if_missing(self):
        """Title key is added when not present."""
        from modules.llm.providers.anthropic_provider import _transform_schema_for_anthropic

        schema = {"type": "object", "properties": {}}
        result = _transform_schema_for_anthropic(schema)
        assert "title" in result

    @pytest.mark.unit
    def test_description_added_if_missing(self):
        """Description key is added when not present."""
        from modules.llm.providers.anthropic_provider import _transform_schema_for_anthropic

        schema = {"type": "object", "properties": {}}
        result = _transform_schema_for_anthropic(schema)
        assert "description" in result

    @pytest.mark.unit
    def test_existing_title_preserved(self):
        """Existing title is not overwritten."""
        from modules.llm.providers.anthropic_provider import _transform_schema_for_anthropic

        schema = {"type": "object", "title": "MyTitle", "properties": {}}
        result = _transform_schema_for_anthropic(schema)
        assert result["title"] == "MyTitle"

    @pytest.mark.unit
    def test_nested_union_types_collapsed(self):
        """Union types in nested properties are also collapsed."""
        from modules.llm.providers.anthropic_provider import _transform_schema_for_anthropic

        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "type": "object",
                    "properties": {
                        "value": {"type": ["integer", "null"]},
                    },
                }
            },
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["properties"]["item"]["properties"]["value"]["type"] == "integer"

    @pytest.mark.unit
    def test_original_schema_not_mutated(self):
        """The original schema dict is not modified (deep copy is used)."""
        from modules.llm.providers.anthropic_provider import _transform_schema_for_anthropic

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
            },
        }
        original_type = schema["properties"]["name"]["type"]
        _transform_schema_for_anthropic(schema)
        assert schema["properties"]["name"]["type"] == original_type  # unchanged
