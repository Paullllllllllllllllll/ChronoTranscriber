"""Unit tests for modules/llm/providers/factory.py.

Tests provider factory for dynamic LLM provider selection.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from modules.llm.providers.factory import (
    ProviderType,
    detect_provider_from_model,
    get_api_key_for_provider,
    get_available_providers,
    get_provider,
    resolve_api_key_env_var,
)


class TestProviderType:
    """Tests for ProviderType enum."""

    @pytest.mark.unit
    def test_all_providers_defined(self) -> None:
        """Test that all expected providers are defined."""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.GOOGLE.value == "google"
        assert ProviderType.OPENROUTER.value == "openrouter"

    @pytest.mark.unit
    def test_provider_count(self) -> None:
        """Test expected number of providers."""
        assert len(ProviderType) == 5


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    @pytest.mark.unit
    def test_no_keys_returns_empty(self) -> None:
        """Test that no API keys returns empty list."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_available_providers()
            assert result == []

    @pytest.mark.unit
    def test_openai_key_available(self) -> None:
        """Test detection of OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = get_available_providers()
            assert ProviderType.OPENAI in result

    @pytest.mark.unit
    def test_anthropic_key_available(self) -> None:
        """Test detection of Anthropic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            result = get_available_providers()
            assert ProviderType.ANTHROPIC in result

    @pytest.mark.unit
    def test_multiple_keys_available(self) -> None:
        """Test detection of multiple API keys."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "GOOGLE_API_KEY": "AIza-test",
            },
            clear=True,
        ):
            result = get_available_providers()
            assert len(result) == 3
            assert ProviderType.OPENAI in result
            assert ProviderType.ANTHROPIC in result
            assert ProviderType.GOOGLE in result


class TestDetectProviderFromModel:
    """Tests for detect_provider_from_model function."""

    @pytest.mark.unit
    def test_gpt_models(self) -> None:
        """Test detection of GPT models."""
        assert detect_provider_from_model("gpt-4o") == ProviderType.OPENAI
        assert detect_provider_from_model("gpt-4-turbo") == ProviderType.OPENAI
        assert detect_provider_from_model("gpt-3.5-turbo") == ProviderType.OPENAI

    @pytest.mark.unit
    def test_o_series_models(self) -> None:
        """Test detection of O-series reasoning models."""
        assert detect_provider_from_model("o1") == ProviderType.OPENAI
        assert detect_provider_from_model("o1-mini") == ProviderType.OPENAI
        assert detect_provider_from_model("o3") == ProviderType.OPENAI
        assert detect_provider_from_model("o4-mini") == ProviderType.OPENAI

    @pytest.mark.unit
    def test_claude_models(self) -> None:
        """Test detection of Claude models."""
        assert detect_provider_from_model("claude-3-opus") == ProviderType.ANTHROPIC
        assert detect_provider_from_model("claude-3-5-sonnet") == ProviderType.ANTHROPIC
        assert detect_provider_from_model("claude-2") == ProviderType.ANTHROPIC

    @pytest.mark.unit
    def test_gemini_models(self) -> None:
        """Test detection of Gemini models."""
        assert detect_provider_from_model("gemini-pro") == ProviderType.GOOGLE
        assert detect_provider_from_model("gemini-1.5-pro") == ProviderType.GOOGLE
        assert detect_provider_from_model("models/gemini-pro") == ProviderType.GOOGLE

    @pytest.mark.unit
    def test_openrouter_format(self) -> None:
        """Test detection of OpenRouter provider/model format."""
        assert detect_provider_from_model("openai/gpt-4") == ProviderType.OPENROUTER
        assert (
            detect_provider_from_model("anthropic/claude-3") == ProviderType.OPENROUTER
        )
        assert detect_provider_from_model("meta/llama-3") == ProviderType.OPENROUTER

    @pytest.mark.unit
    def test_open_source_models(self) -> None:
        """Test detection of open source models (via OpenRouter)."""
        assert detect_provider_from_model("llama-3-70b") == ProviderType.OPENROUTER
        assert detect_provider_from_model("mistral-large") == ProviderType.OPENROUTER
        assert detect_provider_from_model("mixtral-8x7b") == ProviderType.OPENROUTER
        assert detect_provider_from_model("deepseek-coder") == ProviderType.OPENROUTER

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert detect_provider_from_model("GPT-4o") == ProviderType.OPENAI
        assert detect_provider_from_model("CLAUDE-3") == ProviderType.ANTHROPIC
        assert detect_provider_from_model("GEMINI-pro") == ProviderType.GOOGLE

    @pytest.mark.unit
    def test_unknown_defaults_to_openai(self) -> None:
        """Test that unknown models default to OpenAI."""
        assert detect_provider_from_model("unknown-model") == ProviderType.OPENAI


class TestGetApiKeyForProvider:
    """Tests for get_api_key_for_provider function."""

    @pytest.mark.unit
    def test_explicit_key_returned(self) -> None:
        """Test that explicit API key is returned."""
        result = get_api_key_for_provider(ProviderType.OPENAI, "explicit-key")
        assert result == "explicit-key"

    @pytest.mark.unit
    @pytest.mark.usefixtures("no_api_key_remap")
    def test_env_key_used(self) -> None:
        """Test that environment variable key is used."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            result = get_api_key_for_provider(ProviderType.OPENAI)
            assert result == "env-key"

    @pytest.mark.unit
    def test_missing_key_raises(self) -> None:
        """Test that missing key raises ValueError."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="No API key found"),
        ):
            get_api_key_for_provider(ProviderType.OPENAI)

    @pytest.mark.unit
    def test_openrouter_fallback_to_openai_key(self) -> None:
        """Test OpenRouter falls back to OPENAI_API_KEY."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            result = get_api_key_for_provider(ProviderType.OPENROUTER)
            assert result == "openai-key"

    @pytest.mark.unit
    def test_anthropic_key(self) -> None:
        """Test Anthropic API key retrieval."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-key"}, clear=True):
            result = get_api_key_for_provider(ProviderType.ANTHROPIC)
            assert result == "ant-key"


class TestApiKeysConfigRemap:
    """Tests for the optional api_keys_config.yaml env-var remapping."""

    @staticmethod
    def _patched_service(mapping: dict[str, str]) -> MagicMock:
        """Build a fake ConfigService whose api-keys mapping is ``mapping``."""
        fake = MagicMock()
        fake.get_api_keys_config.return_value = mapping
        return fake

    @pytest.mark.unit
    def test_resolve_env_var_returns_override(self) -> None:
        """resolve_api_key_env_var returns the remapped env var name."""
        fake = self._patched_service({"openai": "OPENAI_API_KEY_2"})
        with patch("modules.config.service.get_config_service", return_value=fake):
            assert resolve_api_key_env_var(ProviderType.OPENAI) == "OPENAI_API_KEY_2"

    @pytest.mark.unit
    def test_resolve_env_var_default_when_no_mapping(self) -> None:
        """resolve_api_key_env_var falls back to the hardcoded default."""
        fake = self._patched_service({})
        with patch("modules.config.service.get_config_service", return_value=fake):
            assert resolve_api_key_env_var(ProviderType.OPENAI) == "OPENAI_API_KEY"

    @pytest.mark.unit
    def test_mapping_remaps_provider_env_var(self) -> None:
        """The resolver reads the key from the remapped env var."""
        fake = self._patched_service({"openai": "OPENAI_API_KEY_2"})
        with (
            patch("modules.config.service.get_config_service", return_value=fake),
            patch.dict(os.environ, {"OPENAI_API_KEY_2": "remapped-key"}, clear=True),
        ):
            result = get_api_key_for_provider(ProviderType.OPENAI)
            assert result == "remapped-key"

    @pytest.mark.unit
    def test_mapping_omitted_provider_uses_default_env_var(self) -> None:
        """A provider absent from the mapping uses its default env var."""
        fake = self._patched_service({"anthropic": "ANTHROPIC_API_KEY_2"})
        with (
            patch("modules.config.service.get_config_service", return_value=fake),
            patch.dict(os.environ, {"OPENAI_API_KEY": "default-key"}, clear=True),
        ):
            result = get_api_key_for_provider(ProviderType.OPENAI)
            assert result == "default-key"

    @pytest.mark.unit
    def test_custom_provider_reads_env_var_from_model_config(self) -> None:
        """Custom provider resolves its key via model_config's api_key_env_var."""
        fake = self._patched_service({})
        fake.get_model_config.return_value = {
            "transcription_model": {
                "custom_endpoint": {"api_key_env_var": "MY_CUSTOM_KEY"}
            }
        }
        with (
            patch("modules.config.service.get_config_service", return_value=fake),
            patch.dict(os.environ, {"MY_CUSTOM_KEY": "custom-secret"}, clear=True),
        ):
            result = get_api_key_for_provider(ProviderType.CUSTOM)
            assert result == "custom-secret"


class TestGetProvider:
    """Tests for get_provider function."""

    @pytest.mark.unit
    def test_invalid_provider_raises(self) -> None:
        """Test that invalid provider raises ValueError."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            pytest.raises(ValueError, match="Unknown provider"),
        ):
            get_provider(provider="invalid_provider", model="gpt-4")

    @pytest.mark.unit
    def test_provider_detection_openai(self) -> None:
        """Test that OpenAI models are correctly detected."""
        result = detect_provider_from_model("gpt-4o")
        assert result == ProviderType.OPENAI

    @pytest.mark.unit
    def test_provider_detection_anthropic(self) -> None:
        """Test that Anthropic models are correctly detected."""
        result = detect_provider_from_model("claude-3-opus")
        assert result == ProviderType.ANTHROPIC


class TestProviderTypeIntegration:
    """Integration tests for provider type handling."""

    @pytest.mark.unit
    def test_provider_type_from_string(self) -> None:
        """Test creating ProviderType from string."""
        assert ProviderType("openai") == ProviderType.OPENAI
        assert ProviderType("anthropic") == ProviderType.ANTHROPIC
        assert ProviderType("google") == ProviderType.GOOGLE
        assert ProviderType("openrouter") == ProviderType.OPENROUTER

    @pytest.mark.unit
    def test_provider_type_case_sensitive(self) -> None:
        """Test that ProviderType is case sensitive."""
        with pytest.raises(ValueError):
            ProviderType("OPENAI")

    @pytest.mark.unit
    def test_provider_type_value_access(self) -> None:
        """Test accessing provider type values."""
        for pt in ProviderType:
            assert isinstance(pt.value, str)
            assert len(pt.value) > 0
