"""Unit tests for modules/llm/providers/factory.py.

Tests provider factory for dynamic LLM provider selection.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from modules.llm.providers.factory import (
    ProviderType,
    get_available_providers,
    detect_provider_from_model,
    get_api_key_for_provider,
    get_provider,
)


class TestProviderType:
    """Tests for ProviderType enum."""
    
    @pytest.mark.unit
    def test_all_providers_defined(self):
        """Test that all expected providers are defined."""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.GOOGLE.value == "google"
        assert ProviderType.OPENROUTER.value == "openrouter"
    
    @pytest.mark.unit
    def test_provider_count(self):
        """Test expected number of providers."""
        assert len(ProviderType) == 4


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""
    
    @pytest.mark.unit
    def test_no_keys_returns_empty(self):
        """Test that no API keys returns empty list."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_available_providers()
            assert result == []
    
    @pytest.mark.unit
    def test_openai_key_available(self):
        """Test detection of OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = get_available_providers()
            assert ProviderType.OPENAI in result
    
    @pytest.mark.unit
    def test_anthropic_key_available(self):
        """Test detection of Anthropic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            result = get_available_providers()
            assert ProviderType.ANTHROPIC in result
    
    @pytest.mark.unit
    def test_multiple_keys_available(self):
        """Test detection of multiple API keys."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "GOOGLE_API_KEY": "AIza-test",
        }, clear=True):
            result = get_available_providers()
            assert len(result) == 3
            assert ProviderType.OPENAI in result
            assert ProviderType.ANTHROPIC in result
            assert ProviderType.GOOGLE in result


class TestDetectProviderFromModel:
    """Tests for detect_provider_from_model function."""
    
    @pytest.mark.unit
    def test_gpt_models(self):
        """Test detection of GPT models."""
        assert detect_provider_from_model("gpt-4o") == ProviderType.OPENAI
        assert detect_provider_from_model("gpt-4-turbo") == ProviderType.OPENAI
        assert detect_provider_from_model("gpt-3.5-turbo") == ProviderType.OPENAI
    
    @pytest.mark.unit
    def test_o_series_models(self):
        """Test detection of O-series reasoning models."""
        assert detect_provider_from_model("o1") == ProviderType.OPENAI
        assert detect_provider_from_model("o1-mini") == ProviderType.OPENAI
        assert detect_provider_from_model("o3") == ProviderType.OPENAI
        assert detect_provider_from_model("o4-mini") == ProviderType.OPENAI
    
    @pytest.mark.unit
    def test_claude_models(self):
        """Test detection of Claude models."""
        assert detect_provider_from_model("claude-3-opus") == ProviderType.ANTHROPIC
        assert detect_provider_from_model("claude-3-5-sonnet") == ProviderType.ANTHROPIC
        assert detect_provider_from_model("claude-2") == ProviderType.ANTHROPIC
    
    @pytest.mark.unit
    def test_gemini_models(self):
        """Test detection of Gemini models."""
        assert detect_provider_from_model("gemini-pro") == ProviderType.GOOGLE
        assert detect_provider_from_model("gemini-1.5-pro") == ProviderType.GOOGLE
        assert detect_provider_from_model("models/gemini-pro") == ProviderType.GOOGLE
    
    @pytest.mark.unit
    def test_openrouter_format(self):
        """Test detection of OpenRouter provider/model format."""
        assert detect_provider_from_model("openai/gpt-4") == ProviderType.OPENROUTER
        assert detect_provider_from_model("anthropic/claude-3") == ProviderType.OPENROUTER
        assert detect_provider_from_model("meta/llama-3") == ProviderType.OPENROUTER
    
    @pytest.mark.unit
    def test_open_source_models(self):
        """Test detection of open source models (via OpenRouter)."""
        assert detect_provider_from_model("llama-3-70b") == ProviderType.OPENROUTER
        assert detect_provider_from_model("mistral-large") == ProviderType.OPENROUTER
        assert detect_provider_from_model("mixtral-8x7b") == ProviderType.OPENROUTER
        assert detect_provider_from_model("deepseek-coder") == ProviderType.OPENROUTER
    
    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert detect_provider_from_model("GPT-4o") == ProviderType.OPENAI
        assert detect_provider_from_model("CLAUDE-3") == ProviderType.ANTHROPIC
        assert detect_provider_from_model("GEMINI-pro") == ProviderType.GOOGLE
    
    @pytest.mark.unit
    def test_unknown_defaults_to_openai(self):
        """Test that unknown models default to OpenAI."""
        assert detect_provider_from_model("unknown-model") == ProviderType.OPENAI


class TestGetApiKeyForProvider:
    """Tests for get_api_key_for_provider function."""
    
    @pytest.mark.unit
    def test_explicit_key_returned(self):
        """Test that explicit API key is returned."""
        result = get_api_key_for_provider(ProviderType.OPENAI, "explicit-key")
        assert result == "explicit-key"
    
    @pytest.mark.unit
    def test_env_key_used(self):
        """Test that environment variable key is used."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            result = get_api_key_for_provider(ProviderType.OPENAI)
            assert result == "env-key"
    
    @pytest.mark.unit
    def test_missing_key_raises(self):
        """Test that missing key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key found"):
                get_api_key_for_provider(ProviderType.OPENAI)
    
    @pytest.mark.unit
    def test_openrouter_fallback_to_openai_key(self):
        """Test OpenRouter falls back to OPENAI_API_KEY."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            result = get_api_key_for_provider(ProviderType.OPENROUTER)
            assert result == "openai-key"
    
    @pytest.mark.unit
    def test_anthropic_key(self):
        """Test Anthropic API key retrieval."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-key"}, clear=True):
            result = get_api_key_for_provider(ProviderType.ANTHROPIC)
            assert result == "ant-key"


class TestGetProvider:
    """Tests for get_provider function."""
    
    @pytest.mark.unit
    def test_invalid_provider_raises(self):
        """Test that invalid provider raises ValueError."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="Unknown provider"):
                get_provider(provider="invalid_provider", model="gpt-4")
    
    @pytest.mark.unit
    def test_provider_detection_openai(self):
        """Test that OpenAI models are correctly detected."""
        result = detect_provider_from_model("gpt-4o")
        assert result == ProviderType.OPENAI
    
    @pytest.mark.unit
    def test_provider_detection_anthropic(self):
        """Test that Anthropic models are correctly detected."""
        result = detect_provider_from_model("claude-3-opus")
        assert result == ProviderType.ANTHROPIC


class TestProviderTypeIntegration:
    """Integration tests for provider type handling."""
    
    @pytest.mark.unit
    def test_provider_type_from_string(self):
        """Test creating ProviderType from string."""
        assert ProviderType("openai") == ProviderType.OPENAI
        assert ProviderType("anthropic") == ProviderType.ANTHROPIC
        assert ProviderType("google") == ProviderType.GOOGLE
        assert ProviderType("openrouter") == ProviderType.OPENROUTER
    
    @pytest.mark.unit
    def test_provider_type_case_sensitive(self):
        """Test that ProviderType is case sensitive."""
        with pytest.raises(ValueError):
            ProviderType("OPENAI")
    
    @pytest.mark.unit
    def test_provider_type_value_access(self):
        """Test accessing provider type values."""
        for pt in ProviderType:
            assert isinstance(pt.value, str)
            assert len(pt.value) > 0
