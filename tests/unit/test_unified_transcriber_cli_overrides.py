"""Unit tests for CLI model override behavior in main/unified_transcriber.py."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.ui.core import UserConfiguration


class TestResolveModelConfigFromCLI:
    """Tests for _resolve_model_config_from_cli helper."""

    @pytest.mark.unit
    def test_applies_all_cli_model_overrides(self):
        """Model/provider/reasoning/verbosity/token overrides are applied to a copy."""
        from main.unified_transcriber import _resolve_model_config_from_cli

        base = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-5-mini",
                "max_output_tokens": 4096,
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "medium"},
            }
        }
        args = Namespace(
            model="gpt-5.2",
            provider="openai",
            max_output_tokens=65536,
            reasoning_effort="high",
            model_verbosity="verbose",
        )

        resolved, applied = _resolve_model_config_from_cli(base, args)
        tm = resolved["transcription_model"]

        assert tm["name"] == "gpt-5.2"
        assert tm["provider"] == "openai"
        assert tm["max_output_tokens"] == 65536
        assert tm["reasoning"]["effort"] == "high"
        assert tm["text"]["verbosity"] == "verbose"
        assert "model=gpt-5.2" in applied
        assert "provider=openai" in applied
        assert "max_output_tokens=65536" in applied
        assert "reasoning.effort=high" in applied
        assert "text.verbosity=verbose" in applied

        # ensure base config remains unchanged
        assert base["transcription_model"]["name"] == "gpt-5-mini"
        assert base["transcription_model"]["max_output_tokens"] == 4096

    @pytest.mark.unit
    def test_auto_detects_provider_from_model_when_provider_not_given(self):
        """Provider is inferred from --model when --provider is omitted."""
        from main.unified_transcriber import _resolve_model_config_from_cli

        base = {"transcription_model": {"provider": "openai", "name": "gpt-5-mini"}}
        args = Namespace(
            model="gemini-3-pro-preview",
            provider=None,
            max_output_tokens=None,
            reasoning_effort=None,
            model_verbosity=None,
        )

        resolved, applied = _resolve_model_config_from_cli(base, args)
        tm = resolved["transcription_model"]

        assert tm["name"] == "gemini-3-pro-preview"
        assert tm["provider"] == "google"
        assert "provider=google (auto)" in applied

    @pytest.mark.unit
    def test_rejects_non_positive_max_output_tokens(self):
        """--max-output-tokens must be positive."""
        from main.unified_transcriber import _resolve_model_config_from_cli

        base = {"transcription_model": {"provider": "openai", "name": "gpt-5-mini"}}
        args = Namespace(
            model=None,
            provider=None,
            max_output_tokens=0,
            reasoning_effort=None,
            model_verbosity=None,
        )

        with pytest.raises(ValueError, match="positive integer"):
            _resolve_model_config_from_cli(base, args)


class TestOpenTranscriberFromConfigForwarding:
    """Tests for _open_transcriber_from_config argument forwarding."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_forwards_runtime_model_settings_to_open_transcriber(self):
        """Synchronous transcriber init receives model/runtime overrides from model_config."""
        from main.unified_transcriber import _open_transcriber_from_config

        user_config = UserConfiguration()
        user_config.selected_schema_path = Path("schemas/markdown_transcription_schema.json")
        user_config.additional_context_path = None

        model_config = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-5.2",
                "max_output_tokens": 32000,
                "reasoning": {"effort": "high"},
                "text": {"verbosity": "concise"},
            }
        }

        with patch("main.unified_transcriber.open_transcriber", return_value="ctx") as mock_open:
            result = await _open_transcriber_from_config(user_config, model_config)

        assert result == "ctx"
        kwargs = mock_open.call_args.kwargs
        assert kwargs["model"] == "gpt-5.2"
        assert kwargs["provider"] == "openai"
        assert kwargs["max_output_tokens"] == 32000
        assert kwargs["reasoning_config"] == {"effort": "high"}
        assert kwargs["text_config"] == {"verbosity": "concise"}
