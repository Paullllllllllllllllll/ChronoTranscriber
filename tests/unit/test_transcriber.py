"""Unit tests for modules/llm/transcriber.py.

Tests LangChainTranscriber initialization, config loading, and parameter
forwarding. Includes CT-3 regression tests for text.verbosity extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


def _mock_config_service(
    model_cfg: Dict[str, Any],
    concurrency_cfg: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    """Build a mock ConfigService with the given model and concurrency config."""
    mock_cs = MagicMock()
    mock_cs.get_model_config.return_value = model_cfg
    mock_cs.get_concurrency_config.return_value = concurrency_cfg or {}
    mock_cs.get_paths_config.return_value = {"general": {}}
    mock_cs.get_image_processing_config.return_value = {}
    return mock_cs


from typing import Optional


def _make_transcriber(
    tmp_path: Path,
    model_cfg: Dict[str, Any],
    concurrency_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Initialize a LangChainTranscriber with mocked dependencies.

    Returns the kwargs captured by the get_provider call so tests can assert
    on which parameters were forwarded.
    """
    from modules.config.config_loader import PROJECT_ROOT

    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    prompt_path = PROJECT_ROOT / "system_prompt" / "system_prompt.txt"

    captured: Dict[str, Any] = {}

    def fake_get_provider(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    mock_cs = _mock_config_service(model_cfg, concurrency_cfg)

    with patch("modules.llm.transcriber.get_config_service", return_value=mock_cs):
        with patch("modules.llm.transcriber.get_provider", side_effect=fake_get_provider):
            from modules.llm import transcriber as tr
            tr.LangChainTranscriber(
                schema_path=schema_path,
                system_prompt_path=prompt_path,
                use_hierarchical_context=False,
            )

    return captured


class TestLangChainTranscriberInit:
    """Tests for LangChainTranscriber.__init__ config loading."""

    @pytest.mark.unit
    def test_model_name_loaded_from_config(self, tmp_path: Path):
        """Model name is read from model_config.transcription_model.name."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "temperature": 0.0,
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert kwargs.get("model") == "gpt-4o"

    @pytest.mark.unit
    def test_temperature_loaded_from_config(self, tmp_path: Path):
        """temperature is read from model_config and forwarded."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "temperature": 0.7,
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert kwargs.get("temperature") == pytest.approx(0.7)

    @pytest.mark.unit
    def test_max_tokens_loaded_from_config(self, tmp_path: Path):
        """max_output_tokens is read from model_config and forwarded."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 8192,
                "temperature": 0.0,
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert kwargs.get("max_tokens") == 8192

    @pytest.mark.unit
    def test_reasoning_config_forwarded(self, tmp_path: Path):
        """reasoning config is forwarded to get_provider when present."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-5-mini",
                "max_output_tokens": 4096,
                "temperature": 0.0,
                "reasoning": {"effort": "high"},
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert kwargs.get("reasoning_config") == {"effort": "high"}

    @pytest.mark.unit
    def test_reasoning_config_absent_when_not_configured(self, tmp_path: Path):
        """reasoning_config is not forwarded when model config has no reasoning key."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "temperature": 0.0,
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert "reasoning_config" not in kwargs

    @pytest.mark.unit
    def test_service_tier_loaded_from_concurrency_config(self, tmp_path: Path):
        """service_tier is read from concurrency_config.concurrency.transcription."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "temperature": 0.0,
            }
        }
        concurrency_cfg = {
            "concurrency": {
                "transcription": {"service_tier": "flex"}
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg, concurrency_cfg)
        assert kwargs.get("service_tier") == "flex"

    @pytest.mark.unit
    def test_service_tier_absent_when_not_configured(self, tmp_path: Path):
        """service_tier is not forwarded when not set in concurrency config."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "temperature": 0.0,
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg, {})
        assert "service_tier" not in kwargs


# =============================================================================
# CT-3: text.verbosity extraction and forwarding
# =============================================================================

class TestLangChainTranscriberTextVerbosity:
    """CT-3 regression tests: text_cfg extraction and forwarding to get_provider.

    Before the fix, LangChainTranscriber.__init__() never read tm.get("text"),
    so text.verbosity was silently discarded on all code paths.
    """

    @pytest.mark.unit
    def test_text_config_forwarded_when_configured(self, tmp_path: Path):
        """text_config is passed to get_provider when model config has a text key."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-5-mini",
                "max_output_tokens": 4096,
                "temperature": 0.0,
                "text": {"verbosity": "concise"},
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert "text_config" in kwargs
        assert kwargs["text_config"] == {"verbosity": "concise"}

    @pytest.mark.unit
    def test_text_config_not_forwarded_when_absent(self, tmp_path: Path):
        """text_config is NOT added to get_provider kwargs when model config has no text key."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "temperature": 0.0,
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert "text_config" not in kwargs

    @pytest.mark.unit
    def test_text_config_not_forwarded_when_empty(self, tmp_path: Path):
        """Empty text dict is treated as falsy and not forwarded."""
        model_cfg = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-5-mini",
                "max_output_tokens": 4096,
                "temperature": 0.0,
                "text": {},
            }
        }
        kwargs = _make_transcriber(tmp_path, model_cfg)
        assert "text_config" not in kwargs

    @pytest.mark.unit
    def test_all_verbosity_levels_forwarded(self, tmp_path: Path):
        """All valid verbosity values are forwarded unchanged."""
        for verbosity in ("concise", "medium", "verbose"):
            model_cfg = {
                "transcription_model": {
                    "provider": "openai",
                    "name": "gpt-5-mini",
                    "max_output_tokens": 4096,
                    "temperature": 0.0,
                    "text": {"verbosity": verbosity},
                }
            }
            kwargs = _make_transcriber(tmp_path, model_cfg)
            assert kwargs.get("text_config", {}).get("verbosity") == verbosity
