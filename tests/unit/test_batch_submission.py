"""Tests for modules.core.batch_submission."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.core.batch_submission import (
    submit_batch,
    _load_system_prompt,
    _resolve_additional_context,
)
from modules.llm.batch.backends.base import BatchHandle


# ---------------------------------------------------------------------------
# _load_system_prompt
# ---------------------------------------------------------------------------

class TestLoadSystemPrompt:
    @patch("modules.config.service.get_config_service")
    @patch("modules.config.config_loader.PROJECT_ROOT")
    def test_loads_default_prompt(self, mock_root, mock_svc, tmp_path):
        prompt_dir = tmp_path / "system_prompt"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "system_prompt.txt"
        prompt_file.write_text("  Default prompt text  ", encoding="utf-8")

        mock_root.__truediv__ = MagicMock(side_effect=lambda x: tmp_path / x)
        mock_svc.return_value.get_paths_config.return_value = {"general": {}}

        result = _load_system_prompt()
        assert result == "Default prompt text"

    @patch("modules.config.service.get_config_service")
    def test_loads_override_prompt(self, mock_svc, tmp_path):
        override_file = tmp_path / "custom_prompt.txt"
        override_file.write_text("Custom prompt", encoding="utf-8")

        mock_svc.return_value.get_paths_config.return_value = {
            "general": {"transcription_prompt_path": str(override_file)}
        }

        result = _load_system_prompt()
        assert result == "Custom prompt"

    @patch("modules.config.service.get_config_service")
    @patch("modules.config.config_loader.PROJECT_ROOT")
    def test_raises_when_missing(self, mock_root, mock_svc, tmp_path):
        mock_root.__truediv__ = MagicMock(
            side_effect=lambda x: tmp_path / x
        )
        mock_svc.return_value.get_paths_config.return_value = {"general": {}}

        with pytest.raises(FileNotFoundError):
            _load_system_prompt()


# ---------------------------------------------------------------------------
# _resolve_additional_context
# ---------------------------------------------------------------------------

class TestResolveAdditionalContext:
    def test_explicit_path_exists(self, tmp_path):
        ctx_file = tmp_path / "context.txt"
        ctx_file.write_text("  My context  ", encoding="utf-8")

        user_config = MagicMock()
        user_config.additional_context_path = str(ctx_file)

        result = _resolve_additional_context(user_config, tmp_path)
        assert result == "My context"

    def test_explicit_path_not_exists(self, tmp_path):
        user_config = MagicMock()
        user_config.additional_context_path = str(tmp_path / "missing.txt")

        result = _resolve_additional_context(user_config, tmp_path)
        assert result is None

    @patch("modules.llm.context_utils.resolve_context_for_folder")
    def test_hierarchical_context_resolved(self, mock_resolve, tmp_path):
        mock_resolve.return_value = ("Resolved context", tmp_path / "ctx.md")

        user_config = MagicMock()
        user_config.additional_context_path = None
        user_config.use_hierarchical_context = True

        result = _resolve_additional_context(user_config, tmp_path)
        assert result == "Resolved context"

    @patch("modules.llm.context_utils.resolve_context_for_folder")
    def test_hierarchical_context_none(self, mock_resolve, tmp_path):
        mock_resolve.return_value = (None, None)

        user_config = MagicMock()
        user_config.additional_context_path = None
        user_config.use_hierarchical_context = True

        result = _resolve_additional_context(user_config, tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# submit_batch
# ---------------------------------------------------------------------------

class TestSubmitBatch:
    @pytest.mark.asyncio
    @patch("modules.core.batch_submission.supports_batch", return_value=False)
    async def test_unsupported_provider_returns_none(self, mock_supports, tmp_path):
        result = await submit_batch(
            image_files=[tmp_path / "img.png"],
            temp_jsonl_path=tmp_path / "temp.jsonl",
            parent_folder=tmp_path,
            source_name="test",
            model_config={"transcription_model": {"provider": "unknown"}},
            user_config=MagicMock(),
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("modules.core.batch_submission.supports_batch", return_value=True)
    @patch("modules.core.batch_submission.get_batch_chunk_size", return_value=50)
    @patch("modules.core.batch_submission._load_system_prompt", return_value="prompt")
    @patch("modules.core.batch_submission._resolve_additional_context", return_value=None)
    @patch("modules.core.batch_submission.get_batch_backend")
    async def test_submission_failure_returns_none(
        self, mock_backend, mock_ctx, mock_prompt, mock_chunk, mock_supports, tmp_path
    ):
        mock_backend.return_value.submit_batch.side_effect = RuntimeError("fail")

        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        img = tmp_path / "img.png"
        img.write_bytes(b"")

        user_config = MagicMock()
        user_config.selected_schema_path = None
        result = await submit_batch(
            image_files=[img],
            temp_jsonl_path=jsonl_path,
            parent_folder=tmp_path,
            source_name="test",
            model_config={"transcription_model": {"provider": "openai"}},
            user_config=user_config,
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("modules.core.batch_submission.supports_batch", return_value=True)
    @patch("modules.core.batch_submission.get_batch_chunk_size", return_value=50)
    @patch("modules.core.batch_submission._load_system_prompt", return_value="prompt")
    @patch("modules.core.batch_submission._resolve_additional_context", return_value=None)
    @patch("modules.core.batch_submission.get_batch_backend")
    async def test_successful_submission_returns_handle(
        self, mock_backend_fn, mock_ctx, mock_prompt, mock_chunk, mock_supports, tmp_path
    ):
        mock_handle = BatchHandle(batch_id="batch_123", provider="openai")
        mock_backend = MagicMock()
        mock_backend.submit_batch.return_value = mock_handle
        mock_backend_fn.return_value = mock_backend

        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        img = tmp_path / "img.png"
        img.write_bytes(b"")

        user_config = MagicMock()
        user_config.selected_schema_path = None
        result = await submit_batch(
            image_files=[img],
            temp_jsonl_path=jsonl_path,
            parent_folder=tmp_path,
            source_name="test",
            model_config={"transcription_model": {"provider": "openai"}},
            user_config=user_config,
        )
        assert result is not None
        assert result.batch_id == "batch_123"
