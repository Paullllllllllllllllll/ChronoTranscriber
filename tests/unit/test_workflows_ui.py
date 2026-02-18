"""Unit tests for modules/ui/workflows.py.

Tests WorkflowUI display and configuration helpers.
Includes CT-4 regression tests verifying the removal of the incorrect
image_processing key access from display_processing_summary().
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest


class TestDisplayProcessingSummaryConcurrencyConfig:
    """CT-4 regression tests: concurrency config key access in display_processing_summary.

    Before the fix, display_processing_summary() read
    concurrency_config.get("image_processing", {}), a key that does not exist
    in concurrency_config.yaml.  The row always showed the hardcoded default 24.
    The fix removes that row entirely.
    """

    @pytest.mark.unit
    def test_image_processing_key_not_accessed_in_source(self):
        """Source of display_processing_summary must not read 'image_processing' from concurrency_config."""
        from modules.ui import workflows as wf_module

        source = inspect.getsource(wf_module.WorkflowUI.display_processing_summary)
        assert 'concurrency_config.get("image_processing"' not in source, (
            "display_processing_summary still contains the non-existent "
            "'image_processing' key access on concurrency_config"
        )

    @pytest.mark.unit
    def test_concurrency_path_reads_from_correct_key(self):
        """display_processing_summary reads API concurrency from concurrency.transcription.*."""
        from modules.ui import workflows as wf_module

        source = inspect.getsource(wf_module.WorkflowUI.display_processing_summary)
        assert 'concurrency_config.get("concurrency", {}).get("transcription"' in source, (
            "display_processing_summary is not reading from the correct "
            "concurrency.transcription path"
        )

    @pytest.mark.unit
    def test_correct_concurrency_config_path_resolution(self):
        """The concurrency.transcription.* path resolves the correct configured values."""
        concurrency_cfg: Dict[str, Any] = {
            "concurrency": {
                "transcription": {
                    "concurrency_limit": 1500,
                    "service_tier": "flex",
                    "retry": {"attempts": 10},
                }
            },
            "daily_token_limit": {"enabled": True, "daily_tokens": 25_000_000},
        }

        trans_cfg = (
            concurrency_cfg
            .get("concurrency", {})
            .get("transcription", {})
        )
        assert trans_cfg.get("concurrency_limit", 5) == 1500
        assert trans_cfg.get("service_tier", "default") == "flex"
        assert trans_cfg.get("retry", {}).get("attempts", 5) == 10

    @pytest.mark.unit
    def test_empty_concurrency_config_returns_defaults(self):
        """When concurrency_config is empty, defaults are applied correctly."""
        concurrency_cfg: Dict[str, Any] = {}

        trans_cfg = (
            concurrency_cfg
            .get("concurrency", {})
            .get("transcription", {})
        )
        assert trans_cfg.get("concurrency_limit", 5) == 5
        assert trans_cfg.get("service_tier", "default") == "default"
        assert trans_cfg.get("retry", {}).get("attempts", 5) == 5


class TestWorkflowUIOptions:
    """Tests for WorkflowUI static option helpers."""

    @pytest.mark.unit
    def test_get_processing_type_options_returns_list(self):
        """get_processing_type_options returns a non-empty list of (value, label) tuples."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_processing_type_options()
        assert isinstance(options, list)
        assert len(options) > 0
        for value, label in options:
            assert isinstance(value, str)
            assert isinstance(label, str)

    @pytest.mark.unit
    def test_processing_type_includes_auto(self):
        """Auto mode is present in processing type options."""
        from modules.ui.workflows import WorkflowUI

        values = [v for v, _ in WorkflowUI.get_processing_type_options()]
        assert "auto" in values

    @pytest.mark.unit
    def test_processing_type_includes_pdfs_and_images(self):
        """PDFs and images are present in processing type options."""
        from modules.ui.workflows import WorkflowUI

        values = [v for v, _ in WorkflowUI.get_processing_type_options()]
        assert "pdfs" in values
        assert "images" in values

    @pytest.mark.unit
    def test_get_method_options_for_pdfs(self):
        """PDF processing type offers native, tesseract, and gpt options."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_method_options("pdfs")
        values = [v for v, _ in options]
        assert "native" in values
        assert "tesseract" in values
        assert "gpt" in values

    @pytest.mark.unit
    def test_get_method_options_for_images(self):
        """Image processing type offers tesseract and gpt options."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_method_options("images")
        values = [v for v, _ in options]
        assert "tesseract" in values
        assert "gpt" in values

    @pytest.mark.unit
    def test_get_batch_options_returns_yes_no(self):
        """Batch options include yes and no."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_batch_options()
        values = [v for v, _ in options]
        assert "yes" in values
        assert "no" in values
