"""Unit tests for create_config_from_cli_args() in main/unified_transcriber.py.

Tests CLI-to-config override paths for resume_mode, output_format, and
page_range to verify that CLI flags always take precedence over config
file defaults.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_args(**overrides) -> Namespace:
    """Build a minimal Namespace mimicking parsed CLI args.

    Defaults represent the 'no flag passed' state for every optional argument.
    """
    defaults = dict(
        input=None,
        output=None,
        type="images",
        method="tesseract",
        auto=False,
        batch=False,
        schema=None,
        context=None,
        model=None,
        provider=None,
        reasoning_effort=None,
        model_verbosity=None,
        max_output_tokens=None,
        resume=None,
        force=None,
        files=None,
        recursive=False,
        output_format=None,
        pages=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_paths_config(
    *,
    resume_mode: str | None = None,
    output_format: str = "txt",
) -> dict:
    """Build a minimal paths_config dict for testing."""
    general: dict = {"output_format": output_format}
    if resume_mode is not None:
        general["resume_mode"] = resume_mode
    return {
        "general": general,
        "file_paths": {
            "Images": {"input": "", "output": ""},
            "PDFs": {"input": "", "output": ""},
            "EPUBs": {"input": "", "output": ""},
            "MOBIs": {"input": "", "output": ""},
        },
    }


def _call_create_config(args, paths_config, input_dir, output_dir):
    """Call create_config_from_cli_args with AutoSelector mocked."""
    from main.unified_transcriber import create_config_from_cli_args

    with patch("main.unified_transcriber.AutoSelector"):
        return create_config_from_cli_args(
            args, input_dir, output_dir, paths_config
        )


# =========================================================================
# Resume Mode Override Tests
# =========================================================================


class TestResumeModeOverride:
    """Tests for resume_mode precedence: CLI flag > config > hardcoded default."""

    @pytest.mark.unit
    def test_default_no_flag_no_config_key(self, temp_input_dir, temp_output_dir):
        """Neither flag nor config key -> hardcoded default 'skip'."""
        args = _make_args()
        config = _call_create_config(
            args, _make_paths_config(), temp_input_dir, temp_output_dir
        )
        assert config.resume_mode == "skip"

    @pytest.mark.unit
    def test_force_flag_sets_overwrite(self, temp_input_dir, temp_output_dir):
        """--force flag -> 'overwrite' regardless of config."""
        args = _make_args(force=True)
        config = _call_create_config(
            args, _make_paths_config(), temp_input_dir, temp_output_dir
        )
        assert config.resume_mode == "overwrite"

    @pytest.mark.unit
    def test_resume_flag_sets_skip(self, temp_input_dir, temp_output_dir):
        """--resume flag -> 'skip' regardless of config."""
        args = _make_args(resume=True)
        config = _call_create_config(
            args, _make_paths_config(), temp_input_dir, temp_output_dir
        )
        assert config.resume_mode == "skip"

    @pytest.mark.unit
    def test_config_overwrite_used_when_no_flag(
        self, temp_input_dir, temp_output_dir
    ):
        """No flag + config 'overwrite' -> 'overwrite'."""
        args = _make_args()
        paths_config = _make_paths_config(resume_mode="overwrite")
        config = _call_create_config(
            args, paths_config, temp_input_dir, temp_output_dir
        )
        assert config.resume_mode == "overwrite"

    @pytest.mark.unit
    def test_force_flag_overrides_config_skip(
        self, temp_input_dir, temp_output_dir
    ):
        """--force + config 'skip' -> 'overwrite' (CLI wins)."""
        args = _make_args(force=True)
        paths_config = _make_paths_config(resume_mode="skip")
        config = _call_create_config(
            args, paths_config, temp_input_dir, temp_output_dir
        )
        assert config.resume_mode == "overwrite"

    @pytest.mark.unit
    def test_resume_flag_overrides_config_overwrite(
        self, temp_input_dir, temp_output_dir
    ):
        """--resume + config 'overwrite' -> 'skip' (CLI wins)."""
        args = _make_args(resume=True)
        paths_config = _make_paths_config(resume_mode="overwrite")
        config = _call_create_config(
            args, paths_config, temp_input_dir, temp_output_dir
        )
        assert config.resume_mode == "skip"


# =========================================================================
# Output Format Override Tests
# =========================================================================


class TestOutputFormatOverride:
    """Tests for output_format precedence: CLI flag > config > 'txt'."""

    @pytest.mark.unit
    def test_config_default_used_when_no_flag(
        self, temp_input_dir, temp_output_dir
    ):
        """No --output-format + config 'md' -> 'md'."""
        args = _make_args()
        paths_config = _make_paths_config(output_format="md")
        config = _call_create_config(
            args, paths_config, temp_input_dir, temp_output_dir
        )
        assert config.output_format == "md"

    @pytest.mark.unit
    def test_cli_flag_overrides_config(self, temp_input_dir, temp_output_dir):
        """--output-format json + config 'txt' -> 'json' (CLI wins)."""
        args = _make_args(output_format="json")
        paths_config = _make_paths_config(output_format="txt")
        config = _call_create_config(
            args, paths_config, temp_input_dir, temp_output_dir
        )
        assert config.output_format == "json"

    @pytest.mark.unit
    def test_hardcoded_fallback_when_absent(
        self, temp_input_dir, temp_output_dir
    ):
        """No flag + no config key -> 'txt' (hardcoded fallback)."""
        args = _make_args()
        paths_config = {"general": {}, "file_paths": {}}
        config = _call_create_config(
            args, paths_config, temp_input_dir, temp_output_dir
        )
        assert config.output_format == "txt"


# =========================================================================
# Page Range Override Tests
# =========================================================================


class TestPageRangeOverride:
    """Tests for page_range CLI flag."""

    @pytest.mark.unit
    def test_pages_flag_sets_page_range(self, temp_input_dir, temp_output_dir):
        """--pages '1-5' -> page_range is set."""
        args = _make_args(pages="1-5")
        config = _call_create_config(
            args, _make_paths_config(), temp_input_dir, temp_output_dir
        )
        assert config.page_range is not None

    @pytest.mark.unit
    def test_no_pages_flag_leaves_none(self, temp_input_dir, temp_output_dir):
        """No --pages -> page_range is None."""
        args = _make_args()
        config = _call_create_config(
            args, _make_paths_config(), temp_input_dir, temp_output_dir
        )
        assert config.page_range is None
