"""Tests for modules.ui.cost_display."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from modules.operations.cost_analysis import CostAnalysis, FileStats
from modules.ui.cost_display import display_analysis


def _sample_analysis() -> CostAnalysis:
    return CostAnalysis(
        file_stats=[
            FileStats(
                file_path=Path("test.jsonl"),
                model="gpt-4o",
                total_chunks=5,
                successful_chunks=4,
                failed_chunks=1,
                prompt_tokens=1000,
                cached_tokens=100,
                completion_tokens=500,
                reasoning_tokens=50,
                total_tokens=1500,
                cost_standard=0.015,
                cost_discounted=0.0075,
            )
        ],
        total_files=1,
        total_chunks=5,
        total_prompt_tokens=1000,
        total_cached_tokens=100,
        total_completion_tokens=500,
        total_reasoning_tokens=50,
        total_tokens=1500,
        total_cost_standard=0.015,
        total_cost_discounted=0.0075,
        models_used={"gpt-4o": 1},
    )


class TestDisplayAnalysisInteractive:
    @patch("modules.ui.cost_display.print_info")
    @patch("modules.ui.cost_display.print_success")
    @patch("modules.ui.cost_display.print_header")
    @patch("modules.ui.cost_display.ui_print")
    def test_interactive_mode_does_not_raise(
        self, mock_ui, mock_header, mock_success, mock_info
    ):
        analysis = _sample_analysis()
        display_analysis(analysis, interactive_mode=True)
        assert mock_info.call_count > 0

    @patch("modules.ui.cost_display.print_info")
    @patch("modules.ui.cost_display.print_success")
    @patch("modules.ui.cost_display.print_header")
    @patch("modules.ui.cost_display.ui_print")
    def test_interactive_no_reasoning_tokens(
        self, mock_ui, mock_header, mock_success, mock_info
    ):
        analysis = _sample_analysis()
        analysis.total_reasoning_tokens = 0
        display_analysis(analysis, interactive_mode=True)

    @patch("modules.ui.cost_display.print_info")
    @patch("modules.ui.cost_display.print_success")
    @patch("modules.ui.cost_display.print_header")
    @patch("modules.ui.cost_display.ui_print")
    def test_interactive_empty_file_stats(
        self, mock_ui, mock_header, mock_success, mock_info
    ):
        analysis = CostAnalysis()
        display_analysis(analysis, interactive_mode=True)


class TestDisplayAnalysisCli:
    def test_cli_mode_does_not_raise(self, capsys):
        analysis = _sample_analysis()
        display_analysis(analysis, interactive_mode=False)
        captured = capsys.readouterr()
        assert "Token Cost Analysis" in captured.out

    def test_cli_mode_no_reasoning_tokens(self, capsys):
        analysis = _sample_analysis()
        analysis.total_reasoning_tokens = 0
        display_analysis(analysis, interactive_mode=False)
        captured = capsys.readouterr()
        assert "Reasoning tokens" not in captured.out

    def test_cli_mode_with_reasoning_tokens(self, capsys):
        analysis = _sample_analysis()
        display_analysis(analysis, interactive_mode=False)
        captured = capsys.readouterr()
        assert "Reasoning tokens" in captured.out

    def test_cli_mode_empty_file_stats(self, capsys):
        analysis = CostAnalysis()
        display_analysis(analysis, interactive_mode=False)

    def test_cli_mode_shows_per_file_breakdown(self, capsys):
        analysis = _sample_analysis()
        display_analysis(analysis, interactive_mode=False)
        captured = capsys.readouterr()
        assert "Per-File Breakdown" in captured.out
        assert "test.jsonl" in captured.out
