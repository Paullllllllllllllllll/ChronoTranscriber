"""Unit tests for main/postprocess_transcriptions.py CLI behavior.

Covers directory file collection (legacy + modern naming) and honest exit
codes on partial failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from main.postprocess_transcriptions import (
    collect_transcription_files,
    postprocess_cli,
)


def _cli_args(input_path: Path, **overrides: Any) -> SimpleNamespace:
    """Build a Namespace with all attributes postprocess_cli reads."""
    defaults: dict[str, Any] = {
        "input": str(input_path),
        "recursive": False,
        "in_place": True,
        "output": None,
        "use_config": False,
        "merge_hyphenation": False,
        "no_collapse_spaces": False,
        "max_blank_lines": None,
        "tab_size": None,
        "wrap_width": None,
        "auto_wrap": False,
        "json_summary": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestCollectTranscriptionFiles:
    """Tests for collect_transcription_files (Finding 4)."""

    @pytest.mark.unit
    def test_single_file_passthrough(self, temp_dir: Path) -> None:
        """A file path returns itself regardless of naming."""
        f = temp_dir / "anything.txt"
        f.write_text("x", encoding="utf-8")
        assert collect_transcription_files(f) == [f]

    @pytest.mark.unit
    def test_matches_modern_naming(self, temp_dir: Path) -> None:
        """Modern '{stem}.txt' outputs are found (Finding 4 regression)."""
        modern = temp_dir / "doc.txt"
        modern.write_text("x", encoding="utf-8")
        result = collect_transcription_files(temp_dir)
        assert modern in result

    @pytest.mark.unit
    def test_matches_legacy_naming(self, temp_dir: Path) -> None:
        """Legacy '{stem}_transcription.txt' outputs are still found."""
        legacy = temp_dir / "doc_transcription.txt"
        legacy.write_text("x", encoding="utf-8")
        result = collect_transcription_files(temp_dir)
        assert legacy in result

    @pytest.mark.unit
    def test_matches_both_conventions(self, temp_dir: Path) -> None:
        """Both naming conventions are collected together."""
        modern = temp_dir / "a.txt"
        legacy = temp_dir / "b_transcription.txt"
        modern.write_text("x", encoding="utf-8")
        legacy.write_text("x", encoding="utf-8")
        result = collect_transcription_files(temp_dir)
        assert set(result) == {modern, legacy}

    @pytest.mark.unit
    def test_excludes_context_sidecars(self, temp_dir: Path) -> None:
        """Transcription-context sidecar files are not swept up."""
        good = temp_dir / "doc.txt"
        good.write_text("x", encoding="utf-8")
        (temp_dir / "doc_transcr_context.txt").write_text("c", encoding="utf-8")
        (temp_dir / "transcr_context.txt").write_text("c", encoding="utf-8")
        result = collect_transcription_files(temp_dir)
        assert result == [good]

    @pytest.mark.unit
    def test_recursive_scan(self, temp_dir: Path) -> None:
        """Recursive mode descends into subdirectories."""
        sub = temp_dir / "nested"
        sub.mkdir()
        nested = sub / "page.txt"
        nested.write_text("x", encoding="utf-8")
        assert nested in collect_transcription_files(temp_dir, recursive=True)
        assert nested not in collect_transcription_files(temp_dir, recursive=False)


class TestPostprocessCliExitCodes:
    """Tests for honest exit codes on partial failure (Finding 5)."""

    @staticmethod
    def _patched_config_service() -> MagicMock:
        fake = MagicMock()
        fake.get_image_processing_config.return_value = {"postprocessing": {}}
        return fake

    @pytest.mark.unit
    def test_in_place_returns_1_on_failure(self, temp_dir: Path) -> None:
        """In-place mode returns 1 when any file fails (was hardcoded 0)."""
        (temp_dir / "doc.txt").write_text("x", encoding="utf-8")
        args = _cli_args(temp_dir, in_place=True)

        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._patched_config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.postprocess_file",
                side_effect=RuntimeError("boom"),
            ),
        ):
            assert postprocess_cli(args) == 1

    @pytest.mark.unit
    def test_in_place_returns_0_on_success(self, temp_dir: Path) -> None:
        """In-place mode still returns 0 when all files succeed."""
        (temp_dir / "doc.txt").write_text("x", encoding="utf-8")
        args = _cli_args(temp_dir, in_place=True)

        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._patched_config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.postprocess_file",
                return_value=None,
            ),
        ):
            assert postprocess_cli(args) == 0

    @pytest.mark.unit
    def test_multi_file_output_returns_1_on_failure(self, temp_dir: Path) -> None:
        """Multi-file output mode returns 1 when a file fails (was hardcoded 0)."""
        (temp_dir / "a.txt").write_text("x", encoding="utf-8")
        (temp_dir / "b.txt").write_text("x", encoding="utf-8")
        out_dir = temp_dir / "out"
        args = _cli_args(temp_dir, in_place=False, output=str(out_dir))

        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._patched_config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.postprocess_file",
                side_effect=RuntimeError("boom"),
            ),
        ):
            assert postprocess_cli(args) == 1

    @pytest.mark.unit
    def test_json_summary_reports_failure_exit_code(
        self, temp_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The --json summary reports exit_code=1 on partial failure."""
        (temp_dir / "doc.txt").write_text("x", encoding="utf-8")
        args = _cli_args(temp_dir, in_place=True, json_summary=True)

        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._patched_config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.postprocess_file",
                side_effect=RuntimeError("boom"),
            ),
        ):
            rc = postprocess_cli(args)

        assert rc == 1
        summary_lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("{")
        ]
        payload = json.loads(summary_lines[-1])
        assert payload["exit_code"] == 1
        assert payload["files_failed"] == 1
