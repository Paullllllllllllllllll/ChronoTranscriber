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
    postprocess_interactive,
)
from modules.ui.prompts import NavigationAction, PromptResult


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


def _cont(value: Any) -> PromptResult:
    """Build a CONTINUE prompt result (the only outcome once allow_back=False)."""
    return PromptResult(NavigationAction.CONTINUE, value)


class TestPostprocessInteractive:
    """Tests for the interactive flow (allow_back=False correctness fix)."""

    @staticmethod
    def _config_service() -> MagicMock:
        fake = MagicMock()
        fake.get_image_processing_config.return_value = {"postprocessing": {}}
        return fake

    @pytest.mark.unit
    def test_empty_dir_reports_reworded_message(
        self, temp_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An empty directory returns 1 with the reworded warning."""
        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.prompt_text",
                return_value=_cont(str(temp_dir)),
            ),
            patch(
                "main.postprocess_transcriptions.prompt_yes_no",
                return_value=_cont(False),
            ),
        ):
            rc = postprocess_interactive()

        assert rc == 1
        assert "No transcription .txt files found" in capsys.readouterr().out

    @pytest.mark.unit
    def test_full_flow_builds_config_without_corruption(self, temp_dir: Path) -> None:
        """A scripted single-file run passes a fully-populated config downstream.

        Previously, pressing 'b' returned a BACK result whose ``value`` was
        None and silently corrupted the config; with allow_back=False the
        selected values flow through intact.
        """
        target = temp_dir / "doc.txt"
        target.write_text("x", encoding="utf-8")

        captured: dict[str, Any] = {}

        def _fake_postprocess_file(path: Path, **kwargs: Any) -> None:
            captured.update(kwargs.get("config", {}))

        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.prompt_text",
                return_value=_cont(str(target)),
            ),
            patch(
                "main.postprocess_transcriptions.prompt_yes_no",
                # use-config-base=False, merge-hyphenation=True, proceed=True
                side_effect=[_cont(False), _cont(True), _cont(True)],
            ),
            patch(
                "main.postprocess_transcriptions.prompt_select",
                # wrap mode = "no", output mode = "in_place"
                side_effect=[_cont("no"), _cont("in_place")],
            ),
            patch(
                "main.postprocess_transcriptions.postprocess_file",
                side_effect=_fake_postprocess_file,
            ),
        ):
            rc = postprocess_interactive()

        assert rc == 0
        assert captured["enabled"] is True
        assert captured["merge_hyphenation"] is True
        assert captured["wrap_lines"] is False
        # No key was left as None by a stray BACK result.
        assert None not in captured.values()

    @pytest.mark.unit
    def test_new_dir_mirrors_input_tree(self, temp_dir: Path) -> None:
        """Item 5: 'Save to a new directory' mirrors the input tree so
        same-named files in different subdirectories do not overwrite each
        other (CT-10, matching the CLI path)."""
        input_dir = temp_dir / "in"
        (input_dir / "sub1").mkdir(parents=True)
        (input_dir / "sub2").mkdir(parents=True)
        (input_dir / "sub1" / "page.txt").write_text("a", encoding="utf-8")
        (input_dir / "sub2" / "page.txt").write_text("b", encoding="utf-8")
        output_dir = temp_dir / "out"

        seen: list[Path] = []

        def _capture(path: Path, **kwargs: Any) -> None:
            seen.append(kwargs["output_path"])

        with (
            patch(
                "main.postprocess_transcriptions.get_config_service",
                return_value=self._config_service(),
            ),
            patch(
                "main.postprocess_transcriptions.prompt_text",
                side_effect=[_cont(str(input_dir)), _cont(str(output_dir))],
            ),
            patch(
                "main.postprocess_transcriptions.prompt_yes_no",
                # recursive=True, use-config=False, merge=False, proceed=True
                side_effect=[_cont(True), _cont(False), _cont(False), _cont(True)],
            ),
            patch(
                "main.postprocess_transcriptions.prompt_select",
                # wrap mode = "no", output mode = "new_dir"
                side_effect=[_cont("no"), _cont("new_dir")],
            ),
            patch(
                "main.postprocess_transcriptions.postprocess_file",
                side_effect=_capture,
            ),
        ):
            rc = postprocess_interactive()

        assert rc == 0
        # Two distinct mirrored output paths, not one flattened collision.
        assert len(seen) == 2
        assert len(set(seen)) == 2
        rel = {p.relative_to(output_dir).as_posix() for p in seen}
        assert rel == {"sub1/page.txt", "sub2/page.txt"}


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
