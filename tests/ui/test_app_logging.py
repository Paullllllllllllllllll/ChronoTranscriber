"""The ui holds the wizard's log records for the log file; a dry run writes none."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronotranscriber.common.testing import KEEP, ScriptedPrompter
from chronotranscriber.common.wizard import BACK
from chronotranscriber.settings import load
from chronotranscriber.ui import app
from tests.characterization.fakes_ocr import FakeTesseract, block_real_tesseract
from tests.characterization.inputs import make_named_images, make_scanned_pdf
from tests.ui.conftest import DUMMY_KEYS, SettingsWriter


def _launch(answers: list[Any], settings_path: Path) -> tuple[int, ScriptedPrompter]:
    console = Console(file=io.StringIO(), width=120, color_system=None)
    prompter = ScriptedPrompter(answers)
    with prompter:
        code = app.run_app(
            settings_path,
            prompter=prompter,
            console=console,
            require_tty=False,
            environ=dict(DUMMY_KEYS),
        )
    return code, prompter


def _tesseract_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], end: str
) -> list[Any]:
    """Answers of a Tesseract run on a folder holding a PDF and a loose image."""
    block_real_tesseract(monkeypatch, tmp_path)
    FakeTesseract().install(monkeypatch)
    folder = tmp_path / "inputs"
    make_scanned_pdf(folder / "book.pdf", pages=2)
    make_named_images(folder, ["loose.png"])
    return [
        KEEP,  # task
        str(folder),
        KEEP,  # subfolders
        "pdf",
        "tesseract",
        "folder",
        str(tmp_path / "out"),
        KEEP,  # layout
        KEEP,  # formats
        KEEP,  # Tesseract configuration
        KEEP,  # preprocessing
        KEEP,  # pages
        options,  # run options
        KEEP,  # concurrency
        end,
    ]


def test_wizard_log_records_go_to_the_log_file_not_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_settings: SettingsWriter,
) -> None:
    answers = _tesseract_answers(tmp_path, monkeypatch, [], "quit")
    settings_path = write_settings()
    code, prompter = _launch(answers, settings_path)
    assert code == 0, prompter.transcript
    assert "loose image" not in capsys.readouterr().err
    log_file = load(settings_path).log_file()
    assert "Skipped 1 loose image" in log_file.read_text(encoding="utf-8")


def test_a_dry_run_writes_no_log_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_settings: SettingsWriter,
) -> None:
    answers = _tesseract_answers(tmp_path, monkeypatch, ["dry_run"], "run")
    settings_path = write_settings()
    code, prompter = _launch(answers, settings_path)
    assert code == 0, prompter.transcript
    assert "loose image" not in capsys.readouterr().err
    assert not load(settings_path).log_file().parent.exists()
    assert not (tmp_path / "out").exists()


def test_another_task_also_gets_the_log_file(
    write_settings: SettingsWriter,
) -> None:
    settings_path = write_settings()
    code, prompter = _launch(["render", BACK, BACK, "quit"], settings_path)
    assert code == 0, prompter.transcript
    log_file = load(settings_path).log_file()
    files = [
        Path(handler.baseFilename)
        for handler in logging.getLogger("chronotranscriber").handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert files == [log_file]
