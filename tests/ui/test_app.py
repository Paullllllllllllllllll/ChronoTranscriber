"""The ui subcommand: terminal check, settings, task switches, decisions, Ctrl+C."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import io
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import chronotranscriber.run as core
from chronotranscriber.common.testing import KEEP, ScriptedPrompter
from chronotranscriber.common.wizard import BACK, Review
from chronotranscriber.ui import app
from chronotranscriber.ui.session import TASKS, RunRequest, TaskFlow
from chronotranscriber.ui.transcribe import flow as transcribe_flow
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.inputs import make_scanned_pdf
from tests.characterization.seams import replace
from tests.ui.conftest import DUMMY_KEYS, SettingsWriter

INTERRUPT = object()
_ASYNC_SLEEP = asyncio.sleep


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Interrupting(ScriptedPrompter):
    """Raise KeyboardInterrupt where the script holds ``INTERRUPT``."""

    def _take(self, message: str) -> Any:
        answer = super()._take(message)
        if answer is INTERRUPT:
            raise KeyboardInterrupt
        return answer


async def _yield_only(_delay: float, result: Any = None) -> Any:
    await _ASYNC_SLEEP(0)
    return result


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Install the characterization FakeLLM, the dummy keys and instant sleeps."""
    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(asyncio, "sleep", _yield_only)
    return controller


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """A two-page scanned PDF in its own folder."""
    return make_scanned_pdf(tmp_path / "inputs" / "book.pdf", pages=2)


@pytest.fixture
def settings_path(write_settings: SettingsWriter) -> Path:
    """A settings file without defaults."""
    return write_settings()


def _transcribe(pdf: Path, out: Path, *review: Any) -> list[Any]:
    """Answers of an LLM run on *pdf* into *out*, then the *review* answers."""
    return [
        KEEP,  # task
        str(pdf),
        "pdf",
        "llm",
        "folder",
        str(out),
        KEEP,  # layout
        ["txt", "md"],
        *[KEEP] * 4,  # provider, model, reasoning, model parameters
        *[KEEP] * 4,  # response format, schema, prompt, context
        KEEP,  # items without a sidecar
        KEEP,  # context image
        KEEP,  # page images
        KEEP,  # decision pre-pass
        KEEP,  # pages
        KEEP,  # submission
        [],  # run flags
        KEEP,  # concurrency
        *review,
    ]


def _console() -> Console:
    return Console(file=io.StringIO(), width=120, color_system=None)


def _text(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def _launch(
    answers: Sequence[Any],
    settings_path: Path | None,
    *,
    prompter: ScriptedPrompter | None = None,
    **kwargs: Any,
) -> tuple[int, Console, ScriptedPrompter]:
    console = _console()
    scripted = prompter if prompter is not None else ScriptedPrompter(answers)
    with scripted:
        code = app.run_app(
            settings_path,
            prompter=scripted,
            console=console,
            require_tty=False,
            environ=dict(DUMMY_KEYS),
            **kwargs,
        )
    return code, console, scripted


def _no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(*args: Any, **kwargs: Any) -> core.RunResult:
        raise AssertionError("core.run must not be called")

    monkeypatch.setattr(core, "run", run)


@pytest.mark.parametrize("stream", ["stdin", "stdout"])
def test_a_non_interactive_stream_is_refused_before_reading_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    settings_path: Path,
    stream: str,
) -> None:
    loads: list[Any] = []
    monkeypatch.setattr(app, "load", lambda *a, **kw: loads.append(a))
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(sys, stream, io.StringIO())
    code = app.execute(argparse.Namespace(command="ui", settings=settings_path))
    assert code == 2
    assert loads == []
    err = capsys.readouterr().err
    assert "'chronotranscriber transcribe'" in err
    assert "chronotranscriber transcribe --help" in err


def test_a_settings_error_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    code, console, prompter = _launch([], missing)
    assert code == 2
    assert f"Settings file not found: {missing}" in _text(console)
    assert prompter.lines == []


def test_back_at_the_task_step_and_quit_exits_0(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    code, _console, prompter = _launch([BACK, "quit"], settings_path)
    assert code == 0
    assert "ChronoTranscriber | new transcription   step 1" in prompter.transcript
    assert "? Quit without running?" in prompter.lines


def _headers(prompter: ScriptedPrompter) -> list[str]:
    return [line for line in prompter.lines if line.startswith("ChronoTranscriber | ")]


@pytest.mark.parametrize("task", [name for name in TASKS if name != app.FIRST_TASK])
def test_a_task_switch_starts_at_the_first_step_of_the_new_task(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    _no_run(monkeypatch)
    code, _console, prompter = _launch([task, BACK, BACK, "quit"], settings_path)
    assert code == 0
    title = TASKS[task]
    headers = _headers(prompter)
    assert headers[0] == "ChronoTranscriber | new transcription   step 1"
    assert headers[1] == f"ChronoTranscriber | {title}   step 2"
    assert headers[2] == f"ChronoTranscriber | {title}   step 1"
    asked = [index for index, line in enumerate(prompter.lines) if line == "? Task:"]
    assert len(asked) == 2
    assert prompter.lines[asked[1] + 1 + list(TASKS).index(task)] == f" > {title}"
    assert "? Quit without running?" in prompter.lines


def test_edit_a_step_on_the_review_switches_the_task(
    settings_path: Path, pdf: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    answers = _transcribe(pdf, tmp_path / "out", "edit", "task", "repair")
    code, _console, prompter = _launch([*answers, BACK, BACK, "quit"], settings_path)
    assert code == 0
    assert "Edit which step?" in prompter.transcript
    assert (
        _headers(prompter)[-2] == "ChronoTranscriber | repair a transcription   step 2"
    )
    assert fake_llm.calls == []


@pytest.mark.usefixtures("fake_llm")
def test_quit_on_the_review_exits_0(
    settings_path: Path, pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    out = tmp_path / "out"
    code, _console, _prompter = _launch(_transcribe(pdf, out, "quit"), settings_path)
    assert code == 0
    assert not out.exists()


@pytest.mark.usefixtures("fake_llm")
def test_copy_prints_the_command_and_copies_it(
    settings_path: Path, pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    copied: list[tuple[list[str], bytes]] = []

    def clipboard(argv: Sequence[str], data: bytes) -> None:
        copied.append((list(argv), data))

    stdout = io.StringIO()
    out = tmp_path / "out"
    code, console, _prompter = _launch(
        _transcribe(pdf, out, "copy"),
        settings_path,
        stdout=stdout,
        clipboard=clipboard,
    )
    assert code == 0
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("chronotranscriber transcribe --input ")
    assert str(pdf) in line
    assert line.endswith(f"--settings {settings_path}")
    if sys.platform in ("win32", "darwin"):
        assert len(copied) == 1
        encoding = "utf-16" if sys.platform == "win32" else "utf-8"
        assert copied[0][1].decode(encoding) == line
        assert "Copied the command to the clipboard." in _text(console)
    assert not out.exists()


def _stub(flow: TaskFlow, run: Callable[[RunRequest], int]) -> TaskFlow:
    """Return *flow* without steps, with an empty review and *run*."""
    return dataclasses.replace(
        flow,
        steps=lambda _session: [],
        review=lambda _state, _session: Review(()),
        command=lambda _state, _session: [*flow.program],
        run=run,
    )


def test_the_run_receives_the_ui_arguments(
    settings_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[RunRequest] = []

    def record(request: RunRequest) -> int:
        requests.append(request)
        return 1

    check_flow = app.FLOWS["batch-check"]
    monkeypatch.setitem(app.FLOWS, "batch-check", lambda: _stub(check_flow(), record))
    args = argparse.Namespace(command="ui", settings=settings_path)
    code, console, _prompter = _launch(["batch-check", "run"], settings_path, args=args)
    assert code == 1
    assert len(requests) == 1
    assert requests[0].args is args
    assert requests[0].console is console
    assert requests[0].session.settings_path == settings_path


def test_ctrl_c_in_the_wizard_exits_130(
    settings_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    prompter = _Interrupting([KEEP, str(pdf), INTERRUPT])
    code, console, _prompter = _launch([], settings_path, prompter=prompter)
    assert code == 130
    assert "Interrupted by the user." in _text(console)


@pytest.mark.usefixtures("fake_llm")
def test_ctrl_c_during_the_run_exits_130(
    settings_path: Path, pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run(*args: Any, **kwargs: Any) -> core.RunResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(core, "run", run)
    answers = _transcribe(pdf, tmp_path / "out", "run")
    code, console, _prompter = _launch(answers, settings_path)
    assert code == 130
    assert "Interrupted by the user." in _text(console)


def test_an_unexpected_error_is_logged_and_exits_1(
    settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(_request: RunRequest) -> int:
        raise RuntimeError("boom")

    monkeypatch.setitem(
        app.FLOWS, "transcribe", lambda: _stub(transcribe_flow(), broken)
    )
    with caplog.at_level("ERROR", logger=app.__name__):
        code, console, _prompter = _launch([KEEP, "run"], settings_path)
    assert code == 1
    assert "Unexpected error: boom" in _text(console)
    assert "Critical error in the guided run: boom" in caplog.text
