"""A scripted transcribe session through the wizard and the task's run."""

from __future__ import annotations

import argparse
import asyncio
import io
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronotranscriber.common.testing import KEEP, ScriptedPrompter, WriteGuard
from chronotranscriber.common.wizard import Decision, Outcome, State
from chronotranscriber.spec import OPTIONS, parse
from chronotranscriber.ui.review import PROGRAM
from chronotranscriber.ui.session import RunRequest, Session, build_wizard
from chronotranscriber.ui.transcribe import flow
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.inputs import make_scanned_pdf
from tests.characterization.seams import replace
from tests.ui.conftest import DUMMY_KEYS

_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Only yield, except in the lease heartbeat, which keeps its interval."""
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _ASYNC_SLEEP(delay, result)
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
def two_pages(tmp_path: Path) -> Path:
    """A two-page scanned PDF in its own folder."""
    return make_scanned_pdf(tmp_path / "inputs" / "book.pdf", pages=2)


def _answers(pdf: Path, out: Path, flags: Sequence[str], end: str) -> list[Any]:
    """Answers of an LLM run on *pdf* into *out*, ending with *end*."""
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
        KEEP,  # pages
        KEEP,  # submission
        list(flags),
        KEEP,  # concurrency
        end,
    ]


def _session_run(
    session: Session, answers: Sequence[Any]
) -> tuple[Outcome, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    return outcome, prompter


def _execute(session: Session, outcome: Outcome) -> tuple[int, str]:
    screen = io.StringIO()
    console = Console(file=screen, width=120)
    request = RunRequest(session, outcome.answers, console, argparse.Namespace())
    return flow().run(request), screen.getvalue()


def _files(*folders: Path) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for folder in folders
        if folder.exists()
        for path in folder.rglob("*")
        if path.is_file()
    }


def test_a_session_runs_the_llm_on_a_two_page_pdf(
    session: Session, two_pages: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    outcome, prompter = _session_run(session, _answers(two_pages, out, [], "run"))
    assert outcome.action is Decision.RUN
    assert "Plan: 1 to process, 0 complete" in prompter.transcript
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert sorted(call["page"] for call in fake_llm.calls) == [0, 1]
    assert (out / "chronotranscriber.sqlite").is_file()
    texts = sorted(path.name for path in out.rglob("book.*"))
    assert texts == ["book.md", "book.txt"]
    assert "Run summary" in screen
    assert "1 complete, 0 failed" in screen

    again = _execute(session, outcome)
    assert again[0] == 0
    assert "Nothing to do: 1 item already complete." in again[1]


@pytest.mark.usefixtures("fake_llm")
def test_a_session_with_only_refused_items_exits_1(
    session: Session, two_pages: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    outcome, _prompter = _session_run(session, _answers(two_pages, out, [], "run"))
    first = Outcome(Decision.RUN, {**outcome.answers, "pages": "1"})
    assert _execute(session, first)[0] == 0
    changed = Outcome(Decision.RUN, {**outcome.answers, "jpeg_quality": 70})

    code, screen = _execute(session, changed)

    assert code == 1, screen
    assert "Nothing to do" not in screen
    assert "jpeg_quality" in screen


@pytest.mark.usefixtures("fake_llm")
def test_a_dry_run_session_writes_nothing(
    session: Session,
    two_pages: Path,
    tmp_path: Path,
    write_guard: WriteGuard,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    out = tmp_path / "out"
    outcome, _prompter = _session_run(
        session, _answers(two_pages, out, ["dry_run"], "run")
    )
    assert outcome.answers["dry_run"] is True
    before = _files(two_pages.parent, out)
    write_guard.start([tmp_path / "home", tmp_path / "tmp"])
    try:
        code, screen = _execute(session, outcome)
        stray = write_guard.consume()
    finally:
        write_guard.start([tmp_path_factory.getbasetemp()])
    assert code == 0, screen
    assert stray == []
    assert _files(two_pages.parent, out) == before
    assert not out.exists()
    assert "Plan: 1 to process, 0 already complete" in screen


@pytest.mark.usefixtures("fake_llm")
def test_copy_returns_the_command_of_the_session(
    session: Session, two_pages: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    outcome, prompter = _session_run(session, _answers(two_pages, out, [], "copy"))
    assert outcome.action is Decision.COPY
    assert outcome.command is not None
    words = list(outcome.command)
    assert tuple(words[:2]) == PROGRAM
    assert "Equivalent command:" in prompter.transcript
    state = State(OPTIONS, session.settings.defaults, outcome.answers)
    resolved = flow().resolve(state, session)
    assert parse(words[2:], session.settings.defaults) == resolved.spec
    assert words[words.index("--formats") + 1] == "txt,md"
    assert words[words.index("--output") + 1] == str(out)
