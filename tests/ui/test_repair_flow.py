"""The repair task of the wizard: steps, review, command and run.

Run on Windows with:

    uv --directory <repo> run pytest tests/ui/test_repair_flow.py -q

Each session is scripted through :func:`build_wizard`; the stored documents
come from the helpers of ``tests/batch/test_repair_store.py`` and the model
is their fake transcriber. No request leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronotranscriber import ext
from chronotranscriber.batch import repair as repair_module
from chronotranscriber.common.testing import KEEP, ScriptedPrompter
from chronotranscriber.common.wizard import (
    BACK,
    SETTINGS_LEGEND,
    Decision,
    Outcome,
    State,
    StepContext,
    render_review,
)
from chronotranscriber.settings import Settings
from chronotranscriber.spec import (
    REPAIR_CAUSES,
    REPAIR_OPTIONS,
    CommandSpec,
    _cause_flags,
    parse_repair,
)
from chronotranscriber.ui import repair as repair_ui
from chronotranscriber.ui.session import RunRequest, Session, absolute, build_wizard
from tests.batch.test_repair_store import (
    FakeTranscriber,
    Stored,
    page_name,
    store_document,
    unit_rows,
)
from tests.ui.conftest import SessionMaker

MODEL_KEEPS = [KEEP] * 4
"""Provider, model, reasoning effort and the recommended tuning."""
RESPONSE_KEEPS = [KEEP] * 6
"""Response format, schema, prompt, context, its fallback and context image."""


def stored(tmp_path: Path, outcomes: Sequence[str]) -> Stored:
    return asyncio.run(store_document(tmp_path, outcomes))


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> type[FakeTranscriber]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "PageTranscriber", FakeTranscriber)
    FakeTranscriber.texts = {}
    FakeTranscriber.calls = []
    FakeTranscriber.created = []
    return FakeTranscriber


def session_of(
    session: Session, answers: Sequence[Any]
) -> tuple[Outcome, State, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    wizard = build_wizard(repair_ui.flow(), session, prompter, width=100)
    with prompter:
        outcome = wizard.run()
    return outcome, wizard.state, prompter


def run(session: Session, answers: Any) -> tuple[int, str]:
    console = Console(file=io.StringIO(), width=200)
    request = RunRequest(session, answers, console, argparse.Namespace(command="ui"))
    code = repair_ui.flow().run(request)
    output = console.file.getvalue()  # type: ignore[attr-defined]
    return code, output


def script(path: Path, *failures: Any, start: str = "run") -> list[Any]:
    return [
        KEEP,
        str(path),
        *(failures or (KEEP, KEEP)),
        *MODEL_KEEPS,
        *RESPONSE_KEEPS,
        KEEP,
        KEEP,
        KEEP,
        start,
    ]


def test_the_steps_set_every_repair_option_but_json_and_settings(
    session: Session,
) -> None:
    sets = {name for step in repair_ui.flow().steps(session) for name in step.sets}
    assert sets == set(REPAIR_OPTIONS.names) - {"json", "settings"}


def test_a_scripted_session_repairs_the_failed_pages(
    tmp_path: Path, session: Session, transcriber: type[FakeTranscriber]
) -> None:
    document = stored(tmp_path, ["ok", "failed", "ok", "not_possible"])
    transcriber.texts = {page_name(1): "Fixed one", page_name(3): "Fixed three"}
    answers = [
        KEEP,
        str(document.output),
        KEEP,
        KEEP,
        BACK,
        ["api_error", "not_possible"],
        "",
        *MODEL_KEEPS,
        *RESPONSE_KEEPS,
        KEEP,
        KEEP,
        KEEP,
        "run",
    ]

    outcome, state, prompter = session_of(session, answers)

    assert outcome.action is Decision.RUN
    settings = str(absolute(session.settings_path or ""))
    assert outcome.command == (
        "chronotranscriber",
        "repair",
        "--input",
        str(document.output),
        "--errors-only",
        "--not-possible",
        "--settings",
        settings,
    )
    assert "Pages to repair: 2 (indices 1, 3) of 'doc'" in prompter.transcript
    spec = repair_ui.flow().resolve(state, session).spec
    assert parse_repair(outcome.command[2:], session.settings.defaults) == spec

    code, output = run(session, outcome.answers)

    assert code == 0
    assert [unit[3] for unit in unit_rows(document.db)] == [
        "Text of page 0",
        "Fixed one",
        "Text of page 2",
        "Fixed three",
    ]
    assert "Repair of doc.txt" in output


def ask_failures(answers: Sequence[Any]) -> tuple[State, ScriptedPrompter]:
    state = State(REPAIR_OPTIONS)
    prompter = ScriptedPrompter(answers)
    (step,) = [
        step
        for step in repair_ui.flow().steps(Session(Settings()))
        if step.name == "failures"
    ]
    with prompter:
        step.ask(StepContext(state, prompter))
    return state, prompter


@pytest.mark.parametrize(
    "causes",
    [
        list(chosen)
        for size in range(1, len(REPAIR_CAUSES) + 1)
        for chosen in combinations(REPAIR_CAUSES, size)
    ],
)
def test_the_failure_types_spell_the_cause_flags_of_the_spec(
    causes: list[str],
) -> None:
    state, _prompter = ask_failures([causes, ""])
    assert state.answers == _cause_flags(causes)
    assert repair_ui.cause_flags(causes) == _cause_flags(causes)
    assert repair_ui.selected_causes(state) == tuple(causes)


def test_an_empty_failure_choice_is_asked_again() -> None:
    state, prompter = ask_failures([[], ["no_text"], "0,2-3"])
    assert "  choose at least one failure type" in prompter.lines
    assert state.answers == {"no_text": True, "indices": "0,2-3"}


def test_the_review_marks_settings_defaults_and_previews_the_plan(
    tmp_path: Path, make_session: SessionMaker, transcriber: type[FakeTranscriber]
) -> None:
    document = stored(tmp_path, ["failed", "ok"])
    session = make_session({"reasoning_effort": "low"})

    _outcome, state, prompter = session_of(
        session, script(document.output, start="quit")
    )

    review = repair_ui.flow().review(state, session)
    text = render_review(review)
    assert SETTINGS_LEGEND in text
    assert "reasoning low (s)" in text
    lines = {line.split()[0]: line.split(None, 1)[1] for line in text.splitlines()[1:7]}
    assert lines["transcription"] == str(document.output)
    assert lines["failures"] == "errors | every selected failure"
    assert lines["run"] == "sync"
    assert review.withheld is None
    assert "Pages to repair: 1 (indices 0) of 'doc'" in review.notes[0]
    assert "--reasoning-effort low" in prompter.transcript


def test_run_is_withheld_when_the_repair_cannot_be_planned(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    document = stored(tmp_path, ["failed"])

    _outcome, state, prompter = session_of(
        session, script(document.output, start="quit")
    )

    review = repair_ui.flow().review(state, session)
    assert review.withheld is not None and "OPENAI_API_KEY" in review.withheld
    assert "   Run" not in prompter.lines
    assert "Run unavailable:" in prompter.transcript


def test_a_plan_error_exits_2_without_startup(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    code, output = run(session, {"input": tmp_path / "missing.txt"})
    assert code == 2
    assert "Input path does not exist" in output
    assert calls == []


def test_a_dry_run_starts_the_extension_and_writes_nothing(
    tmp_path: Path,
    session: Session,
    transcriber: type[FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = stored(tmp_path, ["failed", "ok"])
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    before = unit_rows(document.db)

    code, output = run(session, {"input": document.output, "dry_run": True})

    assert code == 0
    assert len(calls) == 1
    assert "[DRY RUN] 1 page(s) would be repaired." in output
    assert unit_rows(document.db) == before
    assert transcriber.calls == []


def test_a_page_that_fails_again_exits_1(
    tmp_path: Path, session: Session, transcriber: type[FakeTranscriber]
) -> None:
    document = stored(tmp_path, ["failed"])
    transcriber.texts = {page_name(0): None}
    code, _output = run(session, {"input": document.output})
    assert code == 1
