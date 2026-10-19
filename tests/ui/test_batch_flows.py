"""The batch check and batch cancel tasks of the wizard.

Run on Windows with:

    uv --directory <repo> run pytest tests/ui/test_batch_flows.py -q

Each session is scripted through :func:`build_wizard`. The batches live in
the fake :class:`~tests.batch.conftest.World` behind the batch clients; no
request leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronotranscriber import ext
from chronotranscriber.batch.submit import submit_batch
from chronotranscriber.common.batch import BatchStatus
from chronotranscriber.common.sqlite import read
from chronotranscriber.common.testing import KEEP, ScriptedPrompter
from chronotranscriber.common.wizard import (
    BACK,
    SETTINGS_LEGEND,
    Decision,
    Outcome,
    State,
    render_review,
)
from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.settings import Settings, SettingsError
from chronotranscriber.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    CommandSpec,
    parse_batch_cancel,
    parse_batch_check,
)
from chronotranscriber.ui import batch as batch_ui
from chronotranscriber.ui.session import (
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    build_wizard,
)
from tests.batch.conftest import (
    KEYS,
    World,
    install_world,
    leased_document,
    page_images,
    run_spec,
)
from tests.ui.conftest import SessionMaker


@pytest.fixture
def batches(monkeypatch: pytest.MonkeyPatch) -> World:
    """Install a fake batch world behind the clients; set every key variable."""
    world = World()
    for variable in KEYS.values():
        monkeypatch.setenv(variable, f"test-{variable.lower()}")
    install_world(world, monkeypatch)
    return world


def submit(root: Path, world: World, *, lose: bool = False) -> str:
    """Submit a two-page document of *root* to the fake OpenAI batch API."""

    async def go() -> str:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            handle = await leased_document(store)
            outcome = await submit_batch(
                page_images(2),
                document=handle,
                spec=run_spec(root),
                settings=Settings(),
                schema_path=DEFAULT_SCHEMA_PATH,
            )
        (submission_id,) = outcome.submission_ids
        return submission_id

    root.mkdir(parents=True, exist_ok=True)
    world.lose_create = lose
    try:
        return asyncio.run(go())
    finally:
        world.lose_create = False


def status(root: Path) -> str:
    return str(
        read(
            database_path(root),
            lambda connection: connection.execute(
                "SELECT status FROM submissions"
            ).fetchone()[0],
        )
    )


def versionless(root: Path) -> Path:
    """Write a database without a schema version below *root*."""
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path(root))
    try:
        connection.executescript(
            "CREATE TABLE documents (name TEXT PRIMARY KEY);"
            "CREATE TABLE pages (name TEXT, idx INTEGER);"
        )
    finally:
        connection.close()
    return database_path(root)


def session_of(
    flow: TaskFlow, session: Session, answers: Sequence[Any]
) -> tuple[Outcome, State, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    wizard = build_wizard(flow, session, prompter, width=100)
    with prompter:
        outcome = wizard.run()
    return outcome, wizard.state, prompter


def run(
    flow: TaskFlow, session: Session, answers: Mapping[str, Any]
) -> tuple[int, str]:
    console = Console(file=io.StringIO(), width=200)
    request = RunRequest(session, answers, console, argparse.Namespace(command="ui"))
    code = flow.run(request)
    return code, console.file.getvalue()  # type: ignore[attr-defined]


def settings_words(session: Session) -> tuple[str, str]:
    return ("--settings", str(absolute(session.settings_path or "")))


@pytest.mark.parametrize(
    ("flow", "table"),
    [
        (batch_ui.check_flow(), BATCH_CHECK_OPTIONS),
        (batch_ui.cancel_flow(), BATCH_CANCEL_OPTIONS),
    ],
    ids=["check", "cancel"],
)
def test_the_steps_set_every_option_but_json_settings_and_wait(
    flow: TaskFlow, table: Any, session: Session
) -> None:
    sets = {name for step in flow.steps(session) for name in step.sets}
    wait = {"wait", "poll_minutes", "wait_hours"}
    assert sets == set(table.names) - {"json", "settings", *wait}


def test_a_check_session_finalizes_a_finished_submission(
    tmp_path: Path, session: Session, batches: World
) -> None:
    root = tmp_path / "jobs" / "out"
    submit(root, batches)
    folder = tmp_path / "jobs"
    answers = [KEEP, str(folder), BACK, str(folder), "", KEEP, KEEP, "run"]

    outcome, state, prompter = session_of(batch_ui.check_flow(), session, answers)

    assert outcome.action is Decision.RUN
    assert outcome.command == (
        "chronotranscriber",
        "batch",
        "check",
        "--input",
        str(folder),
        *settings_words(session),
    )
    assert "  found 1 database" in prompter.lines
    spec = batch_ui.check_flow().resolve(state, session).spec
    assert parse_batch_check(outcome.command[3:], session.settings.defaults) == spec

    code, output = run(batch_ui.check_flow(), session, outcome.answers)

    assert code == 0
    assert status(root) == "finalized"
    assert (root / "doc.txt").is_file()
    assert "1 finalized" in output


def test_the_check_keeps_the_run_formats_over_a_settings_formats_default(
    tmp_path: Path, make_session: SessionMaker
) -> None:
    session = make_session({"formats": ["md"]})
    answers = [KEEP, str(tmp_path), "batch_1, batch_2", KEEP, KEEP, "quit"]

    _outcome, state, prompter = session_of(batch_ui.check_flow(), session, answers)

    text = render_review(batch_ui.check_flow().review(state, session))
    assert SETTINGS_LEGEND not in text
    assert "the formats each run submitted" in text
    assert "batch_1, batch_2" in text
    assert "Only the named batches are finalized" in text
    assert "--batch-ids batch_1,batch_2" in prompter.transcript
    assert "--formats" not in prompter.transcript

    answers = [KEEP, str(tmp_path), "", KEEP, "chosen", ["md"], "quit"]
    _outcome, state, prompter = session_of(batch_ui.check_flow(), session, answers)
    assert state.value("formats") == ("md",)
    assert "--formats md" in prompter.transcript


def test_adopt_and_abandon_are_exclusive_and_spelled(
    tmp_path: Path, session: Session
) -> None:
    answers = [
        KEEP,
        str(tmp_path),
        "",
        "adopt",
        "batch_7",
        KEEP,
        "edit",
        "unresolved",
        "abandon",
        "sub-1",
        "quit",
    ]

    outcome, state, _prompter = session_of(batch_ui.check_flow(), session, answers)

    assert state.answers == {"input": tmp_path, "abandon": "sub-1"}
    spec = batch_ui.check_flow().resolve(state, session).spec
    assert (spec.adopt, spec.abandon) == (None, "sub-1")
    assert outcome.command is not None and "--abandon" in outcome.command
    assert "abandon submission sub-1" in render_review(
        batch_ui.check_flow().review(state, session)
    )


def test_a_check_of_a_failed_batch_exits_1(
    tmp_path: Path, session: Session, batches: World
) -> None:
    batches.status = BatchStatus.FAILED
    submit(tmp_path / "out", batches)

    code, output = run(batch_ui.check_flow(), session, {"input": tmp_path})

    assert code == 1
    assert "1 failed" in output


def test_a_check_target_that_fits_nothing_exits_2(
    tmp_path: Path, session: Session, batches: World
) -> None:
    submit(tmp_path / "out", batches, lose=True)
    code, output = run(
        batch_ui.check_flow(), session, {"input": tmp_path, "abandon": "no-such"}
    )
    assert code == 2
    assert "No submission no-such" in output


def test_a_check_below_a_refused_database_exits_2(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    database = versionless(tmp_path / "old")
    before = database.read_bytes()

    code, output = run(batch_ui.check_flow(), session, {"input": tmp_path})

    assert code == 2
    assert "Refused" in output
    assert database.read_bytes() == before
    assert len(calls) == 1


def test_a_check_of_a_missing_folder_exits_2(tmp_path: Path, session: Session) -> None:
    code, output = run(batch_ui.check_flow(), session, {"input": tmp_path / "gone"})
    assert code == 2
    assert "Error: Input folder does not exist" in output


def test_a_cancel_session_by_id_cancels_the_batch(
    session: Session, batches: World
) -> None:
    batches.add("openai", status=BatchStatus.IN_PROGRESS, batch_id="batch_1")
    answers = [KEEP, "dir", BACK, "ids", "batch_1", "run"]

    outcome, state, prompter = session_of(batch_ui.cancel_flow(), session, answers)

    assert outcome.command == (
        "chronotranscriber",
        "batch",
        "cancel",
        "--batch-ids",
        "batch_1",
        *settings_words(session),
    )
    assert "Cancels 1 batch with the provider" in prompter.transcript
    spec = batch_ui.cancel_flow().resolve(state, session).spec
    assert parse_batch_cancel(outcome.command[3:]) == spec

    code, output = run(batch_ui.cancel_flow(), session, outcome.answers)

    assert code == 0
    assert batches.batches["batch_1"].cancelled
    assert "Cancelled: 1; failed: 0." in output


def test_a_cancel_session_of_every_unfinished_batch(
    session: Session, batches: World
) -> None:
    first = batches.add("openai", status=BatchStatus.IN_PROGRESS)
    done = batches.add("openai", status=BatchStatus.COMPLETED)
    answers = [KEEP, "all", "run"]

    outcome, state, _prompter = session_of(batch_ui.cancel_flow(), session, answers)

    assert state.answers == {"all": True}
    assert outcome.command is not None and "--all" in outcome.command
    assert run(batch_ui.cancel_flow(), session, outcome.answers)[0] == 0
    assert batches.batches[first].cancelled
    assert not batches.batches[done].cancelled


def test_a_cancel_session_below_a_folder_cancels_its_submissions(
    tmp_path: Path, session: Session, batches: World
) -> None:
    batches.status = BatchStatus.IN_PROGRESS
    root = tmp_path / "jobs" / "out"
    submit(root, batches)
    answers = [KEEP, "dir", str(tmp_path / "jobs"), "run"]

    outcome, state, prompter = session_of(batch_ui.cancel_flow(), session, answers)

    assert state.answers == {"input": tmp_path / "jobs"}
    assert "their pages return to pending" in prompter.transcript
    assert run(batch_ui.cancel_flow(), session, outcome.answers)[0] == 0
    assert status(root) == "cancelled"


def test_a_cancel_below_a_refused_database_exits_2(
    tmp_path: Path, session: Session, batches: World
) -> None:
    database = versionless(tmp_path / "old")
    before = database.read_bytes()

    code, output = run(batch_ui.cancel_flow(), session, {"input": tmp_path})

    assert code == 2
    assert "Refused" in output
    assert database.read_bytes() == before


def test_a_failed_cancellation_exits_1(session: Session, batches: World) -> None:
    code, output = run(batch_ui.cancel_flow(), session, {"batch_ids": ("batch_x",)})
    assert code == 1
    assert "Cancelled: 0; failed: 1." in output


def test_an_extension_refusal_exits_2(
    session: Session, batches: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(spec: CommandSpec, **_: Any) -> None:
        raise SettingsError("extension refused the command")

    monkeypatch.setattr(ext, "startup", refuse)
    code, output = run(batch_ui.cancel_flow(), session, {"all": True})
    assert code == 2
    assert "extension refused the command" in output
    assert batches.calls == []
