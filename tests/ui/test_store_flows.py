"""The render and drop tasks of the wizard: steps, review, command and run.

Run on Windows with:

    uv --directory <repo> run pytest tests/ui/test_store_flows.py -q

Scripted sessions drive each flow through the wizard, with Back, to its
review; the outcome's answers then run the task as ``ui`` does. The render
run works on a store seeded with the helpers of ``tests/output``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronotranscriber.common.options import OptionTable
from chronotranscriber.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronotranscriber.common.wizard import (
    BACK,
    BACK_TITLE,
    Decision,
    Outcome,
    State,
)
from chronotranscriber.output.store import DATABASE_NAME, open_store
from chronotranscriber.spec import (
    DROP_OPTIONS,
    RENDER_OPTIONS,
    parse_drop,
    parse_render,
)
from chronotranscriber.ui.session import (
    RunRequest,
    Session,
    TaskFlow,
    build_wizard,
    new_state,
)
from chronotranscriber.ui.store import drop_flow, render_flow
from tests.cli.test_drop_command import ALPHA, seed_two_documents
from tests.output.test_ct_store import page, seed_document
from tests.ui.conftest import SessionMaker


def _versionless_tree(tmp_path: Path) -> Path:
    """Return a folder whose database holds a table but no schema version."""
    out = tmp_path / "out"
    out.mkdir()
    connection = sqlite3.connect(out / DATABASE_NAME)
    try:
        connection.execute("CREATE TABLE documents (name TEXT)")
        connection.commit()
    finally:
        connection.close()
    return out


def _seeded_store(root: Path) -> Path:
    """Seed one document with two done pages; return its txt view path."""

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            await seed_document(store, [page(0), page(1)])

    asyncio.run(main())
    return root / "doc-139d544b" / "doc.txt"


def _listing(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _session_run(
    flow: TaskFlow, session: Session, answers: list[Any]
) -> tuple[Outcome, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    with prompter:
        outcome = build_wizard(flow, session, prompter, width=120).run()
    return outcome, prompter


def _run(flow: TaskFlow, session: Session, outcome: Outcome) -> tuple[int, str]:
    file = io.StringIO()
    console = Console(file=file, force_terminal=False, width=120, color_system=None)
    request = RunRequest(session, outcome.answers, console, argparse.Namespace())
    return flow.run(request), file.getvalue()


def _start_choices(prompter: ScriptedPrompter) -> list[str]:
    """Return the entries of the last Start menu, without Back."""
    lines = prompter.lines
    start = max(index for index, line in enumerate(lines) if line == "? Start:")
    entries = []
    for line in lines[start + 1 :]:
        entry = line[3:]
        if entry == BACK_TITLE:
            return entries
        entries.append(entry)
    raise AssertionError("the Start menu has no Back entry")


def _sets(flow: TaskFlow, session: Session) -> set[str]:
    return {name for step in flow.steps(session) for name in step.sets}


@pytest.mark.parametrize(
    ("flow", "table"),
    [
        (render_flow(), RENDER_OPTIONS),
        (drop_flow(), DROP_OPTIONS),
    ],
    ids=["render", "drop"],
)
def test_every_option_but_json_is_set_by_a_step(
    flow: TaskFlow, table: OptionTable, session: Session
) -> None:
    assert _sets(flow, session) == set(table.names) - {"json"}


def test_a_render_session_with_back_writes_the_chosen_views(
    tmp_path: Path, session: Session
) -> None:
    root = tmp_path / "out"
    txt = _seeded_store(root)
    flow = render_flow()

    outcome, prompter = _session_run(
        flow,
        session,
        [
            KEEP,
            str(tmp_path),
            BACK,
            f'"{root}"',
            "choose",
            BACK,
            "choose",
            ["txt", "md"],
            KEEP,
            KEEP,
            "run",
        ],
    )

    assert outcome.action is Decision.RUN
    assert outcome.answers == {"input": root, "formats": ("txt", "md")}
    assert outcome.command is not None
    assert outcome.command[:2] == ("chronotranscriber", "render")
    assert "--settings" not in outcome.command
    spec = parse_render(list(outcome.command[2:]))
    answered = State(RENDER_OPTIONS, {}, outcome.answers)
    assert spec == flow.resolve(answered, session).spec
    assert (spec.input, spec.formats) == (root, ("txt", "md"))
    assert f"{root}" in prompter.transcript
    assert "1 database, 1 document" in prompter.transcript
    assert "Run" in _start_choices(prompter)

    code, shown = _run(flow, session, outcome)

    assert code == 0
    assert txt.read_text(encoding="utf-8") == "Page 1 text\nPage 2 text\n"
    md = txt.with_suffix(".md")
    assert md.read_text(encoding="utf-8") == "Page 1 text\n\nPage 2 text\n"
    assert "wrote 2 file(s)" in shown


def test_a_render_dry_run_writes_nothing(tmp_path: Path, session: Session) -> None:
    root = tmp_path / "out"
    _seeded_store(root)
    before = _listing(root)
    flow = render_flow()

    outcome, _prompter = _session_run(
        flow, session, [KEEP, str(root), KEEP, True, True, "run"]
    )
    code, shown = _run(flow, session, outcome)

    assert code == 0
    assert outcome.answers == {"input": root, "force": True, "dry_run": True}
    assert "[DRY RUN]" in shown and "would write 1 file(s)" in shown
    assert _listing(root) == before


def test_a_database_without_schema_version_withholds_run(
    tmp_path: Path, session: Session
) -> None:
    out = _versionless_tree(tmp_path)
    before = _listing(out)

    outcome, prompter = _session_run(
        render_flow(), session, [KEEP, str(out), KEEP, KEEP, KEEP, "quit"]
    )

    assert outcome.action is Decision.QUIT
    assert "Run" not in _start_choices(prompter)
    transcript = prompter.transcript
    assert "Run unavailable:" in transcript
    assert "not a chronotranscriber store" in " ".join(transcript.split())
    assert "migrate" not in transcript
    assert _listing(out) == before


def test_render_withholds_run_for_a_missing_input_or_another_file(
    tmp_path: Path, session: Session
) -> None:
    other = tmp_path / "notes.sqlite"
    other.write_bytes(b"")
    for given, reason in (
        (tmp_path / "missing", "does not exist"),
        (other, f"is not a {DATABASE_NAME} file"),
    ):
        _outcome, prompter = _session_run(
            render_flow(), session, [KEEP, str(given), KEEP, KEEP, KEEP, "quit"]
        )
        assert "Run" not in _start_choices(prompter)
        assert reason in prompter.transcript


def test_a_drop_session_previews_then_drops_the_chosen_document(
    tmp_path: Path, session: Session
) -> None:
    root = tmp_path / "out"
    alpha, beta = seed_two_documents(root)
    flow = drop_flow()

    outcome, prompter = _session_run(
        flow, session, [KEEP, str(root), "search", "alpha", KEEP, "run"]
    )

    assert outcome.answers == {"input": root, "select": "alpha"}
    assert outcome.command is not None
    assert parse_drop(list(outcome.command[2:])) == parse_drop(
        ["--input", str(root), "--select", "alpha"]
    )
    assert f"Would drop: {ALPHA}" in prompter.transcript
    assert "Would delete 1 file, keep 0." in prompter.transcript
    assert alpha.is_file()

    code, shown = _run(flow, session, outcome)

    assert code == 0
    assert f"dropped {ALPHA}" in shown
    assert not alpha.exists() and beta.is_file()


def test_a_drop_run_drops_nothing_when_the_store_changed_since_the_review(
    tmp_path: Path, session: Session
) -> None:
    root = tmp_path / "out"
    alpha, beta = seed_two_documents(root)
    flow = drop_flow()
    outcome, _prompter = _session_run(
        flow, session, [KEEP, str(root), "range", "1", KEEP, "run"]
    )

    async def add_first() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            await seed_document(store, [page(0)], name="aardvark-33333333/aardvark")

    asyncio.run(add_first())
    code, shown = _run(flow, session, outcome)

    assert code == 1
    assert "The store changed since the review" in shown
    assert alpha.is_file() and beta.is_file()


def test_drop_takes_no_settings_default_and_asks_for_a_store(
    tmp_path: Path, make_session: SessionMaker
) -> None:
    session = make_session({"all": True})
    assert new_state(session, DROP_OPTIONS).defaults == {}
    prompter = ScriptedPrompter([KEEP, str(tmp_path)])
    wizard = build_wizard(drop_flow(), session, prompter)
    with pytest.raises(ScriptError, match=f"No {DATABASE_NAME}"):
        wizard.run()
