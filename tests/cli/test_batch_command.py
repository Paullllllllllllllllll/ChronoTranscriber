"""batch check and batch cancel on the store, through the console entry point.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_batch_command.py -q

The shipped backends run over the fake SDK clients of
``tests/batch/conftest.py``; a create that loses its response leaves an
unresolved submission. Every exit writes exactly one JSON line.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.batch.clients as clients_module
from chronotranscriber import __main__ as entry
from chronotranscriber.batch import check as check_module
from chronotranscriber.batch.submit import submit_batch
from chronotranscriber.common.batch import create_backend
from chronotranscriber.common.heartbeat import ENV_VAR
from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.settings import Settings
from chronotranscriber.spec import effective_spec
from tests.batch.conftest import (
    KEYS,
    SDK_FAKES,
    Clock,
    leased_document,
    page_images,
    run_spec,
)
from tests.cli.conftest import json_lines

PAGES = 2
ZERO_USAGE = {
    "total": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
}
# The fields of a check that rendered nothing.
NO_WORK = {
    "submitted": 0,
    "complete": 0,
    "failed_pages": 0,
    "outputs": [],
    "usage": ZERO_USAGE,
}


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build the shipped backends over one fake SDK client per provider."""
    clients = {provider: fake() for provider, fake in SDK_FAKES.items()}
    for variable in KEYS.values():
        monkeypatch.setenv(variable, f"test-{variable.lower()}")
    monkeypatch.setattr(
        clients_module, "create_client", lambda provider, key, **_: clients[provider]
    )
    monkeypatch.setattr(clients_module, "create_backend", create_backend)
    return clients


def _submit(
    root: Path,
    provider: str,
    fakes: dict[str, Any],
    *,
    lose: bool,
    formats: tuple[str, ...] | None = None,
) -> str:
    """Submit a document; *formats* stores a run spec with those formats."""
    spec = None
    if formats is not None:
        spec = effective_spec(run_spec(root, provider, formats=formats))

    async def submit() -> str:
        async with open_store(root) as store:
            await store.begin_run("transcribe", spec=spec)
            handle = await leased_document(store, provider=provider)
            outcome = await submit_batch(
                page_images(PAGES),
                document=handle,
                spec=run_spec(root, provider),
                settings=Settings(),
                schema_path=DEFAULT_SCHEMA_PATH,
            )
        (submission_id,) = outcome.submission_ids
        return submission_id

    fakes[provider].lose_create = lose
    try:
        return asyncio.run(submit())
    finally:
        fakes[provider].lose_create = False


def _status(root: Path) -> str:
    connection = sqlite3.connect(database_path(root))
    try:
        (status,) = connection.execute("SELECT status FROM submissions").fetchone()
    finally:
        connection.close()
    return str(status)


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _check(root: Path, *options: str) -> int:
    return entry.main(["batch", "check", "--input", str(root), *options, "--json"])


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "out"
    root.mkdir()
    return root


def test_an_unresolved_submission_exits_1_and_lists_its_candidates(
    fakes: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=True)

    assert _check(root) == 1

    payload, err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "check_batches",
        "finalized": 0,
        "pending": 0,
        "failed": 0,
        "unresolved": 1,
        **NO_WORK,
    }
    assert "candidate msgbatch_1" in err
    assert _status(root) == "unresolved"


def test_adopt_finalizes_and_reports_the_adoption(
    fakes: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=True)
    fakes["anthropic"].complete("msgbatch_1")

    assert _check(root, "--adopt", "msgbatch_1") == 0

    payload, _err = _one_line(capsys)
    assert payload["adopted"] is True
    assert (payload["finalized"], payload["unresolved"]) == (1, 0)
    assert _status(root) == "finalized"
    assert (root / "doc.txt").is_file()


def test_the_check_renders_the_formats_of_the_submitting_run(
    fakes: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=False, formats=("md",))
    fakes["anthropic"].complete("msgbatch_1")

    assert _check(root) == 0

    payload, _err = _one_line(capsys)
    assert (root / "doc.md").is_file()
    assert not (root / "doc.txt").exists()
    assert (payload["finalized"], payload["complete"], payload["submitted"]) == (
        1,
        1,
        0,
    )
    assert payload["outputs"] == [str(root / "doc.md")]
    assert set(payload["usage"]) == {"transcription", "total"}


def test_check_formats_override_the_submitting_run(
    fakes: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=False, formats=("md",))
    fakes["anthropic"].complete("msgbatch_1")

    assert _check(root, "--formats", "txt") == 0

    capsys.readouterr()
    assert (root / "doc.txt").is_file()
    assert not (root / "doc.md").exists()


def test_abandon_reports_the_abandonment(
    fakes: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    submission_id = _submit(root, "openai", fakes, lose=True)

    assert _check(root, "--abandon", submission_id) == 0

    payload, _err = _one_line(capsys)
    assert payload["abandoned"] is True
    assert payload["unresolved"] == 0
    assert _status(root) == "abandoned"
    assert fakes["openai"].deleted == ["file-1"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--abandon", "no-such-submission"), "No submission no-such-submission"),
        (("--adopt", "batch_9"), "No unresolved openai submission"),
        (("--adopt", "batch_9", "--abandon", "x"), "--abandon"),
    ],
)
def test_a_target_that_fits_nothing_exits_2(
    fakes: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    options: tuple[str, ...],
    message: str,
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=True)

    assert _check(root, *options) == 2

    payload, err = _one_line(capsys)
    assert payload["command"] == "check_batches"
    assert message in err
    assert _status(root) == "unresolved"


def test_a_versionless_database_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    _versionless(root)
    before = database_path(root).read_bytes()

    assert _check(tmp_path) == 2

    payload, err = _one_line(capsys)
    assert payload["unresolved"] == 0
    assert "not a chronotranscriber store" in " ".join(err.split())
    assert "migrate" not in err
    assert database_path(root).read_bytes() == before


def _versionless(root: Path) -> None:
    connection = sqlite3.connect(database_path(root))
    connection.executescript(
        "CREATE TABLE documents (name TEXT PRIMARY KEY);"
        "CREATE TABLE pages (name TEXT, idx INTEGER);"
    )
    connection.close()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Replace the clock and the sleep of batch check --wait."""
    fake = Clock()
    monkeypatch.setattr(check_module, "_clock", fake)
    monkeypatch.setattr(check_module, "_sleep", fake.sleep)
    return fake


def _waited(**fields: Any) -> dict[str, Any]:
    """Return the --wait summary with zero counts, updated by *fields*."""
    return {
        "dry_run": False,
        "command": "check_batches",
        "finalized": 0,
        "pending": 0,
        "failed": 0,
        "unresolved": 0,
        **NO_WORK,
        "polls": 0,
        "waited_seconds": 0,
        "stopped": None,
        **fields,
    }


def test_a_wait_that_settles_exits_0_with_the_wait_fields(
    fakes: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clock: Clock,
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=False)
    clock.actions[1] = lambda: fakes["anthropic"].complete("msgbatch_1")

    assert _check(root, "--wait") == 0

    payload, err = _one_line(capsys)
    assert set(payload.pop("usage")) == {"transcription", "total"}
    expected = _waited(
        finalized=1,
        complete=1,
        outputs=[str(root / "doc.txt")],
        polls=2,
        waited_seconds=900,
        stopped="settled",
    )
    del expected["usage"]
    assert payload == expected
    assert re.search(
        r"\[INFO\] Waiting for batches: 1 pending; next check at \d\d:\d\d "
        r"\(check 2; 0\.00 of 50 h\)\.",
        err,
    )
    assert "Stopped after 2 checks and 0.25 h: settled." in err
    assert err.count("not finished yet") == 1
    assert "Waiting until" not in err
    assert (root / "doc.txt").is_file()


def test_a_wait_writes_the_heartbeat_file_before_it_sleeps(
    fakes: dict[str, Any],
    tmp_path: Path,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    _submit(root, "anthropic", fakes, lose=False)
    clock.actions[1] = lambda: fakes["anthropic"].complete("msgbatch_1")
    path = tmp_path / "job.heartbeat.json"
    monkeypatch.setenv(ENV_VAR, str(path))

    assert _check(root, "--wait") == 0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["units_done"], data["units_total"]) == (0, 1)
    assert (data["unit_label"], data["eta_seconds"]) == ("batches", None)


def test_a_wait_past_its_hours_exits_1_as_a_timeout(
    fakes: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clock: Clock,
) -> None:
    root = _root(tmp_path)
    _submit(root, "openai", fakes, lose=False)

    code = _check(root, "--wait", "--poll-minutes", "20", "--wait-hours", "0.5")

    assert code == 1
    payload, err = _one_line(capsys)
    assert payload == _waited(
        pending=1, polls=3, waited_seconds=1800, stopped="timeout"
    )
    assert clock.sleeps == [1200, 600]
    assert err.count("not finished yet") == 2
    assert _status(root) == "submitted"


def test_a_refused_database_in_a_wait_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], clock: Clock
) -> None:
    _versionless(_root(tmp_path))

    assert _check(tmp_path, "--wait") == 2

    payload, err = _one_line(capsys)
    assert payload == _waited(polls=1, stopped="settled")
    assert "not a chronotranscriber store" in " ".join(err.split())


@pytest.mark.parametrize("option", ["--poll-minutes", "--wait-hours"])
def test_a_wait_value_without_wait_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], option: str
) -> None:
    assert _check(_root(tmp_path), option, "5") == 2

    payload, err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "check_batches",
        "finalized": 0,
        "pending": 0,
        "failed": 0,
        "unresolved": 0,
        **NO_WORK,
    }
    assert f"{option} needs --wait" in err


def test_an_interrupted_wait_exits_130_with_the_checks_so_far(
    fakes: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    _submit(root, "openai", fakes, lose=False)

    async def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(check_module, "_sleep", interrupted)

    assert _check(root, "--wait") == 130

    payload, err = _one_line(capsys)
    assert payload == _waited(pending=1, polls=1, stopped="interrupted")
    assert "Stopped after 1 check and 0.00 h: interrupted." in err


def test_an_error_out_of_a_pass_ends_the_wait_with_1(
    fakes: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    _submit(root, "openai", fakes, lose=False)

    async def broken(*_args: Any) -> None:
        raise RuntimeError("disk vanished")

    monkeypatch.setattr(check_module, "_check_database", broken)

    assert _check(root, "--wait") == 1

    payload, err = _one_line(capsys)
    assert payload == _waited(stopped="error")
    assert "disk vanished" in err


def test_cancel_dir_marks_the_submissions_cancelled(
    fakes: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    _submit(root, "openai", fakes, lose=False)

    code = entry.main(["batch", "cancel", "--input", str(tmp_path), "--json"])

    assert code == 0
    payload, _err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "cancel_batches",
        "cancelled": 1,
        "failed": 0,
    }
    assert _status(root) == "cancelled"
    assert fakes["openai"].records[0].status == "cancelling"
