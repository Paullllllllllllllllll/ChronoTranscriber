"""Several items open at once: ``transcribe --open-items`` under one request limit.

The scripted model answers every call after a fixed delay (a scripted error
after half of it) and counts the calls in flight; a wrapper around the run's
item step counts the items open. The oracle is that count, the exit code,
the JSON line, the requests the fake model received and the store's units,
leases and databases.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import ext, run
from chronotranscriber.common.retry import Decision
from chronotranscriber.common.testing import ScriptedChatModel
from tests.characterization.fakes import FakeLLM, Reply, permanent_error
from tests.characterization.inputs import make_born_digital_pdf
from tests.characterization.runner import RunResult, run_cli

pytestmark = pytest.mark.integration

DELAY = 0.5
"""Seconds every answer takes; a scripted error comes after half of it."""

_SLEEP = asyncio.sleep
"""The real sleep: ``run_cli`` makes ``asyncio.sleep`` only yield."""


class _Probe:
    """The calls in flight and the items open, with their peaks.

    *cancel_at* cancels the run once that many items are open; a scripted
    error waits until *fail_open* items are open, at most a few seconds.
    """

    def __init__(self, cancel_at: int | None = None) -> None:
        self.calls = self.calls_peak = 0
        self.items = self.items_peak = 0
        self.cancel_at = cancel_at
        self.fail_open = 0

    async def answer(
        self, model: ScriptedChatModel, messages: Any, kwargs: dict[str, Any]
    ) -> Any:
        self.calls += 1
        self.calls_peak = max(self.calls_peak, self.calls)
        try:
            try:
                result = model._respond(messages, kwargs)
            except Exception:
                await _SLEEP(DELAY / 2)
                for _ in range(500):
                    if self.items >= self.fail_open:
                        break
                    await _SLEEP(0.01)
                raise
            await _SLEEP(DELAY)
            return result
        finally:
            self.calls -= 1

    def opened(self) -> None:
        self.items += 1
        self.items_peak = max(self.items_peak, self.items)
        if self.cancel_at is not None and self.items >= self.cancel_at:
            for task in asyncio.all_tasks():
                if task.get_coro().__name__ == "run":  # type: ignore[union-attr]
                    task.cancel()


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _Probe:
    """Slow down the scripted model and count calls and open items."""
    watch = _Probe()

    async def agenerate(
        self: ScriptedChatModel, messages: Any, *_args: Any, **kwargs: Any
    ) -> Any:
        kwargs.pop("run_manager", None)
        return await watch.answer(self, messages, kwargs)

    item_step = run._run_item

    async def run_item(*args: Any, **kwargs: Any) -> Any:
        watch.opened()
        try:
            return await item_step(*args, **kwargs)
        finally:
            watch.items -= 1

    monkeypatch.setattr(ScriptedChatModel, "_agenerate", agenerate)
    monkeypatch.setattr(run, "_run_item", run_item)
    return watch


@contextlib.contextmanager
def _stopping() -> Iterator[None]:
    """Make every failed attempt decide that the run stops."""
    original = ext.Attempt.on_error

    def on_error(self: ext.Attempt[Any], exc: BaseException) -> Decision:
        original(self, exc)
        return Decision.STOP

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ext.Attempt, "on_error", on_error)
        yield


def _query(database: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(database)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _statuses(database: Path) -> dict[str, list[str]]:
    """Return the unit statuses of each document, by the stem of its input."""
    rows = _query(
        database,
        "SELECT d.name, u.status FROM units u JOIN documents d ON d.doc_id = u.doc_id "
        "ORDER BY d.name, u.idx",
    )
    statuses: dict[str, list[str]] = {}
    for name, status in rows:
        statuses.setdefault(name.rpartition("/")[2], []).append(status)
    return statuses


def _leases(database: Path) -> list[str | None]:
    return [row[0] for row in _query(database, "SELECT lease_run_id FROM documents")]


def _docs(root: Path, pages: dict[str, int]) -> Path:
    folder = root / "docs"
    for name, count in pages.items():
        make_born_digital_pdf(folder / name, count)
    return folder


def _transcribe(llm: FakeLLM, root: Path, source: Path, **options: Any) -> RunResult:
    return run_cli(
        "transcribe",
        llm=llm,
        root=root,
        input=source,
        output=root / "out",
        type="pdf",
        method="llm",
        **options,
    )


SIX = {f"item_{n}.pdf": 2 for n in range(1, 7)}


def test_open_items_share_the_limit_and_report_in_item_order(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path
) -> None:
    docs = _docs(tmp_path, SIX)

    result = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=3)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert probe.calls_peak == 4
    assert probe.items_peak >= 2
    assert len(result.calls) == 12
    statuses = _statuses(tmp_path / "out" / "chronotranscriber.sqlite")
    assert set(statuses) == {name.removesuffix(".pdf") for name in SIX}
    assert all(units == ["done", "done"] for units in statuses.values())
    summary = result.summary
    assert summary is not None
    assert summary["items_complete"] == 6
    outputs = [Path(path).stem for path in summary["outputs"][:-1]]
    assert outputs == [name.removesuffix(".pdf") for name in SIX]


def test_one_open_item_runs_the_items_one_after_another(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path
) -> None:
    docs = _docs(tmp_path, SIX)

    result = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=1)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert probe.calls_peak == 2
    assert probe.items_peak == 1


def test_a_stop_closes_the_open_items_and_opens_no_further_one(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path
) -> None:
    docs = _docs(tmp_path, {"a.pdf": 3, "b.pdf": 2, "c.pdf": 2})
    fake_llm.script(2, always=Reply(exception=permanent_error()))
    probe.fail_open = 2

    with _stopping():
        stopped = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=3)

    assert stopped.exit_code == 3, stopped.stdout + stopped.stderr
    summary = stopped.summary
    assert summary is not None
    assert "invalid request" in summary["stopped"]
    assert (summary["items_complete"], summary["items_failed"]) == (0, 0)
    assert probe.items_peak == 2
    assert "the open items 'a.pdf', 'b.pdf'" in stopped.stderr
    assert "1 further item(s) were left unprocessed" in stopped.stderr
    database = tmp_path / "out" / "chronotranscriber.sqlite"
    statuses = _statuses(database)
    assert set(statuses) == {"a", "b"}
    assert "failed" not in {status for units in statuses.values() for status in units}
    assert _leases(database) == [None, None]

    fake_llm.pages.clear()
    resumed = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=3)

    assert resumed.exit_code == 0, resumed.stdout + resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["items_complete"] == 3
    statuses = _statuses(database)
    assert set(statuses) == {"a", "b", "c"}
    assert {status for units in statuses.values() for status in units} == {"done"}


def test_an_item_that_opens_as_the_run_stops_keeps_its_stored_pages(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = _docs(tmp_path, {"a.pdf": 2, "b.pdf": 2})
    database = tmp_path / "out" / "chronotranscriber.sqlite"
    first = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=2)
    assert first.exit_code == 0, first.stdout + first.stderr
    held: list[str] = []
    lease = run._ItemRun.lease

    async def late(self: Any) -> Any:
        # The second item has its lease and waits here while the first stops
        # the run.
        leased = await lease(self)
        if self.planned.name == "b.pdf":
            held.append(self.planned.name)
            for _ in range(1000):
                if self.job.env.stop.reason is not None:
                    break
                await _SLEEP(0.01)
        return leased

    monkeypatch.setattr(run._ItemRun, "lease", late)
    fake_llm.script(1, always=Reply(exception=permanent_error()))
    fake_llm.script(2, always=Reply(exception=permanent_error()))

    with _stopping():
        stopped = _transcribe(
            fake_llm, tmp_path, docs, concurrency=4, open_items=2, force=True
        )

    assert stopped.exit_code == 3, stopped.stdout + stopped.stderr
    assert held == ["b.pdf"]
    assert _statuses(database)["b"] == ["done", "done"]
    assert _leases(database) == [None, None]


def test_an_item_that_fails_leaves_the_others_complete(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path
) -> None:
    docs = _docs(tmp_path, {"a.pdf": 2, "b.pdf": 3, "c.pdf": 2, "d.pdf": 2})
    fake_llm.script(2, always=Reply(exception=permanent_error()))

    result = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=4)

    assert result.exit_code == 1, result.stdout + result.stderr
    summary = result.summary
    assert summary is not None
    assert (summary["items_complete"], summary["items_failed"]) == (3, 1)
    assert probe.items_peak >= 2
    statuses = _statuses(tmp_path / "out" / "chronotranscriber.sqlite")
    assert statuses["b"] == ["done", "done", "failed"]
    assert all(statuses[name] == ["done", "done"] for name in ("a", "c", "d"))


def test_two_open_items_of_one_root_share_one_store(
    fake_llm: FakeLLM,
    probe: _Probe,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = _docs(tmp_path, {"a.pdf": 1, "b.pdf": 1})
    opened: list[Path] = []
    open_store = run.open_store

    def counted(root: Path, **kwargs: Any) -> Any:
        opened.append(root)
        return open_store(root, **kwargs)

    monkeypatch.setattr(run, "open_store", counted)

    result = _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=2)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert probe.items_peak == 2
    assert opened == [(tmp_path / "out").resolve()]
    statuses = _statuses(tmp_path / "out" / "chronotranscriber.sqlite")
    assert statuses == {"a": ["done"], "b": ["done"]}


def test_a_cancelled_run_leaves_no_lease_held(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path
) -> None:
    docs = _docs(tmp_path, {"a.pdf": 1, "b.pdf": 1, "c.pdf": 1})
    probe.cancel_at = 2

    with pytest.raises(asyncio.CancelledError):
        _transcribe(fake_llm, tmp_path, docs, concurrency=4, open_items=3)

    assert probe.items_peak == 2
    database = tmp_path / "out" / "chronotranscriber.sqlite"
    leases = _leases(database)
    assert leases
    assert set(leases) == {None}


def test_open_items_on_one_item_warns_that_it_has_no_effect(
    fake_llm: FakeLLM, probe: _Probe, tmp_path: Path
) -> None:
    docs = _docs(tmp_path, {"a.pdf": 1})

    result = _transcribe(fake_llm, tmp_path, docs, open_items=3)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert (
        "--open-items applies only to synchronous provider calls of several items "
        "and has no effect here: the run has one item to process"
    ) in result.stderr
