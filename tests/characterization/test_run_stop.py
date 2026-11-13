"""A run the provider stops: ``transcribe`` and ``repair`` end with exit code 3.

The attempt's ``on_error`` is patched to return ``Decision.STOP`` for every
failed call, the way an extension decides that a provider serves no further
calls of the run; the fake LLM gives one page a failed call. The oracle is
the exit code, the JSON line, the requests the fake model received and the
store's units and leases, then a second run without the stop.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import ext, run
from chronotranscriber.common.retry import Decision
from tests.characterization.fakes import FakeLLM, Reply, permanent_error
from tests.characterization.inputs import make_born_digital_pdf, make_scanned_pdf
from tests.characterization.runner import RunResult, run_cli

pytestmark = pytest.mark.integration

STOP_PAGE = 1
"""The page whose failed call stops the run, in every document."""


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


def _units(database: Path) -> dict[tuple[str, int], str]:
    rows = _query(
        database,
        "SELECT d.name, u.idx, u.status FROM units u "
        "JOIN documents d ON d.doc_id = u.doc_id",
    )
    return {(name, idx): status for name, idx, status in rows}


def _leases(database: Path) -> list[str | None]:
    return [row[0] for row in _query(database, "SELECT lease_run_id FROM documents")]


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


def test_a_stopped_transcribe_leaves_its_units_pending_and_starts_no_item(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    docs = tmp_path / "docs"
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        make_born_digital_pdf(docs / name, 3)
    fake_llm.script(STOP_PAGE, always=Reply(exception=permanent_error()))

    with _stopping():
        stopped = _transcribe(fake_llm, tmp_path, docs, concurrency=1)

    assert stopped.exit_code == 3, stopped.stdout + stopped.stderr
    summary = stopped.summary
    assert summary is not None
    assert "invalid request" in summary["stopped"]
    assert (summary["items_complete"], summary["items_failed"]) == (0, 0)
    assert [call["page"] for call in stopped.calls] == [0, STOP_PAGE]
    assert "2 further item(s) were left unprocessed" in stopped.stderr
    database = stopped.paths(docs / "a.pdf").database
    units = _units(database)
    assert {idx for (_name, idx), status in units.items() if status == "done"} == {0}
    assert set(units.values()) == {"done", "pending"}
    assert len({name for name, _idx in units}) == 1
    assert _leases(database) == [None]
    assert stopped.output(docs / "a.pdf").exists()

    fake_llm.pages.clear()
    resumed = _transcribe(fake_llm, tmp_path, docs, concurrency=1)

    assert resumed.exit_code == 0, resumed.stdout + resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["stopped"] is None
    assert resumed.summary["items_complete"] == 3
    assert len(resumed.calls) == 2 + 3 + 3
    assert set(_units(database).values()) == {"done"}


def test_a_stop_outlasts_an_item_that_also_lost_its_lease(
    fake_llm: FakeLLM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        make_born_digital_pdf(docs / name, 3)

    async def transcribe(
        self: run._ItemRun, *_args: Any
    ) -> tuple[tuple[str, ...], str | None]:
        self.job.env.stop.set("gone")
        return (), "its lease was lost: another run took the document"

    monkeypatch.setattr(run._ItemRun, "transcribe", transcribe)

    stopped = _transcribe(fake_llm, tmp_path, docs, concurrency=1)

    assert stopped.exit_code == 3, stopped.stdout + stopped.stderr
    summary = stopped.summary
    assert summary is not None
    assert summary["stopped"] == "gone"
    assert summary["items_failed"] == 1
    assert stopped.calls == []
    assert "2 further item(s) were left unprocessed" in stopped.stderr


def test_a_stopped_repair_fails_no_page_and_a_rerun_completes(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    pdf = make_scanned_pdf(tmp_path / "docs" / "doc.pdf", 4)
    fake_llm.script(STOP_PAGE, always=Reply(exception=permanent_error()))
    fake_llm.script(2, always=Reply(exception=permanent_error()))
    first = _transcribe(fake_llm, tmp_path, pdf)
    assert first.exit_code == 1, first.stdout + first.stderr
    fake_llm.script(2, always=None)
    database = first.paths(pdf).database

    with _stopping():
        repaired = run_cli(
            "repair", llm=fake_llm, root=tmp_path, input=first.output(pdf)
        )

    assert repaired.exit_code == 3, repaired.stdout + repaired.stderr
    summary = repaired.summary
    assert summary is not None
    assert "invalid request" in summary["stopped"]
    assert summary["failed"] == 0
    assert [call["page"] for call in repaired.calls].count(STOP_PAGE) == 1
    statuses = {idx: status for (_name, idx), status in _units(database).items()}
    assert statuses[STOP_PAGE] == "pending"
    assert "failed" not in statuses.values()
    assert _leases(database) == [None]
    assert "The repair stopped" in repaired.stderr

    fake_llm.pages.clear()
    resumed = _transcribe(fake_llm, tmp_path, pdf)

    assert resumed.exit_code == 0, resumed.stdout + resumed.stderr
    assert set(_units(database).values()) == {"done"}
