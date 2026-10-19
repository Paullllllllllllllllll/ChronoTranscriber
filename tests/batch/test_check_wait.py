"""batch check --wait: passes until nothing is pending, totals from the store.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_check_wait.py -q

The shipped backends run over the fake SDK clients (``sdk`` fixture). A
:class:`~tests.batch.conftest.Clock` replaces the wait's clock and sleep and
completes batches between passes; the store keeps real time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.batch import check as check_module
from chronotranscriber.batch.check import (
    CheckStats,
    WaitLedger,
    check_until_settled,
    final_totals,
)
from chronotranscriber.batch.clients import BatchClients
from chronotranscriber.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING
from chronotranscriber.batch.submit import submit_batch
from chronotranscriber.common.polling import PollResult, Stop, WaitLimits
from chronotranscriber.common.sqlite import read
from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.settings import Settings
from chronotranscriber.spec import DEFAULT_WAIT_HOURS, resolve_batch_check
from tests.batch.conftest import Clock, leased_document, page_images, run_spec

PAGES = 2
DAY = WaitLimits(900, 26 * 3600)


class ApiError(Exception):
    """A provider error carrying an HTTP status."""

    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Replace the wait's clock and sleep with a :class:`Clock`."""
    fake = Clock()
    monkeypatch.setattr(check_module, "_clock", fake)
    monkeypatch.setattr(check_module, "_sleep", fake.sleep)
    return fake


async def submit(
    root: Path, provider: str, sdk: dict[str, Any], *, lose: bool = False
) -> str:
    """Submit the pages of document ``doc`` in *root*; return the submission id."""
    root.mkdir(parents=True, exist_ok=True)
    sdk[provider].lose_create = lose
    try:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            handle = await leased_document(store, provider=provider)
            outcome = await submit_batch(
                page_images(PAGES),
                document=handle,
                spec=run_spec(root, provider),
                settings=Settings(),
                schema_path=DEFAULT_SCHEMA_PATH,
            )
    finally:
        sdk[provider].lose_create = False
    (submission_id,) = outcome.submission_ids
    return submission_id


def status(root: Path) -> str:
    def query(connection: sqlite3.Connection) -> str:
        (value,) = connection.execute("SELECT status FROM submissions").fetchone()
        return str(value)

    return read(database_path(root), query)


async def wait(
    directory: Path, limits: WaitLimits = DAY, **options: Any
) -> tuple[PollResult[Any], WaitLedger]:
    ledger = WaitLedger()
    polled = await check_until_settled(directory, Settings(), limits, ledger, **options)
    return polled, ledger


async def test_one_wait_finalizes_and_renders_a_batch(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path, "anthropic", sdk)
    clock.actions[2] = lambda: sdk["anthropic"].complete("msgbatch_1")

    polled, ledger = await wait(tmp_path)

    assert (polled.stop, polled.passes, ledger.polls) == (Stop.SETTLED, 3, 3)
    assert clock.sleeps == [900, 900]
    assert final_totals(ledger) == CheckStats(finalized=1)
    assert status(tmp_path) == "finalized"
    assert (tmp_path / "doc.txt").is_file()


async def test_adopt_acts_on_the_first_pass_only(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path, "anthropic", sdk, lose=True)
    (lost,) = sdk["anthropic"].records
    clock.actions[1] = lambda: sdk["anthropic"].complete(lost.id)

    polled, ledger = await wait(tmp_path, adopt=lost.id)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 2)
    first, second = ledger.checks
    assert first.outcome == PENDING
    assert f"adopted batch {lost.id}" in (first.detail or "")
    assert second.outcome == FINALIZED_OUTCOME
    assert status(tmp_path) == "finalized"


async def test_an_unresolved_submission_waits_with_a_pending_one(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path / "a", "anthropic", sdk, lose=True)
    await submit(tmp_path / "b", "openai", sdk)
    clock.actions[1] = lambda: sdk["openai"].complete("batch_1")

    polled, ledger = await wait(tmp_path)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 2)
    totals = final_totals(ledger)
    assert totals == CheckStats(finalized=1, unresolved=1)
    assert totals.had_failure
    assert status(tmp_path / "a") == "unresolved"


async def test_an_unresolved_submission_alone_stops_after_one_pass(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path, "anthropic", sdk, lose=True)

    polled, ledger = await wait(tmp_path)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 1)
    assert clock.sleeps == []
    assert final_totals(ledger) == CheckStats(unresolved=1)


async def test_a_non_retryable_status_error_ends_the_wait_failed(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path, "openai", sdk)

    def rejected(batch_id: str) -> Any:
        raise ApiError(401, "invalid api key")

    sdk["openai"].batches.retrieve = rejected

    polled, ledger = await wait(tmp_path)

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 1)
    assert final_totals(ledger) == CheckStats(failed=1)
    assert status(tmp_path) == "submitted"


async def test_a_failure_the_store_shows_recovered_counts_as_finalized(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path / "a", "openai", sdk)
    await submit(tmp_path / "b", "anthropic", sdk)
    openai = sdk["openai"]
    openai.complete("batch_1")
    retrieve = openai.batches.retrieve
    errors = [ApiError(401, "invalid api key")]

    def flaky(batch_id: str) -> Any:
        if errors:
            raise errors.pop(0)
        return retrieve(batch_id)

    openai.batches.retrieve = flaky
    clock.actions[1] = lambda: sdk["anthropic"].complete("msgbatch_1")

    polled, ledger = await wait(tmp_path)

    assert polled.passes == 2
    assert [check.outcome for check in ledger.checks] == [
        FAILED,
        PENDING,
        FINALIZED_OUTCOME,
        FINALIZED_OUTCOME,
    ]
    assert final_totals(ledger) == CheckStats(finalized=2)


async def test_the_deadline_leaves_the_submission_submitted(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path, "openai", sdk)

    polled, ledger = await wait(tmp_path, WaitLimits(900, 2000))

    assert (polled.stop, polled.passes) == (Stop.TIMEOUT, 4)
    assert clock.sleeps == [900, 900, 200]
    assert final_totals(ledger) == CheckStats(pending=1)
    assert status(tmp_path) == "submitted"


def test_the_default_wait_outlasts_the_longest_batch_window() -> None:
    spec = resolve_batch_check({"wait": True}).spec
    assert spec.wait_hours == DEFAULT_WAIT_HOURS == 50
    assert spec.wait_hours > 48


async def test_an_expired_batch_ends_the_default_wait_failed(
    sdk: dict[str, Any], tmp_path: Path, clock: Clock
) -> None:
    await submit(tmp_path, "openai", sdk)

    def expire() -> None:
        sdk["openai"]._retrieve("batch_1").status = "expired"

    clock.actions[2] = expire

    polled, ledger = await wait(tmp_path, WaitLimits(900, DEFAULT_WAIT_HOURS * 3600))

    assert (polled.stop, polled.passes) == (Stop.SETTLED, 3)
    assert clock.sleeps == [900, 900]
    assert final_totals(ledger) == CheckStats(failed=1)
    assert status(tmp_path) == "failed"


async def test_a_pass_that_raises_keeps_its_reports_and_closes_the_clients(
    sdk: dict[str, Any],
    tmp_path: Path,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await submit(tmp_path / "a", "anthropic", sdk)
    await submit(tmp_path / "b", "anthropic", sdk)
    sdk["anthropic"].complete("msgbatch_1")
    real = check_module._check_database

    async def broken(database: Any, *args: Any) -> None:
        if database.root == tmp_path / "b":
            raise RuntimeError("disk vanished")
        await real(database, *args)

    closed: list[bool] = []

    class Closing(BatchClients):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr(check_module, "_check_database", broken)
    monkeypatch.setattr(check_module, "BatchClients", Closing)
    ledger = WaitLedger()

    with pytest.raises(RuntimeError, match="disk vanished"):
        await check_until_settled(tmp_path, Settings(), DAY, ledger)

    assert final_totals(ledger) == CheckStats(finalized=1)
    assert ledger.polls == 0
    assert closed == [True]
