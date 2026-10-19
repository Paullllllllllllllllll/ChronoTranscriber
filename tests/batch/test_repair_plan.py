"""Repair planning, its errors, and the wait for repair batches.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_repair_plan.py -q
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from chronotranscriber.batch import repair as repair_module
from chronotranscriber.batch import wait as wait_module
from chronotranscriber.batch.clients import BatchClients
from chronotranscriber.batch.repair import (
    RepairPlanError,
    plan_repair,
    repair_transcription,
    run_repair,
)
from chronotranscriber.batch.wait import wait_for_batches
from chronotranscriber.common.batch import (
    BatchBackend,
    BatchHandle,
    BatchStatus,
    BatchStatusInfo,
)
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.settings import Settings
from tests.batch.conftest import Clock, FakeBackend, World
from tests.batch.test_repair_store import spec_for, store_document, tree, unit_rows
from tests.recording_events import RecordingEvents

T = TypeVar("T")


class BadRequest(Exception):
    """A provider error with a client status: not retryable."""

    status_code = 400


def contents(root: Path) -> dict[str, bytes]:
    return {
        name: (root / name).read_bytes()
        for name in tree(root)
        if (root / name).is_file()
    }


async def test_a_dry_run_plan_warns_of_the_key_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stored = await store_document(tmp_path, ["failed"])
    before = contents(tmp_path)

    plan = plan_repair(spec_for(stored.output, "dry_run"), Settings())

    assert plan.to_repair
    assert any("OPENAI_API_KEY" in warning for warning in plan.warnings)
    assert contents(tmp_path) == before
    with pytest.raises(RepairPlanError, match="OPENAI_API_KEY") as caught:
        plan_repair(spec_for(stored.output), Settings())
    assert caught.value.exit_code == ExitCode.USAGE


async def test_the_plan_resolves_the_prompt_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    context = tmp_path / "context.txt"
    context.write_text("Account book of a Basel merchant.", encoding="utf-8")
    stored = await store_document(tmp_path, ["ok", "failed"])

    plan = plan_repair(spec_for(stored.output, context=context), Settings())

    assert plan.to_repair and plan.failures == [1]
    assert plan.schema_path is not None and plan.schema_path.is_file()
    assert plan.context == "Account book of a Basel merchant."
    assert plan.images is not None and plan.images.recorded


async def test_a_model_without_image_input_is_a_plan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stored = await store_document(tmp_path, ["failed"])

    with pytest.raises(RepairPlanError, match="does not support image") as caught:
        plan_repair(spec_for(stored.output, "dry_run", model="gpt-7-foo"), Settings())

    assert caught.value.exit_code == ExitCode.USAGE


async def test_a_missing_context_file_is_a_plan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stored = await store_document(tmp_path, ["failed"])

    with pytest.raises(RepairPlanError):
        plan_repair(
            spec_for(stored.output, context=tmp_path / "missing.txt"), Settings()
        )


async def test_batch_repair_needs_a_batch_api(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, ["failed"])
    spec = spec_for(
        stored.output, "batch", provider="openrouter", model="openai/gpt-5.4"
    )
    with pytest.raises(RepairPlanError, match="batch API"):
        plan_repair(spec, Settings())


async def test_errors_of_the_repair_work_are_not_plan_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stored = await store_document(tmp_path, ["failed"])
    plan = plan_repair(spec_for(stored.output), Settings())

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("unreadable page image")

    monkeypatch.setattr(repair_module, "page_image", broken)

    with pytest.raises(ValueError, match="unreadable page image") as caught:
        await run_repair(plan)
    assert not isinstance(caught.value, RepairPlanError)


def fail_status(monkeypatch: pytest.MonkeyPatch, errors: list[Exception]) -> None:
    """Make the fake backend raise *errors* on its next status calls."""
    real = FakeBackend.status

    def status(self: FakeBackend, handle: BatchHandle) -> BatchStatusInfo:
        if errors:
            raise errors.pop(0)
        return real(self, handle)

    monkeypatch.setattr(FakeBackend, "status", status)


class ScriptedClients(BatchClients):
    """Batch clients whose calls raise *errors* first, as after their retries."""

    def __init__(self, world: World, errors: list[Exception]) -> None:
        super().__init__(Settings(), "repair")
        self.backend = FakeBackend(world, "openai")
        self.errors = errors

    async def call(
        self, provider: str, operation: Callable[[BatchBackend], T], **_: Any
    ) -> T:
        if self.errors:
            raise self.errors.pop(0)
        return operation(self.backend)


async def test_a_retryable_status_error_keeps_the_batch_waiting(
    world: World,
) -> None:
    batch_id = world.add("openai", status=BatchStatus.COMPLETED)
    clients = ScriptedClients(world, [ConnectionError("connection reset by peer")])
    events = RecordingEvents()

    outcome = await wait_for_batches(
        [BatchHandle("openai", batch_id)], clients, events, poll_seconds=0
    )

    assert (outcome.running, outcome.error) == ([], None)
    (warning,) = events.of("warning")
    assert batch_id in warning and "connection" in warning
    assert len(events.of("waiting")) == 1


async def test_a_non_retryable_status_error_stops_the_wait(world: World) -> None:
    batch_id = world.add("openai", status=BatchStatus.IN_PROGRESS)
    clients = ScriptedClients(world, [BadRequest("unknown batch")])
    events = RecordingEvents()

    outcome = await wait_for_batches(
        [BatchHandle("openai", batch_id)], clients, events, poll_seconds=0
    )

    assert outcome.running == []
    assert outcome.error is not None
    assert batch_id in outcome.error and "unknown batch" in outcome.error
    assert events.of("warning") == []


async def test_the_wait_stops_at_its_timeout_with_the_batch_running(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = world.add("openai", status=BatchStatus.IN_PROGRESS)
    clock = Clock()
    monkeypatch.setattr(wait_module, "_clock", clock)
    monkeypatch.setattr(wait_module, "_sleep", clock.sleep)
    events = RecordingEvents()

    outcome = await wait_for_batches(
        [BatchHandle("openai", batch_id)],
        BatchClients(Settings(), "repair"),
        events,
        poll_seconds=10,
        timeout_seconds=25,
    )

    assert (outcome.running, outcome.error) == ([batch_id], None)
    assert clock.sleeps == [10, 10, 5]
    assert world.count("openai", "status") == 4
    assert [waiting.seconds for waiting in events.of("waiting")] == [10, 10, 5]


async def test_a_batch_repair_whose_status_fails_leaves_the_batch_to_check(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "failed"])
    before = stored.output.read_bytes()
    fail_status(monkeypatch, [BadRequest("unknown batch")])
    events = RecordingEvents()

    result = await repair_transcription(
        spec_for(stored.output, "batch"), Settings(), events
    )

    assert (result.repaired, result.failed, result.selected) == (0, 2, 2)
    assert stored.output.read_bytes() == before
    (warning,) = [text for text in events.of("warning") if "unknown batch" in text]
    assert "batch check" in warning
    assert [unit[1] for unit in unit_rows(stored.db)] == ["done", "pending", "pending"]
