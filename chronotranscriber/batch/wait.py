"""Wait for batches to end: poll their status until each is terminal.

:func:`wait_for_submissions` waits for the provider batches of submissions
of the store; :func:`wait_for_batches` polls given batch handles through
:func:`~chronotranscriber.common.polling.poll_until_settled`. Neither writes
anything: finalization is the caller's next step.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from chronotranscriber.batch.clients import BatchClients, status_of
from chronotranscriber.batch.finalize import submission_handle
from chronotranscriber.common.batch import TERMINAL_STATUSES, BatchHandle
from chronotranscriber.common.polling import WaitLimits, WaitTick, poll_until_settled
from chronotranscriber.common.retry import classify_error
from chronotranscriber.common.store import Store
from chronotranscriber.events import RunEvents, Waiting

__all__ = [
    "POLL_SECONDS",
    "WAIT_TIMEOUT_SECONDS",
    "WaitOutcome",
    "wait_for_batches",
    "wait_for_submissions",
]

logger = logging.getLogger(__name__)

POLL_SECONDS = 10.0
WAIT_TIMEOUT_SECONDS = 7200.0

_clock: Callable[[], float] = time.monotonic
"""The wait's clock; tests replace it together with :func:`_sleep`."""


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass(frozen=True)
class WaitOutcome:
    """How a wait ended.

    *running* lists the batches still running at the end; *error* names the
    batch and the non-retryable error that stopped the wait.
    """

    running: list[str] = field(default_factory=list)
    error: str | None = None


async def wait_for_batches(
    handles: Sequence[BatchHandle],
    clients: BatchClients,
    events: RunEvents,
    *,
    poll_seconds: float = POLL_SECONDS,
    timeout_seconds: float = WAIT_TIMEOUT_SECONDS,
) -> WaitOutcome:
    """Poll *handles* until every batch has ended or the timeout passes.

    A retryable error while reading a status counts the batch as running
    for that poll and is reported as a warning; a non-retryable one ends the
    wait with the error and no batch counted as running.
    """

    async def status_pass(_number: int) -> WaitOutcome:
        running: list[str] = []
        for handle in handles:
            try:
                info = await clients.call(handle.provider, status_of(handle))
            except Exception as exc:  # noqa: BLE001 - classified just below
                kind = classify_error(exc)
                if not kind.retryable:
                    logger.error("Status of batch %s failed: %s", handle.batch_id, exc)
                    return WaitOutcome(error=f"{handle.batch_id}: {exc}")
                events.warning(
                    f"Could not read the status of batch {handle.batch_id} "
                    f"({kind}): {exc}; polling again."
                )
                running.append(handle.batch_id)
                continue
            if info.status not in TERMINAL_STATUSES:
                running.append(handle.batch_id)
        return WaitOutcome(running=running)

    def on_wait(tick: WaitTick) -> None:
        events.waiting(
            Waiting(
                reason=(
                    f"{tick.pending} batch(es) still running "
                    f"({int(tick.elapsed)}s elapsed)"
                ),
                seconds=tick.next_in,
            )
        )

    result = await poll_until_settled(
        status_pass,
        lambda outcome: len(outcome.running),
        WaitLimits(poll_seconds, timeout_seconds),
        on_wait=on_wait,
        clock=_clock,
        sleep=_sleep,
    )
    return result.last if result.last is not None else WaitOutcome()


async def wait_for_submissions(
    store: Store,
    submission_ids: Sequence[str],
    clients: BatchClients,
    events: RunEvents,
    *,
    poll_seconds: float = POLL_SECONDS,
    timeout_seconds: float = WAIT_TIMEOUT_SECONDS,
) -> WaitOutcome:
    """Poll the provider batches of the store's *submission_ids* until they end.

    Submissions without a provider batch id are left out.
    """
    handles = [
        submission_handle(submission)
        for submission in (store.submission(sid) for sid in submission_ids)
        if submission.batch_id
    ]
    return await wait_for_batches(
        handles,
        clients,
        events,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
