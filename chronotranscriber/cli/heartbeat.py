"""The launcher's progress file, written beside the stderr reports.

:func:`with_heartbeat` puts a heartbeat sink beside the run's stderr
reporter when ``CHRONO_HEARTBEAT_FILE`` names a file: each item's pages,
images or audio chunks processed, the item position and, during the last
item, an ETA. :class:`BatchHeartbeat` writes the submissions a
``batch check --wait`` has seen finish, once before each sleep.
"""

from __future__ import annotations

from chronotranscriber.batch.check import CheckResult
from chronotranscriber.batch.finalize import PENDING
from chronotranscriber.common.heartbeat import Heartbeat, ItemProgress
from chronotranscriber.common.polling import WaitTick
from chronotranscriber.events import (
    ItemFinished,
    ItemStarted,
    PageDone,
    RunEvents,
    Waiting,
    item_key,
)

__all__ = ["BatchHeartbeat", "HeartbeatEvents", "with_heartbeat"]

UNIT = "pages"


class HeartbeatEvents:
    """Pass every event to *inner*, then record it in *progress*.

    The units written are those of the open items together: each item's
    latest count, summed, so that items open at once share one line.
    *open_items* is the number of items the run keeps open at once: an item
    the run leaves without a finish event (``--pages`` selected none of its
    units, or the run stopped in it) is dropped once that many younger items
    have started.
    """

    def __init__(
        self, inner: RunEvents, progress: ItemProgress, open_items: int = 1
    ) -> None:
        self.inner = inner
        self.progress = progress
        self.open_items = max(1, open_items)
        self._open: dict[tuple[str, str], tuple[int, int]] = {}

    def _units(self, noun: str | None = None) -> None:
        done = sum(count[0] for count in self._open.values())
        total = sum(count[1] for count in self._open.values())
        self.progress.units(done, total, noun)

    def item_started(self, event: ItemStarted) -> None:
        """Forward the event; start the item's units at zero."""
        self.inner.item_started(event)
        key = item_key(event)
        self._open.pop(key, None)
        while len(self._open) >= self.open_items:
            del self._open[next(iter(self._open))]
        self.progress.item_started(event.index, event.total)
        if self._open:
            self._units()
        self._open[key] = (0, 0)

    def page_done(self, event: PageDone) -> None:
        """Forward the event; record the item's units done, named by their kind."""
        self.inner.page_done(event)
        self._open[item_key(event)] = (event.done, event.total)
        self._units(f"{event.unit}s")

    def waiting(self, event: Waiting) -> None:
        """Forward the event."""
        self.inner.waiting(event)

    def warning(self, message: str) -> None:
        """Forward the warning."""
        self.inner.warning(message)

    def item_finished(self, event: ItemFinished) -> None:
        """Forward the event; write the finished item.

        While other items are open the write is that of their units: the
        run is not at its end, whatever the position of the item that ended.
        """
        self.inner.item_finished(event)
        self._open.pop(item_key(event), None)
        if self._open:
            self._units()
        else:
            self.progress.item_finished()


def with_heartbeat(
    reporter: RunEvents, open_items: int = 1
) -> tuple[RunEvents, ItemProgress | None]:
    """Return the events to run with and the progress to finish, if any.

    Without ``CHRONO_HEARTBEAT_FILE`` the reporter alone receives the events.
    *open_items* is the number of items the run keeps open at once.
    """
    heartbeat = Heartbeat.from_env()
    if heartbeat is None:
        return reporter, None
    progress = ItemProgress(heartbeat, UNIT)
    return HeartbeatEvents(reporter, progress, open_items), progress


class BatchHeartbeat:
    """The submissions of a wait that finished, of those its checks saw."""

    def __init__(self, heartbeat: Heartbeat) -> None:
        self.heartbeat = heartbeat
        self._outcomes: dict[tuple[object, str], str] = {}

    @classmethod
    def from_env(cls) -> BatchHeartbeat | None:
        """Return the writer for ``CHRONO_HEARTBEAT_FILE``; None when unset."""
        heartbeat = Heartbeat.from_env()
        return None if heartbeat is None else cls(heartbeat)

    def on_pass(self, result: CheckResult) -> None:
        """Keep each submission's latest outcome."""
        try:
            for check in result.submissions:
                self._outcomes[(check.database, check.submission_id)] = check.outcome
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            self.heartbeat.failed(exc)

    def on_wait(self, tick: WaitTick) -> None:
        """Write the finished submissions before the wait sleeps."""
        try:
            done = sum(outcome != PENDING for outcome in self._outcomes.values())
            self.heartbeat.update(done, len(self._outcomes), "batches", force=True)
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            self.heartbeat.failed(exc)
