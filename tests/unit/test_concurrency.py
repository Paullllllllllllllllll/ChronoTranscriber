"""Tests for modules.infra.concurrency."""

from __future__ import annotations

import asyncio

import pytest

from modules.infra.concurrency import (
    run_concurrent_transcription_tasks,
    run_streaming_transcription_tasks,
)
from modules.infra.token_budget import DailyTokenTracker


@pytest.mark.asyncio
class TestRunConcurrentTranscriptionTasks:
    async def test_basic_execution(self) -> None:
        async def add(a, b):
            return a + b

        results = await run_concurrent_transcription_tasks(
            add, [(1, 2), (3, 4), (5, 6)], concurrency_limit=2
        )
        assert results == [3, 7, 11]

    async def test_empty_args_list(self) -> None:
        async def noop():
            pass

        results = await run_concurrent_transcription_tasks(noop, [])
        assert results == []

    async def test_concurrency_limit_respected(self) -> None:
        running = 0
        max_running = 0

        async def track(*args):
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.01)
            running -= 1
            return args

        args_list = [(i,) for i in range(10)]
        await run_concurrent_transcription_tasks(track, args_list, concurrency_limit=3)
        assert max_running <= 3

    async def test_delay_between_tasks(self) -> None:
        call_times = []

        async def record_time(idx):
            call_times.append(asyncio.get_event_loop().time())
            return idx

        args_list = [(i,) for i in range(3)]
        await run_concurrent_transcription_tasks(
            record_time, args_list, concurrency_limit=1, delay=0.05
        )
        assert len(call_times) == 3
        # Each call should be delayed by at least ~0.05s
        for i in range(1, len(call_times)):
            assert call_times[i] - call_times[i - 1] >= 0.04

    async def test_on_result_callback_called(self) -> None:
        collected = []

        async def identity(x):
            return x

        async def collect(result):
            collected.append(result)

        args_list = [(10,), (20,), (30,)]
        results = await run_concurrent_transcription_tasks(
            identity, args_list, concurrency_limit=5, on_result=collect
        )
        assert sorted(collected) == [10, 20, 30]
        assert sorted(results) == [10, 20, 30]

    async def test_on_result_callback_error_does_not_crash(self) -> None:
        async def identity(x):
            return x

        async def bad_callback(result):
            raise ValueError("callback error")

        args_list = [(1,), (2,)]
        results = await run_concurrent_transcription_tasks(
            identity, args_list, concurrency_limit=5, on_result=bad_callback
        )
        # Results should still be returned despite callback failures
        assert sorted(results) == [1, 2]

    async def test_task_exception_returns_none(self) -> None:
        async def fail_on_two(x):
            if x == 2:
                raise RuntimeError("fail")
            return x

        args_list = [(1,), (2,), (3,)]
        results = await run_concurrent_transcription_tasks(
            fail_on_two, args_list, concurrency_limit=5
        )
        assert results[0] == 1
        assert results[1] is None
        assert results[2] == 3

    async def test_preserves_order(self) -> None:
        async def delayed_identity(x, delay):
            await asyncio.sleep(delay)
            return x

        # Task 0 finishes last, task 2 finishes first
        args_list = [(0, 0.03), (1, 0.02), (2, 0.01)]
        results = await run_concurrent_transcription_tasks(
            delayed_identity, args_list, concurrency_limit=5
        )
        assert results == [0, 1, 2]

    async def test_bounded_lazy_submission_no_eager_task_pile(self) -> None:
        """Only ``min(limit, n)`` Task objects are created, not one per item."""
        from unittest.mock import patch

        n = 50
        limit = 4
        created = 0
        orig_create_task = asyncio.create_task

        def counting_create_task(coro, *args, **kwargs):
            nonlocal created
            created += 1
            return orig_create_task(coro, *args, **kwargs)

        async def work(i):
            await asyncio.sleep(0)
            return i * 2

        with patch(
            "modules.infra.concurrency.asyncio.create_task", counting_create_task
        ):
            results = await run_concurrent_transcription_tasks(
                work, [(i,) for i in range(n)], concurrency_limit=limit
            )

        assert results == [i * 2 for i in range(n)]
        # A 50-item folder must not materialize 50 tasks.
        assert created == limit

    async def test_gather_failure_cancels_all_workers(self) -> None:
        """A BaseException out of a worker cancels the whole pool (no orphans)."""

        class Boom(BaseException):
            pass

        async def work(i):
            raise Boom

        with pytest.raises(Boom):
            await run_concurrent_transcription_tasks(
                work, [(i,) for i in range(50)], concurrency_limit=4
            )

        current = asyncio.current_task()
        leaked = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        assert leaked == []


@pytest.mark.asyncio
class TestStreamingBudgetGate:
    """The chunk/page-level token-budget gate on the streaming runner."""

    async def test_defers_pages_when_budget_exhausted(self, tmp_path) -> None:
        tracker = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
            estimate_smoothing=0.3,
        )
        exhausted = asyncio.Event()
        n = 8
        processed: list[int] = []

        async def producer():
            for i in range(n):
                yield i

        async def handler(item):
            # Commit tokens as the real provider layer would after each call.
            tracker.add_tokens(30)
            return item

        async def on_result(result):
            processed.append(result)

        await run_streaming_transcription_tasks(
            producer(),
            handler,
            concurrency_limit=1,
            delay=0,
            on_result=on_result,
            tracker=tracker,
            exhausted=exhausted,
        )

        # The budget was hit, so only some pages were processed; the rest were
        # deferred and left no on_result record (the resume path re-runs them).
        assert exhausted.is_set()
        assert 0 < len(processed) < n

    async def test_disabled_tracker_processes_all(self, tmp_path) -> None:
        tracker = DailyTokenTracker(
            daily_limit=0, enabled=False, state_file=tmp_path / "s.json"
        )
        exhausted = asyncio.Event()
        processed: list[int] = []

        async def producer():
            for i in range(5):
                yield i

        async def handler(item):
            tracker.add_tokens(9999)  # ignored: tracker disabled
            return item

        async def on_result(result):
            processed.append(result)

        await run_streaming_transcription_tasks(
            producer(),
            handler,
            concurrency_limit=2,
            on_result=on_result,
            tracker=tracker,
            exhausted=exhausted,
        )

        assert not exhausted.is_set()
        assert sorted(processed) == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
class TestStreamingOrphanFreeFailure:
    """The streaming runner cancels every task if the gather raises (B8)."""

    async def test_gather_failure_cancels_producer_and_workers(self) -> None:
        class Boom(BaseException):
            pass

        async def producer():
            for i in range(100):
                yield i
                await asyncio.sleep(0)

        async def handler(item):
            # BaseException is NOT swallowed by the worker's `except Exception`,
            # so it propagates and makes the gather raise.
            raise Boom

        with pytest.raises(Boom):
            await run_streaming_transcription_tasks(
                producer(), handler, concurrency_limit=4
            )

        # No orphaned producer/worker tasks remain.
        current = asyncio.current_task()
        leaked = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        assert leaked == []
