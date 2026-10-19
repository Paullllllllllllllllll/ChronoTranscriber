"""Bounded concurrent and streaming page pools of the LLM layer.

``run_concurrent_transcription_tasks`` maps a handler over argument tuples
and ``run_streaming_transcription_tasks`` consumes an async producer through
a bounded queue. Both keep result order, record a failed item or a failed
``on_result`` callback as ``None``, and leave no orphaned task behind. Unit
pipelines on top of the pools are covered in ``tests/methods``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from chronotranscriber.llm.pool import (
    run_concurrent_transcription_tasks,
    run_streaming_transcription_tasks,
)


@pytest.mark.asyncio
class TestRunConcurrentTranscriptionTasks:
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

    async def test_on_result_callback_error_marks_slot_failed(self) -> None:
        """A callback failure must not crash the run AND must mark the slot None.

        The callback is the JSONL write; if it fails the page never reaches the
        resume artifact, so keeping the successful result would silently drop the
        page from the final output. The slot is recorded failed instead.
        """

        async def identity(x):
            return x

        async def bad_callback(result):
            raise ValueError("callback error")

        args_list = [(1,), (2,)]
        results = await run_concurrent_transcription_tasks(
            identity, args_list, concurrency_limit=5, on_result=bad_callback
        )
        # No crash, but every slot whose callback failed is marked None.
        assert results == [None, None]

    async def test_on_result_partial_failure_marks_only_failed_slot(self) -> None:
        """Only the slot whose callback raised is nulled; others keep results."""

        async def identity(x):
            return x

        async def selective_callback(result):
            if result == 2:
                raise ValueError("callback error for 2")

        args_list = [(1,), (2,), (3,)]
        results = await run_concurrent_transcription_tasks(
            identity,
            args_list,
            concurrency_limit=1,
            on_result=selective_callback,
        )
        assert results == [1, None, 3]

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
            "chronotranscriber.llm.pool.asyncio.create_task",
            counting_create_task,
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


class TestRunStreamingTranscriptionTasks:
    @pytest.mark.unit
    def test_all_items_processed_and_streamed(self) -> None:
        async def run() -> tuple[list[Any], list[Any]]:
            async def producer() -> AsyncIterator[int]:
                for i in range(10):
                    yield i

            streamed: list[Any] = []

            async def on_result(r: Any) -> None:
                streamed.append(r)

            results = await run_streaming_transcription_tasks(
                producer(), lambda x: _double(x), 3, on_result=on_result
            )
            return results, streamed

        async def _double(x: int) -> int:
            return x * 2

        results, streamed = asyncio.run(run())
        assert sorted(results) == [i * 2 for i in range(10)]
        assert sorted(streamed) == sorted(results)

    @pytest.mark.unit
    def test_queue_stays_bounded(self) -> None:
        limit = 2

        async def run() -> int:
            produced = 0
            consumed = 0
            max_ahead = 0

            async def producer() -> AsyncIterator[int]:
                nonlocal produced
                for i in range(20):
                    produced += 1
                    yield i

            async def handler(x: int) -> int:
                nonlocal consumed, max_ahead
                max_ahead = max(max_ahead, produced - consumed)
                await asyncio.sleep(0.005)
                consumed += 1
                return x

            await run_streaming_transcription_tasks(producer(), handler, limit)
            return max_ahead

        max_ahead = asyncio.run(run())
        # Producer can be at most queue size (2*limit) + in-flight workers
        # (limit) + 1 (item awaiting put) ahead of consumption.
        assert max_ahead <= 2 * limit + limit + 1

    @pytest.mark.unit
    def test_producer_exception_propagates_after_drain(self) -> None:
        async def run() -> tuple[list[Any], Exception | None]:
            async def producer() -> AsyncIterator[int]:
                yield 1
                yield 2
                raise RuntimeError("render guard")

            handled: list[int] = []

            async def handler(x: int) -> int:
                handled.append(x)
                return x

            try:
                await run_streaming_transcription_tasks(producer(), handler, 2)
            except RuntimeError as e:
                return handled, e
            return handled, None

        handled, error = asyncio.run(run())
        assert error is not None and "render guard" in str(error)
        assert sorted(handled) == [1, 2]  # items before the failure survive

    @pytest.mark.unit
    def test_worker_error_recorded_as_none(self) -> None:
        async def run() -> list[Any]:
            async def producer() -> AsyncIterator[int]:
                for i in range(4):
                    yield i

            async def handler(x: int) -> int:
                if x == 2:
                    raise ValueError("boom")
                return x

            return await run_streaming_transcription_tasks(producer(), handler, 2)

        results = asyncio.run(run())
        assert None in results
        assert sorted(r for r in results if r is not None) == [0, 1, 3]


@pytest.mark.asyncio
class TestStreamingCallbackFailure:
    """A failed on_result callback nulls the affected slot (page failure)."""

    async def test_callback_failure_records_none(self) -> None:
        async def producer():
            for i in range(4):
                yield i
                await asyncio.sleep(0)

        async def handler(item):
            return item

        async def bad_callback(result):
            # Only the odd items' JSONL writes fail.
            if result % 2 == 1:
                raise OSError("disk full")

        results = await run_streaming_transcription_tasks(
            producer(),
            handler,
            concurrency_limit=1,
            on_result=bad_callback,
        )
        # concurrency_limit=1 keeps ordering deterministic; odd slots are None.
        assert results == [0, None, 2, None]


@pytest.mark.asyncio
class TestStreamingOrphanFreeFailure:
    """The streaming runner cancels every task if the gather raises."""

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
