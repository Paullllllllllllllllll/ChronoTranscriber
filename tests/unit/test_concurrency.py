"""Tests for modules.infra.concurrency."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from modules.infra.concurrency import run_concurrent_transcription_tasks


@pytest.mark.asyncio
class TestRunConcurrentTranscriptionTasks:
    async def test_basic_execution(self):
        async def add(a, b):
            return a + b

        results = await run_concurrent_transcription_tasks(
            add, [(1, 2), (3, 4), (5, 6)], concurrency_limit=2
        )
        assert results == [3, 7, 11]

    async def test_empty_args_list(self):
        async def noop():
            pass

        results = await run_concurrent_transcription_tasks(noop, [])
        assert results == []

    async def test_concurrency_limit_respected(self):
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
        await run_concurrent_transcription_tasks(
            track, args_list, concurrency_limit=3
        )
        assert max_running <= 3

    async def test_delay_between_tasks(self):
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

    async def test_on_result_callback_called(self):
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

    async def test_on_result_callback_error_does_not_crash(self):
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

    async def test_task_exception_returns_none(self):
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

    async def test_preserves_order(self):
        async def delayed_identity(x, delay):
            await asyncio.sleep(delay)
            return x

        # Task 0 finishes last, task 2 finishes first
        args_list = [(0, 0.03), (1, 0.02), (2, 0.01)]
        results = await run_concurrent_transcription_tasks(
            delayed_identity, args_list, concurrency_limit=5
        )
        assert results == [0, 1, 2]
