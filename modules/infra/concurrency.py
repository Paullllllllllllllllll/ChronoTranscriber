"""Concurrency utilities for async task management.

Provides semaphore-based concurrency control for async operations and a
bounded producer-consumer runner for streaming pipelines.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

_SENTINEL: Any = object()


async def run_concurrent_transcription_tasks(
    corofunc: Callable[..., Awaitable[Any]],
    args_list: list[tuple[Any, ...]],
    concurrency_limit: int = 20,
    delay: float = 0,
    on_result: Callable[[Any], Awaitable[None]] | None = None,
    tracker: Any = None,
    exhausted: asyncio.Event | None = None,
    stamp: tuple[str | None, str | None, str | None] | None = None,
) -> list[Any]:
    """
    Run async function concurrently over argument tuples with concurrency control.

    Args:
        corofunc: The async function to execute.
        args_list: List of argument tuples to pass to the function.
        concurrency_limit: Maximum number of concurrent tasks (default: 20).
        delay: Delay in seconds between task starts (default: 0).
        on_result: Optional async callback to process each result immediately.
        tracker: Optional token tracker; when given, each task reserves an
            estimated cost before running and releases it after, so the daily
            budget is enforced per task. ``None`` disables the gate.
        exhausted: Optional event set when the budget is exhausted; tasks not
            yet started then defer (return None without running), letting the
            caller drain, wait for the daily reset, and re-pass.
        stamp: Optional (provider, key_env, model) so each reservation and
            release lands in the call's per-key pool bucket. ``None`` reserves
            against the unattributed bucket (today's combined-only semantics).

    Returns:
        List of results from all task executions, one per input tuple in the
        original order (``None`` for a failed or deferred task).
    """
    n = len(args_list)
    results: list[Any] = [None] * n
    if n == 0:
        return results

    # Bounded lazy submission: only ``workers_n`` Task objects exist at once,
    # each pulling the next index off a queue, so a 900-item folder never
    # materializes 900 tasks and cancellation stays prompt. Results are stored
    # by index, preserving input order; workers swallow task exceptions (return
    # None) per the existing contract.
    index_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        index_queue.put_nowait(i)
    workers_n = max(1, min(int(concurrency_limit), n))
    s_provider, s_key_env, s_model = stamp if stamp else (None, None, None)

    async def worker() -> None:
        while True:
            try:
                i = index_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            args = args_list[i]
            if exhausted is not None and exhausted.is_set():
                # Budget exhausted: defer (re-processed on the next pass).
                continue
            reserved = None
            if tracker is not None:
                reserved = tracker.try_reserve(
                    provider=s_provider, key_env=s_key_env, model=s_model
                )
                if reserved is None:
                    if exhausted is not None:
                        exhausted.set()
                    continue
            try:
                if delay > 0:
                    await asyncio.sleep(delay)
                result = await corofunc(*args)
                results[i] = result
                # Stream the result to the callback as soon as it is ready.
                if on_result is not None:
                    try:
                        await on_result(result)
                    except Exception as cb_exc:
                        logger.error(
                            f"on_result callback failed for args {args}: {cb_exc}"
                        )
            except Exception as e:
                logger.error(f"Transcription task failed with arguments {args}: {e}")
                results[i] = None
            finally:
                if reserved:
                    tracker.release(
                        reserved,
                        provider=s_provider,
                        key_env=s_key_env,
                        model=s_model,
                    )

    worker_tasks = [asyncio.create_task(worker()) for _ in range(workers_n)]
    try:
        await asyncio.gather(*worker_tasks)
    except BaseException:
        # Cancel and drain the pool so no worker is orphaned on failure.
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        raise
    return results


async def run_streaming_transcription_tasks(
    producer: AsyncIterator[Any],
    handler: Callable[[Any], Awaitable[Any]],
    concurrency_limit: int = 20,
    delay: float = 0,
    on_result: Callable[[Any], Awaitable[None]] | None = None,
    tracker: Any = None,
    exhausted: asyncio.Event | None = None,
    stamp: tuple[str | None, str | None, str | None] | None = None,
) -> list[Any]:
    """Consume an async producer with a bounded queue and worker pool.

    A single producer task fills an ``asyncio.Queue`` bounded to
    ``2 * concurrency_limit`` items; exactly ``concurrency_limit`` workers
    pull items and run ``handler`` on each. The worker count is the
    concurrency cap, so no extra semaphore is needed.

    A producer exception is captured, the workers are drained via
    sentinels, and the exception is re-raised after all workers finish —
    results produced up to that point are processed normally (and streamed
    to ``on_result``), so partial progress is preserved by the caller's
    JSONL writes.

    Args:
        producer: Async iterator yielding work items (e.g. PagePayloads).
        handler: Async function applied to each item; its return value is
            collected and streamed to ``on_result``.
        concurrency_limit: Number of worker tasks (default: 20).
        delay: Delay in seconds before each handler call (default: 0).
        on_result: Optional async callback invoked per result as soon as it
            is ready.

    Returns:
        Unordered list of handler results (failed items return None).
    """
    workers_n = max(1, int(concurrency_limit))
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=2 * workers_n)
    producer_error: list[BaseException] = []
    results: list[Any] = []
    s_provider, s_key_env, s_model = stamp if stamp else (None, None, None)

    async def produce() -> None:
        try:
            async for item in producer:
                # Stop rendering new pages once the budget is exhausted; the
                # remaining items are re-rendered on the next pass.
                if exhausted is not None and exhausted.is_set():
                    break
                await queue.put(item)
        except BaseException as e:  # noqa: BLE001 - re-raised after drain
            producer_error.append(e)
        finally:
            for _ in range(workers_n):
                await queue.put(_SENTINEL)

    async def work() -> None:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            # Budget gate: skip (defer) items once exhausted so they leave no
            # JSONL record and are re-processed after the daily reset.
            if exhausted is not None and exhausted.is_set():
                continue
            reserved = None
            if tracker is not None:
                reserved = tracker.try_reserve(
                    provider=s_provider, key_env=s_key_env, model=s_model
                )
                if reserved is None:
                    if exhausted is not None:
                        exhausted.set()
                    continue
            try:
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    result = await handler(item)
                except Exception as e:
                    logger.error(f"Streaming transcription task failed: {e}")
                    results.append(None)
                    continue
                results.append(result)
                if on_result is not None:
                    try:
                        await on_result(result)
                    except Exception as cb_exc:
                        logger.error(f"on_result callback failed: {cb_exc}")
            finally:
                if reserved:
                    tracker.release(
                        reserved,
                        provider=s_provider,
                        key_env=s_key_env,
                        model=s_model,
                    )

    producer_task = asyncio.create_task(produce())
    worker_tasks = [asyncio.create_task(work()) for _ in range(workers_n)]
    all_tasks = [producer_task, *worker_tasks]
    try:
        await asyncio.gather(*all_tasks)
    except BaseException:
        # If the gather raises, cancel and await the producer and every worker
        # (suppressing their CancelledError) so none is left orphaned before the
        # exception propagates.
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        raise

    if producer_error:
        raise producer_error[0]
    return results
