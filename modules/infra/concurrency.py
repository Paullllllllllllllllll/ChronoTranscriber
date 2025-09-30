"""Concurrency utilities for async task management.

Provides semaphore-based concurrency control for async operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Tuple, Awaitable, Optional

logger = logging.getLogger(__name__)


async def run_concurrent_transcription_tasks(
    corofunc: Callable[..., Awaitable[Any]],
    args_list: List[Tuple[Any, ...]],
    concurrency_limit: int = 20,
    delay: float = 0,
    on_result: Optional[Callable[[Any], Awaitable[None]]] = None,
) -> List[Any]:
    """
    Run async function concurrently over argument tuples with concurrency control.

    Args:
        corofunc: The async function to execute.
        args_list: List of argument tuples to pass to the function.
        concurrency_limit: Maximum number of concurrent tasks (default: 20).
        delay: Delay in seconds between task starts (default: 0).
        on_result: Optional async callback to process each result immediately.

    Returns:
        List of results from all task executions.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def worker(args: Tuple[Any, ...]) -> Any:
        async with semaphore:
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                result = await corofunc(*args)
                # Stream the result to the callback as soon as it is ready
                if on_result is not None:
                    try:
                        await on_result(result)
                    except Exception as cb_exc:
                        logger.error(f"on_result callback failed for args {args}: {cb_exc}")
                return result
            except Exception as e:
                logger.error(
                    f"Transcription task failed with arguments {args}: {e}")
                return None

    tasks = [asyncio.create_task(worker(args)) for args in args_list]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results
