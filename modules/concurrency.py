# modules/concurrency.py

import asyncio
import logging
from typing import Any, Callable, List, Tuple, Awaitable

logger = logging.getLogger(__name__)


async def run_concurrent_transcription_tasks(
		corofunc: Callable[..., Awaitable[Any]],
		args_list: List[Tuple[Any, ...]],
		concurrency_limit: int = 20,
		delay: float = 0
) -> List[Any]:
	"""
    Run the given asynchronous function concurrently over a list of argument tuples, respecting a concurrency limit.

    Parameters:
        corofunc (Callable[..., Awaitable[Any]]): The async function to execute.
        args_list (List[Tuple[Any, ...]]): The list of argument tuples.
        concurrency_limit (int): Maximum number of concurrent tasks.
        delay (float): Delay in seconds between task starts.

    Returns:
        List[Any]: List of results.
    """
	semaphore = asyncio.Semaphore(concurrency_limit)

	async def worker(args: Tuple[Any, ...]) -> Any:
		async with semaphore:
			if delay > 0:
				await asyncio.sleep(delay)
			try:
				return await corofunc(*args)
			except Exception as e:
				logger.error(
					f"Transcription task failed with arguments {args}: {e}")
				return None

	tasks = [asyncio.create_task(worker(args)) for args in args_list]
	results = await asyncio.gather(*tasks, return_exceptions=False)
	return results
