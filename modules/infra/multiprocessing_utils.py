"""Multiprocessing utilities for CPU-bound parallel tasks.

Provides process pool management for parallel execution of functions.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Any, Tuple

logger = logging.getLogger(__name__)


def wrapper(func: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
    """
    Unpack arguments and call the target function with error handling.

    Args:
        func: The function to call.
        args: Arguments to pass to the function.

    Returns:
        The result of the function call, or None if an error occurred.
    """
    try:
        return func(*args)
    except Exception as e:
        logger.error(f"Error in multiprocessing task with args {args}: {e}")
        return None


def run_multiprocessing_tasks(
    func: Callable[..., Any],
    args_list: List[Tuple[Any, ...]],
    processes: int | None = None,
) -> List[Any]:
    """
    Run function over argument tuples using multiprocessing.

    Args:
        func: The function to execute.
        args_list: List of argument tuples to pass to the function.
        processes: Number of processes to use (default: cpu_count - 1).

    Returns:
        List of results from all function calls.
    """
    if processes is None:
        processes = max(1, cpu_count() - 1)

    with Pool(processes=processes) as pool:
        results = pool.starmap(wrapper, [(func, args) for args in args_list])

    return results
