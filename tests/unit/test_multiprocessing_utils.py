"""Tests for modules.infra.multiprocessing_utils."""

from __future__ import annotations

import pytest

from modules.infra.multiprocessing_utils import wrapper, run_multiprocessing_tasks


# Module-level functions are picklable (required by multiprocessing)
def _double(x):
    return x * 2


def _identity(x):
    return x


def _maybe_fail(x):
    if x == 2:
        raise RuntimeError("fail")
    return x


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------

class TestWrapper:
    def test_successful_call(self):
        assert wrapper(_double, (3,)) == 6

    def test_exception_returns_none(self):
        assert wrapper(_maybe_fail, (2,)) is None

    def test_empty_args(self):
        def no_args():
            return 42
        assert wrapper(no_args, ()) == 42


# ---------------------------------------------------------------------------
# run_multiprocessing_tasks
# ---------------------------------------------------------------------------

class TestRunMultiprocessingTasks:
    def test_basic_execution(self):
        results = run_multiprocessing_tasks(_double, [(1,), (2,), (3,)], processes=2)
        assert sorted(results) == [2, 4, 6]

    def test_empty_args_list(self):
        results = run_multiprocessing_tasks(_identity, [], processes=1)
        assert results == []

    def test_single_process(self):
        results = run_multiprocessing_tasks(_identity, [(10,), (20,)], processes=1)
        assert sorted(results) == [10, 20]

    def test_exception_in_task_returns_none(self):
        results = run_multiprocessing_tasks(
            _maybe_fail, [(1,), (2,), (3,)], processes=2
        )
        assert results[0] == 1
        assert results[1] is None
        assert results[2] == 3

    def test_default_processes(self):
        results = run_multiprocessing_tasks(_identity, [(1,), (2,)])
        assert sorted(results) == [1, 2]
