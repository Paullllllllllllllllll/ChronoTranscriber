"""Tests for modules.infra.progress."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from modules.infra.progress import ProgressState, ProgressTracker


# ---------------------------------------------------------------------------
# ProgressState
# ---------------------------------------------------------------------------

class TestProgressStateInit:
    def test_sets_total(self):
        state = ProgressState(total=100)
        assert state.total == 100
        assert state.completed == 0
        assert state.failed == 0

    def test_auto_sets_start_time(self):
        state = ProgressState(total=10)
        assert state.start_time is not None
        assert isinstance(state.start_time, datetime)

    def test_custom_start_time(self):
        custom = datetime(2024, 1, 1)
        state = ProgressState(total=10, start_time=custom)
        assert state.start_time == custom


class TestProgressStateProperties:
    def test_remaining(self):
        state = ProgressState(total=10, completed=3, failed=2)
        assert state.remaining == 5

    def test_percent_complete_partial(self):
        state = ProgressState(total=10, completed=3, failed=2)
        assert state.percent_complete == pytest.approx(50.0)

    def test_percent_complete_zero_total(self):
        state = ProgressState(total=0)
        assert state.percent_complete == 100.0

    def test_percent_complete_all_done(self):
        state = ProgressState(total=5, completed=5)
        assert state.percent_complete == pytest.approx(100.0)

    def test_elapsed_seconds(self):
        past = datetime.now() - timedelta(seconds=10)
        state = ProgressState(total=10, start_time=past)
        assert state.elapsed_seconds >= 9.0

    def test_elapsed_seconds_none_start(self):
        state = ProgressState(total=10)
        state.start_time = None
        assert state.elapsed_seconds == 0.0

    def test_estimated_remaining_no_progress(self):
        state = ProgressState(total=10)
        assert state.estimated_remaining_seconds is None

    def test_estimated_remaining_some_progress(self):
        past = datetime.now() - timedelta(seconds=10)
        state = ProgressState(total=10, completed=5, start_time=past)
        est = state.estimated_remaining_seconds
        assert est is not None
        assert est > 0

    def test_estimated_remaining_all_done(self):
        state = ProgressState(total=5, completed=5)
        assert state.estimated_remaining_seconds is None


class TestProgressStateMutations:
    def test_increment_completed(self):
        state = ProgressState(total=10)
        state.increment_completed()
        assert state.completed == 1

    def test_increment_failed(self):
        state = ProgressState(total=10)
        state.increment_failed()
        assert state.failed == 1


class TestProgressStateFormatSummary:
    def test_format_summary_basic(self):
        past = datetime.now() - timedelta(seconds=65)
        state = ProgressState(total=10, completed=5, start_time=past)
        summary = state.format_summary()
        assert "5/10" in summary
        assert "50.0%" in summary
        assert "1m" in summary

    def test_format_summary_with_failures(self):
        state = ProgressState(total=10, completed=3, failed=2)
        summary = state.format_summary()
        assert "2 failed" in summary

    def test_format_summary_with_estimate(self):
        past = datetime.now() - timedelta(seconds=10)
        state = ProgressState(total=10, completed=5, start_time=past)
        summary = state.format_summary()
        assert "remaining" in summary


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------

class TestProgressTracker:
    @pytest.mark.asyncio
    async def test_increment_completed(self):
        tracker = ProgressTracker(total=10)
        await tracker.increment_completed()
        assert tracker.state.completed == 1

    @pytest.mark.asyncio
    async def test_increment_failed(self):
        tracker = ProgressTracker(total=10)
        await tracker.increment_failed()
        assert tracker.state.failed == 1

    @pytest.mark.asyncio
    async def test_callback_called_at_interval(self):
        callback = MagicMock()
        tracker = ProgressTracker(total=10, on_update=callback, update_interval=5)
        for _ in range(5):
            await tracker.increment_completed()
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_callback_called_at_completion(self):
        callback = MagicMock()
        tracker = ProgressTracker(total=3, on_update=callback, update_interval=10)
        for _ in range(3):
            await tracker.increment_completed()
        # Should fire at total=3 (final item)
        assert callback.call_count >= 1

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(self):
        def bad_callback(state):
            raise ValueError("callback error")

        tracker = ProgressTracker(total=10, on_update=bad_callback, update_interval=1)
        await tracker.increment_completed()
        assert tracker.state.completed == 1

    @pytest.mark.asyncio
    async def test_finalize_calls_callback(self):
        callback = MagicMock()
        tracker = ProgressTracker(total=10, on_update=callback)
        await tracker.finalize()
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_finalize_without_callback(self):
        tracker = ProgressTracker(total=10)
        await tracker.finalize()  # Should not raise

    @pytest.mark.asyncio
    async def test_finalize_callback_error_does_not_crash(self):
        def bad_callback(state):
            raise RuntimeError("finalize error")

        tracker = ProgressTracker(total=10, on_update=bad_callback)
        await tracker.finalize()  # Should not raise
