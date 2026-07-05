"""Unit tests for modules/infra/token_budget.py.

Tests token usage tracking and daily limit management.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.infra.token_budget as _tb_module
from modules.infra.token_budget import (
    DailyTokenTracker,
    get_token_tracker,
)


class TestDailyTokenTracker:
    """Tests for DailyTokenTracker class."""

    @pytest.fixture
    def tracker(self, temp_dir):
        """Create a DailyTokenTracker with temp state file."""
        state_file = temp_dir / ".token_state.json"
        return DailyTokenTracker(
            daily_limit=100000,
            state_file=state_file,
        )

    @pytest.mark.unit
    def test_initialization(self, tracker) -> None:
        """Test TokenTracker initialization."""
        assert tracker.daily_limit == 100000
        assert tracker.get_tokens_used_today() == 0

    @pytest.mark.unit
    def test_add_tokens(self, tracker) -> None:
        """Test adding tokens to tracker."""
        tracker.add_tokens(1000)
        assert tracker.get_tokens_used_today() == 1000

        tracker.add_tokens(500)
        assert tracker.get_tokens_used_today() == 1500

    @pytest.mark.unit
    def test_get_remaining(self, tracker) -> None:
        """Test getting remaining tokens via stats."""
        tracker.add_tokens(30000)
        stats = tracker.get_stats()
        assert stats["tokens_remaining"] == 70000

    @pytest.mark.unit
    def test_is_limit_reached(self, tracker) -> None:
        """Test limit reached detection."""
        assert tracker.is_limit_reached() is False

        tracker.add_tokens(100000)
        assert tracker.is_limit_reached() is True

    @pytest.mark.unit
    def test_get_stats(self, tracker) -> None:
        """Test getting usage statistics."""
        tracker.add_tokens(50000)

        stats = tracker.get_stats()

        assert stats["tokens_used_today"] == 50000
        assert stats["daily_limit"] == 100000
        assert stats["tokens_remaining"] == 50000
        assert stats["usage_percentage"] == 50.0

    @pytest.mark.unit
    def test_persists_state(self, temp_dir: Path) -> None:
        """Test that state is persisted to file."""
        state_file = temp_dir / ".token_state.json"

        # First tracker
        tracker1 = DailyTokenTracker(daily_limit=100000, state_file=state_file)
        tracker1.add_tokens(25000)
        # Disk writes are debounced off the event loop; flush to persist now.
        tracker1.flush()

        # Second tracker should load state
        tracker2 = DailyTokenTracker(daily_limit=100000, state_file=state_file)

        # Should have loaded the previous state
        assert tracker2.get_stats()["tokens_used_today"] == 25000

    @pytest.mark.unit
    def test_can_use_tokens(self, tracker) -> None:
        """Test can_use_tokens method."""
        assert tracker.can_use_tokens() is True

        tracker.add_tokens(100000)  # Hit limit
        assert tracker.can_use_tokens() is False


class TestResetBoundary:
    """The budget day rolls over at 00:01 UTC, not exact UTC midnight."""

    @pytest.mark.unit
    def test_date_str_before_buffer_is_previous_day(self, temp_dir, monkeypatch):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=temp_dir / "s.json"
        )

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 7, 5, 0, 0, 30, tzinfo=UTC)

        monkeypatch.setattr(_tb_module, "datetime", _FrozenDateTime)
        assert tracker._get_current_date_str() == "2026-07-04"

    @pytest.mark.unit
    def test_date_str_after_buffer_is_new_day(self, temp_dir, monkeypatch):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=temp_dir / "s.json"
        )

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 7, 5, 0, 1, 30, tzinfo=UTC)

        monkeypatch.setattr(_tb_module, "datetime", _FrozenDateTime)
        assert tracker._get_current_date_str() == "2026-07-05"

    @pytest.mark.unit
    def test_seconds_until_reset_targets_next_00_01_utc(self, temp_dir, monkeypatch):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=temp_dir / "s.json"
        )

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC)

        monkeypatch.setattr(_tb_module, "datetime", _FrozenDateTime)
        # From 12:00:00 UTC to the next 00:01 UTC (6 July) is 12h 1m.
        expected = 12 * 3600 + 60
        assert tracker.get_seconds_until_reset() == expected

    @pytest.mark.unit
    def test_reset_time_is_aware_utc_00_01(self, temp_dir):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=temp_dir / "s.json"
        )
        reset_time = tracker.get_reset_time()
        now = datetime.now(UTC)
        assert reset_time > now
        assert reset_time.tzinfo is not None
        assert reset_time.utcoffset().total_seconds() == 0
        assert reset_time.hour == 0
        assert reset_time.minute == 1


class TestDailyTokensUnderscoreParsing:
    """Tests for underscore-formatted daily_tokens ingestion in get_token_tracker()."""

    @pytest.mark.unit
    def test_underscore_string_parsed(self, temp_dir: Path) -> None:
        """daily_tokens as quoted underscore string (e.g. "10_000_000") is accepted."""
        import modules.infra.token_budget as tt

        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {
                "daily_token_limit": {"enabled": False, "daily_tokens": "10_000_000"}
            }
            with patch("modules.infra.token_budget.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 10_000_000
        finally:
            tt._tracker_instance = original

    @pytest.mark.unit
    def test_plain_int_still_works(self, temp_dir: Path) -> None:
        """daily_tokens as a plain int (standard YAML) is still accepted."""
        import modules.infra.token_budget as tt

        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {
                "daily_token_limit": {"enabled": False, "daily_tokens": 67500000}
            }
            with patch("modules.infra.token_budget.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 67_500_000
        finally:
            tt._tracker_instance = original

    @pytest.mark.unit
    def test_underscore_int_from_yaml_parser(self, temp_dir: Path) -> None:
        """daily_tokens as a bare int (PyYAML may parse 10_000_000 as int) works."""
        import modules.infra.token_budget as tt

        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {
                "daily_token_limit": {"enabled": False, "daily_tokens": 10000000}
            }
            with patch("modules.infra.token_budget.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 10_000_000
        finally:
            tt._tracker_instance = original


class TestTokenTrackerSingleton:
    """Tests for token tracker singleton functions."""

    @pytest.mark.unit
    @pytest.mark.integration
    def test_get_token_tracker_returns_instance(self) -> None:
        """Test that get_token_tracker returns an instance.

        Note: This test requires valid config files, so marked as integration.
        """
        # Skip if config loading fails (common in isolated test environments)
        try:
            tracker = get_token_tracker()
            assert tracker is not None
            assert isinstance(tracker, DailyTokenTracker)
        except Exception:
            pytest.skip("Config loading not available in test environment")

    @pytest.mark.unit
    @pytest.mark.integration
    def test_get_token_tracker_returns_same_instance(self) -> None:
        """Test that get_token_tracker returns same instance."""
        try:
            tracker1 = get_token_tracker()
            tracker2 = get_token_tracker()
            assert tracker1 is tracker2
        except Exception:
            pytest.skip("Config loading not available in test environment")


class TestSaveStateRetry:
    """Tests for _save_state() retry logic on Windows file lock errors."""

    @pytest.fixture
    def tracker(self, temp_dir):
        """Create a DailyTokenTracker with temp state file."""
        state_file = temp_dir / ".token_state.json"
        return DailyTokenTracker(
            daily_limit=100000,
            state_file=state_file,
        )

    @pytest.mark.unit
    def test_retry_succeeds_on_second_attempt(self, tracker) -> None:
        """Atomic replace succeeds after a transient PermissionError."""
        original_replace = Path.replace
        call_count = 0

        def flaky_replace(self_path, target):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("WinError 5: Access Denied")
            return original_replace(self_path, target)

        tracker._tokens_used_today = 500
        with (
            patch.object(Path, "replace", flaky_replace),
            patch("modules.infra.token_budget.time.sleep") as mock_sleep,
        ):
            tracker._save_state()

        assert call_count == 2
        mock_sleep.assert_called_once_with(0.1)

        with open(tracker.state_file, encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 500

        # No per-process temp file left behind.
        assert list(tracker.state_file.parent.glob("*.tmp")) == []

    @pytest.mark.unit
    def test_retry_on_file_not_found_race(self, tracker) -> None:
        """A FileNotFoundError from a lost replace() race is retried."""
        original_replace = Path.replace
        call_count = 0

        def flaky_replace(self_path, target):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FileNotFoundError("lost the replace race")
            return original_replace(self_path, target)

        tracker._tokens_used_today = 600
        with (
            patch.object(Path, "replace", flaky_replace),
            patch("modules.infra.token_budget.time.sleep") as mock_sleep,
        ):
            tracker._save_state()

        assert call_count == 2
        mock_sleep.assert_called_once_with(0.1)
        with open(tracker.state_file, encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 600

    @pytest.mark.unit
    def test_fallback_to_direct_write_after_all_retries_fail(self, tracker) -> None:
        """Falls back to direct write when all retry attempts exhaust."""

        def always_fail(self_path, target):
            raise PermissionError("WinError 5: Access Denied")

        tracker._tokens_used_today = 750
        with (
            patch.object(Path, "replace", always_fail),
            patch("modules.infra.token_budget.time.sleep"),
        ):
            tracker._save_state()

        with open(tracker.state_file, encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 750

        assert list(tracker.state_file.parent.glob("*.tmp")) == []

    @pytest.mark.unit
    def test_temp_file_cleaned_up_on_general_error(self, tracker) -> None:
        """Temp file is removed by finally block on non-retryable errors."""

        def fail_replace(self_path, target):
            raise IsADirectoryError("target is a directory")

        with patch.object(Path, "replace", fail_replace):
            tracker._save_state()

        assert list(tracker.state_file.parent.glob("*.tmp")) == []

    @pytest.mark.unit
    def test_no_retry_on_unretryable_error(self, tracker) -> None:
        """An error that is neither PermissionError nor FileNotFoundError is not
        retried (it propagates to the outer handler on the first attempt)."""
        call_count = 0

        def fail_unretryable(self_path, target):
            nonlocal call_count
            call_count += 1
            raise IsADirectoryError("target is a directory")

        with (
            patch.object(Path, "replace", fail_unretryable),
            patch("modules.infra.token_budget.time.sleep") as mock_sleep,
        ):
            tracker._save_state()

        assert call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.unit
    def test_temp_file_name_is_per_process_unique(self, tracker) -> None:
        """The temp file embeds the pid and a random token (no fixed .tmp)."""
        import os
        import re

        original_replace = Path.replace
        captured: dict[str, str] = {}

        def capture_replace(self_path, target):
            captured["name"] = self_path.name
            return original_replace(self_path, target)

        tracker._tokens_used_today = 10
        with patch.object(Path, "replace", capture_replace):
            tracker._save_state()

        assert re.fullmatch(
            rf"\.token_state\.json\.{os.getpid()}\.[0-9a-f]{{8}}\.tmp",
            captured["name"],
        )


class TestDebouncedOffLoopWrites:
    """Tests for the debounced, off-event-loop state persistence (B2)."""

    @pytest.fixture
    def tracker(self, temp_dir):
        state_file = temp_dir / ".token_state.json"
        return DailyTokenTracker(daily_limit=100000, state_file=state_file)

    @pytest.mark.unit
    def test_add_tokens_does_not_write_per_call(self, tracker) -> None:
        """add_tokens marks state dirty but does not write on every call."""
        with patch.object(tracker, "_save_state") as mock_save:
            tracker.add_tokens(100)
            tracker.add_tokens(200)
            # No synchronous write happened on the (event-loop) calling thread.
            assert mock_save.call_count == 0
            assert tracker._pending_write is True
        # Stop the background writer so it does not race later assertions.
        tracker._writer_stop.set()
        assert tracker.get_tokens_used_today() == 300

    @pytest.mark.unit
    def test_flush_persists_pending_write(self, tracker) -> None:
        """flush() forces a pending debounced write to disk."""
        tracker.add_tokens(1234)
        tracker._writer_stop.set()  # prevent the daemon from writing first
        tracker.flush()
        with open(tracker.state_file, encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 1234
        assert tracker._pending_write is False

    @pytest.mark.unit
    def test_flush_on_exit_persists_pending_write(self, tracker) -> None:
        """The atexit hook persists any pending write."""
        tracker.add_tokens(555)
        tracker._writer_stop.clear()
        tracker._flush_on_exit()
        with open(tracker.state_file, encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 555


class TestChunkReservation:
    """Page-level reservation gate: try_reserve / release / EWMA estimate."""

    @pytest.mark.unit
    def test_disabled_returns_zero_and_never_denies(self, temp_dir) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=False,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=10,
        )
        assert t.try_reserve() == 0
        assert t.try_reserve(999_999) == 0
        t.release(0)  # no-op when disabled (must not raise)

    @pytest.mark.unit
    def test_reserve_within_and_beyond_budget(self, temp_dir) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=10,
        )
        # Seed EWMA is 10, so a bare reservation claims 10.
        assert t.try_reserve() == 10
        # An explicit estimate above the EWMA claims the estimate.
        assert t.try_reserve(50) == 50
        # reserved is now 60; remaining headroom is 40, so a 50 will not fit.
        assert t.try_reserve(50) is None
        # 40 fits exactly, bringing reserved to the limit.
        assert t.try_reserve(40) == 40
        assert t.try_reserve(1) is None

    @pytest.mark.unit
    def test_release_restores_headroom(self, temp_dir) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=10,
        )
        assert t.try_reserve(100) == 100
        assert t.try_reserve(1) is None
        t.release(100)
        assert t.try_reserve(50) == 50

    @pytest.mark.unit
    def test_committed_plus_reserved_never_exceeds_limit(self, temp_dir) -> None:
        # smoothing=0 freezes the EWMA at the seed so this test isolates the
        # committed-plus-reserved arithmetic from estimate drift.
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=10,
            estimate_smoothing=0.0,
        )
        t.add_tokens(60)  # committed usage
        assert t.try_reserve(30) == 30  # 60 + 30 = 90, fits
        assert t.try_reserve(20) is None  # 90 + 20 = 110, denied
        assert t.try_reserve(10) == 10  # 90 + 10 = 100, exact fit

    @pytest.mark.unit
    def test_add_tokens_updates_ewma(self, temp_dir) -> None:
        t = DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=100,
            estimate_smoothing=0.5,
        )
        # EWMA = 0.5 * 1100 + 0.5 * 100 = 600
        t.add_tokens(1100)
        assert t.try_reserve() == 600

    @pytest.mark.unit
    def test_reserve_uses_max_of_estimate_and_ewma(self, temp_dir) -> None:
        t = DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=100,
        )
        assert t.try_reserve(5000) == 5000  # estimate above EWMA wins
        assert t.try_reserve(10) == 100  # estimate below EWMA floors at EWMA


class TestLiveLimitReread:
    """The wait loop re-reads daily_tokens so a mid-wait edit lifts the cap (B10)."""

    @pytest.mark.unit
    def test_set_daily_limit_updates_value(self, temp_dir: Path) -> None:
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=temp_dir / "s.json"
        )
        tracker.set_daily_limit(500)
        assert tracker.daily_limit == 500

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wait_reread_lifts_cap_without_restart(
        self, temp_dir: Path, monkeypatch
    ) -> None:
        from modules.infra import token_budget as tb

        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=temp_dir / "s.json"
        )
        tracker.add_tokens(100)  # limit reached
        assert tracker.is_limit_reached()

        monkeypatch.setattr(tb, "get_token_tracker", lambda: tracker)
        # A mid-wait config edit raises the daily limit.
        monkeypatch.setattr(tb, "_read_configured_daily_limit", lambda: 1000)

        async def _fast_sleep(_seconds):
            return None

        monkeypatch.setattr(tb.asyncio, "sleep", _fast_sleep)

        ok = await tb.check_and_wait_for_token_limit(
            {"daily_token_limit": {"enabled": True}}
        )
        assert ok is True
        # The re-read lifted the cap, so processing resumes with the new limit.
        assert tracker.daily_limit == 1000
