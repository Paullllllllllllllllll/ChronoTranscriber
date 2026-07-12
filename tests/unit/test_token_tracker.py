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


class TestWouldBlockNextPage:
    """Reservation-aware predicate: blocks when the remaining budget cannot
    cover the current per-page reservation estimate, even though the hard limit
    is not yet reached (CT-6)."""

    @pytest.mark.unit
    def test_disabled_never_blocks(self, temp_dir: Path) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=False,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=30,
        )
        assert t.would_block_next_page() is False

    @pytest.mark.unit
    def test_blocks_near_cap_while_limit_not_reached(self, temp_dir: Path) -> None:
        # smoothing=0 freezes the EWMA estimate at the seed (30).
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=30,
            estimate_smoothing=0.0,
        )
        t.add_tokens(80)  # remaining 20 < per-page estimate 30
        # This is the crux of the bug: the hard limit is NOT reached, yet the
        # next page cannot be admitted. would_block_next_page catches it.
        assert t.is_limit_reached() is False
        assert t.would_block_next_page() is True

    @pytest.mark.unit
    def test_does_not_block_with_ample_budget(self, temp_dir: Path) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=30,
            estimate_smoothing=0.0,
        )
        t.add_tokens(50)  # remaining 50 >= estimate 30
        assert t.would_block_next_page() is False

    @pytest.mark.unit
    def test_matches_try_reserve_admission(self, temp_dir: Path) -> None:
        # The predicate must agree with try_reserve(): if it says "blocked",
        # a bare reservation is denied, and vice versa.
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=30,
            estimate_smoothing=0.0,
        )
        t.add_tokens(80)
        assert t.would_block_next_page() is True
        assert t.try_reserve() is None

    @pytest.mark.unit
    def test_unblocks_after_day_rollover(self, temp_dir: Path, monkeypatch) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=30,
            estimate_smoothing=0.0,
        )
        t.add_tokens(80)
        assert t.would_block_next_page() is True
        # Simulate the daily reset: the internal date-check zeroes usage.
        monkeypatch.setattr(t, "_get_current_date_str", lambda: "2099-01-01")
        assert t.would_block_next_page() is False


class TestEstimateExceedsDailyLimit:
    """Fast-fail predicate: the per-page estimate alone exceeds the whole daily
    limit, so even a fresh daily reset cannot admit the next page (CT-11)."""

    @pytest.mark.unit
    def test_disabled_never_blocks(self, temp_dir: Path) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=False,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=500,
        )
        assert t.estimate_exceeds_daily_limit() is False

    @pytest.mark.unit
    def test_true_when_seed_estimate_exceeds_limit(self, temp_dir: Path) -> None:
        # Per-page estimate (seed 150) > daily limit (100): no reset can help.
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=150,
            estimate_smoothing=0.0,
        )
        assert t.estimate_exceeds_daily_limit() is True
        # Even with a completely fresh budget the page would still be blocked.
        assert t.would_block_next_page() is True

    @pytest.mark.unit
    def test_false_when_estimate_fits_the_daily_limit(self, temp_dir: Path) -> None:
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=30,
            estimate_smoothing=0.0,
        )
        # Blocked only because usage is high, not because the estimate is too
        # big — a reset WOULD help, so this stays False.
        t.add_tokens(90)
        assert t.would_block_next_page() is True
        assert t.estimate_exceeds_daily_limit() is False


# --------------------------------------------------------------------------- #
# Per-provider-key accounting (schema v2): bucket stamping, pool caps, scope
# --------------------------------------------------------------------------- #

_OAI = "OPENAI_API_KEY"
_OAI2 = "OPENAI_API_KEY_2"


def _pooled_tracker(temp_dir: Path, **kwargs) -> DailyTokenTracker:
    """Standalone tracker with small OpenAI pool caps for gate tests."""
    defaults = dict(
        daily_limit=1_000_000,
        enabled=True,
        state_file=temp_dir / "s.json",
        chunk_estimate_seed=1,
        estimate_smoothing=0.0,
        daily_scope="pooled",
        per_key_pool_caps_enabled=True,
        pool_caps={("openai", "large"): 100, ("openai", "small"): 100},
    )
    defaults.update(kwargs)
    return DailyTokenTracker(**defaults)


class TestBucketStamping:
    """add_tokens attributes usage to the (provider, key_env, pool) bucket."""

    @pytest.mark.unit
    def test_stamp_resolves_openai_pool_bucket(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import BucketKey

        t = _pooled_tracker(temp_dir)
        t.add_tokens(50, provider="openai", key_env=_OAI, model="gpt-4o")
        bucket = BucketKey("openai", _OAI, "large")
        assert t._bucket_used_today.get(bucket) == 50
        # The plain daily sum stays a flat total across buckets.
        assert t.get_own_tokens_used_today() == 50

    @pytest.mark.unit
    def test_missing_stamp_lands_unattributed(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import UNATTRIBUTED_BUCKET

        t = _pooled_tracker(temp_dir)
        t.add_tokens(40)  # no stamp
        t.add_tokens(10, provider="openai")  # key_env missing -> unattributed
        assert t._bucket_used_today.get(UNATTRIBUTED_BUCKET) == 50


class TestPerKeyPoolGate:
    """The primary per-key pool cap blocks one key while another admits."""

    @pytest.mark.unit
    def test_key1_blocks_while_key2_admits(self, temp_dir: Path) -> None:
        t = _pooled_tracker(temp_dir)
        # Fill key 1's large pool to the cap.
        t.add_tokens(90, provider="openai", key_env=_OAI, model="gpt-4o")
        # Key 1: 90 + est 20 > cap 100 -> denied.
        assert (
            t.try_reserve(20, provider="openai", key_env=_OAI, model="gpt-4o") is None
        )
        # Key 2 has its own fresh pool -> admitted.
        assert t.try_reserve(20, provider="openai", key_env=_OAI2, model="gpt-4o") == 20

    @pytest.mark.unit
    def test_pools_are_independent_per_model(self, temp_dir: Path) -> None:
        t = _pooled_tracker(temp_dir)
        # Exhaust the large pool on key 1.
        t.add_tokens(100, provider="openai", key_env=_OAI, model="gpt-4o")
        assert (
            t.try_reserve(10, provider="openai", key_env=_OAI, model="gpt-4o") is None
        )
        # A small-pool model on the same key uses a different pool -> admitted.
        assert (
            t.try_reserve(10, provider="openai", key_env=_OAI, model="gpt-4o-mini")
            == 10
        )

    @pytest.mark.unit
    def test_disabled_pool_caps_do_not_gate(self, temp_dir: Path) -> None:
        t = _pooled_tracker(temp_dir, per_key_pool_caps_enabled=False)
        t.add_tokens(100, provider="openai", key_env=_OAI, model="gpt-4o")
        # No per-key gate; combined cap is huge -> admitted.
        assert t.try_reserve(50, provider="openai", key_env=_OAI, model="gpt-4o") == 50


class TestScopeAndPoollessRegression:
    """The bug fix: a pool-less bucket is never blocked by OpenAI usage."""

    @pytest.mark.unit
    def test_custom_admitted_when_openai_and_combined_exhausted(
        self, temp_dir: Path
    ) -> None:
        # daily_limit small so the COMBINED cap is also exhausted.
        t = _pooled_tracker(temp_dir, daily_limit=100)
        # Exhaust OpenAI key 1's large pool AND the combined cap.
        t.add_tokens(100, provider="openai", key_env=_OAI, model="gpt-4o")
        assert (
            t.try_reserve(20, provider="openai", key_env=_OAI, model="gpt-4o") is None
        )
        assert t.is_limit_reached() is True  # combined exhausted too
        # A custom/local endpoint (pool None) must still be admitted under
        # scope=pooled, even with OpenAI pools AND the combined cap spent.
        assert (
            t.try_reserve(50, provider="custom", key_env="UZH_KEY", model="local-ocr")
            == 50
        )
        assert (
            t.would_block_next_page(
                provider="custom", key_env="UZH_KEY", model="local-ocr"
            )
            is False
        )

    @pytest.mark.unit
    def test_scope_all_restores_legacy_blocking(self, temp_dir: Path) -> None:
        t = _pooled_tracker(temp_dir, daily_limit=100, daily_scope="all")
        t.add_tokens(100, provider="openai", key_env=_OAI, model="gpt-4o")
        # scope=all: the combined cap governs even the pool-less custom bucket.
        assert (
            t.try_reserve(50, provider="custom", key_env="UZH_KEY", model="local-ocr")
            is None
        )

    @pytest.mark.unit
    def test_unstamped_keeps_combined_semantics(self, temp_dir: Path) -> None:
        t = _pooled_tracker(temp_dir, daily_limit=100)
        t.add_tokens(100)  # unattributed
        # Un-stamped calls keep today's combined-only blocking under scope=pooled.
        assert t.try_reserve(10) is None


class TestPrivateStateBucketPersistence:
    """The private state file round-trips per-bucket counts and adopts legacy."""

    @pytest.mark.unit
    def test_buckets_persist_and_reload(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import BucketKey

        state_file = temp_dir / "s.json"
        t1 = _pooled_tracker(temp_dir, state_file=state_file)
        t1.add_tokens(70, provider="openai", key_env=_OAI, model="gpt-4o")
        t1.add_tokens(30, provider="custom", key_env="UZH_KEY", model="local-ocr")
        t1._writer_stop.set()
        t1.flush()

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 100
        assert data["buckets"]["openai|OPENAI_API_KEY|large"] == 70
        assert data["buckets"]["custom|UZH_KEY|"] == 30

        t2 = _pooled_tracker(temp_dir, state_file=state_file)
        assert t2._bucket_used_today.get(BucketKey("openai", _OAI, "large")) == 70
        assert t2._bucket_used_today.get(BucketKey("custom", "UZH_KEY", None)) == 30

    @pytest.mark.unit
    def test_legacy_state_adopted_as_unattributed(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import UNATTRIBUTED_BUCKET

        state_file = temp_dir / "s.json"
        # Legacy state without a "buckets" object.
        today = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=temp_dir / "throwaway.json"
        )._get_current_date_str()
        state_file.write_text(
            json.dumps({"date": today, "tokens_used": 500}), encoding="utf-8"
        )
        t = _pooled_tracker(temp_dir, state_file=state_file)
        assert t._bucket_used_today.get(UNATTRIBUTED_BUCKET) == 500
        assert t.get_own_tokens_used_today() == 500


class TestPolicyConfigParsing:
    """get_token_tracker parses scope + per_key_pool_caps."""

    def _tracker_from_cfg(self, cfg: dict):
        import modules.infra.token_budget as tt

        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            with patch("modules.infra.token_budget.get_config_service") as mock_svc:
                svc = mock_svc.return_value
                svc.get_concurrency_config.return_value = cfg
                svc.get_paths_config.return_value = {"general": {}}
                return tt.get_token_tracker()
        finally:
            tt._tracker_instance = original

    @pytest.mark.unit
    def test_absent_block_uses_defaults(self, temp_dir: Path) -> None:
        t = self._tracker_from_cfg(
            {"daily_token_limit": {"enabled": True, "daily_tokens": 100}}
        )
        assert t._scope == "pooled"
        assert t._per_key_pool_caps_enabled is True
        # No configured caps: resolution falls back to the vendored defaults.
        assert t._pool_caps == {}
        assert t._pool_cap_for("openai", "large") == 975_000
        assert t._pool_cap_for("openai", "small") == 9_750_000

    @pytest.mark.unit
    def test_scope_and_partial_caps_parsed(self, temp_dir: Path) -> None:
        t = self._tracker_from_cfg(
            {
                "daily_token_limit": {
                    "enabled": True,
                    "daily_tokens": 100,
                    "scope": "all",
                    "per_key_pool_caps": {
                        "enabled": False,
                        "openai": {"large": "500_000"},
                    },
                }
            }
        )
        assert t._scope == "all"
        assert t._per_key_pool_caps_enabled is False
        assert t._pool_caps[("openai", "large")] == 500_000
        # Unspecified pool falls back to its vendored default.
        assert t._pool_cap_for("openai", "small") == 9_750_000

    @pytest.mark.unit
    def test_invalid_scope_falls_back_to_pooled(self, temp_dir: Path) -> None:
        t = self._tracker_from_cfg(
            {
                "daily_token_limit": {
                    "enabled": True,
                    "daily_tokens": 100,
                    "scope": "nonsense",
                }
            }
        )
        assert t._scope == "pooled"

    @pytest.mark.unit
    def test_mapping_form_parsed_into_caps_and_pools(self) -> None:
        from modules.infra.token_budget import _parse_pool_caps

        enabled, caps, pools = _parse_pool_caps(
            {
                "per_key_pool_caps": {
                    "enabled": True,
                    "openai": {
                        "small": 9_750_000,  # bare int: cap only
                        "large": {"cap": "975_000"},  # mapping, cap only
                    },
                    "myhost": {
                        "standard": {
                            "cap": 5_000_000,
                            "models": ["my-model", 42, "  "],
                        },
                        "free": {"models": ["other-model"]},  # cap-less
                    },
                    "broken": "not-a-dict",  # dropped
                }
            }
        )
        assert enabled is True
        assert caps[("openai", "small")] == 9_750_000
        assert caps[("openai", "large")] == 975_000
        assert caps[("myhost", "standard")] == 5_000_000
        assert ("myhost", "free") not in caps
        # Only entries carrying `models` define pools; junk prefixes dropped.
        assert pools == {"myhost": {"standard": ["my-model"], "free": ["other-model"]}}


class TestDefinablePools:
    """Custom pool definitions: mapping form, built-in replacement, uncapped."""

    @pytest.mark.unit
    def test_custom_provider_pool_derived_capped_enforced(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import BucketKey

        t = DailyTokenTracker(
            daily_limit=1_000_000,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=1,
            estimate_smoothing=0.0,
            daily_scope="pooled",
            pool_caps={("myhost", "standard"): 100},
            provider_pools={"myhost": {"standard": ["my-model"]}},
        )
        # Prefix match at a separator boundary derives the custom pool.
        t.add_tokens(90, provider="myhost", key_env="MY_KEY", model="my-model-v2")
        bucket = BucketKey("myhost", "MY_KEY", "standard")
        assert t._bucket_used_today.get(bucket) == 90
        # The custom cap gates the pool: 90 + 20 > 100 -> denied.
        assert (
            t.try_reserve(20, provider="myhost", key_env="MY_KEY", model="my-model")
            is None
        )
        # A second key for the same host has its own fresh pool.
        assert (
            t.try_reserve(20, provider="myhost", key_env="MY_KEY_2", model="my-model")
            == 20
        )
        # Pool-less regression still holds: an unlisted model on the same host
        # derives NO pool and is admitted under scope=pooled despite the
        # exhausted pool.
        assert (
            t.try_reserve(50, provider="myhost", key_env="MY_KEY", model="unlisted")
            == 50
        )

    @pytest.mark.unit
    def test_configured_openai_models_replace_builtins(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import BucketKey

        t = DailyTokenTracker(
            daily_limit=1_000_000,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=1,
            estimate_smoothing=0.0,
            pool_caps={("openai", "tiny"): 100},
            provider_pools={"openai": {"tiny": ["gpt-4o-mini"]}},
        )
        # gpt-4o is in the BUILT-IN large pool, but configured pools REPLACE
        # the built-ins for openai -> it now derives no pool at all.
        t.add_tokens(40, provider="openai", key_env=_OAI, model="gpt-4o")
        assert t._bucket_used_today.get(BucketKey("openai", _OAI, None)) == 40
        # The configured pool still derives and gates.
        t.add_tokens(95, provider="openai", key_env=_OAI, model="gpt-4o-mini")
        assert t._bucket_used_today.get(BucketKey("openai", _OAI, "tiny")) == 95
        assert (
            t.try_reserve(10, provider="openai", key_env=_OAI, model="gpt-4o-mini")
            is None
        )

    @pytest.mark.unit
    def test_capless_pool_tracked_but_uncapped(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import BucketKey

        t = DailyTokenTracker(
            daily_limit=1_000_000,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=1,
            estimate_smoothing=0.0,
            provider_pools={"myhost": {"free": ["other-model"]}},
            # No cap for (myhost, free); "free" is unknown to DEFAULT_POOL_CAPS.
        )
        t.add_tokens(10_000, provider="myhost", key_env="MY_KEY", model="other-model")
        bucket = BucketKey("myhost", "MY_KEY", "free")
        # Tracked under its derived pool...
        assert t._bucket_used_today.get(bucket) == 10_000
        # ...but never blocked by a pool cap (and scope=pooled: the combined
        # gate applies to pooled buckets, with plenty of combined headroom).
        assert (
            t.try_reserve(500, provider="myhost", key_env="MY_KEY", model="other-model")
            == 500
        )
        assert (
            t.would_block_next_page(
                provider="myhost", key_env="MY_KEY", model="other-model"
            )
            is False
        )

    @pytest.mark.unit
    def test_set_token_policy_refreshes_pools(self, temp_dir: Path) -> None:
        """The wait loop's policy refresh swaps compiled pools live."""
        t = _pooled_tracker(temp_dir)
        # Initially built-ins apply: gpt-4o derives "large".
        assert _pooled_tracker(temp_dir, state_file=temp_dir / "s2.json") is not None
        t.set_token_policy(
            pool_caps={("openai", "solo"): 50},
            provider_pools={"openai": {"solo": ["gpt-4o"]}},
        )
        # After the refresh gpt-4o derives the new label and its new cap.
        t.add_tokens(45, provider="openai", key_env=_OAI, model="gpt-4o")
        assert (
            t.try_reserve(10, provider="openai", key_env=_OAI, model="gpt-4o") is None
        )


class TestDegradedPerBucketPreservation:
    """Degraded shared sync preserves un-pushed per-bucket deltas."""

    @pytest.mark.unit
    def test_per_bucket_deltas_survive_degraded_then_land(self, temp_dir: Path) -> None:
        from modules.infra.shared_ledger import BucketKey, UsageSnapshot

        class _FakeLedger:
            def __init__(self) -> None:
                self.rows: dict = {}
                self.fail = True

            def seed_usage(self, own_total, own_buckets=None):
                if self.fail:
                    return None
                for b, n in (own_buckets or {}).items():
                    self.rows[b] = max(self.rows.get(b, 0), int(n))
                return self._snap()

            def sync_usage(self, deltas):
                if self.fail:
                    return None
                for b, n in deltas.items():
                    self.rows[b] = self.rows.get(b, 0) + int(n)
                return self._snap()

            def _snap(self):
                total = sum(self.rows.values())
                buckets = {
                    b: n for b, n in self.rows.items() if b.provider != "unattributed"
                }
                return UsageSnapshot(total, total, buckets, dict(self.rows))

            def read_breakdown(self):
                return (
                    None
                    if self.fail
                    else {"chronotranscriber": sum(self.rows.values())}
                )

        t = DailyTokenTracker(
            daily_limit=10_000_000,
            enabled=True,
            state_file=temp_dir / "s.json",
            chunk_estimate_seed=1,
            estimate_smoothing=0.0,
            shared_enabled=True,
            shared_ledger_dir=temp_dir / "ledger",
        )
        t._writer_stop.set()
        fake = _FakeLedger()
        with t._lock:
            t._ledger = fake
            t._seeded = False
            t._combined_total = 0
            t._unsynced_deltas = {}
            t._bucket_totals = {}
            t._ledger_degraded = False

        k1 = BucketKey("openai", _OAI, "large")
        k2 = BucketKey("custom", "UZH_KEY", None)
        t.add_tokens(100, provider="openai", key_env=_OAI, model="gpt-4o")
        t.sync_ledger_now()  # seed fails -> degraded, deltas preserved
        assert t._ledger_degraded is True
        t.add_tokens(30, provider="custom", key_env="UZH_KEY", model="local-ocr")
        assert t._unsynced_deltas.get(k1) == 100
        assert t._unsynced_deltas.get(k2) == 30

        fake.fail = False
        t.sync_ledger_now()  # recovery: both per-bucket deltas land
        assert t._ledger_degraded is False
        assert fake.rows.get(k1) == 100
        assert fake.rows.get(k2) == 30
        assert not any(t._unsynced_deltas.values())
