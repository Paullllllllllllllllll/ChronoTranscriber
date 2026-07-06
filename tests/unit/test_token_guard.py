"""Unit tests for the async wait-for-reset helper in modules/infra/token_budget.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


class TestCheckAndWaitForTokenLimit:
    """Tests for check_and_wait_for_token_limit function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_true_when_disabled(self) -> None:
        """Test that function returns True immediately when token limit is disabled."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        config = {"daily_token_limit": {"enabled": False}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_true_when_no_config(self) -> None:
        """Test that function returns True when no token config exists."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        config = {}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_true_when_limit_not_reached(self) -> None:
        """Test that function returns True when limit is not reached."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        mock_tracker.is_limit_reached.return_value = False

        config = {"daily_token_limit": {"enabled": True}}

        with patch(
            "modules.infra.token_budget.get_token_tracker", return_value=mock_tracker
        ):
            result = await check_and_wait_for_token_limit(config)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_waits_when_limit_reached(self) -> None:
        """Function waits when limit is reached and returns True after reset."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        # First call returns True (limit reached), second returns False (limit reset)
        mock_tracker.is_limit_reached.side_effect = [True, False]
        mock_tracker.get_stats.return_value = {
            "tokens_used_today": 1000,
            "daily_limit": 1000,
        }
        mock_tracker.get_reset_time.return_value = datetime.now() + timedelta(seconds=1)
        mock_tracker.get_seconds_until_reset.return_value = 0.1  # Very short wait

        config = {"daily_token_limit": {"enabled": True}}

        with patch(
            "modules.infra.token_budget.get_token_tracker", return_value=mock_tracker
        ):
            result = await check_and_wait_for_token_limit(config)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_false_on_keyboard_interrupt(self) -> None:
        """Test that function returns False when interrupted."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        mock_tracker.is_limit_reached.return_value = True
        mock_tracker.get_stats.return_value = {
            "tokens_used_today": 1000,
            "daily_limit": 1000,
        }
        mock_tracker.get_reset_time.return_value = datetime.now() + timedelta(hours=1)
        mock_tracker.get_seconds_until_reset.return_value = 3600

        config = {"daily_token_limit": {"enabled": True}}

        async def mock_sleep(duration):
            raise KeyboardInterrupt()

        with (
            patch(
                "modules.infra.token_budget.get_token_tracker",
                return_value=mock_tracker,
            ),
            patch("asyncio.sleep", side_effect=KeyboardInterrupt),
        ):
            result = await check_and_wait_for_token_limit(config)
            assert result is False


class TestReservationAwareWait:
    """The mid-document re-pass loop passes reservation_aware=True so the wait
    treats reservation-blocked-near-the-cap as limit-reached (CT-6)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reservation_aware_returns_immediately_when_not_blocked(
        self,
    ) -> None:
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        mock_tracker.would_block_next_page.return_value = False

        config = {"daily_token_limit": {"enabled": True}}
        with patch(
            "modules.infra.token_budget.get_token_tracker", return_value=mock_tracker
        ):
            result = await check_and_wait_for_token_limit(
                config, reservation_aware=True
            )
        assert result is True
        # The reservation predicate — not is_limit_reached — gates the wait.
        mock_tracker.would_block_next_page.assert_called()
        mock_tracker.is_limit_reached.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reservation_aware_waits_then_resumes_after_reset(self) -> None:
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        # Blocked on entry, cleared after the (short) wait — mirrors the daily
        # reset freeing enough budget to admit the next page.
        mock_tracker.would_block_next_page.side_effect = [True, False]
        # The per-page estimate fits the daily limit, so a reset can help (no
        # fast-fail): the wait proceeds and resumes after the reset.
        mock_tracker.estimate_exceeds_daily_limit.return_value = False
        mock_tracker.get_stats.return_value = {
            "tokens_used_today": 95,
            "daily_limit": 100,
        }
        mock_tracker.get_reset_time.return_value = datetime.now() + timedelta(seconds=1)
        mock_tracker.get_seconds_until_reset.return_value = 0.1

        config = {"daily_token_limit": {"enabled": True}}
        with patch(
            "modules.infra.token_budget.get_token_tracker", return_value=mock_tracker
        ):
            result = await check_and_wait_for_token_limit(
                config, reservation_aware=True
            )
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fast_fails_when_estimate_exceeds_daily_limit(self) -> None:
        # When a single page's estimate exceeds the whole daily limit, no reset
        # can admit it. The wait must give up immediately (return False) WITHOUT
        # sleeping, instead of burning ~48 h across two useless resets (CT-11).
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        mock_tracker.would_block_next_page.return_value = True
        mock_tracker.estimate_exceeds_daily_limit.return_value = True
        mock_tracker.get_stats.return_value = {
            "tokens_used_today": 0,
            "daily_limit": 100,
        }

        config = {"daily_token_limit": {"enabled": True}}
        with (
            patch(
                "modules.infra.token_budget.get_token_tracker",
                return_value=mock_tracker,
            ),
            patch("asyncio.sleep", side_effect=AssertionError("must not wait")),
        ):
            result = await check_and_wait_for_token_limit(
                config, reservation_aware=True
            )
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_countdown_expiry_rechecks_and_gives_up_when_still_blocked(
        self,
    ) -> None:
        # If the estimate does not exceed the daily limit on entry (no fast-fail)
        # but the page stays blocked through the whole countdown, the
        # post-countdown fallthrough must re-check _still_blocked() and return
        # False rather than unconditionally returning True and spinning the
        # caller's re-pass loop into a second full-day wait (CT-11).
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        mock_tracker._shared_enabled = False
        mock_tracker.would_block_next_page.return_value = True  # never clears
        mock_tracker.estimate_exceeds_daily_limit.return_value = False
        mock_tracker.get_stats.return_value = {
            "tokens_used_today": 95,
            "daily_limit": 100,
        }
        mock_tracker.get_reset_time.return_value = datetime.now() + timedelta(seconds=1)
        mock_tracker.get_seconds_until_reset.return_value = 0.05

        config = {"daily_token_limit": {"enabled": True}}
        with (
            patch(
                "modules.infra.token_budget.get_token_tracker",
                return_value=mock_tracker,
            ),
            patch(
                "modules.infra.token_budget._read_configured_daily_limit",
                return_value=None,
            ),
        ):
            result = await check_and_wait_for_token_limit(
                config, reservation_aware=True
            )
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_default_ignores_reservation_block(self) -> None:
        # Documents why the bug existed: the default (non-reservation-aware)
        # path consults only is_limit_reached, so a reservation-blocked tracker
        # whose hard limit is not reached returns True immediately (no wait).
        from modules.infra.token_budget import check_and_wait_for_token_limit

        mock_tracker = MagicMock()
        mock_tracker.is_limit_reached.return_value = False
        mock_tracker.would_block_next_page.return_value = True

        config = {"daily_token_limit": {"enabled": True}}
        with patch(
            "modules.infra.token_budget.get_token_tracker", return_value=mock_tracker
        ):
            result = await check_and_wait_for_token_limit(config)
        assert result is True
        mock_tracker.would_block_next_page.assert_not_called()


class TestTokenGuardConfiguration:
    """Tests for token guard configuration handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_empty_daily_token_limit(self) -> None:
        """Test graceful handling of empty daily_token_limit dict."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        config = {"daily_token_limit": {}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_enabled_as_string_false(self) -> None:
        """Test handling enabled as various falsy values."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        # Empty string is falsy
        config = {"daily_token_limit": {"enabled": ""}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_enabled_as_zero(self) -> None:
        """Test handling enabled as zero."""
        from modules.infra.token_budget import check_and_wait_for_token_limit

        config = {"daily_token_limit": {"enabled": 0}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True
