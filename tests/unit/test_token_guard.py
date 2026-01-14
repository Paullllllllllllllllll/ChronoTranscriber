"""Unit tests for modules/core/token_guard.py."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta


class TestCheckAndWaitForTokenLimit:
    """Tests for check_and_wait_for_token_limit function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_true_when_disabled(self):
        """Test that function returns True immediately when token limit is disabled."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        config = {"daily_token_limit": {"enabled": False}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_true_when_no_config(self):
        """Test that function returns True when no token config exists."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        config = {}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_true_when_limit_not_reached(self):
        """Test that function returns True when limit is not reached."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        mock_tracker = MagicMock()
        mock_tracker.is_limit_reached.return_value = False
        
        config = {"daily_token_limit": {"enabled": True}}
        
        with patch('modules.infra.token_tracker.get_token_tracker', return_value=mock_tracker):
            result = await check_and_wait_for_token_limit(config)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_waits_when_limit_reached(self):
        """Test that function waits when limit is reached and returns True after reset."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        mock_tracker = MagicMock()
        # First call returns True (limit reached), second returns False (limit reset)
        mock_tracker.is_limit_reached.side_effect = [True, False]
        mock_tracker.get_stats.return_value = {"tokens_used_today": 1000, "daily_limit": 1000}
        mock_tracker.get_reset_time.return_value = datetime.now() + timedelta(seconds=1)
        mock_tracker.get_seconds_until_reset.return_value = 0.1  # Very short wait
        
        config = {"daily_token_limit": {"enabled": True}}
        
        with patch('modules.infra.token_tracker.get_token_tracker', return_value=mock_tracker):
            result = await check_and_wait_for_token_limit(config)
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_false_on_keyboard_interrupt(self):
        """Test that function returns False when interrupted."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        mock_tracker = MagicMock()
        mock_tracker.is_limit_reached.return_value = True
        mock_tracker.get_stats.return_value = {"tokens_used_today": 1000, "daily_limit": 1000}
        mock_tracker.get_reset_time.return_value = datetime.now() + timedelta(hours=1)
        mock_tracker.get_seconds_until_reset.return_value = 3600
        
        config = {"daily_token_limit": {"enabled": True}}
        
        async def mock_sleep(duration):
            raise KeyboardInterrupt()
        
        with patch('modules.infra.token_tracker.get_token_tracker', return_value=mock_tracker):
            with patch('asyncio.sleep', side_effect=KeyboardInterrupt):
                result = await check_and_wait_for_token_limit(config)
                assert result is False


class TestTokenGuardConfiguration:
    """Tests for token guard configuration handling."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_empty_daily_token_limit(self):
        """Test graceful handling of empty daily_token_limit dict."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        config = {"daily_token_limit": {}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_enabled_as_string_false(self):
        """Test handling enabled as various falsy values."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        # Empty string is falsy
        config = {"daily_token_limit": {"enabled": ""}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handles_enabled_as_zero(self):
        """Test handling enabled as zero."""
        from modules.core.token_guard import check_and_wait_for_token_limit
        
        config = {"daily_token_limit": {"enabled": 0}}
        result = await check_and_wait_for_token_limit(config)
        assert result is True
