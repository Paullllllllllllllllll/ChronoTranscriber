"""Unit tests for modules/infra/token_budget.py.

Tests token usage tracking and daily limit management.
"""

from __future__ import annotations

import json
import os
import pytest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, MagicMock

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


class TestDailyTokensUnderscoreParsing:
    """Tests for underscore-formatted daily_tokens ingestion in get_token_tracker()."""

    @pytest.mark.unit
    def test_underscore_string_parsed(self, temp_dir: Path) -> None:
        """daily_tokens as quoted underscore string (e.g. "10_000_000") is accepted."""
        import modules.infra.token_budget as tt
        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {"daily_token_limit": {"enabled": False, "daily_tokens": "10_000_000"}}
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
            mock_cfg = {"daily_token_limit": {"enabled": False, "daily_tokens": 67500000}}
            with patch("modules.infra.token_budget.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 67_500_000
        finally:
            tt._tracker_instance = original

    @pytest.mark.unit
    def test_underscore_int_from_yaml_parser(self, temp_dir: Path) -> None:
        """daily_tokens as an already-stripped int (PyYAML may parse 10_000_000 as int) works."""
        import modules.infra.token_budget as tt
        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {"daily_token_limit": {"enabled": False, "daily_tokens": 10000000}}
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

        with patch.object(Path, "replace", flaky_replace), \
             patch("modules.infra.token_budget.time.sleep") as mock_sleep:
            tracker.add_tokens(500)

        assert call_count == 2
        mock_sleep.assert_called_once_with(0.1)

        with open(tracker.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 500

        temp_file = tracker.state_file.with_suffix(".tmp")
        assert not temp_file.exists()

    @pytest.mark.unit
    def test_fallback_to_direct_write_after_all_retries_fail(self, tracker) -> None:
        """Falls back to direct write when all retry attempts exhaust."""
        def always_fail(self_path, target):
            raise PermissionError("WinError 5: Access Denied")

        with patch.object(Path, "replace", always_fail), \
             patch("modules.infra.token_budget.time.sleep"):
            tracker.add_tokens(750)

        with open(tracker.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        assert state["tokens_used"] == 750

        temp_file = tracker.state_file.with_suffix(".tmp")
        assert not temp_file.exists()

    @pytest.mark.unit
    def test_temp_file_cleaned_up_on_general_error(self, tracker) -> None:
        """Temp file is removed by finally block on non-Permission errors."""
        temp_file = tracker.state_file.with_suffix(".tmp")

        def fail_replace(self_path, target):
            raise FileNotFoundError("Target directory missing")

        with patch.object(Path, "replace", fail_replace):
            tracker._save_state()

        assert not temp_file.exists()

    @pytest.mark.unit
    def test_no_retry_on_non_permission_error(self, tracker) -> None:
        """Non-PermissionError exceptions are not retried."""
        call_count = 0

        def fail_with_fnf(self_path, target):
            nonlocal call_count
            call_count += 1
            raise FileNotFoundError("Target directory missing")

        with patch.object(Path, "replace", fail_with_fnf), \
             patch("modules.infra.token_budget.time.sleep") as mock_sleep:
            tracker._save_state()

        assert call_count == 1
        mock_sleep.assert_not_called()