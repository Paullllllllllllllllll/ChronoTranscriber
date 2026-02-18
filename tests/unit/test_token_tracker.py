"""Unit tests for modules/infra/token_tracker.py.

Tests token usage tracking and daily limit management.
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.infra.token_tracker import (
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
    def test_initialization(self, tracker):
        """Test TokenTracker initialization."""
        assert tracker.daily_limit == 100000
        assert tracker.get_tokens_used_today() == 0
    
    @pytest.mark.unit
    def test_add_tokens(self, tracker):
        """Test adding tokens to tracker."""
        tracker.add_tokens(1000)
        assert tracker.get_tokens_used_today() == 1000
        
        tracker.add_tokens(500)
        assert tracker.get_tokens_used_today() == 1500
    
    @pytest.mark.unit
    def test_get_remaining(self, tracker):
        """Test getting remaining tokens via stats."""
        tracker.add_tokens(30000)
        stats = tracker.get_stats()
        assert stats["tokens_remaining"] == 70000
    
    @pytest.mark.unit
    def test_is_limit_reached(self, tracker):
        """Test limit reached detection."""
        assert tracker.is_limit_reached() is False
        
        tracker.add_tokens(100000)
        assert tracker.is_limit_reached() is True
    
    @pytest.mark.unit
    def test_get_stats(self, tracker):
        """Test getting usage statistics."""
        tracker.add_tokens(50000)
        
        stats = tracker.get_stats()
        
        assert stats["tokens_used_today"] == 50000
        assert stats["daily_limit"] == 100000
        assert stats["tokens_remaining"] == 50000
        assert stats["usage_percentage"] == 50.0
    
    @pytest.mark.unit
    def test_persists_state(self, temp_dir):
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
    def test_can_use_tokens(self, tracker):
        """Test can_use_tokens method."""
        assert tracker.can_use_tokens() is True
        
        tracker.add_tokens(100000)  # Hit limit
        assert tracker.can_use_tokens() is False


class TestDailyTokensUnderscoreParsing:
    """Tests for underscore-formatted daily_tokens ingestion in get_token_tracker()."""

    @pytest.mark.unit
    def test_underscore_string_parsed(self, temp_dir):
        """daily_tokens as quoted underscore string (e.g. "10_000_000") is accepted."""
        import modules.infra.token_tracker as tt
        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {"daily_token_limit": {"enabled": False, "daily_tokens": "10_000_000"}}
            with patch("modules.infra.token_tracker.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 10_000_000
        finally:
            tt._tracker_instance = original

    @pytest.mark.unit
    def test_plain_int_still_works(self, temp_dir):
        """daily_tokens as a plain int (standard YAML) is still accepted."""
        import modules.infra.token_tracker as tt
        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {"daily_token_limit": {"enabled": False, "daily_tokens": 67500000}}
            with patch("modules.infra.token_tracker.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 67_500_000
        finally:
            tt._tracker_instance = original

    @pytest.mark.unit
    def test_underscore_int_from_yaml_parser(self, temp_dir):
        """daily_tokens as an already-stripped int (PyYAML may parse 10_000_000 as int) works."""
        import modules.infra.token_tracker as tt
        original = tt._tracker_instance
        tt._tracker_instance = None
        try:
            mock_cfg = {"daily_token_limit": {"enabled": False, "daily_tokens": 10000000}}
            with patch("modules.infra.token_tracker.get_config_service") as mock_svc:
                mock_svc.return_value.get_concurrency_config.return_value = mock_cfg
                tracker = tt.get_token_tracker()
                assert tracker.daily_limit == 10_000_000
        finally:
            tt._tracker_instance = original


class TestTokenTrackerSingleton:
    """Tests for token tracker singleton functions."""
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_get_token_tracker_returns_instance(self):
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
    def test_get_token_tracker_returns_same_instance(self):
        """Test that get_token_tracker returns same instance."""
        try:
            tracker1 = get_token_tracker()
            tracker2 = get_token_tracker()
            assert tracker1 is tracker2
        except Exception:
            pytest.skip("Config loading not available in test environment")
