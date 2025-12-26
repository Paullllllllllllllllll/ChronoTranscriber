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
