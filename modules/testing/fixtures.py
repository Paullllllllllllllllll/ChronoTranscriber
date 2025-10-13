"""Test fixtures and utilities.

Provides reusable test fixtures for configuration, file system, and API mocking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock


def create_test_config(
    interactive_mode: bool = False,
    model_name: str = "gpt-4o",
    **overrides: Any,
) -> Dict[str, Any]:
    """Create a test configuration dictionary.
    
    Args:
        interactive_mode: Whether to enable interactive mode.
        model_name: Model name to use.
        **overrides: Additional config overrides.
        
    Returns:
        Test configuration dictionary.
    """
    config = {
        "general": {
            "interactive_mode": interactive_mode,
            "logs_dir": "logs",
        },
        "file_paths": {
            "PDFs": {"input": "pdfs_in", "output": "pdfs_out"},
            "Images": {"input": "images_in", "output": "images_out"},
            "EPUBs": {"input": "epubs_in", "output": "epubs_out"},
            "Auto": {"input": "auto_in", "output": "auto_out"},
        },
        "transcription_model": {
            "name": model_name,
            "max_output_tokens": 4096,
        },
        "concurrency": {
            "transcription": {
                "concurrency_limit": 10,
                "service_tier": "auto",
            }
        },
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "general.interactive_mode"
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        else:
            config[key] = value
    
    return config


def create_temp_test_dir() -> Path:
    """Create a temporary directory for testing.
    
    Returns:
        Path to temporary directory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="chronotranscriber_test_"))
    return temp_dir


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        """Initialize mock client.
        
        Args:
            responses: Dictionary mapping method names to response values.
        """
        self.responses = responses or {}
        self.calls = []
    
    def __getattr__(self, name: str) -> MagicMock:
        """Create mock methods on demand."""
        mock = MagicMock()
        
        if name in self.responses:
            mock.return_value = self.responses[name]
        
        # Track calls
        def track_call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.responses.get(name, MagicMock())
        
        mock.side_effect = track_call
        return mock
    
    def get_call_count(self, method_name: str) -> int:
        """Get number of times a method was called.
        
        Args:
            method_name: Name of method to check.
            
        Returns:
            Number of calls.
        """
        return sum(1 for name, _, _ in self.calls if name == method_name)
