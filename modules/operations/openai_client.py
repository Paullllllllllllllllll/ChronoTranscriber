"""OpenAI client setup utilities for operations.

Provides consistent OpenAI client initialization and error handling
for batch and repair operations.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Get configured OpenAI client.
    
    Args:
        api_key: Optional API key. If None, uses OPENAI_API_KEY environment variable.
        
    Returns:
        Configured OpenAI client.
        
    Raises:
        ValueError: If no API key is available.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    return OpenAI(api_key=api_key)


def validate_api_key() -> bool:
    """Check if OpenAI API key is available.
    
    Returns:
        True if API key is set, False otherwise.
    """
    return bool(os.environ.get("OPENAI_API_KEY"))
