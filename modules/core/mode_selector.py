"""Centralized mode selector for dual CLI/interactive script execution.

This module provides a unified interface for all main scripts to route execution
based on the interactive_mode configuration flag, eliminating code duplication.
"""

from __future__ import annotations

import sys
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from modules.config.service import get_config_service, ConfigService
from modules.infra.logger import setup_logger
from modules.ui import print_error, print_info

logger = setup_logger(__name__)

T = TypeVar('T')


def run_with_mode_detection(
    interactive_handler: Callable[[], Awaitable[None]],
    cli_handler: Callable[[Any, Dict[str, Any]], Awaitable[None]],
    parser_factory: Callable[[], Any],
    script_name: str,
) -> tuple[ConfigService, bool, Any, Dict[str, Any]]:
    """Route execution based on interactive_mode configuration.
    
    This function handles:
    - Configuration loading
    - Mode detection (interactive vs CLI)
    - Argument parsing for CLI mode
    - Error handling for config loading failures
    
    Args:
        interactive_handler: Async function to call in interactive mode
        cli_handler: Async function to call in CLI mode (receives args and config)
        parser_factory: Function that returns an ArgumentParser
        script_name: Name of the calling script (for error messages)
        
    Returns:
        Tuple of (config_service, interactive_mode, args_or_none, paths_config)
        
    Raises:
        SystemExit: On configuration loading failure
    """
    # Load configuration
    try:
        config_service = get_config_service()
        paths_config = config_service.get_paths_config()
    except Exception as e:
        logger.critical(f"{script_name}: Failed to load configurations: {e}")
        print_error(f"Failed to load configurations: {e}")
        sys.exit(1)
    
    # Detect mode
    interactive_mode = paths_config.get("general", {}).get("interactive_mode", True)
    
    # Parse arguments if in CLI mode
    args = None
    if not interactive_mode:
        parser = parser_factory()
        args = parser.parse_args()
    
    return config_service, interactive_mode, args, paths_config


def run_sync_with_mode_detection(
    interactive_handler: Callable[[], None],
    cli_handler: Callable[[Any, Dict[str, Any]], None],
    parser_factory: Callable[[], Any],
    script_name: str,
) -> tuple[ConfigService, bool, Any, Dict[str, Any]]:
    """Route execution based on interactive_mode configuration (synchronous version).
    
    Same as run_with_mode_detection but for synchronous handlers.
    
    Args:
        interactive_handler: Synchronous function to call in interactive mode
        cli_handler: Synchronous function to call in CLI mode (receives args and config)
        parser_factory: Function that returns an ArgumentParser
        script_name: Name of the calling script (for error messages)
        
    Returns:
        Tuple of (config_service, interactive_mode, args_or_none, paths_config)
        
    Raises:
        SystemExit: On configuration loading failure
    """
    # Load configuration
    try:
        config_service = get_config_service()
        paths_config = config_service.get_paths_config()
    except Exception as e:
        logger.critical(f"{script_name}: Failed to load configurations: {e}")
        print_error(f"Failed to load configurations: {e}")
        sys.exit(1)
    
    # Detect mode
    interactive_mode = paths_config.get("general", {}).get("interactive_mode", True)
    
    # Parse arguments if in CLI mode
    args = None
    if not interactive_mode:
        parser = parser_factory()
        args = parser.parse_args()
    
    return config_service, interactive_mode, args, paths_config
