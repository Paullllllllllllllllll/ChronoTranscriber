"""Centralized mode selector for dual CLI/interactive script execution.

This module provides a unified interface for all main scripts to route execution
based on the interactive_mode configuration flag, eliminating code duplication.
"""

from __future__ import annotations

import sys
from typing import Any, Awaitable, Callable, Dict

from modules.config.service import get_config_service, ConfigService
from modules.infra.logger import setup_logger
from modules.ui import print_error

logger = setup_logger(__name__)


def _detect_mode_and_parse_args(
    parser_factory: Callable[[], Any],
    script_name: str,
) -> tuple[ConfigService, bool, Any, Dict[str, Any]]:
    """Internal helper that performs mode detection and argument parsing.
    
    This function handles:
    - Configuration loading
    - Mode detection (interactive vs CLI)
    - Argument parsing for CLI mode
    - Error handling for config loading failures
    
    Args:
        parser_factory: Function that returns an ArgumentParser
        script_name: Name of the calling script (for error messages)
        
    Returns:
        Tuple of (config_service, interactive_mode, args_or_none, paths_config)
        
    Raises:
        SystemExit: On configuration loading failure
    """
    try:
        config_service = get_config_service()
        paths_config = config_service.get_paths_config()
    except Exception as e:
        logger.critical(f"{script_name}: Failed to load configurations: {e}")
        print_error(f"Failed to load configurations: {e}")
        sys.exit(1)
    
    interactive_mode = paths_config.get("general", {}).get("interactive_mode", True)
    
    args = None
    if not interactive_mode:
        parser = parser_factory()
        args = parser.parse_args()
    
    return config_service, interactive_mode, args, paths_config


def run_with_mode_detection(
    interactive_handler: Callable[[], Awaitable[None]],
    cli_handler: Callable[[Any, Dict[str, Any]], Awaitable[None]],
    parser_factory: Callable[[], Any],
    script_name: str,
) -> tuple[ConfigService, bool, Any, Dict[str, Any]]:
    """Route execution based on interactive_mode configuration (async version).
    
    Args:
        interactive_handler: Async function to call in interactive mode (for type hints)
        cli_handler: Async function to call in CLI mode (for type hints)
        parser_factory: Function that returns an ArgumentParser
        script_name: Name of the calling script (for error messages)
        
    Returns:
        Tuple of (config_service, interactive_mode, args_or_none, paths_config)
        
    Raises:
        SystemExit: On configuration loading failure
    """
    return _detect_mode_and_parse_args(parser_factory, script_name)


def run_sync_with_mode_detection(
    interactive_handler: Callable[[], None],
    cli_handler: Callable[[Any, Dict[str, Any]], None],
    parser_factory: Callable[[], Any],
    script_name: str,
) -> tuple[ConfigService, bool, Any, Dict[str, Any]]:
    """Route execution based on interactive_mode configuration (synchronous version).
    
    Args:
        interactive_handler: Synchronous function to call in interactive mode (for type hints)
        cli_handler: Synchronous function to call in CLI mode (for type hints)
        parser_factory: Function that returns an ArgumentParser
        script_name: Name of the calling script (for error messages)
        
    Returns:
        Tuple of (config_service, interactive_mode, args_or_none, paths_config)
        
    Raises:
        SystemExit: On configuration loading failure
    """
    return _detect_mode_and_parse_args(parser_factory, script_name)
