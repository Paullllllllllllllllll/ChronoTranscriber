"""
Unified execution framework for dual-mode (Interactive/CLI) scripts.

This module provides base classes and utilities to eliminate code duplication
across all main entry point scripts that support both interactive UI mode
and command-line argument mode.

Classes:
    DualModeScript: Base class for synchronous scripts
    AsyncDualModeScript: Base class for async scripts (uses asyncio.run)
"""

import asyncio
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, Optional

from modules.config.service import ConfigService, get_config_service
from modules.infra.logger import setup_logger
from modules.ui import print_error, print_info


class _DualModeBase:
    """Shared infrastructure for both sync and async dual-mode scripts.
    
    Provides configuration loading, error handling, and logging helpers
    so that DualModeScript and AsyncDualModeScript avoid duplicating them.
    """
    
    def __init__(self, script_name: str):
        self.script_name = script_name
        self.logger = setup_logger(script_name)
        self.config_service: Optional[ConfigService] = None
        self.is_interactive: bool = False
        
        # Configuration dictionaries (loaded on demand)
        self.paths_config: Dict[str, Any] = {}
        self.model_config: Dict[str, Any] = {}
        self.concurrency_config: Dict[str, Any] = {}
        self.image_processing_config: Dict[str, Any] = {}
    
    def initialize_config(self) -> None:
        """Load all configuration resources."""
        self.config_service = get_config_service()
        self.paths_config = self.config_service.get_paths_config()
        self.model_config = self.config_service.get_model_config()
        self.concurrency_config = self.config_service.get_concurrency_config()
        self.image_processing_config = self.config_service.get_image_processing_config()
    
    def _detect_mode(self) -> bool:
        """Detect execution mode from configuration.
        
        Returns:
            True if interactive mode, False for CLI mode.
        """
        return self.paths_config.get("general", {}).get("interactive_mode", True)
    
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt gracefully."""
        print_info("\nOperation cancelled by user.")
        self.logger.info(f"{self.script_name} cancelled by user")
        sys.exit(0)
    
    def _handle_error(self, error: Exception) -> None:
        """Handle unexpected errors gracefully.
        
        Args:
            error: The exception that was raised
        """
        error_msg = f"Unexpected error: {error}"
        print_error(error_msg)
        self.logger.error(f"{self.script_name} failed", exc_info=error)
        sys.exit(1)
    
    def print_or_log(self, message: str, level: str = "info") -> None:
        """Print message using UI utilities and log.
        
        Args:
            message: Message to display/log
            level: Log level (info, warning, error, success)
        """
        from modules.ui import print_info, print_warning, print_error, print_success
        
        if level == "error":
            print_error(message)
        elif level == "warning":
            print_warning(message)
        elif level == "success":
            print_success(message)
        else:
            print_info(message)
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)


class DualModeScript(_DualModeBase, ABC):
    """
    Base class for synchronous scripts that support both interactive and CLI modes.
    
    Subclasses must implement:
    - create_argument_parser(): Return configured ArgumentParser
    - run_interactive(): Execute interactive workflow
    - run_cli(): Execute CLI workflow
    """
    
    @abstractmethod
    def create_argument_parser(self) -> ArgumentParser:
        """Create and configure the argument parser for CLI mode.
        
        Returns:
            Configured ArgumentParser instance
        """
        pass
    
    @abstractmethod
    def run_interactive(self) -> None:
        """Execute the interactive workflow with UI prompts."""
        pass
    
    @abstractmethod
    def run_cli(self, args: Namespace) -> None:
        """Execute the CLI workflow with parsed arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        pass
    
    def execute(self) -> None:
        """Main entry point that orchestrates mode detection and execution."""
        try:
            self.initialize_config()
            self.is_interactive = self._detect_mode()
            
            if self.is_interactive:
                self.logger.info(f"Starting {self.script_name} (Interactive Mode)")
                self.run_interactive()
            else:
                self.logger.info(f"Starting {self.script_name} (CLI Mode)")
                parser = self.create_argument_parser()
                args = parser.parse_args()
                self.run_cli(args)
                
        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            self._handle_error(e)


class AsyncDualModeScript(_DualModeBase, ABC):
    """
    Base class for async scripts that support both interactive and CLI modes.
    
    Subclasses must implement:
    - create_argument_parser(): Return configured ArgumentParser
    - run_interactive(): Execute async interactive workflow
    - run_cli(): Execute async CLI workflow
    """
    
    @abstractmethod
    def create_argument_parser(self) -> ArgumentParser:
        """Create and configure the argument parser for CLI mode.
        
        Returns:
            Configured ArgumentParser instance
        """
        pass
    
    @abstractmethod
    async def run_interactive(self) -> None:
        """Execute the async interactive workflow with UI prompts."""
        pass
    
    @abstractmethod
    async def run_cli(self, args: Namespace) -> None:
        """Execute the async CLI workflow with parsed arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        pass
    
    def execute(self) -> None:
        """Main entry point that wraps the async execution in asyncio.run()."""
        asyncio.run(self._execute_async())
    
    async def _execute_async(self) -> None:
        """Internal async execution handler."""
        try:
            self.initialize_config()
            self.is_interactive = self._detect_mode()
            
            if self.is_interactive:
                self.logger.info(f"Starting {self.script_name} (Interactive Mode)")
                await self.run_interactive()
            else:
                self.logger.info(f"Starting {self.script_name} (CLI Mode)")
                parser = self.create_argument_parser()
                args = parser.parse_args()
                await self.run_cli(args)
                
        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            self._handle_error(e)
