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


class DualModeScript(ABC):
    """
    Base class for synchronous scripts that support both interactive and CLI modes.
    
    This class handles:
    - Mode detection (interactive vs CLI)
    - Configuration loading
    - Logger setup
    - Common error handling
    
    Subclasses must implement:
    - create_argument_parser(): Return configured ArgumentParser
    - run_interactive(): Execute interactive workflow
    - run_cli(): Execute CLI workflow
    """
    
    def __init__(self, script_name: str):
        """
        Initialize the dual-mode script.
        
        Args:
            script_name: Name of the script for logging purposes
        """
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
    
    @abstractmethod
    def create_argument_parser(self) -> ArgumentParser:
        """
        Create and configure the argument parser for CLI mode.
        
        Returns:
            Configured ArgumentParser instance
        """
        pass
    
    @abstractmethod
    def run_interactive(self) -> None:
        """
        Execute the interactive workflow with UI prompts.
        
        This method is called when the script runs in interactive mode.
        """
        pass
    
    @abstractmethod
    def run_cli(self, args: Namespace) -> None:
        """
        Execute the CLI workflow with parsed arguments.
        
        This method is called when the script runs in CLI mode.
        
        Args:
            args: Parsed command-line arguments
        """
        pass
    
    def execute(self) -> None:
        """
        Main entry point that orchestrates mode detection and execution.
        
        This method:
        1. Loads configuration
        2. Detects execution mode (interactive vs CLI)
        3. Calls the appropriate run method
        4. Handles common error scenarios
        """
        try:
            # Load configuration
            self.initialize_config()
            
            # Determine execution mode
            self.is_interactive = self.paths_config.get("general", {}).get("interactive_mode", True)
            
            if self.is_interactive:
                # Interactive mode
                self.logger.info(f"Starting {self.script_name} (Interactive Mode)")
                self.run_interactive()
            else:
                # CLI mode
                self.logger.info(f"Starting {self.script_name} (CLI Mode)")
                parser = self.create_argument_parser()
                args = parser.parse_args()
                self.run_cli(args)
                
        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            self._handle_error(e)
    
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt gracefully."""
        print_info("\nOperation cancelled by user.")
        self.logger.info(f"{self.script_name} cancelled by user")
        sys.exit(0)
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle unexpected errors gracefully.
        
        Args:
            error: The exception that was raised
        """
        error_msg = f"Unexpected error: {error}"
        print_error(error_msg)
        self.logger.error(f"{self.script_name} failed", exc_info=error)
        sys.exit(1)
    
    def print_or_log(self, message: str, level: str = "info") -> None:
        """
        Print message using UI utilities or log.
        
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
        
        # Always log
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)


class AsyncDualModeScript(ABC):
    """
    Base class for async scripts that support both interactive and CLI modes.
    
    This class handles:
    - Mode detection (interactive vs CLI)
    - Configuration loading
    - Logger setup
    - Common error handling
    - Async execution via asyncio.run()
    
    Subclasses must implement:
    - create_argument_parser(): Return configured ArgumentParser
    - run_interactive(): Execute async interactive workflow
    - run_cli(): Execute async CLI workflow
    """
    
    def __init__(self, script_name: str):
        """
        Initialize the async dual-mode script.
        
        Args:
            script_name: Name of the script for logging purposes
        """
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
    
    @abstractmethod
    def create_argument_parser(self) -> ArgumentParser:
        """
        Create and configure the argument parser for CLI mode.
        
        Returns:
            Configured ArgumentParser instance
        """
        pass
    
    @abstractmethod
    async def run_interactive(self) -> None:
        """
        Execute the async interactive workflow with UI prompts.
        
        This method is called when the script runs in interactive mode.
        """
        pass
    
    @abstractmethod
    async def run_cli(self, args: Namespace) -> None:
        """
        Execute the async CLI workflow with parsed arguments.
        
        This method is called when the script runs in CLI mode.
        
        Args:
            args: Parsed command-line arguments
        """
        pass
    
    def execute(self) -> None:
        """
        Main entry point that orchestrates mode detection and async execution.
        
        This method wraps the async execution in asyncio.run().
        """
        asyncio.run(self._execute_async())
    
    async def _execute_async(self) -> None:
        """
        Internal async execution handler.
        
        This method:
        1. Loads configuration
        2. Detects execution mode (interactive vs CLI)
        3. Calls the appropriate async run method
        4. Handles common error scenarios
        """
        try:
            # Load configuration
            self.initialize_config()
            
            # Determine execution mode
            self.is_interactive = self.paths_config.get("general", {}).get("interactive_mode", True)
            
            if self.is_interactive:
                # Interactive mode
                self.logger.info(f"Starting {self.script_name} (Interactive Mode)")
                await self.run_interactive()
            else:
                # CLI mode
                self.logger.info(f"Starting {self.script_name} (CLI Mode)")
                parser = self.create_argument_parser()
                args = parser.parse_args()
                await self.run_cli(args)
                
        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            self._handle_error(e)
    
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt gracefully."""
        print_info("\nOperation cancelled by user.")
        self.logger.info(f"{self.script_name} cancelled by user")
        sys.exit(0)
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle unexpected errors gracefully.
        
        Args:
            error: The exception that was raised
        """
        error_msg = f"Unexpected error: {error}"
        print_error(error_msg)
        self.logger.error(f"{self.script_name} failed", exc_info=error)
        sys.exit(1)
    
    def print_or_log(self, message: str, level: str = "info") -> None:
        """
        Print message using UI utilities or log.
        
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
        
        # Always log
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
