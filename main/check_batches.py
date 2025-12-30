# check_batches.py
"""
CLI script to check whether batch jobs have finished successfully and
download and process completed batches.

Supports two modes:
1. Interactive mode: Runs with diagnostics and default behavior (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from modules.core.cli_args import create_check_batches_parser, resolve_path, validate_input_path
from modules.core.execution_framework import DualModeScript
from modules.operations.batch.check import run_batch_finalization


class CheckBatchesScript(DualModeScript):
    """Script to check batch job status and download completed results."""
    
    def __init__(self):
        super().__init__("check_batches")
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_check_batches_parser()
    
    def run_interactive(self) -> None:
        """Check batches in interactive mode with diagnostics enabled."""
        run_batch_finalization(run_diagnostics=True)
    
    def run_cli(self, args: Namespace) -> None:
        """Check batches in CLI mode with command-line arguments."""
        run_diagnostics = not args.no_diagnostics
        custom_directory = None
        
        # If directory specified, validate and use it
        if args.directory:
            custom_directory = resolve_path(args.directory, Path.cwd())
            validate_input_path(custom_directory)
        
        run_batch_finalization(run_diagnostics=run_diagnostics, custom_directory=custom_directory)


def main() -> None:
    """Main entry point."""
    CheckBatchesScript().execute()


if __name__ == "__main__":
    main()