# check_batches.py
"""
CLI script to check whether batch jobs have finished successfully and
download and process completed batches.

Supports two modes:
1. Interactive mode: Runs with diagnostics and default behavior (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

from pathlib import Path

from modules.operations.batch.check import run_batch_finalization
from modules.core.cli_args import create_check_batches_parser, resolve_path, validate_input_path
from modules.core.mode_selector import run_sync_with_mode_detection


def check_batches_interactive() -> None:
    """Check batches in interactive mode with diagnostics enabled."""
    run_batch_finalization(run_diagnostics=True)


def check_batches_cli(args, paths_config: dict) -> None:
    """Check batches in CLI mode with command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        paths_config: Paths configuration dictionary
    """
    run_diagnostics = not args.no_diagnostics
    custom_directory = None
    
    # If directory specified, validate and use it
    if args.directory:
        custom_directory = resolve_path(args.directory, Path.cwd())
        validate_input_path(custom_directory)
    
    run_batch_finalization(run_diagnostics=run_diagnostics, custom_directory=custom_directory)


def main() -> None:
    """Main entry point supporting both interactive and CLI modes."""
    # Use centralized mode detection
    config_loader, interactive_mode, args, paths_config = run_sync_with_mode_detection(
        interactive_handler=check_batches_interactive,
        cli_handler=check_batches_cli,
        parser_factory=create_check_batches_parser,
        script_name="check_batches"
    )
    
    # Route to appropriate handler
    if interactive_mode:
        check_batches_interactive()
    else:
        check_batches_cli(args, paths_config)


if __name__ == "__main__":
    main()