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

from modules.config.config_loader import ConfigLoader
from modules.operations.batch.check import run_batch_finalization
from modules.core.cli_args import create_check_batches_parser, resolve_path, validate_input_path


def main() -> None:
    """
    Entrypoint for checking all directories for batch outputs and finalizing them.
    Supports both interactive and CLI modes.
    """
    # Load configuration to check mode
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config = config_loader.get_paths_config()
    interactive_mode = paths_config.get("general", {}).get("interactive_mode", True)
    
    if interactive_mode:
        # Interactive mode - use default behavior
        run_batch_finalization(run_diagnostics=True)
    else:
        # CLI mode - parse arguments
        parser = create_check_batches_parser()
        args = parser.parse_args()
        
        # If directory specified, override paths_config
        if args.directory:
            scan_dir = resolve_path(args.directory, Path.cwd())
            validate_input_path(scan_dir)
            # TODO: Would need to modify run_batch_finalization to accept custom directory
            # For now, still use default behavior with optional diagnostics flag
        
        run_diagnostics = not args.no_diagnostics
        run_batch_finalization(run_diagnostics=run_diagnostics)


if __name__ == "__main__":
    main()