# repair_transcriptions.py
"""
CLI script for repairing transcription batches.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import asyncio
import sys

from modules.infra.logger import setup_logger
from modules.ui import print_info, print_error
from modules.operations.repair.run import main as repair_main_interactive, main_cli
from modules.core.cli_args import create_repair_parser
from modules.core.mode_selector import run_with_mode_detection

logger = setup_logger(__name__)


async def main() -> None:
    """Main entry point supporting both interactive and CLI modes."""
    try:
        # Use centralized mode detection
        config_loader, interactive_mode, args, paths_config = run_with_mode_detection(
            interactive_handler=repair_main_interactive,
            cli_handler=main_cli,
            parser_factory=create_repair_parser,
            script_name="repair_transcriptions"
        )
        
        # Route to appropriate handler
        if interactive_mode:
            await repair_main_interactive()
        else:
            await main_cli(args, paths_config)
            
    except KeyboardInterrupt:
        print_info("\nRepair interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error in repair_transcriptions: %s", e)
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
