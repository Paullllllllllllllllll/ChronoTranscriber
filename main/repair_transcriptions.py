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
from modules.config.config_loader import ConfigLoader
from modules.ui import print_info, print_error
from modules.operations.repair.run import main as repair_main_interactive, main_cli
from modules.core.cli_args import create_repair_parser

logger = setup_logger(__name__)


async def main():
    """Main entry point supporting both interactive and CLI modes."""
    try:
        # Load configuration to check mode
        config_loader = ConfigLoader()
        config_loader.load_configs()
        paths_config = config_loader.get_paths_config()
        interactive_mode = paths_config.get("general", {}).get("interactive_mode", True)
        
        if interactive_mode:
            # Interactive mode with prompts
            await repair_main_interactive()
        else:
            # CLI mode with arguments
            parser = create_repair_parser()
            args = parser.parse_args()
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
