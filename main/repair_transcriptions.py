# repair_transcriptions.py
"""
CLI script for repairing transcription batches.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from modules.core.cli_args import create_repair_parser
from modules.core.execution_framework import AsyncDualModeScript
from modules.operations.repair.run import main as repair_main_interactive, main_cli


class RepairTranscriptionsScript(AsyncDualModeScript):
    """Script for repairing transcription batches."""
    
    def __init__(self) -> None:
        super().__init__("repair_transcriptions")
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_repair_parser()
    
    async def run_interactive(self) -> None:
        """Run repair in interactive mode."""
        await repair_main_interactive()
    
    async def run_cli(self, args: Namespace) -> None:
        """Run repair in CLI mode."""
        await main_cli(args, self.paths_config)


def main() -> None:
    """Main entry point."""
    RepairTranscriptionsScript().execute()


if __name__ == "__main__":
    main()
