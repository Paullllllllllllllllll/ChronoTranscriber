# repair_transcriptions.py
"""
CLI script for repairing transcription batches.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace

from modules.batch.repair import main as repair_main_interactive
from modules.batch.repair import main_cli
from modules.core.cli_args import create_repair_parser
from modules.transcribe.dual_mode import AsyncDualModeScript


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
        """Run repair in CLI mode.

        Emits a one-line JSON summary on stdout when ``--json`` is passed
        (CT-4). Exit code stays 0 on a completed run (existing contract);
        unexpected exceptions still exit 1 via the framework handler.
        """
        summary = await main_cli(args, self.paths_config)
        if getattr(args, "json_summary", False):
            payload = {
                "tool": "chronotranscriber",
                "command": "repair_transcriptions",
                "repaired": int(summary.get("repaired", 0)),
                "failed": int(summary.get("failed", 0)),
                "exit_code": 0,
            }
            print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    """Main entry point."""
    RepairTranscriptionsScript().execute()


if __name__ == "__main__":
    main()
