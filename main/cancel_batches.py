"""
CLI script to cancel all non-terminal batch jobs.

Supports two modes:
1. Interactive mode: Shows summary and cancels batches (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

from modules.batch.cancel import cancel_batch_by_id
from modules.config.constants import TERMINAL_BATCH_STATUSES
from modules.core.cli_args import create_cancel_batches_parser
from modules.llm.openai_sdk_utils import list_all_batches, sdk_to_dict
from modules.transcribe.dual_mode import DualModeScript
from modules.ui import (
    PromptStyle,
    confirm_action,
    print_error,
    print_header,
    print_info,
    print_success,
    ui_print,
)
from modules.ui.batch_display import (
    display_batch_cancellation_results,
    display_batch_summary,
)


def _extract_batch_id_and_status(b: object) -> tuple[str | None, str]:
    """Return the (batch_id, lowercased status) for an SDK batch object.

    Normalizes the batch shape via ``sdk_to_dict`` and falls back to direct
    attribute access so both dict-like and object-like SDK results work.
    """
    bd = sdk_to_dict(b)
    batch_id = bd.get("id") or getattr(b, "id", None)
    status = (str(bd.get("status") or getattr(b, "status", "") or "")).lower()
    return batch_id, status


class CancelBatchesScript(DualModeScript):
    """Script to cancel all non-terminal batch jobs."""

    def __init__(self) -> None:
        super().__init__("cancel_batches")

    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_cancel_batches_parser()

    def run_interactive(self) -> None:
        """Cancel batches in interactive mode with prompts."""
        try:
            from openai import OpenAI
        except Exception as e:
            print_error(
                "Could not import OpenAI SDK. This is often caused by a"
                " pydantic/pydantic-core version mismatch."
            )
            print_info("Try upgrading your environment inside the venv, e.g.:")
            ui_print(
                "  .venv\\Scripts\\python.exe -m pip install --upgrade"
                " --upgrade-strategy eager pydantic-core pydantic openai",
                PromptStyle.DIM,
            )
            print_error(f"Original error: {str(e)}")
            sys.exit(1)

        print_header("BATCH CANCELLATION", "Cancel all non-terminal batch jobs")

        client = OpenAI()
        print_info("Retrieving list of batches from OpenAI (with pagination)...")

        try:
            batches = list_all_batches(client)
        except Exception as e:
            self.logger.error(f"Error listing batches: {e}")
            print_error(f"Error listing batches: {e}")
            return

        # Display batch summary
        display_batch_summary(batches)

        if not batches:
            return

        # Track batches for cancellation and those that are skipped
        cancelled_batches = []  # (batch_id, status, success)
        skipped_batches = []  # (batch_id, status)

        # Partition batches: terminal ones are skipped, the rest are eligible for
        # cancellation. This mirrors the CLI path, which only targets
        # non-terminal batches.
        non_terminal: list[tuple[str, str]] = []
        for b in batches:
            batch_id, status = _extract_batch_id_and_status(b)
            if not batch_id:
                continue
            if status in TERMINAL_BATCH_STATUSES:
                self.logger.info(
                    f"Skipping batch {batch_id} with terminal status '{status}'."
                )
                skipped_batches.append((batch_id, status))
            else:
                non_terminal.append((batch_id, status))

        if not non_terminal:
            print_info("No non-terminal batches to cancel.")
            return

        # Interactive mode must confirm before cancelling, matching the CLI path
        # (which requires confirm_action unless --force).
        if not confirm_action(
            f"Cancel all {len(non_terminal)} non-terminal batch(es)?",
            default=False,
        ):
            print_info("Cancellation aborted.")
            return

        print_info("Processing cancellations...")
        for batch_id, status in non_terminal:
            if cancel_batch_by_id("openai", batch_id):
                self.logger.info(f"Batch {batch_id} cancelled.")
                cancelled_batches.append((batch_id, status, True))
            else:
                self.logger.error(f"Error cancelling batch {batch_id}.")
                cancelled_batches.append((batch_id, status, False))

        # Display cancellation results
        display_batch_cancellation_results(cancelled_batches, skipped_batches)

    @staticmethod
    def _emit_json_summary(
        args: Namespace, cancelled: int, failed: int, exit_code: int
    ) -> None:
        """Print the one-line ``--json`` summary on stdout when requested (CT-4)."""
        if not getattr(args, "json_summary", False):
            return
        import json

        payload = {
            "tool": "chronotranscriber",
            "command": "cancel_batches",
            "cancelled": cancelled,
            "failed": failed,
            "exit_code": exit_code,
        }
        print(json.dumps(payload, ensure_ascii=False))

    def run_cli(self, args: Namespace) -> None:
        """Cancel batches in CLI mode with arguments."""
        try:
            from openai import OpenAI
        except Exception as e:
            print_error(f"Could not import OpenAI SDK: {e}")
            self._emit_json_summary(args, 0, 0, 1)
            sys.exit(1)

        print_header("BATCH CANCELLATION (CLI MODE)", "")

        client = OpenAI()

        # Determine which batches to cancel
        if args.batch_ids:
            # Cancel specific batch IDs
            print_info(f"Targeting {len(args.batch_ids)} specific batch(es)...")
            batch_ids_to_cancel = args.batch_ids
        else:
            # Cancel all non-terminal batches
            print_info("Retrieving list of batches from OpenAI...")
            try:
                batches = list_all_batches(client)
            except Exception as e:
                self.logger.error(f"Error listing batches: {e}")
                print_error(f"Error listing batches: {e}")
                self._emit_json_summary(args, 0, 0, 1)
                sys.exit(1)

            # Filter to non-terminal batches
            batch_ids_to_cancel = []
            for b in batches:
                batch_id, status = _extract_batch_id_and_status(b)
                if batch_id and status not in TERMINAL_BATCH_STATUSES:
                    batch_ids_to_cancel.append(batch_id)

            print_info(
                f"Found {len(batch_ids_to_cancel)} non-terminal batch(es) to cancel."
            )

        if not batch_ids_to_cancel:
            print_info("No batches to cancel.")
            self._emit_json_summary(args, 0, 0, 0)
            return

        # Confirm if not forced
        if not args.force:
            ui_print(
                f"\n  About to cancel {len(batch_ids_to_cancel)} batch(es).",
                PromptStyle.WARNING,
            )
            if not confirm_action("Proceed with cancellation?", default=False):
                print_info("Cancellation aborted.")
                self._emit_json_summary(args, 0, 0, 0)
                return

        # Cancel batches
        success_count = 0
        fail_count = 0

        for batch_id in batch_ids_to_cancel:
            if cancel_batch_by_id("openai", batch_id):
                self.logger.info(f"Batch {batch_id} cancelled.")
                print_success(f"Cancelled: {batch_id}")
                success_count += 1
            else:
                self.logger.error(f"Error cancelling batch {batch_id}.")
                print_error(f"Failed to cancel {batch_id}")
                fail_count += 1

        # Summary
        print_header("CANCELLATION COMPLETE", "")
        print_success(f"Successfully cancelled: {success_count}")
        exit_code = 1 if fail_count > 0 else 0
        self._emit_json_summary(args, success_count, fail_count, exit_code)
        if fail_count > 0:
            print_error(f"Failed to cancel: {fail_count}")
            # CLI agent contract: non-zero exit when any cancellation failed.
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    CancelBatchesScript().execute()


if __name__ == "__main__":
    main()
