"""
CLI script to cancel all non-terminal batch jobs.

Supports two modes:
1. Interactive mode: Shows summary and cancels batches (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

from modules.config.constants import TERMINAL_BATCH_STATUSES
from modules.core.cli_args import create_cancel_batches_parser
from modules.core.execution_framework import DualModeScript
from modules.llm.openai_sdk_utils import sdk_to_dict, list_all_batches
from modules.ui import (
    print_header,
    print_info,
    print_success,
    print_error,
    ui_print,
    PromptStyle,
    confirm_action,
)
from modules.ui.batch_display import (
    display_batch_summary,
    display_batch_cancellation_results,
)


class CancelBatchesScript(DualModeScript):
    """Script to cancel all non-terminal batch jobs."""
    
    def __init__(self):
        super().__init__("cancel_batches")
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_cancel_batches_parser()
    
    def run_interactive(self) -> None:
        """Cancel batches in interactive mode with prompts."""
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            print_error("Could not import OpenAI SDK. This is often caused by a pydantic/pydantic-core version mismatch.")
            print_info("Try upgrading your environment inside the venv, e.g.:")
            ui_print("  .venv\\Scripts\\python.exe -m pip install --upgrade --upgrade-strategy eager pydantic-core pydantic openai", PromptStyle.DIM)
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

        print_info("Processing cancellations...")
        for b in batches:
            # normalize shape
            bd = sdk_to_dict(b)
            batch_id = bd.get("id") or getattr(b, "id", None)
            status = (str(bd.get("status") or getattr(b, "status", "") or "")).lower()
            if not batch_id:
                continue

            if status in TERMINAL_BATCH_STATUSES:
                self.logger.info(f"Skipping batch {batch_id} with terminal status '{status}'.")
                skipped_batches.append((batch_id, status))
                continue

            try:
                client.batches.cancel(batch_id)
                self.logger.info(f"Batch {batch_id} cancelled.")
                cancelled_batches.append((batch_id, status, True))
            except Exception as e:
                self.logger.error(f"Error cancelling batch {batch_id}: {e}")
                cancelled_batches.append((batch_id, status, False))

        # Display cancellation results
        display_batch_cancellation_results(cancelled_batches, skipped_batches)

    def run_cli(self, args: Namespace) -> None:
        """Cancel batches in CLI mode with arguments."""
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            print_error(f"Could not import OpenAI SDK: {e}")
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
                sys.exit(1)
            
            # Filter to non-terminal batches
            batch_ids_to_cancel = []
            for b in batches:
                bd = sdk_to_dict(b)
                batch_id = bd.get("id") or getattr(b, "id", None)
                status = (str(bd.get("status") or getattr(b, "status", "") or "")).lower()
                if batch_id and status not in TERMINAL_BATCH_STATUSES:
                    batch_ids_to_cancel.append(batch_id)
            
            print_info(f"Found {len(batch_ids_to_cancel)} non-terminal batch(es) to cancel.")
        
        if not batch_ids_to_cancel:
            print_info("No batches to cancel.")
            return
        
        # Confirm if not forced
        if not args.force:
            ui_print(f"\n  About to cancel {len(batch_ids_to_cancel)} batch(es).", PromptStyle.WARNING)
            if not confirm_action("Proceed with cancellation?", default=False):
                print_info("Cancellation aborted.")
                return
        
        # Cancel batches
        success_count = 0
        fail_count = 0
        
        for batch_id in batch_ids_to_cancel:
            try:
                client.batches.cancel(batch_id)
                self.logger.info(f"Batch {batch_id} cancelled.")
                print_success(f"Cancelled: {batch_id}")
                success_count += 1
            except Exception as e:
                self.logger.error(f"Error cancelling batch {batch_id}: {e}")
                print_error(f"Failed to cancel {batch_id}: {e}")
                fail_count += 1
        
        # Summary
        print_header("CANCELLATION COMPLETE", "")
        print_success(f"Successfully cancelled: {success_count}")
        if fail_count > 0:
            print_error(f"Failed to cancel: {fail_count}")


def main() -> None:
    """Main entry point."""
    CancelBatchesScript().execute()


if __name__ == "__main__":
    main()