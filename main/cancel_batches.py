"""
cancel_batches.py

Cancel all non-terminal batches using robust pagination and clear summaries.
Terminal statuses: completed, expired, cancelled, failed.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from modules.logger import setup_logger
from modules.utils import console_print
from modules.ui.core import UserPrompt
from modules.openai_sdk_utils import sdk_to_dict, list_all_batches

logger = setup_logger(__name__)
TERMINAL_STATUSES = {"completed", "expired", "cancelled", "failed"}


def main() -> None:
    # Lazy import to avoid crashing on environment import issues and to print a helpful message
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        console_print("[ERROR] Could not import OpenAI SDK. This is often caused by a pydantic/pydantic-core version mismatch.")
        console_print("[HINT] Try upgrading your environment inside the venv, e.g.:")
        console_print("  .venv\\Scripts\\python.exe -m pip install --upgrade --upgrade-strategy eager pydantic-core pydantic openai")
        console_print("Then run again. Original error: " + str(e))
        sys.exit(1)

    client = OpenAI()
    console_print("[INFO] Retrieving list of batches (with pagination)...")
    try:
        batches = list_all_batches(client)
    except Exception as e:
        logger.error(f"Error listing batches: {e}")
        console_print(f"[ERROR] Error listing batches: {e}")
        return

    # Display batch summary
    UserPrompt.display_batch_summary(batches)

    if not batches:
        return

    # Track batches for cancellation and those that are skipped
    cancelled_batches = []  # (batch_id, status, success)
    skipped_batches = []  # (batch_id, status)

    console_print("\n[INFO] Processing cancellations...")
    for b in batches:
        # normalize shape
        bd = sdk_to_dict(b)
        batch_id = bd.get("id") or getattr(b, "id", None)
        status = (str(bd.get("status") or getattr(b, "status", "") or "")).lower()
        if not batch_id:
            continue

        if status in TERMINAL_STATUSES:
            logger.info(f"Skipping batch {batch_id} with terminal status '{status}'.")
            skipped_batches.append((batch_id, status))
            continue

        try:
            client.batches.cancel(batch_id)
            logger.info(f"Batch {batch_id} cancelled.")
            cancelled_batches.append((batch_id, status, True))
        except Exception as e:
            logger.error(f"Error cancelling batch {batch_id}: {e}")
            cancelled_batches.append((batch_id, status, False))

    # Display cancellation results
    UserPrompt.display_batch_cancellation_results(cancelled_batches, skipped_batches)


if __name__ == "__main__":
    main()