# cancel_batches.py
"""
Script to cancel all ongoing batches.
This script retrieves all batches using the OpenAI API and cancels any batch whose status is not terminal.
Terminal statuses are assumed to be: completed, expired, cancelled, or failed.
"""

from openai import OpenAI
from modules.logger import setup_logger
from modules.utils import console_print

logger = setup_logger(__name__)
TERMINAL_STATUSES = {"completed", "expired", "cancelled", "failed"}

def main() -> None:
    client = OpenAI()
    console_print("Retrieving list of batches...")
    try:
        batches = list(client.batches.list(limit=100))
    except Exception as e:
        logger.error(f"Error listing batches: {e}")
        console_print(f"Error listing batches: {e}")
        return

    if not batches:
        console_print("No batches found.")
        return

    console_print(f"Found {len(batches)} batches. Processing cancellations...")
    for batch in batches:
        batch_id = batch.id
        status = batch.status.lower()
        if status in TERMINAL_STATUSES:
            logger.info(f"Skipping batch {batch_id} with terminal status '{status}'.")
            console_print(f"Skipping batch {batch_id} (status: '{status}').")
            continue

        console_print(f"Cancelling batch {batch_id} (status: '{status}')...")
        try:
            client.batches.cancel(batch_id)
            logger.info(f"Batch {batch_id} cancelled.")
            console_print(f"Batch {batch_id} cancelled.")
        except Exception as e:
            logger.error(f"Error cancelling batch {batch_id}: {e}")
            console_print(f"Error cancelling batch {batch_id}: {e}")


if __name__ == "__main__":
    main()
