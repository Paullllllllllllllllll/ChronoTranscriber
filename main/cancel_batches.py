# cancel_batches.py
"""
Script to cancel all ongoing batches.
This script retrieves all batches using the OpenAI API and cancels any batch whose status is not terminal.
Terminal statuses are assumed to be: completed, expired, cancelled, or failed.
"""

from openai import OpenAI
from modules.logger import setup_logger
from modules.utils import console_print
from modules.user_interface import UserPrompt

logger = setup_logger(__name__)
TERMINAL_STATUSES = {"completed", "expired", "cancelled", "failed"}


def main() -> None:
	client = OpenAI()
	console_print("[INFO] Retrieving list of batches...")
	try:
		batches = list(client.batches.list(limit=100))
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
	for batch in batches:
		batch_id = batch.id
		status = batch.status.lower()

		if status in TERMINAL_STATUSES:
			logger.info(
				f"Skipping batch {batch_id} with terminal status '{status}'.")
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
	UserPrompt.display_batch_cancellation_results(cancelled_batches,
	                                              skipped_batches)


if __name__ == "__main__":
	main()