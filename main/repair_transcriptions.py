# repair_transcriptions.py
"""
CLI script for repairing transcription batches.

This is a thin delegator to the operations.repair module.
"""

from __future__ import annotations

import asyncio
import sys

from modules.infra.logger import setup_logger
from modules.core.utils import console_print
from modules.operations.repair.run import main as repair_main

logger = setup_logger(__name__)


if __name__ == "__main__":
    try:
        asyncio.run(repair_main())
    except KeyboardInterrupt:
        console_print("\n[INFO] Repair interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error in repair_transcriptions: %s", e)
        console_print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
