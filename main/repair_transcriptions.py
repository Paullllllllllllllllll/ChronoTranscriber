# repair_transcriptions.py
# Thin CLI delegator to operations.repair

from __future__ import annotations

import asyncio
import sys

from modules.logger import setup_logger
from modules.utils import console_print
from modules.operations import repair as repair_ops

logger = setup_logger(__name__)


async def main() -> None:
    await repair_ops.main()


if __name__ == "__main__":
    try:
        asyncio.run(repair_ops.main())
    except KeyboardInterrupt:
        console_print("\n[INFO] Repair interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error in repair_transcriptions: %s", e)
        console_print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
