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
from modules.user_interface import UserPrompt

logger = setup_logger(__name__)
TERMINAL_STATUSES = {"completed", "expired", "cancelled", "failed"}


def _sdk_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        j = getattr(obj, "json", None)
        if callable(j):
            import json as _json
            return _json.loads(j())
    except Exception:
        pass
    d: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
            if not callable(val):
                d[name] = val
        except Exception:
            pass
    return d


def _list_all_batches(client: Any) -> List[Dict[str, Any]]:
    batches: List[Dict[str, Any]] = []
    after: Optional[str] = None
    page_idx = 0
    while True:
        page_idx += 1
        page = client.batches.list(limit=100, after=after) if after else client.batches.list(limit=100)
        data = getattr(page, "data", None) or page
        items = [_sdk_to_dict(b) for b in data]
        console_print(f"[INFO] Batches page {page_idx}: fetched {len(items)} item(s)")
        batches.extend(items)
        # Pagination flags
        try:
            has_more = bool(getattr(page, "has_more", False))
            last_id = getattr(page, "last_id", None)
        except Exception:
            try:
                has_more = bool(page.get("has_more", False))
                last_id = page.get("last_id")
            except Exception:
                has_more = False
                last_id = None
        if not has_more or not last_id:
            break
        after = last_id
    return batches


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
        batches = _list_all_batches(client)
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
        bd = _sdk_to_dict(b)
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