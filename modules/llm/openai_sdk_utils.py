# modules/openai_sdk_utils.py

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from modules.core.utils import console_print


def sdk_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert an OpenAI SDK object into a plain dict when possible.

    Tries common SDK serialization helpers and finally does a best-effort
    attribute extraction for unknown shapes. If a plain dict is provided,
    it is returned unchanged.
    """
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
            return json.loads(j())
    except Exception:
        pass
    # Last resort: best-effort attribute extraction
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


def list_all_batches(client: Any, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List all batches with robust pagination. Returns a list of dicts.

    Parameters
    ----------
    client : Any
        An initialized OpenAI client (from openai import OpenAI).
    limit : int
        Page size for listing; defaults to 100.
    """
    batches: List[Dict[str, Any]] = []
    after: Optional[str] = None
    page_index = 0
    while True:
        page_index += 1
        page = (
            client.batches.list(limit=limit, after=after)
            if after
            else client.batches.list(limit=limit)
        )
        data = getattr(page, "data", None) or page
        page_items = [sdk_to_dict(b) for b in data]
        batches.extend(page_items)
        # Determine pagination flags
        has_more = False
        last_id = None
        try:
            has_more = bool(getattr(page, "has_more", False))
            last_id = getattr(page, "last_id", None)
        except Exception:
            # Some SDK versions expose as dict
            try:
                has_more = bool(page.get("has_more", False))
                last_id = page.get("last_id")
            except Exception:
                has_more = False
                last_id = None
        console_print(
            f"[INFO] Batches page {page_index}: fetched {len(page_items)} item(s); has_more={has_more}"
        )
        if not has_more or not last_id:
            break
        after = last_id
    return batches


def coerce_file_id(candidate: Any) -> Optional[str]:
    """
    Attempt to coerce various SDK response shapes into a file_id string.
    Supports str, dict-like with id/file_id, and one-item lists of either.
    """
    if isinstance(candidate, str) and candidate:
        return candidate
    if isinstance(candidate, dict):
        cid = candidate.get("id") or candidate.get("file_id")
        return cid if isinstance(cid, str) and cid else None
    if isinstance(candidate, list) and candidate:
        first = candidate[0]
        if isinstance(first, str) and first:
            return first
        if isinstance(first, dict):
            cid = first.get("id") or first.get("file_id")
            return cid if isinstance(cid, str) and cid else None
    return None
