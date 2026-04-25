"""Page-range parsing and filtering for ChronoTranscriber.

Supports specifying subsets of pages/images/sections to process:
  - ``5``          → first 5 pages
  - ``first:5``    → first 5 pages (explicit)
  - ``last:5``     → last 5 pages
  - ``3-7``        → pages 3 through 7 (1-indexed, inclusive)
  - ``3-``         → pages 3 through end
  - ``-7``         → pages 1 through 7
  - ``1,3,5-8``   → pages 1, 3, and 5 through 8 (compound)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pre-compiled patterns
_FIRST_RE = re.compile(r"^first\s*:\s*(\d+)$", re.IGNORECASE)
_LAST_RE = re.compile(r"^last\s*:\s*(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class PageRange:
    """Parsed page-range specification.

    Internally stores either shorthand (first/last *n*) or a list of
    ``(start, end)`` spans where both bounds are **0-indexed** and
    *inclusive*.
    """

    first_n: Optional[int] = None
    last_n: Optional[int] = None
    spans: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    raw: str = ""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def resolve(self, total: int) -> List[int]:
        """Return sorted, deduplicated 0-based page indices for *total* pages.

        If the range exceeds *total*, it is silently clamped.  An empty
        result is possible (e.g. ``first:0``).
        """
        if total <= 0:
            return []

        if self.first_n is not None:
            n = min(self.first_n, total)
            return list(range(n))

        if self.last_n is not None:
            n = min(self.last_n, total)
            return list(range(total - n, total))

        indices: set[int] = set()
        for start, end in self.spans:
            clamped_start = max(0, start)
            clamped_end = min(end, total - 1)
            if clamped_start <= clamped_end:
                indices.update(range(clamped_start, clamped_end + 1))

        return sorted(indices)

    def describe(self) -> str:
        """Human-readable description for UI summaries."""
        if self.first_n is not None:
            return f"first {self.first_n} page(s)"
        if self.last_n is not None:
            return f"last {self.last_n} page(s)"
        if not self.spans:
            return "all pages"
        parts: list[str] = []
        for start, end in self.spans:
            if start == end:
                parts.append(str(start + 1))
            else:
                parts.append(f"{start + 1}-{end + 1}")
        return "pages " + ",".join(parts)

    def is_empty_spec(self) -> bool:
        """Return True when the spec would never select any pages."""
        if self.first_n is not None:
            return self.first_n <= 0
        if self.last_n is not None:
            return self.last_n <= 0
        return len(self.spans) == 0


def parse_page_range(spec: str) -> PageRange:
    """Parse a page-range specification string into a :class:`PageRange`.

    Raises :class:`ValueError` on malformed input.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Page range specification cannot be empty.")

    raw = spec

    # --- shorthand: bare integer → first N ---
    if spec.isdigit():
        n = int(spec)
        if n <= 0:
            raise ValueError(f"Page count must be positive, got {n}.")
        return PageRange(first_n=n, raw=raw)

    # --- shorthand: first:N ---
    m = _FIRST_RE.match(spec)
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError(f"Page count must be positive, got {n}.")
        return PageRange(first_n=n, raw=raw)

    # --- shorthand: last:N ---
    m = _LAST_RE.match(spec)
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError(f"Page count must be positive, got {n}.")
        return PageRange(last_n=n, raw=raw)

    # --- compound / range: comma-separated segments ---
    spans: list[tuple[int, int]] = []
    segments = [s.strip() for s in spec.split(",")]
    for seg in segments:
        if not seg:
            continue
        span = _parse_segment(seg)
        spans.append(span)

    if not spans:
        raise ValueError(f"No valid page segments found in '{spec}'.")

    # Sort and merge overlapping spans
    spans = _merge_spans(spans)
    return PageRange(spans=tuple(spans), raw=raw)


def _parse_segment(seg: str) -> Tuple[int, int]:
    """Parse a single segment like ``3``, ``3-7``, ``3-``, or ``-7``."""
    if "-" not in seg:
        # Single page number
        if not seg.isdigit():
            raise ValueError(
                f"Invalid page number '{seg}'. Expected a positive integer."
            )
        page = int(seg)
        if page <= 0:
            raise ValueError(f"Page numbers are 1-indexed; got {page}.")
        idx = page - 1
        return (idx, idx)

    parts = seg.split("-", 1)
    left, right = parts[0].strip(), parts[1].strip()

    if not left and not right:
        raise ValueError("Invalid range '-'; both sides are empty.")

    if not left:
        # -7 → pages 1..7
        if not right.isdigit():
            raise ValueError(f"Invalid range end '{right}'.")
        end = int(right)
        if end <= 0:
            raise ValueError(f"Page numbers are 1-indexed; got {end}.")
        return (0, end - 1)

    if not right:
        # 3- → pages 3..end (use a very large sentinel; resolve() clamps)
        if not left.isdigit():
            raise ValueError(f"Invalid range start '{left}'.")
        start = int(left)
        if start <= 0:
            raise ValueError(f"Page numbers are 1-indexed; got {start}.")
        return (start - 1, 2**31)

    # Normal range: 3-7
    if not left.isdigit() or not right.isdigit():
        raise ValueError(f"Invalid range '{seg}'.")
    start, end = int(left), int(right)
    if start <= 0 or end <= 0:
        raise ValueError(f"Page numbers are 1-indexed; got '{seg}'.")
    if start > end:
        raise ValueError(
            f"Range start ({start}) must not exceed end ({end}) in '{seg}'."
        )
    return (start - 1, end - 1)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and merge overlapping/adjacent spans."""
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda s: (s[0], s[1]))
    merged: list[tuple[int, int]] = [spans_sorted[0]]
    for start, end in spans_sorted[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
