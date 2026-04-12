"""Content quality validators for transcription output.

Detects four known failure modes observed in Qwen 3.5 Flash transcriptions:
1. Hallucination loops — long repeated substrings
2. Truncation — abnormally short output for content-bearing pages
3. System-prompt bleed — schema/prompt text prepended to content
4. Excessive line repetition — map-label loops

All validators are stateless functions returning None (pass) or a failure
description string.  The orchestrator ``validate_content_quality`` calls
them in order and raises ``ContentQualityError`` on the first failure.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Optional

from modules.processing.content_quality_error import ContentQualityError

logger = logging.getLogger(__name__)

# ── Defaults (used when config keys are missing) ────────────────────────────
# Thresholds are tuned against the 4 known catastrophic Qwen failure modes
# documented in tests/runs/qwen35flash_transcription/grading/:
#   - "de la Playa" repeated ~200 times (hallucination loop)
#   - "M" x 100 (single-char loop)
#   - Truncation to 4-7 lines (< 50 chars)
#   - JSON schema prepended (system-prompt bleed)
# Michelin Guide entries share formatting boilerplate (", AE VISA",
# " - Repas", "fermé"), so thresholds must be well above what routine
# repetition can produce.  The defaults below require a substring to be
# *predominantly alphanumeric* (not whitespace/punctuation noise) and
# to appear a number of times that only a true loop would produce.
_DEFAULT_MAX_REPEATED_SUBSTR_LEN = 30
_DEFAULT_MAX_REPEATED_SUBSTR_COUNT = 15
_DEFAULT_MIN_ALPHANUM_IN_SUBSTR = 15
_DEFAULT_MAX_SINGLE_CHAR_REPEAT = 75
_DEFAULT_MIN_TRANSCRIPTION_LENGTH = 50
_DEFAULT_MAX_LINE_REPEAT_COUNT = 18
_DEFAULT_MIN_LINE_LENGTH_FOR_REPEAT = 10
_DEFAULT_SYSTEM_PROMPT_BLEED_PATTERNS = [
    '"type": "object"',
    '"additionalProperties"',
    '"transcription_not_possible"',
]
_DEFAULT_BLEED_CHECK_CHARS = 200


def _is_meaningful_substring(s: str, min_alphanum: int) -> bool:
    """Return True if substring contains enough alphanumeric characters.

    Filters out whitespace-only, punctuation-only, and whitespace-padded
    short-word substrings that would otherwise produce false positives
    on pages with table-of-contents dots, column padding, or standard
    entry boilerplate.
    """
    alphanum_count = sum(1 for c in s if c.isalnum())
    return alphanum_count >= min_alphanum


def detect_hallucination_loop(text: str, config: Dict[str, Any]) -> Optional[str]:
    """Detect long repeated substrings or single-character runs.

    Returns a failure description or None if clean.
    """
    if not text:
        return None

    max_char_repeat = int(
        config.get("max_single_char_repeat", _DEFAULT_MAX_SINGLE_CHAR_REPEAT)
    )
    pattern = re.compile(r"(.)\1{" + str(max_char_repeat - 1) + r",}")
    m = pattern.search(text)
    if m:
        char = m.group(1)
        run_len = len(m.group(0))
        return (
            f"Single character '{char}' repeated {run_len} times "
            f"(threshold: {max_char_repeat})"
        )

    min_len = int(
        config.get(
            "max_repeated_substring_length", _DEFAULT_MAX_REPEATED_SUBSTR_LEN
        )
    )
    max_count = int(
        config.get(
            "max_repeated_substring_count", _DEFAULT_MAX_REPEATED_SUBSTR_COUNT
        )
    )
    min_alphanum = int(
        config.get(
            "min_alphanum_in_substring", _DEFAULT_MIN_ALPHANUM_IN_SUBSTR
        )
    )
    if len(text) < min_len * max_count:
        return None

    # Sliding window: check substrings of length min_len..min(80, len/max_count)
    upper = min(80, len(text) // max_count)
    for window in range(min_len, upper + 1, 4):
        seen: Dict[str, int] = {}
        for i in range(len(text) - window + 1):
            chunk = text[i : i + window]
            if not _is_meaningful_substring(chunk, min_alphanum):
                continue
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= max_count:
                preview = chunk[:40].replace("\n", "\\n")
                return (
                    f"Substring of length {window} repeated "
                    f"{seen[chunk]}+ times (threshold: {max_count}): "
                    f"'{preview}...'"
                )
    return None


def detect_truncation(
    text: str,
    no_transcribable_text: bool,
    transcription_not_possible: bool,
    config: Dict[str, Any],
) -> Optional[str]:
    """Detect abnormally short transcription for content-bearing pages.

    Returns a failure description or None if clean.
    """
    if no_transcribable_text or transcription_not_possible:
        return None
    if text is None:
        return "Transcription text is None but flags indicate content expected"

    min_length = int(
        config.get("min_transcription_length", _DEFAULT_MIN_TRANSCRIPTION_LENGTH)
    )
    actual = len(text.strip())
    if actual < min_length:
        return (
            f"Transcription length ({actual} chars) below minimum "
            f"({min_length}) for content-bearing page"
        )
    return None


def detect_system_prompt_bleed(
    text: str, config: Dict[str, Any]
) -> Optional[str]:
    """Detect system prompt or schema fragments in transcription output.

    Returns a failure description or None if clean.
    """
    if not text:
        return None

    patterns = config.get(
        "system_prompt_bleed_patterns", _DEFAULT_SYSTEM_PROMPT_BLEED_PATTERNS
    )
    check_chars = int(config.get("bleed_check_chars", _DEFAULT_BLEED_CHECK_CHARS))
    prefix = text[:check_chars]

    for pat in patterns:
        if pat in prefix:
            return (
                f"System prompt bleed detected in first {check_chars} chars: "
                f"pattern '{pat}' found"
            )
    return None


def detect_excessive_line_repetition(
    text: str, config: Dict[str, Any]
) -> Optional[str]:
    """Detect map-label loops where a single line repeats many times.

    Returns a failure description or None if clean.
    """
    if not text:
        return None

    max_repeat = int(
        config.get("max_line_repeat_count", _DEFAULT_MAX_LINE_REPEAT_COUNT)
    )
    min_line_len = int(
        config.get(
            "min_line_length_for_repeat", _DEFAULT_MIN_LINE_LENGTH_FOR_REPEAT
        )
    )

    lines = text.splitlines()
    counts: Counter[str] = Counter(
        line.strip() for line in lines if len(line.strip()) > min_line_len
    )
    for line_text, count in counts.most_common(3):
        if count >= max_repeat:
            preview = line_text[:60].replace("\n", "\\n")
            return (
                f"Line repeated {count} times (threshold: {max_repeat}): "
                f"'{preview}'"
            )
    return None


def validate_content_quality(
    transcription_text: Optional[str],
    no_transcribable_text: bool,
    transcription_not_possible: bool,
    config: Dict[str, Any],
) -> None:
    """Run all content quality validators; raise on first failure.

    Parameters
    ----------
    transcription_text : str or None
        The transcription text from parsed_output["transcription"].
    no_transcribable_text : bool
        Schema flag indicating no text on page.
    transcription_not_possible : bool
        Schema flag indicating transcription impossible.
    config : dict
        Content quality configuration from concurrency_config.yaml.

    Raises
    ------
    ContentQualityError
        If any validator detects a quality issue.
    """
    if not config.get("enabled", True):
        return

    if no_transcribable_text or transcription_not_possible:
        return

    text = transcription_text or ""

    validators = [
        ("hallucination_loop", lambda: detect_hallucination_loop(text, config)),
        (
            "truncation",
            lambda: detect_truncation(
                text, no_transcribable_text, transcription_not_possible, config
            ),
        ),
        ("system_prompt_bleed", lambda: detect_system_prompt_bleed(text, config)),
        (
            "excessive_line_repetition",
            lambda: detect_excessive_line_repetition(text, config),
        ),
    ]

    for failure_type, check in validators:
        result = check()
        if result is not None:
            logger.warning(
                "Content quality validation failed (%s): %s",
                failure_type,
                result[:200],
            )
            raise ContentQualityError(failure_type, result)
