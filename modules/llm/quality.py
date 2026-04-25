"""Content-quality validators and exception for LLM transcription output.

These validators inspect LLM output text (not input document handling),
so they live under ``modules.llm`` next to the providers and response
parsers that produce the text they check.

Detects five known failure modes observed in Qwen 3.5 Flash transcriptions:
1. Hallucination loops — long repeated substrings
2. Truncation — abnormally short output for content-bearing pages
3. System-prompt bleed — schema/prompt text prepended to content
4. Excessive line repetition — map-label loops
5. Invalid transcription markers — icon descriptions, generic placeholder
   flooding, OCR question-mark clusters.

All validators are stateless functions returning None (pass) or a failure
description string. The orchestrator ``validate_content_quality`` calls
them in order and raises ``ContentQualityError`` on the first failure.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class ContentQualityError(Exception):
    """Raised when transcription content fails quality validation.

    Caught by ``_ainvoke_with_retry`` and retried against the
    ``validation_attempts`` budget alongside ``pydantic.ValidationError``
    and ``InputTokensBelowThresholdError``.
    """

    def __init__(self, failure_type: str, detail: str) -> None:
        self.failure_type = failure_type
        self.detail = detail
        super().__init__(
            f"Content quality check failed ({failure_type}): {detail}"
        )


# ── Defaults (used when config keys are missing) ────────────────────────────
# Thresholds are tuned against the 4 known catastrophic Qwen failure modes
# documented in tests/runs/qwen35flash_transcription/grading/:
#   - "de la Playa" repeated ~200 times (hallucination loop)
#   - "M" x 100 (single-char loop)
#   - Truncation to 4-7 lines (< 50 chars)
#   - JSON schema prepended (system-prompt bleed)
# Michelin Guide entries share formatting boilerplate (", AE VISA",
# " - Repas", "fermé"), so thresholds must be well above what routine
# repetition can produce. The defaults below require a substring to be
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

# ── Invalid-transcription-marker defaults ───────────────────────────────────
_DEFAULT_ICON_DESCRIPTION_LITERALS = [
    "!icon:",
    "(icon)",
    "(no icon",
    "[see image",
    "[see attached",
    "[image of",
    "[image:",
    "[figure",
    "[picture",
    "(building icon)",
    "(hotel icon)",
    "(restaurant icon)",
    "(fork and spoon)",
    "(knife and fork)",
    "(telephone symbol)",
    "(phone icon)",
    "(parking icon)",
    "(wheelchair icon)",
    "(star symbol)",
    "(bib gourmand)",
    "[no visible",
    "[cannot transcribe",
    "[omitted",
    "[unreadable",
]
_DEFAULT_GENERIC_PLACEHOLDER_PATTERNS = [
    "[icon]",
    "[symbol]",
    "[character]",
    "[unknown]",
    "[placeholder]",
    "[?]",
    "[marker]",
]
_DEFAULT_MAX_GENERIC_PLACEHOLDER_COUNT = 3
_DEFAULT_MAX_QUESTION_MARK_RATIO = 0.05


def _is_meaningful_substring(s: str, min_alphanum: int) -> bool:
    """Return True if substring contains enough alphanumeric characters."""
    alphanum_count = sum(1 for c in s if c.isalnum())
    return alphanum_count >= min_alphanum


def detect_hallucination_loop(
    text: str, config: Dict[str, Any]
) -> Optional[str]:
    """Detect long repeated substrings or single-character runs."""
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
    """Detect abnormally short transcription for content-bearing pages."""
    if no_transcribable_text or transcription_not_possible:
        return None
    if text is None:
        return (
            "Transcription text is None but flags indicate content expected"
        )

    min_length = int(
        config.get(
            "min_transcription_length", _DEFAULT_MIN_TRANSCRIPTION_LENGTH
        )
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
    """Detect system prompt or schema fragments in transcription output."""
    if not text:
        return None

    patterns = config.get(
        "system_prompt_bleed_patterns",
        _DEFAULT_SYSTEM_PROMPT_BLEED_PATTERNS,
    )
    check_chars = int(
        config.get("bleed_check_chars", _DEFAULT_BLEED_CHECK_CHARS)
    )
    prefix = text[:check_chars]

    for pat in patterns:
        if pat in prefix:
            return (
                f"System prompt bleed detected in first {check_chars} "
                f"chars: pattern '{pat}' found"
            )
    return None


def detect_excessive_line_repetition(
    text: str, config: Dict[str, Any]
) -> Optional[str]:
    """Detect map-label loops where a single line repeats many times."""
    if not text:
        return None

    max_repeat = int(
        config.get("max_line_repeat_count", _DEFAULT_MAX_LINE_REPEAT_COUNT)
    )
    min_line_len = int(
        config.get(
            "min_line_length_for_repeat",
            _DEFAULT_MIN_LINE_LENGTH_FOR_REPEAT,
        )
    )

    lines = text.splitlines()
    counts: Counter[str] = Counter(
        line.strip()
        for line in lines
        if len(line.strip()) > min_line_len
    )
    for line_text, count in counts.most_common(3):
        if count >= max_repeat:
            preview = line_text[:60].replace("\n", "\\n")
            return (
                f"Line repeated {count} times (threshold: {max_repeat}): "
                f"'{preview}'"
            )
    return None


def detect_invalid_transcription_markers(
    text: str, config: Dict[str, Any]
) -> Optional[str]:
    """Detect invalid-transcription markers left by Qwen when the model
    gives up on an icon or abandons specific tags.

    Catches three sub-patterns:

    * **Icon-description drift.**  Literal strings like ``!Icon:``,
      ``(icon)``, ``[see image]``, ``[figure]`` where the model described
      the visible icon instead of transcribing the page.
    * **Generic-placeholder flooding.**  Patterns like ``[icon]`` or
      ``[symbol]`` used many times on one page in place of specific tags.
    * **Question-mark clusters.**  Pages where OCR failure produced
      ``?`` at more than ``max_question_mark_ratio`` of total chars.
    """
    if not text:
        return None

    marker_cfg = config.get("invalid_markers", {}) or {}
    if not marker_cfg.get("enabled", True):
        return None

    text_lower = text.lower()

    literals = marker_cfg.get(
        "icon_description_literals", _DEFAULT_ICON_DESCRIPTION_LITERALS
    )
    for pat in literals:
        if pat.lower() in text_lower:
            preview = pat[:40]
            return f"Icon description marker present: '{preview}'"

    generic_patterns = marker_cfg.get(
        "generic_placeholder_patterns",
        _DEFAULT_GENERIC_PLACEHOLDER_PATTERNS,
    )
    max_count = int(
        marker_cfg.get(
            "max_generic_placeholder_count",
            _DEFAULT_MAX_GENERIC_PLACEHOLDER_COUNT,
        )
    )
    for pat in generic_patterns:
        n = text_lower.count(pat.lower())
        if n > max_count:
            return (
                f"Generic placeholder '{pat}' repeated {n} times "
                f"(threshold: {max_count})"
            )

    max_ratio = float(
        marker_cfg.get(
            "max_question_mark_ratio", _DEFAULT_MAX_QUESTION_MARK_RATIO
        )
    )
    q_count = text.count("?")
    if q_count >= 10 and len(text) > 0:
        ratio = q_count / len(text)
        if ratio > max_ratio:
            return (
                f"Question-mark ratio {ratio:.1%} exceeds threshold "
                f"{max_ratio:.1%} ({q_count} ? in {len(text)} chars)"
            )

    return None


def validate_content_quality(
    transcription_text: Optional[str],
    no_transcribable_text: bool,
    transcription_not_possible: bool,
    config: Dict[str, Any],
) -> None:
    """Run all content quality validators; raise on first failure."""
    if not config.get("enabled", True):
        return

    if no_transcribable_text or transcription_not_possible:
        return

    text = transcription_text or ""

    validators = [
        (
            "hallucination_loop",
            lambda: detect_hallucination_loop(text, config),
        ),
        (
            "truncation",
            lambda: detect_truncation(
                text,
                no_transcribable_text,
                transcription_not_possible,
                config,
            ),
        ),
        (
            "system_prompt_bleed",
            lambda: detect_system_prompt_bleed(text, config),
        ),
        (
            "excessive_line_repetition",
            lambda: detect_excessive_line_repetition(text, config),
        ),
        (
            "invalid_transcription_markers",
            lambda: detect_invalid_transcription_markers(text, config),
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


__all__ = [
    "ContentQualityError",
    "detect_hallucination_loop",
    "detect_truncation",
    "detect_system_prompt_bleed",
    "detect_excessive_line_repetition",
    "detect_invalid_transcription_markers",
    "validate_content_quality",
]
