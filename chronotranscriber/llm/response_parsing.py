"""Turn model responses into transcription text and final-file page lines."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Each pattern allows an optional "image_name:" header before the bracket.
_PLACEHOLDER_RE_ERROR = re.compile(
    r"^(?:[^\[]+?:\s*)?\[\s*transcription\s+error.*\]$", re.IGNORECASE
)
_PLACEHOLDER_RE_NO_TEXT = re.compile(
    r"^(?:[^\[]+?:\s*)?\[\s*No\s+transcribable\s+text.*\]$", re.IGNORECASE
)
_PLACEHOLDER_RE_NOT_POSSIBLE = re.compile(
    r"^(?:[^\[]+?:\s*)?\[\s*Transcription\s+not\s+possible.*\]$", re.IGNORECASE
)


def detect_transcription_cause(text: str) -> str:
    """Detect the cause type of a transcription line.

    Returns one of: 'api_error' | 'no_text' | 'not_possible' | 'ok'
    """
    s = (text or "").strip()
    # Every placeholder pattern ends with "]" at the end of the stripped text.
    if not s.endswith("]"):
        return "ok"
    if _PLACEHOLDER_RE_ERROR.match(s):
        return "api_error"
    if _PLACEHOLDER_RE_NO_TEXT.match(s):
        return "no_text"
    if _PLACEHOLDER_RE_NOT_POSSIBLE.match(s):
        return "not_possible"
    return "ok"


def format_page_line(text: str, page_number: int | None, image_name: str | None) -> str:
    """Return a unified, page-identified output for final files.

    - Prefer the image file name as the header: '<image_name>:'
    - If image_name is missing, fall back to 'Page {n}:' when available,
      otherwise '[unknown image]:'
    - If the text is a known placeholder (error/no_text/not_possible), keep it
      inline on the header line.
    - Otherwise, emit a header line followed by the page transcription on the
      next line(s).
    """
    safe_name = (image_name or "").strip()
    if safe_name:
        header = f"{safe_name}:"
    elif isinstance(page_number, int) and page_number > 0:
        header = f"Page {page_number}:"
    else:
        header = "[unknown image]:"

    s = (text or "").strip()
    cause = detect_transcription_cause(s)
    if cause in {"api_error", "no_text", "not_possible"}:
        return f"{header} {s}"
    return s


def _extract_from_responses_object(data: dict[str, Any]) -> str:
    """Normalize **Responses API** output into a string.

    Tries `output_text`; if missing, reconstructs from `output[*].content[].text`.
    """
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return str(data["output_text"]).strip()

    parts: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []):
                    t = c.get("text")
                    if isinstance(t, str):
                        parts.append(t)
    return "".join(parts).strip()


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    except RecursionError:
        # Deeply nested input exhausts the recursion limit; treat it as unparseable.
        return None


_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)

_FENCE_DELIM_RE = re.compile(r"^\s*```")


def _strip_code_fences(text: str) -> str:
    """Extract content from markdown code fences if present.

    Handles patterns like::

        ```json
        {"transcription": "..."}
        ```

    Applies only when the response starts with a fence. A transcription that
    merely contains a fenced block in running prose passes through unchanged,
    since reducing it to the fence would discard the surrounding text.

    Two shapes start with a fence:

    - A JSON answer wrapped in one fence (wrapped content opens with ``{``):
      only the outermost fence is stripped, so fenced blocks inside the
      ``transcription`` string survive.
    - A "thinking" fenced block followed by the fenced answer: the last
      fenced block is the answer.

    Otherwise returns the original text unchanged.
    """
    if not text.lstrip().startswith("```"):
        return text
    stripped = text.strip()
    lines = stripped.split("\n")
    if (
        len(lines) >= 2
        and _FENCE_DELIM_RE.match(lines[0])
        and _FENCE_DELIM_RE.match(lines[-1])
    ):
        inner = "\n".join(lines[1:-1])
        if inner.lstrip().startswith("{"):
            return inner.strip()
    matches = _CODE_FENCE_RE.findall(stripped)
    if matches:
        return str(matches[-1]).strip()
    return text


def normalize_llm_text(text: str) -> str:
    """Normalize raw LLM text output for JSON extraction.

    Strips surrounding whitespace, extracts the content of a code fence if
    present, and otherwise isolates a JSON object by removing preamble and
    postamble prose. Clean JSON passes through unchanged.
    """
    if not text:
        return text

    stripped = text.strip()

    defenced = _strip_code_fences(stripped)
    if defenced != stripped:
        return defenced.strip()

    trimmed = stripped.lstrip()
    if not trimmed.startswith("{") and "{" in trimmed:
        first_brace = trimmed.index("{")
        candidate = trimmed[first_brace:]
        last_brace = candidate.rfind("}")
        if last_brace != -1:
            json_candidate = candidate[: last_brace + 1]
            parsed = _try_parse_json(json_candidate)
            if parsed is not None:
                return json_candidate.strip()

    return stripped


# Caps the opening braces tried by _salvage_last_json_object, so a string with
# thousands of unbalanced braces cannot cause quadratic re-scanning.
_MAX_SALVAGE_CANDIDATES = 1000


def _salvage_last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last valid JSON object near the end of concatenated or mixed text.

    The backward walk tries at most ``_MAX_SALVAGE_CANDIDATES`` opening
    braces and returns ``None`` beyond that.
    """
    if not text:
        return None
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    attempts = 0
    i = last_close
    while i >= 0:
        if text[i] == "{":
            candidate = text[i : last_close + 1]
            obj = _try_parse_json(candidate)
            if obj is not None:
                return obj
            attempts += 1
            if attempts >= _MAX_SALVAGE_CANDIDATES:
                return None
        i -= 1
    return None


def _check_transcription_flags(parsed: dict[str, Any]) -> str | None:
    """Return a placeholder when the schema flags mark no usable text, else None."""
    if parsed.get("no_transcribable_text", False):
        return "[No transcribable text]"
    if parsed.get("transcription_not_possible", False):
        return "[Transcription not possible]"
    return None


def _transcription_from_object(parsed: dict[str, Any] | None) -> str | None:
    """Return the placeholder or transcription of a schema object, else None."""
    if parsed is None:
        return None
    flag = _check_transcription_flags(parsed)
    if flag is not None:
        return flag
    if "transcription" not in parsed:
        return None
    value = parsed["transcription"]
    # A null transcription must yield "", not the literal "None".
    return str(value).strip() if value is not None else ""


def _extract_from_output_text(text: str) -> str:
    """Parse a response text that may hold fenced, prefixed or plain JSON."""
    normalized = normalize_llm_text(text)
    stripped = normalized.lstrip()
    if stripped.startswith("{"):
        parsed = _try_parse_json(stripped) or _salvage_last_json_object(stripped)
        extracted = _transcription_from_object(parsed)
        if extracted is not None:
            return extracted
    if normalized != text.strip() and "{" in text:
        extracted = _transcription_from_object(_salvage_last_json_object(text))
        if extracted is not None:
            return extracted
    return normalized if normalized else text


def extract_transcribed_text(result: dict[str, Any], image_name: str = "") -> str:
    """Extract a transcription string from a provider response.

    *result* is either a structured object with the schema keys or a
    Responses API payload whose ``output_text`` holds plain text or a loosely
    shaped JSON answer.

    Returns:
        The transcription, or a bracketed placeholder.
    """
    if isinstance(result, dict):
        if result.get("error") and not result.get("output_text"):
            logger.warning("API error for %s: %s", image_name, result["error"])
            return "[transcription error]"

        if "transcription" in result:
            return _transcription_from_object(result) or ""

        if "output_text" in result or "output" in result:
            text = _extract_from_responses_object(result)
            if text:
                return _extract_from_output_text(text)

    logger.error(
        "Unrecognized response shape for image %s: %s", image_name, json.dumps(result)
    )
    return "[transcription error]"
