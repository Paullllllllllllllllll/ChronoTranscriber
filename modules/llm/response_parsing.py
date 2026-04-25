# modules/processing/response_parsing.py
# Python 3.11+ • PEP8-compliant

from __future__ import annotations

import json
from typing import Any, Dict, List
import re

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

# --- Global placeholder detection (exported) ---
# Allow optional leading header like "image_name:" before the bracket
_PLACEHOLDER_RE_ERROR = re.compile(r"^(?:[^\[]+?:\s*)?\[\s*transcription\s+error.*\]$", re.IGNORECASE)
_PLACEHOLDER_RE_NO_TEXT = re.compile(r"^(?:[^\[]+?:\s*)?\[\s*No\s+transcribable\s+text.*\]$", re.IGNORECASE)
_PLACEHOLDER_RE_NOT_POSSIBLE = re.compile(r"^(?:[^\[]+?:\s*)?\[\s*Transcription\s+not\s+possible.*\]$", re.IGNORECASE)


def detect_transcription_cause(text: str) -> str:
    """Detect the cause type of a transcription line.

    Returns one of: 'api_error' | 'no_text' | 'not_possible' | 'ok'
    """
    s = (text or "").strip()
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
    - If the text is a known placeholder (error/no_text/not_possible), keep it inline on the header line.
    - Otherwise, emit a header line followed by the page transcription on the next line(s).
    """
    # Prefer image filename for tracking/debugging
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
        # Keep placeholder inline with header
        return f"{header} {s}"
    # For normal text, return only the transcription without any header
    return s


def _extract_from_responses_object(data: Dict[str, Any]) -> str:
    """
    Normalize **Responses API** output into a string.
    Tries `output_text`; if missing, reconstructs from `output[*].content[].text`.
    """
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return str(data["output_text"]).strip()

    parts: List[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []):
                    t = c.get("text")
                    if isinstance(t, str):
                        parts.append(t)
    return "".join(parts).strip()


def _try_parse_json(text: str) -> Dict[str, Any] | None:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# Regex to match code-fenced blocks: ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """Extract content from markdown code fences if present.

    Handles patterns like::

        ```json
        {"transcription": "..."}
        ```

    If one or more fenced blocks are found, returns the last one (models
    sometimes emit a "thinking" block followed by the real output).
    If no fences are found, returns the original text unchanged.
    """
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return str(matches[-1]).strip()
    return text


def _normalize_llm_text(text: str) -> str:
    """Normalize raw LLM text output for JSON extraction.

    Applies two transformations in order:

    1. Strip leading/trailing whitespace.
    2. Extract content from code fences (if present).
    3. If the result still does not start with ``{``, attempt to isolate a
       JSON object by stripping preamble/postamble prose.

    The function is idempotent: passing already-clean JSON through returns
    it unchanged.
    """
    if not text:
        return text

    stripped = text.strip()

    # Step 1: Code fence extraction
    defenced = _strip_code_fences(stripped)
    if defenced != stripped:
        return defenced.strip()

    # Step 2: If text doesn't start with '{' but contains '{' and '}',
    # try to isolate the JSON object (handles preamble/postamble).
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


def _salvage_last_json_object(text: str) -> Dict[str, Any] | None:
    """
    When the model returns concatenated JSON objects or mixed prose+JSON,
    try to salvage the last valid JSON object near the end of the string.
    """
    if not text:
        return None
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    # Walk backwards to find a starting '{' that yields a valid JSON object
    i = last_close
    while i >= 0:
        if text[i] == "{":
            candidate = text[i:last_close + 1]
            obj = _try_parse_json(candidate)
            if obj is not None:
                return obj
        i -= 1
    return None


def _check_transcription_flags(parsed: dict[str, Any]) -> str | None:
    """Return a placeholder string if schema flags indicate no usable text, else None."""
    if parsed.get("no_transcribable_text", False):
        return "[No transcribable text]"
    if parsed.get("transcription_not_possible", False):
        return "[Transcription not possible]"
    return None


def _extract_from_schema_object(result: dict[str, Any]) -> str:
    """Extract transcription from a normalized schema object."""
    flag = _check_transcription_flags(result)
    if flag is not None:
        return flag
    return str(result.get("transcription", "")).strip()


def _extract_from_chat_completions(result: dict[str, Any], image_name: str) -> str:
    """Extract transcription from a Chat Completions response."""
    choices = result.get("choices")
    if not choices or not isinstance(choices, list):
        return ""

    message = choices[0].get("message", {})

    # Structured (message.parsed)
    if "parsed" in message and message["parsed"]:
        parsed = message["parsed"]
        if isinstance(parsed, dict):
            flag = _check_transcription_flags(parsed)
            if flag is not None:
                return flag
            transcription_value = parsed.get("transcription")
            return transcription_value.strip() if transcription_value is not None else ""
        else:
            try:
                parsed_obj = json.loads(parsed)
                flag = _check_transcription_flags(parsed_obj)
                if flag is not None:
                    return flag
                transcription_value = parsed_obj.get("transcription")
                return transcription_value.strip() if transcription_value is not None else ""
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.error("Error parsing structured output for %s: %s", image_name, exc)
                return ""

    # Plain content
    content = str(message.get("content", "")).strip()
    if content:
        normalized = _normalize_llm_text(content)
        if normalized.lstrip().startswith("{"):
            parsed = _try_parse_json(normalized)
            if parsed is None:
                parsed = _salvage_last_json_object(normalized)
            if parsed is not None:
                flag = _check_transcription_flags(parsed)
                if flag is not None:
                    return flag
                if "transcription" in parsed:
                    return str(parsed["transcription"]).strip()
        return normalized if normalized else content

    logger.error("Empty content field in response for %s: %s", image_name, json.dumps(result))
    return "[transcription error]"


def extract_transcribed_text(result: Dict[str, Any], image_name: str = "") -> str:
    """
    Extract a transcription string from either:
      - A structured object already containing our schema keys
      - A **Responses API** payload (preferred)
      - A legacy Chat Completions payload

    Returns
    -------
    str
        The transcription (or a helpful placeholder).
    """
    # Early exit for error responses (connection failures, etc.)
    if (
        isinstance(result, dict)
        and result.get("error")
        and not result.get("output_text")
        and not result.get("choices")
    ):
        logger.warning("API error for %s: %s", image_name, result["error"])
        return "[transcription error]"

    # Case 1: Already normalized (schema object)
    if isinstance(result, dict) and "transcription" in result:
        return _extract_from_schema_object(result)

    # Case 2: Responses API object
    if isinstance(result, dict) and ("output_text" in result or "output" in result):
        text = _extract_from_responses_object(result)
        if text:
            normalized = _normalize_llm_text(text)
            stripped = normalized.lstrip()
            if stripped.startswith("{"):
                parsed = _try_parse_json(stripped)
                if parsed is None:
                    parsed = _salvage_last_json_object(stripped)
                if parsed is not None:
                    flag = _check_transcription_flags(parsed)
                    if flag is not None:
                        return flag
                    if "transcription" in parsed:
                        return str(parsed["transcription"]).strip()
            # If normalization changed the text but JSON parsing failed,
            # still try salvaging from the original text
            if normalized != text.strip() and "{" in text:
                parsed = _salvage_last_json_object(text)
                if parsed is not None:
                    flag = _check_transcription_flags(parsed)
                    if flag is not None:
                        return flag
                    if "transcription" in parsed:
                        return str(parsed["transcription"]).strip()
            # Fallback: return normalized text (fences stripped)
            return normalized if normalized else text

    # Case 3: Legacy Chat Completions
    chat_result = _extract_from_chat_completions(result, image_name)
    if chat_result:
        return chat_result

    # Last resort
    logger.error("Unrecognized response shape for image %s: %s", image_name, json.dumps(result))
    return "[transcription error]"


def process_batch_output(file_content: bytes) -> List[str]:
    """
    Parse the JSONL content from an OpenAI Batch output file.

    Supports:
      - Responses API lines: { response: { status_code, body: {...} }, ... }
      - Legacy Chat Completions lines (older jobs)
      - A top-level JSON array or plain JSONL lines

    Returns
    -------
    List[str]
        One transcription per line/object.
    """
    content = file_content.decode("utf-8") if isinstance(file_content, bytes) else str(file_content)
    content = content.strip()
    transcriptions: List[str] = []

    # Normalize to a list of JSON lines
    if content.startswith("[") and content.endswith("]"):
        try:
            items = json.loads(content)
            lines = [json.dumps(item) for item in items]
        except (json.JSONDecodeError, ValueError, TypeError):
            lines = content.splitlines()
    else:
        lines = content.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.exception("Error parsing line as JSON: %s", exc)
            continue

        # Prefer Responses 'response.body'
        data: Dict[str, Any] | None = None
        if "response" in obj and isinstance(obj["response"], dict):
            body = obj["response"].get("body")
            if isinstance(body, dict):
                data = body
        # Fallback: legacy 'choices' at the top level
        if data is None and "choices" in obj:
            data = obj

        if data is None:
            # Page/image identification will be added by the caller via format_page_line()
            transcriptions.append("[transcription error]")
            continue

        transcription = extract_transcribed_text(data, obj.get("image_name", ""))
        if transcription:
            transcriptions.append(transcription)

    return transcriptions
