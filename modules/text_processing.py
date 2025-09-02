# modules/text_processing.py
# Python 3.11+ • PEP8-compliant

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_from_responses_object(data: Dict[str, Any]) -> str:
    """
    Normalize **Responses API** output into a string.
    Tries `output_text`; if missing, reconstructs from `output[*].content[].text`.
    """
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

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


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


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
    # Case 1: Already normalized (schema object)
    if isinstance(result, dict) and "transcription" in result:
        if result.get("no_transcribable_text", False):
            return "[No transcribable text]"
        if result.get("transcription_not_possible", False):
            return "[Transcription not possible]"
        return str(result.get("transcription", "")).strip()

    # Case 2: Responses API object
    if isinstance(result, dict) and ("output_text" in result or "output" in result):
        text = _extract_from_responses_object(result)
        if text:
            if text.startswith("{"):
                parsed = _try_parse_json(text)
                if parsed is not None:
                    if parsed.get("no_transcribable_text", False):
                        return "[No transcribable text]"
                    if parsed.get("transcription_not_possible", False):
                        return "[Transcription not possible]"
                    if "transcription" in parsed:
                        return str(parsed["transcription"]).strip()
            return text

    # Case 3: Legacy Chat Completions object
    choices = result.get("choices")
    if choices and isinstance(choices, list) and len(choices) > 0:
        message = choices[0].get("message", {})
        # Structured (message.parsed)
        if "parsed" in message and message["parsed"]:
            parsed = message["parsed"]
            if isinstance(parsed, dict):
                if parsed.get("no_transcribable_text", False):
                    return "[No transcribable text]"
                if parsed.get("transcription_not_possible", False):
                    return "[Transcription not possible]"
                transcription_value = parsed.get("transcription")
                return transcription_value.strip() if transcription_value is not None else ""
            else:
                try:
                    parsed_obj = json.loads(parsed)
                    if parsed_obj.get("no_transcribable_text", False):
                        return "[No transcribable text]"
                    if parsed_obj.get("transcription_not_possible", False):
                        return "[Transcription not possible]"
                    transcription_value = parsed_obj.get("transcription")
                    return transcription_value.strip() if transcription_value is not None else ""
                except Exception as exc:
                    logger.error("Error parsing structured output for %s: %s", image_name, exc)
                    return ""
        # Plain content
        content = str(message.get("content", "")).strip()
        if content:
            if content.startswith("{"):
                parsed = _try_parse_json(content)
                if parsed is not None:
                    if parsed.get("no_transcribable_text", False):
                        return "[No transcribable text]"
                    if parsed.get("transcription_not_possible", False):
                        return "[Transcription not possible]"
                    if "transcription" in parsed:
                        return str(parsed["transcription"]).strip()
            return content
        logger.error("Empty content field in response for %s: %s", image_name, json.dumps(result))
        return json.dumps(result)

    # Last resort: log and return raw JSON
    logger.error("Unrecognized response shape for image %s: %s", image_name, json.dumps(result))
    return json.dumps(result)


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
        except Exception:
            lines = content.splitlines()
    else:
        lines = content.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            logger.exception("Error parsing line as JSON: %s", exc)
            continue

        # Prefer Responses 'response.body'
        data: Optional[Dict[str, Any]] = None
        if "response" in obj and isinstance(obj["response"], dict):
            body = obj["response"].get("body")
            if isinstance(body, dict):
                data = body
        # Fallback: legacy 'choices' at the top level
        if data is None and "choices" in obj:
            data = obj

        if data is None:
            image_name = obj.get("file_name") or obj.get("image_name") or "[unknown image]"
            transcriptions.append(f"[transcription error: {image_name}]")
            continue

        transcription = extract_transcribed_text(data, obj.get("image_name", ""))
        if transcription:
            transcriptions.append(transcription)

    return transcriptions
