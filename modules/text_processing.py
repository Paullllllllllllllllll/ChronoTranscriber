# modules/text_processing.py
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def extract_transcribed_text(result: Dict[str, Any], image_name: str = "") -> str:
    """
    Extract and return the transcribed text from a structured OpenAI API response.
    Supports both batched and non-batched responses.
    If the result already contains the expected schema, it uses that directly.
    """
    # If the result already has the expected keys, use them directly.
    if isinstance(result, dict) and "transcription" in result:
        if result.get("no_transcribable_text", False):
            return "[No transcribable text]"
        if result.get("transcription_not_possible", False):
            return "[Transcription not possible]"
        return result.get("transcription", "").strip()

    # Otherwise, assume the result follows the typical API response structure
    choices = result.get("choices")
    if choices and isinstance(choices, list) and len(choices) > 0:
        message = choices[0].get("message", {})
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
                except Exception as e:
                    logger.error(f"Error parsing structured output for {image_name}: {e}")
                    return ""
        else:
            content = message.get("content", "").strip()
            if content:
                if content.startswith("{"):
                    try:
                        parsed_obj = json.loads(content)
                        if parsed_obj.get("no_transcribable_text", False):
                            return "[No transcribable text]"
                        if parsed_obj.get("transcription_not_possible", False):
                            return "[Transcription not possible]"
                        if "transcription" in parsed_obj:
                            return str(parsed_obj["transcription"]).strip()
                    except Exception as e:
                        logger.error(f"Error parsing content JSON for {image_name}: {e}")
                return content
            else:
                logger.error(f"Empty content field in response for {image_name}: {json.dumps(result)}")
                return json.dumps(result)
    else:
        logger.error(f"No choices in response for image {image_name}: {json.dumps(result)}")
        return json.dumps(result)
