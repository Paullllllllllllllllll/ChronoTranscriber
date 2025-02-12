# modules/text_processing.py
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_transcribed_text(result: Dict[str, Any],
                             image_name: str = "") -> str:
	"""
    Extract and return the transcribed text from a structured OpenAI API response.

    Parameters:
        result (Dict[str, Any]): The API response.
        image_name (str): Optional image name for logging.

    Returns:
        str: The transcribed text or a placeholder.
    """
	choices = result.get("choices")
	if choices and isinstance(choices, list) and len(choices) > 0:
		message = choices[0].get("message", {})
		if "parsed" in message and message["parsed"]:
			parsed = message["parsed"]
			if isinstance(parsed, dict):
				transcription_value = parsed.get("transcription")
				if transcription_value is not None:
					final_text = transcription_value.strip()
				else:
					if parsed.get("no_transcribable_text", False):
						final_text = "[No transcribable text]"
					elif parsed.get("transcription_not_possible", False):
						final_text = "[Transcription not possible]"
					else:
						final_text = ""
			else:
				try:
					parsed_obj = json.loads(parsed)
					transcription_value = parsed_obj.get("transcription")
					if transcription_value is not None:
						final_text = transcription_value.strip()
					else:
						if parsed_obj.get("no_transcribable_text", False):
							final_text = "[No transcribable text]"
						elif parsed_obj.get("transcription_not_possible",
						                    False):
							final_text = "[Transcription not possible]"
						else:
							final_text = ""
				except Exception as e:
					logger.error(
						f"Error parsing structured output for {image_name}: {e}")
					final_text = ""
		else:
			content = message.get("content", "").strip()
			if content:
				final_text = content
			else:
				logger.error(
					f"Empty content field in response for {image_name}: {json.dumps(result)}")
				final_text = json.dumps(result)
	else:
		logger.error(
			f"No choices in response for image {image_name}: {json.dumps(result)}")
		final_text = json.dumps(result)

	if final_text.strip().startswith("{"):
		try:
			parsed_again = json.loads(final_text)
			if isinstance(parsed_again,
			              dict) and "transcription" in parsed_again:
				final_text = str(parsed_again["transcription"]).strip()
		except Exception as e:
			logger.error(
				f"Error re-parsing final_text JSON for {image_name}: {e}")

	return final_text
