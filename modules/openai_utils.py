# modules/openai_utils.py
import aiofiles
import base64
from pathlib import Path
import aiohttp
from typing import Dict, Any
from contextlib import asynccontextmanager
import json

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from tenacity import retry, wait_exponential, stop_after_attempt

logger = setup_logger(__name__)

SUPPORTED_IMAGE_FORMATS = {
	'.png': 'image/png',
	'.jpg': 'image/jpeg',
	'.jpeg': 'image/jpeg'
}


class OpenAITranscriber:
	def __init__(self, api_key: str, system_prompt_path: Path,
	             schema_path: Path, model: str = None) -> None:
		self.api_key = api_key
		self.model = model if model else "gpt-4o-2024-08-06"
		self.endpoint = "https://api.openai.com/v1/chat/completions"
		self.system_prompt_path = system_prompt_path
		self.schema_path = schema_path

		if not self.system_prompt_path.exists():
			logger.error(
				f"System prompt file not found: {self.system_prompt_path}")
			raise FileNotFoundError(
				f"System prompt file does not exist: {self.system_prompt_path}")
		try:
			with self.system_prompt_path.open('r',
			                                  encoding='utf-8') as prompt_file:
				self.system_prompt_text = prompt_file.read().strip()
		except Exception as e:
			logger.error(f"Failed to read system prompt: {e}")
			raise

		if not self.schema_path.exists():
			logger.error(f"Schema file not found: {self.schema_path}")
			raise FileNotFoundError(
				f"Schema file does not exist: {self.schema_path}")
		try:
			with self.schema_path.open('r', encoding='utf-8') as schema_file:
				self.transcription_schema = json.load(schema_file)
		except Exception as e:
			logger.error(f"Failed to load transcription schema: {e}")
			raise

		config_loader = ConfigLoader()
		config_loader.load_configs()
		self.model_config = config_loader.get_model_config()
		self.temperature = self.model_config.get('transcription_model', {}).get(
			'temperature', 0.0)
		self.max_tokens = self.model_config.get('transcription_model', {}).get(
			'max_tokens', 4096)
		self.session = aiohttp.ClientSession()

	async def close(self) -> None:
		if self.session and not self.session.closed:
			await self.session.close()

	@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
	       stop=stop_after_attempt(5))
	async def transcribe_image(self, image_path: Path) -> Dict[str, Any]:
		"""
        Transcribe the given image using the OpenAI API.

        Returns:
            Dict[str, Any]: The API response.
        """
		mime_type = SUPPORTED_IMAGE_FORMATS.get(image_path.suffix.lower())
		if not mime_type:
			logger.error(f"Unsupported image format for {image_path.name}.")
			raise ValueError(f"Unsupported image format: {image_path.suffix}")

		data_url = await self.encode_image(image_path, mime_type)

		messages = [
			{
				"role": "system",
				"content": self.system_prompt_text
			},
			{
				"role": "user",
				"content": [
					{"type": "text",
					 "text": "Please analyze and transcribe the text from this image according to the provided instructions."},
					{"type": "image_url",
					 "image_url": {"url": data_url, "detail": "high"}}
				]
			}
		]

		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json"
		}

		payload: Dict[str, Any] = {
			"model": self.model,
			"messages": messages,
			"temperature": self.temperature,
			"max_tokens": self.max_tokens,
			"top_p": self.model_config.get('transcription_model', {}).get(
				'top_p', 1.0),
			"frequency_penalty": self.model_config.get('transcription_model',
			                                           {}).get(
				'frequency_penalty', 0.0),
			"presence_penalty": self.model_config.get('transcription_model',
			                                          {}).get(
				'presence_penalty', 0.0),
			"response_format": {
				"type": "json_schema",
				"json_schema": self.transcription_schema
			}
		}

		try:
			async with self.session.post(self.endpoint, headers=headers,
			                             json=payload) as response:
				if response.status != 200:
					error_text = await response.text()
					logger.error(
						f"OpenAI API error for {image_path.name}: {error_text}")
					raise Exception(f"OpenAI API error: {error_text}")
				data = await response.json()
				return data
		except aiohttp.ClientError as e:
			logger.error(
				f"HTTP error during OpenAI API call for {image_path.name}: {e}")
			raise

	async def encode_image(self, image_path: Path, mime_type: str) -> str:
		"""
        Encode an image to a data URL.

        Returns:
            str: The data URL.
        """
		try:
			async with aiofiles.open(image_path, 'rb') as image_file:
				content = await image_file.read()
				encoded = base64.b64encode(content).decode("utf-8")
				data_url = f"data:{mime_type};base64,{encoded}"
				return data_url
		except Exception as e:
			logger.error(f"Failed to encode image {image_path.name}: {e}")
			raise


@asynccontextmanager
async def open_transcriber(api_key: str, system_prompt_path: Path,
                           schema_path: Path, model: str = None):
	transcriber = OpenAITranscriber(api_key, system_prompt_path, schema_path,
	                                model)
	try:
		yield transcriber
	finally:
		await transcriber.close()


async def transcribe_image_with_openai(image_path: Path,
                                       transcriber: OpenAITranscriber) -> Dict[
	str, Any]:
	"""
    Transcribe an image using the provided OpenAITranscriber.

    Returns:
        Dict[str, Any]: The transcription result.
    """
	return await transcriber.transcribe_image(image_path)
