# modules/batching.py
import json
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from modules.utils import extract_page_number_from_filename

from openai import OpenAI

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_FORMATS = {
	'.png': 'image/png',
	'.jpg': 'image/jpeg',
	'.jpeg': 'image/jpeg'
}


def encode_image_to_data_url(image_path: Path) -> str:
	"""
	Encode an image file to a data URL.

	Parameters:
		image_path (Path): The image file path.

	Returns:
		str: The encoded data URL.
	"""
	ext = image_path.suffix.lower()
	mime = SUPPORTED_IMAGE_FORMATS.get(ext)
	if not mime:
		logger.error(f"Unsupported image format: {image_path.suffix}")
		raise ValueError(f"Unsupported image format: {image_path.suffix}")
	with image_path.open("rb") as f:
		encoded_data = base64.b64encode(f.read()).decode("utf-8")
	return f"data:{mime};base64,{encoded_data}"


def create_batch_request_line(
		custom_id: str,
		image_url: str,
		image_info: Dict[str, Any],  # Added parameter for image metadata
		model_config: Dict[str, Any],
		system_prompt_path: Optional[Path] = None,
		schema_path: Optional[Path] = None
) -> str:
	"""
	Create a batch request line for an image transcription task.

	Parameters:
		custom_id (str): A custom identifier for the request.
		image_url (str): The data URL of the image.
		image_info (Dict[str, Any]): Metadata about the image including ordering information.
		model_config (Dict[str, Any]): Model configuration parameters.
		system_prompt_path (Optional[Path]): Path to the system prompt file.
			Defaults to "system_prompt/system_prompt.txt".
		schema_path (Optional[Path]): Path to the transcription schema file.
			Defaults to "schemas/transcription_schema.json".

	Returns:
		str: A JSON string representing the batch request.
	"""
	if system_prompt_path is None:
		system_prompt_path = Path("system_prompt/system_prompt.txt")
	if not system_prompt_path.exists():
		raise FileNotFoundError(
			f"System prompt not found at {system_prompt_path}")
	with system_prompt_path.open("r", encoding="utf-8") as sp_file:
		system_prompt = sp_file.read().strip()

	if schema_path is None:
		schema_path = Path("schemas/transcription_schema.json")
	if not schema_path.exists():
		raise FileNotFoundError(f"Schema file not found at {schema_path}")
	with schema_path.open("r", encoding="utf-8") as schema_file:
		transcription_schema = json.load(schema_file)

	request_body = {
		"model": model_config.get("name", "gpt-4o-2024-08-06"),
		"messages": [
			{"role": "system", "content": system_prompt},
			{
				"role": "user",
				"content": [
					{"type": "text",
					 "text": "Please transcribe the text from this image."},
					{"type": "image_url",
					 "image_url": {"url": image_url, "detail": "high"}}
				]
			}
		],
		"temperature": model_config.get("temperature", 0.0),
		"max_tokens": model_config.get("max_tokens", 4096),
		"top_p": model_config.get("top_p", 1.0),
		"frequency_penalty": model_config.get("frequency_penalty", 0.0),
		"presence_penalty": model_config.get("presence_penalty", 0.0),
		"response_format": {
			"type": "json_schema",
			"json_schema": transcription_schema
		}
	}

	# Include image metadata in the request line
	request_line = {
		"custom_id": custom_id,
		"method": "POST",
		"url": "/v1/chat/completions",
		"body": request_body,
		"image_info": image_info  # Store metadata for proper ordering later
	}
	return json.dumps(request_line)


def write_batch_file(request_lines: List[str], output_path: Path) -> Path:
	"""
	Write batch request lines to a file.

	Parameters:
		request_lines (List[str]): List of JSON strings.
		output_path (Path): The file path to write to.

	Returns:
		Path: The path to the written file.
	"""
	with output_path.open("w", encoding="utf-8") as f:
		for line in request_lines:
			f.write(line + "\n")
	logger.info(f"Batch file written to {output_path}")
	return output_path


def submit_batch(batch_file_path: Path) -> Dict[str, Any]:
	"""
	Submit a batch file to OpenAI.

	Parameters:
		batch_file_path (Path): The path of the batch file.

	Returns:
		Dict[str, Any]: The response from the API.
	"""
	client = OpenAI()  # Instantiate a new client instance
	with batch_file_path.open("rb") as f:
		file_response = client.files.create(
			file=f,
			purpose="batch"
		)
	file_id = file_response.id
	logger.info(f"Uploaded batch file, file id: {file_id}")
	batch_response = client.batches.create(
		input_file_id=file_id,
		endpoint="/v1/chat/completions",
		completion_window="24h",
		metadata={"description": "Batch OCR transcription"}
	)
	logger.info(f"Batch submitted, batch id: {batch_response.id}")
	return batch_response


def process_batch_transcription(image_files: List[Path], prompt_text: str,
                                model_config: Dict[str, Any]) -> List[Any]:
	"""
	Process a batch transcription for a list of image files.

	Parameters:
		image_files (List[Path]): List of image file paths.
		prompt_text (str): Additional prompt text (currently unused).
		model_config (Dict[str, Any]): Model configuration parameters.

	Returns:
		List[Any]: List of batch responses.
	"""
	batch_request_lines = []
	batch_request_records = []  # Track metadata for each request

	for index, image_file in enumerate(image_files):
		try:
			data_url = encode_image_to_data_url(image_file)
			custom_id = f"req-{index + 1}"

			# Create image info with order information
			image_info = {
				"image_name": image_file.name,
				"order_index": index,  # Explicit ordering
				"page_number": extract_page_number_from_filename(
					image_file.name)
			}

			line = create_batch_request_line(custom_id, data_url, image_info,
			                                 model_config)
			batch_request_lines.append(line)

			# Store a record mapping custom_id to image information
			batch_request_records.append({
				"batch_request": {
					"custom_id": custom_id,
					"image_info": image_info
				}
			})
		except Exception as e:
			logger.error(f"Error processing image {image_file}: {e}")

	max_batch_size = 200 * 1024 * 1024  # 200MB in bytes
	batch_files = []
	current_lines = []
	current_size = 0
	batch_index = 1

	for line in batch_request_lines:
		line_bytes = len(line.encode("utf-8"))
		if current_size + line_bytes > max_batch_size and current_lines:
			batch_file = Path(f"batch_requests_part_{batch_index}.jsonl")
			write_batch_file(current_lines, batch_file)
			batch_files.append(batch_file)
			batch_index += 1
			current_lines = []
			current_size = 0
		current_lines.append(line)
		current_size += line_bytes

	if current_lines:
		batch_file = Path(f"batch_requests_part_{batch_index}.jsonl")
		write_batch_file(current_lines, batch_file)
		batch_files.append(batch_file)

	batch_responses = []
	for batch_file in batch_files:
		response = submit_batch(batch_file)

		# First write all the metadata records to help with order reconstruction
		with batch_file.open("a", encoding="utf-8") as f:
			for record in batch_request_records:
				f.write(json.dumps(record) + "\n")

			# Then add the batch tracking record
			tracking_record = {
				"batch_tracking": {
					"batch_id": response.id,
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"batch_file": str(response.id)
				}
			}
			f.write(json.dumps(tracking_record) + "\n")

		batch_responses.append(response)
		try:
			batch_file.unlink()
			logger.info(f"Deleted temporary batch request file: {batch_file}")
		except Exception as e:
			logger.error(
				f"Error deleting temporary batch request file {batch_file}: {e}")

	return batch_responses
