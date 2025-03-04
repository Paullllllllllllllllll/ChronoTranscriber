# check_batches.py
"""
Script to check whether batch jobs have finished successfully (i.e.,
are marked as completed) and—if so—to download and process them.
Temporary .jsonl files and image folders will only be deleted if the final
output is successfully written and all batches in a JSONL file are completed.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set

from openai import OpenAI
from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.text_processing import process_batch_output
from modules.path_utils import validate_paths
from modules.utils import extract_page_number_from_filename, console_print
from modules.user_interface import UserPrompt

logger = setup_logger(__name__)


def load_config() -> Tuple[List[Path], Dict[str, Any]]:
	"""
    Load and parse configuration YAML files. Identify directories to scan and
    retrieve general processing settings (e.g., concurrency limits, keep_raw_images flag).
    """
	config_loader = ConfigLoader()
	config_loader.load_configs()
	paths_config = config_loader.get_paths_config()

	# Add path validation with the new function
	validate_paths(paths_config)

	processing_settings = paths_config.get("general", {})

	file_paths = paths_config.get("file_paths", {})
	scan_dirs: List[Path] = []
	for key, folders in file_paths.items():
		for folder_key in ["input", "output"]:
			folder_path = folders.get(folder_key)
			if folder_path:
				dir_path = Path(folder_path)
				dir_path.mkdir(parents=True, exist_ok=True)
				scan_dirs.append(dir_path.resolve())
	scan_dirs = list(set(scan_dirs))
	return scan_dirs, processing_settings


def extract_custom_id_mapping(temp_file: Path) -> Tuple[
	Dict[str, Dict[str, Any]], Dict[str, int], Dict[int, Dict[str, Any]]]:
	"""
	Extract mapping between custom_ids and image information from the JSONL file.
	Collects metadata from both batch_request entries and image_metadata entries.

	Returns:
		Tuple containing:
		- Dict[str, Dict[str, Any]]: Mapping of custom_id to image information
		- Dict[str, int]: Mapping of custom_id to order index
		- Dict[int, Dict[str, Any]]: Mapping of order index to image metadata
	"""
	custom_id_map = {}
	batch_order = {}
	order_index_map = {}  # Maps order_index -> image metadata

	try:
		with temp_file.open("r", encoding="utf-8") as f:
			for line in f:
				try:
					record = json.loads(line)

					# Process batch_request records from batching.py
					if "batch_request" in record:
						request_data = record["batch_request"]
						custom_id = request_data.get("custom_id")
						image_info = request_data.get("image_info", {})
						if custom_id and image_info:
							custom_id_map[custom_id] = image_info
							# Extract order information if available
							if "order_index" in image_info:
								batch_order[custom_id] = image_info[
									"order_index"]
								order_index_map[
									image_info["order_index"]] = image_info

					# Process image_metadata records from workflow.py
					elif "image_metadata" in record:
						metadata = record["image_metadata"]
						custom_id = metadata.get("custom_id")
						if custom_id:
							custom_id_map[custom_id] = metadata
							if "order_index" in metadata:
								batch_order[custom_id] = metadata["order_index"]
								order_index_map[
									metadata["order_index"]] = metadata

				except json.JSONDecodeError:
					continue
	except Exception as e:
		logger.error(
			f"Error extracting custom_id mapping from {temp_file}: {e}")

	return custom_id_map, batch_order, order_index_map


def process_all_batches(root_folder: Path, processing_settings: Dict[str, Any],
                        client: OpenAI) -> None:
	"""
    Scans the root folder for *_transcription.jsonl files, locates batch IDs
    within those files, checks if ALL batches for a file are completed, and if so,
    downloads the results and writes them to a final text file while preserving
    the original image order.
    """
	console_print(
		f"\n[INFO] Scanning directory '{root_folder}' for temporary batch files...")
	temp_files = list(root_folder.rglob("*_transcription.jsonl"))
	if not temp_files:
		console_print(
			f"[INFO] No temporary batch files found in {root_folder}.")
		logger.info(f"No temporary batch files found in {root_folder}.")
		return

	# Retrieve all batches from OpenAI
	console_print("[INFO] Retrieving list of submitted batches from OpenAI...")
	try:
		batches = list(client.batches.list(limit=100))
		# Create a dictionary of batch ID to batch object for faster lookup
		batch_dict = {batch.id: batch for batch in batches}
	except Exception as e:
		console_print(f"[ERROR] Failed to retrieve batches from OpenAI: {e}")
		logger.exception(f"Error retrieving batches: {e}")
		return

	# Display batch summary
	UserPrompt.display_batch_summary(batches)

	# Process each temporary file
	for temp_file in temp_files:
		# Extract all batch IDs from this temporary file
		batch_ids: Set[str] = set()
		original_image_order: List[Dict[str, Any]] = []
		images_by_page: Dict[int, Dict[str, Any]] = {}

		with temp_file.open("r", encoding="utf-8") as f:
			for line in f:
				try:
					record = json.loads(line)
					if "batch_tracking" in record:
						batch_id = record["batch_tracking"].get("batch_id")
						if batch_id:
							batch_ids.add(batch_id)
					elif "image_name" in record or "pre_processed_image" in record:
						# Store original image record to track order
						image_name = record.get("image_name", "")
						page_num = extract_page_number_from_filename(image_name)
						images_by_page[page_num] = record

				except json.JSONDecodeError:
					continue

		# Sort images by page number
		for page_num in sorted(images_by_page.keys()):
			original_image_order.append(images_by_page[page_num])

		if not batch_ids:
			console_print(
				f"[WARN] No batch IDs found in {temp_file.name}. Skipping this file.")
			continue

		# Check if all batches are completed
		all_completed = True
		missing_batches = []
		completed_count = 0

		for batch_id in batch_ids:
			if batch_id not in batch_dict:
				all_completed = False
				missing_batches.append(batch_id)
				logger.warning(
					f"Batch ID {batch_id} not found in OpenAI batches.")
				continue

			batch = batch_dict[batch_id]
			if batch.status.lower() == "completed":
				completed_count += 1
			else:
				all_completed = False
				logger.info(
					f"Batch {batch_id} has status '{batch.status}' - not yet completed.")

		# Display progress information for this temp file
		UserPrompt.display_batch_processing_progress(
			temp_file,
			batch_ids,
			completed_count,
			len(missing_batches)
		)

		if not all_completed:
			continue

		# All batches are completed, now download and process them
		console_print(
			f"[SUCCESS] All batches for {temp_file.name} are completed. Processing results...")

		# Extract custom_id mapping if available
		custom_id_map, batch_order, order_index_map = extract_custom_id_mapping(
			temp_file)

		# Collect all transcriptions from all batches
		all_transcriptions: List[Dict[str, Any]] = []

		for batch_id in batch_ids:
			batch = batch_dict[batch_id]
			try:
				file_obj = client.files.content(batch.output_file_id)
				file_content = file_obj.content

				# Try to parse the file content directly to get custom_id mapping
				try:
					batch_response_text = file_content.decode(
						'utf-8') if isinstance(file_content,
					                           bytes) else file_content
					batch_response_lines = batch_response_text.strip().split(
						'\n')

					for line in batch_response_lines:
						response_obj = json.loads(line)
						custom_id = response_obj.get("custom_id")

						if "response" in response_obj and "body" in \
								response_obj["response"]:
							response_body = response_obj["response"]["body"]

							if "choices" in response_body and len(
									response_body["choices"]) > 0:
								message_content = response_body["choices"][
									0].get("message", {}).get("content", "")

								try:
									# Try to parse JSON content if present
									transcription_data = json.loads(
										message_content)

									# Store the transcription along with custom_id and any order info
									transcription_entry = {
										"custom_id": custom_id,
										"transcription": transcription_data.get(
											"transcription", ""),
										"order_info": batch_order.get(custom_id,
										                              None),
										"image_info": custom_id_map.get(
											custom_id, {})
									}

									all_transcriptions.append(
										transcription_entry)

								except json.JSONDecodeError:
									# If not JSON, just use the raw content
									all_transcriptions.append({
										"custom_id": custom_id,
										"transcription": message_content,
										"order_info": batch_order.get(custom_id,
										                              None),
										"image_info": custom_id_map.get(
											custom_id, {})
									})
				except Exception as json_parse_error:
					# Fall back to the process_batch_output function if direct parsing fails
					logger.warning(
						f"Could not directly parse batch result, falling back to process_batch_output: {json_parse_error}")
					transcriptions = process_batch_output(file_content)

					# In fallback mode, we can't maintain page order reliably
					# Just append in the order they come
					for idx, transcription in enumerate(transcriptions):
						all_transcriptions.append({
							"transcription": transcription,
							"fallback_index": idx
						})

			except Exception as e:
				logger.exception(f"Error downloading batch {batch_id}: {e}")
				console_print(
					f"[ERROR] Failed to download output for Batch ID {batch_id}: {e}")
				# If any batch fails to download, skip writing the output
				all_completed = False

		if not all_completed:
			console_print(
				f"[WARN] Failed to process all batches for {temp_file.name}. Skipping output writing.")
			continue

		if not all_transcriptions:
			logger.warning(
				f"No transcriptions extracted for any batch in {temp_file.name}.")
			console_print(
				f"[WARN] No transcriptions extracted for any batch in {temp_file.name}. Skipping this file.")
			continue

		# Build the final text file path
		identifier = temp_file.stem.replace("_transcription", "")
		final_txt_path = temp_file.parent / f"{identifier}_transcription.txt"

		# Apply a multi-level sorting strategy
		console_print(
			f"[INFO] Arranging transcriptions in the correct order...")
		console_print(
			f"[INFO] Found {len(all_transcriptions)} transcriptions to combine.")
		console_print(
			f"[INFO] Order tracking data: {len(batch_order)} custom IDs mapped to order indices.")
		console_print(
			f"[INFO] Using multi-level sorting to ensure correct page order.")

		# Define a sorting key function that tries multiple sorting methods
		def get_sorting_key(transcription_entry):
			"""Returns a sorting key tuple based on available ordering information"""
			# 1. Try order_info from batch processing
			order_info = transcription_entry.get("order_info")
			if order_info is not None:
				return (0, order_info, 0, 0)

			# 2. Try order_index from image_info in custom_id_map
			custom_id = transcription_entry.get("custom_id")
			if custom_id in batch_order:
				return (1, batch_order[custom_id], 0, 0)

			# 3. Try page_number from image_info
			image_info = transcription_entry.get("image_info", {})
			page_number = image_info.get("page_number")
			if page_number is not None:
				return (2, page_number, 0, 0)

			# 4. Try to extract page number from image name
			image_name = image_info.get("image_name", "")
			if image_name:
				page_num = extract_page_number_from_filename(image_name)
				return (3, page_num, 0, 0)

			# 5. Fallback to fallback_index
			fallback_index = transcription_entry.get("fallback_index", 999999)
			return (4, fallback_index, 0, 0)

		# Sort the transcriptions using the multi-level key
		all_transcriptions.sort(key=get_sorting_key)

		# Log the sorting order for debugging
		logger.info(
			f"Sorted {len(all_transcriptions)} transcriptions for {temp_file.name}")
		for idx, entry in enumerate(all_transcriptions):
			custom_id = entry.get("custom_id", "unknown")
			order_info = entry.get("order_info")
			image_name = entry.get("image_info", {}).get("image_name",
			                                             "unknown")
			logger.info(
				f"Position {idx}: custom_id={custom_id}, order_info={order_info}, image={image_name}")

		# Extract just the transcription text in sorted order
		ordered_transcriptions = [t.get("transcription", "") for t in
		                          all_transcriptions]

		processing_success = False
		try:
			with final_txt_path.open("w", encoding="utf-8") as fout:
				for text in ordered_transcriptions:
					if text:  # Only write non-empty transcriptions
						fout.write(text + "\n")
			logger.info(
				f"All batches for {temp_file.name} processed and saved to {final_txt_path}")
			console_print(
				f"[SUCCESS] Processed all batches for {temp_file.name}. Results saved to {final_txt_path.name}")
			processing_success = True
		except Exception as e:
			logger.exception(
				f"Error writing final output for {temp_file.name}: {e}")
			console_print(
				f"[ERROR] Failed to write final output for {temp_file.name}: {e}")

		# Delete temporary JSONL file if we successfully wrote the final output
		if processing_success and not processing_settings.get(
				"retain_temporary_jsonl", True):
			try:
				temp_file.unlink()
				logger.info(
					f"Deleted temporary file after processing: {temp_file}")
				console_print(
					f"[CLEANUP] Deleted temporary file: {temp_file.name}")
			except Exception as e:
				logger.exception(
					f"Error deleting temporary file {temp_file}: {e}")
				console_print(
					f"[ERROR] Could not delete temporary file {temp_file.name}: {e}")

	console_print(
		f"\n[INFO] Completed processing batches in directory: {root_folder}")
	logger.info(
		f"Batch results processing complete for directory: {root_folder}")


def main() -> None:
	"""
    Entrypoint for checking all directories for batch outputs and finalizing them.
    """
	scan_dirs, processing_settings = load_config()
	client = OpenAI()
	for directory in scan_dirs:
		process_all_batches(directory, processing_settings, client)
	console_print(
		"\n[INFO] Batch results processing complete across all directories.")
	logger.info("Batch results processing complete across all directories.")


if __name__ == "__main__":
	main()
