# modules/workflow.py
import asyncio
import json
import shutil
import datetime
from PIL import Image
import pytesseract
import aiofiles
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from modules.logger import setup_logger
from modules.user_interface import UserConfiguration
from modules.pdf_utils import PDFProcessor, native_extract_pdf_text
from modules.image_utils import ImageProcessor
from modules.openai_utils import transcribe_image_with_openai
from modules.concurrency import run_concurrent_transcription_tasks
from modules.text_processing import extract_transcribed_text
from modules.utils import console_print, extract_page_number_from_filename

logger = setup_logger(__name__)


class WorkflowManager:
	"""
	Manages the processing workflow for PDFs and images based on user configuration.
	"""

	def __init__(self,
	             user_config: UserConfiguration,
	             paths_config: Dict[str, Any],
	             model_config: Dict[str, Any],
	             concurrency_config: Dict[str, Any],
	             image_processing_config: Dict[str, Any]
	             ):
		self.user_config = user_config
		self.paths_config = paths_config
		self.model_config = model_config
		self.concurrency_config = concurrency_config
		self.image_processing_config = image_processing_config
		self.processing_settings = paths_config.get("general", {})

		# Set up output directories
		pdf_output_dir = Path(
			paths_config.get('file_paths', {}).get('PDFs', {}).get('output',
			                                                       'pdfs_out'))
		image_output_dir = Path(
			paths_config.get('file_paths', {}).get('Images', {}).get('output',
			                                                         'images_out'))

		if self.processing_settings.get("input_paths_is_output_path", False):
			pdf_input_dir = Path(
				paths_config.get('file_paths', {}).get('PDFs', {}).get('input',
				                                                       'pdfs_in'))
			image_input_dir = Path(
				paths_config.get('file_paths', {}).get('Images', {}).get(
					'input', 'images_in'))
			self.pdf_output_dir = pdf_input_dir
			self.image_output_dir = image_input_dir
		else:
			self.pdf_output_dir = pdf_output_dir
			self.image_output_dir = image_output_dir

		# Ensure directories exist
		self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
		self.image_output_dir.mkdir(parents=True, exist_ok=True)

	async def tesseract_ocr_image(self, img_path: Path,
	                              tesseract_config: str) -> Optional[str]:
		"""
		Perform OCR on an image using Tesseract.
		Returns the extracted text, or a placeholder if no text is found.
		"""
		try:
			with Image.open(img_path) as img:
				text = pytesseract.image_to_string(img, config=tesseract_config)
				return text.strip() if text.strip() else "[No transcribable text]"
		except Exception as e:
			logger.error(f"Tesseract OCR error on {img_path.name}: {e}")
			return None

	async def process_selected_items(self,
	                                 transcriber: Optional[Any] = None) -> None:
		"""
		Process all selected items based on the user configuration.
		"""
		total_items = len(self.user_config.selected_items)
		console_print(
			f"[INFO] Beginning processing of {total_items} item(s)...")

		for idx, item in enumerate(self.user_config.selected_items, 1):
			console_print(
				f"\n[INFO] Processing item {idx}/{total_items}: {item.name}")

			if self.user_config.processing_type == "images":
				# Process image folder
				await self.process_single_image_folder(
					item,
					transcriber
				)
			else:
				# Process PDF file
				await self.process_single_pdf(
					item,
					transcriber
				)

			console_print(f"[INFO] Completed item {idx}/{total_items}")

		console_print(
			f"[INFO] All {total_items} item(s) processed successfully.")

	async def process_single_pdf(self, pdf_path: Path,
	                             transcriber: Optional[Any]) -> None:
		"""
		Processes a single PDF file for transcription based on the user configuration.
		"""
		pdf_processor = PDFProcessor(pdf_path)
		parent_folder, output_txt_path, temp_jsonl_path = pdf_processor.prepare_output_folder(
			self.pdf_output_dir)
		method = self.user_config.transcription_method

		console_print(f"\n[INFO] Processing PDF: {pdf_path.name}")
		console_print(f"[INFO] Using method: {method}")

		# Check if method is valid for this PDF
		if method == "native" and not pdf_processor.is_native_pdf():
			console_print(
				f"[WARN] PDF '{pdf_path.name}' is not searchable. Switching to tesseract method.")
			method = "tesseract"  # Fall back to Tesseract if native extraction isn't possible

		# Native PDF extraction
		if method == "native":
			text = native_extract_pdf_text(pdf_path)
			try:
				async with aiofiles.open(temp_jsonl_path, 'a',
				                         encoding='utf-8') as jfile:
					record = {
						"file_name": pdf_path.name,
						"timestamp": datetime.datetime.now().isoformat(),
						"method": "native",
						"text_chunk": text,
						"pre_processed_image": None
					}
					await jfile.write(json.dumps(record) + '\n')
				output_txt_path.write_text(text, encoding='utf-8')
				console_print(
					f"[SUCCESS] Extracted text from '{pdf_path.name}' using native method -> {output_txt_path.name}")
			except Exception as e:
				logger.exception(
					f"Error writing native extraction output for {pdf_path.name}: {e}")
				console_print(f"[ERROR] Failed to write output: {e}")

			# Cleanup if not retaining
			if not self.processing_settings.get("retain_temporary_jsonl", True):
				try:
					temp_jsonl_path.unlink()
					console_print(
						f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
				except Exception as e:
					logger.exception(
						f"Error deleting temporary file {temp_jsonl_path}: {e}")
					console_print(
						f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")

			return

		# Non-native PDF: Extract images
		raw_images_folder = parent_folder / "raw_images"
		raw_images_folder.mkdir(exist_ok=True)
		preprocessed_folder = parent_folder / "preprocessed_images"
		preprocessed_folder.mkdir(exist_ok=True)

		target_dpi = self.image_processing_config.get('image_processing',
		                                              {}).get(
			'target_dpi', 300)
		console_print(
			f"[INFO] Extracting and processing images from PDF with {target_dpi} DPI...")
		processed_image_files = await pdf_processor.process_images(
			raw_images_folder, preprocessed_folder, target_dpi)

		# Ensure proper page ordering
		processed_image_files.sort(
			key=lambda p: extract_page_number_from_filename(p.name))
		console_print(
			f"[INFO] Extracted {len(processed_image_files)} page images from PDF.")

		# Delete raw images folder immediately if the flag is false
		if not self.processing_settings.get("keep_raw_images", True):
			if raw_images_folder.exists():
				try:
					shutil.rmtree(raw_images_folder, ignore_errors=True)
					console_print(
						f"[CLEANUP] Deleted raw_images folder: {raw_images_folder.name}")
				except Exception as e:
					logger.exception(
						f"Error deleting raw_images folder {raw_images_folder}: {e}")

		# Handle GPT batch mode
		if method == "gpt" and self.user_config.use_batch_processing:
			try:
				from modules import batching
				console_print(
					f"[INFO] Submitting batch job for {len(processed_image_files)} images...")

				# Record image metadata in the JSONL file before batch submission
				async with aiofiles.open(temp_jsonl_path, 'a',
				                         encoding='utf-8') as f:
					for idx, img_path in enumerate(processed_image_files):
						page_num = extract_page_number_from_filename(
							img_path.name)
						image_record = {
							"image_metadata": {
								"pre_processed_image": str(img_path),
								"image_name": img_path.name,
								"page_number": page_num,
								"order_index": idx,
								"custom_id": f"req-{idx + 1}"
							}
						}
						# Store image metadata before sending batch to help with ordering later
						await f.write(json.dumps(image_record) + "\n")

				# Now submit the batch job - modified to work with updated batching module
				batch_responses, metadata_records = await asyncio.to_thread(
					batching.process_batch_transcription,
					processed_image_files,
					"",
					self.model_config.get("transcription_model", {})
				)

				# Record batch tracking information
				async with aiofiles.open(temp_jsonl_path, 'a',
				                         encoding='utf-8') as f:
					# Write metadata records to help with ordering
					for record in metadata_records:
						await f.write(json.dumps(record) + "\n")

					# Write batch tracking records
					for response in batch_responses:
						tracking_record = {
							"batch_tracking": {
								"batch_id": response.id,
								"timestamp": datetime.datetime.now(
									datetime.timezone.utc).isoformat(),
								"batch_file": str(response.id)
							}
						}
						await f.write(json.dumps(tracking_record) + "\n")

				console_print(
					f"[SUCCESS] Batch submitted for PDF '{pdf_path.name}'.")
				console_print(
					"[INFO] The batch will be processed asynchronously. Use 'check_batches.py' to monitor status.")

				# Delete preprocessed folder if setting indicates
				if not self.processing_settings.get("keep_preprocessed_images",
				                                    True):
					if preprocessed_folder.exists():
						try:
							shutil.rmtree(preprocessed_folder,
							              ignore_errors=True)
						except Exception as e:
							logger.exception(
								f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")

				return
			except Exception as e:
				logger.exception(
					f"Error during GPT batch submission for {pdf_path.name}: {e}")
				console_print(
					f"[ERROR] Failed to submit batch for {pdf_path.name}. Falling back to synchronous processing.")

		# Synchronous processing for GPT or Tesseract
		console_print(
			f"[INFO] Starting {method} transcription for {len(processed_image_files)} images...")

		await self._process_images_with_method(
			processed_image_files,
			method,
			transcriber,
			temp_jsonl_path,
			output_txt_path,
			pdf_path.name
		)

		# If keep_preprocessed_images is false, delete preprocessed folder
		if not self.processing_settings.get("keep_preprocessed_images", True):
			if preprocessed_folder.exists():
				try:
					shutil.rmtree(preprocessed_folder, ignore_errors=True)
				except Exception as e:
					logger.exception(
						f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")

		console_print(
			f"[SUCCESS] Saved transcription for PDF '{pdf_path.name}' -> {output_txt_path.name}")

		# Remove temporary JSONL if not retaining AND not using batch processing
		# For batch processing, we must keep the JSONL files as they contain batch tracking info
		if not self.processing_settings.get("retain_temporary_jsonl",
		                                    True) and not (
				method == "gpt" and self.user_config.use_batch_processing):
			try:
				temp_jsonl_path.unlink()
				console_print(
					f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
			except Exception as e:
				logger.exception(
					f"Error deleting temporary file {temp_jsonl_path}: {e}")
				console_print(
					f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")
		elif method == "gpt" and self.user_config.use_batch_processing:
			console_print(
				f"[INFO] Preserving {temp_jsonl_path.name} for batch tracking (required for retrieval)")

	async def process_single_image_folder(self, folder: Path,
	                                      transcriber: Optional[Any]) -> None:
		"""
		Processes all images in a given folder based on the user configuration.
		"""
		parent_folder, raw_images_folder, preprocessed_folder, temp_jsonl_path, output_txt_path = ImageProcessor.prepare_image_folder(
			folder, self.image_output_dir)
		method = self.user_config.transcription_method

		console_print(f"\n[INFO] Processing folder: {folder.name}")
		console_print(f"[INFO] Using method: {method}")

		# Copy images
		console_print(f"[INFO] Copying and identifying images in folder...")
		copied_images = ImageProcessor.copy_images_to_raw(folder,
		                                                  raw_images_folder)
		if not copied_images:
			console_print(f"[WARN] No images found in {folder}.")
			return

		# Preprocess images
		console_print(f"[INFO] Preprocessing {len(copied_images)} images...")
		processed_image_paths = [
			preprocessed_folder / f"{img.stem}_pre_processed{img.suffix}" for
			img in
			copied_images]
		ImageProcessor.process_images_multiprocessing(copied_images,
		                                              processed_image_paths)

		# Delete raw_images_folder immediately if the flag is false
		if not self.processing_settings.get("keep_raw_images", True):
			if raw_images_folder.exists():
				try:
					shutil.rmtree(raw_images_folder, ignore_errors=True)
					console_print(
						f"[CLEANUP] Deleted raw_images folder: {raw_images_folder.name}")
				except Exception as e:
					logger.exception(
						f"Error deleting raw_images folder {raw_images_folder}: {e}")

		processed_files = [p for p in processed_image_paths if p.exists()]
		try:
			processed_files.sort(
				key=lambda x: extract_page_number_from_filename(x.name))
		except Exception:
			# If page number extraction fails, sort by filename
			processed_files.sort(key=lambda x: x.name)

		if not processed_files:
			console_print(
				f"[WARN] No processed images found in folder '{folder.name}'.")
			return

		# Handle batch mode for GPT
		if method == "gpt" and self.user_config.use_batch_processing:
			try:
				from modules import batching
				console_print(
					f"[INFO] Submitting batch job for {len(processed_files)} images...")

				# Record image metadata in the JSONL file before batch submission
				async with aiofiles.open(temp_jsonl_path, 'a',
				                         encoding='utf-8') as f:
					for idx, img_path in enumerate(processed_files):
						page_num = extract_page_number_from_filename(
							img_path.name)
						image_record = {
							"image_metadata": {
								"pre_processed_image": str(img_path),
								"image_name": img_path.name,
								"folder_name": folder.name,
								"page_number": page_num,
								"order_index": idx,
								"custom_id": f"req-{idx + 1}"
							}
						}
						# Store image metadata before sending batch to help with ordering later
						await f.write(json.dumps(image_record) + "\n")

				# Now submit the batch job - modified to work with updated batching module
				batch_responses, metadata_records = await asyncio.to_thread(
					batching.process_batch_transcription,
					processed_files,
					"",
					self.model_config.get("transcription_model", {})
				)

				# Record batch tracking information
				async with aiofiles.open(temp_jsonl_path, 'a',
				                         encoding='utf-8') as f:
					# Write metadata records to help with ordering
					for record in metadata_records:
						await f.write(json.dumps(record) + "\n")

					# Write batch tracking records
					for response in batch_responses:
						tracking_record = {
							"batch_tracking": {
								"batch_id": response.id,
								"timestamp": datetime.datetime.now(
									datetime.timezone.utc).isoformat(),
								"batch_file": str(response.id)
							}
						}
						await f.write(json.dumps(tracking_record) + "\n")

				console_print(
					f"[SUCCESS] Batch submitted for folder '{folder.name}'.")
				console_print(
					"[INFO] The batch will be processed asynchronously. Use 'check_batches.py' to monitor status.")

				# Delete preprocessed folder if setting indicates
				if not self.processing_settings.get("keep_preprocessed_images",
				                                    True):
					if preprocessed_folder.exists():
						try:
							shutil.rmtree(preprocessed_folder,
							              ignore_errors=True)
						except Exception as e:
							logger.exception(
								f"Error cleaning up preprocessed images for folder '{folder.name}': {e}")

				return
			except Exception as e:
				logger.exception(
					f"Error during GPT batch submission for folder '{folder.name}': {e}")
				console_print(
					f"[ERROR] Failed to submit batch for folder '{folder.name}'. Falling back to synchronous processing.")

		# Synchronous processing (non-batch GPT or Tesseract)
		console_print(
			f"[INFO] Starting {method} transcription for {len(processed_files)} images...")

		await self._process_images_with_method(
			processed_files,
			method,
			transcriber,
			temp_jsonl_path,
			output_txt_path,
			folder.name,
			is_folder=True
		)

		# Delete preprocessed folder if not retaining
		if not self.processing_settings.get("keep_preprocessed_images", True):
			if preprocessed_folder.exists():
				try:
					shutil.rmtree(preprocessed_folder, ignore_errors=True)
				except Exception as e:
					logger.exception(
						f"Error cleaning up preprocessed images for folder '{folder.name}': {e}")

		console_print(
			f"[SUCCESS] Transcription completed for folder '{folder.name}' -> {output_txt_path.name}")

		# Delete temporary JSONL if not retaining AND not using batch processing
		# For batch processing, we must keep the JSONL files as they contain batch tracking info
		if not self.processing_settings.get("retain_temporary_jsonl",
		                                    True) and not (
				method == "gpt" and self.user_config.use_batch_processing):
			try:
				temp_jsonl_path.unlink()
				console_print(
					f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
			except Exception as e:
				logger.exception(
					f"Error deleting temporary file {temp_jsonl_path}: {e}")
				console_print(
					f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")
		elif method == "gpt" and self.user_config.use_batch_processing:
			console_print(
				f"[INFO] Preserving {temp_jsonl_path.name} for batch tracking (required for retrieval)")

	async def _process_images_with_method(
			self,
			image_files: List[Path],
			method: str,
			transcriber: Optional[Any],
			temp_jsonl_path: Path,
			output_txt_path: Path,
			source_name: str,
			is_folder: bool = False
	) -> None:
		"""
		Helper method to process images using the specified method.
		"""

		async def transcribe_single_image_task(
				img_path: Path,
				trans: Optional[Any],
				method: str,
				tesseract_config: str = "--oem 3 --psm 6"
		) -> Tuple[str, str, Optional[str]]:
			"""
			Transcribes a single image file using either GPT (via OpenAI) or Tesseract OCR.
			Returns a tuple containing the image path, image name, and the transcription result.
			"""
			image_name = img_path.name
			try:
				if method == "gpt":
					if not trans:
						logger.error(
							"No transcriber instance provided for GPT usage.")
						return (str(img_path), image_name,
						        f"[transcription error: {image_name}]")
					result = await transcribe_image_with_openai(img_path, trans)
					logger.debug(
						f"OpenAI response for {img_path.name}: {result}")
					try:
						final_text = extract_transcribed_text(result,
						                                      image_name)
					except Exception as e:
						logger.error(
							f"Error extracting transcription for {img_path.name}: {e}. Marking as transcription error.")
						final_text = f"[transcription error: {image_name}]"
				elif method == "tesseract":
					final_text = await self.tesseract_ocr_image(img_path,
					                                            tesseract_config)
				else:
					logger.error(
						f"Unknown transcription method '{method}' for image {img_path.name}")
					final_text = None
				return (str(img_path), image_name, final_text)
			except Exception as e:
				logger.exception(
					f"Error transcribing {img_path.name} with method '{method}': {e}")
				return (
					str(img_path), image_name,
					f"[transcription error: {image_name}]")

		# Set up args for processing
		if method == "gpt":
			args_list = [
				(img, transcriber, method,
				 self.image_processing_config.get('ocr', {}).get(
					 'tesseract_config',
					 "--oem 3 --psm 6"))
				for img in image_files
			]
		else:
			args_list = [
				(img, None, method,
				 self.image_processing_config.get('ocr', {}).get(
					 'tesseract_config',
					 "--oem 3 --psm 6"))
				for img in image_files
			]

		transcription_conf = self.concurrency_config.get("transcription", {})
		concurrency_limit = transcription_conf.get("concurrency_limit", 20)
		delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

		try:
			console_print(
				f"[INFO] Processing with concurrency limit of {concurrency_limit}...")
			results = await run_concurrent_transcription_tasks(
				transcribe_single_image_task, args_list, concurrency_limit,
				delay_between_tasks
			)
		except Exception as e:
			logger.exception(
				f"Error running concurrent transcription tasks for {source_name}: {e}")
			console_print(f"[ERROR] Concurrency error for {source_name}.")
			return

		# Write the transcription results to a temporary JSONL
		async with aiofiles.open(temp_jsonl_path, 'a',
		                         encoding='utf-8') as jfile:
			for result in results:
				if result is None or result[2] is None:
					continue

				if is_folder:
					record = {
						"folder_name": source_name,
						"pre_processed_image": result[0],
						"image_name": result[1],
						"timestamp": datetime.datetime.now().isoformat(),
						"method": method,
						"text_chunk": result[2]
					}
				else:
					record = {
						"file_name": source_name,
						"pre_processed_image": result[0],
						"image_name": result[1],
						"timestamp": datetime.datetime.now().isoformat(),
						"method": method,
						"text_chunk": result[2]
					}

				await jfile.write(json.dumps(record) + "\n")

		# Combine the transcription text in page order
		try:
			combined_text = "\n".join(
				[res[2] for res in results if res and res[2] is not None])
			output_txt_path.write_text(combined_text, encoding='utf-8')
		except Exception as e:
			logger.exception(
				f"Error writing combined text for {source_name}: {e}")
