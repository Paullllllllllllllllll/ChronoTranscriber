# modules/user_interface.py
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import sys
import os

from modules.logger import setup_logger
from modules.utils import console_print, check_exit, safe_input

logger = setup_logger(__name__)


class UserConfiguration:
	"""
	Stores user's processing preferences to avoid re-prompting during workflow.
	"""

	def __init__(self):
		self.processing_type = None  # "images" or "pdfs"
		self.transcription_method = None  # "native", "tesseract", or "gpt"
		self.use_batch_processing = False
		self.selected_items = []  # files or folders to process
		self.process_all = False  # flag to process all files/folders

	def __str__(self):
		"""String representation for logging purposes."""
		method_name = {
			"native": "Native PDF extraction",
			"tesseract": "Tesseract OCR",
			"gpt": "GPT-based transcription"
		}.get(self.transcription_method, self.transcription_method)

		batch_text = " with batch processing" if self.use_batch_processing else ""

		return (f"Processing type: {self.processing_type}, "
		        f"Method: {method_name}{batch_text}, "
		        f"Process all: {self.process_all}, "
		        f"Selected items: {len(self.selected_items)}")


class UserPrompt:
	"""
	A class for handling all user interaction and prompting functionality.
	Consolidates UI functions from unified_transcriber.py.
	"""

	@staticmethod
	def display_banner():
		"""
		Display a welcome banner with basic information about the application.
		"""
		console_print("\n" + "=" * 80)
		console_print(
			"  CHRONO TRANSCRIBER - Historical Document Digitization Tool")
		console_print("=" * 80)
		console_print(
			"  This tool helps you convert historical documents to searchable text using")
		console_print(
			"  multiple transcription methods tailored to different document types.")
		console_print("=" * 80 + "\n")

	@staticmethod
	def enhanced_select_option(prompt: str, options: List[Tuple[str, str]],
	                           allow_quit: bool = True) -> str:
		"""
		Display a prompt with detailed options and return the user's choice.
		"""
		console_print(f"\n{prompt}")
		console_print("-" * 80)

		for idx, (value, description) in enumerate(options, 1):
			console_print(f"  {idx}. {description}")

		if allow_quit:
			console_print(f"\n  (Type 'q' to exit the application)")

		while True:
			choice = safe_input("\nEnter your choice: ").strip()

			if allow_quit:
				check_exit(choice)

			if choice.isdigit() and 1 <= int(choice) <= len(options):
				return options[int(choice) - 1][0]

			console_print("[ERROR] Invalid selection. Please try again.")

	@staticmethod
	def select_pdfs_or_folders(
			directory: Path,
			selection_type: str = "folders",
			process_subfolders: bool = False
	) -> List[Path]:
		"""
		List and allow selection of folders or PDF files.
		"""
		if selection_type == "folders":
			items = [d for d in directory.iterdir() if d.is_dir()]
			item_description = "folders"
		else:  # pdfs
			if process_subfolders:
				items = list(directory.rglob("*.pdf"))
			else:
				items = [f for f in directory.iterdir() if
				         f.is_file() and f.suffix.lower() == ".pdf"]
			item_description = "PDF files"

		if not items:
			console_print(f"[INFO] No {item_description} found in {directory}")
			return []

		console_print(f"\nAvailable {item_description} ({len(items)}):")
		console_print("-" * 80)

		for idx, item in enumerate(items, 1):
			name = item.name
			if len(name) > 70:
				name = name[:67] + "..."
			console_print(f"  {idx}. {name}")

		console_print("\nSelection options:")
		console_print("  - Enter numbers separated by commas (e.g., '1,3,5')")
		console_print("  - Enter 'all' to select all items")
		console_print("  - Type 'q' to exit")

		while True:
			selection = safe_input("\nMake your selection: ").strip().lower()
			check_exit(selection)

			if selection == "all":
				return items

			try:
				indices = [int(i.strip()) - 1 for i in selection.split(',') if
				           i.strip().isdigit()]
				selected_items = [items[i] for i in indices if
				                  0 <= i < len(items)]

				if not selected_items:
					console_print(
						"[ERROR] No valid items selected. Please try again.")
					continue

				return selected_items

			except (ValueError, IndexError):
				console_print(
					"[ERROR] Invalid selection format. Please try again.")

	@staticmethod
	def select_files(directory: Path, extension: str) -> List[Path]:
		"""
		List files with a given extension and allow the user to select files.
		"""
		files = [f for f in directory.iterdir() if
		         f.is_file() and f.suffix.lower() == extension.lower()]
		if not files:
			console_print(
				f"No files with extension '{extension}' found in {directory}.")
			logger.info(
				f"No files with extension '{extension}' found in {directory}.")
			return []

		console_print(
			f"Files with extension '{extension}' found in {directory}:")
		for idx, file in enumerate(files, 1):
			console_print(f"{idx}. {file.name}")

		selected = safe_input(
			"Enter the numbers of the files to select, separated by commas (or type 'q' to exit): ").strip()
		check_exit(selected)
		try:
			indices = [int(i.strip()) - 1 for i in selected.split(',') if
			           i.strip().isdigit()]
			selected_files = [files[i] for i in indices if 0 <= i < len(files)]
			if not selected_files:
				console_print("No valid files selected.")
				logger.info("No valid files selected by the user.")
			return selected_files
		except ValueError:
			console_print(
				"Invalid input. Please enter numbers separated by commas.")
			logger.error("User entered invalid file selection input.")
			return []

	@staticmethod
	def select_option(prompt: str, options: List[str]) -> str:
		"""
		Display a prompt and a numbered list of options, then return the user's choice as a string.
		"""
		console_print(prompt)
		for idx, option in enumerate(options, 1):
			console_print(f"{idx}. {option}")
		choice = safe_input(
			"Enter the number of your choice (or type 'q' to exit): ")
		check_exit(choice)
		return choice.strip()

	@staticmethod
	def get_processing_type_options() -> List[Tuple[str, str]]:
		"""Return options for processing type selection."""
		return [
			("images",
			 "Image Folders - Process collections of image files organized in folders"),
			("pdfs",
			 "PDF Documents - Process PDF files containing documents or scanned pages")
		]

	@staticmethod
	def get_method_options(processing_type: str) -> List[Tuple[str, str]]:
		"""Return transcription method options based on processing type."""
		if processing_type == "pdfs":
			return [
				("native",
				 "Native PDF extraction - Fast extraction from searchable PDFs (text-based, not scanned)"),
				("tesseract",
				 "Tesseract OCR - Open-source OCR suited for printed text and scanned documents"),
				("gpt",
				 "GPT Transcription - High-quality AI-powered transcription ideal for complex documents")
			]
		else:  # Images
			return [
				("tesseract",
				 "Tesseract OCR - Open-source OCR suited for printed text and straightforward layouts"),
				("gpt",
				 "GPT Transcription - High-quality AI-powered transcription ideal for handwriting and complex layouts")
			]

	@staticmethod
	def get_batch_options() -> List[Tuple[str, str]]:
		"""Return options for batch processing selection."""
		return [
			("yes",
			 "Yes - Batch processing (for large jobs, processes asynchronously, lower cost)"),
			("no",
			 "No - Synchronous processing (for small jobs, immediate results)")
		]

	@staticmethod
	def get_pdf_scope_options() -> List[Tuple[str, str]]:
		"""Return options for PDF processing scope."""
		return [
			("all",
			 "Process all PDFs in the input directory and its subfolders"),
			("subfolders", "Process PDFs from specific subfolders"),
			("specific", "Process specific PDF files")
		]

	@staticmethod
	def configure_processing_type(user_config: UserConfiguration) -> None:
		"""
		Configure the processing type for the user configuration.
		"""
		processing_type_options = UserPrompt.get_processing_type_options()
		user_config.processing_type = UserPrompt.enhanced_select_option(
			"What type of documents would you like to process?",
			processing_type_options
		)

	@staticmethod
	def configure_transcription_method(user_config: UserConfiguration) -> None:
		"""
		Configure the transcription method for the user configuration.
		"""
		method_options = UserPrompt.get_method_options(
			user_config.processing_type)
		user_config.transcription_method = UserPrompt.enhanced_select_option(
			f"Which transcription method would you like to use for {user_config.processing_type}?",
			method_options
		)

	@staticmethod
	def configure_batch_processing(user_config: UserConfiguration) -> None:
		"""
		Configure batch processing options for GPT transcription.
		"""
		if user_config.transcription_method == "gpt":
			batch_options = UserPrompt.get_batch_options()
			batch_choice = UserPrompt.enhanced_select_option(
				"Would you like to use batch processing for GPT transcription?",
				batch_options
			)
			user_config.use_batch_processing = (batch_choice == "yes")

			# Check for API key if using GPT
			api_key = os.getenv('OPENAI_API_KEY')
			if not api_key:
				console_print(
					"[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
				sys.exit(1)

	@staticmethod
	def select_pdfs_workflow(user_config: UserConfiguration,
	                         pdf_input_dir: Path) -> None:
		"""
		Guide user through PDF selection workflow.
		"""
		# PDF selection
		source_dir = pdf_input_dir

		# First, decide on the scope of PDF processing
		pdf_scope_options = UserPrompt.get_pdf_scope_options()
		pdf_scope = UserPrompt.enhanced_select_option(
			"How would you like to select PDFs for processing?",
			pdf_scope_options
		)

		if pdf_scope == "all":
			all_pdfs = list(pdf_input_dir.rglob("*.pdf"))
			if not all_pdfs:
				console_print(
					f"[ERROR] No PDF files found in {pdf_input_dir} or its subfolders.")
				return

			console_print(
				f"[INFO] Found {len(all_pdfs)} PDF file(s) to process.")
			user_config.selected_items = all_pdfs
			user_config.process_all = True

		elif pdf_scope == "subfolders":
			subfolders = [d for d in pdf_input_dir.iterdir() if d.is_dir()]
			if not subfolders:
				console_print(f"[ERROR] No subfolders found in {pdf_input_dir}")
				return

			console_print(
				f"[INFO] Found {len(subfolders)} subfolder(s) in the input directory.")
			user_config.selected_items = UserPrompt.select_pdfs_or_folders(
				pdf_input_dir,
				"folders")

			if not user_config.selected_items:
				console_print("[INFO] No folders selected. Exiting.")
				return

			# Convert folder selection to list of PDFs
			pdf_files = []
			for folder in user_config.selected_items:
				folder_pdfs = list(folder.glob("*.pdf"))
				if not folder_pdfs:
					console_print(
						f"[WARN] No PDF files found in {folder.name}.")
				else:
					pdf_files.extend(folder_pdfs)

			if not pdf_files:
				console_print(
					"[ERROR] No PDF files found in the selected folders.")
				return

			console_print(
				f"[INFO] Found {len(pdf_files)} PDF file(s) in the selected folders.")
			user_config.selected_items = pdf_files

		else:  # specific PDFs
			all_pdfs = list(pdf_input_dir.glob("*.pdf"))
			if not all_pdfs:
				console_print(
					f"[ERROR] No PDF files found directly in {pdf_input_dir}")
				return

			user_config.selected_items = UserPrompt.select_pdfs_or_folders(
				pdf_input_dir,
				"pdfs")

			if not user_config.selected_items:
				console_print("[INFO] No PDFs selected. Exiting.")
				return

	@staticmethod
	def select_images_workflow(user_config: UserConfiguration,
	                           image_input_dir: Path) -> None:
		"""
		Guide user through image folder selection workflow.
		"""
		from modules.image_utils import SUPPORTED_IMAGE_EXTENSIONS

		# Image folder selection
		source_dir = image_input_dir

		# Check if there are subfolders or direct images
		subfolders = [f for f in source_dir.iterdir() if f.is_dir()]
		direct_images = [f for f in source_dir.iterdir() if
		                 f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]

		if not subfolders and not direct_images:
			console_print(
				f"[ERROR] No image folders or images found in {source_dir}")
			console_print(
				"[INFO] Please add image folders or images to the input directory and try again.")
			return

		# If there are no subfolders but direct images exist, treat the input dir as a folder to process
		if not subfolders and direct_images:
			console_print(
				f"[INFO] No subfolders found, but {len(direct_images)} images detected directly in the input directory.")
			process_option = UserPrompt.enhanced_select_option(
				"How would you like to proceed?",
				[("process_direct", "Process these images directly"),
				 ("cancel", "Cancel and create proper subfolders first")]
			)

			if process_option == "process_direct":
				user_config.selected_items = [source_dir]
			else:
				console_print(
					"[INFO] Operation cancelled. Please organize your images into subfolders and try again.")
				return
		else:
			# Normal subfolder selection
			console_print(
				f"[INFO] Found {len(subfolders)} image folder(s) in the input directory.")
			selection_options = [
				("specific", "Select specific folders to process"),
				("all", "Process all folders")
			]
			selection_mode = UserPrompt.enhanced_select_option(
				"Would you like to process all folders or select specific ones?",
				selection_options
			)

			if selection_mode == "all":
				user_config.selected_items = subfolders
				user_config.process_all = True
			else:
				user_config.selected_items = UserPrompt.select_pdfs_or_folders(
					source_dir,
					"folders")
				if not user_config.selected_items:
					console_print("[INFO] No folders selected. Exiting.")
					return

	@staticmethod
	def display_processing_summary(user_config: UserConfiguration) -> bool:
		"""
		Display processing summary and ask for confirmation.
		Returns True if user confirms, False otherwise.
		"""
		console_print("\n" + "=" * 80)
		console_print("  PROCESSING SUMMARY")
		console_print("=" * 80)

		if user_config.processing_type == "images":
			item_type = "image folder(s)"
		else:
			item_type = "PDF file(s)"

		console_print(
			f"\nReady to process {len(user_config.selected_items)} {item_type} with the following settings:")
		console_print(
			f"  - Document type: {user_config.processing_type.capitalize()}")
		console_print(
			f"  - Transcription method: {user_config.transcription_method.capitalize()}")

		if user_config.transcription_method == "gpt":
			batch_mode = "Batch (asynchronous)" if user_config.use_batch_processing else "Synchronous"
			console_print(f"  - Processing mode: {batch_mode}")

		# First few items to show
		console_print("\nSelected items (first 5 shown):")
		for i, item in enumerate(user_config.selected_items[:5]):
			console_print(f"  {i + 1}. {item.name}")

		if len(user_config.selected_items) > 5:
			console_print(
				f"  ... and {len(user_config.selected_items) - 5} more")

	@staticmethod
	def display_batch_summary(batches):
		"""
		Display a summary of batches grouped by status.
		Only shows details for in-progress batches.

		Parameters:
			batches: List of batch items from OpenAI API (SDK objects or dicts)
		"""
		if not batches:
			console_print("No batches found.")
			return

		# Group batches by status
		status_groups = {}
		for batch in batches:
			# Support both object-style and dict-style batches
			status_val = getattr(batch, "status", None)
			if status_val is None and isinstance(batch, dict):
				status_val = batch.get("status")
			status = (status_val or "").lower()
			if status not in status_groups:
				status_groups[status] = []
			status_groups[status].append(batch)

		# Display summary counts
		console_print("\n===== Batch Summary =====")
		console_print(f"Total batches: {len(batches)}")
		for status, batch_list in sorted(status_groups.items()):
			console_print(f"{status.capitalize()}: {len(batch_list)} batch(es)")

		# Display in-progress batches with more detail
		in_progress_statuses = {"validating", "in_progress", "finalizing"}
		for status in in_progress_statuses:
			if status in status_groups and status_groups[status]:
				console_print(f"\n----- {status.capitalize()} Batches -----")
				for batch in status_groups[status]:
					batch_id = getattr(batch, "id", None)
					batch_status = getattr(batch, "status", None)
					if isinstance(batch, dict):
						batch_id = batch.get("id", batch_id)
						batch_status = batch.get("status", batch_status)
					console_print(f"  Batch ID: {batch_id} | Status: {batch_status}")

	@staticmethod
	def _format_page_image(page_number: Optional[int], image_name: str) -> str:
		"""
		Format a concise page/image label. If page_number is None, omit the page label to avoid bogus values.
		"""
		if page_number is None:
			return f"({image_name})"
		return f"page {page_number} ({image_name})"

	@staticmethod
	def print_transcription_item_error(
			image_name: str,
			page_number: Optional[int] = None,
			status_code: Optional[int] = None,
			err_code: Optional[str] = None,
			err_message: Optional[str] = None,
	) -> None:
		"""
		Standardized print for a single per-page transcription error.
		Avoids printing unknown page numbers and prettifies the message.
		"""
		label = UserPrompt._format_page_image(page_number, image_name)
		parts = []
		if status_code is not None:
			parts.append(f"status={status_code}")
		if err_code:
			parts.append(f"code={err_code}")
		if err_message:
			parts.append(f"message={err_message}")
		detail = " ".join(parts)
		console_print(f"[ERROR] {label} failed in batch" + (f": {detail}" if detail else ""))

	@staticmethod
	def print_transcription_not_possible(image_name: str, page_number: Optional[int] = None) -> None:
		"""
		Standardized print for a 'transcription not possible' warning.
		"""
		label = UserPrompt._format_page_image(page_number, image_name)
		console_print(f"[WARN] Model reported transcription not possible for {label}.")

	@staticmethod
	def print_no_transcribable_text(image_name: str, page_number: Optional[int] = None) -> None:
		"""
		Standardized print for a 'no transcribable text' informational message.
		"""
		label = UserPrompt._format_page_image(page_number, image_name)
		console_print(f"[INFO] No transcribable text detected for {label}.")

	@staticmethod
	def display_page_error_summary(error_entries: List[Dict[str, Any]]) -> None:
		"""
		Display a summary list of per-page errors captured during batch processing.
		Each entry is expected to include 'image_info' and 'error_details'.
		"""
		if not error_entries:
			return
		console_print(f"[WARN] {len(error_entries)} page(s) failed during batch processing:")
		for e in error_entries:
			img = (e.get("image_info", {}) or {}).get("image_name") or e.get("custom_id", "[unknown image]")
			page = (e.get("image_info", {}) or {}).get("page_number")
			det = (e.get("error_details", {}) or {})
			status = det.get("status_code")
			code = det.get("code")
			msg = det.get("message")
			label = UserPrompt._format_page_image(page, img)
			parts = []
			if status is not None:
				parts.append(f"status={status}")
			if code:
				parts.append(f"code={code}")
			if msg:
				parts.append(f"message={msg}")
			console_print("  - " + label + (": " + " ".join(parts) if parts else ""))

	@staticmethod
	def display_transcription_not_possible_summary(count: int) -> None:
		"""
		Display a short summary for pages where the model reported 'transcription not possible'.
		"""
		if count > 0:
			console_print(f"[INFO] {count} page(s) reported 'transcription not possible' by the model.")

	@staticmethod
	def display_batch_processing_progress(temp_file, batch_ids, completed_count,
	                                      missing_count):
		"""
		Display progress information for batch processing.

		Parameters:
			temp_file: Path to the temporary file
			batch_ids: Set of batch IDs
			completed_count: Number of completed batches
			missing_count: Number of missing batches
		"""
		console_print(f"\n----- Processing File: {temp_file.name} -----")
		console_print(f"Found {len(batch_ids)} batch ID(s)")

		if completed_count == len(batch_ids):
			console_print(f"[SUCCESS] All batches completed!")
		else:
			console_print(
				f"Completed: {completed_count} | Pending: {len(batch_ids) - completed_count - missing_count} | Missing: {missing_count}")
			if missing_count > 0:
				console_print(
					f"[WARN] {missing_count} batch ID(s) were not found in the API response")
			if completed_count < len(batch_ids) - missing_count:
				console_print(
					f"[INFO] Some batches are still processing. Try again later.")

	@staticmethod
	def display_batch_cancellation_results(cancelled_batches, skipped_batches):
		"""
		Display the results of batch cancellation attempts.

		Parameters:
			cancelled_batches: List of tuples (batch_id, status, success)
			skipped_batches: List of tuples (batch_id, status)
		"""
		success_count = sum(1 for _, _, success in cancelled_batches if success)
		fail_count = len(cancelled_batches) - success_count

		console_print(f"\n===== Cancellation Summary =====")
		console_print(
			f"Total batches found: {len(cancelled_batches) + len(skipped_batches)}")
		console_print(f"Skipped (terminal status): {len(skipped_batches)}")
		console_print(f"Attempted to cancel: {len(cancelled_batches)}")
		console_print(f"Successfully cancelled: {success_count}")

		if fail_count > 0:
			console_print(f"Failed to cancel: {fail_count}")
			console_print("\n----- Failed Cancellations -----")
			for batch_id, status, success in cancelled_batches:
				if not success:
					console_print(f"  Batch {batch_id} (status: '{status}')")

		if success_count > 0:
			console_print("\n----- Successfully Cancelled -----")
			for batch_id, status, success in cancelled_batches:
				if success:
					console_print(
						f"  Batch {batch_id} (previous status: '{status}')")