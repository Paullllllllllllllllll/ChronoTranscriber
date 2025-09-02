# unified_transcriber.py

"""
This script orchestrates the transcription of historical documents by:
- Loading configuration from YAML files.
- Guiding users through a structured workflow:
  1. Choose between PDF or image processing
  2. Select transcription method (native, Tesseract OCR, or GPT-based)
  3. Configure batch processing options if applicable
  4. Select specific files or folders to process
- Processing documents based on user selections, with appropriate logging
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path

# Add parent directory to path to help with module imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
	sys.path.insert(0, parent_dir)

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.openai_utils import open_transcriber
from modules.user_interface import UserPrompt, UserConfiguration
from modules.path_utils import validate_paths
from modules.utils import console_print
from modules.workflow import WorkflowManager

logger = setup_logger(__name__)


async def select_files_for_processing(user_config: UserConfiguration,
                                      pdf_input_dir: Path,
                                      image_input_dir: Path) -> None:
	"""
	Select files or folders for processing based on user configuration.
	"""
	console_print("\n" + "=" * 80)
	console_print(
		f"  DOCUMENT SELECTION - {user_config.processing_type.upper()}")
	console_print("=" * 80)

	# Display a summary of selections so far
	console_print("\nSelected processing options:")
	console_print(
		f"  - Processing type: {user_config.processing_type.capitalize()}")
	console_print(
		f"  - Transcription method: {user_config.transcription_method.capitalize()}")
	if user_config.transcription_method == "gpt":
		batch_mode = "Batch (asynchronous)" if user_config.use_batch_processing else "Synchronous"
		console_print(f"  - Processing mode: {batch_mode}")
	console_print("-" * 80)

	# Handle file selection based on processing type
	if user_config.processing_type == "images":
		UserPrompt.select_images_workflow(user_config, image_input_dir)
	else:  # PDFs
		UserPrompt.select_pdfs_workflow(user_config, pdf_input_dir)


async def display_summary_and_confirm(user_config: UserConfiguration) -> bool:
	"""
	Display summary of processing and confirm with user.
	"""
	return UserPrompt.display_processing_summary(user_config)


async def display_final_summary(user_config: UserConfiguration) -> None:
	"""
	Display final processing summary.
	"""
	console_print("\n" + "=" * 80)
	console_print("  PROCESSING COMPLETE")
	console_print("=" * 80)

	if user_config.use_batch_processing and user_config.transcription_method == "gpt":
		console_print("\n[INFO] Batch processing jobs have been submitted.")
		console_print(
			"[INFO] To check the status of your batches, run: python main/check_batches.py")
		console_print(
			"[INFO] To cancel any pending batches, run: python main/cancel_batches.py")
	else:
		console_print("\n[INFO] All selected items have been processed.")

	console_print("\n[INFO] Thank you for using ChronoTranscriber!")


async def get_user_configuration(pdf_input_dir: Path,
                                 image_input_dir: Path) -> UserConfiguration:
	"""
	Guide user through configuration options and return UserConfiguration object.
	"""
	# Create a new UserConfiguration
	user_config = UserConfiguration()

	# Display welcome banner
	UserPrompt.display_banner()

	# Step 1: Choose processing type
	UserPrompt.configure_processing_type(user_config)

	# Step 2: Choose transcription method
	UserPrompt.configure_transcription_method(user_config)

	# Step 3: Configure batch processing if needed
	UserPrompt.configure_batch_processing(user_config)

	# Step 4: Select files/folders
	await select_files_for_processing(user_config, pdf_input_dir,
	                                  image_input_dir)

	return user_config


async def main():
	"""
	Main function implementing the improved workflow.
	"""
	# Load configurations
	config_loader = ConfigLoader()
	try:
		config_loader.load_configs()
	except Exception as e:
		logger.critical(f"Failed to load configurations: {e}")
		console_print(f"[CRITICAL] Failed to load configurations: {e}")
		sys.exit(1)

	# Get configuration values
	paths_config = config_loader.get_paths_config()
	processing_settings = paths_config.get("general", {})

	# Validate paths
	try:
		validate_paths(paths_config)
	except Exception as e:
		logger.error(f"Path validation failed: {e}")
		console_print(f"[ERROR] Path validation failed: {e}")
		sys.exit(1)

	# Get directory paths
	pdf_input_dir = Path(
		paths_config.get('file_paths', {}).get('PDFs', {}).get('input',
		                                                       'pdfs_in'))
	image_input_dir = Path(
		paths_config.get('file_paths', {}).get('Images', {}).get('input',
		                                                         'images_in'))

	# Ensure directories exist
	pdf_input_dir.mkdir(parents=True, exist_ok=True)
	image_input_dir.mkdir(parents=True, exist_ok=True)

	# Additional configs
	model_config = config_loader.get_model_config()
	concurrency_config = config_loader.get_concurrency_config()
	image_processing_config = config_loader.get_image_processing_config()

	# Get user configuration
	user_config = await get_user_configuration(pdf_input_dir, image_input_dir)

	# Display summary and confirm
	if not await display_summary_and_confirm(user_config):
		console_print("[INFO] Processing cancelled by user.")
		return

	# Start processing
	console_print("\n" + "=" * 80)
	console_print("  STARTING PROCESSING")
	console_print("=" * 80)

	# Process the files
	workflow_manager = WorkflowManager(
		user_config,
		paths_config,
		model_config,
		concurrency_config,
		image_processing_config
	)

	# Initialize OpenAI transcriber if needed
	transcriber = None
	if user_config.transcription_method == "gpt" and not user_config.use_batch_processing:
		api_key = os.getenv('OPENAI_API_KEY')
		if not api_key:
			console_print(
				"[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
			sys.exit(1)

		async with open_transcriber(
				api_key=api_key,
				model=model_config.get("transcription_model", {}).get("name",
				                                                      "gpt-4o")
		) as t:
			transcriber = t
			await workflow_manager.process_selected_items(transcriber)
	else:
		# For non-GPT methods or batch processing, no transcriber needed
		await workflow_manager.process_selected_items()

	# Display final summary
	await display_final_summary(user_config)


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		console_print("\n[INFO] Processing interrupted by user.")
		sys.exit(0)
	except Exception as e:
		logger.exception(f"Unexpected error: {e}")
		console_print(f"\n[ERROR] An unexpected error occurred: {e}")
		console_print(f"[INFO] Check the logs for more details.")
		console_print(f"Traceback: {traceback.format_exc()}")
		sys.exit(1)