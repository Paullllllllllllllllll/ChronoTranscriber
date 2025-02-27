# unified_transcriber.py

"""
This script orchestrates the transcription of historical documents by:
- Loading configuration from YAML files.
- Prompting the user to select between processing PDFs or image folders.
- For PDFs: detecting native (searchable) vs. scanned files, then either extracting text directly or processing images using Tesseract OCR or GPT-based methods.
- For image folders: copying, preprocessing, and transcribing images.
- Optionally using batch processing for GPT transcription.
- Writing transcription results to output files, with the option to use the input folder as the output destination.
"""

import asyncio
import os
import sys
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import aiofiles
import pytesseract
from PIL import Image

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.pdf_utils import PDFProcessor, native_extract_pdf_text
from modules.openai_utils import transcribe_image_with_openai, open_transcriber
from modules.concurrency import run_concurrent_transcription_tasks
from modules.image_utils import ImageProcessor, SUPPORTED_IMAGE_EXTENSIONS
from modules.text_processing import extract_transcribed_text
from modules.utils import console_print, check_exit, safe_input
from modules.path_utils import validate_paths

logger = setup_logger(__name__)

# --------------------------------------------------
# OCR and Transcription Functions
# --------------------------------------------------

async def tesseract_ocr_image(img_path: Path, tesseract_config: str) -> Optional[str]:
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

async def transcribe_single_image_task(
        img_path: Path,
        transcriber: Optional[Any],
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
            if not transcriber:
                logger.error("No transcriber instance provided for GPT usage.")
                return (str(img_path), image_name, f"[transcription error: {image_name}]")
            result = await transcribe_image_with_openai(img_path, transcriber)
            logger.debug(f"OpenAI response for {img_path.name}: {result}")
            try:
                final_text = extract_transcribed_text(result, image_name)
            except Exception as e:
                logger.error(f"Error extracting transcription for {img_path.name}: {e}. Marking as transcription error.")
                final_text = f"[transcription error: {image_name}]"
        elif method == "tesseract":
            final_text = await tesseract_ocr_image(img_path, tesseract_config)
        else:
            logger.error(f"Unknown transcription method '{method}' for image {img_path.name}")
            final_text = None
        return (str(img_path), image_name, final_text)
    except Exception as e:
        logger.exception(f"Error transcribing {img_path.name} with method '{method}': {e}")
        return (str(img_path), image_name, f"[transcription error: {image_name}]")

# --------------------------------------------------
# PDF Processing Function
# --------------------------------------------------

async def process_single_pdf(pdf_path: Path,
                             transcriber: Optional[Any],
                             image_processing_config: Dict[str, Any],
                             concurrency_config: Dict[str, Any],
                             pdf_output_dir: Path,
                             processing_settings: Dict[str, Any],
                             model_config: Dict[str, Any],
                             chosen_method: Optional[str] = None) -> None:
    """
    Processes a single PDF file for transcription.
    1) Prepares an output folder structure and determines the method (native, Tesseract, GPT).
    2) If it's native, extract text directly. Otherwise, extract images, possibly use OCR or GPT transcription.
    3) Optionally handle batch processing for GPT requests.
    4) Manage raw/preprocessed image folders based on user settings.
    """
    pdf_processor = PDFProcessor(pdf_path)
    parent_folder, output_txt_path, temp_jsonl_path = pdf_processor.prepare_output_folder(pdf_output_dir)
    valid_methods, _ = pdf_processor.choose_transcription_method()

    console_print(f"\n[INFO] Processing PDF: {pdf_path.name}")

    # Determine or prompt for the chosen method
    if chosen_method is None:
        if pdf_processor.is_native_pdf():
            choice = select_option("Choose transcription method for this PDF:", ["Native", "Tesseract", "GPT"])
        else:
            choice = select_option("This PDF appears to be non-native (scanned).\nChoose transcription method for this PDF:", ["Tesseract", "GPT"])
        check_exit(choice)
    else:
        choice = str(chosen_method).strip()
        if choice not in valid_methods:
            console_print(f"[WARN] Invalid chosen method '{choice}' for PDF: {pdf_path.name}. Defaulting to the first available option.")
            choice = list(valid_methods.keys())[0]
    method = valid_methods[choice]

    # Native PDF extraction
    if method == "native":
        text = native_extract_pdf_text(pdf_path)
        try:
            async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
                record = {
                    "file_name": pdf_path.name,
                    "timestamp": datetime.now().isoformat(),
                    "method": "native",
                    "text_chunk": text,
                    "pre_processed_image": None
                }
                await jfile.write(json.dumps(record) + '\n')
            output_txt_path.write_text(text, encoding='utf-8')
            console_print(f"[SUCCESS] Extracted text from '{pdf_path.name}' using native method -> {output_txt_path.name}")
        except Exception as e:
            logger.exception(f"Error writing native extraction output for {pdf_path.name}: {e}")

        # Cleanup if not retaining
        if not processing_settings.get("retain_temporary_jsonl", True):
            try:
                temp_jsonl_path.unlink()
                console_print(f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
            except Exception as e:
                logger.exception(f"Error deleting temporary file {temp_jsonl_path}: {e}")
                console_print(f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")

        return

    # Non-native PDF: Extract images
    raw_images_folder = parent_folder / "raw_images"
    raw_images_folder.mkdir(exist_ok=True)
    preprocessed_folder = parent_folder / "preprocessed_images"
    preprocessed_folder.mkdir(exist_ok=True)

    target_dpi = image_processing_config.get('target_dpi', 300)
    processed_image_files = await pdf_processor.process_images(raw_images_folder, preprocessed_folder, target_dpi)

    # Ensure proper page ordering
    from modules.utils import extract_page_number_from_filename
    processed_image_files.sort(key=lambda p: extract_page_number_from_filename(p.name))

    # Delete raw images folder immediately if the flag is false
    if not processing_settings.get("keep_raw_images", True):
        if raw_images_folder.exists():
            try:
                shutil.rmtree(raw_images_folder, ignore_errors=True)
                console_print(f"[CLEANUP] Deleted raw_images folder: {raw_images_folder.name}")
            except Exception as e:
                logger.exception(f"Error deleting raw_images folder {raw_images_folder}: {e}")

    # Handle GPT batch or synchronous modes
    if method == "gpt":
        batch_choice = safe_input("Use batch processing for GPT transcription? (y/n): ").lower()
        if batch_choice == "y":
            try:
                from modules import batching
                batch_responses = await asyncio.to_thread(
                    batching.process_batch_transcription,
                    processed_image_files,
                    "",
                    model_config.get("transcription_model", {})
                )
                async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
                    for response in batch_responses:
                        tracking_record = {
                            "batch_tracking": {
                                "batch_id": response.id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "batch_file": str(response.id)
                            }
                        }
                        await f.write(json.dumps(tracking_record) + "\n")

                console_print(f"[SUCCESS] Batch submitted for PDF '{pdf_path.name}'.")
                if not processing_settings.get("keep_preprocessed_images", True):
                    if preprocessed_folder.exists():
                        try:
                            shutil.rmtree(preprocessed_folder, ignore_errors=True)
                        except Exception as e:
                            logger.exception(f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")

                return
            except Exception as e:
                logger.exception(f"Error during GPT batch submission for {pdf_path.name}: {e}")
                console_print(f"[ERROR] Failed to submit batch for {pdf_path.name}.")
                return

    # Non-batch flow for GPT or Tesseract
    if method == "gpt":
        args_list = [
            (img, transcriber, method, image_processing_config.get('ocr', {}).get('tesseract_config', "--oem 3 --psm 6"))
            for img in processed_image_files
        ]
    else:
        args_list = [
            (img, None, method, image_processing_config.get('ocr', {}).get('tesseract_config', "--oem 3 --psm 6"))
            for img in processed_image_files
        ]

    transcription_conf = concurrency_config.get("transcription", {})
    concurrency_limit = transcription_conf.get("concurrency_limit", 20)
    delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

    try:
        results = await run_concurrent_transcription_tasks(
            transcribe_single_image_task, args_list, concurrency_limit, delay_between_tasks
        )
    except Exception as e:
        logger.exception(f"Error running concurrent transcription tasks for {pdf_path.name}: {e}")
        console_print(f"[ERROR] Concurrency error for {pdf_path.name}.")
        return

    # Write the transcription results to a temporary JSONL
    async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
        for result in results:
            if result is None or result[2] is None:
                continue
            record = {
                "file_name": pdf_path.name,
                "pre_processed_image": result[0],
                "image_name": result[1],
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "text_chunk": result[2]
            }
            await jfile.write(json.dumps(record) + "\n")

    # Combine the transcription text in page order
    try:
        combined_text = "\n".join([res[2] for res in results if res and res[2] is not None])
        output_txt_path.write_text(combined_text, encoding='utf-8')
    except Exception as e:
        logger.exception(f"Error writing combined text for {pdf_path.name}: {e}")

    # If keep_preprocessed_images is false, delete preprocessed folder
    if not processing_settings.get("keep_preprocessed_images", True):
        if preprocessed_folder.exists():
            try:
                shutil.rmtree(preprocessed_folder, ignore_errors=True)
            except Exception as e:
                logger.exception(f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")

    console_print(f"[SUCCESS] Saved transcription for PDF '{pdf_path.name}' -> {output_txt_path.name}")

    # Remove temporary JSONL if not retaining
    if not processing_settings.get("retain_temporary_jsonl", True):
        try:
            temp_jsonl_path.unlink()
            console_print(f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
        except Exception as e:
            logger.exception(f"Error deleting temporary file {temp_jsonl_path}: {e}")
            console_print(f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")

# --------------------------------------------------
# Image Folder Processing Function
# --------------------------------------------------

async def process_single_image_folder(
        folder: Path,
        transcriber: Optional[Any],
        image_processing_config: Dict[str, Any],
        concurrency_config: Dict[str, Any],
        image_output_dir: Path,
        processing_settings: Dict[str, Any],
        model_config: Dict[str, Any],
        chosen_method: Optional[str] = None
) -> None:
    """
    Processes all images in a given folder:
    1) Copies images to a raw folder.
    2) Preprocesses them.
    3) Transcribes using GPT or Tesseract.
    4) Manages folder cleanup if flags are set.
    """
    parent_folder, raw_images_folder, preprocessed_folder, temp_jsonl_path, output_txt_path = ImageProcessor.prepare_image_folder(folder, image_output_dir)

    # Copy images
    copied_images = ImageProcessor.copy_images_to_raw(folder, raw_images_folder)
    if not copied_images:
        console_print(f"[WARN] No images found in {folder}.")
        return

    # Preprocess images
    processed_image_paths = [preprocessed_folder / f"{img.stem}_pre_processed{img.suffix}" for img in copied_images]
    ImageProcessor.process_images_multiprocessing(copied_images, processed_image_paths)

    # Delete raw_images_folder immediately if the flag is false
    if not processing_settings.get("keep_raw_images", True):
        if raw_images_folder.exists():
            try:
                shutil.rmtree(raw_images_folder, ignore_errors=True)
                console_print(f"[CLEANUP] Deleted raw_images folder: {raw_images_folder.name}")
            except Exception as e:
                logger.exception(f"Error deleting raw_images folder {raw_images_folder}: {e}")

    processed_files = [p for p in processed_image_paths if p.exists()]
    processed_files.sort(key=lambda x: int(__import__('modules.utils').utils.extract_page_number_from_filename(x.name)))
    if not processed_files:
        console_print(f"[WARN] No processed images found in folder '{folder.name}'.")
        return

    # Determine or prompt for GPT/Tesseract
    if chosen_method is None:
        choice = select_option("Choose image transcription method:", ["GPT", "Tesseract"])
    else:
        choice = str(chosen_method).strip().lower()
    mapping = {"1": "gpt", "2": "tesseract", "gpt": "gpt", "tesseract": "tesseract"}
    method = mapping.get(choice, "gpt")

    # Optionally handle batch mode
    if method == "gpt":
        batch_choice = safe_input("Use batch processing for GPT transcription? (y/n): ").lower()
        if batch_choice == "y":
            try:
                from modules import batching
                batch_responses = await asyncio.to_thread(
                    batching.process_batch_transcription,
                    processed_files,
                    "",
                    model_config.get("transcription_model", {})
                )
                async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
                    for response in batch_responses:
                        tracking_record = {
                            "batch_tracking": {
                                "batch_id": response.id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "batch_file": str(response.id)
                            }
                        }
                        await f.write(json.dumps(tracking_record) + "\n")
                console_print(f"[SUCCESS] Batch submitted for folder '{folder.name}'.")

                # Even for batch usage, delete preprocessed folder if the user doesn't want to keep it
                if not processing_settings.get("keep_preprocessed_images", True):
                    if preprocessed_folder.exists():
                        try:
                            shutil.rmtree(preprocessed_folder, ignore_errors=True)
                        except Exception as e:
                            logger.exception(f"Error cleaning up preprocessed images for folder '{folder.name}': {e}")

                return
            except Exception as e:
                logger.exception(f"Error during GPT batch submission for folder '{folder.name}': {e}")
                console_print(f"[ERROR] Failed to submit batch for folder '{folder.name}'.")
                return

    # Non-batch GPT or Tesseract
    args_list = [
        (img_path, transcriber if method == "gpt" else None, method,
         image_processing_config.get('ocr', {}).get('tesseract_config', "--oem 3 --psm 6"))
        for img_path in processed_files
    ]
    transcription_conf = concurrency_config.get("transcription", {})
    concurrency_limit = transcription_conf.get("concurrency_limit", 20)
    delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

    try:
        results = await run_concurrent_transcription_tasks(
            transcribe_single_image_task, args_list, concurrency_limit, delay_between_tasks
        )
    except Exception as e:
        logger.exception(f"Error running concurrent transcription tasks for folder '{folder.name}': {e}")
        console_print(f"[ERROR] Concurrency error for folder '{folder.name}'.")
        return

    # Write to temporary JSONL
    async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
        for result in results:
            if result is None or result[2] is None:
                continue
            record = {
                "folder_name": folder.name,
                "pre_processed_image": result[0],
                "image_name": result[1],
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "text_chunk": result[2]
            }
            await jfile.write(json.dumps(record) + "\n")

    # Combine final text
    try:
        combined_text = "\n".join([res[2] for res in results if res and res[2] is not None])
        output_txt_path.write_text(combined_text, encoding='utf-8')
    except Exception as e:
        logger.exception(f"Error writing combined text for folder '{folder.name}': {e}")

    # Delete preprocessed folder if not retaining
    if not processing_settings.get("keep_preprocessed_images", True):
        if preprocessed_folder.exists():
            try:
                shutil.rmtree(preprocessed_folder, ignore_errors=True)
            except Exception as e:
                logger.exception(f"Error cleaning up preprocessed images for folder '{folder.name}': {e}")

    console_print(f"[SUCCESS] Transcription completed for folder '{folder.name}' -> {output_txt_path.name}")

    # Delete temporary JSONL if not retaining
    if not processing_settings.get("retain_temporary_jsonl", True):
        try:
            temp_jsonl_path.unlink()
            console_print(f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
        except Exception as e:
            logger.exception(f"Error deleting temporary file {temp_jsonl_path}: {e}")
            console_print(f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")

# --------------------------------------------------
# User Interface Helper
# --------------------------------------------------

def select_option(prompt: str, options: list) -> str:
    """
    Prompt the user for a choice among a list of options.
    """
    console_print(prompt)
    for idx, option in enumerate(options, 1):
        console_print(f"{idx}. {option}")
    choice = safe_input("Enter the number of your choice (or type 'q' to exit): ")
    check_exit(choice)
    return choice.strip()

# --------------------------------------------------
# Main Function
# --------------------------------------------------

async def main() -> None:
    config_loader = ConfigLoader()
    try:
        config_loader.load_configs()
    except Exception as e:
        logger.critical(f"Failed to load configurations: {e}")
        console_print(f"[CRITICAL] Failed to load configurations: {e}")
        sys.exit(1)

    paths_config = config_loader.get_paths_config()
    processing_settings = paths_config.get("general", {})

    validate_paths(paths_config)

    pdf_input_dir = Path(paths_config.get('file_paths', {}).get('PDFs', {}).get('input', 'pdfs_in'))
    image_input_dir = Path(paths_config.get('file_paths', {}).get('Images', {}).get('input', 'images_in'))
    pdf_output_dir = Path(paths_config.get('file_paths', {}).get('PDFs', {}).get('output', 'pdfs_out'))
    image_output_dir = Path(paths_config.get('file_paths', {}).get('Images', {}).get('output', 'images_out'))

    if processing_settings.get("input_paths_is_output_path", False):
        pdf_output_dir = pdf_input_dir
        image_output_dir = image_input_dir

    for d in (pdf_input_dir, image_input_dir, pdf_output_dir, image_output_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_config = config_loader.get_model_config()
    concurrency_config = config_loader.get_concurrency_config()
    image_processing_config = config_loader.get_image_processing_config()

    system_prompt_path = Path(paths_config.get('general', {}).get('transcription_prompt_path', 'system_prompt/system_prompt.txt'))
    schema_path = Path("schemas/transcription_schema.json")

    console_print("\n[INFO] ** Unified Processor **")
    overall_choice = safe_input("Enter 1 for Images, 2 for PDFs (or type 'q' to exit): ")
    check_exit(overall_choice)

    if overall_choice == "1":
        if not image_input_dir.exists():
            console_print(f"[ERROR] Image input directory does not exist: {image_input_dir}")
            return
        subfolders = [f for f in image_input_dir.iterdir() if f.is_dir()]
        if not subfolders:
            images = [f for f in image_input_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
            if images:
                console_print(f"[WARN] No subfolders found in {image_input_dir}. Processing images directly in the input folder.")
                subfolders = [image_input_dir]
            else:
                console_print(f"[WARN] No subfolders or images found in {image_input_dir}.")
                return
        console_print(f"\n[INFO] Found {len(subfolders)} folder(s) to process in '{image_input_dir}'.")
        for idx, sf in enumerate(subfolders, 1):
            console_print(f"  {idx}. {sf.name}")
        all_or_select = safe_input("\nProcess 1) All folders or 2) Specific folders? (or type 'q' to exit): ")
        check_exit(all_or_select)
        if all_or_select == "1":
            method_choice = select_option("Choose transcription method for all image folders:", ["GPT", "Tesseract"])
            mapping = {"1": "gpt", "2": "tesseract"}
            method_choice = mapping.get(method_choice, "gpt")
            for folder in subfolders:
                console_print(f"Processing folder: {folder.name}")
                if method_choice == "gpt":
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        console_print(
                            "[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                        sys.exit(1)
                    async with open_transcriber(
                            api_key=api_key,
                            model=model_config.get("transcription_model",
                                                   {}).get("name", "gpt")
                    ) as transcriber:
                        await process_single_image_folder(folder, transcriber,
                                                          image_processing_config,
                                                          concurrency_config,
                                                          image_output_dir,
                                                          processing_settings,
                                                          model_config,
                                                          chosen_method=method_choice)
                else:
                    await process_single_image_folder(folder, None,
                                                      image_processing_config,
                                                      concurrency_config,
                                                      image_output_dir,
                                                      processing_settings,
                                                      model_config,
                                                      chosen_method=method_choice)
        elif all_or_select == "2":
            selected = safe_input("Enter folder numbers separated by commas (or type 'q' to exit): ")
            check_exit(selected)
            try:
                indices = [int(n.strip()) - 1 for n in selected.split(",") if n.strip().isdigit()]
                chosen_folders = [subfolders[i] for i in indices if 0 <= i < len(subfolders)]
            except ValueError:
                console_print("[ERROR] Invalid input. Aborting.")
                return
            method_choice = select_option("Choose transcription method for all selected folders:", ["GPT", "Tesseract"])
            for folder in chosen_folders:
                console_print(f"Processing folder: {folder.name}")
                if method_choice == "1":
                    api_key = os.getenv('OPENAI_API_KEY')
                    if not api_key:
                        console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                        sys.exit(1)
                    async with open_transcriber(
                            api_key=api_key,
                            model=model_config.get("transcription_model", {}).get("name", "gpt")
                    ) as transcriber:
                        await process_single_image_folder(folder, transcriber,
                                                          image_processing_config,
                                                          concurrency_config,
                                                          image_output_dir,
                                                          processing_settings,
                                                          model_config,
                                                          chosen_method=method_choice)
                else:
                    await process_single_image_folder(folder, None,
                                                      image_processing_config,
                                                      concurrency_config,
                                                      image_output_dir,
                                                      processing_settings,
                                                      model_config,
                                                      chosen_method=method_choice)
        else:
            console_print("[ERROR] Invalid choice. Aborting.")
    elif overall_choice == "2":
        if not pdf_input_dir.exists():
            console_print(f"[ERROR] PDF input directory does not exist: {pdf_input_dir}")
            return
        all_pdfs = list(pdf_input_dir.rglob("*.pdf"))
        if not all_pdfs:
            console_print(f"[WARN] No PDFs found in {pdf_input_dir}.")
            return
        console_print("\n[INFO] How do you want to process PDFs?")
        console_print("  1. All PDFs in the folder (and subfolders)")
        console_print("  2. Selected subfolders containing PDFs")
        console_print("  3. Single PDF by file name")
        pdf_choice = safe_input("Enter your choice (1/2/3 or type 'q' to exit): ")
        check_exit(pdf_choice)
        if pdf_choice == "1":
            method_choice = select_option("Choose transcription method for all PDFs:", ["Native", "Tesseract", "GPT"])
            console_print(f"[INFO] Processing {len(all_pdfs)} PDF(s) with the selected transcription method...")
            if method_choice == "3":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                    sys.exit(1)
                async with open_transcriber(
                        api_key=api_key,
                        model=model_config.get("transcription_model", {}).get("name", "gpt")
                ) as transcriber:
                    for pdf_path in all_pdfs:
                        await process_single_pdf(pdf_path, transcriber,
                                                 image_processing_config,
                                                 concurrency_config,
                                                 pdf_output_dir,
                                                 processing_settings,
                                                 model_config,
                                                 chosen_method=method_choice)
            else:
                for pdf_path in all_pdfs:
                    await process_single_pdf(pdf_path, None,
                                             image_processing_config,
                                             concurrency_config,
                                             pdf_output_dir,
                                             processing_settings,
                                             model_config,
                                             chosen_method=method_choice)
        elif pdf_choice == "2":
            subfolders = [sf for sf in pdf_input_dir.iterdir() if sf.is_dir()]
            if not subfolders:
                console_print("[WARN] No subfolders found containing PDFs.")
                return
            console_print("\n[INFO] Subfolders available:")
            for idx, sf in enumerate(subfolders, 1):
                console_print(f"  {idx}. {sf.name}")
            method_choice = select_option("Choose transcription method for all selected subfolders:", ["Native", "Tesseract", "GPT"])
            selected_indices = safe_input("Enter subfolder numbers separated by commas (or type 'q' to exit): ")
            check_exit(selected_indices)
            try:
                indices = [int(n.strip()) - 1 for n in selected_indices.split(",") if n.strip().isdigit()]
                chosen_folders = [subfolders[i] for i in indices if 0 <= i < len(subfolders)]
            except ValueError:
                console_print("[ERROR] Invalid input. Exiting.")
                return
            if method_choice == "3":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                    sys.exit(1)
                async with open_transcriber(
                        api_key=api_key,
                        model=model_config.get("transcription_model", {}).get("name", "gpt")
                ) as transcriber:
                    for folder in chosen_folders:
                        pdfs_in_folder = list(folder.rglob("*.pdf"))
                        if not pdfs_in_folder:
                            console_print(f"[WARN] No PDF files in {folder.name}.")
                            continue
                        for pdf_path in pdfs_in_folder:
                            await process_single_pdf(pdf_path, transcriber,
                                                 image_processing_config,
                                                 concurrency_config,
                                                 pdf_output_dir,
                                                 processing_settings,
                                                 model_config,
                                                 chosen_method=method_choice)
            else:
                for folder in chosen_folders:
                    pdfs_in_folder = list(folder.rglob("*.pdf"))
                    if not pdfs_in_folder:
                        console_print(f"[WARN] No PDF files in {folder.name}.")
                        continue
                    for pdf_path in pdfs_in_folder:
                        await process_single_pdf(pdf_path, None,
                                                 image_processing_config,
                                                 concurrency_config,
                                                 pdf_output_dir,
                                                 processing_settings,
                                                 model_config,
                                                 chosen_method=method_choice)
        elif pdf_choice == "3":
            while True:
                filename = safe_input("\nEnter the filename (with or without .pdf) or type 'q' to exit: ")
                check_exit(filename)
                pdf_stem = filename.replace(".pdf", "").lower()
                match = next((pdf for pdf in all_pdfs if pdf.stem.lower() == pdf_stem), None)
                if match:
                    method_choice = select_option("Choose transcription method for this PDF:", ["Native", "Tesseract", "GPT"])
                    check_exit(method_choice)
                    if method_choice == "3":
                        api_key = os.getenv('OPENAI_API_KEY')
                        if not api_key:
                            console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                            sys.exit(1)
                        async with open_transcriber(
                                api_key=api_key,
                                model=model_config.get("transcription_model", {}).get("name", "gpt")
                        ) as transcriber:
                            await process_single_pdf(match, transcriber,
                                                 image_processing_config,
                                                 concurrency_config,
                                                 pdf_output_dir,
                                                 processing_settings,
                                                 model_config,
                                                 chosen_method=method_choice)
                    else:
                        await process_single_pdf(match, None,
                                                 image_processing_config,
                                                 concurrency_config,
                                                 pdf_output_dir,
                                                 processing_settings,
                                                 model_config,
                                                 chosen_method=method_choice)
                else:
                    console_print(f"[WARN] No PDF found named '{filename}'.")
        else:
            console_print("[ERROR] Invalid choice for PDF processing.")
    else:
        console_print("[ERROR] Invalid overall choice. Exiting.")

    console_print("\n[INFO] Processing complete.")

if __name__ == "__main__":
    asyncio.run(main())
