# unified_transcriber.py
import asyncio
import os
import sys
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict

import aiofiles
import pytesseract
from PIL import Image

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.utils import extract_page_number_from_filename
from modules.pdf_utils import PDFProcessor
from modules.openai_utils import transcribe_image_with_openai, open_transcriber
from modules.image_utils import process_images_multiprocessing, SUPPORTED_IMAGE_EXTENSIONS
from modules import batching
from modules.text_processing import extract_transcribed_text
from modules.concurrency import run_concurrent_transcription_tasks

logger = setup_logger(__name__)

def console_print(message: str) -> None:
    print(message)

def check_exit(user_input: str) -> None:
    if user_input.lower() in ["q", "exit"]:
        console_print("[INFO] Exiting as requested.")
        sys.exit(0)

def safe_input(prompt: str) -> str:
    try:
        return str(input(prompt)).strip()
    except Exception as e:
        logger.error(f"Error reading input: {e}")
        console_print("[ERROR] Unable to read input. Exiting.")
        sys.exit(1)

async def tesseract_ocr_image(img_path: Path, tesseract_config: str) -> Optional[str]:
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
    image_name = img_path.name
    try:
        if method == "gpt":
            if not transcriber:
                logger.error("No transcriber instance provided for GPT usage.")
                return (str(img_path), image_name, None)
            result = await transcribe_image_with_openai(img_path, transcriber)
            logger.debug(f"OpenAI response for {img_path.name}: {result}")
            final_text = extract_transcribed_text(result, image_name)
        elif method == "tesseract":
            final_text = await tesseract_ocr_image(img_path, tesseract_config)
        else:
            logger.error(f"Unknown transcription method '{method}' for image {img_path.name}")
            final_text = None
        return (str(img_path), image_name, final_text)
    except Exception as e:
        logger.exception(f"Error transcribing {img_path.name} with method '{method}': {e}")
        return (str(img_path), image_name, None)

def native_extract_pdf_text(pdf_path: Path) -> str:
    pdf_processor = PDFProcessor(pdf_path)
    text = ""
    try:
        pdf_processor.open_pdf()
        if pdf_processor.doc:
            for page in pdf_processor.doc:
                text += page.get_text()
        pdf_processor.close_pdf()
    except Exception as e:
        logger.exception(f"Failed native PDF extraction on {pdf_path.name}: {e}")
    return text

async def process_images_for_pdf(
        pdf_path: Path,
        raw_images_folder: Path,
        preprocessed_folder: Path,
        target_dpi: int
) -> List[Path]:
    pdf_processor = PDFProcessor(pdf_path)
    try:
        await pdf_processor.extract_images(raw_images_folder, dpi=target_dpi)
    except Exception as e:
        logger.exception(f"Error extracting images from PDF {pdf_path.name}: {e}")
        console_print(f"[ERROR] Failed to extract images from {pdf_path.name}.")
        return []
    image_files: List[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(list(raw_images_folder.glob(f"*{ext}")))
    if not image_files:
        console_print(f"[WARN] No images extracted from {pdf_path.name}.")
        return []
    processed_image_paths = [
        preprocessed_folder / f"{img.stem}_pre_processed{img.suffix}" for img in image_files
    ]
    process_images_multiprocessing(image_files, processed_image_paths)
    return [p for p in processed_image_paths if p.exists()]

async def process_single_pdf(
        pdf_path: Path,
        transcriber: Optional[Any],
        image_processing_config: Dict[str, Any],
        concurrency_config: Dict[str, Any],
        pdf_output_dir: Path,
        processing_settings: Dict[str, Any],
        model_config: Dict[str, Any],
        chosen_method: Optional[str] = None
) -> None:
    parent_folder = pdf_output_dir / pdf_path.stem
    parent_folder.mkdir(parents=True, exist_ok=True)
    output_txt_path = parent_folder / f"{pdf_path.stem}_transcription.txt"
    temp_jsonl_path = parent_folder / f"{pdf_path.stem}_transcription.jsonl"
    if not temp_jsonl_path.exists():
        temp_jsonl_path.touch()

    pdf_processor = PDFProcessor(pdf_path)
    is_native = pdf_processor.is_native_pdf()
    valid_methods: Dict[str, str] = {}
    methods: List[str] = []
    if is_native:
        valid_methods["1"] = "native"
        methods.append("1. Native text extraction")
        valid_methods["2"] = "tesseract"
        methods.append("2. Tesseract")
        valid_methods["3"] = "gpt"
        methods.append("3. GPT")
    else:
        valid_methods["1"] = "tesseract"
        methods.append("1. Tesseract")
        valid_methods["2"] = "gpt"
        methods.append("2. GPT")

    console_print(f"\n[INFO] Processing PDF: {pdf_path.name}")
    if chosen_method is None:
        if is_native:
            console_print("Choose how to extract/transcribe the PDF:")
            for m in methods:
                console_print(m)
        else:
            console_print("This PDF appears to be non-native (scanned).")
            console_print("Available transcription methods: 1. Tesseract, 2. GPT")
        choice = safe_input("Enter the method number (or type 'q' to exit): ")
        check_exit(choice)
    else:
        choice = str(chosen_method).strip()
        if choice not in valid_methods:
            console_print(f"[WARN] Invalid chosen method '{choice}' for PDF: {pdf_path.name}. Defaulting to the first available option.")
            choice = list(valid_methods.keys())[0]
    method = valid_methods.get(choice)

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
        return

    raw_images_folder = parent_folder / "raw_images"
    raw_images_folder.mkdir(exist_ok=True)
    preprocessed_folder = parent_folder / "preprocessed_images"
    preprocessed_folder.mkdir(exist_ok=True)
    keep_preprocessed_images = processing_settings.get("keep_preprocessed_images", True)
    target_dpi = image_processing_config.get('target_dpi', 300)
    processed_image_files = await process_images_for_pdf(pdf_path, raw_images_folder, preprocessed_folder, target_dpi)
    if not processed_image_files:
        return

    if method == "gpt":
        batch_choice = safe_input("Use batch processing for GPT transcription? (y/n): ").lower()
        if batch_choice == "y":
            try:
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
                return
            except Exception as e:
                logger.exception(f"Error during GPT batch submission for {pdf_path.name}: {e}")
                console_print(f"[ERROR] Failed to submit batch for {pdf_path.name}.")
                return

    args_list = [
        (img, transcriber if method == "gpt" else None, method,
         image_processing_config.get('ocr', {}).get('tesseract_config', "--oem 3 --psm 6"))
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
    async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
        for result in results:
            if result is None:
                continue
            if result[2] is not None:
                record = {
                    "file_name": pdf_path.name,
                    "pre_processed_image": result[0],
                    "image_name": result[1],
                    "timestamp": datetime.now().isoformat(),
                    "method": method,
                    "text_chunk": result[2]
                }
                await jfile.write(json.dumps(record) + "\n")
    try:
        combined_text = "\n".join([res[2] for res in results if res and res[2] is not None])
        output_txt_path.write_text(combined_text, encoding='utf-8')
    except Exception as e:
        logger.exception(f"Error writing combined text for {pdf_path.name}: {e}")
    if not keep_preprocessed_images:
        if preprocessed_folder.exists():
            try:
                shutil.rmtree(preprocessed_folder, ignore_errors=True)
            except Exception as e:
                logger.exception(f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")
    console_print(f"[SUCCESS] Saved transcription for PDF '{pdf_path.name}' -> {output_txt_path.name}")

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
    parent_folder = image_output_dir / folder.name
    parent_folder.mkdir(parents=True, exist_ok=True)
    raw_images_folder = parent_folder / "raw_images"
    raw_images_folder.mkdir(exist_ok=True)
    preprocessed_folder = parent_folder / "preprocessed_images"
    preprocessed_folder.mkdir(exist_ok=True)
    output_txt_path = parent_folder / f"{folder.name}_transcription.txt"
    temp_jsonl_path = parent_folder / f"{folder.name}_transcription.jsonl"
    if not temp_jsonl_path.exists():
        temp_jsonl_path.touch()
    console_print(f"\n[INFO] Processing image folder: {folder.name}")
    valid_methods = {"1": "gpt", "2": "tesseract"}
    if chosen_method is None:
        console_print("Choose image transcription method:")
        console_print("1. GPT")
        console_print("2. Tesseract")
        choice = safe_input("Enter the number of your choice (or type 'q' to exit): ")
        check_exit(choice)
    else:
        choice = str(chosen_method).strip()
        if choice not in valid_methods:
            console_print(f"[WARN] Invalid chosen method '{choice}' for folder '{folder.name}'. Defaulting to '1'.")
            choice = "1"
    method = valid_methods[choice]
    image_files: List[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(list(folder.glob(f'*{ext}')))
    if not image_files:
        console_print(f"[WARN] No images found in {folder}.")
        return
    for file in image_files:
        try:
            shutil.copy(file, raw_images_folder / file.name)
        except Exception as e:
            logger.exception(f"Error copying file {file} to raw_images_folder: {e}")
    image_files = list(raw_images_folder.glob("*"))
    processed_image_paths = [
        preprocessed_folder / f"{img.stem}_pre_processed{img.suffix}" for img in image_files
    ]
    process_images_multiprocessing(image_files, processed_image_paths)
    if not processing_settings.get("keep_raw_images", True):
        for img in image_files:
            if img.exists():
                try:
                    img.unlink()
                except Exception as e:
                    logger.exception(f"Error deleting raw image {img}: {e}")
    processed_files = [p for p in processed_image_paths if p.exists()]
    processed_files.sort(key=lambda x: extract_page_number_from_filename(x.name))
    if not processed_files:
        console_print(f"[WARN] No processed images found in {folder.name}.")
        return
    if method == "gpt":
        batch_choice = safe_input("Use batch processing for GPT transcription? (y/n): ").lower()
        if batch_choice == "y":
            try:
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
                return
            except Exception as e:
                logger.exception(f"Error during GPT batch submission for folder {folder.name}: {e}")
                console_print(f"[ERROR] Failed to submit batch for folder '{folder.name}'.")
                return
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
        logger.exception(f"Error running concurrent transcription tasks for folder {folder.name}: {e}")
        console_print(f"[ERROR] Concurrency error for folder {folder.name}.")
        return
    async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
        for result in results:
            if result is None:
                continue
            if result[2] is not None:
                record = {
                    "folder_name": folder.name,
                    "pre_processed_image": result[0],
                    "image_name": result[1],
                    "timestamp": datetime.now().isoformat(),
                    "method": method,
                    "text_chunk": result[2]
                }
                await jfile.write(json.dumps(record) + "\n")
    try:
        combined_text = "\n".join([res[2] for res in results if res and res[2] is not None])
        output_txt_path.write_text(combined_text, encoding='utf-8')
    except Exception as e:
        logger.exception(f"Error writing combined text for folder {folder.name}: {e}")
    if not processing_settings.get("keep_preprocessed_images", True):
        if preprocessed_folder.exists():
            try:
                shutil.rmtree(preprocessed_folder, ignore_errors=True)
            except Exception as e:
                logger.exception(f"Error cleaning up preprocessed images for folder {folder.name}: {e}")
    console_print(f"[SUCCESS] Transcription completed for folder '{folder.name}' -> {output_txt_path.name}")

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
    pdf_input_dir = Path(paths_config.get('file_paths', {}).get('PDFs', {}).get('input', 'pdfs_in'))
    image_input_dir = Path(paths_config.get('file_paths', {}).get('Images', {}).get('input', 'images_in'))
    pdf_output_dir = Path(paths_config.get('file_paths', {}).get('PDFs', {}).get('output', 'pdfs_out'))
    image_output_dir = Path(paths_config.get('file_paths', {}).get('Images', {}).get('output', 'images_out'))

    # Enforce absolute paths for input directories
    if not pdf_input_dir.is_absolute():
        console_print("[ERROR] PDF input path must be an absolute path. Please update your configuration.")
        sys.exit(1)
    if not image_input_dir.is_absolute():
        console_print("[ERROR] Image input path must be an absolute path. Please update your configuration.")
        sys.exit(1)

    for d in (pdf_input_dir, image_input_dir, pdf_output_dir, image_output_dir):
        d.mkdir(parents=True, exist_ok=True)
    model_config = config_loader.get_model_config()
    concurrency_config = config_loader.get_concurrency_config()
    image_processing_config = config_loader.get_image_processing_config()
    system_prompt_path = Path(paths_config.get('general', {}).get('transcription_prompt_path',
                                                                  'system_prompt/system_prompt.txt'))
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
            method_choice = safe_input("Choose transcription method for all image folders (1 for GPT, 2 for Tesseract): ")
            check_exit(method_choice)
            if method_choice == "1":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                    sys.exit(1)
                async with open_transcriber(
                        api_key=api_key,
                        system_prompt_path=system_prompt_path,
                        schema_path=schema_path,
                        model=model_config.get("transcription_model", {}).get("name", "gpt")
                ) as transcriber:
                    for folder in subfolders:
                        await process_single_image_folder(folder, transcriber,
                                                          image_processing_config,
                                                          concurrency_config,
                                                          image_output_dir,
                                                          processing_settings,
                                                          model_config,
                                                          chosen_method=method_choice)
            else:
                for folder in subfolders:
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
            method_choice = safe_input("Enter transcription method for all selected folders (Native, Tesseract, GPT) (or type 'q' to exit): ")
            check_exit(method_choice)
            mapping = {"native": "native", "tesseract": "tesseract", "gpt": "gpt"}
            method_choice_mapped = mapping.get(method_choice.lower())
            if not method_choice_mapped:
                console_print(f"[WARN] Invalid method choice '{method_choice}'. Defaulting to Native.")
                method_choice_mapped = "native"
            if method_choice_mapped == "gpt":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                    sys.exit(1)
                async with open_transcriber(
                        api_key=api_key,
                        system_prompt_path=system_prompt_path,
                        schema_path=schema_path,
                        model=model_config.get("transcription_model", {}).get("name", "gpt")
                ) as transcriber:
                    for folder in chosen_folders:
                        await process_single_image_folder(folder, transcriber,
                                                          image_processing_config,
                                                          concurrency_config,
                                                          image_output_dir,
                                                          processing_settings,
                                                          model_config,
                                                          chosen_method=method_choice_mapped)
            else:
                for folder in chosen_folders:
                    await process_single_image_folder(folder, None,
                                                      image_processing_config,
                                                      concurrency_config,
                                                      image_output_dir,
                                                      processing_settings,
                                                      model_config,
                                                      chosen_method=method_choice_mapped)
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
            method_choice = safe_input("Choose transcription method for all PDFs (For native PDFs: 1.Native, 2.Tesseract, 3.GPT): ")
            check_exit(method_choice)
            console_print(f"[INFO] Processing {len(all_pdfs)} PDF(s) with the selected transcription method...")
            if method_choice == "3":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                    sys.exit(1)
                async with open_transcriber(
                        api_key=api_key,
                        system_prompt_path=system_prompt_path,
                        schema_path=schema_path,
                        model=model_config.get("transcription_model", {}).get("name", "gpt")
                ) as transcriber:
                    for pdf_path in all_pdfs:
                        await process_single_pdf(pdf_path, transcriber,
                                                 image_processing_config,
                                                 concurrency_config, pdf_output_dir,
                                                 processing_settings, model_config,
                                                 chosen_method=method_choice)
            else:
                for pdf_path in all_pdfs:
                    await process_single_pdf(pdf_path, None,
                                             image_processing_config,
                                             concurrency_config, pdf_output_dir,
                                             processing_settings, model_config,
                                             chosen_method=method_choice)
        elif pdf_choice == "2":
            subfolders = [sf for sf in pdf_input_dir.iterdir() if sf.is_dir()]
            if not subfolders:
                console_print("[WARN] No subfolders found containing PDFs.")
                return
            console_print("\n[INFO] Subfolders available:")
            for idx, sf in enumerate(subfolders, 1):
                console_print(f"  {idx}. {sf.name}")
            method_choice = safe_input("Enter transcription method for all selected subfolders (Native, Tesseract, GPT) (or type 'q' to exit): ")
            check_exit(method_choice)
            selected_indices = safe_input("Enter subfolder numbers separated by commas (or type 'q' to exit): ")
            check_exit(selected_indices)
            try:
                indices = [int(n.strip()) - 1 for n in selected_indices.split(",") if n.strip().isdigit()]
                chosen_folders = [subfolders[i] for i in indices if 0 <= i < len(subfolders)]
            except ValueError:
                console_print("[ERROR] Invalid input. Exiting.")
                return
            mapping = {"native": "native", "tesseract": "tesseract", "gpt": "gpt"}
            method_choice_mapped = mapping.get(method_choice.lower())
            if not method_choice_mapped:
                console_print(f"[WARN] Invalid method choice '{method_choice}'. Defaulting to Native.")
                method_choice_mapped = "native"
            if method_choice_mapped == "gpt":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                    sys.exit(1)
                async with open_transcriber(
                        api_key=api_key,
                        system_prompt_path=system_prompt_path,
                        schema_path=schema_path,
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
                                                     concurrency_config, pdf_output_dir,
                                                     processing_settings, model_config,
                                                     chosen_method=method_choice_mapped)
            else:
                for folder in chosen_folders:
                    pdfs_in_folder = list(folder.rglob("*.pdf"))
                    if not pdfs_in_folder:
                        console_print(f"[WARN] No PDF files in {folder.name}.")
                        continue
                    for pdf_path in pdfs_in_folder:
                        await process_single_pdf(pdf_path, None,
                                                 image_processing_config,
                                                 concurrency_config, pdf_output_dir,
                                                 processing_settings, model_config,
                                                 chosen_method=method_choice_mapped)
        elif pdf_choice == "3":
            while True:
                filename = safe_input("\nEnter the filename (with or without .pdf) or type 'q' to exit: ")
                check_exit(filename)
                pdf_stem = filename.replace(".pdf", "").lower()
                match = next((pdf for pdf in all_pdfs if pdf.stem.lower() == pdf_stem), None)
                if match:
                    method_choice = safe_input("Choose transcription method for this PDF (1.Native, 2.Tesseract, 3.GPT): ")
                    check_exit(method_choice)
                    if method_choice == "3":
                        api_key = os.getenv('OPENAI_API_KEY')
                        if not api_key:
                            console_print("[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again.")
                            sys.exit(1)
                        async with open_transcriber(
                                api_key=api_key,
                                system_prompt_path=system_prompt_path,
                                schema_path=schema_path,
                                model=model_config.get("transcription_model", {}).get("name", "gpt")
                        ) as transcriber:
                            await process_single_pdf(match, transcriber,
                                                     image_processing_config,
                                                     concurrency_config, pdf_output_dir,
                                                     processing_settings, model_config,
                                                     chosen_method=method_choice)
                    else:
                        await process_single_pdf(match, None,
                                                 image_processing_config,
                                                 concurrency_config, pdf_output_dir,
                                                 processing_settings, model_config,
                                                 chosen_method=method_choice)
                else:
                    console_print(f"[WARN] No PDF found named '{filename}'.")
        else:
            console_print("[ERROR] Invalid choice for PDF processing.")
    else:
        console_print("[ERROR] Invalid overall choice. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())
