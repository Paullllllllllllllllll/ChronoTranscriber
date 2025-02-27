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
from modules.utils import extract_page_number_from_filename

logger = setup_logger(__name__)

def console_print(message: str) -> None:
    print(message)

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

def process_all_batches(root_folder: Path, processing_settings: Dict[str, Any], client: OpenAI) -> None:
    """
    Scans the root folder for *_transcription.jsonl files, locates batch IDs
    within those files, checks if ALL batches for a file are completed, and if so,
    downloads the results and writes them to a final text file while preserving
    the original image order.
    """
    console_print(f"\n[INFO] Scanning directory '{root_folder}' for temporary batch files...")
    temp_files = list(root_folder.rglob("*_transcription.jsonl"))
    if not temp_files:
        console_print(f"[INFO] No temporary batch files found in {root_folder}.")
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

    if not batches:
        console_print("[INFO] No batch jobs found online.")
    else:
        console_print(f"[INFO] Found {len(batches)} batch(es) online:")
        for batch in batches:
            info = f"  Batch ID: {batch.id} | Status: {batch.status}"
            console_print(info)
            logger.info(info)

    # Process each temporary file
    for temp_file in temp_files:
        console_print(f"\n[INFO] Checking batch status for file: {temp_file.name}")

        # Extract all batch IDs from this temporary file and map batches to their image file data
        batch_ids: Set[str] = set()
        batch_to_images: Dict[str, List[Dict[str, Any]]] = {}
        image_records: List[Dict[str, Any]] = []

        # First pass: collect all image records and batch IDs
        with temp_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "batch_tracking" in record:
                        batch_id = record["batch_tracking"].get("batch_id")
                        if batch_id:
                            batch_ids.add(batch_id)
                            if batch_id not in batch_to_images:
                                batch_to_images[batch_id] = []
                    elif "image_name" in record or "pre_processed_image" in record:
                        # This is a record with image data - store it to establish order
                        image_records.append(record)
                except json.JSONDecodeError:
                    continue

        if not batch_ids:
            console_print(f"[WARN] No batch IDs found in {temp_file.name}. Skipping this file.")
            continue

        console_print(f"[INFO] Found {len(batch_ids)} batch IDs in {temp_file.name}")

        # Check if all batches are completed
        all_completed = True
        missing_batches = []
        for batch_id in batch_ids:
            if batch_id not in batch_dict:
                all_completed = False
                missing_batches.append(batch_id)
                logger.warning(f"Batch ID {batch_id} not found in OpenAI batches.")
                continue

            batch = batch_dict[batch_id]
            if batch.status.lower() != "completed":
                all_completed = False
                logger.info(f"Batch {batch_id} has status '{batch.status}' - not yet completed.")

        if missing_batches:
            console_print(f"[WARN] {len(missing_batches)} batch IDs from {temp_file.name} not found in OpenAI batches.")

        if not all_completed:
            console_print(f"[INFO] Not all batches for {temp_file.name} are completed yet. Skipping this file.")
            continue

        # All batches are completed, now download and process them
        console_print(f"[SUCCESS] All batches for {temp_file.name} are completed. Processing results...")

        # Sort image records based on page numbers
        image_records.sort(key=lambda r: extract_page_number_from_filename(r.get("image_name", "")))

        # Collect all transcriptions from all batches
        all_batch_transcriptions: Dict[str, List[str]] = {}
        for batch_id in batch_ids:
            batch = batch_dict[batch_id]
            try:
                file_obj = client.files.content(batch.output_file_id)
                file_content = file_obj.content
                transcriptions = process_batch_output(file_content)
                if transcriptions:
                    all_batch_transcriptions[batch_id] = transcriptions
                else:
                    logger.warning(f"No transcriptions extracted for batch {batch_id}.")
            except Exception as e:
                logger.exception(f"Error downloading batch {batch_id}: {e}")
                console_print(f"[ERROR] Failed to download output for Batch ID {batch_id}: {e}")
                # If any batch fails to download, skip writing the output
                all_completed = False

        if not all_completed:
            console_print(f"[WARN] Failed to process all batches for {temp_file.name}. Skipping output writing.")
            continue

        if not all_batch_transcriptions:
            logger.warning(f"No transcriptions extracted for any batch in {temp_file.name}.")
            console_print(f"[WARN] No transcriptions extracted for any batch in {temp_file.name}. Skipping this file.")
            continue

        # Build the final text file path
        identifier = temp_file.stem.replace("_transcription", "")
        final_txt_path = temp_file.parent / f"{identifier}_transcription.txt"

        ordered_transcriptions: List[str] = []

        for batch_id, transcriptions in all_batch_transcriptions.items():
            ordered_transcriptions.extend(transcriptions)

        processing_success = False
        try:
            with final_txt_path.open("w", encoding="utf-8") as fout:
                for text in ordered_transcriptions:
                    fout.write(text + "\n")
            logger.info(f"All batches for {temp_file.name} processed and saved to {final_txt_path}")
            console_print(f"[SUCCESS] Processed all batches for {temp_file.name}. Results saved to {final_txt_path.name}")
            processing_success = True
        except Exception as e:
            logger.exception(f"Error writing final output for {temp_file.name}: {e}")
            console_print(f"[ERROR] Failed to write final output for {temp_file.name}: {e}")

        # Delete temporary JSONL file if we successfully wrote the final output
        if processing_success and not processing_settings.get("retain_temporary_jsonl", True):
            try:
                temp_file.unlink()
                logger.info(f"Deleted temporary file after processing: {temp_file}")
                console_print(f"[CLEANUP] Deleted temporary file: {temp_file.name}")
            except Exception as e:
                logger.exception(f"Error deleting temporary file {temp_file}: {e}")
                console_print(f"[ERROR] Could not delete temporary file {temp_file.name}: {e}")

    console_print(f"\n[INFO] Completed processing batches in directory: {root_folder}")
    logger.info(f"Batch results processing complete for directory: {root_folder}")

def main() -> None:
    """
    Entrypoint for checking all directories for batch outputs and finalizing them.
    """
    scan_dirs, processing_settings = load_config()
    client = OpenAI()
    for directory in scan_dirs:
        process_all_batches(directory, processing_settings, client)
    console_print("\n[INFO] Batch results processing complete across all directories.")
    logger.info("Batch results processing complete across all directories.")

if __name__ == "__main__":
    main()
