# check_batches.py
"""
Script to check whether batch jobs have finished successfully (i.e.,
are marked as completed) and—if so—to download and process them.
Temporary .jsonl files and image folders will only be deleted if the final
output is successfully written and all batches in a JSONL file are completed.
"""

import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set, Optional

import requests
from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.text_processing import process_batch_output
from modules.path_utils import validate_paths
from modules.utils import extract_page_number_from_filename, console_print
from modules.user_interface import UserPrompt

logger = setup_logger(__name__)


class OpenAIHttpClient:
    """
    Minimal HTTP client for OpenAI endpoints used in this script.
    Avoids the OpenAI SDK to sidestep pydantic dependency issues.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")
        self.base_url = "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })

    def list_batches(self, limit: int = 100) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/batches"
        resp = self.session.get(url, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data if isinstance(data, list) else [])

    def retrieve_batch(self, batch_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/batches/{batch_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_file_content(self, file_id: str) -> bytes:
        url = f"{self.base_url}/files/{file_id}/content"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.content

    def list_models(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/models"
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])


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


def diagnose_batch_failure(batch_id: str, client: OpenAIHttpClient) -> str:
    """
    Attempts to diagnose why a batch might have failed.
    Returns a diagnostic message.
    """
    try:
        # Try to get batch details
        batch = client.retrieve_batch(batch_id)
        status = str(batch.get("status", "")).lower()

        if status == "failed":
            return f"Batch {batch_id} failed. Check your OpenAI dashboard for specific error details."
        elif status == "cancelled":
            return f"Batch {batch_id} was cancelled."
        elif status == "expired":
            return f"Batch {batch_id} expired (not completed within 24 hours)."
        else:
            return f"Batch {batch_id} has status '{status}'."
    except Exception as e:
        error_message = str(e).lower()
        if "not found" in error_message:
            return f"Batch {batch_id} not found in OpenAI. It may have been deleted or belong to a different API key."
        elif "unauthorized" in error_message:
            return "API key unauthorized. Check your OpenAI API key permissions."
        elif "quota" in error_message:
            return "API quota exceeded. Check your usage limits."
        else:
            return f"Error checking batch {batch_id}: {e}"


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


def check_batch_debug_file() -> Optional[Dict[str, Any]]:
    """
    Check if a batch debug file exists and return its contents.
    This can help with troubleshooting missing batch IDs.
    """
    debug_path = Path("batch_submission_debug.json")
    if debug_path.exists():
        try:
            with debug_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading batch debug file: {e}")
    return None


def process_all_batches(root_folder: Path, processing_settings: Dict[str, Any],
                        client: OpenAIHttpClient) -> None:
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

    # Check if we have a debug file with batch submission data
    debug_data = check_batch_debug_file()
    if debug_data:
        console_print(
            f"[INFO] Found batch debug data for {debug_data.get('total_batches', 0)} batches")

    # Retrieve all batches from OpenAI
    console_print("[INFO] Retrieving list of submitted batches from OpenAI...")
    try:
        batches = client.list_batches(limit=100)
        # Create a dictionary of batch ID to batch object for faster lookup
        batch_dict = {b.get("id"): b for b in batches if isinstance(b, dict) and b.get("id")}
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

        # Ensure this is always defined even if we skip processing later
        all_transcriptions: List[Dict[str, Any]] = []

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
                f"[WARN] No batch IDs found in {temp_file.name}. Checking if this file needs repair...")

            # Check if the debug data can help us repair this file
            if debug_data and debug_data.get("batch_data"):
                console_print(
                    f"[INFO] Attempting to repair {temp_file.name} using debug data")
                try:
                    # Write batch tracking records from debug data
                    with temp_file.open("a", encoding="utf-8") as f:
                        for batch_data in debug_data.get("batch_data", []):
                            tracking_record = {
                                "batch_tracking": {
                                    "batch_id": batch_data["batch_id"],
                                    "timestamp": batch_data["timestamp"],
                                    "batch_file": str(batch_data["batch_id"])
                                }
                            }
                            f.write(json.dumps(tracking_record) + "\n")
                            batch_ids.add(batch_data["batch_id"])

                    console_print(
                        f"[SUCCESS] Added {len(debug_data.get('batch_data', []))} missing batch IDs to {temp_file.name}")
                except Exception as e:
                    console_print(f"[ERROR] Failed to repair file: {e}")
                    continue
            else:
                console_print(
                    f"[WARN] No debug data available to repair {temp_file.name}. Skipping this file.")
                continue

        # Check if all batches are completed
        all_completed = True
        missing_batches = []
        completed_count = 0
        failed_count = 0
        failed_details = []

        for batch_id in batch_ids:
            if batch_id not in batch_dict:
                all_completed = False
                missing_batches.append(batch_id)
                diagnosis = diagnose_batch_failure(batch_id, client)
                logger.warning(
                    f"Batch ID {batch_id} not found in OpenAI batches. {diagnosis}")
                continue

            batch = batch_dict[batch_id]
            status = str(batch.get("status", "")).lower()

            if status == "completed":
                completed_count += 1
            elif status == "failed":
                failed_count += 1
                failed_details.append(f"Batch {batch_id}: status={status}")
                all_completed = False
            else:
                all_completed = False
                logger.info(
                    f"Batch {batch_id} has status '{status}' - not yet completed.")

        # Display progress information for this temp file
        console_print(f"\n----- Processing File: {temp_file.name} -----")
        console_print(f"Found {len(batch_ids)} batch ID(s)")

        if not all_completed:
            if missing_batches:
                console_print(
                    f"Completed: {completed_count} | Failed: {failed_count} | In Progress: {len(batch_ids) - completed_count - failed_count - len(missing_batches)} | Missing: {len(missing_batches)}")
                console_print(
                    f"[WARN] {len(missing_batches)} batch ID(s) were not found in the API response")
            else:
                console_print(
                    f"Completed: {completed_count} | Failed: {failed_count} | In Progress: {len(batch_ids) - completed_count - failed_count}")

            if failed_count > 0:
                console_print(
                    f"[WARN] {failed_count} batches have failed. Check the OpenAI dashboard for details.")

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
                file_content = client.get_file_content(batch.get("output_file_id"))
            except Exception as e:
                logger.exception(
                    f"Error downloading batch {batch_id}: {e}")
                console_print(
                    f"[ERROR] Failed to download output for Batch ID {batch_id}: {e}")
                # If any batch fails to download, skip writing the output
                all_completed = False
                break

            # Try to parse the file content directly to get custom_id mapping
            try:
                batch_response_text = file_content.decode(
                    'utf-8') if isinstance(file_content,
                                           bytes) else file_content
                batch_response_lines = batch_response_text.strip().split('\n')
                for line in batch_response_lines:
                    if not line.strip():
                        continue
                    response_obj = json.loads(line)
                    custom_id = response_obj.get("custom_id")

                    if "response" in response_obj and "body" in response_obj["response"]:
                        response_body = response_obj["response"]["body"]
                        choices = response_body.get("choices", [])
                        if choices:
                            message_content = choices[0].get("message", {}).get("content", "")

                            try:
                                # Try to parse JSON content if present
                                transcription_data = json.loads(message_content)

                                # Store the transcription along with custom_id and any order info
                                transcription_entry = {
                                    "custom_id": custom_id,
                                    "transcription": transcription_data.get("transcription", ""),
                                    "order_info": batch_order.get(custom_id, None),
                                    "image_info": custom_id_map.get(custom_id, {})
                                }

                                all_transcriptions.append(transcription_entry)

                            except json.JSONDecodeError:
                                # If not JSON, just use the raw content
                                all_transcriptions.append({
                                    "custom_id": custom_id,
                                    "transcription": message_content,
                                    "order_info": batch_order.get(custom_id, None),
                                    "image_info": custom_id_map.get(custom_id, {})
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

    can_write_output = True
    if not all_completed:
        console_print(
            f"[WARN] Failed to process all batches for {temp_file.name}. Skipping output writing.")
        can_write_output = False

    if not all_transcriptions:
        logger.warning(
            f"No transcriptions extracted for any batch in {temp_file.name}.")
        console_print(
            f"[WARN] No transcriptions extracted for any batch in {temp_file.name}. Skipping this file.")
        can_write_output = False

    if can_write_output:

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


def diagnose_api_issues() -> None:
    """
    Provide diagnostics on common API issues.
    """
    console_print("\n=== API Issue Diagnostics ===")

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console_print("[ERROR] No OpenAI API key found in environment variables")
    else:
        key_summary = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 10 else "[hidden]"
        console_print(f"[INFO] OpenAI API key present: {key_summary}")

    # Check for common model issues using HTTP client
    client = OpenAIHttpClient()
    try:
        models = client.list_models()
        has_gpt4o = any("gpt-4o" in (m.get("id", "")) for m in models)
        console_print(f"[INFO] API Connection successful: {len(models)} models available")
        console_print(f"[INFO] GPT-4o models available: {'Yes' if has_gpt4o else 'No'}")
    except Exception as e:
        console_print(f"[ERROR] Failed to list models: {e}")

    # Check for batch issues
    try:
        batch_list = client.list_batches(limit=1)
        console_print("[INFO] Batch API access successful")
    except Exception as e:
        console_print(f"[ERROR] Batch API access failed: {e}")


def main() -> None:
    """
    Entrypoint for checking all directories for batch outputs and finalizing them.
    """
    scan_dirs, processing_settings = load_config()
    client = OpenAIHttpClient()

    # Run diagnostics
    diagnose_api_issues()

    for directory in scan_dirs:
        process_all_batches(directory, processing_settings, client)
    console_print("\n[INFO] Batch results processing complete across all directories.")
    logger.info("Batch results processing complete across all directories.")


if __name__ == "__main__":
    main()