"""Batch finalization operations.

This module encapsulates the logic to scan local JSONL artifacts, verify that
all OpenAI Batch jobs are completed, download their results, merge outputs in
page order, and write final transcription files. It keeps side-effectful UI
printing and logging, but is importable by tests and thin CLIs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from datetime import datetime, timezone

from openai import OpenAI

from modules.llm.batch.batch_utils import diagnose_batch_failure, extract_custom_id_mapping
from modules.config.config_loader import ConfigLoader
from modules.infra.logger import setup_logger
from modules.llm.openai_sdk_utils import coerce_file_id, list_all_batches, sdk_to_dict
from modules.io.path_utils import validate_paths
from modules.processing.text_processing import (
    extract_transcribed_text,
    process_batch_output,
    format_page_line,
)
from modules.ui.core import UserPrompt
from modules.core.utils import console_print

logger = setup_logger(__name__)


def load_config() -> Tuple[List[Path], Dict[str, Any]]:
    """Load YAML configuration and return (scan_dirs, processing_settings).

    - Ensures file paths exist (creates input/output directories on demand)
    - Validates configured paths
    """
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config = config_loader.get_paths_config()

    # Validate configured paths
    validate_paths(paths_config)

    processing_settings = paths_config.get("general", {})

    file_paths = paths_config.get("file_paths", {})
    scan_dirs: List[Path] = []
    for _key, folders in file_paths.items():
        for folder_key in ["input", "output"]:
            folder_path = folders.get(folder_key)
            if folder_path:
                dir_path = Path(folder_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                scan_dirs.append(dir_path.resolve())
    scan_dirs = list(set(scan_dirs))
    return scan_dirs, processing_settings


def process_all_batches(
    root_folder: Path, processing_settings: Dict[str, Any], client: OpenAI
) -> None:
    """Finalize all completed batch jobs for a given root folder.

    Scans for ``*_transcription.jsonl`` files, retrieves batch summaries,
    safeguards ordering using custom_id metadata, and writes a final
    ``*_transcription.txt`` when all parts are available.
    """
    console_print(
        f"\n[INFO] Scanning directory '{root_folder}' for temporary batch files..."
    )
    temp_files = list(root_folder.rglob("*_transcription.jsonl"))
    if not temp_files:
        console_print(f"[INFO] No temporary batch files found in {root_folder}.")
        logger.info(f"No temporary batch files found in {root_folder}.")
        return

    # Retrieve all batches from OpenAI
    console_print("[INFO] Retrieving list of submitted batches from OpenAI...")
    try:
        batches = list_all_batches(client)
        batch_dict: Dict[str, Dict[str, Any]] = {
            b.get("id"): b for b in batches if isinstance(b, dict) and b.get("id")
        }
    except Exception as e:
        console_print(f"[ERROR] Failed to retrieve batches from OpenAI: {e}")
        logger.exception(f"Error retrieving batches: {e}")
        return

    # Display batch summary (handles dicts or SDK objects)
    UserPrompt.display_batch_summary(batches)

    # Process each temporary file
    for temp_file in temp_files:
        identifier = temp_file.stem.replace("_transcription", "")
        batch_ids: Set[str] = set()

        # Track markers to distinguish batch vs non-batch JSONL
        has_batch_metadata = False
        has_batch_request = False
        has_batch_session = False
        image_metadata_count = 0
        batch_request_count = 0
        batch_session_statuses: Set[str] = set()

        all_transcriptions: List[Dict[str, Any]] = []

        with temp_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)

                    if "batch_tracking" in record:
                        batch_id = record["batch_tracking"].get("batch_id")
                        if batch_id:
                            batch_ids.add(batch_id)

                    elif "image_metadata" in record and isinstance(
                        record["image_metadata"], dict
                    ):
                        meta = record["image_metadata"]
                        if meta.get("custom_id"):
                            has_batch_metadata = True
                            image_metadata_count += 1

                    elif "batch_request" in record and isinstance(
                        record["batch_request"], dict
                    ):
                        has_batch_request = True
                        batch_request_count += 1

                    elif "batch_session" in record and isinstance(
                        record["batch_session"], dict
                    ):
                        has_batch_session = True
                        status_val = record["batch_session"].get("status")
                        if isinstance(status_val, str):
                            batch_session_statuses.add(status_val.lower().strip())

                except json.JSONDecodeError:
                    continue

        # Attempt recovery of missing batch IDs from the debug artifact before classification
        if not batch_ids:
            debug_artifact = temp_file.parent / f"{identifier}_batch_submission_debug.json"
            if debug_artifact.exists():
                try:
                    dbg = json.loads(debug_artifact.read_text(encoding="utf-8"))
                    dbg_ids = [bid for bid in (dbg.get("batch_ids") or []) if isinstance(bid, str)]
                    to_add = [bid for bid in dbg_ids if bid not in batch_ids]
                    if to_add:
                        console_print(
                            f"[INFO] Recovered {len(to_add)} missing batch id(s) for {temp_file.name} from debug artifact."
                        )
                        for bid in to_add:
                            batch_ids.add(bid)
                        # Best-effort persist into the JSONL so future runs have them
                        persist = bool(processing_settings.get("persist_recovered_batch_ids", True))
                        if persist:
                            try:
                                with temp_file.open("a", encoding="utf-8") as wf:
                                    for bid in to_add:
                                        rec = {
                                            "batch_tracking": {
                                                "batch_id": bid,
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                                "batch_file": str(bid),
                                            }
                                        }
                                        wf.write(json.dumps(rec) + "\n")
                                console_print(
                                    f"[INFO] Persisted {len(to_add)} recovered batch id(s) into {temp_file.name}."
                                )
                            except Exception as pe:
                                logger.warning(
                                    "Failed to persist recovered batch ids for %s: %s", temp_file.name, pe
                                )
                except Exception as de:
                    logger.warning("Failed to read debug artifact %s: %s", debug_artifact.name, de)

        is_batched_file = has_batch_session and (
            bool(batch_ids) or has_batch_request or has_batch_metadata
        )

        if not is_batched_file:
            if has_batch_session and not batch_ids:
                console_print(
                    f"[WARN] {temp_file.name} has a batch_session marker but no batch IDs; skipping. "
                    f"Use 'main/repair_transcriptions.py' if needed."
                )
            elif batch_ids:
                console_print(
                    f"[WARN] {temp_file.name} contains {len(batch_ids)} batch_tracking entries but no "
                    f"batch_session marker; skipping as non-batched."
                )
            elif has_batch_metadata or has_batch_request:
                console_print(
                    f"[INFO] {temp_file.name} has batch-like metadata but no batch_session marker; "
                    f"treating as non-batched and skipping."
                )
            else:
                console_print(
                    f"[INFO] {temp_file.name} has no batch markers; treating as non-batched and skipping."
                )
            continue

        if not batch_ids:
            console_print(
                f"[WARN] No batch IDs found in {temp_file.name}. This file appears to be batched but "
                f"missing tracking entries. Use 'main/repair_transcriptions.py' if you need to "
                f"reconstruct outputs."
            )
            continue

        # Check if all batches are completed
        all_completed = True
        missing_batches: List[str] = []
        completed_count = 0
        failed_count = 0

        for batch_id in batch_ids:
            if batch_id not in batch_dict:
                # Attempt to retrieve individually (handles pagination and older batches)
                try:
                    b_obj = client.batches.retrieve(batch_id)
                    batch = sdk_to_dict(b_obj)
                    if isinstance(batch, dict) and batch.get("id"):
                        batch_dict[batch_id] = batch
                    else:
                        all_completed = False
                        missing_batches.append(batch_id)
                        logger.warning(
                            "Batch ID %s retrieval returned no object", batch_id
                        )
                        continue
                except Exception:
                    all_completed = False
                    missing_batches.append(batch_id)
                    diagnosis = diagnose_batch_failure(batch_id, client)
                    logger.warning(
                        f"Batch ID {batch_id} not found in OpenAI batches. {diagnosis}"
                    )
                    continue

            batch = batch_dict[batch_id]
            status = str(batch.get("status", "")).lower()

            if status == "completed":
                completed_count += 1
            elif status == "failed":
                failed_count += 1
                all_completed = False
            else:
                all_completed = False
                logger.info(
                    f"Batch {batch_id} has status '{status}' - not yet completed."
                )

        # Display progress information for this temp file (standardized UI helper)
        UserPrompt.display_batch_processing_progress(
            temp_file=temp_file,
            batch_ids=batch_ids,
            completed_count=completed_count,
            missing_count=len(missing_batches),
        )

        if not all_completed:
            if failed_count > 0:
                console_print(
                    f"[WARN] {failed_count} batches have failed. Check the OpenAI dashboard for details."
                )
            continue

        # All batches are completed, now download and process them
        console_print(
            f"[INFO] All batches for {temp_file.name} report 'completed'. "
            f"Attempting to download and process results..."
        )

        # Extract custom_id mapping if available
        custom_id_map, batch_order = extract_custom_id_mapping(temp_file)

        # Collect all transcriptions from all batches
        all_transcriptions = []

        for batch_id in batch_ids:
            batch = batch_dict[batch_id]

            # Resolve the output file id robustly
            output_file_id = coerce_file_id(batch.get("output_file_id"))

            if not output_file_id:
                # Refresh batch details (list endpoint may omit some fields)
                try:
                    refreshed = sdk_to_dict(client.batches.retrieve(batch_id))
                    if isinstance(refreshed, dict) and refreshed.get("id"):
                        batch = refreshed
                        batch_dict[batch_id] = refreshed
                        output_file_id = coerce_file_id(batch.get("output_file_id"))
                except Exception as e:
                    logger.warning("Could not refresh batch %s: %s", batch_id, e)

            if not output_file_id:
                # Try alternate field shapes just in case
                for key in (
                    "output_file_id",
                    "output_file",
                    "output_file_ids",
                    "response_file_id",
                    "result_file_id",
                    "results_file_id",
                    "result_file_ids",
                ):
                    output_file_id = coerce_file_id(batch.get(key))
                    if output_file_id:
                        break

            if not output_file_id:
                # Try alternate diagnostics: error_file_id may contain per-request errors
                error_file_id = None

                for key in (
                    "error_file_id",
                    "error_file",
                    "errors_file_id",
                    "error_file_ids",
                ):
                    error_file_id = coerce_file_id(batch.get(key))
                    if error_file_id:
                        break

                console_print(
                    f"[ERROR] Batch {batch_id} is marked completed but no output_file_id was found. "
                    f"Skipping this batch."
                )
                logger.warning("Batch %s completed without output_file_id.", batch_id)

                if error_file_id:
                    try:
                        # Download error file from OpenAI
                        resp = client.files.content(error_file_id)
                        err_bytes = resp.read()
                        # Use a short, path-safe filename to avoid MAX_PATH issues on Windows
                        short_batch_id = batch_id.replace("batch_", "")[:16]
                        errors_filename = f"errors_{short_batch_id}.jsonl"
                        errors_path = temp_file.parent / errors_filename
                        errors_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            err_text = (
                                err_bytes.decode("utf-8")
                                if isinstance(err_bytes, (bytes, bytearray))
                                else str(err_bytes)
                            )
                        except Exception:
                            err_text = str(err_bytes)
                        with errors_path.open("w", encoding="utf-8") as ef:
                            ef.write(err_text)
                        console_print(
                            f"[INFO] Saved error details for batch {batch_id} to {errors_path.name}"
                        )
                        logger.info(
                            "Saved error file for %s -> %s", batch_id, errors_path
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not download error_file_id for %s: %s", batch_id, e
                        )

                all_completed = False
                continue

            try:
                file_stream = client.files.content(output_file_id)
                file_content = file_stream.read()
            except Exception as e:
                logger.exception(f"Error downloading batch {batch_id}: {e}")
                console_print(
                    f"[ERROR] Failed to download output for Batch ID {batch_id}: {e}"
                )
                # If any batch fails to download, skip writing the output
                all_completed = False
                break

            # Parse the batch output file content (Responses API aware)
            try:
                batch_response_text = (
                    file_content.decode("utf-8")
                    if isinstance(file_content, bytes)
                    else str(file_content)
                )
                batch_response_lines = batch_response_text.strip().split("\n")
                for line in batch_response_lines:
                    if not line.strip():
                        continue
                    try:
                        response_obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    custom_id = response_obj.get("custom_id")
                    image_info = custom_id_map.get(custom_id, {}) if custom_id else {}
                    image_name = image_info.get("image_name") or (
                        custom_id or "[unknown image]"
                    )
                    page_number = image_info.get("page_number")
                    transcription_text = None

                    # Primary: Responses API body with status/diagnostics
                    resp = response_obj.get("response")
                    if isinstance(resp, dict):
                        status_code = resp.get("status_code")
                        body = resp.get("body")
                        if isinstance(body, dict):
                            # Detect explicit API errors per item
                            error_obj = body.get("error")
                            if isinstance(status_code, int) and status_code != 200:
                                err_code = (error_obj or {}).get("code") or (
                                    error_obj or {}
                                ).get("type")
                                err_message = (error_obj or {}).get(
                                    "message"
                                ) or (error_obj or {}).get("error")
                                diag_text = (
                                    f"[transcription error; status {status_code}"
                                )
                                if err_code:
                                    diag_text += f"; code {err_code}"
                                if err_message:
                                    diag_text += f"; {err_message}"
                                diag_text += "]"
                                UserPrompt.print_transcription_item_error(
                                    image_name=image_name,
                                    page_number=page_number,
                                    status_code=status_code,
                                    err_code=err_code,
                                    err_message=err_message,
                                )
                                all_transcriptions.append(
                                    {
                                        "custom_id": custom_id,
                                        "transcription": diag_text,
                                        "order_info": batch_order.get(
                                            custom_id, None
                                        ),
                                        "image_info": image_info,
                                        "error": True,
                                        "error_details": {
                                            "status_code": status_code,
                                            "code": err_code,
                                            "message": err_message,
                                        },
                                    }
                                )
                                continue
                            if isinstance(error_obj, dict):
                                err_code = error_obj.get("code") or error_obj.get("type")
                                err_message = error_obj.get("message") or error_obj.get(
                                    "error"
                                )
                                diag_text = f"[transcription error"
                                if err_code:
                                    diag_text += f"; code {err_code}"
                                if err_message:
                                    diag_text += f"; {err_message}"
                                diag_text += "]"
                                UserPrompt.print_transcription_item_error(
                                    image_name=image_name,
                                    page_number=page_number,
                                    err_code=err_code,
                                    err_message=err_message,
                                )
                                all_transcriptions.append(
                                    {
                                        "custom_id": custom_id,
                                        "transcription": diag_text,
                                        "order_info": batch_order.get(
                                            custom_id, None
                                        ),
                                        "image_info": image_info,
                                        "error": True,
                                        "error_details": {
                                            "status_code": status_code,
                                            "code": err_code,
                                            "message": err_message,
                                        },
                                    }
                                )
                                continue
                            # Success path: extract transcription text
                            transcription_text = extract_transcribed_text(body, "")

                    # Secondary: legacy Chat Completions style at body level
                    if not transcription_text and isinstance(resp, dict):
                        body = resp.get("body")
                        if isinstance(body, dict) and "choices" in body:
                            transcription_text = extract_transcribed_text(body, "")

                    if transcription_text:
                        # Provide page-aware placeholders for special cases
                        if transcription_text == "[Transcription not possible]":
                            UserPrompt.print_transcription_not_possible(
                                image_name=image_name, page_number=page_number
                            )
                            transcription_text = "[Transcription not possible]"
                            entry = {
                                "custom_id": custom_id,
                                "transcription": transcription_text,
                                "order_info": batch_order.get(custom_id, None),
                                "image_info": image_info,
                                "warning": True,
                                "warning_type": "transcription_not_possible",
                            }
                        elif transcription_text == "[No transcribable text]":
                            UserPrompt.print_no_transcribable_text(
                                image_name=image_name, page_number=page_number
                            )
                            transcription_text = "[No transcribable text]"
                            entry = {
                                "custom_id": custom_id,
                                "transcription": transcription_text,
                                "order_info": batch_order.get(custom_id, None),
                                "image_info": image_info,
                                "warning": True,
                                "warning_type": "no_transcribable_text",
                            }
                        else:
                            entry = {
                                "custom_id": custom_id,
                                "transcription": transcription_text,
                                "order_info": batch_order.get(custom_id, None),
                                "image_info": image_info,
                            }
                        all_transcriptions.append(entry)

                # If nothing parsed, fall back to generic processor
                if not all_transcriptions:
                    transcriptions = process_batch_output(file_content)
                    for idx, transcription in enumerate(transcriptions):
                        all_transcriptions.append(
                            {"transcription": transcription, "fallback_index": idx}
                        )
            except Exception as json_parse_error:
                # Fall back to the process_batch_output function if direct parsing fails
                logger.warning(
                    f"Could not directly parse batch result, falling back to process_batch_output: "
                    f"{json_parse_error}"
                )
                transcriptions = process_batch_output(file_content)

                # In fallback mode, we can't maintain page order reliably
                for idx, transcription in enumerate(transcriptions):
                    all_transcriptions.append(
                        {"transcription": transcription, "fallback_index": idx}
                    )

        can_write_output = True
        if not all_completed:
            console_print(
                f"[WARN] Failed to process all batches for {temp_file.name}. Skipping output writing."
            )
            can_write_output = False

        if not all_transcriptions:
            logger.warning(
                f"No transcriptions extracted for any batch in {temp_file.name}."
            )
            console_print(
                f"[WARN] No transcriptions extracted for any batch in {temp_file.name}. "
                f"Skipping this file."
            )
            can_write_output = False

        if can_write_output:
            # Build the final text file path
            identifier = temp_file.stem.replace("_transcription", "")
            final_txt_path = temp_file.parent / f"{identifier}_transcription.txt"

            # Apply a multi-level sorting strategy
            console_print(f"[INFO] Arranging transcriptions in the correct order...")
            console_print(
                f"[INFO] Found {len(all_transcriptions)} transcriptions to combine."
            )
            console_print(
                f"[INFO] Order tracking data: {len(batch_order)} custom IDs mapped to order indices."
            )
            console_print(
                f"[INFO] Using multi-level sorting to ensure correct page order."
            )

            def get_sorting_key(transcription_entry):
                """Return a sorting key tuple for stable page ordering."""
                # 1) order_info attached during parsing (authoritative)
                order_info = transcription_entry.get("order_info")
                if order_info is not None:
                    return (0, order_info)

                # 2) order_index from batch_order mapping derived from JSONL metadata
                custom_id = transcription_entry.get("custom_id")
                if custom_id in batch_order:
                    return (1, batch_order[custom_id])

                # 3) Derive index from a custom_id pattern like 'req-<n>'
                if isinstance(custom_id, str) and custom_id.startswith("req-"):
                    try:
                        req_idx = int(custom_id.split("-", 1)[1]) - 1
                        if req_idx >= 0:
                            return (2, req_idx)
                    except Exception:
                        pass

                # 4) Fallback to the enumeration index used when we had to degrade
                return (3, transcription_entry.get("fallback_index", 999999))

            # Sort the transcriptions using the multi-level key
            all_transcriptions.sort(key=get_sorting_key)

            # Log the sorting order for debugging
            logger.info(
                f"Sorted {len(all_transcriptions)} transcriptions for {temp_file.name}"
            )
            for idx, entry in enumerate(all_transcriptions):
                custom_id = entry.get("custom_id", "unknown")
                order_info = entry.get("order_info")
                image_name = entry.get("image_info", {}).get("image_name", "unknown")
                logger.info(
                    f"Position {idx}: custom_id={custom_id}, order_info={order_info}, image={image_name}"
                )

            # Summarize per-page diagnostics to console
            error_entries = [e for e in all_transcriptions if e.get("error")]
            if error_entries:
                UserPrompt.display_page_error_summary(error_entries)

            np_entries = [
                e
                for e in all_transcriptions
                if e.get("warning_type") == "transcription_not_possible"
            ]
            if np_entries:
                UserPrompt.display_transcription_not_possible_summary(len(np_entries))

            # Build page-identified lines in sorted order
            ordered_lines = []
            for t in all_transcriptions:
                tx = t.get("transcription", "")
                info = t.get("image_info", {}) or {}
                pn = info.get("page_number")
                iname = info.get("image_name")
                ordered_lines.append(format_page_line(tx, pn, iname))

            processing_success = False
            try:
                with final_txt_path.open("w", encoding="utf-8") as fout:
                    for line in ordered_lines:
                        if line:
                            fout.write(line + "\n")
                logger.info(
                    f"All batches for {temp_file.name} processed and saved to {final_txt_path}"
                )
                console_print(
                    f"[SUCCESS] Processed all batches for {temp_file.name}. Results saved to "
                    f"{final_txt_path.name}"
                )
                processing_success = True
            except Exception as e:
                logger.exception(
                    f"Error writing final output for {temp_file.name}: {e}"
                )
                console_print(
                    f"[ERROR] Failed to write final output for {temp_file.name}: {e}"
                )

            # Delete temporary JSONL file if we successfully wrote the final output
            if processing_success and not processing_settings.get(
                "retain_temporary_jsonl", True
            ):
                try:
                    temp_file.unlink()
                    logger.info(f"Deleted temporary file after processing: {temp_file}")
                    console_print(f"[CLEANUP] Deleted temporary file: {temp_file.name}")
                except Exception as e:
                    logger.exception(f"Error deleting temporary file {temp_file}: {e}")
                    console_print(
                        f"[ERROR] Could not delete temporary file {temp_file.name}: {e}"
                    )

    console_print(f"\n[INFO] Completed processing batches in directory: {root_folder}")
    logger.info(f"Batch results processing complete for directory: {root_folder}")


def diagnose_api_issues() -> None:
    """Print quick diagnostics for API key presence and SDK connectivity."""
    console_print("\n=== API Issue Diagnostics ===")

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console_print("[ERROR] No OpenAI API key found in environment variables")
    else:
        key_summary = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 10 else "[hidden]"
        console_print(f"[INFO] OpenAI API key present: {key_summary}")

    # Check for common model issues using SDK
    try:
        client = OpenAI()
        models_page = client.models.list()
        models_data: Iterable[Any] = getattr(models_page, "data", None) or []
        total_models = len(list(models_data))
        console_print(f"[INFO] API Connection successful: {total_models} models available")
    except Exception as e:
        console_print(f"[ERROR] Failed to list models: {e}")

    # Check for batch issues
    try:
        client = OpenAI()
        _ = client.batches.list(limit=1)
        console_print("[INFO] Batch API access successful")
    except Exception as e:
        console_print(f"[ERROR] Batch API access failed: {e}")


def run_batch_finalization(run_diagnostics: bool = True, custom_directory: Path | None = None) -> None:
    """High-level entrypoint used by the CLI to finalize batch results.
    
    Args:
        run_diagnostics: Whether to run API diagnostics before processing
        custom_directory: Optional custom directory to scan instead of config defaults
    """
    if custom_directory:
        # Use the specified directory instead of loading from config
        scan_dirs = [custom_directory]
        # Load minimal processing settings from config
        config_loader = ConfigLoader()
        config_loader.load_configs()
        paths_config = config_loader.get_paths_config()
        processing_settings = paths_config.get("general", {})
    else:
        # Load standard configuration
        scan_dirs, processing_settings = load_config()
    
    client = OpenAI()

    if run_diagnostics:
        diagnose_api_issues()

    for directory in scan_dirs:
        process_all_batches(directory, processing_settings, client)
    console_print("\n[INFO] Batch results processing complete across all directories.")
    logger.info("Batch results processing complete across all directories.")
