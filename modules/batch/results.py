"""Batch result downloading, transcription sorting, and output finalization.

Provides functions to download and parse batch results from OpenAI and
non-OpenAI providers, sort transcriptions into page order, and write the
final transcription output files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI, OpenAIError

from modules.infra.logger import setup_logger
from modules.batch.backends import BatchHandle, BatchStatus, get_batch_backend
from modules.batch.mapping import extract_custom_id_mapping
from modules.llm.openai_sdk_utils import coerce_file_id, sdk_to_dict
from modules.postprocess.writer import write_transcription_output
from modules.llm.response_parsing import (
    extract_transcribed_text,
    process_batch_output,
)
from modules.ui import print_error, print_info, print_success, print_warning
from modules.ui.batch_display import (
    display_batch_processing_progress,
    display_page_error_summary,
    display_transcription_not_possible_summary,
    print_no_transcribable_text,
    print_transcription_item_error,
    print_transcription_not_possible,
)

logger = setup_logger(__name__)


def _process_non_openai_batch(
    temp_file: Path,
    batch_ids: Set[str],
    batch_provider: str,
    batch_tracking_records: List[Dict[str, Any]],
    processing_settings: Dict[str, Any],
    postprocessing_config: Optional[Dict[str, Any]] = None,
    output_format: str = "txt",
) -> None:
    """Process batch results for non-OpenAI providers (Anthropic, Google).

    Uses the BatchBackend abstraction to check status and download results.
    """
    from modules.batch.mapping import extract_custom_id_mapping

    identifier = temp_file.stem.replace("_transcription", "")

    print_info(f"Processing {batch_provider.upper()} batch for {temp_file.name}...")

    try:
        backend = get_batch_backend(batch_provider)
    except (ValueError, ImportError) as e:
        print_error(f"Failed to get batch backend for {batch_provider}: {e}")
        return

    # Check status of all batches
    all_completed = True
    completed_count = 0
    failed_count = 0

    for batch_id in batch_ids:
        # Reconstruct handle from tracking record
        tracking = next(
            (t for t in batch_tracking_records if t.get("batch_id") == batch_id),
            {}
        )
        handle = BatchHandle(
            provider=batch_provider,
            batch_id=batch_id,
            metadata=tracking.get("metadata", {}),
        )

        try:
            status_info = backend.get_status(handle)
        except (OSError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            print_warning(f"Failed to get status for batch {batch_id}: {e}")
            all_completed = False
            continue

        if status_info.status == BatchStatus.COMPLETED:
            completed_count += 1
        elif status_info.status == BatchStatus.FAILED:
            failed_count += 1
            all_completed = False
            print_warning(f"Batch {batch_id} failed: {status_info.error_message or 'Unknown error'}")
        elif status_info.status in (BatchStatus.CANCELLED, BatchStatus.EXPIRED):
            all_completed = False
            print_warning(f"Batch {batch_id} was {status_info.status.value}")
        else:
            all_completed = False
            print_info(f"Batch {batch_id} has status '{status_info.status.value}' - not yet completed.")

    # Display progress
    display_batch_processing_progress(
        temp_file=temp_file,
        batch_ids=list(batch_ids),
        completed_count=completed_count,
        missing_count=0,
    )

    if not all_completed:
        if failed_count > 0:
            print_warning(
                f"{failed_count} batches have failed. Check the {batch_provider} dashboard for details."
            )
        return

    # All batches completed - download results
    print_info(
        f"All batches for {temp_file.name} report 'completed'. "
        f"Attempting to download and process results..."
    )

    # Extract custom_id mapping
    custom_id_map, batch_order = extract_custom_id_mapping(temp_file)

    # Collect all transcriptions
    all_transcriptions: List[Dict[str, Any]] = []

    for batch_id in batch_ids:
        tracking = next(
            (t for t in batch_tracking_records if t.get("batch_id") == batch_id),
            {}
        )
        handle = BatchHandle(
            provider=batch_provider,
            batch_id=batch_id,
            metadata=tracking.get("metadata", {}),
        )

        try:
            for result in backend.download_results(handle):
                image_info = custom_id_map.get(result.custom_id, {})
                image_name = image_info.get("image_name") or result.custom_id or "[unknown image]"
                page_number = image_info.get("page_number")

                if not result.success:
                    print_transcription_item_error(
                        image_name=image_name,
                        page_number=page_number,
                        err_code=result.error_code,
                        err_message=result.error,
                    )
                    all_transcriptions.append({
                        "custom_id": result.custom_id,
                        "transcription": f"[transcription error: {result.error}]",
                        "order_info": batch_order.get(result.custom_id),
                        "image_info": image_info,
                        "error": True,
                    })
                    continue

                # Extract transcription text
                transcription_text = result.content
                if result.parsed_output and isinstance(result.parsed_output, dict):
                    if "transcribed_text" in result.parsed_output:
                        transcription_text = result.parsed_output["transcribed_text"]

                # Handle special cases
                if result.no_transcribable_text:
                    print_no_transcribable_text(image_name=image_name, page_number=page_number)
                    transcription_text = "[No transcribable text]"
                    all_transcriptions.append({
                        "custom_id": result.custom_id,
                        "transcription": transcription_text,
                        "order_info": batch_order.get(result.custom_id),
                        "image_info": image_info,
                        "warning": True,
                        "warning_type": "no_transcribable_text",
                    })
                elif result.transcription_not_possible:
                    print_transcription_not_possible(image_name=image_name, page_number=page_number)
                    transcription_text = "[Transcription not possible]"
                    all_transcriptions.append({
                        "custom_id": result.custom_id,
                        "transcription": transcription_text,
                        "order_info": batch_order.get(result.custom_id),
                        "image_info": image_info,
                        "warning": True,
                        "warning_type": "transcription_not_possible",
                    })
                else:
                    all_transcriptions.append({
                        "custom_id": result.custom_id,
                        "transcription": transcription_text,
                        "order_info": batch_order.get(result.custom_id),
                        "image_info": image_info,
                    })

        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as e:
            print_error(f"Failed to download results for batch {batch_id}: {e}")
            logger.exception(f"Error downloading batch {batch_id}: {e}")
            return

    if not all_transcriptions:
        print_warning(f"No transcriptions extracted for {temp_file.name}. Skipping.")
        return

    # Sort transcriptions by order
    def get_sorting_key(entry: Dict[str, Any]) -> tuple[int, Any]:
        order_info = entry.get("order_info")
        if order_info is not None:
            return (0, order_info)
        custom_id = entry.get("custom_id")
        if custom_id in batch_order:
            return (1, batch_order[custom_id])
        if isinstance(custom_id, str) and custom_id.startswith("req-"):
            try:
                req_idx = int(custom_id.split("-", 1)[1]) - 1
                if req_idx >= 0:
                    return (2, req_idx)
            except (ValueError, IndexError):
                logger.debug("Could not parse req index from custom_id: %s", custom_id)
        return (3, 999999)

    all_transcriptions.sort(key=get_sorting_key)

    # Build final output
    final_txt_path = temp_file.parent / f"{identifier}.txt"

    pages = []
    for t in all_transcriptions:
        tx = t.get("transcription", "")
        info = t.get("image_info", {}) or {}
        pn = info.get("page_number")
        iname = info.get("image_name")
        pages.append({"text": tx, "page_number": pn, "image_name": iname})

    # Write output
    try:
        actual_path = write_transcription_output(
            pages,
            final_txt_path,
            output_format=output_format,
            postprocess=bool(postprocessing_config),
            postprocessing_config=postprocessing_config,
        )
        print_success(f"Saved transcription to {actual_path.name}")
        logger.info(f"Wrote final transcription: {actual_path}")
    except (OSError, ValueError, TypeError) as e:
        print_error(f"Failed to write transcription file: {e}")
        logger.exception(f"Error writing {final_txt_path}: {e}")
        return

    # Optionally delete temp JSONL
    if not processing_settings.get("retain_temporary_jsonl", True):
        try:
            temp_file.unlink()
            print_info(f"Deleted temporary file: {temp_file.name}")
        except OSError as e:
            logger.warning(f"Could not delete temp file {temp_file.name}: {e}")


def _download_and_parse_openai_results(
    batch_ids: Set[str],
    batch_dict: Dict[str, Dict[str, Any]],
    client: OpenAI,
    custom_id_map: Dict[str, Any],
    batch_order: Dict[str, Any],
    temp_file: Path,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Download results from OpenAI and parse them into transcription entries.

    Iterates over each completed batch, resolves its output file ID, downloads
    the JSONL result, and parses each response line (Responses API and legacy
    Chat Completions formats).  Falls back to ``process_batch_output`` when
    direct parsing fails.

    Returns:
        (all_transcriptions, all_completed) where *all_completed* may be set
        to ``False`` if any batch could not be downloaded or lacked an output file.
    """
    all_transcriptions: List[Dict[str, Any]] = []
    all_completed = True

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
            except (OpenAIError, OSError, ValueError, TypeError) as e:
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

            print_error(
                f"Batch {batch_id} is marked completed but no output_file_id was found. "
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
                    except (UnicodeDecodeError, ValueError):
                        logger.debug("Could not decode error bytes as UTF-8, falling back to str()")
                        err_text = str(err_bytes)
                    with errors_path.open("w", encoding="utf-8") as ef:
                        ef.write(err_text)
                    print_info(
                        f"Saved error details for batch {batch_id} to {errors_path.name}"
                    )
                    logger.info(
                        "Saved error file for %s -> %s", batch_id, errors_path
                    )
                except (OpenAIError, OSError, ValueError, TypeError) as e:
                    logger.warning(
                        "Could not download error_file_id for %s: %s", batch_id, e
                    )

            all_completed = False
            continue

        try:
            file_stream = client.files.content(output_file_id)
            file_content = file_stream.read()
        except (OpenAIError, OSError, ValueError) as e:
            logger.exception(f"Error downloading batch {batch_id}: {e}")
            print_error(
                f"Failed to download output for Batch ID {batch_id}: {e}"
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
                            print_transcription_item_error(
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
                            print_transcription_item_error(
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
                        print_transcription_not_possible(
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
                        print_no_transcribable_text(
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
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, AttributeError, ValueError, UnicodeDecodeError) as json_parse_error:
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

    return all_transcriptions, all_completed


def _sort_transcriptions(
    all_transcriptions: List[Dict[str, Any]],
    batch_order: Dict[str, Any],
    temp_file: Path,
) -> None:
    """Sort transcriptions in-place using the multi-level sorting strategy.

    The strategy uses four priority tiers:
    1. ``order_info`` attached during parsing (authoritative).
    2. ``batch_order`` mapping derived from JSONL metadata.
    3. Index parsed from a ``req-<n>`` custom_id pattern.
    4. Fallback enumeration index for degraded entries.

    Also logs the final sort order for debugging.
    """
    print_info(f"Arranging transcriptions in the correct order...")
    print_info(
        f"Found {len(all_transcriptions)} transcriptions to combine."
    )
    print_info(
        f"Order tracking data: {len(batch_order)} custom IDs mapped to order indices."
    )
    print_info(
        f"Using multi-level sorting to ensure correct page order."
    )

    def get_sorting_key(transcription_entry: Dict[str, Any]) -> tuple[int, Any]:
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
            except (ValueError, IndexError):
                logger.debug("Could not parse req index from custom_id: %s", custom_id)

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


def _finalize_batch_output(
    all_transcriptions: List[Dict[str, Any]],
    temp_file: Path,
    identifier: str,
    processing_settings: Dict[str, Any],
    postprocessing_config: Optional[Dict[str, Any]],
    output_format: str,
) -> None:
    """Display diagnostics, write output file, and optionally delete the temp file.

    Summarizes per-page errors and transcription-not-possible warnings to the
    console, builds the page list in sorted order, writes the final output via
    ``write_transcription_output``, and removes the temporary JSONL when
    configured to do so.
    """
    final_txt_path = temp_file.parent / f"{identifier}.txt"

    # Summarize per-page diagnostics to console
    error_entries = [e for e in all_transcriptions if e.get("error")]
    if error_entries:
        display_page_error_summary(error_entries)

    np_entries = [
        e
        for e in all_transcriptions
        if e.get("warning_type") == "transcription_not_possible"
    ]
    if np_entries:
        display_transcription_not_possible_summary(len(np_entries))

    # Build page dicts in sorted order
    pages = []
    for t in all_transcriptions:
        tx = t.get("transcription", "")
        info = t.get("image_info", {}) or {}
        pn = info.get("page_number")
        iname = info.get("image_name")
        pages.append({"text": tx, "page_number": pn, "image_name": iname})

    processing_success = False
    try:
        actual_path = write_transcription_output(
            pages,
            final_txt_path,
            output_format=output_format,
            postprocess=bool(postprocessing_config),
            postprocessing_config=postprocessing_config,
        )
        logger.info(
            f"All batches for {temp_file.name} processed and saved to {actual_path}"
        )
        print_success(
            f"Processed all batches for {temp_file.name}. Results saved to "
            f"{actual_path.name}"
        )
        processing_success = True
    except (OSError, ValueError, TypeError) as e:
        logger.exception(
            f"Error writing final output for {temp_file.name}: {e}"
        )
        print_error(
            f"Failed to write final output for {temp_file.name}: {e}"
        )

    # Delete temporary JSONL file if we successfully wrote the final output
    if processing_success and not processing_settings.get(
        "retain_temporary_jsonl", True
    ):
        try:
            temp_file.unlink()
            logger.info(f"Deleted temporary file after processing: {temp_file}")
            print_info(f"Deleted temporary file: {temp_file.name}")
        except OSError as e:
            logger.exception(f"Error deleting temporary file {temp_file}: {e}")
            print_error(
                f"Could not delete temporary file {temp_file.name}: {e}"
            )
