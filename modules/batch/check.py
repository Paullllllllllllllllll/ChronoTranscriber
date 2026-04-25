"""Batch finalization operations.

This module orchestrates the batch finalization workflow: scanning local JSONL
artifacts, verifying batch completion, downloading results, merging outputs in
page order, and writing final transcription files.

The heavy lifting is delegated to two submodules:

- ``status`` -- status checking, metadata parsing, batch ID recovery
- ``results`` -- result downloading, transcription sorting, output finalization

This module retains the public API surface (``process_all_batches``,
``run_batch_finalization``, ``load_config``, etc.) so that existing callers and
tests continue to work without changes.

Multi-provider support:
- OpenAI: Uses legacy direct SDK calls for backward compatibility
- Anthropic: Uses BatchBackend abstraction via Message Batches API
- Google: Uses BatchBackend abstraction via Gemini Batch API
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI, OpenAIError

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.batch.mapping import extract_custom_id_mapping
from modules.batch.backends.factory import supports_batch
from modules.llm.openai_sdk_utils import list_all_batches
from modules.ui import print_error, print_info, print_success, print_warning
from modules.ui.batch_display import (
    display_batch_processing_progress,
    display_batch_summary,
)

# Re-export status submodule functions for backward compatibility
from modules.batch.status import (  # noqa: F401
    load_config,
    _parse_temp_file_metadata,
    _recover_batch_ids,
    _check_openai_batch_status,
    diagnose_api_issues,
)

# Re-export results submodule functions for backward compatibility
from modules.batch.results import (  # noqa: F401
    _process_non_openai_batch,
    _download_and_parse_openai_results,
    _sort_transcriptions,
    _finalize_batch_output,
)

logger = setup_logger(__name__)


def process_all_batches(
    root_folder: Path,
    processing_settings: Dict[str, Any],
    client: OpenAI,
    postprocessing_config: Optional[Dict[str, Any]] = None,
    output_format: str = "txt",
) -> None:
    """Finalize all completed batch jobs for a given root folder.

    Scans for ``.jsonl`` files (both legacy ``*_transcription.jsonl`` and new format),
    retrieves batch summaries, safeguards ordering using custom_id metadata, and writes
    a final ``.txt`` output when all parts are available.
    """
    print_info(f"Scanning directory '{root_folder}' for temporary batch files...")
    # Search for both new format (*.jsonl) and legacy format (*_transcription.jsonl)
    temp_files = list(root_folder.rglob("*.jsonl"))
    if not temp_files:
        print_info(f"No temporary batch files found in {root_folder}.")
        logger.info(f"No temporary batch files found in {root_folder}.")
        return

    # Retrieve all batches from OpenAI
    print_info("Retrieving list of submitted batches from OpenAI...")
    try:
        batches = list_all_batches(client)
        batch_dict: Dict[str, Dict[str, Any]] = {
            str(b.get("id")): b for b in batches if isinstance(b, dict) and b.get("id")
        }
    except (OpenAIError, OSError, ValueError, TypeError) as e:
        print_error(f"Failed to retrieve batches from OpenAI: {e}")
        logger.exception(f"Error retrieving batches: {e}")
        return

    # Display batch summary (handles dicts or SDK objects)
    display_batch_summary(batches)

    # Process each temporary file
    for temp_file in temp_files:
        identifier = temp_file.stem.replace("_transcription", "")

        # --- 1. Parse temp file metadata ---
        meta = _parse_temp_file_metadata(temp_file)
        batch_ids: Set[str] = meta["batch_ids"]
        batch_provider: str = meta["batch_provider"]
        batch_tracking_records: List[Dict[str, Any]] = meta["batch_tracking_records"]
        has_batch_session: bool = meta["has_batch_session"]
        has_batch_request: bool = meta["has_batch_request"]
        has_batch_metadata: bool = meta["has_batch_metadata"]

        # --- 2. Recover missing batch IDs from debug artifact ---
        batch_ids = _recover_batch_ids(
            temp_file, identifier, batch_ids, processing_settings
        )

        # --- 3. Classification: is this a batched file? ---
        is_batched_file = has_batch_session and (
            bool(batch_ids) or has_batch_request or has_batch_metadata
        )

        if not is_batched_file:
            if has_batch_session and not batch_ids:
                print_warning(
                    f"{temp_file.name} has a batch_session marker but no batch IDs; skipping. "
                    f"Use 'main/repair_transcriptions.py' if needed."
                )
            elif batch_ids:
                print_warning(
                    f"{temp_file.name} contains {len(batch_ids)} batch_tracking entries but no "
                    f"batch_session marker; skipping as non-batched."
                )
            elif has_batch_metadata or has_batch_request:
                print_info(
                    f"{temp_file.name} has batch-like metadata but no batch_session marker; "
                    f"treating as non-batched and skipping."
                )
            else:
                print_info(
                    f"{temp_file.name} has no batch markers; treating as non-batched and skipping."
                )
            continue

        if not batch_ids:
            print_warning(
                f"No batch IDs found in {temp_file.name}. This file appears to be batched but "
                f"missing tracking entries. Use 'main/repair_transcriptions.py' if you need to "
                f"reconstruct outputs."
            )
            continue

        # --- 4. Route non-OpenAI providers to backend abstraction ---
        if batch_provider != "openai" and supports_batch(batch_provider):
            _process_non_openai_batch(
                temp_file=temp_file,
                batch_ids=batch_ids,
                batch_provider=batch_provider,
                batch_tracking_records=batch_tracking_records,
                processing_settings=processing_settings,
                postprocessing_config=postprocessing_config,
                output_format=output_format,
            )
            continue

        # --- 5. Check OpenAI batch status ---
        all_completed, missing_batches, completed_count, failed_count = (
            _check_openai_batch_status(batch_ids, batch_dict, client)
        )

        # Display progress information for this temp file (standardized UI helper)
        display_batch_processing_progress(
            temp_file=temp_file,
            batch_ids=list(batch_ids),
            completed_count=completed_count,
            missing_count=len(missing_batches),
        )

        if not all_completed:
            if failed_count > 0:
                print_warning(
                    f"{failed_count} batches have failed. Check the OpenAI dashboard for details."
                )
            continue

        # --- 6. Download and parse OpenAI results ---
        print_info(
            f"All batches for {temp_file.name} report 'completed'. "
            f"Attempting to download and process results..."
        )

        custom_id_map, batch_order = extract_custom_id_mapping(temp_file)

        all_transcriptions, all_completed = _download_and_parse_openai_results(
            batch_ids, batch_dict, client, custom_id_map, batch_order, temp_file
        )

        # --- 7. Validate results before writing ---
        can_write_output = True
        if not all_completed:
            print_warning(
                f"Failed to process all batches for {temp_file.name}. Skipping output writing."
            )
            can_write_output = False

        if not all_transcriptions:
            logger.warning(
                f"No transcriptions extracted for any batch in {temp_file.name}."
            )
            print_warning(
                f"No transcriptions extracted for any batch in {temp_file.name}. "
                f"Skipping this file."
            )
            can_write_output = False

        if can_write_output:
            # --- 8. Sort transcriptions ---
            _sort_transcriptions(all_transcriptions, batch_order, temp_file)

            # --- 9. Finalize output ---
            _finalize_batch_output(
                all_transcriptions,
                temp_file,
                identifier,
                processing_settings,
                postprocessing_config,
                output_format,
            )

    print_info(f"Completed processing batches in directory: {root_folder}")
    logger.info(f"Batch results processing complete for directory: {root_folder}")


def run_batch_finalization(
    run_diagnostics: bool = True,
    custom_directory: Path | None = None,
    output_format: str = "txt",
) -> None:
    """High-level entrypoint used by the CLI to finalize batch results.

    Args:
        run_diagnostics: Whether to run API diagnostics before processing
        custom_directory: Optional custom directory to scan instead of config defaults
        output_format: Output file format (``"txt"``, ``"md"``, or ``"json"``)
    """
    if custom_directory:
        # Use the specified directory instead of loading from config
        scan_dirs = [custom_directory]
        # Load minimal processing settings from config
        config_service = get_config_service()
        paths_config = config_service.get_paths_config()
        image_processing_config = config_service.get_image_processing_config()
        processing_settings = paths_config.get("general", {})
        postprocessing_config = image_processing_config.get("postprocessing", {})
    else:
        # Load standard configuration
        scan_dirs, processing_settings, postprocessing_config = load_config()

    client = OpenAI()

    if run_diagnostics:
        diagnose_api_issues()

    for directory in scan_dirs:
        process_all_batches(
            directory, processing_settings, client, postprocessing_config,
            output_format=output_format,
        )
    print_info("Batch results processing complete across all directories.")
    logger.info("Batch results processing complete across all directories.")
