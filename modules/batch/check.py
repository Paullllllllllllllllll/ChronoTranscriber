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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from modules.batch.backends.factory import supports_batch
from modules.batch.mapping import extract_custom_id_mapping

# Re-export results submodule functions for backward compatibility
from modules.batch.results import (  # noqa: F401
    _download_and_parse_openai_results,
    _finalize_batch_output,
    _process_non_openai_batch,
    _sort_transcriptions,
)

# Re-export status submodule functions for backward compatibility
from modules.batch.status import (  # noqa: F401
    _check_openai_batch_status,
    _parse_temp_file_metadata,
    _recover_batch_ids,
    diagnose_api_issues,
    load_config,
)
from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.llm.openai_sdk_utils import list_all_batches
from modules.ui import (  # noqa: F401
    print_error,
    print_info,
    print_success,
    print_warning,
)
from modules.ui.batch_display import (
    display_batch_processing_progress,
    display_batch_summary,
)

logger = setup_logger(__name__)


@dataclass
class BatchCheckStats:
    """Per-job outcome counts for a batch finalization run (CT-4).

    Each batched temp JSONL counts once: ``finalized`` when its output file
    was written, ``failed`` when any of its batches reached a terminal
    failure (or the batch list could not be retrieved), ``pending``
    otherwise (still running or retryable download problem).

    ``had_failure`` preserves the previous boolean return contract: exit
    non-zero when any batch reached a terminal failure (CLI agent contract).
    """

    finalized: int = 0
    pending: int = 0
    failed: int = 0

    @property
    def had_failure(self) -> bool:
        return self.failed > 0

    def merge(self, other: BatchCheckStats) -> None:
        self.finalized += other.finalized
        self.pending += other.pending
        self.failed += other.failed


def process_all_batches(
    root_folder: Path,
    processing_settings: dict[str, Any],
    client: OpenAI | None = None,
    postprocessing_config: dict[str, Any] | None = None,
    output_format: str = "txt",
) -> BatchCheckStats:
    """Finalize all completed batch jobs for a given root folder.

    Scans for ``.jsonl`` files (both legacy ``*_transcription.jsonl`` and new format),
    retrieves batch summaries, safeguards ordering using custom_id metadata, and writes
    a final ``.txt`` output when all parts are available.

    Returns:
        A :class:`BatchCheckStats` with finalized/pending/failed counts;
        ``stats.had_failure`` is True if any batch reached a terminal failure
        (failed/expired/cancelled), so callers can surface a non-zero exit
        code (CLI agent contract).
    """
    stats = BatchCheckStats()
    print_info(f"Scanning directory '{root_folder}' for temporary batch files...")
    # Search for both new format (*.jsonl) and legacy format (*_transcription.jsonl)
    temp_files = list(root_folder.rglob("*.jsonl"))
    if not temp_files:
        print_info(f"No temporary batch files found in {root_folder}.")
        logger.info(f"No temporary batch files found in {root_folder}.")
        return stats

    # OpenAI's batch list is fetched lazily: only when at least one temp file
    # resolves to an OpenAI batch. This lets Anthropic/Google-only setups (with
    # no OpenAI credentials) finalize their batches, and confines any OpenAI-list
    # failure to the OpenAI files rather than aborting the whole run (B4).
    batch_dict: dict[str, dict[str, Any]] = {}
    openai_listing_done = False
    openai_listing_ok = False

    def ensure_openai_batches() -> bool:
        """List OpenAI batches on first need; return False if unavailable."""
        nonlocal client, batch_dict, openai_listing_done, openai_listing_ok
        if openai_listing_done:
            return openai_listing_ok
        openai_listing_done = True
        if client is None:
            try:
                client = OpenAI()
            except (OpenAIError, OSError, ValueError, TypeError) as e:
                print_error(f"OpenAI client unavailable: {e}")
                logger.exception(f"Could not construct OpenAI client: {e}")
                return False
        print_info("Retrieving list of submitted batches from OpenAI...")
        try:
            batches = list_all_batches(client)
        except (OpenAIError, OSError, ValueError, TypeError) as e:
            print_error(f"Failed to retrieve batches from OpenAI: {e}")
            logger.exception(f"Error retrieving batches: {e}")
            return False
        batch_dict = {
            str(b.get("id")): b for b in batches if isinstance(b, dict) and b.get("id")
        }
        # Display batch summary (handles dicts or SDK objects)
        display_batch_summary(batches)
        openai_listing_ok = True
        return True

    # Process each temporary file
    for temp_file in temp_files:
        identifier = temp_file.stem.replace("_transcription", "")

        # --- 1. Parse temp file metadata ---
        meta = _parse_temp_file_metadata(temp_file)
        batch_ids: set[str] = meta["batch_ids"]
        batch_provider: str = meta["batch_provider"]
        batch_tracking_records: list[dict[str, Any]] = meta["batch_tracking_records"]
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
                    f"{temp_file.name} has a batch_session marker but no batch IDs;"
                    f" skipping. Use 'main/repair_transcriptions.py' if needed."
                )
            elif batch_ids:
                n = len(batch_ids)
                print_warning(
                    f"{temp_file.name} contains {n} batch_tracking entries but no"
                    f" batch_session marker; skipping as non-batched."
                )
            elif has_batch_metadata or has_batch_request:
                print_info(
                    f"{temp_file.name} has batch-like metadata but no"
                    f" batch_session marker; treating as non-batched and skipping."
                )
            else:
                print_info(
                    f"{temp_file.name} has no batch markers;"
                    f" treating as non-batched and skipping."
                )
            continue

        if not batch_ids:
            print_warning(
                f"No batch IDs found in {temp_file.name}. This file appears to be"
                f" batched but missing tracking entries. Use"
                f" 'main/repair_transcriptions.py' if you need to reconstruct outputs."
            )
            continue

        # --- 4. Route non-OpenAI providers to backend abstraction ---
        if batch_provider != "openai" and supports_batch(batch_provider):
            outcome = _process_non_openai_batch(
                temp_file=temp_file,
                batch_ids=batch_ids,
                batch_provider=batch_provider,
                batch_tracking_records=batch_tracking_records,
                processing_settings=processing_settings,
                postprocessing_config=postprocessing_config,
                output_format=output_format,
            )
            if outcome == "finalized":
                stats.finalized += 1
            elif outcome == "failed":
                stats.failed += 1
            else:
                stats.pending += 1
            continue

        # --- 5. Check OpenAI batch status ---
        # Lazily fetch the OpenAI batch list; if it cannot be retrieved, fail
        # only this OpenAI file instead of the whole run (B4).
        if not ensure_openai_batches():
            stats.failed += 1
            continue
        assert client is not None  # ensured by ensure_openai_batches()

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
                stats.failed += 1
                print_warning(
                    f"{failed_count} batches have failed."
                    f" Check the OpenAI dashboard for details."
                )
            else:
                stats.pending += 1
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
                f"Failed to process all batches for {temp_file.name}."
                f" Skipping output writing."
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
            stats.finalized += 1
        else:
            # Download/extraction problem with completed batches: retryable,
            # counted as pending (matches the previous exit-code semantics).
            stats.pending += 1

    print_info(f"Completed processing batches in directory: {root_folder}")
    logger.info(f"Batch results processing complete for directory: {root_folder}")
    return stats


def run_batch_finalization(
    run_diagnostics: bool = True,
    custom_directory: Path | None = None,
    output_format: str = "txt",
) -> BatchCheckStats:
    """High-level entrypoint used by the CLI to finalize batch results.

    Args:
        run_diagnostics: Whether to run API diagnostics before processing
        custom_directory: Optional custom directory to scan instead of config defaults
        output_format: Output file format (``"txt"``, ``"md"``, or ``"json"``)

    Returns:
        Aggregated :class:`BatchCheckStats` across all scanned directories;
        ``stats.had_failure`` is True if any directory reported a terminal
        batch failure.
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

    # Construct the OpenAI client defensively: an Anthropic/Google-only setup
    # may have no OpenAI credentials, in which case process_all_batches builds
    # one lazily only if an OpenAI batch is actually encountered (B4).
    client: OpenAI | None
    try:
        client = OpenAI()
    except (OpenAIError, OSError, ValueError, TypeError) as e:
        logger.info(
            "OpenAI client not available up front (%s); "
            "will construct lazily if an OpenAI batch is found.",
            e,
        )
        client = None

    if run_diagnostics:
        diagnose_api_issues()

    stats = BatchCheckStats()
    for directory in scan_dirs:
        stats.merge(
            process_all_batches(
                directory,
                processing_settings,
                client,
                postprocessing_config,
                output_format=output_format,
            )
        )
    print_info("Batch results processing complete across all directories.")
    logger.info("Batch results processing complete across all directories.")
    return stats
