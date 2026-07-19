"""Batch status checking and metadata parsing.

Provides functions to parse JSONL batch artifacts, detect providers, recover
missing batch IDs from debug artifacts, check batch completion status via the
OpenAI API, and run quick API diagnostics.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from modules.batch.mapping import diagnose_batch_failure
from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.infra.paths import collect_scan_directories
from modules.llm.openai_sdk_utils import sdk_to_dict
from modules.ui import print_error, print_info

logger = setup_logger(__name__)


def load_config() -> tuple[list[Path], dict[str, Any], dict[str, Any]]:
    """Load YAML configuration and return scan_dirs, processing_settings,
    and postprocessing_config.

    Ensures file paths exist (creates input/output directories on demand).
    """
    config_service = get_config_service()
    paths_config = config_service.get_paths_config()
    image_processing_config = config_service.get_image_processing_config()

    processing_settings = paths_config.get("general", {})
    postprocessing_config = image_processing_config.get("postprocessing", {})

    # Use centralized directory collection utility
    scan_dirs = collect_scan_directories(paths_config)

    return scan_dirs, processing_settings, postprocessing_config


def _parse_temp_file_metadata(
    temp_file: Path,
) -> dict[str, Any]:
    """Parse a single JSONL temp file and extract batch-related metadata.

    Reads the file line-by-line, collecting batch tracking records, session
    markers, image metadata counts, and provider information.

    Returns a dict with keys:
        batch_ids, batch_provider, batch_tracking_records, has_batch_session,
        has_batch_request, has_batch_metadata, image_metadata_count,
        batch_request_count, batch_session_statuses,
        post_finalize_batch_ids, post_finalize_tracking_records.

    ``post_finalize_batch_ids``/``post_finalize_tracking_records`` cover only
    batch_tracking records appended AFTER the last ``finalized`` session
    marker (a resubmission of an already-finalized item). Callers use them to
    process the new submission instead of skipping the file as finalized —
    otherwise the freshly paid batch results would be unreachable forever.
    """
    batch_ids: set[str] = set()
    batch_provider: str | None = None
    batch_tracking_records: list[dict[str, Any]] = []
    has_batch_session = False
    has_batch_request = False
    has_batch_metadata = False
    image_metadata_count = 0
    batch_request_count = 0
    batch_session_statuses: set[str] = set()
    post_finalize_batch_ids: set[str] = set()
    post_finalize_tracking_records: list[dict[str, Any]] = []
    finalized_seen = False

    with temp_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)

                if "batch_tracking" in record:
                    tracking = record["batch_tracking"]
                    batch_id = tracking.get("batch_id")
                    if batch_id:
                        batch_ids.add(batch_id)
                        batch_tracking_records.append(tracking)
                        if finalized_seen:
                            post_finalize_batch_ids.add(batch_id)
                            post_finalize_tracking_records.append(tracking)
                        # Extract provider if present (new format)
                        if tracking.get("provider"):
                            batch_provider = tracking["provider"]

                elif "batch_session" in record and isinstance(
                    record["batch_session"], dict
                ):
                    has_batch_session = True
                    session = record["batch_session"]
                    status_val = session.get("status")
                    if isinstance(status_val, str):
                        status_norm = status_val.lower().strip()
                        batch_session_statuses.add(status_norm)
                        if status_norm == "finalized":
                            # Only records after the LAST finalized marker
                            # count as a fresh resubmission.
                            finalized_seen = True
                            post_finalize_batch_ids = set()
                            post_finalize_tracking_records = []
                    # Extract provider from session if present
                    if session.get("provider"):
                        batch_provider = session["provider"]

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

            except json.JSONDecodeError:
                continue

    # Default to OpenAI for backward compatibility with old batch files
    if batch_provider is None:
        batch_provider = "openai"

    return {
        "batch_ids": batch_ids,
        "batch_provider": batch_provider,
        "batch_tracking_records": batch_tracking_records,
        "has_batch_session": has_batch_session,
        "has_batch_request": has_batch_request,
        "has_batch_metadata": has_batch_metadata,
        "image_metadata_count": image_metadata_count,
        "batch_request_count": batch_request_count,
        "batch_session_statuses": batch_session_statuses,
        "post_finalize_batch_ids": post_finalize_batch_ids,
        "post_finalize_tracking_records": post_finalize_tracking_records,
    }


def _recover_batch_ids(
    temp_file: Path,
    identifier: str,
    batch_ids: set[str],
    processing_settings: dict[str, Any],
    tracking_records: list[dict[str, Any]] | None = None,
) -> set[str]:
    """Attempt to recover missing batch IDs from the debug artifact file.

    Looks for a ``<identifier>_batch_submission_debug.json`` file alongside the
    temp file, reads any batch IDs stored there, and optionally persists them
    back into the JSONL for future runs.

    Recovery runs even when *batch_ids* is non-empty: a crash can lose the
    tracking write for one part of a multi-part submission while another part
    was recorded, and finalizing from the tracked parts alone would brand the
    untracked part's pages as permanent errors while its paid results expire
    unread.

    When *tracking_records* is given, reconstructed tracking records
    (including any per-part provider/metadata stored in the debug artifact,
    e.g. the ``custom_id_map`` Google inline results need for correlation)
    are appended to it so the current run can use them, not just future runs.

    Returns the (potentially augmented) *batch_ids* set.
    """
    debug_artifact = temp_file.parent / f"{identifier}_batch_submission_debug.json"
    if not debug_artifact.exists():
        return batch_ids

    try:
        dbg = json.loads(debug_artifact.read_text(encoding="utf-8"))
        dbg_ids = [bid for bid in (dbg.get("batch_ids") or []) if isinstance(bid, str)]
        provider = dbg.get("provider") if isinstance(dbg.get("provider"), str) else None
        parts_info: dict[str, dict[str, Any]] = {}
        for part in dbg.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("batch_id"), str):
                parts_info[part["batch_id"]] = part
        to_add = [bid for bid in dbg_ids if bid not in batch_ids]
        if to_add:
            print_info(
                f"Recovered {len(to_add)} missing batch id(s) for"
                f" {temp_file.name} from debug artifact."
            )
            recovered_records: list[dict[str, Any]] = []
            for bid in to_add:
                batch_ids.add(bid)
                info = parts_info.get(bid, {})
                tracking: dict[str, Any] = {
                    "batch_id": bid,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "batch_file": str(bid),
                }
                part_provider = info.get("provider") or provider
                if isinstance(part_provider, str) and part_provider:
                    tracking["provider"] = part_provider
                metadata = info.get("metadata")
                if isinstance(metadata, dict):
                    tracking["metadata"] = metadata
                recovered_records.append(tracking)
            if tracking_records is not None:
                tracking_records.extend(recovered_records)
            # Best-effort persist into the JSONL so future runs have them
            persist = bool(processing_settings.get("persist_recovered_batch_ids", True))
            if persist:
                try:
                    with temp_file.open("a", encoding="utf-8") as wf:
                        for tracking in recovered_records:
                            rec = {"batch_tracking": tracking}
                            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print_info(
                        f"Persisted {len(to_add)} recovered batch id(s)"
                        f" into {temp_file.name}."
                    )
                except OSError as pe:
                    logger.warning(
                        "Failed to persist recovered batch ids for %s: %s",
                        temp_file.name,
                        pe,
                    )
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as de:
        logger.warning("Failed to read debug artifact %s: %s", debug_artifact.name, de)

    return batch_ids


def _check_openai_batch_status(
    batch_ids: set[str],
    batch_dict: dict[str, dict[str, Any]],
    client: OpenAI,
) -> tuple[bool, list[str], int, int]:
    """Check OpenAI batch status for all *batch_ids*.

    Attempts to retrieve any batch not already present in *batch_dict* from
    the API, then classifies each batch as completed, failed, or in-progress.

    Returns:
        (all_completed, missing_batches, completed_count, failed_count)
    """
    all_completed = True
    missing_batches: list[str] = []
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
                    logger.warning("Batch ID %s retrieval returned no object", batch_id)
                    continue
            except (OpenAIError, OSError, ValueError, TypeError) as e:
                all_completed = False
                missing_batches.append(batch_id)
                diagnosis = diagnose_batch_failure(batch_id, client)
                logger.warning(
                    "Batch ID %s not found in OpenAI batches. %s (%s)",
                    batch_id,
                    diagnosis,
                    e,
                )
                continue

        batch = batch_dict[batch_id]
        status = str(batch.get("status", "")).lower()

        if status == "completed":
            completed_count += 1
        elif status in ("failed", "expired", "cancelled", "canceled"):
            # Terminal non-success states. Previously expired/cancelled fell
            # into the else branch and were reported as "not yet completed"
            # forever (B6); they never progress, so classify them as failed.
            failed_count += 1
            all_completed = False
            logger.warning(
                "Batch %s reached terminal state '%s'; it will not complete.",
                batch_id,
                status,
            )
        else:
            all_completed = False
            logger.info(f"Batch {batch_id} has status '{status}' - not yet completed.")

    return all_completed, missing_batches, completed_count, failed_count


def diagnose_api_issues() -> None:
    """Print quick diagnostics for API key presence and SDK connectivity."""
    print_info("\n=== API Issue Diagnostics ===")

    # Check API key (honors the optional api_keys_config.yaml remap)
    from modules.llm.providers.factory import (
        ProviderType,
        resolve_api_key_env_var,
    )

    env_var = resolve_api_key_env_var(ProviderType.OPENAI) or "OPENAI_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        # These diagnostics probe only the OpenAI API. On an Anthropic- or
        # Google-only setup no OpenAI key is present, so reporting an "error"
        # and running live probes (which would fail) is misleading. Report it as
        # informational and skip the probes to keep the diagnostic cheap.
        print_info(
            "No OpenAI API key found; skipping OpenAI connectivity checks."
            " (These diagnostics currently cover only the OpenAI API.)"
        )
        return
    print_info("OpenAI API key present: True")

    # Check for common model issues using SDK
    try:
        client = OpenAI()
        models_page = client.models.list()
        models_data: Iterable[Any] = getattr(models_page, "data", None) or []
        total_models = len(list(models_data))
        print_info(f"API Connection successful: {total_models} models available")
    except Exception as e:
        print_error(f"Failed to list models: {e}")

    # Check for batch issues
    try:
        client = OpenAI()
        _ = client.batches.list(limit=1)
        print_info("Batch API access successful")
    except Exception as e:
        print_error(f"Batch API access failed: {e}")
