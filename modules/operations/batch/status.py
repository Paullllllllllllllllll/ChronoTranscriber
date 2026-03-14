"""Batch status checking and metadata parsing.

Provides functions to parse JSONL batch artifacts, detect providers, recover
missing batch IDs from debug artifacts, check batch completion status via the
OpenAI API, and run quick API diagnostics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI, OpenAIError

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.io.directory_utils import collect_scan_directories
from modules.llm.batch.batch_utils import diagnose_batch_failure
from modules.llm.openai_sdk_utils import list_all_batches, sdk_to_dict
from modules.ui import print_error, print_info, print_warning

logger = setup_logger(__name__)


def load_config() -> Tuple[List[Path], Dict[str, Any], Dict[str, Any]]:
    """Load YAML configuration and return (scan_dirs, processing_settings, postprocessing_config).

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
) -> Dict[str, Any]:
    """Parse a single JSONL temp file and extract batch-related metadata.

    Reads the file line-by-line, collecting batch tracking records, session
    markers, image metadata counts, and provider information.

    Returns a dict with keys:
        batch_ids, batch_provider, batch_tracking_records, has_batch_session,
        has_batch_request, has_batch_metadata, image_metadata_count,
        batch_request_count, batch_session_statuses.
    """
    batch_ids: Set[str] = set()
    batch_provider: Optional[str] = None
    batch_tracking_records: List[Dict[str, Any]] = []
    has_batch_session = False
    has_batch_request = False
    has_batch_metadata = False
    image_metadata_count = 0
    batch_request_count = 0
    batch_session_statuses: Set[str] = set()

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
                        batch_session_statuses.add(status_val.lower().strip())
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
    }


def _recover_batch_ids(
    temp_file: Path,
    identifier: str,
    batch_ids: Set[str],
    processing_settings: Dict[str, Any],
) -> Set[str]:
    """Attempt to recover missing batch IDs from the debug artifact file.

    Looks for a ``<identifier>_batch_submission_debug.json`` file alongside the
    temp file, reads any batch IDs stored there, and optionally persists them
    back into the JSONL for future runs.

    Returns the (potentially augmented) *batch_ids* set.
    """
    if batch_ids:
        return batch_ids

    debug_artifact = temp_file.parent / f"{identifier}_batch_submission_debug.json"
    if not debug_artifact.exists():
        return batch_ids

    try:
        dbg = json.loads(debug_artifact.read_text(encoding="utf-8"))
        dbg_ids = [bid for bid in (dbg.get("batch_ids") or []) if isinstance(bid, str)]
        to_add = [bid for bid in dbg_ids if bid not in batch_ids]
        if to_add:
            print_info(
                f"Recovered {len(to_add)} missing batch id(s) for {temp_file.name} from debug artifact."
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
                    print_info(
                        f"Persisted {len(to_add)} recovered batch id(s) into {temp_file.name}."
                    )
                except OSError as pe:
                    logger.warning(
                        "Failed to persist recovered batch ids for %s: %s", temp_file.name, pe
                    )
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as de:
        logger.warning("Failed to read debug artifact %s: %s", debug_artifact.name, de)

    return batch_ids


def _check_openai_batch_status(
    batch_ids: Set[str],
    batch_dict: Dict[str, Dict[str, Any]],
    client: OpenAI,
) -> Tuple[bool, List[str], int, int]:
    """Check OpenAI batch status for all *batch_ids*.

    Attempts to retrieve any batch not already present in *batch_dict* from
    the API, then classifies each batch as completed, failed, or in-progress.

    Returns:
        (all_completed, missing_batches, completed_count, failed_count)
    """
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
            except (OpenAIError, OSError, ValueError, TypeError) as e:
                all_completed = False
                missing_batches.append(batch_id)
                diagnosis = diagnose_batch_failure(batch_id, client)
                logger.warning(
                    f"Batch ID {batch_id} not found in OpenAI batches. {diagnosis} ({e})"
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

    return all_completed, missing_batches, completed_count, failed_count


def diagnose_api_issues() -> None:
    """Print quick diagnostics for API key presence and SDK connectivity."""
    print_info("\n=== API Issue Diagnostics ===")

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print_error("No OpenAI API key found in environment variables")
    else:
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
