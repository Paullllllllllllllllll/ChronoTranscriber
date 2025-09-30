"""Batch processing utilities for OpenAI Batch API operations.

Provides diagnostic tools and metadata extraction for batch job management.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from modules.llm.openai_sdk_utils import sdk_to_dict

logger = logging.getLogger(__name__)


def diagnose_batch_failure(batch_id: str, client: Any) -> str:
    """
    Diagnose a batch failure by retrieving its status from the OpenAI API.

    Args:
        batch_id: The ID of the batch to diagnose.
        client: An initialized OpenAI client instance.

    Returns:
        A human-readable diagnostic message about the batch status or error.
    """
    try:
        batch_obj = client.batches.retrieve(batch_id)
        batch = sdk_to_dict(batch_obj)
        status = str(batch.get("status", "")).lower()

        if status == "failed":
            return (
                f"Batch {batch_id} failed. Check your OpenAI dashboard for specific error details."
            )
        elif status == "cancelled":
            return f"Batch {batch_id} was cancelled."
        elif status == "expired":
            return f"Batch {batch_id} expired (not completed within 24 hours)."
        else:
            return f"Batch {batch_id} has status '{status}'."
    except Exception as e:
        error_message = str(e).lower()
        if "not found" in error_message:
            return (
                f"Batch {batch_id} not found in OpenAI. It may have been deleted or belong to a different API key."
            )
        elif "unauthorized" in error_message:
            return "API key unauthorized. Check your OpenAI API key permissions."
        elif "quota" in error_message:
            return "API quota exceeded. Check your usage limits."
        else:
            return f"Error checking batch {batch_id}: {e}"


def extract_custom_id_mapping(
    temp_file: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Extract mapping between custom_ids and image information from a local JSONL file.

    Scans for either "batch_request" (from batching.py) or "image_metadata" (from
    workflow.py) lines and builds a map of custom_id -> image_info, and a separate
    ordering map of custom_id -> order_index.

    Args:
        temp_file: Path to the temporary transcription JSONL file.

    Returns:
        Tuple of (custom_id -> image_info dict, custom_id -> order_index dict).
    """
    custom_id_map: Dict[str, Dict[str, Any]] = {}
    batch_order: Dict[str, int] = {}

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
                            if "order_index" in image_info:
                                batch_order[custom_id] = image_info["order_index"]

                    # Process image_metadata records from workflow.py
                    elif "image_metadata" in record:
                        metadata = record["image_metadata"]
                        custom_id = metadata.get("custom_id")
                        if custom_id:
                            custom_id_map[custom_id] = metadata
                            if "order_index" in metadata:
                                batch_order[custom_id] = metadata["order_index"]

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error("Error extracting custom_id mapping from %s: %s", temp_file, e)

    return custom_id_map, batch_order
