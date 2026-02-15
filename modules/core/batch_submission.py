"""Batch submission orchestration extracted from WorkflowManager.

Handles system-prompt loading, context resolution, batch-request building,
backend submission, and tracking/debug artifact writing.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import aiofiles

from modules.infra.logger import setup_logger
from modules.llm.batch.batching import get_batch_chunk_size
from modules.llm.batch.backends import get_batch_backend, BatchRequest, BatchHandle
from modules.llm.batch.backends.factory import supports_batch
from modules.ui import print_info, print_warning, print_error, print_success

if TYPE_CHECKING:
    from modules.ui.core import UserConfiguration

logger = setup_logger(__name__)


async def submit_batch(
    image_files: List[Path],
    temp_jsonl_path: Path,
    parent_folder: Path,
    source_name: str,
    model_config: Dict[str, Any],
    user_config: "UserConfiguration",
) -> Optional[BatchHandle]:
    """Submit a batch using the provider-agnostic batch backend.

    Args:
        image_files: List of image paths to process.
        temp_jsonl_path: Path to the temp JSONL file for tracking.
        parent_folder: Parent folder for debug artifacts.
        source_name: Name of the source (PDF or folder name).
        model_config: Model configuration dictionary.
        user_config: Current user configuration.

    Returns:
        BatchHandle if submission successful, None otherwise.
    """
    tm = model_config.get("transcription_model", {})
    provider = tm.get("provider", "openai")

    # Check if provider supports batch processing
    if not supports_batch(provider):
        print_warning(
            f"Provider '{provider}' does not support batch processing. "
            f"Falling back to synchronous mode."
        )
        return None

    total_images = len(image_files)
    chunk_size = get_batch_chunk_size()
    expected_batches = math.ceil(total_images / max(1, chunk_size))

    # Telemetry
    logger.info(
        "[Batch] Preparing submission: provider=%s, images=%d, chunk_size=%d",
        provider, total_images, chunk_size,
    )
    print_info(
        f"Batch telemetry -> provider={provider}, images={total_images}, chunk_size={chunk_size}"
    )
    print_info(f"Submitting batch job for {total_images} images...")

    # Early marker
    try:
        async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
            await f.write(json.dumps({"batch_session": {"status": "submitting", "provider": provider}}) + "\n")
    except Exception:
        logger.warning("Could not write early batch_session marker")

    # Record image metadata
    try:
        async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
            for idx, img_path in enumerate(image_files):
                image_record = {
                    "image_metadata": {
                        "pre_processed_image": str(img_path),
                        "image_name": img_path.name,
                        "order_index": idx,
                        "custom_id": f"req-{idx + 1}"
                    }
                }
                await f.write(json.dumps(image_record) + "\n")
    except Exception as e:
        logger.warning("Failed writing image_metadata before batch submit: %s", e)

    # Build batch requests
    batch_requests = [
        BatchRequest(
            custom_id=f"req-{idx + 1}",
            image_path=img_path,
            order_index=idx,
            image_info={
                "image_name": img_path.name,
                "order_index": idx,
                "page_number": idx + 1,
            },
        )
        for idx, img_path in enumerate(image_files)
    ]

    # Load system prompt
    system_prompt = _load_system_prompt()

    # Load additional context
    additional_context = _resolve_additional_context(user_config, parent_folder)

    # Submit via backend
    try:
        backend = get_batch_backend(provider)
        handle = await asyncio.to_thread(
            backend.submit_batch,
            batch_requests,
            tm,
            system_prompt=system_prompt,
            schema_path=user_config.selected_schema_path,
            additional_context=additional_context,
        )
    except Exception as e:
        logger.exception(f"Batch submission failed for {source_name}: {e}")
        print_error(
            f"Batch submission failed for {source_name}. "
            f"Falling back to synchronous processing."
        )
        return None

    # Write tracking record with provider info
    try:
        async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
            tracking_record = {
                "batch_tracking": {
                    "batch_id": handle.batch_id,
                    "provider": handle.provider,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "metadata": handle.metadata,
                }
            }
            await f.write(json.dumps(tracking_record) + "\n")
    except Exception as e:
        logger.warning("Post-submission tracking write failed for %s: %s", source_name, e)

    # Write debug artifact
    try:
        debug_payload = {
            "source": source_name,
            "provider": provider,
            "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "expected_batches": int(expected_batches),
            "batch_ids": [handle.batch_id],
            "total_images": int(total_images),
            "chunk_size": int(chunk_size),
        }
        debug_path = parent_folder / f"{source_name}_batch_submission_debug.json"
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding='utf-8')
    except Exception as e:
        logger.warning("Failed to write batch submission debug artifact for %s: %s", source_name, e)

    # Mark submitted
    try:
        async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
            await f.write(json.dumps({"batch_session": {"status": "submitted", "provider": provider}}) + "\n")
    except Exception:
        logger.warning("Could not write submitted batch_session marker")

    print_success(f"Batch submitted for '{source_name}'.")
    print_info("The batch will be processed asynchronously. Use 'check_batches.py' to monitor status.")

    return handle


def _load_system_prompt() -> str:
    """Load the system prompt from the configured or default path."""
    from modules.config.config_loader import PROJECT_ROOT
    from modules.config.service import get_config_service

    pcfg = get_config_service().get_paths_config()
    general = pcfg.get("general", {})
    override_prompt = general.get("transcription_prompt_path")
    system_prompt_path = (
        Path(override_prompt)
        if override_prompt
        else (PROJECT_ROOT / "system_prompt" / "system_prompt.txt")
    )
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt missing: {system_prompt_path}")
    return system_prompt_path.read_text(encoding="utf-8").strip()


def _resolve_additional_context(
    user_config: "UserConfiguration",
    parent_folder: Path,
) -> Optional[str]:
    """Resolve additional context from explicit path or hierarchical resolution."""
    if user_config.additional_context_path:
        ctx_path = Path(user_config.additional_context_path)
        if ctx_path.exists():
            return ctx_path.read_text(encoding="utf-8").strip()
    elif getattr(user_config, 'use_hierarchical_context', True):
        from modules.llm.context_utils import resolve_context_for_folder
        context_content, context_path = resolve_context_for_folder(parent_folder)
        if context_content:
            logger.info(f"Using resolved context from: {context_path}")
            return context_content
    return None
