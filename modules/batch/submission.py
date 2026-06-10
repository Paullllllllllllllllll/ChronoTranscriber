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
from typing import TYPE_CHECKING, Any

import aiofiles

from modules.batch.backends import BatchHandle, BatchRequest, get_batch_backend
from modules.batch.backends.factory import supports_batch
from modules.batch.requests import get_batch_chunk_size
from modules.infra.logger import setup_logger
from modules.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from modules.images.page_stream import PagePayload
    from modules.transcribe.user_config import UserConfiguration

logger = setup_logger(__name__)


async def submit_batch(
    payloads: list[PagePayload],
    temp_jsonl_path: Path,
    parent_folder: Path,
    source_name: str,
    model_config: dict[str, Any],
    user_config: UserConfiguration,
    file_provenance: dict[str, Any] | None = None,
) -> BatchHandle | None:
    """Submit a batch using the provider-agnostic batch backend.

    Args:
        payloads: In-memory page payloads (base64 JPEG + provenance) to
            process; no preprocessed image files exist on disk.
        temp_jsonl_path: Path to the temp JSONL file for tracking.
        parent_folder: Parent folder for debug artifacts.
        source_name: Name of the source (PDF or folder name).
        model_config: Model configuration dictionary.
        user_config: Current user configuration.
        file_provenance: Optional file-level provenance record to persist
            in the temp JSONL.

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

    total_images = len(payloads)
    chunk_size = get_batch_chunk_size()
    expected_batches = math.ceil(total_images / max(1, chunk_size))

    # Telemetry
    logger.info(
        "[Batch] Preparing submission: provider=%s, images=%d, chunk_size=%d",
        provider,
        total_images,
        chunk_size,
    )
    print_info(
        f"Batch telemetry -> provider={provider},"
        f" images={total_images}, chunk_size={chunk_size}"
    )
    print_info(f"Submitting batch job for {total_images} images...")

    # Early marker
    try:
        async with aiofiles.open(temp_jsonl_path, "a", encoding="utf-8") as f:
            await f.write(
                json.dumps(
                    {"batch_session": {"status": "submitting", "provider": provider}}
                )
                + "\n"
            )
            if file_provenance is not None:
                await f.write(json.dumps(file_provenance) + "\n")
    except Exception as e:
        logger.warning(
            "Could not write early batch_session marker: %s: %s", type(e).__name__, e
        )

    # Record image metadata (no preprocessed files exist; carry the source
    # reference and per-page provenance so repair can re-render).
    try:
        async with aiofiles.open(temp_jsonl_path, "a", encoding="utf-8") as f:
            for pos, payload in enumerate(payloads):
                image_record = {
                    "image_metadata": {
                        "pre_processed_image": None,
                        "image_name": payload.image_name,
                        "order_index": payload.index,
                        "custom_id": f"req-{pos + 1}",
                        "source_file": payload.source_file,
                        "page_index": payload.page_index,
                        "image_provenance": payload.provenance(),
                    }
                }
                await f.write(json.dumps(image_record) + "\n")
    except Exception as e:
        logger.warning("Failed writing image_metadata before batch submit: %s", e)

    # Build batch requests from in-memory payloads
    batch_requests = [
        BatchRequest(
            custom_id=f"req-{pos + 1}",
            image_base64=payload.base64,
            mime_type=payload.mime_type,
            order_index=payload.index,
            image_info={
                "image_name": payload.image_name,
                "order_index": payload.index,
                "page_number": payload.index + 1,
            },
        )
        for pos, payload in enumerate(payloads)
    ]

    # Load system prompt
    system_prompt = _load_system_prompt()

    # Load additional context
    additional_context = _resolve_additional_context(user_config, parent_folder)

    # Resolve context image
    ctx_image_url = _resolve_context_image(user_config, parent_folder)

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
            context_image_url=ctx_image_url,
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
        async with aiofiles.open(temp_jsonl_path, "a", encoding="utf-8") as f:
            tracking_record = {
                "batch_tracking": {
                    "batch_id": handle.batch_id,
                    "provider": handle.provider,
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "metadata": handle.metadata,
                }
            }
            await f.write(json.dumps(tracking_record) + "\n")
    except Exception as e:
        logger.warning(
            "Post-submission tracking write failed for %s: %s", source_name, e
        )

    # Write debug artifact
    try:
        debug_payload = {
            "source": source_name,
            "provider": provider,
            "submitted_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "expected_batches": int(expected_batches),
            "batch_ids": [handle.batch_id],
            "total_images": int(total_images),
            "chunk_size": int(chunk_size),
        }
        debug_path = parent_folder / f"{source_name}_batch_submission_debug.json"
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(
            "Failed to write batch submission debug artifact for %s: %s", source_name, e
        )

    # Mark submitted
    try:
        async with aiofiles.open(temp_jsonl_path, "a", encoding="utf-8") as f:
            await f.write(
                json.dumps(
                    {"batch_session": {"status": "submitted", "provider": provider}}
                )
                + "\n"
            )
    except Exception as e:
        logger.warning(
            "Could not write submitted batch_session marker: %s: %s",
            type(e).__name__,
            e,
        )

    print_success(f"Batch submitted for '{source_name}'.")
    print_info(
        "The batch will be processed asynchronously."
        " Use 'check_batches.py' to monitor status."
    )

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
        else (PROJECT_ROOT / "system_prompt" / "transcription_prompt_schema.txt")
    )
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt missing: {system_prompt_path}")
    return system_prompt_path.read_text(encoding="utf-8").strip()


def _resolve_additional_context(
    user_config: UserConfiguration,
    parent_folder: Path,
) -> str | None:
    """Resolve additional context from explicit path or hierarchical resolution."""
    if user_config.additional_context_path:
        ctx_path = Path(user_config.additional_context_path)
        if ctx_path.exists():
            return ctx_path.read_text(encoding="utf-8").strip()
    elif getattr(user_config, "use_hierarchical_context", True):
        from modules.config.context import resolve_context_for_folder

        context_content, context_path = resolve_context_for_folder(parent_folder)
        if context_content:
            logger.info(f"Using resolved context from: {context_path}")
            return context_content
    return None


def _resolve_context_image(
    user_config: UserConfiguration,
    parent_folder: Path,
) -> str | None:
    """Resolve a context image and encode it as a data URL.

    Returns a ``data:`` URL string or ``None``.
    """
    image_path: Path | None = None

    if user_config.additional_context_image_path:
        p = Path(user_config.additional_context_image_path)
        if p.exists():
            image_path = p
    elif getattr(user_config, "use_hierarchical_context", True):
        from modules.config.context import resolve_context_image_for_folder

        image_path = resolve_context_image_for_folder(parent_folder)

    if image_path is None:
        return None

    from modules.images.encoding import encode_image_to_data_url

    try:
        data_url = encode_image_to_data_url(image_path)
        logger.info("Using context image for batch: %s", image_path)
        return data_url
    except Exception as exc:
        logger.warning("Failed to encode context image %s: %s", image_path, exc)
        return None
