"""Image transcription pipeline extracted from WorkflowManager.

Handles resume-aware image filtering, concurrent transcription dispatch,
streaming JSONL writes, result ordering, and final output assembly.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiofiles

from modules.batch.jsonl import (
    ensure_resume_marker,
    extract_transcription_records,
    get_processed_image_names,
    read_jsonl_records,
    verify_resume_compatible,
)
from modules.images.page_stream import PagePayload
from modules.images.tesseract_runtime import perform_ocr
from modules.infra.concurrency import (
    run_concurrent_transcription_tasks,
    run_streaming_transcription_tasks,
)
from modules.infra.logger import setup_logger
from modules.llm import transcribe_image_with_llm
from modules.llm.response_parsing import extract_transcribed_text
from modules.postprocess.writer import write_transcription_output
from modules.ui import print_error, print_info, print_success, print_warning

logger = setup_logger(__name__)


async def transcribe_single_image(
    img_path: Path,
    transcriber: Any | None,
    method: str,
    tesseract_config: str = "--oem 3 --psm 6",
    order_index: int = 0,
) -> tuple[str, str, str | None, dict[str, Any] | None, int]:
    """Transcribe a single image file using either GPT or Tesseract OCR.

    Returns:
        Tuple of (image_path_str, image_name, text, raw_response, order_index).
    """
    image_name = img_path.name
    final_text: str | None = None
    try:
        if method == "gpt":
            if not transcriber:
                logger.error("No transcriber instance provided for GPT usage.")
                return (
                    str(img_path),
                    image_name,
                    f"[transcription error: {image_name}]",
                    None,
                    order_index,
                )
            result = await transcribe_image_with_llm(img_path, transcriber)
            logger.debug(f"LLM response for {img_path.name}: {result}")
            try:
                final_text = extract_transcribed_text(result, image_name)
            except Exception as e:
                logger.error(
                    "Error extracting transcription for %s: %s."
                    " Marking as transcription error.",
                    img_path.name,
                    e,
                )
                final_text = f"[transcription error: {image_name}]"
            return (str(img_path), image_name, final_text, result, order_index)
        elif method == "tesseract":
            final_text = perform_ocr(img_path, tesseract_config)
        else:
            logger.error(
                f"Unknown transcription method '{method}' for image {img_path.name}"
            )
            final_text = None
        return (str(img_path), image_name, final_text, None, order_index)
    except Exception as e:
        logger.exception(
            f"Error transcribing {img_path.name} with method '{method}': {e}"
        )
        return (
            str(img_path),
            image_name,
            f"[transcription error: {image_name}]",
            None,
            order_index,
        )


def _build_jsonl_record(
    result_tuple: tuple[str, str, str | None, dict[str, Any] | None, int],
    source_name: str,
    method: str,
    is_folder: bool,
    transcriber: Any | None,
    payload: PagePayload | None = None,
) -> dict[str, Any] | None:
    """Build a JSONL record dict from a transcription result tuple.

    When ``payload`` is given (streaming pipeline), no preprocessed image
    file exists on disk: ``pre_processed_image`` is None and the record
    carries the source reference and per-page provenance instead.

    Returns None if the result should be skipped (no text).
    """
    if not result_tuple or len(result_tuple) < 5:
        return None
    img_path_str, image_name, text_chunk, raw_response, order_index = result_tuple
    if text_chunk is None:
        return None

    # Build base record — only the source key differs between PDF and folder
    source_key = "folder_name" if is_folder else "file_name"
    record: dict[str, Any] = {
        source_key: source_name,
        "pre_processed_image": None if payload is not None else img_path_str,
        "image_name": image_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "method": method,
        "order_index": order_index,
        "text_chunk": text_chunk,
    }

    if payload is not None:
        record["source_file"] = payload.source_file
        record["page_index"] = payload.page_index
        record["image_provenance"] = payload.provenance()

    # Include raw_response and request_context only for GPT
    if method == "gpt" and raw_response is not None and transcriber is not None:
        try:
            ctx = {}
            extractor = getattr(transcriber, "extractor", None)
            if extractor is not None:
                ctx = {
                    "model": getattr(extractor, "model", None),
                    "service_tier": getattr(extractor, "service_tier", None),
                    "max_output_tokens": getattr(extractor, "max_output_tokens", None),
                    "temperature": getattr(extractor, "temperature", None),
                    "top_p": getattr(extractor, "top_p", None),
                    "presence_penalty": getattr(extractor, "presence_penalty", None),
                    "frequency_penalty": getattr(extractor, "frequency_penalty", None),
                    "stop": getattr(extractor, "stop", None),
                    "seed": getattr(extractor, "seed", None),
                    "reasoning": getattr(extractor, "reasoning", None),
                    "text": getattr(extractor, "text_params", None),
                    "detail": None,
                }
            record["request_context"] = ctx
            record["raw_response"] = raw_response
        except (AttributeError, TypeError, ValueError) as exc:
            # Diagnostic context is best-effort; never drop it silently, as
            # batch repair relies on raw_response/request_context downstream.
            logger.warning(
                "Failed to attach request_context/raw_response for %s: %s",
                image_name,
                exc,
            )

    return record


async def transcribe_payload(
    payload: PagePayload,
    transcriber: Any,
) -> tuple[Any, str, str | None, dict[str, Any] | None, int]:
    """Transcribe one in-memory page payload via the LLM provider.

    Returns the same 5-tuple shape as :func:`transcribe_single_image`, with
    the payload itself in the first slot (so JSONL writers can access
    provenance) and the absolute page index as ``order_index``.
    """
    image_name = payload.image_name
    try:
        result = await transcriber.transcribe_image_from_base64(
            payload.base64, payload.mime_type
        )
        logger.debug(f"LLM response for {image_name}: {result}")
        try:
            final_text: str | None = extract_transcribed_text(result, image_name)
        except Exception as e:
            logger.error(
                "Error extracting transcription for %s: %s."
                " Marking as transcription error.",
                image_name,
                e,
            )
            final_text = f"[transcription error: {image_name}]"
        return (payload, image_name, final_text, result, payload.index)
    except Exception as e:
        logger.exception(f"Error transcribing {image_name}: {e}")
        return (
            payload,
            image_name,
            f"[transcription error: {image_name}]",
            None,
            payload.index,
        )


def _compute_file_sha256(path: Path) -> str | None:
    """SHA-256 of a file via streamed read (sources can be ~1 GB)."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as e:
        logger.warning("Could not hash source file %s: %s", path, e)
        return None


def build_file_provenance(
    source_file: Path,
    img_cfg: dict[str, Any],
    model_type: str,
    max_pixels: int,
) -> dict[str, Any]:
    """File-level reproducibility record for one streaming run."""
    from importlib.metadata import PackageNotFoundError, version

    import PIL

    try:
        pymupdf_version = version("pymupdf")
    except PackageNotFoundError:
        pymupdf_version = None

    return {
        "file_provenance": {
            "source_file": str(source_file),
            "source_sha256": _compute_file_sha256(source_file)
            if source_file.is_file()
            else None,
            "pymupdf_version": pymupdf_version,
            "pillow_version": PIL.__version__,
            "model_type": model_type,
            "image_config": {
                "target_dpi": img_cfg.get("target_dpi"),
                "max_pixels_per_page": max_pixels,
                "resize_profile": img_cfg.get("resize_profile"),
                "llm_detail": img_cfg.get("llm_detail"),
                "media_resolution": img_cfg.get("media_resolution"),
                "jpeg_quality": img_cfg.get("jpeg_quality"),
                "grayscale_conversion": img_cfg.get("grayscale_conversion"),
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }
    }


async def run_streaming_transcription_pipeline(
    payload_source: AsyncIterator[PagePayload],
    transcriber: Any,
    temp_jsonl_path: Path,
    output_txt_path: Path,
    source_name: str,
    concurrency_config: dict[str, Any],
    postprocessing_config: dict[str, Any],
    is_folder: bool = False,
    output_format: str = "txt",
    file_provenance: dict[str, Any] | None = None,
    tracker: Any = None,
    exhausted: asyncio.Event | None = None,
) -> None:
    """Execute the in-memory streaming transcription pipeline (GPT method).

    A bounded producer-consumer pool transcribes payloads as they are
    rendered; each result is streamed to the temp JSONL immediately. The
    final output is regenerated from the complete JSONL, so resumed pages
    from earlier runs are included.

    Resume filtering happens BEFORE rendering (the caller passes a
    producer that already excludes completed pages).
    """
    transcription_conf = concurrency_config.get("concurrency", {}).get(
        "transcription", {}
    )
    concurrency_limit = transcription_conf.get("concurrency_limit", 20)
    delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

    # Version the resume artifact before any streamed write (decision 1).
    ensure_resume_marker(temp_jsonl_path)

    write_lock = asyncio.Lock()
    async with aiofiles.open(temp_jsonl_path, "a", encoding="utf-8") as jfile:
        if file_provenance is not None:
            await jfile.write(json.dumps(file_provenance, ensure_ascii=False) + "\n")
            await jfile.flush()

        async def handle(payload: PagePayload) -> Any:
            return await transcribe_payload(payload, transcriber)

        async def on_result_write(result_tuple: Any) -> None:
            if not result_tuple or len(result_tuple) < 5:
                return
            payload = result_tuple[0]
            record = _build_jsonl_record(
                (
                    payload.source_file,
                    result_tuple[1],
                    result_tuple[2],
                    result_tuple[3],
                    result_tuple[4],
                ),
                source_name,
                "gpt",
                is_folder,
                transcriber,
                payload=payload,
            )
            if record is None:
                return
            async with write_lock:
                await jfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                await jfile.flush()

        try:
            print_info(f"Processing with concurrency limit of {concurrency_limit}...")
            await run_streaming_transcription_tasks(
                payload_source,
                handle,
                concurrency_limit,
                delay_between_tasks,
                on_result=on_result_write,
                tracker=tracker,
                exhausted=exhausted,
            )
        except Exception as e:
            # Producer failures (e.g. the render failure-rate guard) must
            # propagate so the item is counted as failed; results already
            # written to the JSONL survive for a later resume.
            logger.exception(f"Error in streaming transcription for {source_name}: {e}")
            print_error(f"Streaming transcription error for {source_name}.")
            raise

    # Regenerate the combined output from the full JSONL so pages completed
    # in earlier (resumed) runs are included alongside this run's results.
    write_output_from_jsonl(
        temp_jsonl_path,
        output_txt_path,
        postprocessing_config,
        output_format=output_format,
    )


async def run_transcription_pipeline(
    image_files: list[Path],
    method: str,
    transcriber: Any | None,
    temp_jsonl_path: Path,
    output_txt_path: Path,
    source_name: str,
    concurrency_config: dict[str, Any],
    image_processing_config: dict[str, Any],
    postprocessing_config: dict[str, Any],
    is_folder: bool = False,
    resume_mode: str = "skip",
    output_format: str = "txt",
    retry_errors: bool = False,
) -> None:
    """Execute the full image transcription pipeline.

    Handles resume filtering, concurrent dispatch, streaming JSONL writes,
    result ordering, and final output assembly with post-processing.

    Args:
        image_files: Pre-processed image paths to transcribe.
        method: Transcription method (``"gpt"`` or ``"tesseract"``).
        transcriber: LLM transcriber instance (required for GPT).
        temp_jsonl_path: JSONL tracking file for streaming writes.
        output_txt_path: Final combined text output path.
        source_name: Human-readable source identifier (PDF or folder name).
        concurrency_config: Concurrency configuration dictionary.
        image_processing_config: Image processing configuration dictionary.
        postprocessing_config: Post-processing configuration dictionary.
        is_folder: True when processing an image folder (affects JSONL key).
        resume_mode: ``"skip"`` to reuse cached JSONL results; ``"overwrite"``
            to clear the JSONL and reprocess all images from scratch.
    """
    # Absolute page ordering: index each image by its position in the FULL
    # sorted list BEFORE any resume filtering, so a resumed run keeps the same
    # order_index it would have had on a single pass (see B2). Assigning the
    # index by position in the *filtered* list collides run-2 indices with
    # run-1's and scrambles the merged output.
    absolute_index = {img.name: idx for idx, img in enumerate(image_files)}

    if resume_mode == "overwrite":
        # Clear stale JSONL so old results do not mix with the new run.
        if temp_jsonl_path.exists():
            temp_jsonl_path.write_text("", encoding="utf-8")
            logger.info(f"Cleared stale JSONL cache: {temp_jsonl_path.name}")
    else:
        # Page-level resume: skip images already recorded in the JSONL.
        # This is the second layer of resume filtering; the first (item-level)
        # layer in WorkflowManager.process_selected_items() skips items whose
        # final output file already exists. Refuse artifacts written by an
        # incompatible resume format (decision 1).
        verify_resume_compatible(temp_jsonl_path)
        already_processed = get_processed_image_names(
            temp_jsonl_path, exclude_errors=retry_errors
        )
        if already_processed:
            original_count = len(image_files)
            image_files = [
                img for img in image_files if img.name not in already_processed
            ]
            skipped_count = original_count - len(image_files)
            if skipped_count > 0:
                print_info(
                    f"Skipping {skipped_count} already-processed"
                    f" images (found in JSONL)"
                )
                logger.info(
                    f"Skipped {skipped_count} images already in {temp_jsonl_path.name}"
                )

    if not image_files:
        print_info(
            "All images already processed. Regenerating output file from JSONL..."
        )
        write_output_from_jsonl(
            temp_jsonl_path,
            output_txt_path,
            postprocessing_config,
            output_format=output_format,
        )
        return

    # Build args list for concurrent dispatch
    tesseract_cfg = (
        image_processing_config.get("tesseract_image_processing", {})
        .get("ocr", {})
        .get("tesseract_config", "--oem 3 --psm 6")
    )
    args_list = [
        (
            img,
            transcriber if method == "gpt" else None,
            method,
            tesseract_cfg,
            absolute_index[img.name],
        )
        for img in image_files
    ]

    transcription_conf = concurrency_config.get("concurrency", {}).get(
        "transcription", {}
    )
    concurrency_limit = transcription_conf.get("concurrency_limit", 20)
    delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

    # Version the resume artifact before the first streamed write (decision 1).
    ensure_resume_marker(temp_jsonl_path)

    # Streaming JSONL writes as results arrive
    write_lock = asyncio.Lock()
    async with aiofiles.open(temp_jsonl_path, "a", encoding="utf-8") as jfile:

        async def on_result_write(result_tuple: Any) -> None:
            record = _build_jsonl_record(
                result_tuple, source_name, method, is_folder, transcriber
            )
            if record is None:
                return
            async with write_lock:
                await jfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                await jfile.flush()

        try:
            print_info(f"Processing with concurrency limit of {concurrency_limit}...")
            await run_concurrent_transcription_tasks(
                transcribe_single_image,
                args_list,
                concurrency_limit,
                delay_between_tasks,
                on_result=on_result_write,
            )
        except Exception as e:
            logger.exception(
                f"Error running concurrent transcription tasks for {source_name}: {e}"
            )
            print_error(f"Concurrency error for {source_name}.")
            # Propagate so the item is counted as failed rather than silently
            # succeeding with a partial output (B9). Results already streamed to
            # the JSONL survive for a later resume.
            raise

    # Regenerate the combined output from the FULL JSONL (this run's results
    # plus any completed in earlier resumed runs), ordered by absolute
    # order_index. Writing only from this run's `results` would drop pages
    # transcribed on a previous run and scramble page order (B2).
    write_output_from_jsonl(
        temp_jsonl_path,
        output_txt_path,
        postprocessing_config,
        output_format=output_format,
    )


def write_output_from_jsonl(
    jsonl_path: Path,
    output_path: Path,
    postprocessing_config: dict[str, Any],
    output_format: str = "txt",
) -> bool:
    """Write combined output text from JSONL transcription records.

    Args:
        jsonl_path: Path to JSONL file with transcription records.
        output_path: Path to write combined text output.
        postprocessing_config: Post-processing configuration dictionary.
        output_format: Output format (``"txt"``, ``"md"``, or ``"json"``).

    Returns:
        True if successful, False otherwise.
    """
    try:
        records = read_jsonl_records(jsonl_path)
        transcriptions = extract_transcription_records(records, deduplicate=True)

        if not transcriptions:
            print_warning(f"No valid transcription records found in {jsonl_path.name}")
            return False

        ordered = sorted(transcriptions, key=lambda r: r.get("order_index", 0))

        pages: list[dict[str, Any]] = []
        for record in ordered:
            image_name = record.get("image_name", "")
            text_chunk = record.get("text_chunk", "")
            order_index = record.get("order_index", 0)
            page_number = int(order_index) + 1 if isinstance(order_index, int) else None
            pages.append(
                {
                    "text": text_chunk,
                    "page_number": page_number,
                    "image_name": image_name,
                }
            )

        actual_path = write_transcription_output(
            pages,
            output_path,
            output_format=output_format,
            postprocess=True,
            postprocessing_config=postprocessing_config,
        )
        print_success(f"Output written: {actual_path.name}")
        return True
    except Exception as e:
        logger.exception(f"Error writing output from JSONL {jsonl_path.name}: {e}")
        print_error(f"Failed to write output from {jsonl_path.name}.")
        return False
