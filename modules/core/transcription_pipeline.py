"""Image transcription pipeline extracted from WorkflowManager.

Handles resume-aware image filtering, concurrent transcription dispatch,
streaming JSONL writes, result ordering, and final output assembly.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

from modules.infra.logger import setup_logger
from modules.llm import transcribe_image_with_llm
from modules.infra.concurrency import run_concurrent_transcription_tasks
from modules.processing.text_processing import extract_transcribed_text, format_page_line
from modules.processing.postprocess import postprocess_transcription
from modules.operations.jsonl_utils import (
    get_processed_image_names,
    read_jsonl_records,
    extract_transcription_records,
)
from modules.processing.tesseract_utils import perform_ocr
from modules.ui import print_info, print_warning, print_error, print_success

logger = setup_logger(__name__)


async def transcribe_single_image(
    img_path: Path,
    transcriber: Optional[Any],
    method: str,
    tesseract_config: str = "--oem 3 --psm 6",
    order_index: int = 0,
) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]], int]:
    """Transcribe a single image file using either GPT or Tesseract OCR.

    Returns:
        Tuple of (image_path_str, image_name, text, raw_response, order_index).
    """
    image_name = img_path.name
    final_text: Optional[str] = None
    try:
        if method == "gpt":
            if not transcriber:
                logger.error("No transcriber instance provided for GPT usage.")
                return (str(img_path), image_name,
                        f"[transcription error: {image_name}]", None, order_index)
            result = await transcribe_image_with_llm(img_path, transcriber)
            logger.debug(f"LLM response for {img_path.name}: {result}")
            try:
                final_text = extract_transcribed_text(result, image_name)
            except Exception as e:
                logger.error(
                    f"Error extracting transcription for {img_path.name}: {e}. Marking as transcription error.")
                final_text = f"[transcription error: {image_name}]"
            return (str(img_path), image_name, final_text, result, order_index)
        elif method == "tesseract":
            final_text = perform_ocr(img_path, tesseract_config)
        else:
            logger.error(f"Unknown transcription method '{method}' for image {img_path.name}")
            final_text = None
        return (str(img_path), image_name, final_text, None, order_index)
    except Exception as e:
        logger.exception(f"Error transcribing {img_path.name} with method '{method}': {e}")
        return (
            str(img_path), image_name,
            f"[transcription error: {image_name}]",
            None,
            order_index
        )


def _build_jsonl_record(
    result_tuple: Tuple[str, str, Optional[str], Optional[Dict[str, Any]], int],
    source_name: str,
    method: str,
    is_folder: bool,
    transcriber: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Build a JSONL record dict from a transcription result tuple.

    Returns None if the result should be skipped (no text).
    """
    if not result_tuple or len(result_tuple) < 5:
        return None
    img_path_str, image_name, text_chunk, raw_response, order_index = result_tuple
    if text_chunk is None:
        return None

    # Build base record — only the source key differs between PDF and folder
    source_key = "folder_name" if is_folder else "file_name"
    record: Dict[str, Any] = {
        source_key: source_name,
        "pre_processed_image": img_path_str,
        "image_name": image_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "method": method,
        "order_index": order_index,
        "text_chunk": text_chunk,
    }

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
        except Exception:
            pass

    return record


async def run_transcription_pipeline(
    image_files: List[Path],
    method: str,
    transcriber: Optional[Any],
    temp_jsonl_path: Path,
    output_txt_path: Path,
    source_name: str,
    concurrency_config: Dict[str, Any],
    image_processing_config: Dict[str, Any],
    postprocessing_config: Dict[str, Any],
    is_folder: bool = False,
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
    """
    # Resume filtering — skip images already in JSONL
    already_processed = get_processed_image_names(temp_jsonl_path)
    if already_processed:
        original_count = len(image_files)
        image_files = [img for img in image_files if img.name not in already_processed]
        skipped_count = original_count - len(image_files)
        if skipped_count > 0:
            print_info(f"Skipping {skipped_count} already-processed images (found in JSONL)")
            logger.info(f"Skipped {skipped_count} images already in {temp_jsonl_path.name}")

    if not image_files:
        print_info("All images already processed. Regenerating output file from JSONL...")
        write_output_from_jsonl(temp_jsonl_path, output_txt_path, postprocessing_config)
        return

    # Build args list for concurrent dispatch
    tesseract_cfg = (
        image_processing_config
        .get('tesseract_image_processing', {})
        .get('ocr', {})
        .get('tesseract_config', "--oem 3 --psm 6")
    )
    all_image_names = {img.name: idx for idx, img in enumerate(image_files)}
    args_list = [
        (
            img,
            transcriber if method == "gpt" else None,
            method,
            tesseract_cfg,
            all_image_names[img.name],
        )
        for img in image_files
    ]

    transcription_conf = concurrency_config.get("concurrency", {}).get("transcription", {})
    concurrency_limit = transcription_conf.get("concurrency_limit", 20)
    delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

    # Streaming JSONL writes as results arrive
    write_lock = asyncio.Lock()
    async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
        async def on_result_write(result_tuple: Any) -> None:
            record = _build_jsonl_record(
                result_tuple, source_name, method, is_folder, transcriber
            )
            if record is None:
                return
            async with write_lock:
                await jfile.write(json.dumps(record) + "\n")
                await jfile.flush()

        try:
            print_info(f"Processing with concurrency limit of {concurrency_limit}...")
            results = await run_concurrent_transcription_tasks(
                transcribe_single_image,
                args_list,
                concurrency_limit,
                delay_between_tasks,
                on_result=on_result_write,
            )
        except Exception as e:
            logger.exception(
                f"Error running concurrent transcription tasks for {source_name}: {e}")
            print_error(f"Concurrency error for {source_name}.")
            return

    # Combine transcription text in page order with unified page headers
    try:
        ordered = sorted(
            [r for r in results if r and r[2] is not None], key=lambda r: r[4]
        )
        lines: List[str] = []
        for (_p, image_name, text_chunk, _raw, order_index) in ordered:
            page_number = int(order_index) + 1 if isinstance(order_index, int) else None
            lines.append(format_page_line(text_chunk, page_number, image_name))
        combined_text = "\n".join(lines)
        processed_text = postprocess_transcription(combined_text, postprocessing_config)
        output_txt_path.write_text(processed_text, encoding='utf-8')
    except Exception as e:
        logger.exception(
            f"Error writing combined transcription output for {source_name}: {e}")
        print_error(f"Failed to write combined output for {source_name}.")


def write_output_from_jsonl(
    jsonl_path: Path,
    output_path: Path,
    postprocessing_config: Dict[str, Any],
) -> bool:
    """Write combined output text from JSONL transcription records.

    Args:
        jsonl_path: Path to JSONL file with transcription records.
        output_path: Path to write combined text output.
        postprocessing_config: Post-processing configuration dictionary.

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

        lines: List[str] = []
        for record in ordered:
            image_name = record.get("image_name", "")
            text_chunk = record.get("text_chunk", "")
            order_index = record.get("order_index", 0)
            page_number = int(order_index) + 1 if isinstance(order_index, int) else None
            lines.append(format_page_line(text_chunk, page_number, image_name))

        combined_text = "\n".join(lines)
        processed_text = postprocess_transcription(combined_text, postprocessing_config)
        output_path.write_text(processed_text, encoding='utf-8')
        print_success(f"Output written: {output_path.name}")
        return True
    except Exception as e:
        logger.exception(f"Error writing output from JSONL {jsonl_path.name}: {e}")
        print_error(f"Failed to write output from {jsonl_path.name}.")
        return False
