# modules/batching.py

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from modules.model_capabilities import detect_capabilities
from modules.utils import console_print, extract_page_number_from_filename
from modules.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_FORMATS: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _render_prompt_with_schema(prompt_text: str, schema_obj: Dict[str, Any]) -> str:
    """
    Render the system prompt with the latest schema content injected.
    Supports either a placeholder token "{{TRANSCRIPTION_SCHEMA}}" or
    replacing any existing JSON object that follows the text marker
    "The JSON schema:".
    """
    try:
        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
    except Exception:
        schema_str = str(schema_obj)

    token = "{{TRANSCRIPTION_SCHEMA}}"
    if token in prompt_text:
        return prompt_text.replace(token, schema_str)

    marker = "The JSON schema:"
    if marker in prompt_text:
        idx = prompt_text.find(marker)
        start_brace = prompt_text.find("{", idx)
        if start_brace != -1:
            end_brace = prompt_text.rfind("}")
            if end_brace != -1 and end_brace > start_brace:
                return prompt_text[:start_brace] + schema_str + prompt_text[end_brace + 1:]
        # Fallback: append schema after the marker
        return prompt_text + "\n" + schema_str

    # Fallback: append a full section
    return prompt_text + "\n\nThe JSON schema:\n" + schema_str


def encode_image_to_data_url(image_path: Path) -> str:
    """
    Encode an image file as a data URL.
    """
    ext = image_path.suffix.lower()
    mime = SUPPORTED_IMAGE_FORMATS.get(ext)
    if not mime:
        logger.error("Unsupported image format: %s", image_path.suffix)
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _build_responses_body_for_image(
    *,
    model_config: Dict[str, Any],
    system_prompt: str,
    image_url: str,
    transcription_schema: Dict[str, Any],
    llm_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct a **Responses API** request body for vision transcription,
    with feature gating based on the capabilities registry.
    """
    tm = model_config or {}
    model_name: str = tm.get("name", "gpt-4o-2024-08-06")
    caps = detect_capabilities(model_name)

    # Normalize detail from caller/config
    detail_norm: Optional[str] = None
    if isinstance(llm_detail, str):
        d = llm_detail.lower().strip()
        if d in ("low", "high"):
            detail_norm = d
        elif d == "auto":
            detail_norm = None

    # Base body
    body: Dict[str, Any] = {
        "model": model_name,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Please transcribe the text from this image."},
                    {
                        "type": "input_image",
                        # Responses API expects image_url as a STRING (URL or data URL)
                        # with optional detail as a sibling property.
                        "image_url": image_url,
                        **(
                            {"detail": detail_norm}
                            if (detail_norm is not None and caps.supports_image_detail)
                            else ({"detail": caps.default_ocr_detail} if caps.supports_image_detail and detail_norm is None and False else {})
                        ),
                    },
                ],
            },
        ],
        "max_output_tokens": int(
            tm.get("max_output_tokens")
            if tm.get("max_output_tokens") is not None
            else (tm.get("max_completion_tokens") if tm.get("max_completion_tokens") is not None else tm.get("max_tokens", 4096))
        ),
    }

    # Optional service tier (now sourced from concurrency_config.yaml)
    try:
        cfg = ConfigLoader()
        cfg.load_configs()
        cc = cfg.get_concurrency_config()
        st = (cc.get("concurrency", {}) or {}).get("transcription", {}).get("service_tier")
    except Exception:
        st = None
    # Backward-compat: fall back to model_config if concurrency_config missing
    effective_service_tier = st if st is not None else tm.get("service_tier")
    if effective_service_tier:
        body["service_tier"] = effective_service_tier

    # Structured outputs (avoid on o-series)
    if caps.supports_structured_outputs:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "TranscriptionSchema",
                "schema": transcription_schema,
                "strict": True,
            }
        }

    # GPT-5 public reasoning controls (no classic sampler)
    if caps.supports_reasoning_effort and tm.get("reasoning"):
        body["reasoning"] = tm["reasoning"]
        if tm.get("text") and isinstance(tm["text"], dict) and tm["text"].get("verbosity") is not None:
            body.setdefault("text", {})["verbosity"] = tm["text"]["verbosity"]

    # Classic sampler controls only for families that support them
    if caps.supports_sampler_controls:
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            if k in tm and tm[k] is not None:
                body[k] = tm[k]

    return body


def create_batch_request_line(
    custom_id: str,
    image_url: str,
    image_info: Dict[str, Any],
    model_config: Dict[str, Any],
    system_prompt_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Create a **Responses API** batch request line for an image transcription task.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        (json_line_for_api, local_metadata_record)
    """
    # Load system prompt
    if system_prompt_path is None:
        system_prompt_path = Path("system_prompt/system_prompt.txt")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found at {system_prompt_path}")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    # Load transcription schema
    if schema_path is None:
        schema_path = Path("schemas/transcription_schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    with schema_path.open("r", encoding="utf-8") as sfile:
        loaded_schema = json.load(sfile)
        # Accept either a wrapper {name, strict, schema: {...}} or a bare JSON Schema
        if isinstance(loaded_schema, dict) and "schema" in loaded_schema and isinstance(loaded_schema["schema"], dict):
            transcription_schema = loaded_schema["schema"]
        else:
            transcription_schema = loaded_schema

    # Inject current schema into system prompt
    system_prompt = _render_prompt_with_schema(system_prompt, loaded_schema)

    # Load image processing config for llm_detail
    try:
        cfg = ConfigLoader()
        cfg.load_configs()
        image_cfg = cfg.get_image_processing_config().get("image_processing", {})
        raw_detail = str(image_cfg.get("llm_detail", "high")).lower().strip()
        llm_detail: Optional[str]
        if raw_detail in ("low", "high"):
            llm_detail = raw_detail
        elif raw_detail == "auto":
            llm_detail = "auto"
        else:
            llm_detail = "auto"
    except Exception:
        llm_detail = "auto"

    # Build **Responses** body (typed input + text.format where supported)
    request_body = _build_responses_body_for_image(
        model_config=model_config,
        system_prompt=system_prompt,
        image_url=image_url,
        transcription_schema=transcription_schema,
        llm_detail=llm_detail,
    )

    logger.debug(
        "Batch image body: model=%s include_detail=%s detail=%s",
        model_config.get("name"),
        isinstance(llm_detail, str) and llm_detail.lower().strip() in ("low", "high"),
        llm_detail,
    )

    # Final request line for the Batch API (must have exactly these fields)
    request_line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": request_body,
    }

    # Separate metadata for local ordering/tracking
    metadata_record = {"batch_request": {"custom_id": custom_id, "image_info": image_info}}
    return json.dumps(request_line), metadata_record


def write_batch_file(request_lines: List[str], output_path: Path) -> Path:
    """
    Write JSONL lines to disk for Batch submission.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for line in request_lines:
            f.write(line + "\n")
    logger.info("Batch file written to %s", output_path)
    return output_path


def submit_batch(batch_file_path: Path) -> Dict[str, Any]:
    """
    Submit a prepared JSONL batch to the OpenAI Batch API targeting **/v1/responses**.
    """
    client = OpenAI()
    try:
        console_print(f"[INFO] Uploading batch file {batch_file_path.name} to OpenAI...")
        with batch_file_path.open("rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
        file_id = file_response.id
        logger.info("Uploaded batch file; file id: %s", file_id)
        console_print("[INFO] Batch file uploaded successfully, creating batch job...")

        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": "Batch OCR transcription via Responses API"},
        )
        logger.info("Batch submitted; batch id: %s", batch_response.id)
        console_print(f"[INFO] Batch job created successfully with ID: {batch_response.id}")
        return batch_response
    except Exception as exc:  # propagate after logging
        logger.error("Error submitting batch file %s: %s", batch_file_path, exc)
        console_print(f"[ERROR] Failed to submit batch file: {exc}")
        raise


def process_batch_transcription(
    image_files: List[Path],
    prompt_text: str,  # kept for signature parity
    model_config: Dict[str, Any],
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Prepare and submit batched image transcriptions using the **Responses API**.

    Returns
    -------
    Tuple[List[Any], List[Dict[str, Any]]]
        (batch_responses, all_metadata_records)
    """
    chunk_size = 50
    total_images = len(image_files)
    batch_responses: List[Any] = []
    all_metadata_records: List[Dict[str, Any]] = []

    # Safety margin under 180 MB limit
    max_batch_size = 150 * 1024 * 1024
    batch_index = 1

    # Note: Previously captured submitted_batch_data for a local debug snapshot; removed.
    console_print(f"[INFO] Processing {total_images} images in chunks of {chunk_size}...")

    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_images = image_files[chunk_start:chunk_end]

        console_print(
            f"[INFO] Processing chunk {chunk_start // chunk_size + 1}/"
            f"{(total_images + chunk_size - 1) // chunk_size}: "
            f"images {chunk_start + 1}-{chunk_end} of {total_images}..."
        )

        batch_request_lines: List[str] = []
        metadata_records: List[Dict[str, Any]] = []

        for idx, image_file in enumerate(chunk_images):
            try:
                global_idx = chunk_start + idx
                custom_id = f"req-{global_idx + 1}"
                print(
                    f"[INFO] Encoding image {chunk_start + idx + 1}/{total_images}: {image_file.name}",
                    end="\r",
                )
                data_url = encode_image_to_data_url(image_file)
                image_info = {
                    "image_name": image_file.name,
                    "order_index": global_idx,
                    "page_number": extract_page_number_from_filename(image_file.name),
                }
                request_line, metadata_record = create_batch_request_line(
                    custom_id=custom_id,
                    image_url=data_url,
                    image_info=image_info,
                    model_config=model_config,
                )
                batch_request_lines.append(request_line)
                metadata_records.append(metadata_record)
                all_metadata_records.append(metadata_record)
            except Exception as exc:
                logger.error("Error processing image %s: %s", image_file, exc)
                console_print(f"[ERROR] Failed to process image {image_file.name}: {exc}")

        console_print(f"[INFO] Creating batch files for chunk {chunk_start // chunk_size + 1}...")

        current_lines: List[str] = []
        current_size = 0
        current_metadata: List[Dict[str, Any]] = []

        for i, line in enumerate(batch_request_lines):
            line_bytes = len(line.encode("utf-8"))

            if current_size + line_bytes > max_batch_size and current_lines:
                batch_file = Path(f"batch_requests_part_{batch_index}.jsonl")
                write_batch_file(current_lines, batch_file)
                try:
                    response = submit_batch(batch_file)
                    batch_id = response.id
                    console_print(f"[SUCCESS] Successfully submitted batch {batch_index} with ID: {batch_id}")
                    batch_responses.append(response)
                except Exception as exc:
                    logger.error("Error submitting batch file %s: %s", batch_file, exc)
                    console_print(f"[ERROR] Failed to submit batch file {batch_file}: {exc}")
                try:
                    batch_file.unlink()
                except Exception:
                    logger.warning("Could not delete temporary batch file: %s", batch_file)

                batch_index += 1
                current_lines = []
                current_size = 0
                current_metadata = []

            current_lines.append(line)
            current_size += line_bytes
            if i < len(metadata_records):
                current_metadata.append(metadata_records[i])

        if current_lines:
            batch_file = Path(f"batch_requests_part_{batch_index}.jsonl")
            write_batch_file(current_lines, batch_file)
            try:
                response = submit_batch(batch_file)
                batch_id = response.id
                console_print(f"[SUCCESS] Successfully submitted batch {batch_index} with ID: {batch_id}")
                batch_responses.append(response)
            except Exception as exc:
                logger.error("Error submitting batch file %s: %s", batch_file, exc)
                console_print(f"[ERROR] Failed to submit batch file {batch_file}: {exc}")
            try:
                batch_file.unlink()
            except Exception:
                logger.warning("Could not delete temporary batch file: %s", batch_file)
            batch_index += 1
    console_print(f"[INFO] All {total_images} images processed and submitted in {batch_index - 1} batch files")
    return batch_responses, all_metadata_records
