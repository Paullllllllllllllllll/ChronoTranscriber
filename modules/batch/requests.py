"""OpenAI Batch API processing utilities.

This module provides batch processing capabilities specifically for OpenAI's Batch API.
Unlike synchronous transcription (which supports multiple providers via LangChain),
batch processing is currently OpenAI-specific due to the proprietary nature of the
Batch API endpoint (/v1/responses).

Note:
    For synchronous transcription with multi-provider support, use:
    - ``modules.llm.transcriber.LangChainTranscriber``
    - ``modules.llm.providers`` for direct provider access

    Batch processing will remain OpenAI-specific until other providers offer
    equivalent batch processing APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types import Batch

from modules.config.capabilities import detect_capabilities
from modules.config.config_loader import PROJECT_ROOT
from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.llm.prompt_utils import (
    inject_additional_context,
    render_prompt_with_schema,
)
from modules.llm.structured_outputs import build_structured_text_format
from modules.ui import print_error, print_info, print_success

logger = setup_logger(__name__)


# Centralized batch chunk size default and getter
# (configurable via concurrency_config.yaml)
DEFAULT_BATCH_CHUNK_SIZE: int = 50


def get_batch_chunk_size() -> int:
    """
    Returns the batch chunk size from concurrency_config.yaml if present,
    otherwise a safe default.

    Expected path in YAML:
    concurrency:
      transcription:
        batch_chunk_size: 50
    """
    try:
        cc = get_config_service().get_concurrency_config() or {}
        val = (
            (cc.get("concurrency", {}) or {})
            .get("transcription", {})
            .get("batch_chunk_size", DEFAULT_BATCH_CHUNK_SIZE)
        )
        # Coerce to int and guard against invalid values
        try:
            ival = int(val)
            if ival < 1:
                return DEFAULT_BATCH_CHUNK_SIZE
            return ival
        except (ValueError, TypeError):
            return DEFAULT_BATCH_CHUNK_SIZE
    except (KeyError, AttributeError, TypeError):
        return DEFAULT_BATCH_CHUNK_SIZE


def encode_image_to_data_url(image_path: Path) -> str:
    """Encode an image file as a data URL.

    Delegates to modules.images.encoding for the shared implementation.
    """
    from modules.images.encoding import encode_image_to_data_url as _encode

    return _encode(image_path)


def _build_responses_body_for_image(
    *,
    model_config: dict[str, Any],
    system_prompt: str,
    image_url: str,
    transcription_schema: dict[str, Any],
    llm_detail: str | None = None,
) -> dict[str, Any]:
    """
    Construct a Responses API request body for vision transcription,
    with feature gating based on the capabilities registry.
    """
    tm = model_config or {}
    model_name: str = tm.get("name", "gpt-4o-2024-08-06")
    caps = detect_capabilities(model_name)

    # Normalize detail from caller/config
    detail_norm: str | None = None
    if isinstance(llm_detail, str):
        d = llm_detail.lower().strip()
        valid = {"low", "high"}
        if caps.supports_image_detail_original:
            valid.add("original")
        if d in valid:
            detail_norm = d
        elif d == "auto":
            detail_norm = None

    # Build user content blocks
    user_instruction = tm.get("user_instruction", "The image:")
    user_content: list[dict[str, Any]] = []
    if user_instruction:
        user_content.append({"type": "input_text", "text": user_instruction})
    user_content.append(
        {
            "type": "input_image",
            "image_url": image_url,
            **(
                {"detail": detail_norm}
                if (
                    detail_norm in ("low", "high", "original")
                    and caps.supports_image_detail
                )
                else {}
            ),
        },
    )

    # Base body
    body: dict[str, Any] = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "max_output_tokens": int(
            tm.get("max_output_tokens")
            or tm.get("max_completion_tokens")
            or tm.get("max_tokens")
            or 4096
        ),
    }

    # Optional service tier (now sourced from concurrency_config.yaml)
    try:
        cc = get_config_service().get_concurrency_config()
        st = (
            (cc.get("concurrency", {}) or {})
            .get("transcription", {})
            .get("service_tier")
        )
    except (KeyError, AttributeError, TypeError):
        st = None
    # Fallback: use model_config if service_tier not in concurrency_config
    effective_service_tier = st if st is not None else tm.get("service_tier")

    # IMPORTANT: Flex processing is only available for synchronous API calls, NOT
    # batch API. If flex is configured, use "auto" instead for batch requests.
    if effective_service_tier:
        allowed_service_tiers = {"auto", "default", "priority"}
        tier_str = str(effective_service_tier)

        # Replace "flex" with "auto" for batch processing since flex is not supported
        if tier_str == "flex":
            logger.info(
                "Batch API does not support service_tier='flex'. Using 'auto' instead."
            )
            body["service_tier"] = "auto"
        elif tier_str in allowed_service_tiers:
            body["service_tier"] = tier_str
        else:
            logger.warning(
                "Ignoring unsupported service_tier=%s for model %s",
                effective_service_tier,
                model_name,
            )

    # Structured outputs (avoid on o-series if disabled)
    if caps.supports_structured_outputs:
        fmt = build_structured_text_format(
            transcription_schema, "TranscriptionSchema", True
        )
        if fmt is not None:
            body.setdefault("text", {})
            body["text"]["format"] = fmt

    # GPT-5 public reasoning controls (no classic sampler)
    if caps.supports_reasoning_effort and tm.get("reasoning"):
        body["reasoning"] = tm["reasoning"]
        if (
            tm.get("text")
            and isinstance(tm["text"], dict)
            and tm["text"].get("verbosity") is not None
        ):
            body.setdefault("text", {})["verbosity"] = tm["text"]["verbosity"]

    # Classic sampler controls only for families that support them
    if caps.supports_sampler_controls:
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            if k in tm and tm[k] is not None:
                body[k] = tm[k]

    # Prompt cache retention (OpenAI automatic caching extension)
    try:
        caching_cfg = get_config_service().get_prompt_caching_config()
        if caching_cfg.get("enabled"):
            openai_cfg = caching_cfg.get("openai", {})
            retention = (
                openai_cfg.get("prompt_cache_retention")
                if isinstance(openai_cfg, dict)
                else None
            )
            if retention:
                body["prompt_cache_retention"] = retention
    except (KeyError, AttributeError, TypeError):
        logger.debug(
            "Could not load prompt caching config for batch request; skipping."
        )

    return body


@dataclass
class _BatchRequestPrep:
    """Run-invariant inputs for building batch request lines.

    Computed once per :func:`process_batch_transcription` run and reused for
    every page, hoisting the system-prompt read, schema parse, schema render,
    explicit-context load, and llm_detail resolution out of the per-image loop.
    Only per-image hierarchical context resolution and the body build remain
    per page.
    """

    base_system_prompt: str
    transcription_schema: Any
    llm_detail: str | None
    use_explicit_context: bool
    explicit_context: str | None
    model_config: dict[str, Any]


def _prepare_batch_request(
    model_config: dict[str, Any],
    system_prompt_path: Path | None = None,
    schema_path: Path | None = None,
    additional_context_path: Path | None = None,
) -> _BatchRequestPrep:
    """Compute the run-invariant parts of a batch request line once.

    Performs path resolution, the system-prompt read, schema parse and render,
    explicit additional-context load, and llm_detail resolution -- all of which
    are identical for every page in a run. The result is consumed by
    :func:`_build_batch_request_line` per image.
    """
    # Resolve prompt/schema paths with PROJECT_ROOT defaults and optional overrides
    if system_prompt_path is None or schema_path is None:
        try:
            pcfg = get_config_service().get_paths_config()
            general = pcfg.get("general", {})
        except (KeyError, AttributeError, TypeError):
            general = {}

        if system_prompt_path is None:
            override_prompt = general.get("transcription_prompt_path")
            system_prompt_path = (
                Path(override_prompt)
                if override_prompt
                else (
                    PROJECT_ROOT / "system_prompt" / "transcription_prompt_schema.txt"
                )
            )
        if schema_path is None:
            override_schema = general.get("transcription_schema_path")
            schema_path = (
                Path(override_schema)
                if override_schema
                else (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json")
            )

    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found at {system_prompt_path}")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    with schema_path.open("r", encoding="utf-8") as sfile:
        loaded_schema = json.load(sfile)
        # Accept either a wrapper {name, strict, schema: {...}} or a bare JSON Schema
        if (
            isinstance(loaded_schema, dict)
            and "schema" in loaded_schema
            and isinstance(loaded_schema["schema"], dict)
        ):
            transcription_schema = loaded_schema["schema"]
        else:
            transcription_schema = loaded_schema

    # Inject current schema into system prompt (optional)
    inject_schema_into_prompt = True
    try:
        # Respect a model_config flag if provided
        inject_schema_into_prompt = bool(
            model_config.get("inject_schema_into_prompt", True)
        )
    except (AttributeError, TypeError):
        inject_schema_into_prompt = True
    if inject_schema_into_prompt:
        system_prompt = render_prompt_with_schema(system_prompt, loaded_schema)

    # Explicit additional context (run-invariant). When an explicit path is
    # given and exists, it wins over per-image hierarchical resolution, exactly
    # as in the original single-call flow.
    use_explicit_context = False
    explicit_context: str | None = None
    if additional_context_path is not None and additional_context_path.exists():
        use_explicit_context = True
        try:
            explicit_context = additional_context_path.read_text(
                encoding="utf-8"
            ).strip()
        except (OSError, PermissionError, UnicodeDecodeError) as e:
            logger.warning(
                "Failed to load additional context from %s: %s",
                additional_context_path,
                e,
            )

    # Load image processing config for llm_detail
    try:
        image_cfg = (
            get_config_service()
            .get_image_processing_config()
            .get("api_image_processing", {})
        )
        raw_detail = str(image_cfg.get("llm_detail", "high")).lower().strip()
        llm_detail: str | None
        if raw_detail in ("low", "high", "original"):
            llm_detail = raw_detail
        elif raw_detail == "auto":
            llm_detail = "auto"
        else:
            llm_detail = "auto"
    except (KeyError, AttributeError, TypeError):
        llm_detail = "auto"

    return _BatchRequestPrep(
        base_system_prompt=system_prompt,
        transcription_schema=transcription_schema,
        llm_detail=llm_detail,
        use_explicit_context=use_explicit_context,
        explicit_context=explicit_context,
        model_config=model_config,
    )


def _build_batch_request_line(
    prep: _BatchRequestPrep,
    custom_id: str,
    image_url: str,
    image_info: dict[str, Any],
    use_hierarchical_context: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Build one batch request line from precomputed run-invariant inputs.

    Only per-image work remains here: hierarchical context resolution (when no
    explicit context path was supplied), context injection, and the body build.
    """
    system_prompt = prep.base_system_prompt

    # Inject additional context - use explicit context or hierarchical resolution
    additional_context = None
    if prep.use_explicit_context:
        additional_context = prep.explicit_context
    elif use_hierarchical_context:
        # Use hierarchical context resolution for file-specific context
        from modules.config.context import resolve_context_for_file

        # Extract image path from image_url if it's a file path
        if image_url.startswith("file://"):
            image_path = Path(image_url[7:])
        elif image_url.startswith("data:"):
            # Base64 data URL - no file path available for context resolution
            image_path = None
        else:
            # Assume it might be a local path
            image_path = Path(image_url) if not image_url.startswith("http") else None

        if image_path:
            context_content, context_path = resolve_context_for_file(image_path)
            if context_content:
                additional_context = context_content
                logger.debug(f"Using resolved context from: {context_path}")

    # Inject context into prompt (or remove section if empty)
    system_prompt = inject_additional_context(system_prompt, additional_context or "")

    # Build Responses body (typed input + text.format where supported)
    request_body = _build_responses_body_for_image(
        model_config=prep.model_config,
        system_prompt=system_prompt,
        image_url=image_url,
        transcription_schema=prep.transcription_schema,
        llm_detail=prep.llm_detail,
    )

    logger.debug(
        "Batch image body: model=%s include_detail=%s detail=%s",
        prep.model_config.get("name"),
        isinstance(prep.llm_detail, str)
        and prep.llm_detail.lower().strip() in ("low", "high", "original"),
        prep.llm_detail,
    )

    # Final request line for the Batch API (must have exactly these fields)
    request_line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": request_body,
    }

    # Separate metadata for local ordering/tracking
    metadata_record = {
        "batch_request": {"custom_id": custom_id, "image_info": image_info}
    }
    return json.dumps(request_line), metadata_record


def create_batch_request_line(
    custom_id: str,
    image_url: str,
    image_info: dict[str, Any],
    model_config: dict[str, Any],
    system_prompt_path: Path | None = None,
    schema_path: Path | None = None,
    additional_context_path: Path | None = None,
    use_hierarchical_context: bool = True,
) -> tuple[str, dict[str, Any]]:
    """
    Create a Responses API batch request line for an image transcription task.

    Thin wrapper: computes the run-invariant prep and builds a single line.
    The batch loop hoists the prep step and calls
    :func:`_build_batch_request_line` directly to avoid redundant per-page work.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        (json_line_for_api, local_metadata_record)
    """
    prep = _prepare_batch_request(
        model_config,
        system_prompt_path=system_prompt_path,
        schema_path=schema_path,
        additional_context_path=additional_context_path,
    )
    return _build_batch_request_line(
        prep,
        custom_id=custom_id,
        image_url=image_url,
        image_info=image_info,
        use_hierarchical_context=use_hierarchical_context,
    )


def write_batch_file(request_lines: list[str], output_path: Path) -> Path:
    """
    Write JSONL lines to disk for Batch submission.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for line in request_lines:
            f.write(line + "\n")
    logger.info("Batch file written to %s", output_path)
    return output_path


def submit_batch(batch_file_path: Path) -> Batch:
    """
    Submit a prepared JSONL batch to the OpenAI Batch API targeting /v1/responses.
    """
    # Lazy import to avoid import-time dependency issues when this module is
    # imported for config/telemetry only.
    from openai import OpenAI

    client = OpenAI()
    try:
        print_info(f"Uploading batch file {batch_file_path.name} to OpenAI...")
        with batch_file_path.open("rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
        file_id = file_response.id
        logger.info("Uploaded batch file; file id: %s", file_id)
        print_info("Batch file uploaded successfully, creating batch job...")

        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": "Batch OCR transcription via Responses API"},
        )
        logger.info("Batch submitted; batch id: %s", batch_response.id)
        print_info(f"Batch job created successfully with ID: {batch_response.id}")
        return batch_response
    except Exception as exc:  # propagate after logging
        logger.error("Error submitting batch file %s: %s", batch_file_path, exc)
        print_error(f"Failed to submit batch file: {exc}")
        raise


def _submit_and_cleanup_batch_file(
    batch_file: Path, batch_index: int, batch_responses: list[Any]
) -> bool:
    """Submit one batch file, record its response, and delete the temp file.

    Appends the API response to *batch_responses* on success. Always attempts
    to remove the temporary JSONL file afterwards. Returns ``True`` when the
    submission succeeded, ``False`` otherwise.
    """
    submitted = False
    try:
        response = submit_batch(batch_file)
        print_success(
            f"Successfully submitted batch {batch_index} with ID: {response.id}"
        )
        batch_responses.append(response)
        submitted = True
    except Exception as exc:
        logger.error("Error submitting batch file %s: %s", batch_file, exc)
        print_error(f"Failed to submit batch file {batch_file}: {exc}")
    try:
        batch_file.unlink()
    except OSError:
        logger.warning("Could not delete temporary batch file: %s", batch_file)
    return submitted


def _item_to_data_url_and_name(item: Any) -> tuple[str, str]:
    """Resolve a batch item to (data_url, image_name).

    Accepts a ``Path`` to an image file, or any in-memory object carrying
    base64 data (``image_base64``/``base64``), ``mime_type``, and
    ``image_name`` attributes (e.g. RepairTarget, PagePayload).
    """
    if isinstance(item, Path):
        return encode_image_to_data_url(item), item.name
    b64 = getattr(item, "image_base64", None) or getattr(item, "base64", None)
    mime = getattr(item, "mime_type", None) or "image/jpeg"
    name = getattr(item, "image_name", "") or "[in-memory]"
    if not b64:
        raise ValueError(f"Batch item for {name} has no image data")
    return f"data:{mime};base64,{b64}", name


def process_batch_transcription(
    image_files: list[Path | Any],
    prompt_text: str,  # kept for signature parity
    model_config: dict[str, Any],
    *,
    schema_path: Path | None = None,
    additional_context_path: Path | None = None,
    use_hierarchical_context: bool = True,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Prepare and submit batched image transcriptions using the Responses API.

    Items may be image file paths or in-memory objects with base64 data
    (see :func:`_item_to_data_url_and_name`).

    Returns
    -------
    Tuple[List[Any], List[Dict[str, Any]]]
        (batch_responses, all_metadata_records)
    """
    chunk_size = get_batch_chunk_size()
    total_images = len(image_files)
    batch_responses: list[Any] = []
    all_metadata_records: list[dict[str, Any]] = []

    # Safety margin under 180 MB limit
    max_batch_size = 150 * 1024 * 1024
    batch_index = 1

    print_info(f"Processing {total_images} images in chunks of {chunk_size}...")

    # Compute the run-invariant request prep once (system prompt read, schema
    # parse/render, explicit-context load, llm_detail). On failure fall back to
    # per-image create_batch_request_line so the original per-image error
    # handling and outcome (logged failure -> RuntimeError) are preserved.
    try:
        request_prep: _BatchRequestPrep | None = _prepare_batch_request(
            model_config,
            schema_path=schema_path,
            additional_context_path=additional_context_path,
        )
    except Exception:
        request_prep = None

    submitted_parts = 0
    attempted_parts = 0

    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_images = image_files[chunk_start:chunk_end]

        print_info(
            f"Processing chunk {chunk_start // chunk_size + 1}/"
            f"{(total_images + chunk_size - 1) // chunk_size}: "
            f"images {chunk_start + 1}-{chunk_end} of {total_images}..."
        )

        batch_request_lines: list[str] = []
        metadata_records: list[dict[str, Any]] = []

        for idx, image_file in enumerate(chunk_images):
            try:
                global_idx = chunk_start + idx
                custom_id = f"req-{global_idx + 1}"
                pos = chunk_start + idx + 1
                data_url, image_name = _item_to_data_url_and_name(image_file)
                print(
                    f"[INFO] Encoding image {pos}/{total_images}: {image_name}",
                    end="\r",
                )
                image_info = {
                    "image_name": image_name,
                    "order_index": global_idx,
                    "page_number": global_idx + 1,
                }
                if request_prep is not None:
                    request_line, metadata_record = _build_batch_request_line(
                        request_prep,
                        custom_id=custom_id,
                        image_url=data_url,
                        image_info=image_info,
                        use_hierarchical_context=use_hierarchical_context,
                    )
                else:
                    request_line, metadata_record = create_batch_request_line(
                        custom_id=custom_id,
                        image_url=data_url,
                        image_info=image_info,
                        model_config=model_config,
                        schema_path=schema_path,
                        additional_context_path=additional_context_path,
                        use_hierarchical_context=use_hierarchical_context,
                    )
                batch_request_lines.append(request_line)
                metadata_records.append(metadata_record)
                all_metadata_records.append(metadata_record)
            except Exception as exc:
                item_name = (
                    image_file.name
                    if isinstance(image_file, Path)
                    else getattr(image_file, "image_name", str(image_file))
                )
                logger.error("Error processing image %s: %s", item_name, exc)
                print_error(f"Failed to process image {item_name}: {exc}")

        print_info(f"Creating batch files for chunk {chunk_start // chunk_size + 1}...")

        current_lines: list[str] = []
        current_size = 0
        current_metadata: list[dict[str, Any]] = []

        for i, line in enumerate(batch_request_lines):
            line_bytes = len(line.encode("utf-8"))

            if current_size + line_bytes > max_batch_size and current_lines:
                batch_file = Path(f"batch_requests_part_{batch_index}.jsonl")
                write_batch_file(current_lines, batch_file)
                attempted_parts += 1
                if _submit_and_cleanup_batch_file(
                    batch_file, batch_index, batch_responses
                ):
                    submitted_parts += 1

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
            attempted_parts += 1
            if _submit_and_cleanup_batch_file(batch_file, batch_index, batch_responses):
                submitted_parts += 1
            batch_index += 1
    total_parts = attempted_parts
    if submitted_parts == total_parts and total_parts > 0:
        print_info(
            f"All {total_images} images processed and submitted"
            f" in {total_parts} batch file(s)"
        )
    else:
        print_info(
            f"Submitted {submitted_parts}/{total_parts} batch file(s)"
            f" for {total_images} images"
        )
        if submitted_parts == 0:
            # Propagate failure to caller so workflow can decide on fallback
            raise RuntimeError("No batch submissions succeeded; see logs for details.")
    return batch_responses, all_metadata_records
