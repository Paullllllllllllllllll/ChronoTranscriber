"""OpenAI Batch API backend implementation.

Uses OpenAI's Batch API with the /v1/responses endpoint for async processing.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modules.llm.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
)
from modules.llm.model_capabilities import detect_capabilities
from modules.llm.structured_outputs import build_structured_text_format
from modules.llm.prompt_utils import prepare_prompt_with_context
from modules.config.constants import SUPPORTED_IMAGE_FORMATS
from modules.config.service import get_config_service

logger = logging.getLogger(__name__)

# Limits for OpenAI Batch API
MAX_BATCH_REQUESTS = 50000
MAX_BATCH_BYTES = 150 * 1024 * 1024  # 150 MB safety margin (limit is 200MB)


def _encode_image_to_data_url(image_path: Path) -> str:
    """Encode an image file as a data URL."""
    ext = image_path.suffix.lower()
    mime = SUPPORTED_IMAGE_FORMATS.get(ext)
    if not mime:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _build_responses_body(
    *,
    model_config: Dict[str, Any],
    system_prompt: str,
    image_url: str,
    transcription_schema: Optional[Dict[str, Any]] = None,
    llm_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a Responses API request body for vision transcription."""
    tm = model_config or {}
    model_name: str = tm.get("name", "gpt-4o-2024-08-06")
    caps = detect_capabilities(model_name)

    # Normalize detail
    detail_norm: Optional[str] = None
    if isinstance(llm_detail, str):
        d = llm_detail.lower().strip()
        if d in ("low", "high"):
            detail_norm = d

    # Base body
    body: Dict[str, Any] = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "The image:"},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        **(
                            {"detail": detail_norm}
                            if detail_norm and caps.supports_image_detail
                            else {}
                        ),
                    },
                ],
            },
        ],
        "max_output_tokens": int(
            tm.get("max_output_tokens")
            or tm.get("max_completion_tokens")
            or tm.get("max_tokens", 4096)
        ),
    }

    # Service tier handling
    try:
        cc = get_config_service().get_concurrency_config()
        st = (
            (cc.get("concurrency", {}) or {})
            .get("transcription", {})
            .get("service_tier")
        )
    except Exception:
        st = None
    effective_service_tier = st if st is not None else tm.get("service_tier")

    # Flex is not supported for batch - convert to auto
    if effective_service_tier:
        allowed_service_tiers = {"auto", "default", "priority"}
        tier_str = str(effective_service_tier)
        if tier_str == "flex":
            logger.info("Batch API does not support service_tier='flex'. Using 'auto'.")
            body["service_tier"] = "auto"
        elif tier_str in allowed_service_tiers:
            body["service_tier"] = tier_str

    # Structured outputs
    if transcription_schema and caps.supports_structured_outputs:
        fmt = build_structured_text_format(transcription_schema, "TranscriptionSchema", True)
        if fmt is not None:
            body.setdefault("text", {})
            body["text"]["format"] = fmt

    # Reasoning controls for GPT-5
    if caps.supports_reasoning_effort and tm.get("reasoning"):
        body["reasoning"] = tm["reasoning"]

    # Sampler controls
    if caps.supports_sampler_controls:
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            if k in tm and tm[k] is not None:
                body[k] = tm[k]

    return body


class OpenAIBatchBackend(BatchBackend):
    """OpenAI Batch API backend using /v1/responses endpoint."""

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def max_batch_size(self) -> int:
        return MAX_BATCH_REQUESTS

    @property
    def max_batch_bytes(self) -> int:
        return MAX_BATCH_BYTES

    def submit_batch(
        self,
        requests: List[BatchRequest],
        model_config: Dict[str, Any],
        *,
        system_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        schema_path: Optional[Path] = None,
        additional_context: Optional[str] = None,
    ) -> BatchHandle:
        """Submit a batch to OpenAI's Batch API."""
        client = self._get_client()

        # Load schema if path provided
        transcription_schema = None
        full_schema_obj = schema
        if schema_path and schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as f:
                full_schema_obj = json.load(f)
        if full_schema_obj:
            if isinstance(full_schema_obj, dict) and "schema" in full_schema_obj:
                transcription_schema = full_schema_obj["schema"]
            else:
                transcription_schema = full_schema_obj

        # Prepare system prompt with schema and context
        final_prompt = prepare_prompt_with_context(
            system_prompt, full_schema_obj, additional_context
        )

        # Get image detail setting
        try:
            image_cfg = get_config_service().get_image_processing_config().get("api_image_processing", {})
            raw_detail = str(image_cfg.get("llm_detail", "high")).lower().strip()
            llm_detail = raw_detail if raw_detail in ("low", "high") else None
        except Exception:
            llm_detail = None

        # Build JSONL content
        jsonl_lines = []
        for req in requests:
            # Get image URL
            if req.image_path:
                image_url = _encode_image_to_data_url(req.image_path)
            elif req.image_base64 and req.mime_type:
                image_url = f"data:{req.mime_type};base64,{req.image_base64}"
            else:
                raise ValueError(f"Request {req.custom_id} has no image data")

            # Build request body
            body = _build_responses_body(
                model_config=model_config,
                system_prompt=final_prompt,
                image_url=image_url,
                transcription_schema=transcription_schema,
                llm_detail=llm_detail,
            )

            request_line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            jsonl_lines.append(json.dumps(request_line))

        # Write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8"
        ) as f:
            for line in jsonl_lines:
                f.write(line + "\n")
            temp_path = Path(f.name)

        try:
            # Upload file
            logger.info("Uploading batch file to OpenAI...")
            with temp_path.open("rb") as f:
                file_response = client.files.create(file=f, purpose="batch")
            file_id = file_response.id
            logger.info("Uploaded batch file; file id: %s", file_id)

            # Create batch
            batch_response = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/responses",
                completion_window="24h",
                metadata={"description": "Batch transcription via BatchBackend"},
            )
            logger.info("Batch submitted; batch id: %s", batch_response.id)

            return BatchHandle(
                provider="openai",
                batch_id=batch_response.id,
                metadata={
                    "input_file_id": file_id,
                    "request_count": len(requests),
                },
            )
        finally:
            # Cleanup temp file
            try:
                temp_path.unlink()
            except Exception:
                pass

    def get_status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Get status of an OpenAI batch job."""
        client = self._get_client()

        try:
            batch = client.batches.retrieve(handle.batch_id)
        except Exception as e:
            return BatchStatusInfo(
                status=BatchStatus.UNKNOWN,
                error_message=str(e),
            )

        # Map OpenAI status to our enum
        status_map = {
            "validating": BatchStatus.PENDING,
            "in_progress": BatchStatus.IN_PROGRESS,
            "finalizing": BatchStatus.IN_PROGRESS,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "cancelled": BatchStatus.CANCELLED,
            "expired": BatchStatus.EXPIRED,
        }
        status = status_map.get(batch.status, BatchStatus.UNKNOWN)

        # Extract counts
        request_counts = getattr(batch, "request_counts", None)
        total = completed = failed = 0
        if request_counts:
            total = getattr(request_counts, "total", 0)
            completed = getattr(request_counts, "completed", 0)
            failed = getattr(request_counts, "failed", 0)

        # Get output file id
        output_file_id = getattr(batch, "output_file_id", None)

        return BatchStatusInfo(
            status=status,
            total_requests=total,
            completed_requests=completed,
            failed_requests=failed,
            pending_requests=total - completed - failed,
            results_available=status == BatchStatus.COMPLETED and output_file_id is not None,
            output_file_id=output_file_id,
        )

    def download_results(self, handle: BatchHandle) -> Iterator[BatchResultItem]:
        """Download and parse OpenAI batch results."""
        client = self._get_client()

        # Get batch to find output file
        batch = client.batches.retrieve(handle.batch_id)
        output_file_id = getattr(batch, "output_file_id", None)

        if not output_file_id:
            raise RuntimeError(f"Batch {handle.batch_id} has no output file")

        # Download content
        file_stream = client.files.content(output_file_id)
        content = file_stream.read()
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)

        # Parse JSONL lines
        for line in text.strip().split("\n"):
            if not line.strip():
                continue

            try:
                response_obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = response_obj.get("custom_id", "")
            result_item = BatchResultItem(custom_id=custom_id)

            # Parse response
            resp = response_obj.get("response", {})
            if isinstance(resp, dict):
                status_code = resp.get("status_code")
                body = resp.get("body", {})

                # Check for errors
                if status_code and status_code != 200:
                    error_obj = body.get("error", {}) if isinstance(body, dict) else {}
                    result_item.success = False
                    result_item.error = error_obj.get("message", f"HTTP {status_code}")
                    result_item.error_code = error_obj.get("code")
                    result_item.raw_response = resp
                    yield result_item
                    continue

                if isinstance(body, dict) and body.get("error"):
                    error_obj = body["error"]
                    result_item.success = False
                    result_item.error = error_obj.get("message", "Unknown error")
                    result_item.error_code = error_obj.get("code")
                    result_item.raw_response = resp
                    yield result_item
                    continue

                # Extract content from successful response
                result_item.raw_response = resp
                result_item.success = True

                # Try to extract text from Responses API format
                if isinstance(body, dict):
                    # Extract from output array
                    output = body.get("output", [])
                    for item in output if isinstance(output, list) else []:
                        if isinstance(item, dict) and item.get("type") == "message":
                            content_list = item.get("content", [])
                            for c in content_list if isinstance(content_list, list) else []:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    result_item.content = c.get("text", "")
                                    break

                    # Try to parse as JSON for structured output
                    if result_item.content:
                        try:
                            parsed = json.loads(result_item.content)
                            if isinstance(parsed, dict):
                                result_item.parsed_output = parsed
                                # Extract transcribed_text if present
                                if "transcribed_text" in parsed:
                                    result_item.content = parsed["transcribed_text"]
                        except json.JSONDecodeError:
                            pass

                    # Extract usage
                    usage = body.get("usage", {})
                    if isinstance(usage, dict):
                        result_item.input_tokens = usage.get("input_tokens", 0)
                        result_item.output_tokens = usage.get("output_tokens", 0)

            yield result_item

    def cancel(self, handle: BatchHandle) -> bool:
        """Cancel an OpenAI batch job."""
        client = self._get_client()
        try:
            client.batches.cancel(handle.batch_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel batch %s: %s", handle.batch_id, e)
            return False
