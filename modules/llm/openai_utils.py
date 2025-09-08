from __future__ import annotations

import base64
import json
import logging
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import aiofiles
import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
from modules.llm.model_capabilities import Capabilities, detect_capabilities
from modules.llm.structured_outputs import build_structured_text_format
from modules.llm.prompt_utils import render_prompt_with_schema
from modules.config.constants import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)



# ---------- Exceptions (retry control) ----------


class TransientOpenAIError(Exception):
    """Error category that is safe to retry (429/5xx/timeouts)."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NonRetryableOpenAIError(Exception):
    """Error category that should not be retried (e.g., 4xx other than 429)."""


# ---------- Retry wait helpers ----------

_WAIT_IMAGE_BASE = wait_exponential(multiplier=1, min=4, max=60)
_WAIT_TRANSCRIBE_BASE = wait_exponential(multiplier=1, min=4, max=10)


def _wait_with_server_hint_factory(base_wait):
    """Respect server-provided Retry-After if present; otherwise use base wait."""

    def _wait(retry_state):
        try:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            ra = getattr(exc, "retry_after", None)
            if isinstance(ra, (int, float)) and ra and ra > 0:
                return ra
        except Exception:
            pass
        return base_wait(retry_state)

    return _wait


# ---------- Responses API adapter ----------


class OpenAIExtractor:
    """
    Minimal, robust adapter for the OpenAI Responses API.

    - Uses typed `input` (input_text / input_image).
    - Structured outputs via `text.format` where supported.
    - GPT-5 public `reasoning` / `text.verbosity` when applicable.
    - Excludes sampler controls for reasoning families and GPT-5.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        service_tier: Optional[str] = None,
        timeout: Optional[float] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided.")
        if not model:
            raise ValueError("model must be provided.")

        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.openai.com/v1/responses"
        self.service_tier = service_tier

        self.caps: Capabilities = detect_capabilities(model)

        # Load model configuration dictionary (preserving legacy structure)
        if model_config is None:
            cl = ConfigLoader()
            cl.load_configs()
            mc = cl.get_model_config()
        else:
            mc = model_config

        tm = mc.get("transcription_model") or mc.get("extraction_model") or {}

        # Token budget: Responses uses max_output_tokens (fallbacks for backward-compat)
        self.max_output_tokens: int = int(
            tm.get("max_output_tokens")
            if tm.get("max_output_tokens") is not None
            else (
                tm.get("max_completion_tokens")
                if tm.get("max_completion_tokens") is not None
                else tm.get("max_tokens", 4096)
            )
        )

        # Classic sampler controls (only used on non-reasoning, non-GPT-5)
        self.temperature: float = float(tm.get("temperature", 0.0))
        self.top_p: float = float(tm.get("top_p", 1.0))
        self.presence_penalty: float = float(tm.get("presence_penalty", 0.0))
        self.frequency_penalty: float = float(tm.get("frequency_penalty", 0.0))
        self.stop = tm.get("stop") or []
        self.seed = tm.get("seed")

        # GPT-5 specifics (only attached if caps.supports_reasoning_effort is True)
        self.reasoning: Dict[str, Any] = tm.get("reasoning", {"effort": "medium"})
        self.text_params: Dict[str, Any] = tm.get("text", {"verbosity": "medium"})

        # Timeouts: explicit per-stage; longer for "flex" (tolerate connector queueing)
        if timeout is not None:
            total_timeout = float(timeout)
            connect_timeout = 30.0
            sock_connect_timeout = 30.0
            sock_read = 600.0
            if service_tier == "flex":
                connect_timeout = 180.0
                sock_connect_timeout = 180.0
                sock_read = 900.0
        else:
            if service_tier == "flex":
                total_timeout = 1800.0
                connect_timeout = 180.0
                sock_connect_timeout = 180.0
                sock_read = 1200.0
            else:
                # Default tier: raise connect timeouts to better tolerate pool queueing under load
                total_timeout = 900.0
                connect_timeout = 120.0
                sock_connect_timeout = 120.0
                sock_read = 600.0
        client_timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_connect=sock_connect_timeout,
            sock_read=sock_read,
        )
        # Align connector pool with configured transcription concurrency to avoid queue wait timeouts
        try:
            conc_cfg = ConfigLoader().get_concurrency_config()
            trans_cfg = conc_cfg.get("concurrency", {}).get("transcription", {})
            conn_limit = int(trans_cfg.get("concurrency_limit", 100))
            if conn_limit <= 0:
                conn_limit = 100
        except Exception:
            conn_limit = 100
        connector = aiohttp.TCPConnector(limit=conn_limit, limit_per_host=conn_limit)
        self.session = aiohttp.ClientSession(timeout=client_timeout, connector=connector)

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with self.session.post(self.endpoint, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                retry_after_val: Optional[float] = None
                if resp.status == 429 or 500 <= resp.status < 600:
                    # Respect Retry-After when provided by the server
                    ra_hdr = resp.headers.get("Retry-After")
                    if ra_hdr is not None:
                        try:
                            retry_after_val = float(ra_hdr)
                        except Exception:
                            # Retry-After may be an HTTP-date; convert to seconds if in the future
                            try:
                                dt = parsedate_to_datetime(ra_hdr)
                                if dt is not None:
                                    delta = (
                                        dt - datetime.now(timezone.utc)
                                    ).total_seconds()
                                    if delta and delta > 0:
                                        retry_after_val = delta
                            except Exception:
                                retry_after_val = None
                    logger.warning(
                        "Transient OpenAI error (%s): %s", resp.status, error_text
                    )
                    raise TransientOpenAIError(
                        f"{resp.status}: {error_text}", retry_after=retry_after_val
                    )
                logger.error("Non-retryable OpenAI error (%s): %s", resp.status, error_text)
                raise NonRetryableOpenAIError(f"{resp.status}: {error_text}")
            return await resp.json()

    def _build_base_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.service_tier:
            payload["service_tier"] = self.service_tier

        # GPT-5 public reasoning/text controls
        if self.caps.supports_reasoning_effort:
            payload["reasoning"] = self.reasoning
            if (
                isinstance(self.text_params, dict)
                and self.text_params.get("verbosity") is not None
            ):
                payload.setdefault("text", {})["verbosity"] = self.text_params["verbosity"]

        # Sampler controls only for non-reasoning, non-GPT-5 families
        if self.caps.supports_sampler_controls:
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
            if self.stop:
                payload["stop"] = self.stop

        return payload

    def _maybe_add_text_format(
        self, payload: Dict[str, Any], json_schema: Optional[Dict[str, Any]]
    ) -> None:
        """
        Add `text.format` (Structured Outputs) where supported (avoid on o-series).
        """
        if not json_schema or not self.caps.supports_structured_outputs:
            return

        fmt = build_structured_text_format(json_schema, "TranscriptionSchema", True)
        if fmt is None:
            return

        payload.setdefault("text", {})
        payload["text"]["format"] = fmt

    @staticmethod
    def _collect_output_text(data: Dict[str, Any]) -> str:
        """
        Normalize Responses output into a single text string.
        """
        if isinstance(data, dict) and isinstance(data.get("output_text"), str):
            return data["output_text"].strip()

        parts: list[str] = []
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message":
                    for c in item.get("content", []):
                        t = c.get("text")
                        if isinstance(t, str):
                            parts.append(t)
        return "".join(parts).strip()

    @retry(
        wait=_wait_with_server_hint_factory(_WAIT_IMAGE_BASE),
        stop=stop_after_attempt(5),
        retry=(
            retry_if_exception_type(TransientOpenAIError)
            | retry_if_exception_type(aiohttp.ClientError)
            | retry_if_exception_type(asyncio.TimeoutError)
        ),
    )
    async def process_image(
        self,
        *,
        system_message: str,
        image_data_url: str,
        json_schema: Optional[Dict[str, Any]] = None,
        user_instruction: str = "Please transcribe the text from this image.",
        detail: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call Responses API with an image and optional JSON schema.

        Returns
        -------
        tuple[str, dict]
            (content_text, raw_response_dict)
        """
        if not self.caps.supports_image_input:
            raise NonRetryableOpenAIError(
                f"Selected model '{self.model}' does not support image inputs."
            )

        # Normalize detail: when explicitly set to 'auto', omit the field.
        detail_norm = detail or None
        if isinstance(detail_norm, str):
            dlow = detail_norm.lower().strip()
            if dlow == "auto":
                detail_norm = None
            elif dlow in ("low", "high"):
                detail_norm = dlow
            else:
                detail_norm = None

        effective_detail = (
            detail_norm
            if detail_norm is not None
            else (self.caps.default_ocr_detail if self.caps.supports_image_detail else None)
        )

        image_part: Dict[str, Any] = {
            "type": "input_image",
            # Responses API expects image_url as a STRING (URL or data URL)
            # and optional detail as a sibling property.
            "image_url": image_data_url,
        }
        if detail_norm is not None and self.caps.supports_image_detail:
            image_part["detail"] = effective_detail

        input_messages = [
            {"role": "system", "content": [{"type": "input_text", "text": system_message}]},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_instruction}, image_part],
            },
        ]

        payload = self._build_base_payload()
        payload["input"] = input_messages

        # Add structured outputs when supported
        self._maybe_add_text_format(payload, json_schema)

        logger.debug(
            "Submitting Responses image request: model=%s, has_text_format=%s, service_tier=%s",
            self.model,
            "text" in payload and isinstance(payload["text"], dict) and "format" in payload["text"],
            self.service_tier,
        )
        logger.debug(
            "Responses image call: model=%s include_detail=%s effective_detail=%s",
            self.model,
            detail_norm is not None and self.caps.supports_image_detail,
            effective_detail,
        )
        data = await self._post(payload)
        content_text = self._collect_output_text(data)
        return content_text, data


# ---------- Backward-compatible façade (same public interface) ----------


class OpenAITranscriber:
    """
    Backward-compatible façade used by existing workflow code.

    Preserves:
      - constructor signature (api_key, model)
      - `transcribe_image(Path)` method
    but internally routes to the Responses API via `OpenAIExtractor`.
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        *,
        schema_path: Optional[Path] = None,
        system_prompt_path: Optional[Path] = None,
    ) -> None:
        cfg = ConfigLoader()
        cfg.load_configs()

        mc = cfg.get_model_config()
        tm = mc.get("transcription_model", {})
        self.model = model or tm.get("name", "gpt-4o-2024-08-06")

        self.api_key = api_key
        # service_tier now sourced from concurrency_config.yaml
        try:
            cc = cfg.get_concurrency_config()
            st = (
                (cc.get("concurrency", {}) or {})
                .get("transcription", {})
                .get("service_tier")
            )
        except Exception:
            st = None
        # Backward-compat fallback to model_config.yaml if present
        self.service_tier: Optional[str] = st if st is not None else tm.get("service_tier")

        self.extractor = OpenAIExtractor(
            api_key=api_key,
            model=self.model,
            service_tier=self.service_tier,
            timeout=None,
            model_config=mc,
        )

        # Load image processing config for LLM image detail
        ipc = cfg.get_image_processing_config()
        self.image_cfg = (
            ipc.get("api_image_processing", {}) if isinstance(ipc, dict) else {}
        )
        raw_detail = str(self.image_cfg.get("llm_detail", "high")).lower().strip()
        self.llm_detail: Optional[str]
        if raw_detail in ("low", "high"):
            self.llm_detail = raw_detail
        elif raw_detail == "auto":
            self.llm_detail = "auto"
        else:
            # Fallback to model capability default if misconfigured
            self.llm_detail = "auto"

        # Resolve prompt/schema with priority: explicit args -> config overrides -> defaults
        pcfg = cfg.get_paths_config()
        general = pcfg.get("general", {})
        override_prompt = general.get("transcription_prompt_path")
        override_schema = general.get("transcription_schema_path")
        self.system_prompt_path = (
            Path(system_prompt_path)
            if system_prompt_path is not None
            else (
                Path(override_prompt)
                if override_prompt
                else (PROJECT_ROOT / "system_prompt" / "system_prompt.txt")
            )
        )
        self.schema_path = (
            Path(schema_path)
            if schema_path is not None
            else (
                Path(override_schema)
                if override_schema
                else (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json")
            )
        )

        if not self.system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt missing: {self.system_prompt_path}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file missing: {self.schema_path}")
        # Load prompt and schema, then inject schema into prompt at runtime
        raw_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()
        with self.schema_path.open("r", encoding="utf-8") as sf:
            loaded_schema = json.load(sf)
            full_schema_obj = loaded_schema
            # Accept wrapper form {name, strict, schema: {...}} or bare schema {...}
            if (
                isinstance(loaded_schema, dict)
                and "schema" in loaded_schema
                and isinstance(loaded_schema["schema"], dict)
            ):
                self.transcription_schema = loaded_schema["schema"]
            else:
                self.transcription_schema = loaded_schema

        # Render system prompt with current schema content
        self.system_prompt_text = render_prompt_with_schema(raw_prompt, full_schema_obj)

    async def close(self) -> None:
        await self.extractor.close()

    @staticmethod
    async def _encode_image_to_data_url(image_path: Path, mime_type: str) -> str:
        """
        Async file read + Base64 to data URL.
        """
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @retry(
        wait=_wait_with_server_hint_factory(_WAIT_TRANSCRIBE_BASE),
        stop=stop_after_attempt(7),
        retry=(
            retry_if_exception_type(TransientOpenAIError)
            | retry_if_exception_type(aiohttp.ClientError)
            | retry_if_exception_type(asyncio.TimeoutError)
        ),
    )
    async def transcribe_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Backward-compatible call used by workflow.

        Returns
        -------
        dict
            Raw Responses dict (safest for downstream parsers).
        """
        mime = SUPPORTED_IMAGE_FORMATS.get(image_path.suffix.lower())
        if not mime:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        data_url = await self._encode_image_to_data_url(image_path, mime)

        _, data = await self.extractor.process_image(
            system_message=self.system_prompt_text,
            image_data_url=data_url,
            # Pass bare JSON Schema; helper will wrap if needed
            json_schema=self.transcription_schema,
            user_instruction="Please analyze and transcribe the text from this image according to the provided instructions.",
            # Use configured detail; 'auto' will omit the field
            detail=self.llm_detail,
        )
        return data


@asynccontextmanager
async def open_transcriber(
    api_key: str,
    model: Optional[str] = None,
    *,
    schema_path: Optional[Path] = None,
    system_prompt_path: Optional[Path] = None,
) -> AsyncGenerator[OpenAITranscriber, None]:
    """
    Context manager matching legacy import sites.
    """
    transcriber = OpenAITranscriber(
        api_key=api_key,
        model=model,
        schema_path=schema_path,
        system_prompt_path=system_prompt_path,
    )
    try:
        yield transcriber
    finally:
        await transcriber.close()


async def transcribe_image_with_openai(
    image_path: Path, transcriber: OpenAITranscriber
) -> Dict[str, Any]:
    """
    Legacy helper retained for callers in the workflow.
    """
    return await transcriber.transcribe_image(image_path)