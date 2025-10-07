from __future__ import annotations

import base64
import json
import logging
import asyncio
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import aiofiles
import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_random

from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
from modules.llm.model_capabilities import Capabilities, detect_capabilities
from modules.llm.structured_outputs import build_structured_text_format
from modules.llm.prompt_utils import render_prompt_with_schema, inject_additional_context
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


class TranscriptionFailureError(Exception):
    """Raised when the model returns a transcription failure indicator.
    
    Used to trigger retries for specific transcription outcomes like:
    - no_transcribable_text: true
    - transcription_not_possible: true
    """
    
    def __init__(self, message: str, failure_type: str) -> None:
        super().__init__(message)
        self.failure_type = failure_type  # 'no_transcribable_text' or 'transcription_not_possible'


# ---------- Retry wait helpers ----------

def _load_retry_policy() -> Tuple[int, float, float, float]:
    """Load retry attempts and wait window from concurrency configuration.

    Returns
    -------
    tuple
        (attempts, wait_min_seconds, wait_max_seconds, jitter_max_seconds)
    """
    try:
        conc_cfg = ConfigLoader().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = (trans_cfg.get("retry", {}) or {})
        attempts = int(retry_cfg.get("attempts", 5))
        wait_min = float(retry_cfg.get("wait_min_seconds", 4))
        wait_max = float(retry_cfg.get("wait_max_seconds", 60))
        jitter_max = float(retry_cfg.get("jitter_max_seconds", 1))
        # Sanity bounds
        if attempts <= 0:
            attempts = 1
        if wait_min < 0:
            wait_min = 0
        if wait_max < wait_min:
            wait_max = wait_min
        if jitter_max < 0:
            jitter_max = 0
        return attempts, wait_min, wait_max, jitter_max
    except Exception:
        # Fallback to sensible defaults matching prior behavior
        return 5, 4.0, 60.0, 1.0


def _load_transcription_failure_retry_policy() -> Tuple[int, int, float, float, float]:
    """Load transcription failure retry configuration.
    
    Returns
    -------
    tuple
        (no_text_retries, not_possible_retries, wait_min, wait_max, jitter_max)
    """
    try:
        conc_cfg = ConfigLoader().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = (trans_cfg.get("retry", {}) or {})
        tf_cfg = (retry_cfg.get("transcription_failures", {}) or {})
        
        no_text_retries = int(tf_cfg.get("no_transcribable_text_retries", 0))
        not_possible_retries = int(tf_cfg.get("transcription_not_possible_retries", 0))
        wait_min = float(tf_cfg.get("wait_min_seconds", 2))
        wait_max = float(tf_cfg.get("wait_max_seconds", 30))
        jitter_max = float(tf_cfg.get("jitter_max_seconds", 1))
        
        # Sanity bounds
        if no_text_retries < 0:
            no_text_retries = 0
        if not_possible_retries < 0:
            not_possible_retries = 0
        if wait_min < 0:
            wait_min = 0
        if wait_max < wait_min:
            wait_max = wait_min
        if jitter_max < 0:
            jitter_max = 0
            
        return no_text_retries, not_possible_retries, wait_min, wait_max, jitter_max
    except Exception:
        # Fallback: no retries for transcription failures
        return 0, 0, 2.0, 30.0, 1.0


_RETRY_ATTEMPTS, _RETRY_WAIT_MIN, _RETRY_WAIT_MAX, _RETRY_JITTER_MAX = _load_retry_policy()

# Load transcription failure retry configuration
(
    _TF_NO_TEXT_RETRIES,
    _TF_NOT_POSSIBLE_RETRIES,
    _TF_WAIT_MIN,
    _TF_WAIT_MAX,
    _TF_JITTER_MAX,
) = _load_transcription_failure_retry_policy()

# Base exponential wait, augmented with small random jitter to avoid synchronized retries.
_WAIT_IMAGE_BASE = (
    wait_exponential(multiplier=1, min=_RETRY_WAIT_MIN, max=_RETRY_WAIT_MAX)
    + wait_random(0, _RETRY_JITTER_MAX)
)

# Wait strategy for transcription failure retries
_WAIT_TF_BASE = (
    wait_exponential(multiplier=1, min=_TF_WAIT_MIN, max=_TF_WAIT_MAX)
    + wait_random(0, _TF_JITTER_MAX)
)


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


def _check_transcription_failure(content_text: str, raw_response: Dict[str, Any]) -> Optional[str]:
    """Check if the response indicates a transcription failure.
    
    Returns
    -------
    Optional[str]
        The failure type ('no_transcribable_text' or 'transcription_not_possible') if detected, None otherwise.
    """
    # Try to parse the response as JSON
    parsed = None
    if content_text:
        stripped = content_text.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # Try to salvage the last valid JSON object
                last_close = stripped.rfind("}")
                if last_close != -1:
                    i = last_close
                    while i >= 0:
                        if stripped[i] == "{":
                            candidate = stripped[i:last_close + 1]
                            try:
                                parsed = json.loads(candidate)
                                break
                            except json.JSONDecodeError:
                                pass
                        i -= 1
    
    # Check for failure flags in parsed JSON
    if isinstance(parsed, dict):
        if parsed.get("no_transcribable_text", False):
            return "no_transcribable_text"
        if parsed.get("transcription_not_possible", False):
            return "transcription_not_possible"
    
    return None


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

        # Load model configuration dictionary
        if model_config is None:
            cl = ConfigLoader()
            cl.load_configs()
            mc = cl.get_model_config()
        else:
            mc = model_config

        tm = mc.get("transcription_model") or mc.get("extraction_model") or {}

        # Token budget: Responses API uses max_output_tokens (with fallbacks for flexibility)
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
            
            # Parse response and capture token usage
            data = await resp.json()
            
            # Report token usage immediately after successful API call
            try:
                usage = data.get("usage")
                if isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens")
                    if isinstance(total_tokens, int) and total_tokens > 0:
                        from modules.token_tracker import get_token_tracker
                        token_tracker = get_token_tracker()
                        token_tracker.add_tokens(total_tokens)
                        logger.debug(
                            f"[TOKEN] API call consumed {total_tokens:,} tokens "
                            f"(daily total: {token_tracker.get_tokens_used_today():,})"
                        )
            except Exception as e:
                logger.warning(f"Error reporting token usage: {e}")
            
            return data

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
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
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

    async def process_image_with_transcription_retry(
        self,
        *,
        system_message: str,
        image_data_url: str,
        json_schema: Optional[Dict[str, Any]] = None,
        user_instruction: str = "Please transcribe the text from this image.",
        detail: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call Responses API with an image and retry on transcription failures.
        
        This wraps process_image() and adds an additional retry layer for transcription-specific
        failures (no_transcribable_text, transcription_not_possible) based on configuration.
        
        The retry strategy follows OpenAI's recommendations:
        - General API errors (429, 5xx, timeouts) are handled by process_image with exponential backoff
        - Transcription failures are retried separately with their own configurable limits
        
        Returns
        -------
        tuple[str, dict]
            (content_text, raw_response_dict)
        """
        # Track retry attempts for each failure type
        no_text_attempts = 0
        not_possible_attempts = 0
        
        while True:
            # Call the API (with its own retry logic for transient errors)
            content_text, raw_response = await self.process_image(
                system_message=system_message,
                image_data_url=image_data_url,
                json_schema=json_schema,
                user_instruction=user_instruction,
                detail=detail,
            )
            
            # Check if the response indicates a transcription failure
            failure_type = _check_transcription_failure(content_text, raw_response)
            
            if failure_type is None:
                # Success - return the result
                return content_text, raw_response
            
            # Determine if we should retry based on the failure type and configured limits
            should_retry = False
            wait_time = 0.0
            
            if failure_type == "no_transcribable_text":
                if no_text_attempts < _TF_NO_TEXT_RETRIES:
                    should_retry = True
                    no_text_attempts += 1
                    # Calculate exponential backoff with jitter
                    base_wait = min(_TF_WAIT_MIN * (2 ** (no_text_attempts - 1)), _TF_WAIT_MAX)
                    wait_time = base_wait + random.uniform(0, _TF_JITTER_MAX)
                    logger.warning(
                        "Transcription returned no_transcribable_text=true (attempt %d/%d). "
                        "Retrying after %.2f seconds...",
                        no_text_attempts,
                        _TF_NO_TEXT_RETRIES,
                        wait_time,
                    )
            elif failure_type == "transcription_not_possible":
                if not_possible_attempts < _TF_NOT_POSSIBLE_RETRIES:
                    should_retry = True
                    not_possible_attempts += 1
                    # Calculate exponential backoff with jitter
                    base_wait = min(_TF_WAIT_MIN * (2 ** (not_possible_attempts - 1)), _TF_WAIT_MAX)
                    wait_time = base_wait + random.uniform(0, _TF_JITTER_MAX)
                    logger.warning(
                        "Transcription returned transcription_not_possible=true (attempt %d/%d). "
                        "Retrying after %.2f seconds...",
                        not_possible_attempts,
                        _TF_NOT_POSSIBLE_RETRIES,
                        wait_time,
                    )
            
            if not should_retry:
                # Exhausted retries or retries disabled - return the failure response
                logger.info(
                    "Transcription failure (%s) - no more retries configured. Returning response.",
                    failure_type,
                )
                return content_text, raw_response
            
            # Wait before retrying
            await asyncio.sleep(wait_time)


# ---------- Public transcriber interface ----------


class OpenAITranscriber:
    """
    High-level transcriber interface for the ChronoTranscriber workflow.

    Encapsulates:
      - Configuration loading (model, prompts, schema)
      - Image encoding to data URLs
      - Transcription via OpenAI Responses API with retry logic
    
    Used by workflow orchestration and repair operations.
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        *,
        schema_path: Optional[Path] = None,
        system_prompt_path: Optional[Path] = None,
        additional_context_path: Optional[Path] = None,
    ) -> None:
        cfg = ConfigLoader()
        cfg.load_configs()

        mc = cfg.get_model_config()
        tm = mc.get("transcription_model", {})
        self.model = model or tm.get("name", "gpt-4o-2024-08-06")

        self.api_key = api_key
        # service_tier sourced from concurrency_config.yaml with fallback to model_config.yaml
        try:
            cc = cfg.get_concurrency_config()
            st = (
                (cc.get("concurrency", {}) or {})
                .get("transcription", {})
                .get("service_tier")
            )
        except Exception:
            st = None
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
        
        # Inject additional context if provided (fail-safe: no context = empty replacement)
        if additional_context_path is not None and additional_context_path.exists():
            try:
                additional_context = additional_context_path.read_text(encoding="utf-8").strip()
                self.system_prompt_text = inject_additional_context(
                    self.system_prompt_text, additional_context
                )
            except Exception as e:
                logger.warning(
                    "Failed to load additional context from %s: %s",
                    additional_context_path,
                    e,
                )
                # Fail-safe: remove marker with empty string
                self.system_prompt_text = inject_additional_context(self.system_prompt_text, "")
        else:
            # No context provided: remove marker with empty string
            self.system_prompt_text = inject_additional_context(self.system_prompt_text, "")

    async def close(self) -> None:
        await self.extractor.close()

    async def transcribe_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Transcribe an image file using the OpenAI Responses API.

        Returns
        -------
        dict
            Raw Responses API response dictionary.
        """
        mime = SUPPORTED_IMAGE_FORMATS.get(image_path.suffix.lower())
        if not mime:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        data_url = await self._encode_image_to_data_url(image_path, mime)

        _, data = await self.extractor.process_image_with_transcription_retry(
            system_message=self.system_prompt_text,
            image_data_url=data_url,
            json_schema=self.transcription_schema,
            user_instruction="The image:",
            detail=self.llm_detail,
        )
        return data

    @staticmethod
    async def _encode_image_to_data_url(image_path: Path, mime_type: str) -> str:
        """
        Async file read + Base64 to data URL.
        """
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

@asynccontextmanager
async def open_transcriber(
    api_key: str,
    model: Optional[str] = None,
    *,
    schema_path: Optional[Path] = None,
    system_prompt_path: Optional[Path] = None,
    additional_context_path: Optional[Path] = None,
) -> AsyncGenerator[OpenAITranscriber, None]:
    """
    Context manager for OpenAITranscriber with automatic cleanup.
    
    Ensures the underlying HTTP session is properly closed.
    """
    transcriber = OpenAITranscriber(
        api_key=api_key,
        model=model,
        schema_path=schema_path,
        system_prompt_path=system_prompt_path,
        additional_context_path=additional_context_path,
    )
    try:
        yield transcriber
    finally:
        await transcriber.close()


async def transcribe_image_with_openai(
    image_path: Path, transcriber: OpenAITranscriber
) -> Dict[str, Any]:
    """
    Convenience helper for transcribing a single image.
    
    Used by workflow orchestration code.
    """
    return await transcriber.transcribe_image(image_path)