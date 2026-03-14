"""Base provider abstraction for LLM integrations.

Defines the common interface that all LLM providers must implement.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.llm.model_capabilities import Capabilities

logger = setup_logger(__name__)


class InputTokensBelowThresholdError(Exception):
    """Raised when an image request returns fewer input tokens than expected.

    This signals that the API silently dropped the image payload and the
    response likely contains cross-contaminated content from another session.
    """

    def __init__(self, actual: int, threshold: int) -> None:
        self.actual = actual
        self.threshold = threshold
        super().__init__(
            f"Input tokens ({actual}) below minimum threshold ({threshold}); "
            f"image payload may have been dropped"
        )


def load_max_retries() -> int:
    """Load max retries from concurrency_config.yaml.

    Used by all LangChain-based providers to configure retry attempts.
    LangChain handles retry logic internally with exponential backoff.
    """
    try:
        conc_cfg = get_config_service().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = trans_cfg.get("retry", {}) or {}
        attempts = int(retry_cfg.get("attempts", 5))
        return max(1, attempts)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.debug("Could not load max_retries from config, using default: %s", e)
        return 5


def load_max_validation_retries() -> int:
    """Load max validation retries from concurrency_config.yaml.

    Controls how many times to retry when the model returns unparseable
    structured output (pydantic.ValidationError).  Defaults to 3 since
    these indicate model behavior issues rather than transient
    infrastructure problems.
    """
    try:
        conc_cfg = get_config_service().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = trans_cfg.get("retry", {}) or {}
        attempts = int(retry_cfg.get("validation_attempts", 3))
        return max(1, attempts)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.debug("Could not load max_validation_retries from config, using default: %s", e)
        return 3


def load_min_input_tokens() -> int:
    """Load minimum expected input tokens for image requests.

    Responses below this threshold are treated as cross-contaminated
    (image payload silently dropped by the API) and retried.
    Defaults to 500.  Set to 0 to disable the check.
    """
    try:
        conc_cfg = get_config_service().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = trans_cfg.get("retry", {}) or {}
        value = int(retry_cfg.get("min_input_tokens", 500))
        return max(0, value)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.debug("Could not load min_input_tokens from config, using default: %s", e)
        return 500


@dataclass(frozen=True)
class TokenUsageMapping:
    """Maps provider-specific token usage keys in LangChain response_metadata.
    
    Each provider stores token counts under different key paths. This mapping
    allows _process_llm_response() to extract them generically.
    """
    usage_key: str = "token_usage"
    input_key: str = "prompt_tokens"
    output_key: str = "completion_tokens"
    total_key: str = "total_tokens"


# Pre-built mappings for each supported provider
OPENAI_TOKEN_MAPPING = TokenUsageMapping()  # defaults match OpenAI
ANTHROPIC_TOKEN_MAPPING = TokenUsageMapping(
    usage_key="usage",
    input_key="input_tokens",
    output_key="output_tokens",
    total_key="",  # Anthropic doesn't provide total; compute from in+out
)
GOOGLE_TOKEN_MAPPING = TokenUsageMapping(
    usage_key="usage_metadata",
    input_key="prompt_token_count",
    output_key="candidates_token_count",
    total_key="total_token_count",
)
OPENROUTER_TOKEN_MAPPING = TokenUsageMapping()  # same as OpenAI


# Backward-compatibility alias — existing code that imports ProviderCapabilities
# from this module will get the unified Capabilities class instead.
ProviderCapabilities = Capabilities


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    
    # Core result
    content: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    # Parsed structured output (if schema was provided)
    parsed_output: Dict[str, Any] | None = None
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cache token usage (prompt caching)
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_hit: bool = False
    
    # Transcription status flags (from schema response)
    no_transcribable_text: bool = False
    transcription_not_possible: bool = False
    
    # Error information
    error: str | None = None
    
    def __post_init__(self) -> None:
        """Parse transcription status flags from content if available."""
        if self.content and not self.parsed_output:
            try:
                stripped = self.content.strip()
                if stripped.startswith("{"):
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        self.parsed_output = parsed
                        self.no_transcribable_text = parsed.get("no_transcribable_text", False)
                        self.transcription_not_possible = parsed.get("transcription_not_possible", False)
            except json.JSONDecodeError:
                pass


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.
    
    All providers must implement these methods to work with the transcription pipeline.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the provider.
        
        Args:
            api_key: API key for the provider
            model: Model name/identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_config = kwargs

        # Load prompt caching configuration
        try:
            caching_cfg = get_config_service().get_prompt_caching_config()
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.debug("Could not load prompt caching config, disabling: %s", e)
            caching_cfg = {"enabled": False}
        self._caching_enabled: bool = bool(caching_cfg.get("enabled", False))
        self._caching_config: Dict[str, Any] = caching_cfg
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider/model combination."""
        pass
    
    async def transcribe_image(
        self,
        image_path: Path,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: Dict[str, Any] | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe text from an image file.
        
        Encodes the image to base64 and delegates to transcribe_image_from_base64.
        Subclasses typically only need to override transcribe_image_from_base64.
        
        Args:
            image_path: Path to the image file
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            media_resolution: Media resolution for Google ("low", "medium", "high", "ultra_high", "auto")
        
        Returns:
            TranscriptionResult with the transcription and metadata
        """
        base64_data, mime_type = self.encode_image_to_base64(image_path)
        return await self.transcribe_image_from_base64(
            image_base64=base64_data,
            mime_type=mime_type,
            system_prompt=system_prompt,
            user_instruction=user_instruction,
            json_schema=json_schema,
            image_detail=image_detail,
            media_resolution=media_resolution,
        )
    
    @abstractmethod
    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: Dict[str, Any] | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image data
            mime_type: MIME type of the image (e.g., "image/jpeg")
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            media_resolution: Media resolution for Google ("low", "medium", "high", "ultra_high", "auto")
        
        Returns:
            TranscriptionResult with the transcription and metadata
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g., HTTP sessions)."""
        pass
    
    def _normalize_list_content(self, content_list: list) -> str:
        """Normalize list-type content from LLM response to a string.
        
        Override in subclasses for provider-specific list content handling.
        Default behavior: convert to string representation.
        
        Args:
            content_list: List content from the LLM response.
            
        Returns:
            Normalized string content.
        """
        return str(content_list)

    def _extract_content(
        self,
        response: Any,
    ) -> tuple[str, dict[str, Any] | None, Any]:
        """Extract content, parsed_output, and raw_message from an LLM response.

        Returns:
            Tuple of (content_string, parsed_output_or_none, raw_message_or_none).
        """
        parsed_output: dict[str, Any] | None = None
        raw_message = None

        if isinstance(response, dict) and "raw" in response and "parsed" in response:
            # with_structured_output(include_raw=True) → {"raw": AIMessage, "parsed": ...}
            raw_message = response.get("raw")
            parsed_data = response.get("parsed")

            if parsed_data is not None:
                if hasattr(parsed_data, "model_dump"):
                    # Pydantic model
                    content = parsed_data.model_dump_json()
                    parsed_output = parsed_data.model_dump()
                elif isinstance(parsed_data, dict):
                    content = json.dumps(parsed_data)
                    parsed_output = parsed_data
                else:
                    content = str(parsed_data)
            else:
                # Parsing failed — fall back to raw message content
                content = (
                    raw_message.content
                    if raw_message and hasattr(raw_message, "content")
                    else ""
                )
                if isinstance(content, dict):
                    parsed_output = content
                    content = json.dumps(content)
                elif isinstance(content, list):
                    content = self._normalize_list_content(content)
        elif hasattr(response, "content"):
            # Standard AIMessage response (no structured output)
            raw_message = response
            content = response.content
            if isinstance(content, dict):
                parsed_output = content
                content = json.dumps(content)
            elif isinstance(content, list):
                content = self._normalize_list_content(content)
            elif not isinstance(content, str):
                content = str(content)
        elif isinstance(response, dict):
            content = json.dumps(response)
            parsed_output = response
        else:
            content = str(response)

        return content, parsed_output, raw_message

    @staticmethod
    def _extract_token_usage(
        raw_message: Any,
        token_mapping: TokenUsageMapping,
    ) -> tuple[int, int, int, int, int]:
        """Extract token counts from the raw LLM message.

        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens,
                      cached_input_tokens, cache_creation_tokens).
        """
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cached_input_tokens = 0
        cache_creation_tokens = 0

        if raw_message is None:
            return input_tokens, output_tokens, total_tokens, cached_input_tokens, cache_creation_tokens

        # Extract token usage from response_metadata
        if hasattr(raw_message, "response_metadata"):
            metadata = raw_message.response_metadata
            if isinstance(metadata, dict):
                usage = metadata.get(token_mapping.usage_key, {})
                if isinstance(usage, dict):
                    input_tokens = usage.get(token_mapping.input_key, 0)
                    output_tokens = usage.get(token_mapping.output_key, 0)
                    if token_mapping.total_key:
                        total_tokens = usage.get(token_mapping.total_key, 0)
                    if total_tokens == 0:
                        total_tokens = input_tokens + output_tokens

                    # Extract cache tokens from provider-specific usage dicts:
                    # Anthropic: usage.cache_read_input_tokens / cache_creation_input_tokens
                    cached_input_tokens = int(
                        usage.get("cache_read_input_tokens", 0) or 0
                    )
                    cache_creation_tokens = int(
                        usage.get("cache_creation_input_tokens", 0) or 0
                    )

                    # OpenAI: token_usage.prompt_tokens_details.cached_tokens
                    if cached_input_tokens == 0:
                        prompt_details = usage.get("prompt_tokens_details")
                        if isinstance(prompt_details, dict):
                            cached_input_tokens = int(
                                prompt_details.get("cached_tokens", 0) or 0
                            )

        # Fallback: LangChain 1.x stores token usage in AIMessage.usage_metadata
        # (a UsageMetadata TypedDict — i.e. a plain dict) rather than
        # response_metadata["token_usage"].  This is the primary path for the
        # OpenAI Responses API where response_metadata has no "token_usage" key.
        if total_tokens == 0:
            usage_meta = getattr(raw_message, "usage_metadata", None)
            if usage_meta is not None:
                # UsageMetadata is a TypedDict (plain dict), so use dict
                # access (.get) rather than getattr which cannot read dict keys.
                if isinstance(usage_meta, dict):
                    input_tokens = int(usage_meta.get("input_tokens", 0) or 0)
                    output_tokens = int(usage_meta.get("output_tokens", 0) or 0)
                    total_tokens = int(usage_meta.get("total_tokens", 0) or 0)
                else:
                    # Defensive: support object-attribute style (e.g. MagicMock in tests)
                    input_tokens = int(getattr(usage_meta, "input_tokens", 0) or 0)
                    output_tokens = int(getattr(usage_meta, "output_tokens", 0) or 0)
                    total_tokens = int(getattr(usage_meta, "total_tokens", 0) or 0)
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens

                # Extract cache tokens from usage_metadata
                if cached_input_tokens == 0 and isinstance(usage_meta, dict):
                    details = usage_meta.get("input_token_details")
                    if isinstance(details, dict):
                        cached_input_tokens = int(
                            details.get("cache_read", 0) or 0
                        )
                        cache_creation_tokens = int(
                            details.get("cache_creation", 0) or 0
                        )

        return input_tokens, output_tokens, total_tokens, cached_input_tokens, cache_creation_tokens

    def _track_token_usage(
        self,
        total_tokens: int,
        cached_input_tokens: int,
        input_tokens: int,
    ) -> None:
        """Record token consumption in the daily tracker."""
        if total_tokens > 0:
            try:
                from modules.infra.token_tracker import get_token_tracker
                token_tracker = get_token_tracker()
                token_tracker.add_tokens(total_tokens)
                cache_msg = ""
                if cached_input_tokens > 0 and input_tokens > 0:
                    pct = cached_input_tokens / input_tokens * 100
                    cache_msg = f", cache hit {pct:.0f}%"
                logger.debug(
                    f"[TOKEN] API call consumed {total_tokens:,} tokens "
                    f"(daily total: {token_tracker.get_tokens_used_today():,}"
                    f"{cache_msg})"
                )
            except Exception as e:
                logger.warning(f"Error tracking tokens: {e}")

    async def _process_llm_response(
        self,
        response: Any,
        token_mapping: TokenUsageMapping,
    ) -> TranscriptionResult:
        """Process a LangChain LLM response into a TranscriptionResult.

        Shared logic for all providers: extracts content, parses structured output,
        extracts token usage, and tracks daily token consumption.

        Args:
            response: Raw response from llm.ainvoke() — either an AIMessage,
                      a dict (from with_structured_output(include_raw=True)),
                      or another type.
            token_mapping: Provider-specific mapping for token usage keys.

        Returns:
            TranscriptionResult with content, tokens, and parsed output.
        """
        content, parsed_output, raw_message = self._extract_content(response)

        input_tokens, output_tokens, total_tokens, cached_input_tokens, cache_creation_tokens = (
            self._extract_token_usage(raw_message, token_mapping)
        )

        self._track_token_usage(total_tokens, cached_input_tokens, input_tokens)

        raw_response: Dict[str, Any] = {}
        if raw_message and hasattr(raw_message, "response_metadata"):
            metadata = raw_message.response_metadata
            if isinstance(metadata, dict):
                raw_response = metadata

        result = TranscriptionResult(
            content=content,
            raw_response=raw_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_hit=cached_input_tokens > 0 or cache_creation_tokens > 0,
        )

        if parsed_output and isinstance(parsed_output, dict):
            result.parsed_output = parsed_output
            result.no_transcribable_text = parsed_output.get("no_transcribable_text", False)
            result.transcription_not_possible = parsed_output.get("transcription_not_possible", False)

        return result

    def _build_disabled_params(self) -> Dict[str, Any] | None:
        """Build disabled_params dict based on model capabilities.
        
        LangChain's disabled_params feature automatically filters out
        unsupported parameters before sending to the API.
        
        Subclasses that set self._capabilities can use this directly.
        Returns None if no params need disabling.
        """
        caps = getattr(self, "_capabilities", None)
        if caps is None:
            return None
        disabled: Dict[str, Any] = {}
        if not getattr(caps, "supports_sampler_controls", True):
            # Master flag: disable ALL sampler controls (reasoning models)
            disabled["temperature"] = None
            disabled["top_p"] = None
            disabled["frequency_penalty"] = None
            disabled["presence_penalty"] = None
        else:
            # Fine-grained flags (e.g. Claude 4.5 accepts temp but not top_p)
            if not getattr(caps, "supports_top_p", True):
                disabled["top_p"] = None
            if not getattr(caps, "supports_frequency_penalty", True):
                disabled["frequency_penalty"] = None
            if not getattr(caps, "supports_presence_penalty", True):
                disabled["presence_penalty"] = None
        return disabled if disabled else None

    @staticmethod
    def _extract_input_tokens(result: Any) -> int:
        """Extract input token count from an LLM response.

        Checks ``usage_metadata`` first (Responses API / LangChain 1.x path),
        then falls back to ``response_metadata.token_usage.prompt_tokens``.
        Returns 0 when extraction fails so callers can guard against false
        positives.
        """
        raw_message = None
        if isinstance(result, dict) and "raw" in result:
            raw_message = result.get("raw")
        elif hasattr(result, "usage_metadata") or hasattr(result, "response_metadata"):
            raw_message = result

        if raw_message is None:
            return 0

        # Primary path: usage_metadata (TypedDict / plain dict)
        usage_meta = getattr(raw_message, "usage_metadata", None)
        if usage_meta is not None:
            if isinstance(usage_meta, dict):
                val = usage_meta.get("input_tokens", 0)
            else:
                val = getattr(usage_meta, "input_tokens", 0)
            val = int(val or 0)
            if val > 0:
                return val

        # Fallback: response_metadata["token_usage"]["prompt_tokens"]
        resp_meta = getattr(raw_message, "response_metadata", None)
        if isinstance(resp_meta, dict):
            usage = resp_meta.get("token_usage", {})
            if isinstance(usage, dict):
                val = int(usage.get("prompt_tokens", 0) or 0)
                if val > 0:
                    return val

        return 0

    async def _ainvoke_with_retry(
        self,
        llm: Any,
        messages: List[Any],
        *,
        expect_image_tokens: bool = False,
        **invoke_kwargs: Any,
    ) -> Any:
        """Invoke the LangChain LLM with retry on transient and validation errors.

        Acts as a safety net on top of LangChain's internal max_retries,
        targeting:
            httpx.ConnectError                — TCP/DNS connection failure
            httpx.TimeoutException            — covers all httpx timeout subclasses
            pydantic.ValidationError          — model returned unparseable structured
                                                output (capped at validation_attempts)
            InputTokensBelowThresholdError    — image payload silently dropped;
                                                shares the validation retry budget

        Also detects *silent* parsing failures where LangChain's
        ``with_structured_output(include_raw=True)`` returns
        ``{"parsed": None, "parsing_error": <exc>}`` instead of raising.
        In that case the parsing error is re-raised so tenacity can retry.

        Uses exponential backoff (2 s -> 4 s -> ... -> 60 s max) with jitter.
        After exhausting all attempts the exception is re-raised to the caller.
        """
        import httpx
        import tenacity
        from pydantic import ValidationError

        max_attempts = load_max_retries()
        max_validation_attempts = load_max_validation_retries()
        min_input_tokens = load_min_input_tokens() if expect_image_tokens else 0
        validation_attempt_count = 0

        def _should_retry(exc: BaseException) -> bool:
            if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
                return True
            if isinstance(exc, (ValidationError, InputTokensBelowThresholdError)):
                nonlocal validation_attempt_count
                validation_attempt_count += 1
                if validation_attempt_count >= max_validation_attempts:
                    return False
                if isinstance(exc, InputTokensBelowThresholdError):
                    logger.warning(
                        "Input tokens below threshold on attempt %d/%d "
                        "(got %d, need %d), retrying",
                        validation_attempt_count,
                        max_validation_attempts,
                        exc.actual,
                        exc.threshold,
                    )
                else:
                    logger.warning(
                        "Structured-output validation error on attempt %d/%d, "
                        "retrying: %s",
                        validation_attempt_count,
                        max_validation_attempts,
                        str(exc)[:200],
                    )
                return True
            return False

        async for attempt in tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception(_should_retry),
            wait=tenacity.wait_exponential_jitter(initial=2, max=60),
            stop=tenacity.stop_after_attempt(max_attempts),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                result = await llm.ainvoke(messages, **invoke_kwargs)
                # Detect silent parsing failures from
                # with_structured_output(include_raw=True).
                if (
                    isinstance(result, dict)
                    and result.get("parsing_error") is not None
                ):
                    logger.warning(
                        "Structured output parsing failed silently, "
                        "re-raising for retry: %s",
                        str(result["parsing_error"])[:200],
                    )
                    raise result["parsing_error"]
                # Check for suspiciously low input tokens (image dropped).
                if min_input_tokens > 0:
                    actual = self._extract_input_tokens(result)
                    if 0 < actual < min_input_tokens:
                        raise InputTokensBelowThresholdError(
                            actual, min_input_tokens
                        )
                return result

    async def __aenter__(self) -> "BaseProvider":
        """Async context manager entry."""
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool:
        """Async context manager exit."""
        await self.close()
        return False
    
    @staticmethod
    def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64.

        Delegates to modules.llm.image_encoding for the shared implementation.
        """
        from modules.llm.image_encoding import encode_image_to_base64 as _encode
        return _encode(image_path)
    
    @staticmethod
    def create_data_url(base64_data: str, mime_type: str) -> str:
        """Create a data URL from base64 data.
        
        Args:
            base64_data: Base64-encoded image data
            mime_type: MIME type of the image
        
        Returns:
            Data URL string
        """
        return f"data:{mime_type};base64,{base64_data}"
