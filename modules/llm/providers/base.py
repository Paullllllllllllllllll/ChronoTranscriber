"""Base provider abstraction for LLM integrations.

Defines the common interface that all LLM providers must implement.
"""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modules.config.constants import SUPPORTED_IMAGE_FORMATS
from modules.config.service import get_config_service
from modules.llm.model_capabilities import Capabilities

logger = logging.getLogger(__name__)


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
    except Exception:
        return 5


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
    parsed_output: Optional[Dict[str, Any]] = None
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Transcription status flags (from schema response)
    no_transcribable_text: bool = False
    transcription_not_possible: bool = False
    
    # Error information
    error: Optional[str] = None
    
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
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
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
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
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
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
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
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        raw_response: Dict[str, Any] = {}
        raw_message = None
        parsed_output: Optional[Dict[str, Any]] = None

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

        # Extract token usage from response_metadata
        if raw_message and hasattr(raw_message, "response_metadata"):
            metadata = raw_message.response_metadata
            if isinstance(metadata, dict):
                raw_response = metadata
                usage = metadata.get(token_mapping.usage_key, {})
                if isinstance(usage, dict):
                    input_tokens = usage.get(token_mapping.input_key, 0)
                    output_tokens = usage.get(token_mapping.output_key, 0)
                    if token_mapping.total_key:
                        total_tokens = usage.get(token_mapping.total_key, 0)
                    if total_tokens == 0:
                        total_tokens = input_tokens + output_tokens

        # Fallback: LangChain 1.x stores token usage in AIMessage.usage_metadata
        # (a UsageMetadata TypedDict — i.e. a plain dict) rather than
        # response_metadata["token_usage"].  This is the primary path for the
        # OpenAI Responses API where response_metadata has no "token_usage" key.
        if total_tokens == 0 and raw_message is not None:
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

        # Track tokens using daily tracker
        if total_tokens > 0:
            try:
                from modules.infra.token_tracker import get_token_tracker
                token_tracker = get_token_tracker()
                token_tracker.add_tokens(total_tokens)
                logger.debug(
                    f"[TOKEN] API call consumed {total_tokens:,} tokens "
                    f"(daily total: {token_tracker.get_tokens_used_today():,})"
                )
            except Exception as e:
                logger.warning(f"Error tracking tokens: {e}")

        result = TranscriptionResult(
            content=content,
            raw_response=raw_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
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

    async def _ainvoke_with_retry(
        self,
        llm: Any,
        messages: List[Any],
        **invoke_kwargs: Any,
    ) -> Any:
        """Invoke the LangChain LLM with retry on transient connection errors.

        Acts as a safety net on top of LangChain's internal max_retries,
        specifically targeting raw httpx network failures that can slip through
        the Responses API code path.

        Retries on:
            httpx.ConnectError     — TCP/DNS connection failure
            httpx.TimeoutException — covers all httpx timeout subclasses

        Uses exponential backoff (2 s -> 4 s -> ... -> 60 s max) with jitter.
        After exhausting all attempts the exception is re-raised to the caller.
        """
        import httpx
        import tenacity

        max_attempts = load_max_retries()

        async for attempt in tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception_type(
                (httpx.ConnectError, httpx.TimeoutException)
            ),
            wait=tenacity.wait_exponential_jitter(initial=2, max=60),
            stop=tenacity.stop_after_attempt(max_attempts),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                return await llm.ainvoke(messages, **invoke_kwargs)

    async def __aenter__(self) -> "BaseProvider":
        """Async context manager entry."""
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Async context manager exit."""
        await self.close()
        return False
    
    @staticmethod
    def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Tuple of (base64_data, mime_type)
        
        Raises:
            ValueError: If the image format is not supported
        """
        ext = image_path.suffix.lower()
        mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
        if not mime_type:
            raise ValueError(f"Unsupported image format: {ext}")
        
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return data, mime_type
    
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
