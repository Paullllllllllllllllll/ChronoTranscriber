"""Base provider abstraction for LLM integrations.

Defines the common interface that all LLM providers must implement.
"""

from __future__ import annotations

import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modules.config.capabilities import Capabilities
from modules.config.service import get_config_service
from modules.infra.logger import setup_logger

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


class OutputTokensTruncatedError(Exception):
    """Raised when a response is truncated because it hit max_output_tokens.

    Retrying is pointless — the same limit applies.  Reasoning models
    (GPT-5, o-series) can consume the entire output budget on the
    internal reasoning chain, leaving zero tokens for the actual answer.
    The fix is to increase ``max_output_tokens`` in model_config.yaml or
    via ``--max-output-tokens`` on the CLI.
    """

    def __init__(self, output_tokens: int, reasoning_tokens: int) -> None:
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        super().__init__(
            f"Response truncated: model used {output_tokens} output tokens "
            f"({reasoning_tokens} on reasoning) and hit max_output_tokens. "
            f"Increase --max-output-tokens or lower --reasoning-effort."
        )


def _load_retry_config() -> dict[str, Any]:
    """Return the ``concurrency.transcription.retry`` config block.

    Shared lookup for all retry-related loaders below; returns an empty dict
    when any level of the config is missing so callers can apply their own
    defaults via ``.get``.
    """
    conc_cfg = get_config_service().get_concurrency_config() or {}
    trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
    return trans_cfg.get("retry", {}) or {}


def load_max_retries() -> int:
    """Load the tenacity retry-attempt budget from concurrency_config.yaml.

    This feeds ONLY the tenacity loop in :meth:`BaseProvider._ainvoke_with_retry`
    (the single retry authority): every provider constructs its LangChain/SDK
    client with ``max_retries=0``, so ``retry.attempts`` no longer feeds the SDK.
    Defaults to 8 transient attempts.
    """
    try:
        attempts = int(_load_retry_config().get("attempts", 8))
        return max(1, attempts)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.debug("Could not load max_retries from config, using default: %s", e)
        return 8


def load_max_validation_retries() -> int:
    """Load max validation retries from concurrency_config.yaml.

    Controls how many times to retry when the model returns unparseable
    structured output (pydantic.ValidationError).  Defaults to 3 since
    these indicate model behavior issues rather than transient
    infrastructure problems.
    """
    try:
        attempts = int(_load_retry_config().get("validation_attempts", 3))
        return max(1, attempts)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.debug(
            "Could not load max_validation_retries from config, using default: %s", e
        )
        return 3


def load_min_input_tokens() -> int:
    """Load minimum expected input tokens for image requests.

    Responses below this threshold are treated as cross-contaminated
    (image payload silently dropped by the API) and retried.
    Defaults to 500.  Set to 0 to disable the check.
    """
    try:
        value = int(_load_retry_config().get("min_input_tokens", 500))
        return max(0, value)
    except (KeyError, AttributeError, TypeError, ValueError) as e:
        logger.debug(
            "Could not load min_input_tokens from config, using default: %s", e
        )
        return 500


def _classify_status(exc: BaseException) -> tuple[bool, bool]:
    """Classify an exception by its HTTP status code (authoritative when present).

    Reads ``status_code`` first, then ``status``. Returns
    ``(is_rate_limit, is_server_error)``: a 429 is rate-limit-retryable and a
    5xx is server-error-retryable. Both default to False when no numeric status
    is present.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status", None)
    if not isinstance(status, int):
        status = None
    is_rate_limit = status == 429
    is_server_error = status is not None and 500 <= status <= 599
    return is_rate_limit, is_server_error


def _is_connection_error(exc: BaseException) -> bool:
    """Return True when the exception is a transient connection/timeout failure.

    Provider SDKs wrap the underlying ``httpx`` transport error before it
    reaches the retry loop (e.g. the openai and anthropic SDKs raise
    ``APIConnectionError from httpx.ConnectError``), so checking only the
    top-level exception type misses them. Walk the ``__cause__``/``__context__``
    chain (bounded, cycle-safe) looking for ``httpx.ConnectError`` or
    ``httpx.TimeoutException``.
    """
    import httpx

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def parse_retry_after(exc: BaseException | None) -> float | None:
    """Extract a Retry-After delay (in seconds) from an exception's HTTP headers.

    Reads the ``Retry-After`` header off the exception's response (or a
    top-level ``headers`` attribute) across the openai/anthropic SDK exception
    shapes, tolerating both the integer/float seconds form and the HTTP-date
    form. Returns ``None`` when no usable value is present. Defensive: any
    parsing problem yields ``None`` rather than raising.
    """
    if exc is None:
        return None
    try:
        headers = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
        if headers is None:
            headers = getattr(exc, "headers", None)
        if headers is None:
            return None

        getter = getattr(headers, "get", None)
        if not callable(getter):
            return None
        raw = getter("retry-after")
        if raw is None:
            raw = getter("Retry-After")
        if raw is None:
            return None

        value = str(raw).strip()
        if not value:
            return None

        # Seconds form (integer or float).
        try:
            return max(0.0, float(value))
        except ValueError:
            pass

        # HTTP-date form (e.g. "Wed, 21 Oct 2026 07:28:00 GMT").
        from email.utils import parsedate_to_datetime

        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        import datetime as _dt

        now = _dt.datetime.now(target.tzinfo) if target.tzinfo else _dt.datetime.now()
        return max(0.0, (target - now).total_seconds())
    except Exception:
        return None


def _commit_tokens_from_exception(exc: BaseException) -> None:
    """Best-effort: recover token usage from a failed call and commit it.

    Provider SDK exceptions often carry usage data in ``exc.body["usage"]`` or
    ``exc.response.json()["usage"]``; recovering it keeps the daily budget
    honest even for calls that ultimately errored. Tries ``total_tokens``, then
    ``prompt_tokens`` + ``completion_tokens`` (OpenAI), then ``input_tokens`` +
    ``output_tokens`` (Anthropic). Never raises.
    """
    try:
        usage: dict[str, Any] | None = None

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            candidate = body.get("usage")
            if isinstance(candidate, dict):
                usage = candidate

        if usage is None:
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    resp_json = resp.json()
                    if isinstance(resp_json, dict) and isinstance(
                        resp_json.get("usage"), dict
                    ):
                        usage = resp_json["usage"]
                except Exception:
                    usage = None

        if not isinstance(usage, dict):
            return

        total = usage.get("total_tokens")
        if not isinstance(total, int) or total <= 0:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            if (
                isinstance(prompt, int)
                and isinstance(completion, int)
                and (prompt + completion) > 0
            ):
                total = prompt + completion
            else:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                if isinstance(inp, int) and isinstance(out, int) and (inp + out) > 0:
                    total = inp + out

        if isinstance(total, int) and total > 0:
            from modules.infra.token_budget import get_token_tracker

            get_token_tracker().add_tokens(total)
            logger.info(
                "[TOKEN] Recovered %s tokens from a failed request.", f"{total:,}"
            )
    except Exception:
        logger.debug("Token recovery from exception failed", exc_info=True)


async def _aclose_maybe(obj: Any) -> None:
    """Call ``obj.close()`` if present, awaiting an async closer. Never raises.

    Used to tear down provider SDK clients (which own httpx connections) when a
    LangChain chat model is disposed. A missing or non-callable ``close`` is a
    no-op; any exception is logged at debug level and swallowed.
    """
    if obj is None:
        return
    close = getattr(obj, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # Best-effort cleanup; never propagate.
        logger.debug("Error closing client %r: %s", type(obj).__name__, exc)


async def aclose_chat_model(chat_model: Any) -> None:
    """Close every known SDK client attribute of a LangChain chat model.

    LangChain chat models wrap provider SDK clients that in turn hold an httpx
    client; without an explicit close those connections leak. ChatOpenAI exposes
    ``root_client`` / ``root_async_client``; ChatAnthropic exposes ``_client`` /
    ``_async_client``; providers exposing none (e.g. Google) are simply skipped.
    Never raises: any close failure is logged at debug level and swallowed.
    """
    if chat_model is None:
        return
    for attr in (
        "root_async_client",
        "root_client",
        "async_client",
        "client",
        "_async_client",
        "_client",
    ):
        await _aclose_maybe(getattr(chat_model, attr, None))


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
    raw_response: dict[str, Any] = field(default_factory=dict)

    # Parsed structured output (if schema was provided)
    parsed_output: dict[str, Any] | None = None

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
                from modules.llm.response_parsing import _normalize_llm_text

                normalized = _normalize_llm_text(self.content)
                if normalized.lstrip().startswith("{"):
                    parsed = json.loads(normalized)
                    if isinstance(parsed, dict):
                        self.parsed_output = parsed
                        self.no_transcribable_text = parsed.get(
                            "no_transcribable_text", False
                        )
                        self.transcription_not_possible = parsed.get(
                            "transcription_not_possible", False
                        )
            except (json.JSONDecodeError, ImportError):
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
        self._caching_config: dict[str, Any] = caching_cfg

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass

    @property
    def use_plain_text_prompt(self) -> bool:
        """Whether the provider requires a simplified plain-text prompt.

        Subclasses may override to return ``True`` when a custom endpoint
        is configured with ``use_plain_text_prompt: true``.
        """
        return False

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
        json_schema: dict[str, Any] | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
        context_image_base64: str | None = None,
        context_image_mime_type: str | None = None,
        context_image_detail: str | None = None,
        context_image_instruction: str = "Context image:",
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
            media_resolution: Media resolution for Google
                ("low", "medium", "high", "ultra_high", "auto")
            context_image_base64: Optional base64-encoded context image
            context_image_mime_type: MIME type of the context image
            context_image_detail: Detail level for the context image
            context_image_instruction: Label for the context image block;
                empty string omits the label entirely

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
            context_image_base64=context_image_base64,
            context_image_mime_type=context_image_mime_type,
            context_image_detail=context_image_detail,
            context_image_instruction=context_image_instruction,
        )

    @abstractmethod
    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: dict[str, Any] | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
        context_image_base64: str | None = None,
        context_image_mime_type: str | None = None,
        context_image_detail: str | None = None,
        context_image_instruction: str = "Context image:",
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image.

        Args:
            image_base64: Base64-encoded image data
            mime_type: MIME type of the image (e.g., "image/jpeg")
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            media_resolution: Media resolution for Google
                ("low", "medium", "high", "ultra_high", "auto")
            context_image_base64: Optional base64-encoded context image
            context_image_mime_type: MIME type of the context image
            context_image_detail: Detail level for the context image
            context_image_instruction: Label for the context image block;
                empty string omits the label entirely

        Returns:
            TranscriptionResult with the transcription and metadata
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g., HTTP sessions)."""
        pass

    def _normalize_list_content(self, content_list: list[Any]) -> str:
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
            # with_structured_output(include_raw=True)
            # → {"raw": AIMessage, "parsed": ...}
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
    ) -> tuple[int, int, int, int, int, int]:
        """Extract token counts from the raw LLM message.

        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens,
                      cached_input_tokens, cache_creation_tokens,
                      additive_cache_tokens).

        ``additive_cache_tokens`` is the cache total (creation + read) that the
        provider reports SEPARATELY from ``input_tokens`` and thus omits from
        ``total_tokens`` — i.e. the raw-Anthropic ``cache_*_input_tokens`` shape.
        It is zero for OpenAI (``prompt_tokens_details.cached_tokens`` is a subset
        of ``prompt_tokens``, already counted) and for LangChain
        ``usage_metadata`` (cache is already folded into ``input_tokens``), so the
        daily budget can add it at full weight without double counting.
        """
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cached_input_tokens = 0
        cache_creation_tokens = 0
        additive_cache_tokens = 0

        if raw_message is None:
            return (
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
                cache_creation_tokens,
                additive_cache_tokens,
            )

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
                    # Anthropic: usage.cache_read_input_tokens /
                    # cache_creation_input_tokens. These are reported SEPARATELY
                    # from input_tokens (which excludes them), so they are
                    # additive for the daily-budget total.
                    cached_input_tokens = int(
                        usage.get("cache_read_input_tokens", 0) or 0
                    )
                    cache_creation_tokens = int(
                        usage.get("cache_creation_input_tokens", 0) or 0
                    )
                    additive_cache_tokens = cached_input_tokens + cache_creation_tokens

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
                    # Defensive: support object-attribute style
                    # (e.g. MagicMock in tests)
                    input_tokens = int(getattr(usage_meta, "input_tokens", 0) or 0)
                    output_tokens = int(getattr(usage_meta, "output_tokens", 0) or 0)
                    total_tokens = int(getattr(usage_meta, "total_tokens", 0) or 0)
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens

                # Extract cache tokens from usage_metadata
                if cached_input_tokens == 0 and isinstance(usage_meta, dict):
                    details = usage_meta.get("input_token_details")
                    if isinstance(details, dict):
                        cached_input_tokens = int(details.get("cache_read", 0) or 0)
                        cache_creation_tokens = int(
                            details.get("cache_creation", 0) or 0
                        )

        return (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            cache_creation_tokens,
            additive_cache_tokens,
        )

    def _track_token_usage(
        self,
        total_tokens: int,
        cached_input_tokens: int,
        input_tokens: int,
    ) -> None:
        """Record token consumption in the daily tracker.

        ``total_tokens`` is the committed daily-budget total, already inclusive
        of any separately-reported (raw-Anthropic) cache tokens at full weight;
        the caller computes it in :meth:`_process_llm_response`. The cache-hit
        percentage log below uses ``cached_input_tokens`` / ``input_tokens``.
        """
        if total_tokens > 0:
            try:
                from modules.infra.token_budget import get_token_tracker

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

        (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            cache_creation_tokens,
            additive_cache_tokens,
        ) = self._extract_token_usage(raw_message, token_mapping)

        # Commit prompt-cache tokens at full weight: for the raw-Anthropic shape
        # they sit OUTSIDE total_tokens, so add them back. OpenAI/usage_metadata
        # shapes already include cache in total_tokens (additive is 0 there), so
        # no double counting occurs.
        committed_total = total_tokens + additive_cache_tokens
        self._track_token_usage(committed_total, cached_input_tokens, input_tokens)

        raw_response: dict[str, Any] = {}
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
            result.no_transcribable_text = parsed_output.get(
                "no_transcribable_text", False
            )
            result.transcription_not_possible = parsed_output.get(
                "transcription_not_possible", False
            )

        # Content-quality validation is applied inside _ainvoke_with_retry so it
        # shares the validation retry budget (B11); it is NOT re-run here, so an
        # exhausted budget keeps the last attempt's text instead of raising.
        return result

    def _get_content_quality_config(self) -> dict[str, Any]:
        """Load content quality validator config from concurrency_config.yaml."""
        try:
            return _load_retry_config().get("content_quality", {}) or {}
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug("Could not load content_quality config, using defaults: %s", e)
            return {}

    def _validate_result_content_quality(self, response: Any) -> None:
        """Validate the transcription text of a raw response, if enabled.

        Extracts the transcription text from *response* and runs the
        content-quality validators. Raises ``ContentQualityError`` on failure so
        the caller's retry loop can retry against the validation budget (B11).
        A no-op when content-quality validation is disabled (the shipped
        default) or the page carries a no-text/not-possible flag.
        """
        cq_config = self._get_content_quality_config()
        if not cq_config.get("enabled", False):
            return

        content, parsed_output, _ = self._extract_content(response)

        transcription_text: str | None = None
        skip_quality = False
        if parsed_output and isinstance(parsed_output, dict):
            transcription_text = parsed_output.get("transcription")
            skip_quality = bool(
                parsed_output.get("no_transcribable_text")
                or parsed_output.get("transcription_not_possible")
            )
        elif content:
            stripped = content.strip()
            if stripped in ("[No transcribable text]", "[Transcription not possible]"):
                skip_quality = True
            else:
                transcription_text = stripped

        if transcription_text and not skip_quality:
            from modules.llm.quality import validate_content_quality

            validate_content_quality(
                transcription_text=transcription_text,
                no_transcribable_text=False,
                transcription_not_possible=False,
                config=cq_config,
            )

    def _build_disabled_params(self) -> dict[str, Any] | None:
        """Build disabled_params dict based on model capabilities.

        LangChain's disabled_params feature automatically filters out
        unsupported parameters before sending to the API.

        Subclasses that set self._capabilities can use this directly.
        Returns None if no params need disabling.
        """
        caps = getattr(self, "_capabilities", None)
        if caps is None:
            return None
        disabled: dict[str, Any] = {}
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

    @staticmethod
    def _extract_total_tokens(result: Any) -> int:
        """Extract total token count from an LLM response.

        Uses ``usage_metadata`` (LangChain's provider-normalized dict) so it
        works for all providers without needing ``TokenUsageMapping``.
        Falls back to ``input_tokens + output_tokens`` when ``total_tokens``
        is zero.  Returns 0 when extraction fails.
        """
        raw_message = None
        if isinstance(result, dict) and "raw" in result:
            raw_message = result.get("raw")
        elif hasattr(result, "usage_metadata"):
            raw_message = result

        if raw_message is None:
            return 0

        usage_meta = getattr(raw_message, "usage_metadata", None)
        if usage_meta is None:
            return 0

        if isinstance(usage_meta, dict):
            total = int(usage_meta.get("total_tokens", 0) or 0)
            if total > 0:
                return total
            inp = int(usage_meta.get("input_tokens", 0) or 0)
            out = int(usage_meta.get("output_tokens", 0) or 0)
            return inp + out
        else:
            total = int(getattr(usage_meta, "total_tokens", 0) or 0)
            if total > 0:
                return total
            inp = int(getattr(usage_meta, "input_tokens", 0) or 0)
            out = int(getattr(usage_meta, "output_tokens", 0) or 0)
            return inp + out

    async def _ainvoke_with_retry(
        self,
        llm: Any,
        messages: list[Any],
        *,
        expect_image_tokens: bool = False,
        **invoke_kwargs: Any,
    ) -> Any:
        """Invoke the LangChain LLM with retry on transient and validation errors.

        This is the SINGLE retry authority: every provider constructs its
        SDK/LangChain client with ``max_retries=0``, so all retries happen here.
        Retryable classes:
            HTTP 429 / 5xx (status-code-first)  — transient API errors, full budget
            httpx.ConnectError                  — TCP/DNS connection failure,
                                                  including SDK-wrapped forms
                                                  (openai/anthropic
                                                  APIConnectionError) detected
                                                  via the __cause__ chain
            httpx.TimeoutException              — covers all httpx timeout subclasses
            pydantic.ValidationError            — unparseable structured output
                                                  (capped at validation_attempts)
            OutputParserException (langchain)   — invalid JSON model output
                                                  (capped at validation_attempts)
            InputTokensBelowThresholdError      — image payload silently dropped;
                                                  shares the validation retry budget

        Also detects *silent* parsing failures where LangChain's
        ``with_structured_output(include_raw=True)`` returns
        ``{"parsed": None, "parsing_error": <exc>}`` instead of raising.
        In that case the parsing error is re-raised so tenacity can retry.

        Backoff is exponential with jitter (floor 2 s, cap 120 s); a server-sent
        ``Retry-After`` header raises the wait to at least that value (still
        capped at 120 s). Usage reported by each failed attempt is recovered and
        committed to the daily budget. After exhausting all attempts the
        exception is re-raised to the caller.
        """
        import tenacity
        from langchain_core.exceptions import OutputParserException
        from pydantic import ValidationError

        from modules.llm.quality import ContentQualityError

        max_attempts = load_max_retries()
        max_validation_attempts = load_max_validation_retries()
        min_input_tokens = load_min_input_tokens() if expect_image_tokens else 0
        validation_attempt_count = 0

        def _should_retry(exc: BaseException) -> bool:
            nonlocal validation_attempt_count
            # Cause-chain-aware: provider SDKs wrap httpx transport errors
            # (e.g. openai.APIConnectionError from httpx.ConnectError), so
            # the underlying connection failure is found via __cause__.
            if _is_connection_error(exc):
                return True
            # Status-code-first classification (authoritative). Since every
            # provider builds its SDK client with max_retries=0, this loop is the
            # ONLY retry authority: 429 (rate limit) and 5xx (server error) must
            # be retried here or they fail permanently. Transient infrastructure
            # errors — retried on the full attempt budget, NOT the validation one.
            is_rate_limit, is_server_error = _classify_status(exc)
            if is_rate_limit or is_server_error:
                logger.warning(
                    "Transient API %s error, retrying: %s",
                    "rate-limit (429)" if is_rate_limit else "server (5xx)",
                    str(exc)[:200],
                )
                return True
            # Transient provider-side response-shape bugs surface as
            # TypeError/AttributeError from LangChain/OpenAI SDK internals when a
            # provider (e.g. OpenRouter) returns an unexpected response shape.
            # Treat these as retryable under the validation budget to avoid
            # permanent "[transcription error]" fallbacks on otherwise recoverable
            # pages. Messages observed in the wild include:
            #   "'NoneType' object is not iterable"
            #   "expected string or bytes-like object, got 'int'"
            if isinstance(exc, (TypeError, AttributeError)):
                validation_attempt_count += 1
                if validation_attempt_count >= max_validation_attempts:
                    return False
                logger.warning(
                    "Provider response-shape error on attempt %d/%d, retrying: %s",
                    validation_attempt_count,
                    max_validation_attempts,
                    str(exc)[:200],
                )
                return True
            # OutputParserException covers LangChain's "Invalid json output"
            # from with_structured_output when a model returns flaky JSON
            # (CT-3): transient invalid model output, retried on the
            # validation budget like a pydantic ValidationError.
            if isinstance(
                exc,
                (
                    ValidationError,
                    OutputParserException,
                    InputTokensBelowThresholdError,
                    ContentQualityError,
                ),
            ):
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
                elif isinstance(exc, ContentQualityError):
                    logger.warning(
                        "Content quality check failed on attempt %d/%d "
                        "(%s): %s — retrying",
                        validation_attempt_count,
                        max_validation_attempts,
                        exc.failure_type,
                        exc.detail[:200],
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

        _base_wait = tenacity.wait_exponential_jitter(initial=2, max=120)

        def _wait(retry_state: tenacity.RetryCallState) -> float:
            """Exponential-jitter backoff, raised to honor Retry-After (cap 120s)."""
            computed = float(_base_wait(retry_state))
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            retry_after = parse_retry_after(exc)
            if retry_after is not None:
                return min(120.0, max(computed, float(retry_after)))
            return computed

        def _before_sleep(retry_state: tenacity.RetryCallState) -> None:
            """Recover usage from the failed attempt, then log the retry."""
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc is not None:
                _commit_tokens_from_exception(exc)
            tenacity.before_sleep_log(logger, logging.WARNING)(retry_state)

        last_result: Any = None
        try:
            try:
                async for attempt in tenacity.AsyncRetrying(
                    retry=tenacity.retry_if_exception(_should_retry),
                    wait=_wait,
                    stop=tenacity.stop_after_attempt(max_attempts),
                    before_sleep=_before_sleep,
                    reraise=True,
                ):
                    with attempt:
                        result = await self._ainvoke_once(
                            llm, messages, min_input_tokens, **invoke_kwargs
                        )
                        last_result = result
                        # Content-quality validation shares the validation retry
                        # budget; on the final attempt _should_retry returns False
                        # and the error propagates to the except below (B11).
                        self._validate_result_content_quality(result)
                        return result
            except ContentQualityError:
                logger.warning(
                    "Content-quality retry budget exhausted; keeping the last "
                    "attempt's text rather than discarding the page."
                )
                if last_result is not None:
                    return last_result
                raise
        except Exception as exc:
            # Terminal failure of the retry loop: the final attempt is not
            # followed by a before_sleep, so recover its usage here too. Earlier
            # attempts were committed in _before_sleep, so no double counting.
            _commit_tokens_from_exception(exc)
            raise
        # Unreachable: the retry loop always returns or raises.
        raise RuntimeError("retry loop exited without a result")

    async def _ainvoke_once(
        self,
        llm: Any,
        messages: list[Any],
        min_input_tokens: int,
        **invoke_kwargs: Any,
    ) -> Any:
        """Single guarded ainvoke: truncation, parsing, and input-token checks.

        Extracted from the retry loop so content-quality validation can run
        against the same result within the retried callable (B11).

        Every synchronous provider call passes through the per-provider client-
        side rate limiter exactly once, immediately before the API call: it
        blocks off the event loop until capacity is free, then feeds the outcome
        back (429/5xx => is_rate_limit) so the adaptive multiplier smooths bursts.
        Batch submission/polling never reaches this seam.
        """
        from modules.infra.rate_limit import await_capacity, get_shared_rate_limiter

        limiter = get_shared_rate_limiter(self.provider_name)
        await await_capacity(limiter)
        try:
            result = await llm.ainvoke(messages, **invoke_kwargs)
        except BaseException as exc:
            is_rate_limit, is_server_error = _classify_status(exc)
            limiter.report_error(is_rate_limit=is_rate_limit or is_server_error)
            raise
        limiter.report_success()
        # Detect max_output_tokens truncation before checking parsing errors —
        # retrying a truncated response is pointless and wastes tokens + time.
        if isinstance(result, dict):
            raw_msg = result.get("raw")
            if raw_msg is not None:
                meta = getattr(raw_msg, "response_metadata", {}) or {}
                incomplete = meta.get("incomplete_details") or {}
                if incomplete.get("reason") == "max_output_tokens":
                    usage = getattr(raw_msg, "usage_metadata", {}) or {}
                    out_tok = usage.get("output_tokens", 0)
                    out_detail = usage.get("output_token_details", {}) or {}
                    reasoning_tok = out_detail.get(
                        "flex_reasoning",
                        out_detail.get("reasoning", 0),
                    )
                    discarded = self._extract_total_tokens(result)
                    if discarded > 0:
                        self._track_token_usage(discarded, 0, 0)
                    raise OutputTokensTruncatedError(out_tok, reasoning_tok)
        # Detect silent parsing failures from
        # with_structured_output(include_raw=True).
        if isinstance(result, dict) and result.get("parsing_error") is not None:
            logger.warning(
                "Structured output parsing failed silently, re-raising for retry: %s",
                str(result["parsing_error"])[:200],
            )
            discarded_tokens = self._extract_total_tokens(result)
            if discarded_tokens > 0:
                self._track_token_usage(discarded_tokens, 0, 0)
                logger.debug(
                    "Tracked %d tokens from discarded retry (parsing failure)",
                    discarded_tokens,
                )
            raise result["parsing_error"]
        # Check for suspiciously low input tokens (image dropped).
        if min_input_tokens > 0:
            actual = self._extract_input_tokens(result)
            if 0 < actual < min_input_tokens:
                discarded_tokens = self._extract_total_tokens(result)
                if discarded_tokens > 0:
                    self._track_token_usage(discarded_tokens, 0, 0)
                    logger.debug(
                        "Tracked %d tokens from discarded retry "
                        "(input below threshold)",
                        discarded_tokens,
                    )
                raise InputTokensBelowThresholdError(actual, min_input_tokens)
        return result

    async def __aenter__(self) -> BaseProvider:
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

        Delegates to modules.images.encoding for the shared implementation.
        """
        from modules.images.encoding import encode_image_to_base64 as _encode

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
