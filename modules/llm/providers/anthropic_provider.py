"""Anthropic (Claude) provider implementation using LangChain.

Supports the full Claude model family via ChatAnthropic:
- Claude 4.7 Opus (adaptive thinking only)
- Claude 4.6 (Opus, Sonnet -- adaptive thinking recommended)
- Claude 4.5 (Opus, Sonnet, Haiku -- extended thinking)
- Claude 4.1 Opus, Claude 4 (Opus, Sonnet)
- Claude 3.7 Sonnet, 3.5 (Sonnet, Haiku), 3 (Opus, Sonnet, Haiku)

LangChain handles:
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)

Retries are owned solely by BaseProvider._ainvoke_with_retry (tenacity); the SDK
client is built with max_retries=0.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from modules.config.capabilities import Capabilities, detect_capabilities
from modules.infra.logger import setup_logger
from modules.llm.providers.base import (
    ANTHROPIC_TOKEN_MAPPING,
    BaseProvider,
    TranscriptionResult,
    aclose_chat_model,
)

logger = setup_logger(__name__)


def _transform_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform JSON schema to be Anthropic-compatible.

    Anthropic's SDK doesn't support union types like ["string", "null"].
    This function converts them to simple types.
    Also adds required 'title' and 'description' keys for LangChain compatibility.
    """
    import copy

    result = copy.deepcopy(schema)

    def transform_type(obj: dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            return

        # Handle union types like ["string", "null"]
        if "type" in obj and isinstance(obj["type"], list):
            # Filter out "null" and keep the first non-null type
            non_null_types = [t for t in obj["type"] if t != "null"]
            if non_null_types:
                obj["type"] = non_null_types[0]
            else:
                obj["type"] = "string"  # fallback

        # Recursively handle properties
        if "properties" in obj and isinstance(obj["properties"], dict):
            for prop in obj["properties"].values():
                transform_type(prop)

        # Handle items in arrays
        if "items" in obj and isinstance(obj["items"], dict):
            transform_type(obj["items"])

        # Handle anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    transform_type(item)

    transform_type(result)

    # Add required top-level keys for LangChain/Anthropic compatibility
    if "title" not in result:
        result["title"] = "TranscriptionSchema"
    if "description" not in result:
        result["description"] = "Schema for document transcription output"

    return result


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider using LangChain."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float | None = None,
        top_p: float = 1.0,
        top_k: int | None = None,
        reasoning_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        caps = detect_capabilities(model)
        effective_max_tokens = int(min(max_tokens, caps.max_output_tokens))

        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            timeout=timeout,
            **kwargs,
        )

        self.top_p = top_p
        self.top_k = top_k
        self.reasoning_config = reasoning_config

        self._capabilities = caps

        # Build LangChain model kwargs
        model_kwargs: dict[str, Any] = {}
        if self._capabilities.supports_sampler_controls:
            model_kwargs["temperature"] = temperature
        if self._capabilities.supports_top_p:
            model_kwargs["top_p"] = top_p
        if top_k is not None:
            model_kwargs["top_k"] = top_k

        # Apply thinking for Claude models that support it.
        # Opus 4.7: adaptive only (budget_tokens returns 400).
        # Opus 4.6 / Sonnet 4.6: adaptive recommended, enabled deprecated.
        # Older models (4.5, 4.1, etc.): extended thinking with budget_tokens.
        thinking_enabled = False
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            effort = reasoning_config.get("effort", "medium")
            family = self._capabilities.family

            _ADAPTIVE_FAMILIES = (
                "claude-opus-4.7",
                "claude-opus-4.6",
                "claude-sonnet-4.6",
            )
            if family in _ADAPTIVE_FAMILIES:
                model_kwargs["thinking"] = {"type": "adaptive"}
                logger.info(f"Using adaptive thinking for model {model}")
            else:
                effort_to_budget = {
                    "low": 1024,
                    "medium": 4096,
                    "high": 16384,
                }
                budget = effort_to_budget.get(effort, 4096)
                model_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                logger.info(
                    f"Using extended thinking (budget={budget}) for model {model}"
                )
            thinking_enabled = True

        # Anthropic rejects sampler controls while any thinking block is active:
        # the API requires temperature=1 and forbids top_p/top_k for both the
        # "enabled" (budget_tokens) and "adaptive" branches. The upstream default
        # temperature is 0.0, so leaving these in model_kwargs alongside thinking
        # produces an unretryable 400 on every page. Drop the incompatible
        # samplers here so a thinking-enabled model transcribes instead of failing.
        if thinking_enabled:
            dropped = [
                key for key in ("temperature", "top_p", "top_k") if key in model_kwargs
            ]
            for key in dropped:
                model_kwargs.pop(key, None)
            if dropped:
                logger.info(
                    "Dropped sampler kwargs (%s) for model %s: Anthropic requires "
                    "temperature=1 and forbids top_p/top_k when a thinking block "
                    "is active.",
                    ", ".join(dropped),
                    model,
                )

        # Initialize LangChain ChatAnthropic. max_retries=0 disables SDK-internal
        # retries so the tenacity loop in BaseProvider._ainvoke_with_retry is the
        # single retry authority.
        self._llm = ChatAnthropic(  # type: ignore[call-arg]
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            max_tokens=effective_max_tokens,
            timeout=timeout,
            max_retries=0,
            **model_kwargs,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_capabilities(self) -> Capabilities:
        return self._capabilities

    def _normalize_list_content(self, content_list: list) -> str:
        """Anthropic can return content as a list of text blocks."""
        text_parts: list[str] = []
        for item in content_list:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    text_parts.append(t)
            elif isinstance(item, str) and item.strip():
                text_parts.append(item)
        return "\n".join(text_parts) if text_parts else str(content_list)

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
        """Transcribe text from a base64-encoded image using LangChain.

        Note: Anthropic doesn't use image_detail or media_resolution parameters.
        These parameters are accepted for API compatibility but ignored.
        """
        caps = self._capabilities

        if not caps.supports_image_input:
            return TranscriptionResult(
                content="",
                error=f"Model {self.model} does not support vision/image inputs.",
                transcription_not_possible=True,
            )

        # Anthropic uses a different image format
        # See: https://docs.anthropic.com/claude/docs/vision
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_base64,
            },
        }

        # Build system message — with cache_control when prompt caching is enabled
        if self._caching_enabled:
            anthropic_cfg = self._caching_config.get("anthropic", {})
            ttl = (
                anthropic_cfg.get("ttl", "5m")
                if isinstance(anthropic_cfg, dict)
                else "5m"
            )
            cache_control: dict[str, Any] = {"type": "ephemeral"}
            if ttl == "1h":
                cache_control["ttl"] = "1h"
            system_message = SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": cache_control,
                    }
                ]
            )
        else:
            system_message = SystemMessage(content=system_prompt)

        human_content: list[dict[str, Any]] = []
        if user_instruction:
            human_content.append({"type": "text", "text": user_instruction})
        human_content.append(image_content)

        messages = [
            system_message,
            HumanMessage(content=human_content),  # type: ignore[arg-type]
        ]

        # Native structured outputs (no function calling)
        # We require json_schema mode so the model returns a validated JSON object.
        if json_schema and not caps.supports_structured_outputs:
            raise ValueError(
                f"Selected Anthropic model '{self.model}' does not support"
                f" native structured outputs. "
                f"Choose a Claude model that supports structured outputs"
                f" (e.g. claude-sonnet-4-5-* or claude-opus-4-1-*)."
            )

        # Use include_raw=True to get token usage from the underlying AIMessage
        llm_to_use = self._llm
        if json_schema:
            # Unwrap schema if wrapped
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema

            # Transform schema for Anthropic compatibility (handle nullable types)
            actual_schema = _transform_schema_for_anthropic(actual_schema)

            llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                actual_schema,
                method="json_schema",
                include_raw=True,
            )

        # Invoke LLM - LangChain handles retries internally
        return await self._invoke_llm(llm_to_use, messages)

    async def _invoke_llm(
        self,
        llm: Any,
        messages: list[Any],
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.

        LangChain handles retry logic internally.
        Response parsing and token tracking are handled by the shared
        BaseProvider._process_llm_response() method.
        """
        try:
            response = await self._ainvoke_with_retry(
                llm, messages, expect_image_tokens=True
            )
            return await self._process_llm_response(response, ANTHROPIC_TOKEN_MAPPING)
        except Exception as e:
            # logger.exception captures the traceback so a programming error
            # (e.g. KeyError from a refactor) is not masked as an API error.
            logger.exception(f"Error invoking Anthropic: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
            )

    async def close(self) -> None:
        """Dispose the underlying LangChain/SDK HTTP clients. Never raises."""
        await aclose_chat_model(getattr(self, "_llm", None))
