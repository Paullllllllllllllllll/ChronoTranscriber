"""Anthropic (Claude) provider implementation using LangChain.

Supports Claude 3 family models:
- Claude 3 Opus (most capable)
- Claude 3 Sonnet (balanced)
- Claude 3 Haiku (fastest)
- Claude 3.5 Sonnet (latest)

LangChain handles:
- Retry logic with exponential backoff (max_retries parameter)
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from modules.llm.providers.base import (
    ANTHROPIC_TOKEN_MAPPING,
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
    load_max_retries,
)

logger = logging.getLogger(__name__)


def _transform_schema_for_anthropic(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Transform JSON schema to be Anthropic-compatible.
    
    Anthropic's SDK doesn't support union types like ["string", "null"].
    This function converts them to simple types.
    Also adds required 'title' and 'description' keys for LangChain compatibility.
    """
    import copy
    result = copy.deepcopy(schema)
    
    def transform_type(obj: Dict[str, Any]) -> None:
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


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on Anthropic model name.
    
    Supports (as of November 2025):
    - Claude 4.5: claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5
    - Claude 4.1: claude-opus-4-1
    - Claude 4: claude-sonnet-4, claude-opus-4
    - Claude 3.5: claude-3-5-sonnet, claude-3-5-haiku
    - Claude 3: claude-3-opus, claude-3-sonnet, claude-3-haiku
    
    Model ID formats:
    - claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)
    - claude-opus-4-5-XXXXXXXX (Claude Opus 4.5)
    - claude-haiku-4-5-XXXXXXXX (Claude Haiku 4.5)
    - claude-opus-4-1-20250805 (Claude Opus 4.1)
    - claude-sonnet-4-20250514 (Claude Sonnet 4)
    """
    m = model_name.lower().strip()
    
    # Claude 4.5 Opus (most capable)
    if "claude-opus-4-5" in m or "claude-opus-4.5" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,  # Extended thinking support
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,  # Claude 4.5 doesn't allow both temperature AND top_p
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=32768,
        )
    
    # Claude 4.5 Sonnet (balanced, recommended)
    if "claude-sonnet-4-5" in m or "claude-sonnet-4.5" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,  # Extended thinking support
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,  # Claude 4.5 doesn't allow both temperature AND top_p
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 4.5 Haiku (fastest, cost-efficient)
    # Structured outputs GA: supported on Haiku 4.5
    if "claude-haiku-4-5" in m or "claude-haiku-4.5" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,  # Extended thinking support
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,  # Claude 4.5 doesn't allow both temperature AND top_p
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 4.1 Opus
    if "claude-opus-4-1" in m or "claude-opus-4.1" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 4 Sonnet
    if "claude-sonnet-4" in m and "4-5" not in m and "4.5" not in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 4 Opus
    if "claude-opus-4" in m and "4-1" not in m and "4.1" not in m and "4-5" not in m and "4.5" not in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 3.5 Sonnet
    if "claude-3-5-sonnet" in m or "claude-3.5-sonnet" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 3.5 Haiku (does NOT support structured outputs)
    if "claude-3-5-haiku" in m or "claude-3.5-haiku" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 3 Opus (does NOT support structured outputs)
    if "claude-3-opus" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=4096,
        )
    
    # Claude 3 Sonnet (does NOT support structured outputs)
    if "claude-3-sonnet" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=4096,
        )
    
    # Claude 3 Haiku (does NOT support structured outputs)
    if "claude-3-haiku" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=4096,
        )
    
    # Default/fallback for Claude models (assume latest capabilities)
    return ProviderCapabilities(
        provider_name="anthropic",
        model_name=model_name,
        supports_vision=True,
        supports_image_detail=False,
        default_image_detail="auto",
        supports_structured_output=True,
        supports_json_mode=True,
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_temperature=True,
        supports_top_p=True,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        max_context_tokens=200000,
        max_output_tokens=8192,
    )


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider using LangChain."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        caps = _get_model_capabilities(model)
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
        max_retries = load_max_retries()
        
        # Build LangChain model kwargs
        model_kwargs: Dict[str, Any] = {}
        if self._capabilities.supports_temperature:
            model_kwargs["temperature"] = temperature
        if self._capabilities.supports_top_p:
            model_kwargs["top_p"] = top_p
        if top_k is not None:
            model_kwargs["top_k"] = top_k
        
        # Apply extended thinking for Claude 4.5+ models that support it
        # Maps reasoning_config.effort to Anthropic's thinking parameter
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            effort = reasoning_config.get("effort", "medium")
            # Map effort levels to thinking budget tokens
            # Anthropic uses budget_tokens to control thinking depth
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
            logger.info(f"Using extended thinking (budget={budget}) for model {model}")
        
        # Initialize LangChain ChatAnthropic
        # LangChain handles retry logic with exponential backoff internally
        self._llm = ChatAnthropic(  # type: ignore[call-arg]
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            max_tokens=effective_max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **model_kwargs,
        )
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def get_capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def _normalize_list_content(self, content_list: list) -> str:
        """Anthropic can return content as a list of text blocks."""
        text_parts: List[str] = []
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
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image using LangChain.
        
        Note: Anthropic doesn't use image_detail or media_resolution parameters.
        These parameters are accepted for API compatibility but ignored.
        """
        caps = self._capabilities
        
        if not caps.supports_vision:
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
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Native structured outputs (no function calling)
        # We require json_schema mode so the model returns a validated JSON object.
        if json_schema and not caps.supports_structured_output:
            raise ValueError(
                f"Selected Anthropic model '{self.model}' does not support native structured outputs. "
                f"Choose a Claude model that supports structured outputs (e.g. claude-sonnet-4-5-* or claude-opus-4-1-*)."
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
        messages: List[Any],
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic internally.
        Response parsing and token tracking are handled by the shared
        BaseProvider._process_llm_response() method.
        """
        try:
            response = await llm.ainvoke(messages)
            return await self._process_llm_response(response, ANTHROPIC_TOKEN_MAPPING)
        except Exception as e:
            logger.error(f"Error invoking Anthropic: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
