"""OpenAI provider implementation using LangChain.

Supports all OpenAI models including:
- GPT-4o, GPT-4o-mini
- GPT-4.1 family
- GPT-5 family (with reasoning controls)
- o1, o3 reasoning models

LangChain handles:
- Retry logic with exponential backoff (max_retries parameter)
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from modules.llm.providers.base import (
    BaseProvider,
    OPENAI_TOKEN_MAPPING,
    ProviderCapabilities,
    TranscriptionResult,
    load_max_retries,
)

logger = logging.getLogger(__name__)


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on OpenAI model name.
    
    Supports (as of November 2025):
    - GPT-5.1 family: gpt-5.1, gpt-5.1-mini, gpt-5.1-nano (with thinking variants)
    - GPT-5 family: gpt-5, gpt-5-mini, gpt-5-nano
    - o4-mini: Latest small reasoning model
    - o3 family: o3, o3-pro, o3-mini
    - o1 family: o1, o1-pro, o1-mini
    - GPT-4.1 family: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
    - GPT-4o family: gpt-4o, gpt-4o-mini
    """
    m = model_name.lower().strip()
    
    # GPT-5.1 family (newest, with adaptive thinking)
    # gpt-5.1, gpt-5.1-mini, gpt-5.1-nano, gpt-5.1-instant, gpt-5.1-thinking
    if m.startswith("gpt-5.1"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=256000,
            max_output_tokens=128000,
        )
    
    # GPT-5 family (standard, mini, nano variants)
    # gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5-chat
    if m.startswith("gpt-5"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=256000,
            max_output_tokens=128000,
        )
    
    # o4-mini (newest reasoning model, optimized for speed and cost)
    if m.startswith("o4-mini") or m.startswith("o4"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o3-pro (highest capability reasoning)
    if m.startswith("o3-pro"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o3 family (not o3-mini, not o3-pro)
    if m == "o3" or m.startswith("o3-20") or (m.startswith("o3") and not m.startswith("o3-mini") and not m.startswith("o3-pro")):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o3-mini
    if m.startswith("o3-mini"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=False,  # No vision
            supports_image_detail=False,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o1-pro
    if m.startswith("o1-pro"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=False,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o1 family (not o1-mini, not o1-pro)
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini") and not m.startswith("o1-pro")):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=False,  # Conservative for o-series
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o1-mini
    if m.startswith("o1-mini"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=False,  # No vision
            supports_image_detail=False,
            default_image_detail="high",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=128000,
            max_output_tokens=65536,
        )
    
    # GPT-4o family (multimodal workhorse)
    if m.startswith("gpt-4o"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
            max_context_tokens=128000,
            max_output_tokens=16384,
        )
    
    # GPT-4.1 family (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)
    if m.startswith("gpt-4.1"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
            max_context_tokens=1000000,  # Million token context
            max_output_tokens=32768,
        )
    
    # Default/fallback (conservative)
    return ProviderCapabilities(
        provider_name="openai",
        model_name=model_name,
        supports_vision=True,
        supports_image_detail=True,
        default_image_detail="high",
        supports_structured_output=True,
        supports_json_mode=True,
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_temperature=True,
        supports_top_p=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
        max_context_tokens=128000,
        max_output_tokens=4096,
    )


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider using LangChain.
    
    LangChain handles:
    - Automatic retry with exponential backoff (via max_retries)
    - Token usage tracking (via response_metadata)
    - Structured output parsing (via with_structured_output)
    - Parameter filtering for unsupported models (via disabled_params)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        service_tier: Optional[str] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
        
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.service_tier = service_tier
        self.reasoning_config = reasoning_config
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        # LangChain will automatically filter these out before sending to API
        disabled_params = self._build_disabled_params()
        
        # Initialize LangChain ChatOpenAI
        # LangChain handles:
        # - Retry logic with exponential backoff (max_retries)
        # - Parameter filtering for unsupported models (disabled_params)
        # - Converting max_completion_tokens to correct API parameter for reasoning models
        llm_kwargs = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout,
            "max_retries": max_retries,
            "disabled_params": disabled_params,
        }
        
        # Pass service_tier to LangChain (OpenAI API supports auto/default/flex/priority)
        if service_tier:
            llm_kwargs["service_tier"] = service_tier
            logger.info(f"Using service_tier={service_tier} for model {model}")
        
        # For reasoning models (GPT-5, o-series), use max_completion_tokens instead of max_tokens
        # and skip sampler parameters which are not supported
        caps = self._capabilities
        if caps.is_reasoning_model:
            llm_kwargs["max_completion_tokens"] = max_tokens
            logger.info(f"Using max_completion_tokens={max_tokens} for reasoning model {model}")
            
            # Apply reasoning controls for models that support them
            if caps.supports_reasoning_effort and reasoning_config:
                effort = reasoning_config.get("effort")
                if effort:
                    llm_kwargs["reasoning_effort"] = effort
                    logger.info(f"Using reasoning_effort={effort} for model {model}")
        else:
            llm_kwargs["max_tokens"] = max_tokens
            # Only pass sampler parameters for non-reasoning models
            llm_kwargs["temperature"] = temperature
            llm_kwargs["top_p"] = top_p
            llm_kwargs["frequency_penalty"] = frequency_penalty
            llm_kwargs["presence_penalty"] = presence_penalty
        
        self._llm = ChatOpenAI(**llm_kwargs)  # type: ignore[arg-type]
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def get_capabilities(self) -> ProviderCapabilities:
        return self._capabilities
    
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
        
        Note: OpenAI uses image_detail parameter, not media_resolution.
        The media_resolution parameter is accepted for API compatibility but ignored.
        """
        caps = self._capabilities
        
        if not caps.supports_vision:
            return TranscriptionResult(
                content="",
                error=f"Model {self.model} does not support vision/image inputs.",
                transcription_not_possible=True,
            )
        
        # Normalize image detail
        detail = image_detail
        if detail:
            detail = detail.lower().strip()
            if detail not in ("low", "high"):
                detail = None
        if detail is None:
            detail = caps.default_image_detail if caps.supports_image_detail else None
        
        # Build data URL
        data_url = self.create_data_url(image_base64, mime_type)
        
        # Build message content with image
        image_content: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        if detail and caps.supports_image_detail:
            image_content["image_url"]["detail"] = detail
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Use LangChain's structured output if schema provided and supported
        llm_to_use = self._llm
        use_pydantic = False
        
        if json_schema and caps.supports_structured_output:
            # Try to use Pydantic model for better validation
            # Use include_raw=True to get token usage from the underlying AIMessage
            try:
                from modules.llm.schemas import TranscriptionOutput
                llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                    TranscriptionOutput,
                    include_raw=True,
                )
                use_pydantic = True
            except ImportError:
                # Fallback to JSON schema if Pydantic model not available
                if isinstance(json_schema, dict) and "schema" in json_schema:
                    actual_schema = json_schema["schema"]
                else:
                    actual_schema = json_schema
                llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                    actual_schema,
                    method="json_schema",
                    strict=True,
                    include_raw=True,
                )
        
        # Invoke LLM - LangChain handles retries internally
        return await self._invoke_llm(llm_to_use, messages, use_pydantic)
    
    async def _invoke_llm(
        self,
        llm: Any,
        messages: List[Any],
        use_pydantic: bool = False,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic with exponential backoff internally.
        Response parsing and token tracking are handled by the shared
        BaseProvider._process_llm_response() method.
        """
        try:
            response = await llm.ainvoke(messages)
            return await self._process_llm_response(response, OPENAI_TOKEN_MAPPING)
        except Exception as e:
            logger.error(f"Error invoking OpenAI: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources - LangChain handles session cleanup internally."""
        pass
