"""Google Gemini provider implementation using LangChain.

Supports Gemini models:
- Gemini 3 Flash (latest, Pro-level intelligence at Flash speed)
- Gemini 3 Pro (most capable, state-of-the-art reasoning)
- Gemini 2.5 Pro/Flash (adaptive thinking)
- Gemini 2.0 Flash
- Gemini 1.5 Pro/Flash

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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.llm.providers.base import (
    BaseProvider,
    GOOGLE_TOKEN_MAPPING,
    ProviderCapabilities,
    TranscriptionResult,
    load_max_retries,
)

logger = logging.getLogger(__name__)


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on Google model name.
    
    Supports (as of February 2026):
    - Gemini 3: gemini-3-pro, gemini-3-flash-preview, gemini-3-preview (latest, state-of-the-art)
    - Gemini 2.5: gemini-2.5-pro, gemini-2.5-flash (with adaptive thinking)
    - Gemini 2.0: gemini-2.0-flash
    - Gemini 1.5: gemini-1.5-pro, gemini-1.5-flash
    """
    m = model_name.lower().strip()
    
    # Gemini 3 Flash (Pro-level intelligence at Flash speed)
    # Must check before general "gemini-3" pattern
    if "gemini-3-flash" in m or "gemini-3.0-flash" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,  # Thinking capabilities
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1048576,  # 1M token context
            max_output_tokens=65536,
        )
    
    # Gemini 3 Pro (state-of-the-art reasoning)
    if "gemini-3-pro" in m or "gemini-3.0-pro" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=2000000,  # 2M token context
            max_output_tokens=65536,
        )
    
    # Gemini 3.0 Preview (non-flash; gemini-3-preview, gemini-3.0-preview)
    if ("gemini-3-preview" in m or "gemini-3.0-preview" in m) and "flash" not in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1048576,  # 1M token context
            max_output_tokens=65536,
        )

    # Catch-all for other Gemini 3 variants
    if "gemini-3" in m or "gemini-3.0" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1000000,
            max_output_tokens=65536,
        )
    
    # Gemini 2.5 Pro (with adaptive thinking)
    if "gemini-2.5-pro" in m or "gemini-2.5" in m and "pro" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,  # Thinking mode support
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=2000000,  # 2M token context
            max_output_tokens=65536,
        )
    
    # Gemini 2.5 Flash
    if "gemini-2.5-flash" in m or ("gemini-2.5" in m and "flash" in m):
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1000000,
            max_output_tokens=32768,
        )
    
    # Gemini 2.0 Flash
    if "gemini-2.0" in m or "gemini-2" in m and "flash" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1000000,
            max_output_tokens=8192,
        )
    
    # Gemini 1.5 Pro
    if "gemini-1.5-pro" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=2000000,
            max_output_tokens=8192,
        )
    
    # Gemini 1.5 Flash
    if "gemini-1.5-flash" in m:
        return ProviderCapabilities(
            provider_name="google",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1000000,
            max_output_tokens=8192,
        )
    
    # Default/fallback for Gemini models (assume modern capabilities)
    return ProviderCapabilities(
        provider_name="google",
        model_name=model_name,
        supports_vision=True,
        supports_image_detail=False,
        default_image_detail="auto",
        supports_media_resolution=True,
        default_media_resolution="high",
        supports_structured_output=True,
        supports_json_mode=True,
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_temperature=True,
        supports_top_p=True,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
        max_context_tokens=1000000,
        max_output_tokens=8192,
    )


class GoogleProvider(BaseProvider):
    """Google Gemini LLM provider using LangChain."""
    
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
        
        # Build LLM kwargs
        llm_kwargs: Dict[str, Any] = {
            "google_api_key": api_key,
            "model": model,
            "temperature": temperature if self._capabilities.supports_temperature else None,
            "max_tokens": effective_max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "top_p": top_p if self._capabilities.supports_top_p else None,
            "top_k": top_k,
        }
        
        # Apply thinking mode for Gemini 2.5+ models that support it
        # Maps reasoning_config.effort to Google's thinking_level parameter
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            effort = reasoning_config.get("effort", "medium")
            # Map effort levels to Gemini thinking_level
            # Gemini uses "low" or "high" for thinking_level
            if effort == "low":
                llm_kwargs["thinking_level"] = "low"
            else:
                # medium and high both map to "high" thinking
                llm_kwargs["thinking_level"] = "high"
            logger.info(f"Using thinking_level={llm_kwargs['thinking_level']} for model {model}")
        
        # Initialize LangChain ChatGoogleGenerativeAI
        # LangChain handles retry logic with exponential backoff internally
        self._llm = ChatGoogleGenerativeAI(**llm_kwargs)
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def get_capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def _normalize_list_content(self, content_list: list) -> str:
        """Gemini can return content as a list of text parts."""
        text_parts = []
        for part in content_list:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts) if text_parts else str(content_list)

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
        """Transcribe text from a base64-encoded image using LangChain."""
        caps = self._capabilities
        
        if not caps.supports_vision:
            return TranscriptionResult(
                content="",
                error=f"Model {self.model} does not support vision/image inputs.",
                transcription_not_possible=True,
            )
        
        # Normalize media_resolution parameter
        resolution = media_resolution
        if resolution:
            resolution = resolution.lower().strip()
            if resolution not in ("low", "medium", "high", "ultra_high", "auto"):
                resolution = None
        if resolution is None:
            resolution = caps.default_media_resolution if caps.supports_media_resolution else None
        
        # Map to Google's MediaResolution enum values
        resolution_map = {
            "low": "MEDIA_RESOLUTION_LOW",
            "medium": "MEDIA_RESOLUTION_MEDIUM",
            "high": "MEDIA_RESOLUTION_HIGH",
            "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
            "auto": "MEDIA_RESOLUTION_UNSPECIFIED",
        }
        media_resolution_enum = resolution_map.get(resolution) if resolution else None
        
        # Build data URL for Gemini
        data_url = self.create_data_url(image_base64, mime_type)
        
        # Gemini uses standard image_url format
        # Note: LangChain's ChatGoogleGenerativeAI may not directly support per-part
        # media_resolution in image_url. We'll set it globally via generation config instead.
        image_content = {
            "type": "image_url",
            "image_url": data_url,
        }
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Use structured output if schema provided
        # Use include_raw=True to get token usage from the underlying AIMessage
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            # Unwrap schema if wrapped
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
            llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                actual_schema,
                method="json_schema",
                include_raw=True,
            )
        
        # Apply media_resolution if supported
        # LangChain's ChatGoogleGenerativeAI doesn't directly expose generation_config,
        # so we pass it via model_kwargs if needed
        invoke_kwargs: Dict[str, Any] = {}
        if media_resolution_enum and caps.supports_media_resolution:
            # For LangChain, we can't easily set per-request generation config
            # Log the resolution for debugging
            logger.debug(f"Using media_resolution: {resolution} ({media_resolution_enum})")
            # Note: Current LangChain implementation may not expose this parameter
            # Future enhancement: Pass via generation_config if LangChain supports it
        
        # Invoke LLM - LangChain handles retries internally
        return await self._invoke_llm(llm_to_use, messages, invoke_kwargs)
    
    async def _invoke_llm(
        self,
        llm: Any,
        messages: List[Any],
        invoke_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic internally.
        Response parsing and token tracking are handled by the shared
        BaseProvider._process_llm_response() method.
        """
        try:
            kwargs = invoke_kwargs or {}
            response = await llm.ainvoke(messages, **kwargs)
            return await self._process_llm_response(response, GOOGLE_TOKEN_MAPPING)
        except Exception as e:
            logger.error(f"Error invoking Google Gemini: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
