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
    ProviderCapabilities,
    TranscriptionResult,
)
from modules.config.service import get_config_service

logger = logging.getLogger(__name__)


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on Google model name.
    
    Supports (as of December 2025):
    - Gemini 3: gemini-3-pro, gemini-3-flash-preview (latest, state-of-the-art)
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


def _load_max_retries() -> int:
    """Load max retries from concurrency_config.yaml."""
    try:
        conc_cfg = get_config_service().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = trans_cfg.get("retry", {}) or {}
        attempts = int(retry_cfg.get("attempts", 5))
        return max(1, attempts)
    except Exception:
        return 5


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
        **kwargs,
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
        self.top_k = top_k
        self.reasoning_config = reasoning_config
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build LLM kwargs
        llm_kwargs: Dict[str, Any] = {
            "google_api_key": api_key,
            "model": model,
            "temperature": temperature if self._capabilities.supports_temperature else None,
            "max_tokens": max_tokens,
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
        """Transcribe text from an image file."""
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
            
            llm_to_use = self._llm.with_structured_output(
                actual_schema,
                method="json_schema",
                include_raw=True,
            )
        
        # Apply media_resolution if supported
        # LangChain's ChatGoogleGenerativeAI doesn't directly expose generation_config,
        # so we pass it via model_kwargs if needed
        invoke_kwargs = {}
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
        llm,
        messages: List,
        invoke_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic internally.
        
        When using with_structured_output(include_raw=True), the response is a dict:
        - "raw": The underlying AIMessage with response_metadata containing token usage
        - "parsed": The parsed dict
        - "parsing_error": Any parsing error that occurred
        """
        try:
            # Merge invoke_kwargs if provided
            kwargs = invoke_kwargs or {}
            response = await llm.ainvoke(messages, **kwargs)
            
            # Extract token usage and content
            # Handle include_raw=True response format (dict with raw/parsed/parsing_error)
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            raw_response = {}
            raw_message = None
            parsed_output = None
            
            if isinstance(response, dict) and "raw" in response and "parsed" in response:
                # with_structured_output(include_raw=True) returns {"raw": AIMessage, "parsed": dict}
                raw_message = response.get("raw")
                parsed_data = response.get("parsed")
                
                # Extract parsed content
                if parsed_data is not None:
                    if isinstance(parsed_data, dict):
                        content = json.dumps(parsed_data)
                        parsed_output = parsed_data
                    else:
                        content = str(parsed_data)
                else:
                    # Parsing failed, try to get content from raw message
                    content = raw_message.content if raw_message and hasattr(raw_message, 'content') else ""
                    if isinstance(content, dict):
                        parsed_output = content
                        content = json.dumps(content)
                    elif isinstance(content, list):
                        # Gemini can return content as list of parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "".join(text_parts)
            elif hasattr(response, 'content'):
                # Standard AIMessage response (no structured output)
                raw_message = response
                content = response.content
                if isinstance(content, dict):
                    parsed_output = content
                    content = json.dumps(content)
                elif isinstance(content, list):
                    # Gemini can return content as list of parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "".join(text_parts)
                elif not isinstance(content, str):
                    content = str(content)
            elif isinstance(response, dict):
                # Dict response without raw/parsed structure
                content = json.dumps(response)
                parsed_output = response
            else:
                content = str(response)
            
            # Extract token usage from the raw AIMessage's response_metadata
            if raw_message and hasattr(raw_message, 'response_metadata'):
                metadata = raw_message.response_metadata
                if isinstance(metadata, dict):
                    raw_response = metadata
                    # Gemini uses 'usage_metadata' with different key names
                    usage = metadata.get('usage_metadata', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('prompt_token_count', 0)
                        output_tokens = usage.get('candidates_token_count', 0)
                        total_tokens = usage.get('total_token_count', 0)
                        if total_tokens == 0:
                            total_tokens = input_tokens + output_tokens
            
            # Track tokens
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
            
            # Create result
            result = TranscriptionResult(
                content=content,
                raw_response=raw_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            
            # Set parsed output flags if available
            if parsed_output and isinstance(parsed_output, dict):
                result.parsed_output = parsed_output
                result.no_transcribable_text = parsed_output.get('no_transcribable_text', False)
                result.transcription_not_possible = parsed_output.get('transcription_not_possible', False)
            
            return result
            
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
