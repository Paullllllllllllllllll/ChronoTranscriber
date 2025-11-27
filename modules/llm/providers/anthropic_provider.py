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
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
)
from modules.config.service import get_config_service

logger = logging.getLogger(__name__)


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
            supports_top_p=True,
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
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 4.5 Haiku (fastest, cost-efficient)
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
            supports_top_p=True,
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
    
    # Claude 3.5 Haiku
    if "claude-3-5-haiku" in m or "claude-3.5-haiku" in m:
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
    
    # Claude 3 Opus
    if "claude-3-opus" in m:
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
            max_output_tokens=4096,
        )
    
    # Claude 3 Sonnet
    if "claude-3-sonnet" in m:
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
            max_output_tokens=4096,
        )
    
    # Claude 3 Haiku
    if "claude-3-haiku" in m:
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


def _load_max_retries() -> int:
    """Load max retries from concurrency_config.yaml.
    
    LangChain's ChatAnthropic handles retry logic internally.
    """
    try:
        conc_cfg = get_config_service().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = trans_cfg.get("retry", {}) or {}
        attempts = int(retry_cfg.get("attempts", 5))
        return max(1, attempts)
    except Exception:
        return 5


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
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build LangChain model kwargs
        model_kwargs: Dict[str, Any] = {}
        if self._capabilities.supports_temperature:
            model_kwargs["temperature"] = temperature
        if self._capabilities.supports_top_p:
            model_kwargs["top_p"] = top_p
        if top_k is not None:
            model_kwargs["top_k"] = top_k
        
        # Initialize LangChain ChatAnthropic
        # LangChain handles retry logic with exponential backoff internally
        self._llm = ChatAnthropic(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **model_kwargs,
        )
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
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
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image using LangChain."""
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
        
        # Use structured output if schema provided
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            # Unwrap schema if wrapped
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
            llm_to_use = self._llm.with_structured_output(
                actual_schema,
                method="json_mode",
            )
        
        # Invoke LLM - LangChain handles retries internally
        return await self._invoke_llm(llm_to_use, messages)
    
    async def _invoke_llm(
        self,
        llm,
        messages: List,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic and token tracking internally.
        """
        try:
            response = await llm.ainvoke(messages)
            
            # Extract content - LangChain returns different types based on structured output
            parsed_output = None
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, dict):
                    parsed_output = content
                    content = json.dumps(content)
                elif not isinstance(content, str):
                    content = str(content)
            elif isinstance(response, dict):
                content = json.dumps(response)
                parsed_output = response
            else:
                content = str(response)
            
            # Extract token usage from response_metadata (LangChain standard)
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            raw_response = {}
            
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if isinstance(metadata, dict):
                    raw_response = metadata
                    # Anthropic uses 'usage' with input_tokens/output_tokens
                    usage = metadata.get('usage', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        total_tokens = input_tokens + output_tokens
            
            # Track tokens
            if total_tokens > 0:
                try:
                    from modules.token_tracker import get_token_tracker
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
            logger.error(f"Error invoking Anthropic: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
