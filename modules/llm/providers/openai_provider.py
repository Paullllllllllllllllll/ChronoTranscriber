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
    ProviderCapabilities,
    TranscriptionResult,
)
from modules.config.service import get_config_service

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
            supports_reasoning_effort=False,
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
            supports_reasoning_effort=False,
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
            supports_reasoning_effort=False,
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
            supports_reasoning_effort=False,
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
            supports_reasoning_effort=False,
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
            supports_reasoning_effort=False,
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
            supports_reasoning_effort=False,
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


def _load_max_retries() -> int:
    """Load max retries from concurrency_config.yaml.
    
    LangChain's ChatOpenAI handles retry logic internally with exponential backoff.
    We just need to configure the max attempts.
    """
    try:
        conc_cfg = get_config_service().get_concurrency_config() or {}
        trans_cfg = (conc_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = trans_cfg.get("retry", {}) or {}
        attempts = int(retry_cfg.get("attempts", 5))
        return max(1, attempts)
    except Exception:
        return 5


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
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.service_tier = service_tier
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        # LangChain will automatically filter these out before sending to API
        disabled_params = self._build_disabled_params()
        
        # Build model kwargs - include all params, LangChain will filter via disabled_params
        model_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # Initialize LangChain ChatOpenAI
        # LangChain handles:
        # - Retry logic with exponential backoff (max_retries)
        # - Parameter filtering for unsupported models (disabled_params)
        self._llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            disabled_params=disabled_params,
            **model_kwargs,
        )
    
    def _build_disabled_params(self) -> Dict[str, Any]:
        """Build disabled_params dict based on model capabilities.
        
        LangChain's disabled_params feature automatically filters out
        unsupported parameters before sending to the API.
        """
        caps = self._capabilities
        disabled = {}
        
        # Disable sampler controls for reasoning models
        if not caps.supports_temperature:
            disabled["temperature"] = None
        if not caps.supports_top_p:
            disabled["top_p"] = None
        if not caps.supports_frequency_penalty:
            disabled["frequency_penalty"] = None
        if not caps.supports_presence_penalty:
            disabled["presence_penalty"] = None
        
        return disabled if disabled else None
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
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
            try:
                from modules.llm.schemas import TranscriptionOutput
                llm_to_use = self._llm.with_structured_output(TranscriptionOutput)
                use_pydantic = True
            except ImportError:
                # Fallback to JSON schema if Pydantic model not available
                if isinstance(json_schema, dict) and "schema" in json_schema:
                    actual_schema = json_schema["schema"]
                else:
                    actual_schema = json_schema
                llm_to_use = self._llm.with_structured_output(
                    actual_schema,
                    method="json_schema",
                    strict=True,
                )
        
        # Invoke LLM - LangChain handles retries internally
        return await self._invoke_llm(llm_to_use, messages, use_pydantic)
    
    async def _invoke_llm(
        self,
        llm,
        messages: List,
        use_pydantic: bool = False,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles:
        - Retry logic with exponential backoff
        - Token usage tracking in response_metadata
        - Structured output parsing
        """
        try:
            # LangChain handles retries for transient errors internally
            response = await llm.ainvoke(messages)
            
            # Extract content - LangChain returns different types based on structured output
            if use_pydantic and hasattr(response, 'model_dump'):
                # Pydantic model response - LangChain parsed it for us
                content = response.model_dump_json()
                parsed_output = response.model_dump()
            elif hasattr(response, 'content'):
                # Standard AIMessage response
                content = response.content
                parsed_output = None
                if isinstance(content, dict):
                    parsed_output = content
                    content = json.dumps(content)
                elif not isinstance(content, str):
                    content = str(content)
            elif isinstance(response, dict):
                # Dict response (structured output)
                content = json.dumps(response)
                parsed_output = response
            else:
                content = str(response)
                parsed_output = None
            
            # Extract token usage from response_metadata (LangChain standard)
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            raw_response = {}
            
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if isinstance(metadata, dict):
                    raw_response = metadata
                    # LangChain standardizes token usage in 'token_usage' key
                    usage = metadata.get('token_usage', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('prompt_tokens', 0)
                        output_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)
            
            # Track tokens using our daily tracker
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
            
            # Create result - TranscriptionResult will parse flags from content
            result = TranscriptionResult(
                content=content,
                raw_response=raw_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            
            # If we have parsed output, set the flags directly
            if parsed_output and isinstance(parsed_output, dict):
                result.parsed_output = parsed_output
                result.no_transcribable_text = parsed_output.get('no_transcribable_text', False)
                result.transcription_not_possible = parsed_output.get('transcription_not_possible', False)
            
            return result
            
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
