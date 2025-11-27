"""OpenRouter provider implementation using LangChain.

OpenRouter provides access to 200+ models from multiple providers
through a unified OpenAI-compatible API.

Supported model families include:
- OpenAI models (GPT-4o, o1, etc.)
- Anthropic models (Claude 3 family)
- Google models (Gemini)
- Meta models (Llama)
- Mistral models
- And many more

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

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.llm.providers.base import (
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
)
from modules.config.service import get_config_service

logger = logging.getLogger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on OpenRouter model name.
    
    OpenRouter model names follow the format: provider/model-name
    e.g., openai/gpt-4o, anthropic/claude-3-sonnet, google/gemini-pro
    """
    m = model_name.lower().strip()
    
    # OpenAI models via OpenRouter
    if "openai/" in m or "gpt-4" in m or "gpt-5" in m:
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model="o1" in m or "o3" in m,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        )
    
    # Anthropic models via OpenRouter
    if "anthropic/" in m or "claude" in m:
        return ProviderCapabilities(
            provider_name="openrouter",
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
    
    # Google models via OpenRouter
    if "google/" in m or "gemini" in m:
        return ProviderCapabilities(
            provider_name="openrouter",
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
            max_context_tokens=1000000,
            max_output_tokens=8192,
        )
    
    # Meta Llama models via OpenRouter
    if "meta/" in m or "llama" in m:
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision="vision" in m or "llama-3.2" in m,
            supports_image_detail=False,
            default_image_detail="auto",
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
    
    # Mistral models via OpenRouter
    if "mistral/" in m or "mistral" in m or "mixtral" in m:
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision="pixtral" in m,
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
            max_context_tokens=128000,
            max_output_tokens=4096,
        )
    
    # Default/fallback - assume basic capabilities
    return ProviderCapabilities(
        provider_name="openrouter",
        model_name=model_name,
        supports_vision=True,  # Optimistic default
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
        max_context_tokens=128000,
        max_output_tokens=4096,
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


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider using LangChain.
    
    Uses the OpenAI-compatible API endpoint provided by OpenRouter
    to access 200+ models from various providers.
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
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
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
        self.site_url = site_url
        self.app_name = app_name or "ChronoTranscriber"
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        disabled_params = self._build_disabled_params()
        
        # Build model kwargs - include all params, LangChain will filter via disabled_params
        model_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # OpenRouter-specific headers
        default_headers = {
            "HTTP-Referer": site_url or "https://github.com/ChronoTranscriber",
            "X-Title": self.app_name,
        }
        
        # Initialize LangChain ChatOpenAI with OpenRouter endpoint
        # LangChain handles:
        # - Retry logic with exponential backoff (max_retries)
        # - Parameter filtering for unsupported models (disabled_params)
        self._llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=OPENROUTER_BASE_URL,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            disabled_params=disabled_params,
            default_headers=default_headers,
            **model_kwargs,
        )
    
    def _build_disabled_params(self) -> Dict[str, Any]:
        """Build disabled_params dict based on model capabilities."""
        caps = self._capabilities
        disabled = {}
        
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
        return "openrouter"
    
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
        
        # Build data URL
        data_url = self.create_data_url(image_base64, mime_type)
        
        # Normalize image detail
        detail = image_detail
        if detail:
            detail = detail.lower().strip()
            if detail not in ("low", "high"):
                detail = None
        if detail is None and caps.supports_image_detail:
            detail = caps.default_image_detail
        
        # Build image content block (OpenAI format for OpenRouter)
        image_content: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        if detail and caps.supports_image_detail and detail in ("low", "high"):
            image_content["image_url"]["detail"] = detail
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Use structured output if schema provided and supported
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            # Unwrap schema if wrapped
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
            # Note: Not all OpenRouter models support structured output
            # For those that do, use json_mode method as it's more widely supported
            try:
                llm_to_use = self._llm.with_structured_output(
                    actual_schema,
                    method="json_mode",
                )
            except Exception as e:
                logger.warning(
                    f"Structured output not available for {self.model}, "
                    f"falling back to standard output: {e}"
                )
        
        # Invoke LLM - LangChain handles retries internally
        return await self._invoke_llm(llm_to_use, messages)
    
    async def _invoke_llm(
        self,
        llm,
        messages: List,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic internally.
        """
        try:
            response = await llm.ainvoke(messages)
            
            # Extract content
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
                    usage = metadata.get('token_usage', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('prompt_tokens', 0)
                        output_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)
            
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
            logger.error(f"Error invoking OpenRouter: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
