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
    TranscriptionResult,
    load_max_retries,
)
from modules.llm.model_capabilities import Capabilities, detect_capabilities

logger = logging.getLogger(__name__)


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
        text_config: Optional[Dict[str, Any]] = None,
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
        self.text_config = text_config
        
        self._capabilities = detect_capabilities(model)
        max_retries = load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        # LangChain will automatically filter these out before sending to API
        disabled_params = self._build_disabled_params()
        
        # Initialize LangChain ChatOpenAI with Responses API
        # LangChain handles:
        # - Retry logic with exponential backoff (max_retries)
        # - Parameter filtering for unsupported models (disabled_params)
        # - Converting max_completion_tokens to correct API parameter for reasoning models
        # - Responses API routing (use_responses_api=True)
        llm_kwargs = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout,
            "max_retries": max_retries,
            "disabled_params": disabled_params,
            "use_responses_api": True,
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
            
            # Apply reasoning controls via Responses API reasoning dict
            if caps.supports_reasoning_effort and reasoning_config:
                llm_kwargs["reasoning"] = reasoning_config
                logger.info(f"Using reasoning={reasoning_config} for model {model}")
            
            # Apply text verbosity via Responses API verbosity parameter
            if text_config:
                verbosity = text_config.get("verbosity")
                if verbosity:
                    llm_kwargs["verbosity"] = verbosity
                    logger.info(f"Using verbosity={verbosity} for model {model}")
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
    
    def get_capabilities(self) -> Capabilities:
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
        
        if not caps.supports_image_input:
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
        
        if json_schema and caps.supports_structured_outputs:
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
