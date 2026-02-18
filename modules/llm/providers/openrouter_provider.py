"""OpenRouter provider implementation using LangChain.

OpenRouter provides access to 200+ models from multiple providers
through a unified OpenAI-compatible API.

Supported model families include:
- OpenAI models (GPT-5, GPT-4o, o1/o3, GPT-OSS-120B/20B)
- Anthropic models (Claude 4.5, Claude 3.5)
- Google models (Gemini 3, Gemini 2.5)
- DeepSeek models (R1, V3.2, V3.1)
- Meta models (Llama 3.2/3.3)
- Mistral models (Mistral Large, Pixtral)
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
    OPENROUTER_TOKEN_MAPPING,
    TranscriptionResult,
    load_max_retries,
)
from modules.llm.model_capabilities import Capabilities, detect_capabilities
from modules.config.service import get_config_service

logger = logging.getLogger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _effort_to_ratio(effort: str) -> float:
    e = (effort or "").strip().lower()
    return {
        "xhigh": 0.95,
        "high": 0.80,
        "medium": 0.50,
        "low": 0.20,
        "minimal": 0.10,
        "none": 0.0,
    }.get(e, 0.50)


def _compute_openrouter_reasoning_max_tokens(*, max_tokens: int, effort: str) -> int:
    ratio = _effort_to_ratio(effort)
    if ratio <= 0:
        return 0
    # Keep budget within OpenRouter docs guidance for Anthropic reasoning.
    # Ensure some tokens remain for the final response.
    reserve_for_answer = 256
    upper = max(0, int(max_tokens) - reserve_for_answer)
    budget = int(int(max_tokens) * ratio)
    budget = min(budget, 32000, upper)
    budget = max(budget, 1024)
    return budget



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
        self.site_url = site_url
        self.app_name = app_name or "ChronoTranscriber"
        self.reasoning_config = reasoning_config
        
        self._capabilities = detect_capabilities(model)
        max_retries = load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        disabled_params = self._build_disabled_params()
        
        # Build model kwargs - include all params, LangChain will filter via disabled_params
        model_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # Apply OpenRouter unified reasoning controls.
        # OpenRouter accepts a top-level `reasoning` object and will route/translate
        # it when supported by the selected model/provider.
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            reasoning_payload: Dict[str, Any] = {}

            effort = reasoning_config.get("effort")
            if effort:
                reasoning_payload["effort"] = str(effort)

            max_reasoning_tokens = reasoning_config.get("max_tokens")
            if max_reasoning_tokens is not None:
                try:
                    reasoning_payload["max_tokens"] = int(max_reasoning_tokens)
                except Exception:
                    pass

            exclude = reasoning_config.get("exclude")
            if exclude is not None:
                reasoning_payload["exclude"] = bool(exclude)

            enabled = reasoning_config.get("enabled")
            if enabled is not None:
                reasoning_payload["enabled"] = bool(enabled)

            if reasoning_payload:
                # Avoid sending both effort and max_tokens for models where OpenRouter expects
                # one or the other.
                m = (model or "").lower().strip()

                # For Anthropic and Gemini thinking models, OpenRouter supports reasoning.max_tokens.
                # Map effort -> max_tokens budget when max_tokens isn't explicitly provided.
                if ("anthropic/" in m or "claude" in m or "gemini" in m) and "max_tokens" not in reasoning_payload:
                    eff = (reasoning_payload.get("effort") or "medium")
                    budget = _compute_openrouter_reasoning_max_tokens(max_tokens=max_tokens, effort=str(eff))
                    if budget > 0:
                        reasoning_payload.pop("effort", None)
                        reasoning_payload["max_tokens"] = budget

                # For DeepSeek models, OpenRouter docs emphasize enabling reasoning, rather than effort.
                if "deepseek/" in m or "deepseek" in m:
                    eff = str(reasoning_payload.get("effort") or "medium").lower().strip()
                    reasoning_payload.pop("effort", None)
                    if "enabled" not in reasoning_payload:
                        reasoning_payload["enabled"] = eff != "none"

                # OpenRouter's OpenAI-compatible endpoint expects this under extra_body.
                extra_body = model_kwargs.get("extra_body")
                if not isinstance(extra_body, dict):
                    extra_body = {}
                extra_body["reasoning"] = reasoning_payload
                model_kwargs["extra_body"] = extra_body

                logger.info(f"Using OpenRouter reasoning={reasoning_payload} for model {model}")
        
        # OpenRouter-specific headers
        default_headers = {
            "HTTP-Referer": site_url or "https://github.com/ChronoTranscriber",
            "X-Title": self.app_name,
        }
        
        # Initialize LangChain ChatOpenAI with OpenRouter endpoint
        # LangChain handles:
        # - Retry logic with exponential backoff (max_retries)
        # - Parameter filtering for unsupported models (disabled_params)
        self._llm = ChatOpenAI(  # type: ignore[call-arg]
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            base_url=OPENROUTER_BASE_URL,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            disabled_params=disabled_params,
            default_headers=default_headers,
            **model_kwargs,
        )
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
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
        
        Note: OpenRouter uses image_detail for OpenAI-compatible models, not media_resolution.
        The media_resolution parameter is accepted for API compatibility but ignored.
        """
        caps = self._capabilities
        
        if not caps.supports_image_input:
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
        # Use include_raw=True to get token usage from the underlying AIMessage
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_outputs:
            # Unwrap schema if wrapped
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
            # Note: Not all OpenRouter models support structured output
            # For those that do, use json_mode method as it's more widely supported
            try:
                llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                    actual_schema,
                    method="json_mode",
                    include_raw=True,
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
            return await self._process_llm_response(response, OPENROUTER_TOKEN_MAPPING)
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
