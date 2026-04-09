"""Custom OpenAI-compatible endpoint provider using LangChain.

Supports any self-hosted or third-party endpoint that implements the
OpenAI Chat Completions API with vision (base64 image) support.

Configuration is entirely user-driven via model_config.yaml:
  - custom_endpoint.base_url: the endpoint URL
  - custom_endpoint.api_key_env_var: env var name for the Bearer token

The model name is passed verbatim to the endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.llm.providers.base import (
    BaseProvider,
    OPENAI_TOKEN_MAPPING,
    TranscriptionResult,
    load_max_retries,
)
from modules.infra.logger import setup_logger
from modules.llm.model_capabilities import Capabilities

logger = setup_logger(__name__)


class CustomProvider(BaseProvider):
    """Provider for custom OpenAI-compatible endpoints.

    Uses LangChain's ChatOpenAI with a user-supplied base_url to
    communicate with any endpoint that follows the OpenAI Chat
    Completions API contract (including vision/base64 image input).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

        self._base_url = base_url

        # Conservative capabilities for arbitrary custom models
        self._capabilities = Capabilities(
            model=model,
            family="custom",
            provider="custom",
            supports_responses_api=False,
            supports_chat_completions=True,
            api_preference="langchain",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=False,
            default_image_detail="high",
            supports_structured_outputs=False,
            supports_json_mode=False,
            supports_function_calling=False,
            supports_sampler_controls=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=kwargs.get("max_context_tokens", 8192),
            max_output_tokens=max_tokens,
        )

        max_retries = load_max_retries()

        self._llm = ChatOpenAI(
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
        )

        logger.info(
            f"CustomProvider initialized: model={model}, "
            f"base_url={base_url}, max_tokens={max_tokens}"
        )

    @property
    def provider_name(self) -> str:
        return "custom"

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
        """Transcribe text from a base64-encoded image.

        Uses the OpenAI-compatible Chat Completions format with a
        base64 data URL. Structured output (json_schema) is not
        supported by most custom endpoints and will be ignored.
        """
        if not self._capabilities.supports_image_input:
            return TranscriptionResult(
                content="",
                error=f"Model {self.model} does not support vision/image inputs.",
                transcription_not_possible=True,
            )

        # Build data URL
        data_url = self.create_data_url(image_base64, mime_type)

        # Build image content block (OpenAI format)
        image_content: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]

        if json_schema:
            logger.warning(
                "Custom endpoint provider does not support structured "
                "output (json_schema). Using standard text output."
            )

        # Invoke LLM -- LangChain handles retries internally
        try:
            response = await self._ainvoke_with_retry(
                self._llm, messages, expect_image_tokens=True
            )
            return await self._process_llm_response(
                response, OPENAI_TOKEN_MAPPING
            )
        except Exception as e:
            logger.error(f"Error invoking custom endpoint: {e}")
            return TranscriptionResult(content="", error=str(e))

    async def close(self) -> None:
        """Clean up resources."""
        pass
