"""Custom OpenAI-compatible endpoint provider using LangChain.

Supports any self-hosted or third-party endpoint that implements the
OpenAI Chat Completions API with vision (base64 image) support.

Configuration is entirely user-driven via model_config.yaml:
  - custom_endpoint.base_url: the endpoint URL
  - custom_endpoint.api_key_env_var: env var name for the Bearer token
  - custom_endpoint.capabilities: optional capability overrides
  - custom_endpoint.use_plain_text_prompt: use simplified plain-text prompt

Three operating modes (determined by config):

  Structured (supports_structured_output: true):
    Full schema enforcement via with_structured_output(). Pydantic
    validation catches missing/wrong fields and triggers validation
    retries via _ainvoke_with_retry().

  JSON-instructed (default; supports_structured_output: false,
    use_plain_text_prompt: false):
    Normal prompt with schema as a text instruction. No API-level
    enforcement; response sanitisation in response_parsing.py handles
    code-fenced JSON and conversational preamble.

  Plain text (supports_structured_output: false,
    use_plain_text_prompt: true):
    Simplified prompt — no JSON, markdown, or special tags.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from modules.config.capabilities import Capabilities
from modules.infra.logger import setup_logger
from modules.llm.providers.base import (
    OPENAI_TOKEN_MAPPING,
    BaseProvider,
    TranscriptionResult,
    load_max_retries,
)

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
        timeout: float | None = None,
        custom_capabilities: dict[str, Any] | None = None,
        use_plain_text_prompt: bool = False,
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
        self._use_plain_text_prompt = use_plain_text_prompt

        # Build capabilities from conservative defaults + user overrides
        caps = custom_capabilities or {}
        supports_vision = bool(caps.get("supports_vision", True))
        supports_structured = bool(caps.get("supports_structured_output", False))

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
            supports_image_input=supports_vision,
            supports_image_detail=False,
            default_image_detail="high",
            supports_structured_outputs=supports_structured,
            supports_json_mode=supports_structured,
            supports_function_calling=supports_structured,
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
            max_tokens=max_tokens,  # type: ignore[call-arg]  # langchain-openai stubs omit max_tokens
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
        )

        mode = (
            "structured"
            if supports_structured
            else ("plain-text" if use_plain_text_prompt else "json-instructed")
        )
        logger.info(
            "CustomProvider initialized: model=%s, base_url=%s, max_tokens=%s, mode=%s",
            model,
            base_url,
            max_tokens,
            mode,
        )

    @property
    def provider_name(self) -> str:
        return "custom"

    @property
    def use_plain_text_prompt(self) -> bool:
        return self._use_plain_text_prompt

    def get_capabilities(self) -> Capabilities:
        return self._capabilities

    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: dict[str, Any] | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
        context_image_base64: str | None = None,
        context_image_mime_type: str | None = None,
        context_image_detail: str | None = None,
        context_image_instruction: str = "Context image:",
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image.

        Uses the OpenAI-compatible Chat Completions format with a
        base64 data URL.

        When ``supports_structured_outputs`` is ``True`` (Mode A),
        ``with_structured_output()`` is applied with the Pydantic
        ``TranscriptionOutput`` model for field-level validation.
        Invalid responses trigger retries via ``_ainvoke_with_retry()``.
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
        image_content: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

        content_blocks: list[str | dict[str, Any]] = []

        if context_image_base64 and context_image_mime_type:
            ctx_data_url = self.create_data_url(
                context_image_base64, context_image_mime_type
            )
            ctx_block: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": ctx_data_url},
            }
            if context_image_instruction:
                content_blocks.append(
                    {"type": "text", "text": context_image_instruction}
                )
            content_blocks.append(ctx_block)

        if user_instruction:
            content_blocks.append({"type": "text", "text": user_instruction})
        content_blocks.append(image_content)

        messages: list[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_blocks),
        ]

        caps = self._capabilities
        llm_to_use = self._llm

        # Mode A: Structured output via Pydantic validation
        if json_schema and caps.supports_structured_outputs:
            try:
                from modules.llm.schemas import TranscriptionOutput

                llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                    TranscriptionOutput,
                    include_raw=True,
                )
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
        elif json_schema and not caps.supports_structured_outputs:
            logger.debug(
                "Custom endpoint operating in %s mode; "
                "json_schema passed as text instruction only.",
                "plain-text" if self._use_plain_text_prompt else "json-instructed",
            )

        # Invoke LLM -- LangChain handles retries internally
        try:
            response = await self._ainvoke_with_retry(
                llm_to_use, messages, expect_image_tokens=True
            )
            return await self._process_llm_response(response, OPENAI_TOKEN_MAPPING)
        except Exception as e:
            # logger.exception captures the traceback so a programming error
            # (e.g. KeyError from a refactor) is not masked as an API error.
            logger.exception("Error invoking custom endpoint: %s", e)
            return TranscriptionResult(content="", error=str(e))

    async def close(self) -> None:
        """Clean up resources."""
        pass
