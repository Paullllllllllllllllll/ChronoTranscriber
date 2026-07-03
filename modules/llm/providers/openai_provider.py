"""OpenAI provider implementation using LangChain.

Supports all OpenAI models including:
- GPT-5.4, GPT-5.4 Pro (original image detail, xhigh reasoning)
- GPT-5.3 Instant (non-reasoning, fast)
- GPT-5.2, GPT-5.1, GPT-5 family (with reasoning controls)
- GPT-4o, GPT-4o-mini, GPT-4.1 family
- o1, o3 reasoning models

LangChain handles:
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)

Retries are owned solely by BaseProvider._ainvoke_with_retry (tenacity); the SDK
client is built with max_retries=0.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from modules.config.capabilities import Capabilities, detect_capabilities
from modules.infra.logger import setup_logger
from modules.llm.providers.base import (
    OPENAI_TOKEN_MAPPING,
    BaseProvider,
    TranscriptionResult,
    aclose_chat_model,
)

logger = setup_logger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider using LangChain.

    LangChain handles:
    - Token usage tracking (via response_metadata)
    - Structured output parsing (via with_structured_output)
    - Parameter filtering for unsupported models (via disabled_params)

    Retries are owned solely by the tenacity loop in BaseProvider (max_retries=0).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        service_tier: str | None = None,
        reasoning_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
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

        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.service_tier = service_tier
        self.reasoning_config = reasoning_config
        self.text_config = text_config

        self._capabilities = detect_capabilities(model)

        # Build disabled_params for models that don't support certain features
        # LangChain will automatically filter these out before sending to API
        disabled_params = self._build_disabled_params()

        # Initialize LangChain ChatOpenAI with Responses API
        # LangChain handles:
        # - Parameter filtering for unsupported models (disabled_params)
        # - Converting max_completion_tokens to correct API parameter for
        #   reasoning models
        # - Responses API routing (use_responses_api=True)
        # max_retries=0 disables SDK-internal retries so the tenacity loop in
        # BaseProvider._ainvoke_with_retry is the single retry authority.
        llm_kwargs = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout,
            "max_retries": 0,
            "disabled_params": disabled_params,
            "use_responses_api": True,
        }

        # Pass service_tier to LangChain
        # (OpenAI API supports auto/default/flex/priority)
        if service_tier:
            llm_kwargs["service_tier"] = service_tier
            logger.info(f"Using service_tier={service_tier} for model {model}")

        # For reasoning models (GPT-5, o-series), use max_completion_tokens
        # instead of max_tokens
        # and skip sampler parameters which are not supported
        caps = self._capabilities
        if caps.is_reasoning_model:
            llm_kwargs["max_completion_tokens"] = max_tokens
            # Explicitly null out sampler params unsupported by the Responses API.
            # LangChain's disabled_params only filters bind-time kwargs in
            # with_structured_output, NOT _default_params used in the main call.
            # Setting these to None ensures they pass LangChain's exclude_if_none
            # filter and never reach responses.parse() / responses.create().
            llm_kwargs["temperature"] = None
            llm_kwargs["top_p"] = None
            llm_kwargs["frequency_penalty"] = None
            llm_kwargs["presence_penalty"] = None
            logger.info(
                f"Using max_completion_tokens={max_tokens} for reasoning model {model}"
            )

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
            # Only pass sampler parameters for non-reasoning models, and
            # only those that the capability registry marks as supported.
            # The Responses API rejects frequency_penalty / presence_penalty
            # for gpt-4o / gpt-4.1 etc., so omit them entirely when the
            # capability flag is False — LangChain's disabled_params does
            # not filter _default_params, and passing None confuses the
            # Responses API (it returns a generic invalid_request_error).
            if caps.supports_sampler_controls:
                llm_kwargs["temperature"] = temperature
            if caps.supports_top_p:
                llm_kwargs["top_p"] = top_p
            if caps.supports_frequency_penalty:
                llm_kwargs["frequency_penalty"] = frequency_penalty
            if caps.supports_presence_penalty:
                llm_kwargs["presence_penalty"] = presence_penalty

        # Prompt cache retention (OpenAI automatic caching extension)
        if self._caching_enabled:
            openai_cfg = self._caching_config.get("openai", {})
            retention = (
                openai_cfg.get("prompt_cache_retention")
                if isinstance(openai_cfg, dict)
                else None
            )
            if retention:
                model_kwargs = llm_kwargs.setdefault("model_kwargs", {})
                assert isinstance(model_kwargs, dict)
                model_kwargs["prompt_cache_retention"] = retention

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
        json_schema: dict[str, Any] | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
        context_image_base64: str | None = None,
        context_image_mime_type: str | None = None,
        context_image_detail: str | None = None,
        context_image_instruction: str = "Context image:",
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
            valid_details = {"low", "high"}
            if caps.supports_image_detail_original:
                valid_details.add("original")
            if detail not in valid_details:
                detail = None
        if detail is None:
            detail = caps.default_image_detail if caps.supports_image_detail else None

        # Build data URL
        data_url = self.create_data_url(image_base64, mime_type)

        # Build message content with image
        image_content: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        if detail and caps.supports_image_detail:
            image_content["image_url"]["detail"] = detail

        # Assemble content blocks: optional context image, then page image
        content_blocks: list[str | dict[str, Any]] = []

        if context_image_base64 and context_image_mime_type:
            ctx_data_url = self.create_data_url(
                context_image_base64, context_image_mime_type
            )
            ctx_block: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": ctx_data_url},
            }
            ctx_det = context_image_detail
            if ctx_det:
                ctx_det = ctx_det.lower().strip()
            if ctx_det and caps.supports_image_detail:
                ctx_block["image_url"]["detail"] = ctx_det
            if context_image_instruction:
                content_blocks.append(
                    {"type": "text", "text": context_image_instruction}
                )
            content_blocks.append(ctx_block)

        if user_instruction:
            content_blocks.append({"type": "text", "text": user_instruction})
        content_blocks.append(image_content)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_blocks),
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
        messages: list[Any],
        use_pydantic: bool = False,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.

        LangChain handles retry logic with exponential backoff internally.
        Response parsing and token tracking are handled by the shared
        BaseProvider._process_llm_response() method.
        """
        try:
            response = await self._ainvoke_with_retry(
                llm, messages, expect_image_tokens=True
            )
            return await self._process_llm_response(response, OPENAI_TOKEN_MAPPING)
        except Exception as e:
            # logger.exception captures the traceback so a programming error
            # (e.g. KeyError from a refactor) is not masked as an API error.
            logger.exception(f"Error invoking OpenAI: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
            )

    async def close(self) -> None:
        """Dispose the underlying LangChain/SDK HTTP clients. Never raises."""
        await aclose_chat_model(getattr(self, "_llm", None))
