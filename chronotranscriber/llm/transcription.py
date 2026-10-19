"""Page transcription: one page image in, the page's response dict out.

:class:`PageTranscriber` builds each provider's messages (system prompt with
the item's context, an optional context image, the page image with its
detail) and sends them through :class:`~chronotranscriber.llm.caller.ModelCaller`
in the run's response format: ``schema`` has the provider enforce the
selected schema, ``json`` puts the schema in the prompt and validates the
answer, ``text`` uses the plain-text prompt. The answer becomes the response
dict the page pipeline records::

    {
        "output_text": str,
        "usage": {"input_tokens": int, "output_tokens": int,
                  "total_tokens": int, "cached_tokens": int,
                  "reasoning_tokens": int},
        "metadata": {...},                # the provider's response metadata
        "parsed": {...},                  # when the answer is a JSON object
        "error": str,                     # only for a failed page
        "error_type": str,                # a stopped answer's outcome kind
    }

:func:`~chronotranscriber.llm.response_parsing.extract_transcribed_text`
turns it into the page text. A failed page returns the dict with ``error``
set instead of raising; an answer the provider stopped (a content filter, a
refusal, an answer cut at the output-token limit, in every response format)
fails the page.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chronotranscriber.common.provider_requests import (
    Part,
    message_content,
    system_content,
)
from chronotranscriber.common.responses import unwrap_raw_response
from chronotranscriber.common.structured import (
    ResponseFormat,
    StructuredRequest,
    StructuredResult,
    fit_response_format,
)
from chronotranscriber.common.usage import Usage, usage_from_response
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.media import SUPPORTED_IMAGE_FORMATS
from chronotranscriber.llm.options import (
    model_capabilities,
    request_plan,
    response_format_for,
)
from chronotranscriber.llm.prompt_utils import (
    inject_additional_context,
    render_prompt_with_schema,
)
from chronotranscriber.llm.prompts import (
    DEFAULT_SCHEMA_PATH,
    PLAIN_PROMPT_PATH,
    SCHEMA_PROMPT_PATH,
)
from chronotranscriber.llm.response_parsing import normalize_llm_text
from chronotranscriber.spec import ModelSpec

if TYPE_CHECKING:
    from langchain_core.messages import SystemMessage

__all__ = [
    "ROLE",
    "PageTranscriber",
    "anthropic_schema",
    "answer_object",
    "error_response",
    "transcribe_image_with_llm",
    "usage_record",
    "wrapped_schema",
]

logger = logging.getLogger(__name__)

ROLE = "transcription"
"""The usage role of page transcription calls."""

USER_INSTRUCTION = "The image:"
CONTEXT_IMAGE_INSTRUCTION = "Context image:"
DEFAULT_DETAIL = "original"
_REQUESTED_DETAILS = ("low", "high", "auto", "original")

BARE_SCHEMA_NAME = "transcription"
ANTHROPIC_SCHEMA_TITLE = "TranscriptionSchema"
ANTHROPIC_SCHEMA_DESCRIPTION = "Schema for document transcription output"

# Per provider: usage block, input, output and total token keys in
# ``response_metadata`` (total is empty when the provider reports none).
_USAGE_KEYS: Mapping[str, tuple[str, str, str, str]] = {
    "anthropic": ("usage", "input_tokens", "output_tokens", ""),
    "google": (
        "usage_metadata",
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
    ),
}
_OPENAI_USAGE_KEYS = (
    "token_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def error_response(message: str) -> dict[str, Any]:
    """Return the response dict of a page that failed before an answer."""
    return {
        "output_text": "",
        "usage": Usage().as_dict(),
        "error": message,
    }


def _int(value: Any) -> int:
    return int(value or 0)


def usage_record(message: Any, provider: str) -> dict[str, int]:
    """Return the token counts of an answer message for the response dict.

    The counts have the shape of :class:`~chronotranscriber.common.usage.
    Usage`, as every stored usage record: LangChain's ``usage_metadata``
    first, else the provider's own usage block in ``response_metadata``.
    Cache reads and writes count as input; ``cached_tokens`` are the reads.
    """
    usage = usage_from_response(message)
    if usage is None:
        usage = _provider_usage(message, provider)
    return usage.as_dict()


def _provider_usage(message: Any, provider: str) -> Usage:
    """Return the usage of the provider's block in ``response_metadata``."""
    block, input_key, output_key, total_key = _USAGE_KEYS.get(
        provider, _OPENAI_USAGE_KEYS
    )
    metadata = getattr(message, "response_metadata", None)
    usage = metadata.get(block) if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return Usage()
    input_tokens = _int(usage.get(input_key))
    output_tokens = _int(usage.get(output_key))
    total_tokens = _int(usage.get(total_key)) if total_key else 0
    total_tokens = total_tokens or input_tokens + output_tokens
    read = _int(usage.get("cache_read_input_tokens"))
    written = _int(usage.get("cache_creation_input_tokens"))
    cached = read
    details = usage.get("prompt_tokens_details")
    if not cached and isinstance(details, dict):
        cached = _int(details.get("cached_tokens"))
    return Usage(
        input_tokens=input_tokens + read + written,
        output_tokens=output_tokens,
        total_tokens=total_tokens + read + written,
        cached_tokens=cached,
    )


def answer_object(text: str) -> dict[str, Any] | None:
    """Return the JSON object an answer holds, after fence and prose removal."""
    normalized = normalize_llm_text(text)
    if not normalized or not normalized.lstrip().startswith("{"):
        return None
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _encode_image(path: Path) -> tuple[str, str]:
    """Return the base64 text and MIME type of an image file.

    Raises:
        ValueError: The file type is not a supported image format.
    """
    mime_type = SUPPORTED_IMAGE_FORMATS.get(path.suffix.lower())
    if not mime_type:
        raise ValueError(f"Unsupported image format: {path.suffix.lower()}")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime_type


def wrapped_schema(loaded: Any) -> dict[str, Any]:
    """Return a schema file's ``{name, strict, schema}`` wrapper.

    A bare schema is wrapped under the name ``transcription``. Synchronous
    and batch requests name the schema from it.
    """
    if isinstance(loaded, dict) and isinstance(loaded.get("schema"), dict):
        return loaded
    return {"name": BARE_SCHEMA_NAME, "strict": True, "schema": loaded}


def anthropic_schema(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return *spec* with the title and description Anthropic's schema takes."""
    body = spec.get("schema", spec)
    inner = dict(body) if isinstance(body, Mapping) else {}
    inner.setdefault("title", ANTHROPIC_SCHEMA_TITLE)
    inner.setdefault("description", ANTHROPIC_SCHEMA_DESCRIPTION)
    return {**{k: v for k, v in spec.items() if k != "schema"}, "schema": inner}


class PageTranscriber:
    """Transcribe page images with the run's model.

    The system prompt is ``prompt_path``, else the schema prompt (rendered
    with the schema of ``schema_path``) or, for the text format, the
    plain-text prompt. ``image_detail`` is the requested detail of the page
    and context images; OpenAI and OpenRouter keep only the tiers the model
    takes. ``context`` is the context text until :meth:`update_context`.

    Raises:
        FileNotFoundError: The prompt or schema file is missing.
        OSError: The key variable is unset.
    """

    def __init__(
        self,
        model: ModelSpec,
        env: CallEnv,
        *,
        schema_path: Path | None = None,
        prompt_path: Path | None = None,
        image_detail: str | None = None,
        context: str | None = None,
        role: str = ROLE,
    ) -> None:
        self.provider = model.provider
        self.model_name = model.model
        self.capabilities = model_capabilities(model, env.settings)
        self.plan = request_plan(model, env.settings, self.capabilities)
        self.page_detail: str | None = None
        self.response_format: ResponseFormat = response_format_for(
            model, self.capabilities
        )
        self.settings = env.settings
        self.schema: dict[str, Any] | None = None
        self._base_prompt = self._load_prompt(prompt_path, schema_path)
        self.response_format = fit_response_format(
            self.provider, self.response_format, self.schema
        )
        self.system_prompt = inject_additional_context(self._base_prompt, context or "")
        detail = (image_detail or "").lower().strip()
        self.image_detail = detail if detail in _REQUESTED_DETAILS else DEFAULT_DETAIL
        self._context_image: tuple[str, str] | None = None
        self.caller = env.caller(role, model)

    @property
    def plain_text(self) -> bool:
        """Whether the model answers in plain text (the text format)."""
        return self.response_format == "text"

    def _load_prompt(self, prompt_path: Path | None, schema_path: Path | None) -> str:
        default = PLAIN_PROMPT_PATH if self.plain_text else SCHEMA_PROMPT_PATH
        path = Path(prompt_path) if prompt_path is not None else default
        if not path.exists():
            raise FileNotFoundError(f"System prompt missing: {path}")
        raw = path.read_text(encoding="utf-8").strip()
        if self.plain_text:
            return raw
        schema_file = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file missing: {schema_file}")
        with schema_file.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.schema = wrapped_schema(loaded)
        return render_prompt_with_schema(raw, loaded)

    def update_context(self, context: str | None) -> None:
        """Put *context* into the system prompt; None removes the section."""
        self.system_prompt = inject_additional_context(self._base_prompt, context or "")

    def update_context_image(self, path: Path | None) -> None:
        """Send *path* as the context image of later pages; None sends none."""
        if path is None or not path.exists():
            self._context_image = None
            return
        self._context_image = _encode_image(path)
        logger.info("Context image loaded: %s", path.name)

    def _system_message(self) -> SystemMessage:
        from langchain_core.messages import SystemMessage

        return SystemMessage(content=system_content(self.plan, self.system_prompt))

    def messages(
        self, image_base64: str, mime_type: str, *, image_detail: str | None = None
    ) -> list[Any]:
        """Return the system and user messages for one page image.

        The context image and its instruction come first, then the page
        instruction and the page image. A page image carries *image_detail*,
        else the detail its page settings resolved (:attr:`page_detail`),
        else the requested detail.
        """
        from langchain_core.messages import HumanMessage

        parts: list[Part] = []
        default = self.page_detail or self.image_detail
        if self._context_image is not None:
            data, mime = self._context_image
            parts.append(Part(text=CONTEXT_IMAGE_INSTRUCTION))
            parts.append(Part(data=data, mime=mime, detail=default))
        parts.append(Part(text=USER_INSTRUCTION))
        detail = image_detail or default
        parts.append(Part(data=image_base64, mime=mime_type, detail=detail))
        content = message_content(self.plan, parts)
        return [self._system_message(), HumanMessage(content=content)]

    def request(self) -> StructuredRequest:
        """Return the structured request of every page.

        ``schema`` checks only that the answer is a JSON object; ``json``
        also checks the schema's required keys. Invalid answers are asked
        again within ``retry.validation_attempts``.
        """
        schema: Mapping[str, Any] | None = self.schema
        if schema is not None and self.provider == "anthropic":
            schema = anthropic_schema(schema)
        policy = self.settings.retry.policy()
        return StructuredRequest(
            provider=self.provider,
            response_format=self.response_format,
            schema=schema,
            required=() if self.response_format != "json" else None,
            validation_retries=max(0, self.settings.retry.validation_attempts - 1),
            validation_delay=lambda retry: policy.backoff(retry - 1, "validation"),
        )

    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        label: str | None = None,
        image_detail: str | None = None,
        media_resolution: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe one page image; a failure returns the dict with ``error``.

        *image_detail* replaces the page image's detail; a Gemini image
        carries the media resolution of *media_resolution*, else of the
        detail.
        """
        if not self.capabilities.supports_vision:
            return error_response(
                f"Model {self.model_name} does not support vision/image inputs."
            )
        if self.provider == "google" and media_resolution:
            image_detail = media_resolution
        messages = self.messages(image_base64, mime_type, image_detail=image_detail)
        try:
            result = await self.caller.call(
                messages,
                self.request(),
                label=label or "page",
                min_input_tokens=self.settings.retry.min_input_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - a failed page returns an error dict
            logger.error(
                "Transcription failed for %s: %s - %s",
                label or "page",
                type(exc).__name__,
                exc,
            )
            return error_response(str(exc))
        return self._response(result)

    async def transcribe_image(self, image_path: Path) -> dict[str, Any]:
        """Transcribe an image file; see :meth:`transcribe_image_from_base64`."""
        data, mime_type = _encode_image(image_path)
        return await self.transcribe_image_from_base64(
            data, mime_type, label=image_path.name
        )

    def _response(self, result: StructuredResult) -> dict[str, Any]:
        message = unwrap_raw_response(result.response)
        response: dict[str, Any] = {
            "output_text": result.text,
            "usage": usage_record(message, self.provider),
        }
        metadata = getattr(message, "response_metadata", None)
        if isinstance(metadata, dict) and metadata:
            response["metadata"] = metadata
        error: str | None = None
        if result.stopped:
            error = result.outcome.message
            response["error_type"] = result.outcome.kind
        elif result.response_format == "schema" and result.text and result.error:
            error = f"Structured output validation failed: {result.error}"
        elif result.not_object:
            error = f"Answer holds no JSON object: {result.error}"
        if error is not None:
            logger.warning("Page answer rejected: %s", error)
            response["output_text"] = ""
            response["error"] = error
            return response
        parsed = answer_object(result.text)
        if parsed:
            response["parsed"] = parsed
        return response

    async def close(self) -> None:
        """Close the clients of the caller. Never raises."""
        await self.caller.aclose()


async def transcribe_image_with_llm(
    image_path: Path, transcriber: PageTranscriber
) -> dict[str, Any]:
    """Transcribe one image file with *transcriber*."""
    return await transcriber.transcribe_image(image_path)
