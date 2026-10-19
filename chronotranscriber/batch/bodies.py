"""Request bodies of batch transcription, built by the shared request builder.

:func:`build_plan` reads the run-invariant parts once: the system prompt
with the schema rendered and the context injected, the schema, the response
format, the image detail, the request plan and the context image.
:meth:`BodyPlan.body` then turns each :class:`~chronotranscriber.common.batch.
BatchRequest` into the provider-native body the shared backends wrap, through
:func:`~chronotranscriber.common.provider_requests.batch_body`: Responses API
bodies for OpenAI, Messages API parameters for Anthropic and
``InlinedRequest`` bodies for Google.

The bodies carry what the synchronous calls send, from the same
:class:`~chronotranscriber.common.provider_requests.RequestPlan`
(:func:`~chronotranscriber.llm.options.request_plan`): output limit,
reasoning effort or thinking, samplers, verbosity, service tier, prompt
caching and, for a Gemini page, its media resolution. ``schema`` enforces the
schema (OpenAI ``text.format`` named as the schema file names it,
``transcription`` for a bare schema; Anthropic ``output_config.format`` on
the schema the synchronous request sends; Google ``response_json_schema``),
``json`` sends it only in the prompt and ``text`` uses the plain-text prompt
without a schema. The format follows the synchronous rules: without
``--response-format`` a model with structured output enforces the schema and
any other model gets prompted JSON, as does an Anthropic schema with more
union parameters than Anthropic compiles.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronotranscriber.common.batch import BatchRequest, supports_batch
from chronotranscriber.common.provider_requests import Part, RequestPlan, batch_body
from chronotranscriber.common.structured import ResponseFormat, fit_response_format
from chronotranscriber.llm.options import (
    model_capabilities,
    request_plan,
    response_format_for,
)
from chronotranscriber.llm.prompt_utils import prepare_prompt_with_context
from chronotranscriber.llm.prompts import (
    DEFAULT_SCHEMA_PATH,
    PLAIN_PROMPT_PATH,
    SCHEMA_PROMPT_PATH,
)
from chronotranscriber.llm.transcription import anthropic_schema, wrapped_schema
from chronotranscriber.settings import Settings
from chronotranscriber.sources.images import encode_image_to_base64
from chronotranscriber.spec import ModelSpec

__all__ = [
    "CONTEXT_IMAGE_INSTRUCTION",
    "USER_INSTRUCTION",
    "BodyPlan",
    "build_plan",
]

USER_INSTRUCTION = "The image:"
CONTEXT_IMAGE_INSTRUCTION = "Context image:"


@dataclass(frozen=True)
class BodyPlan:
    """The run-invariant inputs of every request body of one submission.

    *schema* is the schema file's content (a ``{name, strict, schema}``
    wrapper or a bare schema), None in the text format; *detail* is the
    image detail of requests that name none; *context_image* the base64
    data and MIME type of the context image.
    """

    plan: RequestPlan
    system_prompt: str
    response_format: ResponseFormat
    schema: Mapping[str, Any] | None = None
    detail: str | None = None
    context_image: tuple[str, str] | None = None

    @property
    def provider(self) -> str:
        """The provider of the submission."""
        return self.plan.provider

    @property
    def model(self) -> str:
        """The model every body names."""
        return self.plan.model

    def _enforced(self) -> Mapping[str, Any] | None:
        """Return the schema spec the provider enforces, None under JSON or text."""
        if self.response_format != "schema" or self.schema is None:
            return None
        if self.plan.provider == "anthropic":
            return anthropic_schema(self.schema)
        return wrapped_schema(self.schema)

    def body(self, request: BatchRequest) -> dict[str, Any]:
        """Return the provider-native body of *request*."""
        if request.image_base64 is None:
            raise ValueError(f"Request {request.custom_id} has no image data")
        parts: list[Part] = []
        if self.context_image is not None:
            data, mime = self.context_image
            parts.append(Part(text=CONTEXT_IMAGE_INSTRUCTION))
            parts.append(Part(data=data, mime=mime, detail=self.detail))
        parts.append(Part(text=USER_INSTRUCTION))
        parts.append(
            Part(
                data=request.image_base64,
                mime=request.mime_type,
                detail=request.image_detail or self.detail,
            )
        )
        return batch_body(self.plan, self.system_prompt, parts, self._enforced())


def _read_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Schema file {path} does not hold a JSON object")
    return loaded


def build_plan(
    model: ModelSpec,
    settings: Settings,
    *,
    schema_path: Path | None,
    prompt_path: Path | None = None,
    context: str | None = None,
    context_image: Path | None = None,
    detail: str | None = None,
) -> BodyPlan:
    """Return the body plan of a submission for *model*.

    *prompt_path* (``--prompt``) replaces the prompt of the response format;
    *schema_path* defaults to the markdown schema.

    Raises:
        ValueError: The provider has no batch API, or the schema is invalid.
        OSError: The prompt, schema or context image cannot be read.
    """
    if not supports_batch(model.provider):
        raise ValueError(f"Provider '{model.provider}' has no batch API")
    caps = model_capabilities(model, settings)
    response_format = response_format_for(model, caps)
    if prompt_path is None:
        prompt_path = (
            PLAIN_PROMPT_PATH if response_format == "text" else SCHEMA_PROMPT_PATH
        )
    raw_prompt = prompt_path.read_text(encoding="utf-8").strip()
    schema: dict[str, Any] | None = None
    if response_format != "text":
        schema = _read_schema(schema_path or DEFAULT_SCHEMA_PATH)
        response_format = fit_response_format(model.provider, response_format, schema)
    image: tuple[str, str] | None = None
    if context_image is not None and context_image.exists():
        image = encode_image_to_base64(context_image)
    return BodyPlan(
        plan=request_plan(model, settings, caps),
        system_prompt=prepare_prompt_with_context(raw_prompt, schema, context),
        response_format=response_format,
        schema=schema,
        detail=detail,
        context_image=image,
    )
