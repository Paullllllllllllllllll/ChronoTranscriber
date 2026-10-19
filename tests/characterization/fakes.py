"""Fake provider classes for the characterization tests.

``FakeLLM.constructor`` builds stand-ins for the provider classes the client
factory instantiates (``ChatOpenAI``, ``ChatAnthropic``,
``ChatGoogleGenerativeAI``). Each stand-in records its constructor kwargs and
returns a ``ScriptedChatModel`` that answers from per-page scripts, so the
real client factory, request assembly, structured-output routing, retries and
parsing all run. A request is matched to a page by the marker drawn into its
last image (see ``inputs.decode_marker``), never by call order, so the
matching does not depend on the task that sends the request. A schema the
provider enforces is recorded in the form of its ``json_schema``
structured-output call, whether it travels as an invoke kwarg or through
``with_structured_output`` (see ``_enforcement``).

A page script names the answer form (schema JSON, prompted JSON with a code
fence or a preamble, plain text, the no-text and not-possible flags,
malformed JSON) and per-attempt replies that raise or return raw text, so a
test scripts any case without editing this module.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from PIL import Image

from chronotranscriber.common.testing import (
    FakeRequest,
    ScriptedChatModel,
    ScriptedStructured,
    StructuredCall,
)
from tests.characterization.inputs import decode_marker

CUSTOM_KEY_ENV = "CT_FAKE_CUSTOM_API_KEY"
DUMMY_KEYS = {
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_API_KEY": "test-google-key",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    CUSTOM_KEY_ENV: "test-custom-key",
}
_KEY_NAMES = {value: name for name, value in DUMMY_KEYS.items()}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_CLASS_PROVIDERS = {"ChatAnthropic": "anthropic", "ChatGoogleGenerativeAI": "google"}

PROVIDER_CLASSES: tuple[tuple[str, str], ...] = (
    ("langchain_openai", "ChatOpenAI"),
    ("langchain_anthropic", "ChatAnthropic"),
    ("langchain_google_genai", "ChatGoogleGenerativeAI"),
)
"""(public module, class name) of every provider class the fake replaces.

``seams.replace`` swaps each class wherever the tool binds it.
"""

SDK_STRUCTURED_METHODS: dict[str, str] = {
    "ChatOpenAI": "json_schema",
    "ChatAnthropic": "function_calling",
    "ChatGoogleGenerativeAI": "json_schema",
}
"""Default ``with_structured_output`` method of each provider class."""

USAGE: dict[str, int] = {
    "input_tokens": 1500,
    "output_tokens": 400,
    "total_tokens": 1900,
}

Form = Literal[
    "auto",
    "schema",
    "fenced",
    "preamble",
    "text",
    "no_text",
    "not_possible",
    "malformed",
]
"""Answer forms of a page script.

``auto`` answers schema JSON to a structured-output call and to a request
whose system prompt mentions JSON, and plain text otherwise. ``schema`` is
bare JSON, ``fenced`` JSON in a json code fence, ``preamble`` JSON after one
sentence of prose, ``text`` the plain transcription. ``no_text`` and
``not_possible`` set the matching flag (in JSON) or return its bracketed
marker (in text). ``malformed`` is truncated JSON.
"""

NO_TEXT_MARKER = "[No transcribable text]"
NOT_POSSIBLE_MARKER = "[Transcription not possible]"
PREAMBLE = "Here is the transcription of the page:"
MALFORMED = '{"transcription": "Grain deli'


# ============================================================================
# Scripts
# ============================================================================
class FakeAPIError(Exception):
    """Provider error carrying an HTTP status code, as the SDK errors do."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def transient_error() -> FakeAPIError:
    """Return a 503 error, which the provider retry loop retries."""
    return FakeAPIError(503, "Error code: 503 - service unavailable")


def permanent_error() -> FakeAPIError:
    """Return a 400 error, which the provider retry loop does not retry."""
    return FakeAPIError(400, "Error code: 400 - invalid request")


@dataclass(frozen=True)
class Reply:
    """Replace one answer: raise, return raw text, or answer in another form.

    ``exception`` is raised from the model call. Otherwise ``text`` is the
    raw message content, or ``form`` picks the content built from the page
    script. ``refusal`` lands in ``additional_kwargs["refusal"]`` and
    ``response_metadata`` replaces the default metadata.
    """

    exception: BaseException | None = None
    text: str | None = None
    form: Form | None = None
    refusal: str | None = None
    response_metadata: Mapping[str, Any] | None = None


@dataclass
class PageScript:
    """Scripted answers for one page.

    ``replies`` maps an attempt number (counted per page and run from 1) to
    a ``Reply``; ``always`` answers every attempt without an entry there.
    A transient error then success is ``replies={1: Reply(transient_error())}``;
    a permanent failure is ``always=Reply(permanent_error())``.
    """

    transcription: str
    form: Form = "auto"
    replies: dict[int, Reply] = field(default_factory=dict)
    always: Reply | None = None

    def payload(self, form: Form) -> dict[str, Any]:
        """Return the schema-shaped answer for *form*."""
        no_text = form == "no_text"
        not_possible = form == "not_possible"
        return {
            "transcription": None if no_text or not_possible else self.transcription,
            "no_transcribable_text": no_text,
            "transcription_not_possible": not_possible,
        }

    def content(self, form: Form, json_answer: bool) -> str:
        """Return the message content of an answer in *form*.

        *json_answer* tells ``auto``, ``no_text`` and ``not_possible`` whether
        the request expects JSON.
        """
        if form == "auto":
            form = "schema" if json_answer else "text"
        if form == "text":
            return self.transcription
        if form == "malformed":
            return MALFORMED
        if form in ("no_text", "not_possible") and not json_answer:
            return NO_TEXT_MARKER if form == "no_text" else NOT_POSSIBLE_MARKER
        body = json.dumps(self.payload(form), ensure_ascii=False)
        if form == "fenced":
            return f"```json\n{body}\n```"
        if form == "preamble":
            return f"{PREAMBLE}\n\n{body}"
        return body


def default_page(index: int) -> PageScript:
    """Return the default script for page *index* (0-based).

    The text holds a hyphenated line break, a run of internal spaces and a
    run of blank lines, so the outputs show that text is written as returned.
    """
    number = index + 1
    text = (
        f"## Market Records, Part {number:02d}\n\n"
        f"Grain deliveries to the city market fell in {1600 + number}, and the magis-\n"
        "trates fixed the bread assize twice.\n"
        "Wheat      12 sous      Rye      9 sous\n"
        "\n\n\n\n"
        f"<page_number>{number}</page_number>"
    )
    return PageScript(transcription=text)


# ============================================================================
# Request digests
# ============================================================================
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def summarize_schema(schema: Any) -> dict[str, Any]:
    """Return a compact, stable description of a structured-output schema."""
    if not isinstance(schema, Mapping):
        return {"value": repr(schema)}
    inner = schema.get("schema", schema)
    properties = inner.get("properties") if isinstance(inner, Mapping) else None
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str)
    summary: dict[str, Any] = {
        "properties": sorted(properties) if isinstance(properties, Mapping) else None,
        "digest": _sha(canonical),
    }
    for key in ("type", "name", "strict", "title"):
        if key in schema:
            summary[key] = schema[key]
    return summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def describe_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return invoke or bound kwargs with schema payloads summarized."""
    described: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "text" and isinstance(value, Mapping) and "format" in value:
            text = dict(value)
            text["format"] = summarize_schema(text["format"])
            described[key] = _jsonable(text)
        elif key == "response_format" and isinstance(value, Mapping):
            fmt = dict(value)
            if isinstance(fmt.get("json_schema"), Mapping):
                fmt["json_schema"] = summarize_schema(fmt["json_schema"])
            described[key] = _jsonable(fmt)
        elif key == "response_schema":
            described[key] = summarize_schema(value)
        else:
            described[key] = _jsonable(value)
    return described


def _decode_data_url(url: str) -> tuple[str | None, bytes | None]:
    if not url.startswith("data:") or "," not in url:
        return None, None
    header, data = url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or None
    try:
        return mime, base64.b64decode(data)
    except (binascii.Error, ValueError):
        return mime, None


def _image_from_block(block: Mapping[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    kind = block.get("type")
    if kind == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url", "") if isinstance(image_url, Mapping) else image_url
        mime, data = _decode_data_url(str(url or ""))
        info: dict[str, Any] = {"type": "image_url", "mime": mime}
        if isinstance(image_url, Mapping) and "detail" in image_url:
            info["detail"] = image_url["detail"]
        return info, data
    source = block.get("source")
    if kind == "image" and isinstance(source, Mapping):
        raw = source.get("data")
        try:
            data = base64.b64decode(str(raw)) if raw is not None else None
        except (binascii.Error, ValueError):
            data = None
        return {
            "type": "image",
            "source_type": source.get("type"),
            "mime": source.get("media_type"),
        }, data
    if kind == "image" and "base64" in block:
        try:
            data = base64.b64decode(str(block["base64"]))
        except (binascii.Error, ValueError):
            data = None
        info = {
            "type": "image",
            "source_type": "base64",
            "mime": block.get("mime_type"),
        }
        if "media_resolution" in block:
            info["media_resolution"] = block["media_resolution"]
        return info, data
    extra = {key: _jsonable(value) for key, value in block.items() if key != "type"}
    return {"type": str(kind), **extra}, None


_IMAGE_BLOCKS = frozenset({"image_url", "image"})


def _message_shapes(
    messages: list[BaseMessage],
) -> tuple[list[dict[str, Any]], int | None, str]:
    """Return the message digests, the last image's page marker, the system text."""
    shapes: list[dict[str, Any]] = []
    page: int | None = None
    system_parts: list[str] = []
    for message in messages:
        content = message.content
        entry: dict[str, Any] = {"role": message.type}
        blocks: list[dict[str, Any]] = []
        parts = [content] if isinstance(content, str) else list(content)
        for block in parts:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
                if message.type == "system":
                    system_parts.append(block)
                continue
            if block.get("type") == "text":
                text = str(block.get("text", ""))
                blocks.append({"type": "text", "text": text})
                if message.type == "system":
                    system_parts.append(text)
                continue
            info, data = _image_from_block(block)
            if data is not None and info["type"] in _IMAGE_BLOCKS:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    info["width"], info["height"] = image.size
                    page = decode_marker(image)
            blocks.append(info)
        entry["blocks"] = blocks
        shapes.append(entry)
    return shapes, page, "\n".join(system_parts)


# ============================================================================
# Provider-class constructions
# ============================================================================
def redact(value: Any) -> Any:
    """Return a JSON-ready copy with dummy keys named by their variable."""
    if isinstance(value, httpx.Timeout):
        return {"httpx.Timeout": value.as_dict()}
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    if isinstance(value, Mapping):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str) and value in _KEY_NAMES:
        return f"<env:{_KEY_NAMES[value]}>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def provider_of(constructor: str, base_url: Any) -> str:
    """Return the provider a constructor call addresses."""
    if constructor in _CLASS_PROVIDERS:
        return _CLASS_PROVIDERS[constructor]
    if base_url is None:
        return "openai"
    if str(base_url).rstrip("/") == OPENROUTER_BASE_URL:
        return "openrouter"
    return "custom"


@dataclass
class Construction:
    """One provider-class constructor call."""

    constructor: str
    kwargs: dict[str, Any]
    provider: str
    model: str

    def record(self) -> dict[str, Any]:
        """Return the golden entry: provider, constructor and redacted kwargs."""
        return {
            "provider": self.provider,
            "constructor": self.constructor,
            "kwargs": redact(self.kwargs),
        }


class SDKDefaultsChatModel(ScriptedChatModel):
    """Scripted model with its provider class's structured-output defaults.

    A call that omits ``method`` uses ``default_method``.
    """

    default_method: str = "function_calling"

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        method: str | None = None,
        **kwargs: Any,
    ) -> ScriptedStructured:
        chosen = method if method is not None else self.default_method
        return ScriptedStructured(self, StructuredCall(schema, chosen, include_raw))


def _describe_structured(request: FakeRequest) -> dict[str, Any] | None:
    call = request.structured
    if call is None:
        return None
    return {
        "method": call.method,
        "include_raw": call.include_raw,
        "schema": summarize_schema(call.schema),
    }


def _json_schema_call(schema: Any) -> dict[str, Any]:
    return {
        "method": "json_schema",
        "include_raw": True,
        "schema": summarize_schema(schema),
    }


def _enforcement(
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return invoke kwargs without a schema they enforce, and that schema's call.

    A schema sent as an invoke kwarg (OpenAI ``text.format``, Chat
    Completions ``response_format``, Gemini ``response_mime_type`` with
    ``response_schema``) reaches the provider as the same request field that
    ``with_structured_output(schema, method="json_schema")`` fills, so it is
    recorded as that call: ``text.format`` and ``response_format`` without
    their ``type`` wrapper, the Gemini schema as given.
    """
    rest = dict(kwargs)
    text = rest.get("text")
    if isinstance(text, Mapping):
        fmt = text.get("format")
        if isinstance(fmt, Mapping) and fmt.get("type") == "json_schema":
            others = {key: value for key, value in text.items() if key != "format"}
            if others:
                rest["text"] = others
            else:
                del rest["text"]
            schema = {key: value for key, value in fmt.items() if key != "type"}
            return rest, _json_schema_call(schema)
    fmt = rest.get("response_format")
    if isinstance(fmt, Mapping) and fmt.get("type") == "json_schema":
        del rest["response_format"]
        return rest, _json_schema_call(fmt.get("json_schema"))
    if (
        rest.get("response_mime_type") == "application/json"
        and "response_schema" in rest
    ):
        schema = rest.pop("response_schema")
        del rest["response_mime_type"]
        return rest, _json_schema_call(schema)
    return rest, None


# ============================================================================
# Fake LLM
# ============================================================================
class FakeLLM:
    """Provider-class stand-ins and the scripts, records and routing behind them.

    ``pages`` holds the page scripts (``default_page`` fills a missing one).
    ``calls`` holds one digest per request with the full text of every text
    block; ``constructions`` one entry per constructed client. With
    ``sdk_defaults`` set, clients constructed afterwards are
    ``SDKDefaultsChatModel`` instances with their class's default method
    from ``SDK_STRUCTURED_METHODS``.
    """

    def __init__(
        self,
        pages: Mapping[int, PageScript] | None = None,
        *,
        sdk_defaults: bool = False,
    ) -> None:
        self.pages: dict[int, PageScript] = dict(pages or {})
        self.sdk_defaults = sdk_defaults
        self.calls: list[dict[str, Any]] = []
        self.constructions: list[Construction] = []
        self._lock = threading.Lock()
        self._attempts: dict[int | None, int] = {}

    # -- configuration ---------------------------------------------------------
    def page(self, index: int) -> PageScript:
        """Return (creating from the default if needed) the script for a page."""
        with self._lock:
            script = self.pages.get(index)
            if script is None:
                script = default_page(index)
                self.pages[index] = script
            return script

    def script(self, index: int, **changes: Any) -> PageScript:
        """Replace fields of page *index*'s script and return it."""
        script = self.page(index)
        for name, value in changes.items():
            if not hasattr(script, name):
                raise AttributeError(f"PageScript has no field {name!r}")
            setattr(script, name, value)
        return script

    def reply(self, index: int, attempt: int, reply: Reply) -> None:
        """Script answer *attempt* (from 1) for page *index* (from 0)."""
        self.page(index).replies[attempt] = reply

    def reset_records(self) -> None:
        """Forget recorded calls, constructions and attempt counters."""
        with self._lock:
            self.calls.clear()
            self.constructions.clear()
            self._attempts.clear()

    # -- provider classes ------------------------------------------------------
    def constructor(self, name: str) -> Callable[..., ScriptedChatModel]:
        """Return a stand-in for the provider class *name*."""

        def construct(*args: Any, **kwargs: Any) -> ScriptedChatModel:
            if args:
                raise TypeError(f"{name} stand-in takes keyword arguments only")
            construction = Construction(
                constructor=name,
                kwargs=kwargs,
                provider=provider_of(name, kwargs.get("base_url")),
                model=str(kwargs.get("model", kwargs.get("model_name", ""))),
            )
            with self._lock:
                self.constructions.append(construction)
            if self.sdk_defaults:
                return SDKDefaultsChatModel(
                    responder=self.respond,
                    recorder=self.record,
                    tag=construction,
                    default_method=SDK_STRUCTURED_METHODS.get(name, "function_calling"),
                )
            return ScriptedChatModel(
                responder=self.respond, recorder=self.record, tag=construction
            )

        return construct

    @property
    def models(self) -> list[dict[str, Any]]:
        """Return the golden entry of every recorded construction."""
        with self._lock:
            return [construction.record() for construction in self.constructions]

    # -- answering -------------------------------------------------------------
    def record(self, request: FakeRequest) -> None:
        """Record the digest of one request; note its page and attempt.

        A schema enforced through invoke kwargs is recorded as its
        structured-output call (see ``_enforcement``).
        """
        construction: Construction = request.tag
        shapes, page, system = _message_shapes(request.messages)
        kwargs, enforced = _enforcement(request.kwargs)
        structured = _describe_structured(request) or enforced
        with self._lock:
            attempt = self._attempts.get(page, 0) + 1
            self._attempts[page] = attempt
            self.calls.append(
                {
                    "page": page,
                    "attempt": attempt,
                    "provider": construction.provider,
                    "model": construction.model,
                    "kwargs": describe_kwargs(kwargs),
                    "bound": describe_kwargs(request.bound),
                    "structured": structured,
                    "messages": shapes,
                }
            )
        json_answer = structured is not None or "json" in system.lower()
        request.notes.update(page=page, attempt=attempt, json_answer=json_answer)

    def respond(self, request: FakeRequest) -> AIMessage:
        """Return the scripted answer to a recorded request."""
        construction: Construction = request.tag
        page: int | None = request.notes["page"]
        attempt: int = request.notes["attempt"]
        json_answer: bool = request.notes["json_answer"]
        script = self.page(page) if page is not None else None
        reply: Reply | None = None
        if script is not None:
            reply = script.replies.get(attempt, script.always)
        metadata: dict[str, Any] = {
            "model_name": construction.model,
            "status": "completed",
        }
        additional: dict[str, Any] = {}
        form: Form = script.form if script is not None else "auto"
        if reply is not None:
            if reply.exception is not None:
                raise reply.exception
            if reply.response_metadata is not None:
                metadata = dict(reply.response_metadata)
            if reply.refusal is not None:
                additional["refusal"] = reply.refusal
            if reply.form is not None:
                form = reply.form
        if reply is not None and reply.text is not None:
            content = reply.text
        elif script is None:
            content = _unknown_answer(json_answer)
        else:
            content = script.content(form, json_answer)
        return AIMessage(
            content=content,
            usage_metadata=dict(USAGE),  # type: ignore[arg-type]
            response_metadata=metadata,
            additional_kwargs=additional,
        )

    def sorted_calls(self) -> list[dict[str, Any]]:
        """Return the call digests in a concurrency-independent order."""

        def key(call: dict[str, Any]) -> tuple[int, int]:
            page = call["page"] if isinstance(call["page"], int) else 10_000
            return page, call["attempt"]

        with self._lock:
            return sorted((copy.deepcopy(call) for call in self.calls), key=key)


def _unknown_answer(json_answer: bool) -> str:
    script = PageScript(transcription="Unrecognized page.")
    return script.content("auto", json_answer)
