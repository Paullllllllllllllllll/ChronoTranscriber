"""The audio-API method: recordings sent to a remote speech-to-text model.

OpenAI's transcription endpoint or a Gemini model, as the run's provider
says. Recordings above the provider's request limit are cut into chunks
first (:mod:`chronotranscriber.methods.recording`).

Each backend turns one chunk into the response dict the page pipeline
parses::

    {
        "output_text": str,          # always present
        "usage": {"input_tokens": int, "output_tokens": int,
                  "total_tokens": int, "cached_tokens": int,
                  "reasoning_tokens": int},
        "metadata": {...},           # model, provider, chunk_index, ...
        "parsed": {...},             # only when a structured payload exists
        "error": str,                # only on failure
    }

An empty transcript is reported as the shared ``[No transcribable text]``
placeholder, and a failure returns the dict with ``error`` set instead of
raising, so the item records a placeholder for the chunk and goes on.

Every OpenAI request runs inside :func:`chronotranscriber.ext.attempt`,
retried by :func:`~chronotranscriber.common.retry.call_with_retry` with the
retry policy of the settings: the attempt builds the client for its key
variable, ``commit`` records the usage of a response and ``on_error``
decides what follows a failure. The usage is added to the run's totals
under the ``audio`` role. The client's httpx client carries the capture
hooks, and each attempt is recorded as
:class:`~chronotranscriber.llm.caller.HttpAttempt` describes.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from chronotranscriber import ext
from chronotranscriber.common.capture import async_capture_hooks, redact_payloads
from chronotranscriber.common.http_timeouts import build_httpx_timeout
from chronotranscriber.common.responses import unwrap_raw_response
from chronotranscriber.common.retry import RetryEvent, RetryHooks, call_with_retry
from chronotranscriber.common.structured import StructuredRequest, StructuredResult
from chronotranscriber.common.usage import Usage
from chronotranscriber.events import Waiting
from chronotranscriber.llm.caller import CallEnv, HttpAttempt
from chronotranscriber.llm.capabilities import (
    detect_capabilities,
    ensure_audio_support,
)
from chronotranscriber.llm.client import get_api_key
from chronotranscriber.llm.retrying import AttemptDecisions
from chronotranscriber.llm.transcription import answer_object, usage_record
from chronotranscriber.methods import Item, Job
from chronotranscriber.methods.pages import transcribe_audio_payload
from chronotranscriber.methods.recording import transcribe_recording
from chronotranscriber.methods.units import StreamHandler
from chronotranscriber.settings import AudioApi
from chronotranscriber.sources.audio.constants import (
    GEMINI_ALLOWED_EXTENSIONS,
    GEMINI_INLINE_LIMIT_BYTES,
    NO_TRANSCRIBABLE_TEXT,
    OPENAI_ALLOWED_EXTENSIONS,
)
from chronotranscriber.sources.audio.stream import AudioChunkPayload
from chronotranscriber.spec import (
    DEFAULT_SERVICE_TIER,
    DEFAULT_TOP_P,
    DEFAULT_VERBOSITY,
    ModelSpec,
    RunSpec,
)

__all__ = [
    "GOOGLE_AUDIO_PROMPT",
    "ROLE",
    "AudioApiTranscriber",
    "AudioBackend",
    "GoogleAudioBackend",
    "OpenAIAudioBackend",
    "build_response",
    "error_response",
    "open_audio_backend",
]

logger = logging.getLogger(__name__)

ROLE = "audio"
"""The usage role of audio calls."""

GOOGLE_AUDIO_PROMPT = (
    "Transcribe the speech in this audio recording verbatim into plain text.\n"
    "Do not add timestamps, speaker labels, headings, or commentary.\n"
    "Output only the transcript."
)
GOOGLE_AUDIO_INSTRUCTION = "Please transcribe the speech in this audio recording."
GOOGLE_AUDIO_MAX_OUTPUT_TOKENS = 32768
GOOGLE_AUDIO_TEMPERATURE = 0.0

# Models that take ``languages[]`` and ``keywords[]`` lists, sent in
# ``extra_body``; every other model takes the scalar ``language``.
OPENAI_PLURAL_PARAM_MODELS: tuple[str, ...] = ("gpt-transcribe",)

# whisper-1 silently truncates its prompt at 224 tokens; the backend trims
# first so the log shows the discarded tail. Four characters per token is
# a rough estimate.
WHISPER1_PROMPT_TOKEN_CAP = 224
_CHARS_PER_TOKEN = 4

_FORMAT_HINT = "Set audio_chunking.chunk_format to 'mp3' or 'wav16' to convert."


class AudioBackend(Protocol):
    """One remote speech-to-text provider."""

    provider_name: str
    model: str

    async def transcribe_audio_chunk(
        self, payload: AudioChunkPayload
    ) -> dict[str, Any]:
        """Transcribe one chunk; a failure is reported in the ``error`` key."""
        ...

    async def close(self) -> None:
        """Release the clients. Never raises."""
        ...


def build_response(  # noqa: PLR0913 - keyword-only fields, used by tests
    *,
    output_text: str,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    metadata: dict[str, Any] | None = None,
    parsed: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Return the response dict of one chunk (see the module docstring).

    An empty transcript becomes the no-text placeholder unless the call
    failed, where an empty ``output_text`` is the signal.
    """
    text = (output_text or "").strip()
    if not text and error is None:
        text = NO_TRANSCRIBABLE_TEXT
    response: dict[str, Any] = {
        "output_text": "" if error is not None else text,
        "usage": Usage(input_tokens, output_tokens, total_tokens).as_dict(),
    }
    combined: dict[str, Any] = {"model": model, "provider": provider}
    if metadata:
        combined.update(metadata)
    response["metadata"] = combined
    if parsed:
        response["parsed"] = parsed
    if error is not None:
        response["error"] = error
    return response


def error_response(
    message: str, *, provider: str, model: str, **metadata: Any
) -> dict[str, Any]:
    """Return the response dict of a failed chunk."""
    return build_response(
        output_text="",
        provider=provider,
        model=model,
        metadata=metadata or None,
        error=message,
    )


def _token_usage(usage: Any) -> Usage:
    """Return the tokens of a transcription response's usage.

    Only token-billed usage (``type == "tokens"``) has counts; duration
    billing and absent usage count zero.
    """
    if usage is None or getattr(usage, "type", None) != "tokens":
        logger.debug(
            "No token usage on the transcription response (type=%s).",
            getattr(usage, "type", None),
        )
        return Usage()
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return Usage(input_tokens, output_tokens, total_tokens)


class OpenAIAudioBackend:
    """Transcribe chunks through the OpenAI speech-to-text endpoint.

    The endpoint is a multipart upload called through the raw ``AsyncOpenAI``
    SDK. Clients are built with ``max_retries=0``, one per key variable, by
    the attempt of each request.
    """

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        *,
        env: CallEnv,
        language: str | None = None,
        extras: AudioApi | None = None,
    ) -> None:
        """Check the model and build the client of the default identity.

        Raises:
            CapabilityError: The model cannot accept audio.
            OSError: The key variable is unset.
        """
        self.model = model.strip()
        ensure_audio_support(self.model, detect_capabilities(self.model, "openai"))
        self.env = env
        self.language = (language or "").strip()
        self.extras = extras or AudioApi()
        self._policy = env.settings.retry.policy()
        self._clients: dict[str, Any] = {}
        self._attempt().build()
        logger.info("OpenAI audio backend ready: model=%s", self.model)

    def _attempt(self) -> ext.Attempt[Any]:
        return ext.attempt(
            ROLE,
            self.provider_name,
            None,
            settings=self.env.settings,
            build_client=self.client_for,
            model=self.model,
            kind="audio",
        )

    def client_for(self, identity: ext.Identity) -> Any:
        """Return the client of an attempt identity, building it if new.

        The read phase takes the request timeout; connect, write and pool
        keep their own shorter timeouts. The extension's client options of
        the identity may set the base URL and the key.
        """
        client = self._clients.get(identity.api_key_env)
        if client is None:
            options = ext.client_options(identity)
            timeouts = self.env.settings.timeouts
            timeout = build_httpx_timeout(
                timeouts.request,
                connect=timeouts.connect,
                write=timeouts.write,
                pool=timeouts.pool,
            )
            endpoint: dict[str, Any] = (
                {"base_url": options.base_url} if options.base_url else {}
            )
            client = AsyncOpenAI(
                api_key=options.api_key
                or get_api_key(self.provider_name, identity.api_key_env),
                **endpoint,
                max_retries=0,
                timeout=timeout,
                http_client=DefaultAsyncHttpxClient(
                    timeout=timeout, event_hooks=async_capture_hooks()
                ),
            )
            self._clients[identity.api_key_env] = client
        return client

    def resolve_prompt(self) -> str:
        """Return the settings prompt, trimmed to whisper-1's token cap."""
        prompt = self.extras.prompt.strip()
        if not prompt or self.model != "whisper-1":
            return prompt
        max_chars = WHISPER1_PROMPT_TOKEN_CAP * _CHARS_PER_TOKEN
        if len(prompt) <= max_chars:
            return prompt
        logger.warning(
            "whisper-1 caps the prompt at %d tokens; truncating from %d to %d "
            "characters.",
            WHISPER1_PROMPT_TOKEN_CAP,
            len(prompt),
            max_chars,
        )
        return prompt[:max_chars]

    def request_kwargs(
        self, filename: str, data: bytes, mime_type: str
    ) -> dict[str, Any]:
        """Return the ``transcriptions.create`` keyword arguments."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "file": (filename, data, mime_type),
        }
        prompt = self.resolve_prompt()
        if prompt:
            kwargs["prompt"] = prompt
        if self.model in OPENAI_PLURAL_PARAM_MODELS:
            extra_body: dict[str, Any] = {}
            languages = [str(x).strip() for x in self.extras.languages if x]
            if not languages and self.language:
                languages = [self.language]
            keywords = [str(x).strip() for x in self.extras.keywords if x]
            if languages:
                extra_body["languages"] = languages
            if keywords:
                extra_body["keywords"] = keywords
            if extra_body:
                kwargs["extra_body"] = extra_body
        elif self.language:
            kwargs["language"] = self.language
        return kwargs

    def _record(self, usage: Usage) -> None:
        self.env.record(ROLE, usage)

    async def _request(self, kwargs: dict[str, Any], label: str) -> Any:
        """Send one request; transient errors are retried, the last propagates.

        Each attempt's record goes to the recorder in the context, if any,
        before the next attempt is sent.
        """
        decisions = AttemptDecisions()
        numbers = itertools.count(1)

        async def one_attempt() -> Any:
            attempt = self._attempt()
            async with contextlib.AsyncExitStack() as stack:
                call = await decisions.enter(
                    stack, attempt, on_failure=lambda: self._record(attempt.usage)
                )
                http = HttpAttempt(
                    provider=self.provider_name,
                    model=self.model,
                    role=ROLE,
                    attempt=next(numbers),
                    label=label,
                    request=lambda: redact_payloads(kwargs),
                )
                try:
                    with http.capturing():
                        response = await call.client.audio.transcriptions.create(
                            **kwargs
                        )
                except Exception as exc:
                    decisions.failed(call, exc)
                    self._record(call.usage)
                    await http.store(usage=call.usage, error=exc)
                    raise
                usage = _token_usage(getattr(response, "usage", None))
                call.commit(usage)
                self._record(call.usage)
                await http.store(response=response, usage=usage)
                return response

        hooks = RetryHooks(
            on_error=decisions.decide,
            on_retry=self._report_retry,
            gate=self.env.limiters.get(self.provider_name),
        )
        return await call_with_retry(
            one_attempt, policy=self._policy, hooks=hooks, label=label
        )

    def _report_retry(self, event: RetryEvent) -> None:
        self.env.events.waiting(
            Waiting(
                reason=(
                    f"{event.label or ROLE}: {event.kind.value} on attempt "
                    f"{event.attempt}, {event.decision.value}"
                ),
                seconds=event.delay,
            )
        )

    async def transcribe_audio_chunk(
        self, payload: AudioChunkPayload
    ) -> dict[str, Any]:
        """Transcribe one chunk; see :class:`AudioBackend`."""
        suffix = payload.path.suffix.lower()
        if suffix not in OPENAI_ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(OPENAI_ALLOWED_EXTENSIONS))
            return error_response(
                f"OpenAI transcription does not accept '{suffix}' files "
                f"(allowed: {allowed}). {_FORMAT_HINT}",
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )
        try:
            data = await payload.read_bytes()
            kwargs = self.request_kwargs(payload.image_name, data, payload.mime_type)
            response = await self._request(kwargs, payload.image_name)
        except Exception as exc:  # noqa: BLE001 - a failed chunk returns an error dict
            logger.error(
                "OpenAI audio transcription failed for %s: %s", payload.image_name, exc
            )
            return error_response(
                str(exc),
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )
        usage = _token_usage(getattr(response, "usage", None))
        return build_response(
            output_text=str(getattr(response, "text", "") or ""),
            provider=self.provider_name,
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            metadata={"chunk_index": payload.index},
        )

    async def close(self) -> None:
        """Close every client."""
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001 - close never raises
                logger.debug("Error closing OpenAI audio client: %s", exc)


class GoogleAudioBackend:
    """Transcribe chunks with an audio-capable Gemini chat model.

    The recording rides as an inline audio part on a plain-text call through
    :class:`~chronotranscriber.llm.caller.ModelCaller`, which runs every
    attempt in an extension attempt, retries, rate-limits and records the
    usage under the ``audio`` role. The chat model takes no thinking level
    and no request timeout.
    """

    provider_name = "google"

    def __init__(
        self, model: str, *, env: CallEnv, language: str | None = None
    ) -> None:
        """Check the model and build its chat model.

        Raises:
            CapabilityError: The model cannot accept audio.
            OSError: The key variable is unset.
        """
        self.model = model.strip()
        capabilities = detect_capabilities(self.model, "google")
        ensure_audio_support(self.model, capabilities)
        self.env = env
        self.language = (language or "").strip()
        self.max_output_tokens = min(
            GOOGLE_AUDIO_MAX_OUTPUT_TOKENS, capabilities.max_output_tokens
        )
        spec = ModelSpec(
            provider=self.provider_name,
            model=self.model,
            endpoint=None,
            reasoning_effort="none",
            verbosity=DEFAULT_VERBOSITY,
            max_output_tokens=GOOGLE_AUDIO_MAX_OUTPUT_TOKENS,
            temperature=GOOGLE_AUDIO_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            service_tier=DEFAULT_SERVICE_TIER,
            response_format="text",
        )
        self._caller = env.caller(ROLE, spec, overrides={"timeout": None})
        logger.info("Google audio backend ready: model=%s", self.model)

    def system_prompt(self) -> str:
        """Return the prompt sent with the audio part, with the language."""
        if self.language:
            return f"{GOOGLE_AUDIO_PROMPT}\nThe recording is in {self.language}."
        return GOOGLE_AUDIO_PROMPT

    def messages(self, audio_base64: str, mime_type: str) -> list[Any]:
        """Return the system and user messages for one chunk.

        langchain-google-genai turns the base64 ``file`` block into an
        inline-data part.
        """
        content: list[str | dict[str, Any]] = [
            {"type": "text", "text": GOOGLE_AUDIO_INSTRUCTION},
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": mime_type,
                "data": audio_base64,
            },
        ]
        return [
            SystemMessage(content=self.system_prompt()),
            HumanMessage(content=content),
        ]

    def _response(self, result: StructuredResult, chunk_index: int) -> dict[str, Any]:
        if result.stopped:
            failed = error_response(
                result.outcome.message,
                provider=self.provider_name,
                model=self.model,
                chunk_index=chunk_index,
            )
            failed["error_type"] = result.outcome.kind
            return failed
        message = unwrap_raw_response(result.response)
        metadata: dict[str, Any] = {"chunk_index": chunk_index}
        raw_metadata = getattr(message, "response_metadata", None)
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        usage = usage_record(message, self.provider_name)
        response = build_response(
            output_text=result.text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            metadata=metadata,
            parsed=answer_object(result.text),
        )
        response["usage"] = usage
        return response

    async def transcribe_audio_chunk(
        self, payload: AudioChunkPayload
    ) -> dict[str, Any]:
        """Transcribe one chunk; see :class:`AudioBackend`."""
        suffix = payload.path.suffix.lower()
        if suffix not in GEMINI_ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(GEMINI_ALLOWED_EXTENSIONS))
            return error_response(
                f"Gemini does not accept '{suffix}' audio (allowed: {allowed}). "
                f"{_FORMAT_HINT}",
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )
        if payload.byte_size > GEMINI_INLINE_LIMIT_BYTES:
            return error_response(
                f"Chunk is {payload.byte_size / 1_048_576:.1f} MB, above "
                f"Gemini's {GEMINI_INLINE_LIMIT_BYTES // 1_048_576} MB inline "
                f"limit. Lower --audio-chunk-seconds or set "
                f"audio_chunking.max_request_bytes.",
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )
        try:
            audio_base64 = await payload.read_base64()
            request = StructuredRequest(
                provider=self.provider_name,
                response_format="text",
                invoke_kwargs={"max_output_tokens": self.max_output_tokens},
            )
            result = await self._caller.call(
                self.messages(audio_base64, payload.mime_type),
                request,
                label=payload.image_name,
            )
        except Exception as exc:  # noqa: BLE001 - a failed chunk returns an error dict
            logger.error(
                "Gemini audio transcription failed for %s: %s", payload.image_name, exc
            )
            return error_response(
                str(exc),
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )
        return self._response(result, payload.index)

    async def close(self) -> None:
        """Close the chat model's HTTP clients. Never raises."""
        await self._caller.aclose()


def open_audio_backend(spec: RunSpec, env: CallEnv) -> AudioBackend:
    """Build the backend of the run's provider.

    Raises:
        ValueError: The provider has no audio API.
        CapabilityError: The model cannot accept audio.
        OSError: The key variable is unset.
    """
    model = spec.model
    language = spec.audio.language
    if model.provider == "openai":
        return OpenAIAudioBackend(
            model.model, env=env, language=language, extras=env.settings.audio_api
        )
    if model.provider == "google":
        return GoogleAudioBackend(model.model, env=env, language=language)
    raise ValueError(
        f"Unknown audio provider '{model.provider}'. Supported: openai, google."
    )


class AudioApiTranscriber:
    """Transcribe recordings through the provider's audio API."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self._client: AudioBackend | None = None

    async def open(self) -> None:
        """Build the audio backend of the run's provider."""
        self._client = open_audio_backend(self.job.spec, self.job.env)

    async def close(self) -> None:
        """Close the audio backend."""
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def transcribe(self, item: Item) -> list[Path]:
        """Transcribe one recording into its chunk units; return [].

        Raises:
            ValueError: The audio backend was not opened.
        """
        client = self._client
        if client is None:
            raise ValueError(
                "No audio transcriber was opened for the audio-api method."
            )
        return await transcribe_recording(
            self.job,
            item,
            client,
            self._handler,
            provider=self.job.spec.model.provider,
            model=client.model,
        )

    @staticmethod
    def _handler() -> StreamHandler:
        return transcribe_audio_payload
