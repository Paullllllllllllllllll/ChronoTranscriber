"""The audio backend contract and the legacy response shape it must emit.

A backend is the thin, provider-specific adapter between an
:class:`~modules.audio.audio_stream.AudioChunkPayload` and one speech-to-text
request. It owns request construction, usage accounting, and its SDK client's
lifecycle; chunk planning, resume, and JSONL assembly stay in the pipeline.

RESPONSE SHAPE. Every backend returns the same dict the image pipeline already
consumes -- the one built by
``modules.llm.transcriber.LangChainTranscriber._result_to_dict`` -- so
``modules.llm.response_parsing.extract_transcribed_text`` works unchanged::

    {
        "output_text": str,          # always present
        "usage": {                   # always present
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            # "cached_input_tokens" / "cache_creation_tokens" only when > 0
        },
        "metadata": {...},           # only when non-empty
        "parsed": {...},             # only when a structured payload exists
        "error": str,                # only on failure
    }

Two invariants beyond the shape:

* An empty or whitespace-only transcript is reported as the shared
  ``[No transcribable text]`` sentinel, never as ``""`` -- an empty string
  would be indistinguishable from a dropped page downstream.
* A failure returns a dict with ``error`` set and ``output_text`` empty rather
  than raising, matching how ``_result_to_dict`` surfaces a
  ``TranscriptionResult`` that carries an error.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from modules.audio.audio_stream import AudioChunkPayload
from modules.audio.constants import NO_TRANSCRIBABLE_TEXT


@runtime_checkable
class AudioBackend(Protocol):
    """One speech-to-text venue, adapted to the pipeline's chunk contract."""

    provider_name: str
    model: str

    async def transcribe_chunk(self, payload: AudioChunkPayload) -> dict[str, Any]:
        """Transcribe one chunk and return the legacy response dict.

        Never raises for an API-level failure: the error is reported through
        the ``error`` key so the pipeline can record a placeholder for this
        chunk and carry on with the rest of the recording.
        """
        ...

    async def close(self) -> None:
        """Release the SDK client and its connection pool. Never raises."""
        ...

    def rekey(self) -> None:
        """Rebuild the client against a freshly resolved API key env var.

        Not called during a run; provided so a caller that has changed the
        ``api_keys_config.yaml`` mapping can adopt it without a restart.
        Backends with nothing to re-resolve implement it as a no-op.
        """
        ...


def build_legacy_response(
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
    """Assemble the legacy response dict documented in this module's docstring.

    Mirrors ``LangChainTranscriber._result_to_dict`` key for key, including its
    conditional inclusion of ``metadata``, ``parsed``, and ``error``. Empty
    transcripts collapse to the no-text sentinel unless the call errored, where
    an empty ``output_text`` is the meaningful signal.
    """
    text = (output_text or "").strip()
    if not text and error is None:
        text = NO_TRANSCRIBABLE_TEXT

    response: dict[str, Any] = {
        "output_text": "" if error is not None else text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }

    combined_metadata: dict[str, Any] = {"model": model, "provider": provider}
    if metadata:
        combined_metadata.update(metadata)
    response["metadata"] = combined_metadata

    if parsed:
        response["parsed"] = parsed
    if error is not None:
        response["error"] = error
    return response


def error_response(
    message: str, *, provider: str, model: str, **metadata: Any
) -> dict[str, Any]:
    """Shorthand for a failed chunk: empty text plus an ``error`` key."""
    return build_legacy_response(
        output_text="",
        provider=provider,
        model=model,
        metadata=metadata or None,
        error=message,
    )


__all__ = ["AudioBackend", "build_legacy_response", "error_response"]
