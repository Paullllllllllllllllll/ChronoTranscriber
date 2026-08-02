"""Google Gemini audio backend (inline audio part on a chat request).

Unlike OpenAI, Gemini has no dedicated transcription endpoint: the recording
rides along as an inline audio part on an ordinary generate-content call. That
is already implemented in
:meth:`modules.llm.providers.google_provider.GoogleProvider.transcribe_audio_from_base64`,
so this backend is a thin adapter -- it owns the prompt assembly, the inline
size and container checks, and the conversion of the returned
``TranscriptionResult`` into the pipeline's legacy response dict. Retry,
rate limiting, and token accounting all happen inside the provider.
"""

from __future__ import annotations

from typing import Any

from modules.audio.audio_stream import AudioChunkPayload
from modules.audio.backends.base import build_legacy_response, error_response
from modules.audio.constants import (
    GEMINI_ALLOWED_EXTENSIONS,
    GEMINI_INLINE_LIMIT_BYTES,
)
from modules.config.capabilities import ensure_audio_support
from modules.infra.logger import setup_logger
from modules.llm.providers.base import TranscriptionResult

logger = setup_logger(__name__)

DEFAULT_GOOGLE_AUDIO_MODEL = "gemini-3.6-flash"

DEFAULT_GOOGLE_PROMPT = (
    "Transcribe the speech in this audio recording verbatim into plain text.\n"
    "Do not add timestamps, speaker labels, headings, or commentary.\n"
    "Output only the transcript."
)


class GoogleAudioBackend:
    """Transcribes chunks through an audio-capable Gemini chat model."""

    def __init__(
        self,
        audio_config: dict[str, Any],
        concurrency_config: dict[str, Any],
    ) -> None:
        """Resolve the model and build the underlying ``GoogleProvider``.

        Args:
            audio_config: Parsed ``audio_config.yaml``.
            concurrency_config: Parsed ``concurrency_config.yaml``. Unused
                here: the provider reads the timeout and retry knobs itself.

        Raises:
            CapabilityError: When the configured model cannot accept audio.
            ValueError: When no Google API key can be resolved.
        """
        del concurrency_config  # provider-managed; kept for a uniform ctor

        settings = (audio_config.get("audio_transcription", {}) or {}).get(
            "google", {}
        ) or {}
        self.provider_name = "google"
        self.model = str(settings.get("model") or DEFAULT_GOOGLE_AUDIO_MODEL).strip()
        ensure_audio_support(self.model)

        self._settings = settings
        raw_max_tokens = settings.get("max_output_tokens")
        self._max_output_tokens = (
            int(raw_max_tokens) if raw_max_tokens not in (None, "", 0) else None
        )
        raw_temperature = settings.get("temperature")
        temperature = float(raw_temperature) if raw_temperature is not None else None

        from modules.llm.providers.factory import get_provider

        self._provider = get_provider(
            provider="google",
            model=self.model,
            temperature=temperature,
            max_tokens=self._max_output_tokens,
        )

        logger.info(
            "Google audio backend initialized: model=%s, max_output_tokens=%s",
            self.model,
            self._max_output_tokens,
        )

    def _system_prompt(self) -> str:
        """Prompt sent alongside the audio part, plus the language hint."""
        prompt = str(self._settings.get("prompt") or "").strip()
        if not prompt:
            prompt = DEFAULT_GOOGLE_PROMPT
        hint = str(self._settings.get("language_hint") or "").strip()
        if hint:
            prompt = f"{prompt}\nThe recording is in {hint}."
        return prompt

    def _to_legacy_dict(
        self, result: TranscriptionResult, chunk_index: int
    ) -> dict[str, Any]:
        """Convert a ``TranscriptionResult`` to the legacy response dict.

        Mirrors ``LangChainTranscriber._result_to_dict`` field for field
        (transcriber.py is left untouched): the token triple, the conditional
        cache keys, the raw response as ``metadata``, and the parsed payload
        and error only when present.
        """
        metadata: dict[str, Any] = {"chunk_index": chunk_index}
        if result.raw_response:
            metadata.update(result.raw_response)

        response = build_legacy_response(
            output_text=result.content,
            provider=self.provider_name,
            model=self.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            metadata=metadata,
            parsed=result.parsed_output,
            error=result.error,
        )
        if result.cached_input_tokens > 0:
            response["usage"]["cached_input_tokens"] = result.cached_input_tokens
        if result.cache_creation_tokens > 0:
            response["usage"]["cache_creation_tokens"] = result.cache_creation_tokens
        return response

    async def transcribe_chunk(self, payload: AudioChunkPayload) -> dict[str, Any]:
        """Transcribe one chunk; see :mod:`modules.audio.backends.base`."""
        suffix = payload.path.suffix.lower()
        if suffix not in GEMINI_ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(GEMINI_ALLOWED_EXTENSIONS))
            return error_response(
                f"Gemini does not accept '{suffix}' audio (allowed: {allowed}). "
                f"Set chunking.chunk_format to 'mp3' or 'wav16' to convert on "
                f"the fly.",
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )

        if 0 < GEMINI_INLINE_LIMIT_BYTES < payload.byte_size:
            return error_response(
                f"Chunk is {payload.byte_size / 1_048_576:.1f} MB, above "
                f"Gemini's ~{GEMINI_INLINE_LIMIT_BYTES // 1_048_576} MB inline "
                f"limit. Lower chunking.target_seconds or set "
                f"chunking.max_request_bytes.",
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )

        try:
            audio_base64 = await payload.read_base64()
            result = await self._provider.transcribe_audio_from_base64(
                audio_base64,
                payload.mime_type,
                system_prompt=self._system_prompt(),
                max_output_tokens=self._max_output_tokens,
            )
        except Exception as exc:
            logger.error(
                "Gemini audio transcription failed for %s: %s",
                payload.image_name,
                exc,
            )
            return error_response(
                str(exc),
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )

        return self._to_legacy_dict(result, payload.index)

    def rekey(self) -> None:
        """No-op: the Google provider owns its own client lifecycle.

        The underlying :class:`GoogleProvider` resolves its key at construction
        and is rebuilt with the backend, so there is nothing to re-key here.
        Kept to satisfy the backend protocol.
        """
        return None

    async def close(self) -> None:
        """Dispose the provider's HTTP clients. Never raises."""
        try:
            await self._provider.close()
        except Exception as exc:
            logger.debug("Error closing Google audio provider: %s", exc)


__all__ = ["DEFAULT_GOOGLE_AUDIO_MODEL", "GoogleAudioBackend"]
