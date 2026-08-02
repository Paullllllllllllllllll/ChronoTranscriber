"""Provider-agnostic entry point for remote audio transcription.

The audio counterpart to :class:`modules.llm.transcriber.LangChainTranscriber`:
the pipeline talks to one object regardless of which venue serves the request,
and the backend behind it is chosen once from
``audio_transcription.provider``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from modules.audio.audio_stream import AudioChunkPayload
from modules.audio.backends.base import AudioBackend
from modules.audio.backends.factory import get_audio_backend
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_AUDIO_PROVIDER = "openai"


class AudioTranscriber:
    """One remote speech-to-text venue, selected from configuration."""

    def __init__(
        self,
        audio_config: dict[str, Any],
        concurrency_config: dict[str, Any],
    ) -> None:
        """Select and construct the configured backend.

        Args:
            audio_config: Parsed ``audio_config.yaml``.
            concurrency_config: Parsed ``concurrency_config.yaml``.

        Raises:
            ValueError: When ``audio_transcription.provider`` is unknown or no
                API key can be resolved for it.
            CapabilityError: When the configured model cannot accept audio.
        """
        provider = str(
            (audio_config.get("audio_transcription", {}) or {}).get(
                "provider", DEFAULT_AUDIO_PROVIDER
            )
            or DEFAULT_AUDIO_PROVIDER
        )
        self._backend: AudioBackend = get_audio_backend(
            provider, audio_config, concurrency_config
        )
        logger.info(
            "AudioTranscriber initialized: provider=%s, model=%s",
            self.provider_name,
            self.model,
        )

    @property
    def provider_name(self) -> str:
        """Name of the venue serving requests (e.g. ``"openai"``)."""
        return self._backend.provider_name

    @property
    def model(self) -> str:
        """Model id the backend submits chunks to."""
        return self._backend.model

    async def transcribe_audio_chunk(
        self, payload: AudioChunkPayload
    ) -> dict[str, Any]:
        """Transcribe one chunk, returning the pipeline's legacy response dict.

        The pipeline's per-chunk handler calls exactly this. The shape and its
        failure semantics are documented in :mod:`modules.audio.backends.base`.
        """
        return await self._backend.transcribe_chunk(payload)

    def rekey(self) -> None:
        """Re-resolve the backend's API key env var; see ``AudioBackend``."""
        self._backend.rekey()

    async def close(self) -> None:
        """Release the backend's SDK clients."""
        await self._backend.close()


@asynccontextmanager
async def open_audio_transcriber(
    *,
    audio_config: dict[str, Any] | None = None,
    concurrency_config: dict[str, Any] | None = None,
) -> AsyncIterator[AudioTranscriber]:
    """Context manager for :class:`AudioTranscriber` with automatic cleanup.

    Mirrors :func:`modules.llm.transcriber.open_transcriber`. Either config may
    be omitted, in which case it is read from the ``ConfigService`` singleton.

    Yields:
        An ``AudioTranscriber`` whose backend is closed on exit.
    """
    if audio_config is None or concurrency_config is None:
        from modules.config.service import get_config_service

        service = get_config_service()
        if audio_config is None:
            audio_config = service.get_audio_config()
        if concurrency_config is None:
            concurrency_config = service.get_concurrency_config()

    transcriber = AudioTranscriber(audio_config, concurrency_config)
    try:
        yield transcriber
    finally:
        await transcriber.close()


__all__ = ["DEFAULT_AUDIO_PROVIDER", "AudioTranscriber", "open_audio_transcriber"]
