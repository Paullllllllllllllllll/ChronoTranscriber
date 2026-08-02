"""Backend selection for the remote audio venues.

Backend classes are imported lazily so choosing one provider never constructs
the other's SDK client or drags in its dependencies.
"""

from __future__ import annotations

from typing import Any

from modules.audio.backends.base import AudioBackend

_VALID_PROVIDERS = ("openai", "google")


def get_audio_backend(
    provider: str,
    audio_config: dict[str, Any],
    concurrency_config: dict[str, Any],
) -> AudioBackend:
    """Build the audio backend for *provider*.

    Args:
        provider: ``"openai"`` or ``"google"`` (case-insensitive), from
            ``audio_transcription.provider``.
        audio_config: Parsed ``audio_config.yaml``.
        concurrency_config: Parsed ``concurrency_config.yaml``.

    Returns:
        A ready-to-use backend.

    Raises:
        ValueError: When *provider* is not a known audio venue.
        CapabilityError: When the configured model cannot accept audio.
    """
    name = (provider or "").strip().lower()

    if name == "openai":
        from modules.audio.backends.openai_audio import OpenAIAudioBackend

        return OpenAIAudioBackend(audio_config, concurrency_config)

    if name == "google":
        from modules.audio.backends.google_audio import GoogleAudioBackend

        return GoogleAudioBackend(audio_config, concurrency_config)

    raise ValueError(
        f"Unknown audio provider '{provider}'. "
        f"Supported: {', '.join(_VALID_PROVIDERS)}."
    )


__all__ = ["get_audio_backend"]
