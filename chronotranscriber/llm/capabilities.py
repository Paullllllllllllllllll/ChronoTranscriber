"""ChronoTranscriber's view of the shared model capability table.

The family rules and their sources are in
:mod:`chronotranscriber.common.capability_table`. Ids are matched after a
``provider:`` or ``models/`` prefix is removed, and ``vendor/model`` ids
resolve through the OpenRouter rules. Unmatched ids get the conservative
profile: no image input and no structured output.

This tool extends the table with its audio flags in ``Capabilities.extras``:
``supports_audio_input`` marks the families that take audio parts
(``AUDIO_FAMILIES``), and ``audio_transcription_only`` the speech-to-text
models served only by the OpenAI transcription endpoint.
``image_original_max_side_px`` is the default long-side limit in pixels of
OpenAI pages sent at ``original`` detail.
"""

from __future__ import annotations

from chronotranscriber.common import capabilities as engine
from chronotranscriber.common.capabilities import Capabilities, CapabilityError
from chronotranscriber.common.capability_table import (
    AUDIO_FAMILIES,
    ORIGINAL_MAX_SIDE,
    SPEECH_TO_TEXT_FAMILIES,
    TableExtension,
    build_table,
)

__all__ = [
    "AUDIO_INPUT",
    "ORIGINAL_MAX_SIDE",
    "TABLE",
    "TRANSCRIPTION_ONLY",
    "Capabilities",
    "CapabilityError",
    "custom_capabilities",
    "detect_capabilities",
    "detect_provider",
    "ensure_audio_support",
    "ensure_image_support",
    "supports_audio",
]

AUDIO_INPUT = "supports_audio_input"
TRANSCRIPTION_ONLY = "audio_transcription_only"

TABLE = build_table(
    TableExtension(
        family_extras={
            family: {
                AUDIO_INPUT: True,
                **(
                    {TRANSCRIPTION_ONLY: True}
                    if family in SPEECH_TO_TEXT_FAMILIES
                    else {}
                ),
            }
            for family in AUDIO_FAMILIES
        }
    )
)


def detect_provider(model: str) -> str:
    """Return the provider a model id names, or ``unknown``."""
    return engine.detect_provider(model, TABLE)


def detect_capabilities(model: str, provider: str | None = None) -> Capabilities:
    """Return the capabilities of *model*; *provider* overrides detection."""
    return engine.resolve_capabilities(model, TABLE, provider)


def custom_capabilities(
    model: str, *, supports_vision: bool, supports_schema: bool
) -> Capabilities:
    """Return the capabilities a named custom endpoint declares for *model*."""
    return Capabilities(
        provider_name=engine.CUSTOM,
        model_name=model,
        family=engine.CUSTOM,
        supports_vision=supports_vision,
        supports_structured_output=supports_schema,
        supports_json_mode=supports_schema,
    )


def supports_audio(capabilities: Capabilities) -> bool:
    """Whether the model accepts audio input."""
    return bool(capabilities.extra(AUDIO_INPUT, False))


def ensure_image_support(model_name: str, capabilities: Capabilities) -> None:
    """Raise ``CapabilityError`` unless the model accepts image input."""
    if not capabilities.supports_vision:
        raise CapabilityError(
            f"Model '{model_name}' does not support image inputs. Choose an "
            "image-capable model (e.g., gpt-5.6-luna, claude-sonnet-5, "
            "gemini-3.5-flash) or a local method."
        )


def ensure_audio_support(model_name: str, capabilities: Capabilities) -> None:
    """Raise ``CapabilityError`` unless the model accepts audio input."""
    if not supports_audio(capabilities):
        raise CapabilityError(
            f"Model '{model_name}' does not support audio inputs. Choose an "
            "audio-capable model (e.g., gpt-transcribe, gpt-4o-transcribe, "
            "whisper-1 or a gemini model) or the whisper method."
        )
