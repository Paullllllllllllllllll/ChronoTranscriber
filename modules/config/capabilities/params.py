"""Capability-based parameter gating and fail-fast safety checks.

Currently exports the CapabilityError and the image/audio-support assertions
used to bail out before the pipeline sends a request the selected model can't
handle.
"""

from __future__ import annotations

import logging

from modules.config.capabilities.detection import detect_capabilities

# Use the stdlib logger directly: modules.infra.logger imports
# modules.config.config_loader, which imports this package, so importing the
# project logger here would create a circular import.
logger = logging.getLogger(__name__)


class CapabilityError(ValueError):
    """Raised when a selected model is incompatible with the pipeline."""


def ensure_image_support(model_name: str, images_required: bool) -> None:
    """Fail fast if the pipeline intends to send images but the model can't
    accept them.

    Parameters
    ----------
    model_name : str
        Selected model id/alias from configuration.
    images_required : bool
        True if the current pipeline path will submit image inputs
        (our OCR path).
    """
    caps = detect_capabilities(model_name)
    if images_required and not caps.supports_image_input:
        if caps.supports_audio_input:
            # Audio-only model configured for an audio run: config load must
            # not fail here, the audio backend gates the run instead.
            logger.warning(
                "Model '%s' accepts audio but not image inputs; image "
                "transcription will not work with this model.",
                model_name,
            )
            return
        raise CapabilityError(
            "The current pipeline sends image inputs, but the selected "
            f"model '{model_name}' does not support image inputs. Choose "
            "an image-capable model (e.g., gpt-5.4, gpt-5, o1, o3, gpt-4o, "
            "gpt-4.1) or set 'expects_image_inputs: false' in "
            "model_config.yaml to run a text-only flow."
        )


def ensure_audio_support(model_name: str) -> None:
    """Fail fast if the selected model cannot accept audio inputs.

    Called at run time by the audio backend factory (never at config load),
    so a model that is fine for the image pipeline does not break startup.

    Parameters
    ----------
    model_name : str
        Selected model id/alias from the audio configuration.
    """
    caps = detect_capabilities(model_name)
    if not caps.supports_audio_input:
        raise CapabilityError(
            "The audio pipeline sends audio inputs, but the selected model "
            f"'{model_name}' does not support them. Choose an audio-capable "
            "model (e.g., gpt-transcribe, gpt-4o-transcribe, whisper-1, or a "
            "gemini-* model) in audio_config.yaml."
        )


__all__ = ["CapabilityError", "ensure_audio_support", "ensure_image_support"]
