"""Capability-based parameter gating and fail-fast safety checks.

Currently exports the CapabilityError and the image-support assertion used
to bail out before the pipeline sends a request the selected model can't
handle.
"""

from __future__ import annotations

from modules.config.capabilities.detection import detect_capabilities


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
        raise CapabilityError(
            "The current pipeline sends image inputs, but the selected "
            f"model '{model_name}' does not support image inputs. Choose "
            "an image-capable model (e.g., gpt-5.4, gpt-5, o1, o3, gpt-4o, "
            "gpt-4.1) or set 'expects_image_inputs: false' in "
            "model_config.yaml to run a text-only flow."
        )


__all__ = ["CapabilityError", "ensure_image_support"]
