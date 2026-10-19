"""The payload handlers the LLM and audio-API unit pipelines share.

:func:`transcribe_payload` transcribes one rendered page or folder image
with the LLM transcriber; :func:`transcribe_audio_payload` one audio chunk
through the remote speech-to-text venue. Both return a
:data:`~chronotranscriber.methods.units.PayloadResult` and turn a failure
into the error placeholder instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any

from chronotranscriber.llm.response_parsing import extract_transcribed_text
from chronotranscriber.methods.units import PayloadResult

__all__ = ["transcribe_audio_payload", "transcribe_payload"]

logger = logging.getLogger(__name__)


def _error_text(image_name: str) -> str:
    return f"[transcription error: {image_name}]"


def _extracted(result: Any, image_name: str) -> str:
    try:
        return extract_transcribed_text(result, image_name)
    except Exception as exc:  # noqa: BLE001 - an unparsable answer becomes a placeholder
        logger.error(
            "Error extracting transcription for %s: %s. Marking as transcription "
            "error.",
            image_name,
            exc,
        )
        return _error_text(image_name)


async def transcribe_payload(payload: Any, transcriber: Any) -> PayloadResult:
    """Transcribe one in-memory page payload with the LLM transcriber.

    A page that never produced an image (``render_error``) is recorded as an
    error without a request.
    """
    image_name = payload.image_name
    if getattr(payload, "render_error", None) is not None:
        return (payload, image_name, _error_text(image_name), None, payload.index)
    try:
        result = await transcriber.transcribe_image_from_base64(
            payload.base64, payload.mime_type, label=image_name
        )
    except Exception as exc:
        logger.exception("Error transcribing %s: %s", image_name, exc)
        return (payload, image_name, _error_text(image_name), None, payload.index)
    logger.debug("LLM response for %s: %s", image_name, result)
    return (payload, image_name, _extracted(result, image_name), result, payload.index)


async def transcribe_audio_payload(payload: Any, transcriber: Any) -> PayloadResult:
    """Transcribe one audio chunk through the remote speech-to-text venue."""
    image_name = payload.image_name
    try:
        result = await transcriber.transcribe_audio_chunk(payload)
    except Exception as exc:
        logger.exception("Error transcribing %s: %s", image_name, exc)
        return (payload, image_name, _error_text(image_name), None, payload.index)
    logger.debug("Audio response for %s: %s", image_name, result)
    return (payload, image_name, _extracted(result, image_name), result, payload.index)
