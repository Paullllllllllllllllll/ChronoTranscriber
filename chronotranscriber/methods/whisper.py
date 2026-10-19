"""The Whisper method: audio recordings transcribed locally with faster-whisper.

``faster-whisper`` ships as an optional extra, so it is imported only when
the first chunk is transcribed; a missing installation fails the item with
:class:`WhisperMissingError`. The model is loaded once per run and shared by
its chunks.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronotranscriber.methods import Item, Job
from chronotranscriber.methods.recording import transcribe_recording
from chronotranscriber.methods.units import PayloadResult, StreamHandler
from chronotranscriber.settings import Settings
from chronotranscriber.sources.audio.constants import NO_TRANSCRIBABLE_TEXT
from chronotranscriber.spec import RunSpec

__all__ = [
    "FASTER_WHISPER_HINT",
    "WhisperMissingError",
    "WhisperOptions",
    "WhisperTranscriber",
    "is_faster_whisper_available",
]

logger = logging.getLogger(__name__)

FASTER_WHISPER_HINT = (
    "faster-whisper is not installed; local audio transcription is unavailable. "
    "Install the optional extra (uv sync --extra audio) or choose the "
    "audio-api method."
)


class WhisperMissingError(RuntimeError):
    """faster-whisper is not installed, so local transcription cannot run."""


def is_faster_whisper_available() -> bool:
    """Check whether the optional ``faster-whisper`` package is importable."""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class WhisperOptions:
    """The faster-whisper model and decoding options of a run.

    *model* is the model folder of the settings or the ``--whisper-model``
    size; *compute_type* ``default`` lets the library choose; *language*
    None detects the language.
    """

    model: str
    device: str = "auto"
    compute_type: str = "default"
    download_root: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    language: str | None = None

    @classmethod
    def of_run(cls, spec: RunSpec, settings: Settings) -> WhisperOptions:
        """Return the options *spec* and the ``whisper`` settings set."""
        whisper = settings.whisper
        compute_type = whisper.compute_type.strip().lower()
        return cls(
            model=str(whisper.model_path or spec.audio.whisper_model),
            device=whisper.device.strip().lower() or "auto",
            compute_type="default" if compute_type in ("", "auto") else compute_type,
            download_root=str(whisper.download_root) if whisper.download_root else None,
            beam_size=whisper.beam_size,
            vad_filter=whisper.vad_filter,
            language=(spec.audio.language or "").strip() or None,
        )


class WhisperTranscriber:
    """Transcribe recordings with the local Whisper model.

    A missing faster-whisper installation fails an item with chunks left
    to transcribe; the model is loaded on the first chunk and kept for the
    run.
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self.options = WhisperOptions.of_run(job.spec, job.settings)
        self._model: Any | None = None
        self._lock = threading.Lock()

    async def open(self) -> None:
        """Nothing to open; the model loads on the first chunk."""

    async def close(self) -> None:
        """Release the model."""
        self._model = None

    async def transcribe(self, item: Item) -> list[Path]:
        """Transcribe one recording into its chunk units; return []."""
        return await transcribe_recording(
            self.job,
            item,
            None,
            self._handler,
            provider="faster-whisper",
            model=self.options.model,
        )

    def load_model(self) -> Any:
        """Return the run's ``WhisperModel``, loading it on the first call.

        Raises:
            RuntimeError: faster-whisper is not installed.
        """
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install the optional extra "
                    "with 'uv sync --extra audio' to use local transcription."
                ) from exc
            options = self.options
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
                options.model,
                options.device,
                options.compute_type,
            )
            kwargs: dict[str, Any] = {
                "device": options.device,
                "compute_type": options.compute_type,
            }
            if options.download_root:
                kwargs["download_root"] = options.download_root
            self._model = WhisperModel(options.model, **kwargs)
            return self._model

    def transcribe_file(self, path: Path) -> str | None:
        """Transcribe one audio file; blocking, so run it in a thread.

        An empty transcript becomes the shared no-text placeholder; any
        failure is logged and returns None.
        """
        try:
            model = self.load_model()
            segments, _info = model.transcribe(
                str(path),
                beam_size=self.options.beam_size,
                vad_filter=self.options.vad_filter,
                language=self.options.language,
            )
            text = " ".join(
                piece for segment in segments if (piece := (segment.text or "").strip())
            ).strip()
            return text if text else NO_TRANSCRIBABLE_TEXT
        except Exception as exc:  # noqa: BLE001 - faster-whisper raises many types
            logger.error("faster-whisper error on %s: %s", path.name, exc)
            return None

    async def transcribe_payload(self, payload: Any, _client: Any) -> PayloadResult:
        """Transcribe one chunk off the event loop.

        A failure becomes an error placeholder that ``--retry-errors``
        retries.
        """
        image_name = payload.image_name
        try:
            text = await asyncio.to_thread(self.transcribe_file, payload.path)
        except Exception as exc:
            logger.exception(
                "Error transcribing %s with local Whisper: %s", image_name, exc
            )
            text = None
        final_text = (
            text if text is not None else f"[transcription error: {image_name}]"
        )
        return (payload, image_name, final_text, None, payload.index)

    def _handler(self) -> StreamHandler:
        if not is_faster_whisper_available():
            raise WhisperMissingError(FASTER_WHISPER_HINT)
        return self.transcribe_payload
