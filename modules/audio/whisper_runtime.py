"""Local faster-whisper runtime: availability, model cache, and transcription.

The offline counterpart to :mod:`modules.images.tesseract_runtime`.
``faster-whisper`` ships as an optional extra, so every import of it is lazy
and confined to this module: nothing here fails at import time when the extra
is absent, and :func:`ensure_faster_whisper_available` prints the install hint
instead.

Loaded models are cached process-wide — a large-v3 load costs seconds and
gigabytes, and the pipeline transcribes many chunks per run.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

from modules.audio.constants import NO_TRANSCRIBABLE_TEXT
from modules.infra.logger import setup_logger
from modules.ui import print_error

logger = setup_logger(__name__)

# (model id or path, device, compute_type) -> loaded WhisperModel.
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def is_faster_whisper_available() -> bool:
    """Check whether the optional ``faster-whisper`` package is importable."""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def ensure_faster_whisper_available() -> bool:
    """Verify that faster-whisper is installed, printing a hint if not.

    Returns:
        True if available, False otherwise.
    """
    if is_faster_whisper_available():
        return True

    print_error(
        "faster-whisper is not installed; local audio transcription is"
        " unavailable.\n"
        "- Install the optional extra: uv sync --extra audio\n"
        "- Or choose the remote 'audio-api' method instead of local Whisper."
    )
    return False


def _resolve_runtime(cfg: dict[str, Any]) -> tuple[str, str, str, str | None]:
    """Return ``(model_ref, device, compute_type, download_root)`` from config.

    ``model_path`` overrides ``model_size`` when set. ``auto`` maps to the
    ``WhisperModel`` defaults: ``device="auto"`` is a valid value the library
    resolves itself, while its ``compute_type`` sentinel is ``"default"``.
    """
    model_path = str(cfg.get("model_path") or "").strip()
    model_ref = model_path or str(cfg.get("model_size") or "large-v3").strip()

    device = str(cfg.get("device") or "auto").strip().lower() or "auto"

    compute_type = str(cfg.get("compute_type") or "auto").strip().lower()
    if compute_type in ("", "auto"):
        compute_type = "default"

    download_root = str(cfg.get("download_root") or "").strip() or None
    return model_ref, device, compute_type, download_root


def get_whisper_model(cfg: dict[str, Any]) -> Any:
    """Load (or return the cached) faster-whisper model for *cfg*.

    Args:
        cfg: The ``local_whisper`` section of ``audio_config.yaml``.

    Returns:
        A ``faster_whisper.WhisperModel`` instance.

    Raises:
        RuntimeError: When the optional dependency is not installed.
    """
    model_ref, device, compute_type, download_root = _resolve_runtime(cfg)
    key = (model_ref, device, compute_type)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install the optional extra "
                "with 'uv sync --extra audio' to use local transcription."
            ) from exc

        logger.info(
            "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
            model_ref,
            device,
            compute_type,
        )
        kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
        if download_root:
            kwargs["download_root"] = download_root
        model = WhisperModel(model_ref, **kwargs)
        _MODEL_CACHE[key] = model
        return model


def transcribe_file(path: Path, cfg: dict[str, Any]) -> str | None:
    """Transcribe one audio file with the local Whisper model.

    Blocking, matching the :func:`modules.images.tesseract_runtime.perform_ocr`
    contract exactly: callers on the event loop run it through
    ``asyncio.to_thread``, an empty result becomes the shared
    ``[No transcribable text]`` sentinel, and any failure returns ``None`` after
    logging.

    Args:
        path: Audio file (a chunk, or a whole recording — faster-whisper
            handles long inputs natively).
        cfg: The ``local_whisper`` section of ``audio_config.yaml``.

    Returns:
        The transcript, the no-text placeholder, or None on error.
    """
    try:
        model = get_whisper_model(cfg)
        language = str(cfg.get("language") or "").strip() or None
        segments, _info = model.transcribe(
            str(path),
            beam_size=int(cfg.get("beam_size", 5) or 5),
            vad_filter=bool(cfg.get("vad_filter", True)),
            language=language,
        )
        text = " ".join(
            piece for segment in segments if (piece := (segment.text or "").strip())
        ).strip()
        return text if text else NO_TRANSCRIBABLE_TEXT
    except Exception as exc:
        logger.error("faster-whisper error on %s: %s", path.name, exc)
        return None


def clear_model_cache() -> None:
    """Drop every cached model (test isolation; reloaded on next request)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


__all__ = [
    "clear_model_cache",
    "ensure_faster_whisper_available",
    "get_whisper_model",
    "is_faster_whisper_available",
    "transcribe_file",
]
