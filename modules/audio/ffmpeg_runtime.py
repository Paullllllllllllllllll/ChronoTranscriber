"""FFmpeg/ffprobe executable lookup, duration probing, and segment cutting.

Mirrors :mod:`modules.images.tesseract_runtime` for the audio pipeline: the
configured executable paths are resolved once from ``audio_config.ffmpeg``,
availability is cached, and a missing binary yields an actionable install
hint rather than an opaque ``FileNotFoundError`` deep inside the workflow.

Every ``ffmpeg`` invocation passes ``-nostdin``: the interactive wizard shares
this process's stdin, and ffmpeg would otherwise consume the user's keystrokes
while a chunk is being cut. (``ffprobe`` never reads stdin.)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from modules.infra.logger import setup_logger
from modules.ui import print_error

logger = setup_logger(__name__)

# Configured executable paths ("" means "resolve from PATH").
_FFMPEG_CMD: str = ""
_FFPROBE_CMD: str = ""

# Availability cache, invalidated by configure_ffmpeg_executables().
_FFMPEG_AVAILABLE: bool | None = None
_FFPROBE_AVAILABLE: bool | None = None

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")

# ffprobe on a multi-hour recording is still sub-second; a generous ceiling
# only guards against a wedged process.
_PROBE_TIMEOUT_S = 120


def configure_ffmpeg_executables(audio_config: dict[str, Any]) -> None:
    """Configure the ffmpeg/ffprobe executable paths from configuration.

    Args:
        audio_config: Parsed ``audio_config.yaml`` mapping. The ``ffmpeg``
            section may carry ``ffmpeg_cmd`` and ``ffprobe_cmd``; an empty
            string (the default) means "use whatever is on PATH".
    """
    global _FFMPEG_CMD, _FFPROBE_CMD, _FFMPEG_AVAILABLE, _FFPROBE_AVAILABLE

    section = audio_config.get("ffmpeg", {}) or {}
    ffmpeg_cmd = str(section.get("ffmpeg_cmd") or "").strip()
    ffprobe_cmd = str(section.get("ffprobe_cmd") or "").strip()

    for label, cmd in (("ffmpeg_cmd", ffmpeg_cmd), ("ffprobe_cmd", ffprobe_cmd)):
        if cmd and not Path(cmd).exists():
            logger.warning("Configured %s does not exist: %s", label, cmd)

    _FFMPEG_CMD = ffmpeg_cmd
    _FFPROBE_CMD = ffprobe_cmd
    _FFMPEG_AVAILABLE = None
    _FFPROBE_AVAILABLE = None
    if ffmpeg_cmd:
        logger.info("Using ffmpeg executable: %s", ffmpeg_cmd)
    if ffprobe_cmd:
        logger.info("Using ffprobe executable: %s", ffprobe_cmd)


def get_ffmpeg_command() -> str:
    """Return the ffmpeg executable to invoke (configured path or ``ffmpeg``)."""
    return _FFMPEG_CMD or "ffmpeg"


def get_ffprobe_command() -> str:
    """Return the ffprobe executable to invoke (configured path or ``ffprobe``)."""
    return _FFPROBE_CMD or "ffprobe"


def _probe_executable(command: str) -> bool:
    """Return True when ``command -version`` runs successfully."""
    if not Path(command).is_absolute() and shutil.which(command) is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, never shell=True
            [command, "-version"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not run '%s -version': %s", command, exc)
        return False
    return result.returncode == 0


def is_ffmpeg_available() -> bool:
    """Check whether the ffmpeg executable is available (cached)."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        _FFMPEG_AVAILABLE = _probe_executable(get_ffmpeg_command())
    return _FFMPEG_AVAILABLE


def is_ffprobe_available() -> bool:
    """Check whether the ffprobe executable is available (cached)."""
    global _FFPROBE_AVAILABLE
    if _FFPROBE_AVAILABLE is None:
        _FFPROBE_AVAILABLE = _probe_executable(get_ffprobe_command())
    return _FFPROBE_AVAILABLE


def ensure_ffmpeg_available() -> bool:
    """Verify that ffmpeg is available, printing an install hint if not.

    Returns:
        True if available, False otherwise.
    """
    if is_ffmpeg_available():
        return True

    print_error(
        "FFmpeg is not installed or not in PATH.\n"
        "- Windows: winget install Gyan.FFmpeg  (or: choco install ffmpeg)\n"
        "- macOS: brew install ffmpeg | Linux: apt install ffmpeg\n"
        "- Or set 'ffmpeg.ffmpeg_cmd' (and 'ffmpeg.ffprobe_cmd') in"
        " config/audio_config.yaml to the full path, e.g.:\n"
        "  C:\\\\Program Files\\\\ffmpeg\\\\bin\\\\ffmpeg.exe"
    )
    return False


def _duration_from_ffprobe(path: Path) -> float | None:
    """Read the container duration via ``ffprobe -show_entries format=duration``."""
    if not is_ffprobe_available():
        return None
    argv = [
        get_ffprobe_command(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, never shell=True
            argv, capture_output=True, timeout=_PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("ffprobe failed on %s: %s", path.name, exc)
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout.decode("utf-8", errors="replace"))
        raw = (payload.get("format") or {}).get("duration")
        if raw is None:
            return None
        duration = float(raw)
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return None
    return duration if duration > 0 else None


def _duration_from_ffmpeg(path: Path) -> float | None:
    """Parse ``Duration: HH:MM:SS.ss`` out of ``ffmpeg -i`` stderr."""
    if not is_ffmpeg_available():
        return None
    argv = [get_ffmpeg_command(), "-nostdin", "-hide_banner", "-i", str(path)]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, never shell=True
            argv, capture_output=True, timeout=_PROBE_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("ffmpeg probe failed on %s: %s", path.name, exc)
        return None
    match = _DURATION_RE.search(result.stderr.decode("utf-8", errors="replace"))
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total if total > 0 else None


def probe_duration_seconds(path: Path) -> float | None:
    """Return the recording's duration in seconds, or None when unknown.

    Tries ``ffprobe`` first (structured JSON, authoritative) and falls back to
    scraping ``Duration:`` from ``ffmpeg -i`` stderr for containers whose
    ffprobe metadata is missing. Callers treat ``None`` as "cannot plan chunks
    by time" and must fall back to a whole-file request.
    """
    duration = _duration_from_ffprobe(path)
    if duration is not None:
        return duration
    return _duration_from_ffmpeg(path)


def _encoder_args(chunk_format: str, mono: bool, sample_rate: int) -> list[str]:
    """Return the codec/filter argv tail for the requested chunk format."""
    fmt = (chunk_format or "mp3").strip().lower()
    if fmt == "copy":
        return ["-c:a", "copy"]
    if fmt == "wav16":
        # 16-bit PCM is always emitted mono at the configured rate: the format
        # exists to feed size-predictable, provider-friendly chunks.
        return ["-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le"]
    if fmt != "mp3":
        logger.warning("Unknown chunk_format '%s'; falling back to mp3.", chunk_format)
    args = []
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sample_rate), "-c:a", "libmp3lame", "-q:a", "4"]
    return args


def cut_segment(
    src: Path,
    dst: Path,
    *,
    start: float,
    duration: float | None,
    chunk_format: str,
    mono: bool,
    sample_rate: int,
) -> None:
    """Cut ``[start, start + duration)`` out of *src* into *dst* with ffmpeg.

    Blocking; callers on the event loop run it through ``asyncio.to_thread``.
    ``-ss`` precedes ``-i`` so ffmpeg seeks the input rather than decoding and
    discarding everything before the cut point, and ``-vn -map 0:a:0`` keeps
    only the first audio stream (cover art in an m4a would otherwise ride
    along). A ``duration`` of ``None`` runs to the end of the recording.

    Args:
        src: Source recording.
        dst: Destination chunk file; its suffix must match *chunk_format*
            (``.mp3``, ``.wav`` for ``wav16``, the source suffix for ``copy``).
        start: Seek offset in seconds.
        duration: Segment length in seconds, or None for "to the end".
        chunk_format: ``mp3``, ``wav16``, or ``copy``.
        mono: Downmix to one channel (ignored by ``copy``; always on for
            ``wav16``).
        sample_rate: Output sample rate in Hz (ignored by ``copy``).

    Raises:
        RuntimeError: When ffmpeg is unavailable or exits non-zero. The message
            carries the tail of ffmpeg's stderr so the cause is visible in the
            log without re-running the command.
    """
    if not is_ffmpeg_available():
        raise RuntimeError(
            "FFmpeg is required to split audio but was not found. Install it "
            "(Windows: winget install Gyan.FFmpeg) or set ffmpeg.ffmpeg_cmd in "
            "config/audio_config.yaml."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        get_ffmpeg_command(),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
    ]
    if duration is not None:
        argv += ["-t", f"{duration:.3f}"]
    argv += ["-i", str(src), "-vn", "-map", "0:a:0"]
    argv += _encoder_args(chunk_format, mono, sample_rate)
    argv.append(str(dst))

    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, never shell=True
            argv, capture_output=True, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not run ffmpeg on {src.name}: {exc}") from exc

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) cutting {src.name} at "
            f"{start:.1f}s: {stderr_tail}"
        )


__all__ = [
    "configure_ffmpeg_executables",
    "cut_segment",
    "ensure_ffmpeg_available",
    "get_ffmpeg_command",
    "get_ffprobe_command",
    "is_ffmpeg_available",
    "is_ffprobe_available",
    "probe_duration_seconds",
]
