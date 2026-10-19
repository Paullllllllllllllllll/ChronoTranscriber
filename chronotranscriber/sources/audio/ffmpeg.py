"""FFmpeg/ffprobe executable lookup, duration probing, and segment cutting.

The executable paths come from the ``tools`` block of the settings (set
once per run with :func:`configure_ffmpeg_executables`), availability is
cached, and a missing binary yields :data:`FFMPEG_INSTALL_HINT` rather than
an opaque ``FileNotFoundError`` deep inside the run.

Every ``ffmpeg`` invocation passes ``-nostdin``: the interactive wizard shares
this process's stdin, and ffmpeg would otherwise consume the user's keystrokes
while a chunk is being cut. (``ffprobe`` never reads stdin.)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

FFMPEG_INSTALL_HINT = (
    "Install FFmpeg (Windows: winget install Gyan.FFmpeg; macOS: brew install "
    "ffmpeg; Linux: apt install ffmpeg) or set tools.ffmpeg_cmd and "
    "tools.ffprobe_cmd in the settings file to the full paths."
)

_FFMPEG_CMD: str = ""  # "" resolves from PATH
_FFPROBE_CMD: str = ""

# Reset by configure_ffmpeg_executables().
_FFMPEG_AVAILABLE: bool | None = None
_FFPROBE_AVAILABLE: bool | None = None

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")

# Probing takes under a second; the ceiling only guards against a wedged process.
_PROBE_TIMEOUT_S = 120


def configure_ffmpeg_executables(
    ffmpeg_cmd: str | None = None, ffprobe_cmd: str | None = None
) -> None:
    """Set the ffmpeg and ffprobe executables; None or blank finds them on PATH."""
    global _FFMPEG_CMD, _FFPROBE_CMD, _FFMPEG_AVAILABLE, _FFPROBE_AVAILABLE

    ffmpeg_cmd = (ffmpeg_cmd or "").strip()
    ffprobe_cmd = (ffprobe_cmd or "").strip()

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
        result = subprocess.run(
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
        result = subprocess.run(
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
        result = subprocess.run(
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
        # wav16 is always mono, so its chunk size is predictable.
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
            f"FFmpeg is required to split audio but was not found. "
            f"{FFMPEG_INSTALL_HINT}"
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
        result = subprocess.run(argv, capture_output=True, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not run ffmpeg on {src.name}: {exc}") from exc

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) cutting {src.name} at "
            f"{start:.1f}s: {stderr_tail}"
        )


__all__ = [
    "FFMPEG_INSTALL_HINT",
    "configure_ffmpeg_executables",
    "cut_segment",
    "get_ffmpeg_command",
    "get_ffprobe_command",
    "is_ffmpeg_available",
    "is_ffprobe_available",
    "probe_duration_seconds",
]
