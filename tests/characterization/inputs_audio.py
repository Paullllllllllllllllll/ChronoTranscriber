"""Deterministic WAV recordings whose samples encode their position.

Every sample of second *s* of a recording holds the value ``s + 1``, so the
first frame of any slice names the second the slice starts at, and its frame
count names its length. ``decode_wav`` reads both back from raw bytes, which
lets the fakes identify a chunk by content instead of by file name or call
order. Recordings use a low frame rate to stay a few kilobytes long.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

FRAME_RATE = 200
"""Frames per second of every recording (16-bit mono)."""

SAMPLE_WIDTH = 2
CHANNELS = 1


@dataclass(frozen=True)
class WavInfo:
    """What ``decode_wav`` reads from WAV bytes.

    ``start_second`` is the second of the source recording the bytes start
    at; ``seconds`` is their length; ``frames`` and ``frame_rate`` come from
    the header.
    """

    start_second: int
    seconds: float
    frames: int
    frame_rate: int


def second_frames(second: int, frame_rate: int = FRAME_RATE) -> bytes:
    """Return the frames of second *second* (0-based) of a recording."""
    value = (second + 1).to_bytes(SAMPLE_WIDTH, "little", signed=True)
    return value * frame_rate


def wav_bytes(frames: bytes, frame_rate: int = FRAME_RATE) -> bytes:
    """Return a complete 16-bit mono WAV file holding *frames*."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(frame_rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def make_wav(path: Path, seconds: int) -> Path:
    """Write a recording of *seconds* seconds to *path* and return it."""
    if seconds <= 0:
        raise ValueError("a recording needs at least one second")
    frames = b"".join(second_frames(second) for second in range(seconds))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(frames))
    return path


def read_frames(data: bytes) -> tuple[bytes, int]:
    """Return the raw frames and the frame rate of WAV *data*."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate()


def decode_wav(data: bytes) -> WavInfo:
    """Return the position and length that WAV *data* encodes."""
    frames, frame_rate = read_frames(data)
    count = len(frames) // SAMPLE_WIDTH
    first = int.from_bytes(frames[:SAMPLE_WIDTH], "little", signed=True) if count else 0
    return WavInfo(
        start_second=first - 1,
        seconds=round(count / frame_rate, 3),
        frames=count,
        frame_rate=frame_rate,
    )


def duration_seconds(path: Path) -> float:
    """Return the length of the recording at *path* in seconds."""
    return decode_wav(path.read_bytes()).seconds


__all__ = [
    "FRAME_RATE",
    "WavInfo",
    "decode_wav",
    "duration_seconds",
    "make_wav",
    "read_frames",
    "second_frames",
    "wav_bytes",
]
