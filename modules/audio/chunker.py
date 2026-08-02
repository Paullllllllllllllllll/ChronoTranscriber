"""Deterministic chunk planning for long recordings.

Remote speech-to-text venues cap each request by payload size (OpenAI 25 MB,
Gemini ~20 MB inline), so a recording that exceeds either the size cap or the
nominal chunk length is cut into fixed-length segments before submission.
Planning is separated from cutting on purpose: :func:`plan_chunks` is pure and
side-effect free, so the same inputs always yield the same segment boundaries
and a resumed run re-derives exactly the chunk indices the first run wrote.

RESUME CAVEAT: chunk indices are positional, not content-addressed. Changing
``chunking.target_seconds``, ``overlap_seconds``, ``max_request_bytes``, or the
output format between runs shifts every boundary, so already-transcribed chunk
records no longer describe the same audio. Compare
:func:`chunk_plan_signature` against the stored provenance before honoring a
skip set; on mismatch, restart the item rather than resume it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from modules.audio.constants import DEFAULT_MIN_CHUNK_SECONDS

# Fraction of the hard request cap a planned chunk is allowed to fill. The
# bitrate estimate is nominal, so a 10 % headroom keeps a slightly denser than
# expected segment from tipping over the provider's ceiling.
_SIZE_SAFETY_FACTOR = 0.9

# Nominal encoder output rates, in bytes per second of audio.
_MP3_Q4_MONO_BYTES_PER_SECOND = 8000.0


@dataclass(frozen=True)
class ChunkSpec:
    """One planned segment of a recording.

    Attributes:
        index: 0-based position in the plan; drives the chunk file name and
            the resume skip set.
        start_seconds: Seek offset into the source recording.
        duration_seconds: Segment length, or ``None`` for "run to the end of
            the recording" (always the last spec, and the only spec of a
            whole-file plan).
    """

    index: int
    start_seconds: float
    duration_seconds: float | None


def estimate_output_bytes_per_second(
    chunk_format: str,
    sample_rate: int,
    mono: bool,
    *,
    source_bytes_per_second: float | None = None,
) -> float:
    """Estimate the encoded size of one second of chunk audio, in bytes.

    ``mp3`` (libmp3lame ``-q:a 4``) and ``wav16`` (PCM s16le) have predictable
    rates derived from the configured sample rate and channel count. Stream
    ``copy`` re-emits the source bitrate, which is only knowable from the
    source itself, so the caller must pass *source_bytes_per_second*
    (``size_bytes / duration_seconds``); without it the mp3 estimate is used as
    a conservative stand-in and a chunk may overshoot the request cap.

    Args:
        chunk_format: ``mp3``, ``wav16``, or ``copy``.
        sample_rate: Output sample rate in Hz.
        mono: Whether the chunk is downmixed to one channel.
        source_bytes_per_second: Measured source rate, required for ``copy``.

    Returns:
        Estimated bytes per second of audio; always strictly positive.
    """
    fmt = (chunk_format or "mp3").strip().lower()
    channels = 1 if mono else 2
    if fmt == "wav16":
        # wav16 is always emitted mono (see ffmpeg_runtime._encoder_args).
        return float(max(1, sample_rate) * 2)
    if fmt == "copy":
        if source_bytes_per_second and source_bytes_per_second > 0:
            return float(source_bytes_per_second)
        return _MP3_Q4_MONO_BYTES_PER_SECOND
    return _MP3_Q4_MONO_BYTES_PER_SECOND * channels


def plan_chunks(
    *,
    duration_seconds: float,
    size_bytes: int,
    target_seconds: int,
    max_request_bytes: int,
    output_bytes_per_second: float,
    overlap_seconds: float = 0.0,
    min_chunk_seconds: int = DEFAULT_MIN_CHUNK_SECONDS,
) -> list[ChunkSpec]:
    """Plan the segments a recording is cut into before submission.

    Pure and deterministic. A recording that fits the request cap AND the
    nominal chunk length is left whole: the single returned spec has
    ``duration_seconds=None``, which the streamer reads as "send the source
    file untouched", so no ffmpeg pass and no re-encode happen at all.

    Otherwise the effective chunk length is the smaller of *target_seconds* and
    the length that fills ``max_request_bytes * 0.9`` at the estimated bitrate,
    floored at *min_chunk_seconds* so a pathological bitrate estimate cannot
    plan thousands of one-second requests. Chunks then start at ``0``,
    ``effective - overlap``, ``2 * (effective - overlap)``, ... until the
    recording is covered; in a multi-chunk plan the final chunk's duration is
    ``None`` so it runs to the true end regardless of probe rounding. A derived
    single-chunk plan (oversized but shorter than one chunk) keeps its finite
    duration so the streamer still re-encodes it instead of sending the
    oversized source untouched.

    Args:
        duration_seconds: Probed recording length; ``<= 0`` means unknown.
        size_bytes: Source file size in bytes.
        target_seconds: Nominal chunk length from ``chunking.target_seconds``.
        max_request_bytes: Hard per-request payload ceiling.
        output_bytes_per_second: Estimated encoded rate of the emitted chunks
            (see :func:`estimate_output_bytes_per_second`).
        overlap_seconds: Audio repeated at each boundary; 0 disables overlap.
        min_chunk_seconds: Lower bound on the effective chunk length.

    Returns:
        A non-empty list of specs with contiguous 0-based indices.
    """
    if duration_seconds <= 0 or max_request_bytes <= 0:
        # Nothing to plan against: send the file whole and let the provider's
        # own error surface if it is oversized.
        return [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]

    if size_bytes <= max_request_bytes and duration_seconds <= target_seconds:
        return [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]

    rate = output_bytes_per_second if output_bytes_per_second > 0 else 1.0
    size_bound = int(math.floor(max_request_bytes * _SIZE_SAFETY_FACTOR / rate))
    effective = min(int(target_seconds), size_bound)
    effective = max(effective, int(min_chunk_seconds))

    step = float(effective) - max(0.0, float(overlap_seconds))
    if step <= 0:
        # Overlap at or beyond the chunk length would never advance; ignore it.
        step = float(effective)

    specs: list[ChunkSpec] = []
    start = 0.0
    index = 0
    while start < duration_seconds:
        specs.append(
            ChunkSpec(
                index=index,
                start_seconds=round(start, 3),
                duration_seconds=float(effective),
            )
        )
        index += 1
        start += step

    # The last chunk runs to the end so probe rounding cannot clip the tail.
    # A single-spec plan keeps its finite duration: ``None`` there would make
    # it indistinguishable from the send-untouched whole-file spec and skip
    # the re-encode an oversized-but-short recording exists to get. Clipping
    # is impossible in that case because one chunk implies
    # ``duration_seconds <= effective``.
    if len(specs) > 1:
        last = specs[-1]
        specs[-1] = ChunkSpec(
            index=last.index, start_seconds=last.start_seconds, duration_seconds=None
        )
    return specs


def chunk_plan_signature(specs: list[ChunkSpec]) -> str:
    """Return a stable provenance token identifying a chunk plan.

    Deterministic for a given list of specs and stable across runs and
    platforms: ``"<count>x<nominal-length>s-<hash>"``, where the hash is the
    first 8 hex digits of the SHA-256 of the rounded ``(index, start,
    duration)`` triples. Persist it beside the chunk records and compare before
    honoring a resume skip set; a mismatch means the boundaries moved.
    """
    if not specs:
        return "0x0s-00000000"
    payload = ";".join(
        f"{s.index}:{s.start_seconds:.3f}:"
        f"{'end' if s.duration_seconds is None else format(s.duration_seconds, '.3f')}"
        for s in specs
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    nominal = specs[0].duration_seconds or 0.0
    return f"{len(specs)}x{int(nominal)}s-{digest}"


__all__ = [
    "ChunkSpec",
    "chunk_plan_signature",
    "estimate_output_bytes_per_second",
    "plan_chunks",
]
