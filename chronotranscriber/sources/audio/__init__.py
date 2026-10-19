"""Audio recordings: the chunk plan of one recording.

A recording is sent whole or cut into chunks: a remote request has a size
limit, and ``audio_chunking.apply_to_local`` or a page range cuts local runs
too. :func:`plan_recording` plans the chunks of one item and
:func:`describe_chunks` the chunk count a dry run reports. The submodules
hold the chunk planner (:mod:`.chunker`), the FFmpeg runtime (:mod:`.ffmpeg`),
the chunk stream (:mod:`.stream`) and the format tables (:mod:`.constants`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from chronotranscriber.settings import Settings
from chronotranscriber.sources.audio.chunker import (
    ChunkSpec,
    estimate_output_bytes_per_second,
    plan_chunks,
)
from chronotranscriber.sources.audio.constants import (
    DEFAULT_MIN_CHUNK_SECONDS,
    GEMINI_INLINE_LIMIT_BYTES,
    OPENAI_UPLOAD_LIMIT_BYTES,
)
from chronotranscriber.sources.audio.ffmpeg import (
    FFMPEG_INSTALL_HINT,
    is_ffmpeg_available,
    probe_duration_seconds,
)
from chronotranscriber.spec import RunSpec

__all__ = [
    "ChunkPlan",
    "ChunkSettings",
    "RecordingSplitError",
    "chunk_settings",
    "describe_chunks",
    "plan_recording",
]

logger = logging.getLogger(__name__)


class RecordingSplitError(RuntimeError):
    """A recording above the request limit cannot be cut into chunks."""


@dataclass(frozen=True)
class ChunkSettings:
    """How the recordings of one run are cut.

    *provider* is the remote provider whose request limit applies; local
    Whisper runs take OpenAI's.
    """

    provider: str
    max_request_bytes: int
    target_seconds: int
    overlap_seconds: float
    min_chunk_seconds: int
    chunk_format: str
    mono: bool
    sample_rate: int
    apply_to_local: bool


def chunk_settings(spec: RunSpec, settings: Settings) -> ChunkSettings:
    """Return the chunking values of a run.

    The request limit is ``audio_chunking.max_request_bytes``, else the limit
    of the remote provider.
    """
    provider = spec.model.provider if spec.method.name == "audio-api" else "openai"
    provider_limit = (
        GEMINI_INLINE_LIMIT_BYTES if provider == "google" else OPENAI_UPLOAD_LIMIT_BYTES
    )
    chunking = settings.audio_chunking
    return ChunkSettings(
        provider=provider,
        max_request_bytes=chunking.max_request_bytes or provider_limit,
        target_seconds=spec.audio.chunk_seconds,
        overlap_seconds=chunking.overlap_seconds,
        min_chunk_seconds=DEFAULT_MIN_CHUNK_SECONDS,
        chunk_format=chunking.chunk_format,
        mono=chunking.mono,
        sample_rate=chunking.sample_rate,
        apply_to_local=chunking.apply_to_local,
    )


@dataclass(frozen=True)
class ChunkPlan:
    """The chunks of one recording.

    *planned* is False for an unprobed whole-file plan, under which chunk
    page ranges do not apply. *warning* tells the user about a fallback;
    *note* describes a plan of several chunks.
    """

    specs: list[ChunkSpec]
    planned: bool
    warning: str | None = None
    note: str | None = None


def _split(
    duration: float, size_bytes: int, settings: ChunkSettings
) -> list[ChunkSpec]:
    return plan_chunks(
        duration_seconds=duration,
        size_bytes=size_bytes,
        target_seconds=settings.target_seconds,
        max_request_bytes=settings.max_request_bytes,
        output_bytes_per_second=estimate_output_bytes_per_second(
            settings.chunk_format,
            settings.sample_rate,
            settings.mono,
            source_bytes_per_second=size_bytes / duration,
        ),
        overlap_seconds=settings.overlap_seconds,
        min_chunk_seconds=settings.min_chunk_seconds,
    )


def plan_recording(
    path: Path,
    method: str,
    settings: ChunkSettings,
    *,
    page_range_active: bool,
) -> ChunkPlan:
    """Plan the chunks *path* is cut into for *method*.

    Probing runs ffprobe, so it is skipped where it cannot change the
    answer: a local Whisper run without ``apply_to_local`` and without a
    page range sends the file whole. Only the remote method has a size
    limit; a local run may always fall back to the whole file.

    Raises:
        RecordingSplitError: a remote recording above the limit, with FFmpeg
            unavailable or unable to measure it.
    """
    whole_file = [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]
    size_bytes = path.stat().st_size
    if not (page_range_active or method == "audio-api" or settings.apply_to_local):
        return ChunkPlan(whole_file, planned=False)

    oversized = method == "audio-api" and size_bytes > settings.max_request_bytes
    if not is_ffmpeg_available():
        if oversized:
            raise RecordingSplitError(
                f"'{path.name}' is {size_bytes / 1048576:.1f} MB and must be "
                f"split for the {settings.provider} API, but FFmpeg is "
                f"unavailable. {FFMPEG_INSTALL_HINT}"
            )
        logger.info("FFmpeg unavailable; sending %s as a single request.", path.name)
        return ChunkPlan(whole_file, planned=False)

    duration = probe_duration_seconds(path)
    if duration is None or duration <= 0:
        if oversized:
            raise RecordingSplitError(
                f"Could not determine the duration of '{path.name}', which "
                f"exceeds the per-request size limit."
            )
        return ChunkPlan(
            whole_file,
            planned=False,
            warning=(
                f"Could not determine the duration of '{path.name}'; "
                f"transcribing it as a single request."
            ),
        )

    specs = _split(duration, size_bytes, settings)
    note = (
        f"Planned {len(specs)} chunk(s) for {duration / 60:.1f} minute(s) of audio."
        if len(specs) > 1
        else None
    )
    return ChunkPlan(specs, planned=True, note=note)


def describe_chunks(path: Path, method: str, settings: ChunkSettings) -> str:
    """Return how many chunks *path* would be cut into, for a dry run.

    A recording within the request limit is one request without probing;
    an oversized one is ``unknown`` when FFmpeg cannot measure it.
    """
    if method == "whisper" and not settings.apply_to_local:
        return "1 chunk"
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return "unknown (file unreadable)"
    if size_bytes <= settings.max_request_bytes:
        return "1 chunk"
    if not is_ffmpeg_available():
        return "unknown (ffmpeg not available)"
    duration = probe_duration_seconds(path)
    if duration is None or duration <= 0:
        return "unknown (duration not probeable)"
    return f"{len(_split(duration, size_bytes, settings))} chunk(s)"
