"""Streaming chunk-payload producer for the audio transcription pipeline.

The audio analogue of :func:`chronotranscriber.sources.images.stream_pages`: it
turns a planned list of :class:`~chronotranscriber.sources.audio.chunker.ChunkSpec`
segments into payloads the transcription pipeline can submit, materializing
each segment with ffmpeg only when it is about to be sent. A recording that
fits a single request is yielded without any ffmpeg pass at all: the payload
points straight at the source file.

Chunks are named ``{stem}_chunk_NNNN{suffix}``, the label of their store unit;
the caller passes the indices the store holds as done in *skip_indices*.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronotranscriber.sources.audio.chunker import ChunkSpec
from chronotranscriber.sources.audio.constants import SUPPORTED_AUDIO_FORMATS
from chronotranscriber.sources.audio.ffmpeg import cut_segment

logger = logging.getLogger(__name__)

_AUDIO_CHUNK_NAME_RE = re.compile(r"_chunk_(\d{4,})\.[^.]+$", re.IGNORECASE)

_DEFAULT_AUDIO_MIME = "application/octet-stream"


@dataclass
class AudioChunkPayload:
    """One audio segment ready for a speech-to-text request.

    ``index`` is the absolute 0-based chunk index within the plan, so chunk
    numbering stays correct under resume. ``image_name`` carries the pipeline's
    generic per-unit name, which for audio is the chunk file name.
    """

    index: int
    image_name: str
    path: Path
    mime_type: str
    source_file: str
    page_index: int | None = None
    start_seconds: float = 0.0
    duration_seconds: float = 0.0
    byte_size: int = 0
    sha256: str = ""
    is_whole_file: bool = False

    def provenance(self) -> dict[str, Any]:
        """Per-chunk reproducibility record, stored as the unit's provenance.

        Mirrors :meth:`chronotranscriber.sources.images.PageImage.provenance`:
        a flat dict of the facts needed to identify exactly which bytes were
        sent, with the image pipeline's ``sha256``/``byte_size`` keys kept
        under the same names.
        """
        return {
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "source_file": self.source_file,
            "chunk_index": self.index,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "is_whole_file": self.is_whole_file,
        }

    async def read_bytes(self) -> bytes:
        """Read the chunk's bytes off the event loop."""
        return await asyncio.to_thread(self.path.read_bytes)

    async def read_base64(self) -> str:
        """Read the chunk and return it base64-encoded (for inline API parts)."""
        raw = await self.read_bytes()
        return base64.b64encode(raw).decode("utf-8")


def audio_chunk_name(source: Path, index: int, suffix: str) -> str:
    """Virtual chunk name for a 0-based chunk index.

    ``suffix`` includes the leading dot and is lowercased, so the name matches
    :func:`parse_audio_chunk_index` regardless of how the source was cased.
    """
    return f"{source.stem}_chunk_{index + 1:04d}{suffix.lower()}"


def parse_audio_chunk_index(image_name: str) -> int | None:
    """Parse the 0-based chunk index out of a chunk name, else None."""
    match = _AUDIO_CHUNK_NAME_RE.search(image_name.strip())
    if match:
        return int(match.group(1)) - 1
    return None


def chunk_names(source: Path, specs: list[ChunkSpec], chunk_format: str) -> list[str]:
    """Return the name of every chunk of a plan, in plan order.

    A whole-file plan keeps the source's suffix; cut chunks take the suffix
    of *chunk_format*.
    """
    if is_whole_file_plan(specs):
        suffix = source.suffix.lower()
    else:
        suffix = _chunk_suffix(source, chunk_format)
    return [audio_chunk_name(source, spec.index, suffix) for spec in specs]


def _chunk_suffix(source: Path, chunk_format: str) -> str:
    """Return the file suffix chunks of *chunk_format* are written with."""
    fmt = (chunk_format or "mp3").strip().lower()
    if fmt == "wav16":
        return ".wav"
    if fmt == "copy":
        return source.suffix.lower()
    return ".mp3"


def _mime_for(suffix: str) -> str:
    """MIME type for a chunk suffix, falling back to a generic binary type."""
    return SUPPORTED_AUDIO_FORMATS.get(suffix.lower(), _DEFAULT_AUDIO_MIME)


def _materialize_chunk(
    source: Path,
    dst: Path,
    spec: ChunkSpec,
    chunk_format: str,
    mono: bool,
    sample_rate: int,
) -> tuple[int, str]:
    """Cut one chunk and return its ``(byte_size, sha256)`` (thread worker)."""
    cut_segment(
        source,
        dst,
        start=spec.start_seconds,
        duration=spec.duration_seconds,
        chunk_format=chunk_format,
        mono=mono,
        sample_rate=sample_rate,
    )
    raw = dst.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _measure_file(path: Path) -> tuple[int, str]:
    """Return ``(byte_size, sha256)`` for a file (thread worker)."""
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def is_whole_file_plan(specs: list[ChunkSpec]) -> bool:
    """True when the plan is the single "send the source untouched" spec."""
    return (
        len(specs) == 1
        and specs[0].index == 0
        and specs[0].start_seconds == 0.0
        and specs[0].duration_seconds is None
    )


async def stream_audio_chunks(
    source: Path,
    *,
    specs: list[ChunkSpec],
    chunk_format: str,
    mono: bool,
    sample_rate: int,
    work_dir: Path | None = None,
    skip_indices: set[int] | None = None,
) -> AsyncIterator[AudioChunkPayload]:
    """Yield payloads for the chunks that still need transcribing.

    A whole-file plan (a single spec at offset 0 with no duration) skips ffmpeg
    entirely and yields a payload pointing at *source*; the chunk still gets a
    ``_chunk_0001`` name so resume behaves identically either way.

    Otherwise each spec not in *skip_indices* is cut into *work_dir* through
    ``asyncio.to_thread`` and yielded as soon as it exists. Exactly one chunk is
    materialized at a time: the transcription pipeline's bounded queue supplies
    the concurrency, and cutting ahead would only pile decoded audio on disk.

    Args:
        source: The recording being transcribed.
        specs: Plan from :func:`~chronotranscriber.sources.audio.chunker.plan_chunks`.
        chunk_format: ``mp3``, ``wav16``, or ``copy``.
        mono: Downmix to a single channel.
        sample_rate: Output sample rate in Hz.
        work_dir: Directory the chunk files are written into; a plan that
            cuts chunks requires it.
        skip_indices: Chunk indices the store holds as done (resume).

    Raises:
        ValueError: The plan cuts chunks and *work_dir* is None.
        RuntimeError: A chunk could not be cut; raised after the other chunks
            have been yielded.
    """
    skip = skip_indices or set()

    if is_whole_file_plan(specs):
        if 0 in skip:
            return
        suffix = source.suffix.lower()
        byte_size, digest = await asyncio.to_thread(_measure_file, source)
        yield AudioChunkPayload(
            index=0,
            image_name=audio_chunk_name(source, 0, suffix),
            path=source,
            mime_type=_mime_for(suffix),
            source_file=str(source),
            page_index=None,
            start_seconds=0.0,
            duration_seconds=0.0,
            byte_size=byte_size,
            sha256=digest,
            is_whole_file=True,
        )
        return

    if work_dir is None:
        raise ValueError(f"Cutting '{source.name}' into chunks needs a work_dir.")
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = _chunk_suffix(source, chunk_format)

    uncut = 0
    for spec in specs:
        if spec.index in skip:
            continue
        name = audio_chunk_name(source, spec.index, suffix)
        dst = work_dir / name
        try:
            byte_size, digest = await asyncio.to_thread(
                _materialize_chunk,
                source,
                dst,
                spec,
                chunk_format,
                mono,
                sample_rate,
            )
        except (RuntimeError, OSError) as exc:
            logger.error(
                "Error cutting chunk %d of %s at %.1fs: %s",
                spec.index + 1,
                source.name,
                spec.start_seconds,
                exc,
            )
            uncut += 1
            continue
        yield AudioChunkPayload(
            index=spec.index,
            image_name=name,
            path=dst,
            mime_type=_mime_for(suffix),
            source_file=str(source),
            page_index=None,
            start_seconds=spec.start_seconds,
            duration_seconds=float(spec.duration_seconds or 0.0),
            byte_size=byte_size,
            sha256=digest,
            is_whole_file=False,
        )
    if uncut:
        raise RuntimeError(f"{uncut} chunk(s) of '{source.name}' could not be cut.")


__all__ = [
    "AudioChunkPayload",
    "audio_chunk_name",
    "chunk_names",
    "is_whole_file_plan",
    "parse_audio_chunk_index",
    "stream_audio_chunks",
]
