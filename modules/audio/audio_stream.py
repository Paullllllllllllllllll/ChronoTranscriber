"""Streaming chunk-payload producer for the audio transcription pipeline.

The audio analogue of :mod:`modules.images.page_stream`: it turns a planned
list of :class:`~modules.audio.chunker.ChunkSpec` segments into payloads the
transcription pipeline can submit, materializing each segment with ffmpeg only
when it is about to be sent. A recording that fits a single request is yielded
without any ffmpeg pass at all — the payload points straight at the source file.

Chunks are named ``{stem}_chunk_NNNN{suffix}`` and recorded under that name in
the temporary JSONL, so :func:`compute_audio_skip_indices` can rebuild the
resume skip set from the same records the image pipeline uses.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.audio.chunker import ChunkSpec
from modules.audio.constants import SUPPORTED_AUDIO_FORMATS
from modules.audio.ffmpeg_runtime import cut_segment
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

_AUDIO_CHUNK_NAME_RE = re.compile(r"_chunk_(\d{4,})\.[^.]+$", re.IGNORECASE)

_DEFAULT_AUDIO_MIME = "application/octet-stream"


@dataclass
class AudioChunkPayload:
    """One audio segment ready for a speech-to-text request.

    ``index`` is the absolute 0-based chunk index within the plan, so chunk
    numbering stays correct under resume. ``image_name`` carries the pipeline's
    generic per-unit name (the JSONL schema is shared with the image workflow),
    which for audio is the chunk file name.
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
        """Per-chunk reproducibility record for JSONL persistence.

        Mirrors :meth:`modules.images.page_stream.PagePayload.provenance`: a
        flat dict of the facts needed to identify exactly which bytes were
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


def compute_audio_skip_indices(
    jsonl_path: Path, *, exclude_errors: bool = False
) -> set[int]:
    """0-based chunk indices already transcribed according to the temp JSONL.

    The audio analogue of
    :func:`modules.images.page_stream.compute_pdf_skip_indices`, including its
    ``exclude_errors`` semantics: when True, chunks whose latest record is a
    ``[transcription error]`` placeholder are left out of the skip set so
    ``--retry-errors`` re-submits them.

    The plan must be unchanged for these indices to mean anything; see the
    resume caveat in :mod:`modules.audio.chunker`.
    """
    from modules.batch.jsonl import get_processed_image_names

    skip: set[int] = set()
    for name in get_processed_image_names(jsonl_path, exclude_errors=exclude_errors):
        index = parse_audio_chunk_index(name)
        if index is not None:
            skip.add(index)
    return skip


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


def _is_whole_file_plan(specs: list[ChunkSpec]) -> bool:
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
    work_dir: Path,
    chunk_format: str,
    mono: bool,
    sample_rate: int,
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
        specs: Plan from :func:`modules.audio.chunker.plan_chunks`.
        work_dir: Directory the chunk files are written into.
        chunk_format: ``mp3``, ``wav16``, or ``copy``.
        mono: Downmix to a single channel.
        sample_rate: Output sample rate in Hz.
        skip_indices: Chunk indices already transcribed (resume).
    """
    skip = skip_indices or set()

    if _is_whole_file_plan(specs):
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

    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = _chunk_suffix(source, chunk_format)

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
        except Exception as exc:
            logger.error(
                "Error cutting chunk %d of %s at %.1fs: %s",
                spec.index + 1,
                source.name,
                spec.start_seconds,
                exc,
            )
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


__all__ = [
    "AudioChunkPayload",
    "audio_chunk_name",
    "compute_audio_skip_indices",
    "parse_audio_chunk_index",
    "stream_audio_chunks",
]
