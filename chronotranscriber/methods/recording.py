"""The flow of one audio recording, shared by the Whisper and audio-API methods.

The recording is planned into chunks (:mod:`chronotranscriber.sources.audio`),
and every chunk of the plan is one ``chunk`` unit of the item's store
document, created as ``pending`` before the first call. The document records
the plan's signature; chunk indices are positional, so a changed plan deletes
the chunk units beyond it and returns the others to ``pending`` (their
responses stay). The selected chunks
still to do are cut into the item's working directory and transcribed
through the streaming pipeline; the run core renders the item's files.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from chronotranscriber.common.store import PlannedUnit
from chronotranscriber.methods import Item, Job, NothingSelectedError, PageCounter
from chronotranscriber.methods.tesseract import store_document, units_to_do
from chronotranscriber.methods.units import (
    DocumentHandle,
    StreamHandler,
    run_unit_stream,
)
from chronotranscriber.output.store import KIND_CHUNK
from chronotranscriber.sources.audio import (
    ChunkPlan,
    ChunkSettings,
    chunk_settings,
    plan_recording,
)
from chronotranscriber.sources.audio.chunker import ChunkSpec, chunk_plan_signature
from chronotranscriber.sources.audio.stream import (
    chunk_names,
    is_whole_file_plan,
    stream_audio_chunks,
)

__all__ = ["CHUNK_FOLDER", "SIGNATURE_KEY", "recording_record", "transcribe_recording"]

logger = logging.getLogger(__name__)

CHUNK_FOLDER = "audio_chunks"

SIGNATURE_KEY = "chunk_plan_signature"
"""The key of the plan signature in the document's recorded settings."""


def recording_record(
    path: Path,
    method: str,
    venue: tuple[str, str],
    settings: ChunkSettings,
    specs: list[ChunkSpec],
) -> dict[str, Any]:
    """Return the recorded settings of one recording, plan signature included.

    *venue* is the ``(provider, model)`` that transcribes the chunks. The
    store keeps the record as the document's settings record.
    """
    provider, model = venue
    return {
        "source_file": str(path),
        "source_bytes": path.stat().st_size,
        "method": method,
        "provider": provider,
        "model": model,
        SIGNATURE_KEY: chunk_plan_signature(specs),
        "chunk_count": len(specs),
        "chunking": {
            "target_seconds": settings.target_seconds,
            "overlap_seconds": settings.overlap_seconds,
            "max_request_bytes": settings.max_request_bytes,
            "chunk_format": settings.chunk_format,
            "mono": settings.mono,
            "sample_rate": settings.sample_rate,
        },
    }


def _plan(job: Job, item: Item, settings: ChunkSettings) -> ChunkPlan:
    plan = plan_recording(
        item.path, item.method, settings, page_range_active=item.pages is not None
    )
    if plan.warning is not None:
        job.events.warning(plan.warning)
    if plan.note is not None:
        logger.info("%s: %s", item.name, plan.note)
    return plan


def _selected(job: Job, item: Item, plan: ChunkPlan) -> list[ChunkSpec]:
    """Return the chunks the page range selects.

    Raises:
        NothingSelectedError: The range selects no chunk.
    """
    page_range = item.pages
    specs = plan.specs
    if page_range is None:
        return specs
    if not plan.planned:
        job.events.warning(
            "Page ranges select audio chunks and require FFmpeg; "
            f"transcribing all of '{item.name}' instead."
        )
        return specs
    keep = set(page_range.resolve(len(specs)))
    if not keep:
        raise NothingSelectedError(
            f"Page range '{page_range.describe()}' selected no chunks for "
            f"'{item.name}' ({len(specs)} chunk(s)). Skipping."
        )
    return [spec for spec in specs if spec.index in keep]


def _stored_signature(handle: DocumentHandle) -> str | None:
    document = handle.store.document(handle.doc_id)
    record = document.image_settings if document is not None else None
    if not isinstance(record, Mapping):
        return None
    signature = record.get(SIGNATURE_KEY)
    return signature if isinstance(signature, str) and signature else None


async def _reset_changed_plan(
    job: Job, handle: DocumentHandle, name: str, signature: str, count: int
) -> None:
    """Replace the chunk units of a changed plan of *count* chunks.

    A plan with other boundaries describes other audio: the units beyond
    the new plan are deleted and the others return to ``pending`` with a
    warning, so the recording is transcribed again. The responses of the
    earlier chunks stay. ``--force`` replaces the plan in one transaction
    instead (:func:`~chronotranscriber.methods.tesseract.units_to_do`).
    """
    stored = _stored_signature(handle)
    if stored is None or stored == signature:
        return
    await handle.discard(count)
    earlier = handle.store.units(handle.doc_id, kind=handle.kind)
    if not any(unit.status != "pending" for unit in earlier):
        return
    job.events.warning(
        f"The chunk plan for '{name}' changed since the last run "
        f"({stored} -> {signature}); discarding the stale chunk records "
        f"and re-transcribing the recording."
    )
    await handle.reset()


async def transcribe_recording(
    job: Job,
    item: Item,
    transcriber: Any,
    handler: Callable[[], StreamHandler | None],
    *,
    provider: str,
    model: str,
) -> list[Path]:
    """Transcribe one recording into its chunk units; return [].

    *handler* returns the per-chunk handler and raises when the method's
    runtime is missing; it is called only when chunks remain to be
    transcribed, and a None return skips the item. *provider* and *model*
    name what transcribes the chunks; the units and the document's record
    carry them.

    Raises:
        NothingSelectedError: The page range selects no chunk.
        RecordingSplitError: The recording must be cut but cannot be.
        UnitsFailedError: A chunk failed or could not be stored.
        LeaseLostError: The document's lease was lost.
    """
    settings = chunk_settings(job.spec, job.settings)
    plan = _plan(job, item, settings)
    selected = _selected(job, item, plan)

    handle = store_document(item, KIND_CHUNK, provider=provider, model=model)
    record = recording_record(
        item.path, item.method, (provider, model), settings, plan.specs
    )
    if not job.force:
        await _reset_changed_plan(
            job, handle, item.name, record[SIGNATURE_KEY], len(plan.specs)
        )
        await handle.store.set_image_settings(handle.lease, record)
    names = chunk_names(item.path, plan.specs, settings.chunk_format)
    units = [
        PlannedUnit(spec.index, label=names[n]) for n, spec in enumerate(plan.specs)
    ]
    todo = set(
        await units_to_do(
            job, handle, [spec.index for spec in selected], units, image_settings=record
        )
    )
    if not todo:
        return []

    chosen = handler()
    if chosen is None:
        return []

    counter = PageCounter(
        job.events,
        item.name,
        total=len(todo),
        unit="chunk",
        already_complete=len(selected) - len(todo),
    )
    # A whole-file plan sends the source itself, so it needs no working files.
    cut = not is_whole_file_plan(plan.specs)
    working = job.working_dir(item.path) if cut else contextlib.nullcontext(None)
    with working as base:
        await run_unit_stream(
            handle,
            stream_audio_chunks(
                item.path,
                specs=selected,
                chunk_format=settings.chunk_format,
                mono=settings.mono,
                sample_rate=settings.sample_rate,
                work_dir=base / CHUNK_FOLDER if base is not None else None,
                skip_indices={spec.index for spec in selected} - todo,
            ),
            transcriber,
            chosen,
            concurrency=job.settings.concurrency.audio,
            delay=job.settings.concurrency.delay_between_tasks,
            on_unit=counter,
        )
    return []
