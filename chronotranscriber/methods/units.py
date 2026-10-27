"""Store-backed unit pipelines: each page, image or chunk is one store row.

A :class:`DocumentHandle` is what a method needs to work one document of the
run's store: the store, the document's id and name, the run's lease on it
and the model and provider stamped on its units. It plans the expected units
as ``pending``, lists those still to do, resets units and writes one unit in
one short transaction that checks the lease.

:func:`run_unit_stream` transcribes in-memory payloads (rendered pages,
folder images, audio chunks) as a producer yields them; :func:`run_unit_files`
transcribes image files (Tesseract). For every unit both pipelines wait for
room in the store's write queue before the provider call, attribute the
call's attempts to the unit and record them through
:class:`~chronotranscriber.output.attempts.AttemptRows`, write the unit, and
only then report it to *on_unit*. After all units they raise
:class:`UnitsFailedError` when a unit failed or could not be written; a lost
lease stops the document and is raised as
:class:`~chronotranscriber.common.store.LeaseLostError`.

Unit conventions (the kinds of :mod:`chronotranscriber.output.store`):

- ``kind`` is the handle's: ``page`` for PDF pages and folder images (the
  default), ``chunk`` for audio chunks.
- ``idx`` is the unit's absolute 0-based position in its item, so runs over
  different page selections share the rows.
- ``label`` is the image or chunk name, the header the txt renderer writes.
- ``source_index`` is the 0-based PDF page index, the chunk's time range in
  seconds (``start-end``) or else the image name.
- ``status`` is ``failed`` for a unit without text or with an error
  placeholder, else ``done``; ``outcome`` is ``failed``, ``ok``, ``no_text``
  or ``not_possible``; a failed unit keeps its placeholder as ``content``.
- ``response_id`` is the selected attempt (the last one without an error,
  else the last one), whose provider and model the unit carries; ``attempts``
  counts the provider attempts of this pass, and a unit without provider
  calls counts one more attempt.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronotranscriber.common.capture import use_recorder
from chronotranscriber.common.store import (
    Lease,
    LeaseLostError,
    PlannedUnit,
    Store,
    UnitKey,
    UnitResult,
    unit_context,
)
from chronotranscriber.llm.pool import (
    run_concurrent_transcription_tasks,
    run_streaming_transcription_tasks,
)
from chronotranscriber.llm.response_parsing import detect_transcription_cause
from chronotranscriber.output.attempts import AttemptRows

__all__ = [
    "UNIT_KIND",
    "DocumentHandle",
    "FileHandler",
    "StreamHandler",
    "UnitsFailedError",
    "absolute_order",
    "run_unit_files",
    "run_unit_stream",
]

logger = logging.getLogger(__name__)

UNIT_KIND = "page"
"""The default unit kind: a PDF page or a folder image."""

# Tesseract page images of a PDF carry the 1-based page number in their name.
_TESS_PAGE_RE = re.compile(r"^page_(\d+)_tess_preprocessed\.", re.IGNORECASE)

PayloadResult = tuple[Any, str, str | None, Any, int]
"""``(payload or path, image_name, text, raw_response, order_index)``."""

StreamHandler = Callable[[Any, Any], Awaitable[PayloadResult]]
"""``(payload, transcriber) -> PayloadResult`` for one streamed unit."""

FileHandler = Callable[[Path, int], Awaitable[PayloadResult]]
"""``(image file, order index) -> PayloadResult`` for one file unit."""

UnitCallback = Callable[[int, bool], None]
"""``(index, failed)`` after each unit's row is committed."""


class UnitsFailedError(Exception):
    """One or more units of a document failed or could not be written.

    Raised after every unit was processed; the units written stay in the
    store for a later resume or repair. *write_errors* are the errors of
    the unit writes that failed; the first is the exception's cause.
    """

    def __init__(
        self,
        source_name: str,
        failed_pages: int,
        total_pages: int,
        write_errors: Sequence[BaseException] = (),
    ) -> None:
        self.source_name = source_name
        self.failed_pages = failed_pages
        self.total_pages = total_pages
        self.write_errors = list(write_errors)
        message = (
            f"{failed_pages} of {total_pages} page(s) failed to transcribe "
            f"for '{source_name}'"
        )
        if self.write_errors:
            first = self.write_errors[0]
            message += (
                f"; {len(self.write_errors)} could not be stored: "
                f"{type(first).__name__}: {first}"
            )
        super().__init__(message)


@dataclass(frozen=True)
class DocumentHandle:
    """One document of the run's store, worked under the run's lease.

    *name* is the document's unique name (the item name), *model* and
    *provider* the stamps of its units when no attempt names others, and
    *kind* the unit kind. The run core builds the handle after acquiring
    the lease; the lease keeper renews it while the handle is in use.
    """

    store: Store
    doc_id: int
    name: str
    lease: Lease
    model: str
    provider: str
    kind: str = UNIT_KIND

    @property
    def tool(self) -> str:
        """Return the tool name the store stamps on every row."""
        return self.store.tool

    @property
    def run_id(self) -> str | None:
        """Return the id of the run the store records."""
        return self.store.run_id

    def key(self, idx: int) -> UnitKey:
        """Return the key of unit *idx*."""
        return UnitKey(self.doc_id, self.kind, idx)

    async def plan(self, units: Sequence[PlannedUnit]) -> int:
        """Create the expected units as ``pending``; return the number created."""
        return await self.store.plan_units(
            self.lease, self.kind, units, model=self.model, provider=self.provider
        )

    def todo(self, *, retry_errors: bool = False) -> list[int]:
        """Return the indices of the pending units no open batch holds.

        With *retry_errors* the failed units are included.
        """
        state = self.store.resume_state(self.doc_id)
        if state is None:
            return []
        blocked = set(state.blocked)
        keys = list(state.ready)
        if retry_errors:
            keys.extend(key for key in state.failed if key not in blocked)
        return sorted(key.idx for key in keys if key.kind == self.kind)

    async def reset(
        self,
        idxs: Sequence[int] | None = None,
        *,
        statuses: Sequence[str] | None = None,
    ) -> int:
        """Return units to ``pending``, keeping their responses; return the count.

        None selects every unit of the kind, or every status.
        """
        return await self.store.reset_units(
            self.lease, kind=self.kind, idxs=idxs, statuses=statuses
        )

    async def discard(self, count: int | None = None) -> int:
        """Delete units, keeping their responses; return the count deleted.

        Deletes the units of other kinds and, with *count*, those of this
        kind from index *count* on.

        Raises:
            SubmissionStateError: An open or unresolved batch submission
                holds one of them.
        """
        return await self.store.delete_units(self.lease, self.kind, count=count)

    async def write(  # noqa: PLR0913 - mirrors the UnitResult fields; tests call it
        self,
        idx: int,
        *,
        status: str,
        content: str | None,
        outcome: str | None = None,
        label: str | None = None,
        source_index: int | str | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        error_type: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        response_id: int | None = None,
        attempts: int | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Commit unit *idx* in one transaction that checks the lease.

        Raises:
            LeaseLostError: The lease expired or passed to another holder.
            sqlite3.Error: The write failed (a busy database only delays it).
        """
        await self.store.write_unit(
            self.lease,
            UnitResult(
                kind=self.kind,
                idx=idx,
                status=status,
                model=model or self.model,
                provider=provider or self.provider,
                content=content,
                outcome=outcome,
                label=label,
                source_index=source_index,
                usage=usage,
                error=error,
                error_type=error_type,
                provenance=provenance,
                response_id=response_id,
                attempts=attempts,
            ),
        )


def absolute_order(
    files: Sequence[Path], order: Mapping[str, int] | None = None
) -> dict[str, int]:
    """Map each image name to its absolute 0-based position in the item.

    PDF Tesseract images carry their page number in the name; folder images
    take their position in the full listing from *order*; other names fall
    back to their position in *files*.
    """
    override = order or {}
    mapping: dict[str, int] = {}
    for position, path in enumerate(files):
        match = _TESS_PAGE_RE.match(path.name)
        if match:
            mapping[path.name] = int(match.group(1)) - 1
        elif path.name in override:
            mapping[path.name] = override[path.name]
        else:
            mapping[path.name] = position
    return mapping


def _error_text(label: str) -> str:
    return f"[transcription error: {label}]"


def _source_index(payload: Any, label: str) -> int | str:
    page_index = getattr(payload, "page_index", None)
    if isinstance(page_index, int):
        return page_index
    start = getattr(payload, "start_seconds", None)
    duration = getattr(payload, "duration_seconds", None)
    if isinstance(start, int | float) and isinstance(duration, int | float):
        return f"{start:.3f}-{start + duration:.3f}"
    return label


def _provenance(payload: Any) -> Mapping[str, Any] | None:
    provenance = getattr(payload, "provenance", None)
    if not callable(provenance):
        return None
    value = provenance()
    return value if isinstance(value, Mapping) and value else None


class _Units:
    """The per-unit steps the pipelines share, and their tally."""

    def __init__(self, handle: DocumentHandle, on_unit: UnitCallback | None) -> None:
        self.handle = handle
        self.on_unit = on_unit
        self.total = 0
        self.failed = 0
        self.write_errors: list[BaseException] = []
        self.lost: LeaseLostError | None = None

    async def run(
        self,
        idx: int,
        label: str,
        call: Callable[[], Awaitable[PayloadResult]],
        *,
        source_index: int | str,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Transcribe one unit through *call*, write it, then report it."""
        handle = self.handle
        if self.lost is not None:
            return
        await handle.store.wait_capacity()
        if self.lost is not None:
            return
        recorder = AttemptRows(
            handle.store, provider=handle.provider, model=handle.model
        )
        result: PayloadResult | None = None
        raised: BaseException | None = None
        with unit_context(handle.key(idx)), use_recorder(recorder):
            try:
                result = await call()
            except Exception as exc:
                logger.exception("Error transcribing %s: %s", label, exc)
                raised = exc
        self.total += 1
        try:
            if recorder.failures:
                raise recorder.failures[0]
            failed = await self._write(
                idx, label, result, raised, recorder, source_index, provenance
            )
        except LeaseLostError as exc:
            logger.error(
                "Stopping '%s': unit %d was not stored: %s", handle.name, idx, exc
            )
            self.lost = exc
            self.failed += 1
            return
        except Exception as exc:  # noqa: BLE001 - tallied, raised after all units
            logger.error(
                "Unit %d of '%s' could not be stored in %s: %s",
                idx,
                handle.name,
                handle.store.path,
                exc,
            )
            self.write_errors.append(exc)
            self.failed += 1
            return
        if failed:
            self.failed += 1
        if self.on_unit is not None:
            self.on_unit(idx, failed)

    async def _write(
        self,
        idx: int,
        label: str,
        result: PayloadResult | None,
        raised: BaseException | None,
        recorder: AttemptRows,
        source_index: int | str,
        provenance: Mapping[str, Any] | None,
    ) -> bool:
        """Write the unit's row; return whether the unit failed."""
        text: str | None = None
        raw: Any = None
        if result is not None and len(result) >= 5:
            label = result[1] or label
            text, raw = result[2], result[3]
        content = text if text is not None else _error_text(label)
        cause = detect_transcription_cause(content)
        failed = text is None or cause == "api_error"
        outcome = "failed" if failed else cause
        raw_dict = raw if isinstance(raw, Mapping) else {}
        selected = recorder.selected()
        error: str | None = None
        error_type: str | None = None
        if raised is not None:
            error = str(raised) or type(raised).__name__
            error_type = type(raised).__name__
        elif failed:
            last = recorder.stored[-1].record if recorder.stored else None
            error = raw_dict.get("error") or (last.error if last else None) or content
            error_type = raw_dict.get("error_type") or (
                last.error_type if last else None
            )
        usage = raw_dict.get("usage")
        await self.handle.write(
            idx,
            status="failed" if failed else "done",
            content=content,
            outcome=outcome,
            label=label,
            source_index=source_index,
            usage=usage if isinstance(usage, Mapping) else None,
            error=error,
            error_type=error_type,
            provenance=provenance,
            response_id=selected.response_id if selected else None,
            attempts=len(recorder.stored) or None,
            model=selected.record.model if selected else None,
            provider=selected.record.provider if selected else None,
        )
        return failed

    def finish(self) -> None:
        """Raise the lost lease or the item failure, if any."""
        if self.lost is not None:
            raise self.lost
        if self.failed:
            cause = self.write_errors[0] if self.write_errors else None
            raise UnitsFailedError(
                self.handle.name, self.failed, self.total, self.write_errors
            ) from cause


async def run_unit_stream(
    handle: DocumentHandle,
    payloads: AsyncIterator[Any],
    transcriber: Any,
    handler: StreamHandler,
    *,
    concurrency: int = 20,
    delay: float = 0.0,
    on_unit: UnitCallback | None = None,
) -> int:
    """Transcribe streamed payloads into the document's units.

    A pool of *concurrency* workers runs *handler* on each payload as the
    producer yields it, each call *delay* seconds after the worker's
    previous one. A payload carries ``index`` and ``image_name``; the
    producer leaves out the units already done. Returns the number of
    units processed.

    Raises:
        UnitsFailedError: A unit failed or could not be written.
        LeaseLostError: The lease was lost; the remaining units were skipped.
    """
    units = _Units(handle, on_unit)

    async def handle_payload(payload: Any) -> None:
        label = str(payload.image_name)
        await units.run(
            int(payload.index),
            label,
            lambda: handler(payload, transcriber),
            source_index=_source_index(payload, label),
            provenance=_provenance(payload),
        )

    await run_streaming_transcription_tasks(
        payloads, handle_payload, concurrency, delay
    )
    units.finish()
    return units.total


async def run_unit_files(
    handle: DocumentHandle,
    files: Sequence[Path],
    handler: FileHandler,
    *,
    order: Mapping[str, int] | None = None,
    concurrency: int = 20,
    delay: float = 0.0,
    on_unit: UnitCallback | None = None,
) -> int:
    """Transcribe image files into the document's units.

    Each file's unit index is its absolute position (:func:`absolute_order`
    with *order*); the caller passes only the files still to do. Returns
    the number of units processed.

    Raises:
        UnitsFailedError: A unit failed or could not be written.
        LeaseLostError: The lease was lost; the remaining units were skipped.
    """
    units = _Units(handle, on_unit)
    positions = absolute_order(files, order)

    async def handle_file(path: Path, idx: int) -> None:
        page = _TESS_PAGE_RE.match(path.name) is not None
        await units.run(
            idx,
            path.name,
            lambda: handler(path, idx),
            source_index=idx if page else path.name,
        )

    await run_concurrent_transcription_tasks(
        handle_file,
        [(path, positions[path.name]) for path in files],
        concurrency,
        delay,
    )
    units.finish()
    return units.total
