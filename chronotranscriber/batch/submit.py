"""Batch submission of an item's pages on the store.

:func:`submit_batch` is the entry point of the LLM method and
:func:`submit_units` the submission of given units, which repair reuses.
:func:`submit_batch` takes the pages as they render, in slices of at most
:data:`SLICE_BYTES` of page images. The requests are split into parts by
the provider's count and byte limits, and every part becomes one
submission of the store: the submission and its
requests, whose custom ids are scoped to the submission id, are committed
before the upload, and each request's body as submitted (payloads redacted)
is recorded next, so finalization can store it as the request of the unit's
response row; the upload reference is recorded before the create; the
provider batch id is recorded after it. Each part is attempted once, since
a retried create could duplicate a batch.

A definite refusal marks the submission failed and deletes its upload; its
units stay pending. A create whose outcome is unknown marks the submission
unresolved and keeps its upload; its units stay blocked until ``batch
check`` adopts the provider's batch or the user abandons the submission.
Provider errors end in a :class:`SubmitOutcome` and never propagate.
"""

from __future__ import annotations

import logging
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from chronotranscriber.batch.bodies import BodyPlan, build_plan
from chronotranscriber.batch.clients import BatchClients
from chronotranscriber.batch.finalize import MISSING_TEXT
from chronotranscriber.common.batch import (
    MISSING_RESULT_ERROR,
    BatchBackend,
    BatchCreateError,
    BatchPart,
    BatchRequest,
    PartUpload,
    supports_batch,
)
from chronotranscriber.common.provider_requests import RequestTooLargeError
from chronotranscriber.common.store import (
    Lease,
    PlannedUnit,
    Store,
    Submission,
    SubmissionRejectedError,
    UnitKey,
    UnitResult,
)
from chronotranscriber.llm.prompts import system_prompt_path
from chronotranscriber.settings import Settings
from chronotranscriber.sources.images import resolve_image_settings
from chronotranscriber.spec import RunSpec

if TYPE_CHECKING:
    from chronotranscriber.sources.images import PageImage

__all__ = [
    "LABEL",
    "REQUEST_TOO_LARGE",
    "ROLE",
    "SLICE_BYTES",
    "BatchSubmissionError",
    "SubmitDocument",
    "SubmitOutcome",
    "UnitRequest",
    "fail_oversized",
    "split_oversized",
    "submit_batch",
    "submit_units",
]

logger = logging.getLogger(__name__)

ROLE = "transcription"
LABEL = "batch-transcription"
REQUEST_TOO_LARGE = "request_too_large"
"""The error type of a unit whose request exceeds a size limit of the provider."""

SubmitStatus = Literal["submitted", "refused", "unresolved"]

SLICE_BYTES: Final = 64 * 1024 * 1024
"""Bytes of base64 page images a submission holds in memory at once.

The bodies, envelopes and upload file of a part copy its images, so the
peak of a submission is a small multiple of this figure, whatever the size
of the document.
"""

_PROBE_PREFIX = "0" * 8
"""Stands in for a submission id prefix while parts are planned."""


class BatchSubmissionError(RuntimeError):
    """The requests of a submission cannot be built."""


@dataclass(frozen=True)
class SubmitOutcome:
    """How a submission ended.

    ``submitted``: every part was accepted; the units wait for ``batch
    check``. ``refused``: a part was refused; its submission is failed, its
    upload deleted and its units pending, as are those of the parts after
    it. ``unresolved``: a create got no confirmed response; that part's
    units stay blocked. *submission_ids* lists every submission opened, in
    part order; *detail* says what went wrong.
    """

    status: SubmitStatus
    submission_ids: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class UnitRequest:
    """One unit of a submission and the request that transcribes it.

    A unit without a request (its page could not be rendered) joins the
    first part's submission without being sent, so its finalization records
    the missing result as an error placeholder.
    """

    unit: UnitKey
    request: BatchRequest | None = None


class SubmitDocument(Protocol):
    """The per-document handle :func:`submit_batch` works on.

    The store must be open with a run begun and hold *lease*.
    """

    @property
    def store(self) -> Store:
        """The open store the submission is recorded in."""
        ...

    @property
    def lease(self) -> Lease:
        """The document's lease in *store*."""
        ...

    @property
    def doc_id(self) -> int:
        """The document's id in *store*."""
        ...

    @property
    def name(self) -> str:
        """The document's name."""
        ...

    @property
    def kind(self) -> str:
        """The kind of the document's units."""
        ...


def _probe_id(position: int) -> str:
    """Return a stand-in custom id as long as the store's scoped ids."""
    return f"{_PROBE_PREFIX}-{position:05d}"


def _plan_parts(
    planner: BatchBackend, units: Sequence[UnitRequest], plan: BodyPlan
) -> list[tuple[BatchPart, list[UnitRequest]]]:
    """Split the units that carry a request into parts, in order.

    The requests are planned under stand-in ids of the store's id length,
    so renaming a part's requests to its scoped ids keeps the split. Units
    without a request join the first part.
    """
    probes: dict[str, UnitRequest] = {}
    requests: list[BatchRequest] = []
    for unit in units:
        if unit.request is None:
            continue
        probe = _probe_id(len(requests))
        probes[probe] = unit
        requests.append(replace(unit.request, custom_id=probe))
    unsent = [unit for unit in units if unit.request is None]
    planned = []
    for part in planner.plan_parts(requests, plan.body):
        members = [probes[custom_id] for custom_id in part.custom_ids]
        if part.index == 1:
            members.extend(unsent)
        planned.append((part, members))
    return planned


def _scoped_part(
    store: Store,
    submission_id: str,
    part: BatchPart,
    members: Sequence[UnitRequest],
    plan: BodyPlan,
    planner: BatchBackend,
) -> BatchPart:
    """Return *part* with its requests renamed to the submission's custom ids."""
    scoped = {
        unit: custom_id
        for custom_id, unit in store.submission_requests(submission_id).items()
    }
    requests = [
        replace(member.request, custom_id=scoped[member.unit])
        for member in members
        if member.request is not None
    ]
    replanned = planner.plan_parts(requests, plan.body)
    if len(replanned) != 1:
        raise BatchSubmissionError(
            f"Batch part {part.index} split again under its scoped custom ids"
        )
    return BatchPart(part.index, part.total, requests, replanned[0].entries)


async def _submit_part(
    store: Store,
    part: BatchPart,
    members: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    planner: BatchBackend,
    clients: BatchClients,
    label: str,
    opened: list[str],
) -> SubmitOutcome | None:
    """Submit one part; return the outcome that ends the submission, if any.

    The store commits the submission before :func:`upload` runs, records
    the upload reference before :func:`create` runs and the batch id after.
    """
    provider, model = plan.provider, plan.model
    stage = "open"
    scoped: list[BatchPart] = []
    uploads: list[PartUpload] = []

    async def upload(submission: Submission) -> str | None:
        nonlocal stage
        submission_id = submission.submission_id
        opened.append(submission_id)
        stage = "upload"
        final = _scoped_part(store, submission_id, part, members, plan, planner)
        scoped.append(final)
        await store.record_requests(submission_id, planner.request_bodies(final))
        uploaded = await clients.call(
            provider,
            lambda backend: backend.upload_part(
                final, submission_id=submission_id, label=label
            ),
            model=model,
            retry=False,
            label=f"upload of batch part {part.index}",
        )
        uploads.append(uploaded)
        stage = "uploaded"
        return uploaded.file_id

    async def create(submission: Submission) -> str:
        nonlocal stage
        stage = "create"
        final, uploaded = scoped[0], uploads[0]
        try:
            handle = await clients.call(
                provider,
                lambda backend: backend.create_part(
                    final, uploaded, submission_id=submission.submission_id, label=label
                ),
                model=model,
                retry=False,
                label=f"create of batch part {part.index}",
            )
        except BatchCreateError as exc:
            if exc.definite:
                raise SubmissionRejectedError(str(exc)) from exc
            raise
        except Exception as exc:
            raise SubmissionRejectedError(str(exc)) from exc
        stage = "created"
        return handle.batch_id

    try:
        submission = await store.submit(
            provider=provider,
            model=model,
            units=[member.unit for member in members],
            upload=upload,
            create=create,
            part=part.index,
        )
    except SubmissionRejectedError as exc:
        for uploaded in uploads:
            await clients.delete_upload(uploaded, model=model)
        return SubmitOutcome("refused", tuple(opened), str(exc))
    except Exception as exc:
        if stage == "upload":
            logger.warning("Upload of batch part %d failed: %s", part.index, exc)
            return SubmitOutcome("refused", tuple(opened), f"upload failed: {exc}")
        if stage != "create":
            raise
        submission_id = opened[-1]
        logger.warning(
            "Batch submission %s has no confirmed create: %s", submission_id, exc
        )
        return SubmitOutcome(
            "unresolved",
            tuple(opened),
            f"the create of batch part {part.index} got no confirmed response "
            f"({exc}); run 'chronotranscriber batch check', which finds the "
            f"batch, or abandon submission {submission_id} with "
            f"'chronotranscriber batch check --abandon {submission_id}'",
        )
    logger.info(
        "Submitted %s batch %s (submission %s, part %d of %d, %d unit(s))",
        provider,
        submission.batch_id,
        submission.submission_id,
        part.index,
        part.total,
        len(members),
    )
    return None


def split_oversized(
    units: Sequence[UnitRequest], plan: BodyPlan
) -> tuple[list[UnitRequest], dict[UnitKey, str]]:
    """Split off the units whose request exceeds a size limit of the provider.

    Returns the other units and the error of each unit split off
    (:class:`~chronotranscriber.common.provider_requests.RequestTooLargeError`);
    those units are not submitted.
    """
    kept: list[UnitRequest] = []
    errors: dict[UnitKey, str] = {}
    for unit in units:
        if unit.request is not None:
            try:
                plan.body(unit.request)
            except RequestTooLargeError as exc:
                key = unit.unit
                logger.warning(
                    "%s unit %d of document %d: %s", key.kind, key.idx, key.doc_id, exc
                )
                errors[key] = str(exc)
                continue
        kept.append(unit)
    return kept, errors


async def fail_oversized(
    store: Store, errors: Mapping[UnitKey, str], *, model: str, provider: str
) -> None:
    """Store each unit of *errors* as failed with its error, keeping its provenance.

    The store must hold the lease of each unit's document.
    """
    for key, error in errors.items():
        await _fail(
            store,
            key,
            UnitResult(
                kind=key.kind,
                idx=key.idx,
                status="failed",
                model=model,
                provider=provider,
                content=f"[transcription error: {error}]",
                outcome="failed",
                error=error,
                error_type=REQUEST_TOO_LARGE,
            ),
        )


async def _fail(store: Store, key: UnitKey, result: UnitResult) -> None:
    """Store the failed *result* of the unit *key*, keeping its provenance."""
    row = next(row for row in store.units(key.doc_id) if row.key == key)
    await store.write_unit(
        store.held_leases()[key.doc_id],
        replace(result, provenance=row.provenance, attempts=row.attempts),
    )


async def _fail_unrendered(
    store: Store, units: Sequence[UnitRequest], *, model: str, provider: str
) -> None:
    """Store *units*, which have no request, as finalization stores such units.

    A page that could not be rendered joins a submission and fails at its
    finalization with the ``no result returned`` placeholder. The ones left
    when the last slice holds no request to send have no submission to join.
    """
    for unit in units:
        key = unit.unit
        await _fail(
            store,
            key,
            UnitResult(
                kind=key.kind,
                idx=key.idx,
                status="failed",
                model=model,
                provider=provider,
                content=MISSING_TEXT,
                outcome="failed",
                error=MISSING_RESULT_ERROR,
                error_type="missing",
            ),
        )


async def submit_units(
    store: Store,
    units: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    clients: BatchClients,
    label: str = LABEL,
) -> SubmitOutcome:
    """Submit *units* to the batch API of *plan*'s model, one submission per part.

    The units must exist in the store, which must hold the lease of each
    unit's document. The parts are submitted in order; the first refused or
    unresolved part ends the submission, and the units of the parts after
    it stay pending.

    Raises:
        BatchSubmissionError: No unit has a request, or the requests cannot
            be built.
    """
    if not any(unit.request is not None for unit in units):
        raise BatchSubmissionError("No unit of the submission has a request to send")
    planner = clients.planner(plan.provider)
    try:
        parts = _plan_parts(planner, units, plan)
    except ValueError as exc:
        raise BatchSubmissionError(f"Cannot build the batch requests: {exc}") from exc
    opened: list[str] = []
    for part, members in parts:
        ended = await _submit_part(
            store,
            part,
            members,
            plan=plan,
            planner=planner,
            clients=clients,
            label=label,
            opened=opened,
        )
        if ended is not None:
            return ended
    return SubmitOutcome("submitted", tuple(opened))


def _source_index(payload: PageImage) -> int | str:
    if isinstance(payload.page_index, int):
        return payload.page_index
    return payload.image_name


async def _record_pages(
    document: SubmitDocument, payloads: Sequence[PageImage], model: str, provider: str
) -> None:
    """Create the payloads' units and record the provenance of their images."""
    store, lease, kind = document.store, document.lease, document.kind
    await store.plan_units(
        lease,
        kind,
        [
            PlannedUnit(payload.index, payload.image_name, _source_index(payload))
            for payload in payloads
        ],
        model=model,
        provider=provider,
    )
    for payload in payloads:
        provenance = payload.provenance()
        if not provenance:
            continue
        await store.write_unit(
            lease,
            UnitResult(
                kind=kind,
                idx=payload.index,
                status="pending",
                model=model,
                provider=provider,
                label=payload.image_name,
                source_index=_source_index(payload),
                provenance=provenance,
                attempts=0,
            ),
        )


def _request(payload: PageImage) -> BatchRequest | None:
    if payload.render_error is not None:
        return None
    return BatchRequest(
        custom_id=f"page-{payload.index}",
        image_base64=payload.base64,
        mime_type=payload.mime_type,
        order_index=payload.index,
    )


async def _slices(
    payloads: Iterable[PageImage] | AsyncIterable[PageImage],
    limit: int,
    shared: int = 0,
) -> AsyncIterator[list[PageImage]]:
    """Yield the payloads in slices whose requests hold at most *limit* bytes.

    A rendered page counts with its image and the *shared* bytes every
    request carries besides (the context image). A slice closes before a
    rendered page that would take it over *limit*, so it always holds a
    rendered page when one follows; a page larger than *limit* forms a slice
    of its own. Pages that could not be rendered carry no image and join the
    slice they come in.
    """
    current: list[PageImage] = []
    held = 0
    async for payload in _each(payloads):
        rendered = payload.render_error is None
        size = len(payload.base64 or "") + shared if rendered else 0
        if held and size and held + size > limit:
            yield current
            current, held = [], 0
        current.append(payload)
        held += size
    if current:
        yield current


async def _each(
    payloads: Iterable[PageImage] | AsyncIterable[PageImage],
) -> AsyncIterator[PageImage]:
    if isinstance(payloads, AsyncIterable):
        async for payload in payloads:
            yield payload
    else:
        for payload in payloads:
            yield payload


async def submit_batch(
    payloads: Iterable[PageImage] | AsyncIterable[PageImage],
    *,
    document: SubmitDocument,
    spec: RunSpec,
    settings: Settings,
    schema_path: Path | None,
    context: str | None = None,
    context_image: Path | None = None,
    schema_text: str | None = None,
) -> SubmitOutcome:
    """Submit the rendered pages of one document to the batch API of *spec*.

    *schema_text* is the schema the run read at planning; without it the
    file *schema_path* is read.

    The pages are taken as they render, in slices whose page images hold at
    most :data:`SLICE_BYTES` (:func:`_slices`); each slice is recorded,
    split into parts and submitted, one submission per part, before the next
    one renders, so the first refused or unresolved part leaves the pages
    after it pending and unrendered. Each payload is the unit of its order
    index in *document*; its unit is created when missing and the provenance
    of its image recorded. A page that could not be rendered is submitted
    without a request (see :class:`UnitRequest`), with the next slice when
    its own holds no request to send, and is stored as failed when no slice
    follows; a page whose request exceeds a size limit of the provider is
    stored as failed with that error and not submitted.

    Raises:
        BatchSubmissionError: The provider has no batch API, no page was
            rendered or within the size limits, or the requests cannot be
            built.
    """
    model = spec.model
    if not supports_batch(model.provider):
        raise BatchSubmissionError(
            f"Provider '{model.provider}' does not support batch processing"
        )
    detail = resolve_image_settings(model.provider, model.model, spec.images)[0][
        "resolved_detail"
    ]
    try:
        plan = build_plan(
            model,
            settings,
            schema_path=schema_path,
            prompt_path=system_prompt_path(spec.prompt.prompt),
            context=context,
            context_image=context_image,
            detail=detail,
            schema_text=schema_text,
        )
    except (OSError, ValueError) as exc:
        raise BatchSubmissionError(f"Cannot build the batch requests: {exc}") from exc
    clients = BatchClients(settings, ROLE)
    opened: list[str] = []
    rendered = False
    carried: list[UnitRequest] = []
    outcome = SubmitOutcome("submitted")
    shared = len(plan.context_image[0]) if plan.context_image is not None else 0
    async for pages in _slices(payloads, SLICE_BYTES, shared):
        rendered = rendered or any(page.render_error is None for page in pages)
        await _record_pages(document, pages, model.model, model.provider)
        units = [
            UnitRequest(
                UnitKey(document.doc_id, document.kind, page.index), _request(page)
            )
            for page in pages
        ]
        del pages
        units, oversized = split_oversized(units, plan)
        await fail_oversized(
            document.store, oversized, model=model.model, provider=model.provider
        )
        units = carried + units
        if not any(unit.request is not None for unit in units):
            carried = units
            continue
        carried = []
        sent = await submit_units(document.store, units, plan=plan, clients=clients)
        del units
        opened += sent.submission_ids
        if sent.status != "submitted":
            outcome = SubmitOutcome(sent.status, tuple(opened), sent.detail)
            break
    else:
        if not rendered:
            raise BatchSubmissionError(
                f"No page of '{document.name}' could be rendered for batch submission"
            )
        if not opened:
            raise BatchSubmissionError(
                f"No page of '{document.name}' is within the provider's size limits"
            )
        await _fail_unrendered(
            document.store, carried, model=model.model, provider=model.provider
        )
        outcome = SubmitOutcome("submitted", tuple(opened))
    logger.info(
        "Batch submission of '%s' %s: %s",
        document.name,
        outcome.status,
        ", ".join(outcome.submission_ids) or "no submission",
    )
    return outcome
