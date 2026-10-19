"""Finalize submitted batches.

:func:`finalize_submission` finalizes one submission of the store, for batch
check and repair: it asks the provider for the batch status and, once the
batch has finished, downloads the results. Every result record becomes one
``responses`` row holding the record as downloaded; then every unit of the
submission is written in a transaction that checks its document's lease: a
success as the page's transcription, a failed request (a refusal, a content
filter, an answer cut at the output-token limit) as an error placeholder
and a unit without a result as the ``no result returned``
placeholder; an answer of a run that asked for JSON fails when it holds no
JSON object, as in a synchronous run. Last, the submission is marked
finalized, its counts taken from
the units written (:func:`~chronotranscriber.common.batch.outcome_counts`).
A batch that ended
without results marks the submission failed, so its units are pending
again. A provider error that outlasts the retries of :class:`BatchClients`
ends in an outcome and changes nothing. Each response row holds as its
request the body kept for the record's custom id when it was submitted.
:func:`delete_provider_files` deletes the batch's input and result files once
the caller has rendered the documents.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chronotranscriber.batch.clients import (
    BatchClients,
    MissingKeyError,
    results_of,
    status_of,
)
from chronotranscriber.common.batch import (
    MISSING_RESULT_ERROR,
    BatchHandle,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    outcome_counts,
    supports_batch,
)
from chronotranscriber.common.retry import ErrorKind, classify_error
from chronotranscriber.common.store import (
    LeaseLostError,
    ResponseRow,
    Store,
    StoreError,
    Submission,
    UnitKey,
    UnitResult,
    UnitRow,
)
from chronotranscriber.common.structured import parse_json_object
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.response_parsing import (
    detect_transcription_cause,
    extract_transcribed_text,
)

__all__ = [
    "FAILED",
    "FINALIZED_OUTCOME",
    "MISSING_TEXT",
    "PENDING",
    "SKIPPED",
    "SubmissionFinalization",
    "delete_provider_files",
    "finalize_submission",
    "placeholder",
    "submission_handle",
]

logger = logging.getLogger(__name__)

FINALIZED_OUTCOME = "finalized"
PENDING = "pending"
FAILED = "failed"
SKIPPED = "skipped"

MISSING_TEXT = "[transcription error: no result returned]"
NO_OBJECT_TEXT = "[transcription error: answer holds no JSON object]"
INVALID_JSON = "invalid_json"
"""The error type of a page whose JSON-format answer holds no JSON object."""
_RETRYABLE = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVER_ERROR,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
    }
)


def placeholder(provider: str, item: BatchResultItem) -> str:
    """Return the error placeholder of a failed result.

    OpenAI results name the HTTP status of an error in the output file and
    the error code; the other providers give their error message.
    """
    if item.origin == "missing":
        return MISSING_TEXT
    if provider != "openai":
        return f"[transcription error: {item.error or 'unknown error'}]"
    parts = ["[transcription error"]
    status = item.response.get("status_code") if item.origin == "output" else None
    if isinstance(status, int) and status != 200:
        parts.append(f"status {status}")
    if item.error_code:
        parts.append(f"code {item.error_code}")
    if item.error and item.error != f"HTTP {status}":
        parts.append(item.error)
    return "; ".join(parts) + "]"


def _page_text(
    provider: str, item: BatchResultItem, label: str | None
) -> tuple[str, bool]:
    """Return a result's page text and whether it is an error placeholder."""
    if not item.success:
        return placeholder(provider, item), True
    text = extract_transcribed_text({"output_text": item.text}, label or item.custom_id)
    return text, detect_transcription_cause(text) == "api_error"


def _usage_of(items: Sequence[BatchResultItem]) -> Usage:
    total = Usage()
    for item in items:
        total = total + item.usage
    return total


def _error_suffix(info: BatchStatusInfo) -> str:
    return f": {info.error_message}" if info.error_message else ""


@dataclass(frozen=True)
class SubmissionFinalization:
    """The outcome of finalizing one submission of the store.

    *outcome* is ``finalized``, ``pending`` (still running, or a retryable
    provider error), ``failed`` (the batch ended without results, or a
    provider error that is not retryable) or ``skipped`` (the submission is
    not submitted); *detail* says why. *documents* are the ids of the
    documents whose units were written and *errors* the number of units
    written as error placeholders.
    """

    submission_id: str
    outcome: str
    detail: str | None = None
    provider_status: str | None = None
    documents: tuple[int, ...] = ()
    units: int = 0
    errors: int = 0
    usage: Usage = Usage()


def submission_handle(
    submission: Submission, custom_ids: Iterable[str] = ()
) -> BatchHandle:
    """Return the backend handle of *submission*'s provider batch.

    The upload reference becomes the input file the backend deletes in its
    cleanup.
    """
    metadata: dict[str, Any] = {}
    if submission.upload_ref:
        key = "input_file_name" if submission.provider == "google" else "input_file_id"
        metadata[key] = submission.upload_ref
    return BatchHandle(
        provider=submission.provider,
        batch_id=submission.batch_id or "",
        custom_ids=list(custom_ids),
        metadata=metadata,
    )


def _status_label(info: BatchStatusInfo) -> str:
    return info.provider_status or info.status.value


def _usage_record(usage: Usage) -> dict[str, int]:
    """Return *usage* in the shape every stored usage record has."""
    return usage.as_dict()


def _http_status(item: BatchResultItem) -> int | None:
    status = item.response.get("status_code")
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _missing(custom_id: str) -> BatchResultItem:
    return BatchResultItem(
        custom_id=custom_id,
        error=MISSING_RESULT_ERROR,
        error_code="missing",
        origin="missing",
    )


async def _append_records(
    store: Store,
    submission: Submission,
    requests: Mapping[str, UnitKey],
    items: Sequence[BatchResultItem],
    role: str,
) -> tuple[dict[str, BatchResultItem], dict[str, int]]:
    """Store every result record as one response row, in provider order.

    A record without a custom id (a malformed result line) is stored with
    neither a custom id nor a unit and is never attributed to a unit. A
    record's request is the body kept for its custom id at submission.
    Returns the first result of each requested custom id and the response
    id of each stored record.
    """
    results: dict[str, BatchResultItem] = {}
    response_ids: dict[str, int] = {}
    bodies = store.submission_bodies(submission.submission_id)
    for item in items:
        if not item.custom_id:
            if item.raw is not None:
                logger.warning(
                    "Batch %s of submission %s returned a record without a "
                    "custom id; stored without a unit: %s",
                    submission.batch_id,
                    submission.submission_id,
                    item.error or "no custom id",
                )
                await store.append_response(
                    ResponseRow(
                        provider=submission.provider,
                        model=submission.model,
                        role=role,
                        attempt=1,
                        submission_id=submission.submission_id,
                        raw=item.raw,
                        raw_kind=item.raw_kind,
                        error=item.error,
                        error_type=item.error_code,
                    )
                )
            continue
        unit = requests.get(item.custom_id)
        if unit is None:
            logger.warning(
                "Batch %s returned %r, which submission %s did not request",
                submission.batch_id,
                item.custom_id,
                submission.submission_id,
            )
        if item.raw is not None:
            response_ids.setdefault(
                item.custom_id,
                await store.append_response(
                    ResponseRow(
                        provider=submission.provider,
                        model=submission.model,
                        unit=unit,
                        role=role,
                        attempt=1,
                        submission_id=submission.submission_id,
                        custom_id=item.custom_id,
                        request=bodies.get(item.custom_id),
                        raw=item.raw,
                        raw_kind=item.raw_kind,
                        usage=_usage_record(item.usage),
                        error=item.error,
                        error_type=item.error_code,
                        http_status=_http_status(item),
                    )
                ),
            )
        if unit is not None:
            results.setdefault(item.custom_id, item)
    return results, response_ids


def _expects_object(store: Store, run_id: str) -> bool:
    """Whether the run that submitted a batch asked for JSON answers.

    A ``transcribe`` run stores ``model.response_format`` in its spec: every
    value but ``text`` (unset included, which resolves to schema or json)
    asks for a JSON object. A run that stored no such field (a repair) is
    not checked, as before.
    """
    try:
        row = store.run(run_id)
    except (StoreError, sqlite3.Error):
        return False
    raw: Any = row.get("spec") if row else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return False
    entry = raw.get("model.response_format") if isinstance(raw, Mapping) else None
    if not isinstance(entry, Mapping):
        return False
    return entry.get("value") != "text"


async def _write_units(
    store: Store,
    submission: Submission,
    requests: Mapping[str, UnitKey],
    results: Mapping[str, BatchResultItem],
    response_ids: Mapping[str, int],
) -> tuple[int, Usage]:
    """Write every requested unit; return the error count and the usage."""
    doc_ids = sorted({key.doc_id for key in requests.values()})
    held = store.held_leases()
    for doc_id in doc_ids:
        if doc_id not in held:
            raise LeaseLostError(f"Document {doc_id} is not leased by this run.")
    rows: dict[UnitKey, UnitRow] = {
        row.key: row for doc_id in doc_ids for row in store.units(doc_id)
    }
    errors = 0
    usage = Usage()
    expects_object = _expects_object(store, submission.run_id)
    for custom_id, key in requests.items():
        item = results.get(custom_id) or _missing(custom_id)
        row = rows.get(key)
        label = row.label if row is not None else None
        text, error = _page_text(submission.provider, item, label)
        error_type = item.error_code
        answer = item.text or ""
        if (
            not error
            and expects_object
            and answer.strip()
            and parse_json_object(answer) is None
        ):
            text, error, error_type = NO_OBJECT_TEXT, True, INVALID_JSON
        cause = detect_transcription_cause(text)
        failed = error or cause == "api_error"
        errors += failed
        usage = usage + item.usage
        await store.write_unit(
            held[key.doc_id],
            UnitResult(
                kind=key.kind,
                idx=key.idx,
                status="failed" if failed else "done",
                model=submission.model,
                provider=submission.provider,
                content=text,
                outcome="failed" if failed else cause,
                label=label,
                usage=_usage_record(item.usage),
                error=(item.error or text) if failed else None,
                error_type=error_type if failed else None,
                provenance=row.provenance if row is not None else None,
                response_id=response_ids.get(custom_id),
                attempts=1,
            ),
        )
    return errors, usage


def _submission_failure(
    submission_id: str, exc: Exception, what: str
) -> SubmissionFinalization:
    """Return the outcome of a provider call that failed after its retries.

    A missing key and a non-retryable error fail the submission; a
    retryable error leaves it pending for the next check.
    """
    if isinstance(exc, MissingKeyError):
        return SubmissionFinalization(submission_id, FAILED, str(exc))
    outcome = PENDING if classify_error(exc) in _RETRYABLE else FAILED
    logger.warning("%s: %s: %s", what, type(exc).__name__, exc)
    return SubmissionFinalization(submission_id, outcome, f"{what}: {exc}")


async def finalize_submission(
    store: Store, submission: Submission, clients: BatchClients
) -> SubmissionFinalization:
    """Finalize *submission* from its provider batch into the store.

    The store must hold the lease of every document the submission covers.
    Nothing is written until the batch has finished and its results are
    downloaded, apart from the provider status of each check.

    Raises:
        LeaseLostError: A document's lease was lost before its units were
            written; the submission stays submitted for the next check.
    """
    submission_id = submission.submission_id
    if submission.status != "submitted" or not submission.batch_id:
        return SubmissionFinalization(
            submission_id, SKIPPED, f"submission is {submission.status}"
        )
    provider = submission.provider
    if not supports_batch(provider):
        return SubmissionFinalization(
            submission_id, FAILED, f"provider {provider!r} has no batch API"
        )
    requests = store.submission_requests(submission_id)
    handle = submission_handle(submission, requests)
    try:
        info = await clients.call(
            provider,
            status_of(handle),
            model=submission.model,
            label=f"status of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - classified by _submission_failure
        return _submission_failure(
            submission_id, exc, f"status of batch {handle.batch_id} not retrieved"
        )
    status = _status_label(info)
    await store.record_check(
        submission_id, provider_status=status, counts=info.counts()
    )
    if not info.is_terminal:
        detail = f"batch {handle.batch_id} {status}{_error_suffix(info)}"
        return SubmissionFinalization(submission_id, PENDING, detail, status)
    if info.status is not BatchStatus.COMPLETED and not info.results_available:
        reason = f"batch {handle.batch_id} {info.status.value}{_error_suffix(info)}"
        await store.mark_failed(submission_id, reason)
        return SubmissionFinalization(submission_id, FAILED, reason, status)
    try:
        items = await clients.call(
            provider,
            results_of(handle),
            usage=_usage_of,
            model=submission.model,
            label=f"results of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - classified by _submission_failure
        return _submission_failure(
            submission_id, exc, f"results of batch {handle.batch_id} not downloaded"
        )
    results, response_ids = await _append_records(
        store, submission, requests, items, clients.role
    )
    errors, usage = await _write_units(
        store, submission, requests, results, response_ids
    )
    await store.mark_finalized(
        submission_id,
        provider_status=status,
        counts=outcome_counts(len(requests), errors),
    )
    documents = tuple(sorted({key.doc_id for key in requests.values()}))
    logger.info(
        "Finalized submission %s from batch %s: %d unit(s), %d error(s)",
        submission_id,
        handle.batch_id,
        len(requests),
        errors,
    )
    return SubmissionFinalization(
        submission_id,
        FINALIZED_OUTCOME,
        None,
        status,
        documents,
        len(requests),
        errors,
        usage,
    )


async def delete_provider_files(clients: BatchClients, submission: Submission) -> None:
    """Delete the input and result files of *submission*'s batch.

    Call it after the results are stored and the documents rendered; the
    store then holds every response. A failure is logged.
    """
    if not submission.batch_id:
        return
    await clients.cleanup(submission_handle(submission), model=submission.model)
