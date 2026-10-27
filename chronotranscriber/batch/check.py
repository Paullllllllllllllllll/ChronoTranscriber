"""Batch check: finalize the batch submissions below a folder.

:func:`check_submissions` finds every ``chronotranscriber.sqlite`` below a
folder and works its open submissions. A submission left ``submitting``
without a provider batch id is marked unresolved first, and every
unresolved submission is reconciled with the provider's batch list (see
:mod:`chronotranscriber.common.batch_reconcile`); ``--adopt`` and ``--abandon``
settle one by hand. Each submitted submission is checked and, once its
batch has finished, finalized into the store
(:func:`~chronotranscriber.batch.finalize.finalize_submission`) under the
leases of its documents. The finalized documents are then rendered through
:mod:`chronotranscriber.output.render` while the leases are held, in the
formats of the run that submitted the batch unless the check names others,
and the batch's input and result files are deleted from the provider.

:func:`check_until_settled` repeats the check until no batch work is pending
or a deadline has passed; :func:`final_totals` then counts every submission
the passes saw by its state in the store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from chronotranscriber.batch.clients import BatchClients
from chronotranscriber.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    PENDING,
    SKIPPED,
    SubmissionFinalization,
    delete_provider_files,
    finalize_submission,
)
from chronotranscriber.common.batch import BatchListing
from chronotranscriber.common.batch_reconcile import (
    abandon,
    adopt,
    count_matches,
    lease_submission,
    list_candidates,
    provider_of,
    reconcile,
)
from chronotranscriber.common.polling import (
    PollResult,
    WaitLimits,
    WaitTick,
    poll_until_settled,
)
from chronotranscriber.common.retry import classify_error
from chronotranscriber.common.store import (
    LeaseLostError,
    Store,
    StoreError,
    Submission,
    SubmissionStateError,
)
from chronotranscriber.common.usage import Usage
from chronotranscriber.output.render import TEXT_FORMATS, render_item, text_formats
from chronotranscriber.output.store import DATABASE_NAME, open_store, read_store
from chronotranscriber.settings import Settings

__all__ = [
    "ABANDONED",
    "UNRESOLVED",
    "CheckResult",
    "CheckStats",
    "CheckTargetError",
    "SubmissionCheck",
    "WaitLedger",
    "check_outputs",
    "check_submissions",
    "check_usage",
    "check_until_settled",
    "failed_units",
    "final_totals",
    "find_databases",
    "pending_work",
]

logger = logging.getLogger(__name__)

ROLE = "transcription"

UNRESOLVED: Final = "unresolved"
ABANDONED: Final = "abandoned"
_WORKED: Final = ("submitting", "submitted", UNRESOLVED)


@dataclass
class CheckStats:
    """How many submissions ended in each outcome.

    ``skipped`` counts submissions that were no longer submitted;
    ``unresolved`` the submissions that wait for an adoption or an
    abandonment.
    """

    finalized: int = 0
    pending: int = 0
    failed: int = 0
    skipped: int = 0
    unresolved: int = 0
    abandoned: int = 0

    @property
    def had_failure(self) -> bool:
        """Whether something failed or waits for the user; exit code 1."""
        return self.failed > 0 or self.unresolved > 0

    def count(self, outcome: str) -> None:
        """Count one outcome."""
        if outcome == FINALIZED_OUTCOME:
            self.finalized += 1
        elif outcome == PENDING:
            self.pending += 1
        elif outcome == FAILED:
            self.failed += 1
        elif outcome == SKIPPED:
            self.skipped += 1
        elif outcome == UNRESOLVED:
            self.unresolved += 1
        elif outcome == ABANDONED:
            self.abandoned += 1


@dataclass(frozen=True)
class SubmissionCheck:
    """What one check did with one submission.

    *outcome* is ``finalized``, ``pending``, ``failed``, ``unresolved`` or
    ``abandoned``; *detail* says why. *outputs* are the rendered files of
    the finalized documents, *errors* the units finalized as error
    placeholders and *candidates* the batches an unresolved submission may
    belong to. *complete* counts the finalized documents rendered with every
    unit done; *usage* holds the tokens of the finalized results.
    """

    database: Path
    submission_id: str
    provider: str
    outcome: str
    batch_id: str | None = None
    detail: str | None = None
    errors: int = 0
    outputs: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    candidates: tuple[BatchListing, ...] = ()
    complete: int = 0
    usage: Usage = field(default_factory=Usage)


class CheckTargetError(ValueError):
    """``--adopt`` or ``--abandon`` names nothing the check can act on."""


@dataclass
class CheckResult:
    """The counts and the per-submission results of one check.

    *submissions* are the per-submission results and *refused* the
    databases the store refused (versionless, newer or unreadable), with
    the reason.
    """

    stats: CheckStats = field(default_factory=CheckStats)
    submissions: list[SubmissionCheck] = field(default_factory=list)
    refused: dict[Path, str] = field(default_factory=dict)

    def add(self, check: SubmissionCheck) -> None:
        """Record and count one submission's result."""
        self.submissions.append(check)
        self.stats.count(check.outcome)

    @property
    def outputs(self) -> list[Path]:
        """The files the check rendered, submission by submission."""
        return check_outputs(self.submissions)

    @property
    def complete(self) -> int:
        """The documents the check rendered with every unit done."""
        return sum(check.complete for check in self.submissions)

    @property
    def had_failure(self) -> bool:
        """Whether the check ends with exit code 1.

        True when a submission failed, waits for the user or finalized a
        failed page (:func:`failed_units`).
        """
        return self.stats.had_failure or failed_units(self.submissions) > 0

    @property
    def usage(self) -> Mapping[str, Usage]:
        """The tokens of the finalized results, by role."""
        return check_usage(self.submissions)


def failed_units(checks: Sequence[SubmissionCheck]) -> int:
    """Return the units *checks* finalized as error placeholders."""
    return sum(check.errors for check in checks)


def check_outputs(checks: Sequence[SubmissionCheck]) -> list[Path]:
    """Return the files *checks* rendered, each once, in order."""
    return list(dict.fromkeys(path for check in checks for path in check.outputs))


def check_usage(checks: Sequence[SubmissionCheck]) -> dict[str, Usage]:
    """Return the tokens of the finalized results of *checks*, by role.

    Without a finalized submission there is no role.
    """
    finalized = [check for check in checks if check.outcome == FINALIZED_OUTCOME]
    if not finalized:
        return {}
    total = Usage()
    for check in finalized:
        total = total + check.usage
    return {ROLE: total}


def find_databases(directory: Path) -> list[Path]:
    """Return the tool's databases at or below *directory*, sorted."""
    return sorted(
        path for path in Path(directory).rglob(DATABASE_NAME) if path.is_file()
    )


@dataclass
class _Database:
    """A database found by the check, read before any write."""

    path: Path
    reader: Store
    submissions: list[Submission]

    @property
    def root(self) -> Path:
        return self.path.parent


@dataclass(frozen=True)
class _Targets:
    """The submissions ``--adopt`` and ``--abandon`` name: (database, id)."""

    adopt: tuple[Path, str, str] | None = None
    abandon: tuple[Path, str] | None = None

    def adopts(self, path: Path, submission_id: str) -> str | None:
        """Return the batch to adopt for this submission, if any."""
        if self.adopt is not None and self.adopt[:2] == (path, submission_id):
            return self.adopt[2]
        return None


def _read_databases(directory: Path, result: CheckResult) -> list[_Database]:
    found = []
    for path in find_databases(directory):
        try:
            reader = read_store(path.parent)
        except StoreError as exc:
            logger.error("Batch check skips %s: %s", path, exc)
            result.refused[path] = str(exc)
            continue
        if reader is not None:
            found.append(_Database(path, reader, reader.submissions(_WORKED)))
    return found


def _unsettled(submission: Submission) -> bool:
    """Whether *submission* is unresolved or was left without a batch id."""
    return submission.status == UNRESOLVED or (
        submission.status == "submitting" and submission.batch_id is None
    )


def _abandon_target(
    found: Sequence[_Database], submission_id: str, directory: Path
) -> tuple[Path, str]:
    for database in found:
        try:
            submission = database.reader.submission(submission_id)
        except KeyError:
            continue
        if not (_unsettled(submission) or submission.status == "submitted"):
            raise CheckTargetError(
                f"Submission {submission_id} is {submission.status}; only an "
                "unresolved or submitted submission can be abandoned"
            )
        return database.path, submission_id
    raise CheckTargetError(
        f"No submission {submission_id} in the databases below {directory}"
    )


async def _fitting(
    choices: Sequence[tuple[_Database, Submission]],
    batch_id: str,
    clients: BatchClients,
) -> list[tuple[_Database, Submission]]:
    """Keep the submissions whose time window holds *batch_id*.

    Those with the batch's request count win over the others.
    """
    fitting: list[tuple[_Database, Submission, bool]] = []
    for database, submission in choices:
        try:
            candidates = await list_candidates(submission, clients)
        except Exception as exc:
            raise CheckTargetError(
                f"Cannot list the provider's batches: {exc}"
            ) from exc
        listing = next((c for c in candidates if c.batch_id == batch_id), None)
        if listing is not None:
            fitting.append((database, submission, count_matches(listing, submission)))
    counted = [entry for entry in fitting if entry[2]]
    return [(database, submission) for database, submission, _ in counted or fitting]


async def _adopt_target(
    found: Sequence[_Database],
    batch_id: str,
    clients: BatchClients,
    directory: Path,
) -> tuple[Path, str, str]:
    provider = provider_of(batch_id)
    for database in found:
        for submission in database.submissions:
            if submission.batch_id == batch_id:
                raise CheckTargetError(
                    f"Batch {batch_id} already belongs to submission "
                    f"{submission.submission_id}"
                )
    choices = [
        (database, submission)
        for database in found
        for submission in database.submissions
        if submission.provider == provider and _unsettled(submission)
    ]
    if not choices:
        raise CheckTargetError(
            f"No unresolved {provider} submission below {directory} to adopt "
            f"batch {batch_id} into"
        )
    if len(choices) > 1:
        choices = await _fitting(choices, batch_id, clients)
    if len(choices) != 1:
        ids = ", ".join(submission.submission_id for _, submission in choices)
        raise CheckTargetError(
            f"Batch {batch_id} fits {len(choices)} unresolved submissions"
            + (f" ({ids})" if ids else "")
            + "; abandon the others first"
        )
    database, submission = choices[0]
    return database.path, submission.submission_id, batch_id


def _output_path(store: Store, root: Path, doc_id: int, name: str) -> Path:
    """Return the main output of a document: its registered file, else by name."""
    for row in store.outputs(doc_id):
        path = Path(row.path)
        return path if path.is_absolute() else root / path
    return root / f"{name}.txt"


def _run_formats(store: Store, run_id: str) -> tuple[str, ...] | None:
    """Return the text formats the run *run_id* stored in its spec, if any.

    A ``transcribe`` run stores ``output.formats``; a repair stores none.
    """
    try:
        row = store.run(run_id)
    except (StoreError, sqlite3.Error):
        return None
    raw: Any = row.get("spec") if row else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    entry = raw.get("output.formats") if isinstance(raw, Mapping) else None
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, list):
        return None
    chosen = tuple(name for name in value if name in TEXT_FORMATS)
    return chosen or None


def _document_formats(store: Store, doc_id: int) -> tuple[str, ...]:
    """Return the text formats registered for a document; txt when none is."""
    registered = {row.format for row in store.outputs(doc_id)}
    return tuple(name for name in TEXT_FORMATS if name in registered) or ("txt",)


def _complete(store: Store, doc_id: int) -> bool:
    """Whether every unit of the document is done and none a batch holds."""
    if store.blocked_units(doc_id):
        return False
    return all(unit.status == "done" for unit in store.units(doc_id))


async def _render(
    store: Store,
    root: Path,
    doc_ids: Sequence[int],
    formats: Sequence[str] | None,
) -> tuple[list[Path], list[str], int]:
    """Render the finalized documents under the leases the check holds.

    *formats* None renders each document in its registered formats. Returns
    the files, the warnings and the number of documents rendered complete.
    """
    outputs: list[Path] = []
    warnings: list[str] = []
    complete = 0
    held = store.held_leases()
    for doc_id in doc_ids:
        document = store.document(doc_id)
        lease = held.get(doc_id)
        if document is None or lease is None:
            warnings.append(f"Document {doc_id} was not rendered: it is not leased")
            continue
        output = _output_path(store, root, doc_id, document.name)
        chosen = formats if formats is not None else _document_formats(store, doc_id)
        try:
            report = await render_item(store, doc_id, chosen, output, lease)
        except (LeaseLostError, OSError, sqlite3.Error) as exc:
            warnings.append(f"'{document.name}' was not rendered: {exc}")
            continue
        outputs.extend([*report.written, *report.unchanged])
        warnings.extend(report.warnings)
        if _complete(store, doc_id):
            complete += 1
    return outputs, warnings, complete


async def _finalize(
    store: Store,
    path: Path,
    submission: Submission,
    clients: BatchClients,
    formats: Sequence[str] | None,
    note: str | None,
) -> SubmissionCheck:
    """Finalize a submitted submission, render it and delete its provider files.

    *formats* None renders the formats of the run that submitted it, else
    each document's registered ones. The files are deleted only once the
    results are stored and rendered.
    """
    finalization: SubmissionFinalization = await finalize_submission(
        store, submission, clients
    )
    detail = "; ".join(part for part in (note, finalization.detail) if part) or None
    if finalization.outcome != FINALIZED_OUTCOME:
        return SubmissionCheck(
            path,
            submission.submission_id,
            submission.provider,
            finalization.outcome,
            submission.batch_id,
            detail,
        )
    if formats is None:
        formats = _run_formats(store, submission.run_id)
    outputs, warnings, complete = await _render(
        store, path.parent, finalization.documents, formats
    )
    await delete_provider_files(clients, submission)
    return SubmissionCheck(
        path,
        submission.submission_id,
        submission.provider,
        FINALIZED_OUTCOME,
        submission.batch_id,
        detail,
        errors=finalization.errors,
        outputs=tuple(outputs),
        warnings=tuple(warnings),
        complete=complete,
        usage=finalization.usage,
    )


async def _settle(
    store: Store,
    submission: Submission,
    clients: BatchClients,
    batch_id: str | None,
) -> tuple[Submission | None, str | None, tuple[BatchListing, ...]]:
    """Adopt *batch_id*, else reconcile; return the submission once submitted.

    Without a submitted submission the detail and the candidates explain
    what the user can do.
    """
    if batch_id is not None:
        adopted = await adopt(store, submission, batch_id, clients)
        return adopted, f"adopted batch {batch_id}", ()
    found = await reconcile(
        store, submission, clients, program="chronotranscriber", units="pages"
    )
    if found.adopted is None:
        return None, found.detail, found.candidates
    return found.submission, found.detail, ()


async def _check_one(
    store: Store,
    path: Path,
    submission: Submission,
    clients: BatchClients,
    formats: Sequence[str] | None,
    targets: _Targets,
) -> SubmissionCheck:
    submission_id = submission.submission_id

    def result(outcome: str, detail: str | None = None) -> SubmissionCheck:
        return SubmissionCheck(
            path,
            submission_id,
            submission.provider,
            outcome,
            submission.batch_id,
            detail,
        )

    abandoning = targets.abandon == (path, submission_id)
    batch_id = targets.adopts(path, submission_id)
    if submission.status == "submitting" and not abandoning and batch_id is None:
        return result(PENDING, "another run is submitting it")
    if not await lease_submission(store, submission_id):
        return result(PENDING, "a document of the submission is leased by another run")
    try:
        if abandoning:
            await abandon(store, submission, clients)
            return result(ABANDONED, "its pages are pending again")
        note: str | None = None
        if submission.status != "submitted":
            try:
                settled, note, candidates = await _settle(
                    store, submission, clients, batch_id
                )
            except LeaseLostError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failure is reported
                action = f"batch {batch_id} not adopted" if batch_id else "not settled"
                return result(FAILED, f"{action}: {exc}")
            if settled is None:
                return replace(result(UNRESOLVED, note), candidates=candidates)
            submission = settled
        return await _finalize(store, path, submission, clients, formats, note)
    except LeaseLostError as exc:
        return result(FAILED, f"the lease was lost: {exc}")
    except SubmissionStateError as exc:
        return result(FAILED, str(exc))


async def _check_database(
    database: _Database,
    clients: BatchClients,
    formats: Sequence[str] | None,
    wanted: set[str],
    targets: _Targets,
    result: CheckResult,
) -> None:
    """Mark, reconcile and finalize the submissions of one database."""
    async with open_store(database.root) as store:
        await store.begin_run(
            "batch check",
            spec={
                "directory": str(database.root),
                "formats": None if formats is None else list(formats),
                "batch_ids": sorted(wanted),
            },
        )
        checks: list[SubmissionCheck] = []
        try:
            async with store.heartbeat():
                for submission_id in await store.mark_stale_submissions():
                    logger.warning(
                        "Submission %s in %s was left without a provider batch id; "
                        "it is unresolved",
                        submission_id,
                        database.path,
                    )
                for submission in store.submissions(_WORKED):
                    if (
                        wanted
                        and submission.status == "submitted"
                        and submission.batch_id not in wanted
                        and targets.abandon != (database.path, submission.submission_id)
                    ):
                        continue
                    check = await _check_one(
                        store, database.path, submission, clients, formats, targets
                    )
                    checks.append(check)
                    result.add(check)
        finally:
            failed = failed_units(checks) > 0 or any(
                check.outcome in (FAILED, UNRESOLVED) for check in checks
            )
            await store.end_run(1 if failed else 0)


async def check_submissions(  # noqa: PLR0913 - the check's options and seams
    directory: Path,
    settings: Settings,
    *,
    formats: Sequence[str] | None = None,
    batch_ids: Sequence[str] = (),
    adopt: str | None = None,
    abandon: str | None = None,
    args: argparse.Namespace | None = None,
    clients: BatchClients | None = None,
    result: CheckResult | None = None,
) -> CheckResult:
    """Check the open batch submissions of every database below *directory*.

    The text formats among *formats* are rendered (txt when there is none);
    *formats* None renders the formats of the run that submitted each
    batch, else each document's registered formats.
    *batch_ids*, when given, limit the finalization to those batches;
    unresolved submissions are reconciled in every check. *adopt* records a
    provider batch for the unresolved submission it fits, *abandon* gives an
    unresolved submission up. A database the store refuses is listed in
    :attr:`CheckResult.refused` and left alone. Only databases with a
    submission to work are opened for writing. *args* holds the parsed
    command line, if any, for the extension's key variables. *clients*
    replaces the batch clients the check builds; the caller closes them.
    The check fills *result*, when given, as it goes, so a caller keeps what
    the databases before an exception reported.

    Raises:
        CheckTargetError: *adopt* or *abandon* fits no submission, or more
            than one.
    """
    directory = Path(directory)
    result = result if result is not None else CheckResult()
    if clients is None:
        clients = BatchClients(settings, ROLE, args=args)
    found = _read_databases(directory, result)
    targets = _Targets(
        adopt=(
            await _adopt_target(found, adopt, clients, directory) if adopt else None
        ),
        abandon=_abandon_target(found, abandon, directory) if abandon else None,
    )
    outputs = None if formats is None else text_formats(formats)
    wanted = set(batch_ids)
    for database in found:
        if not database.submissions:
            continue
        await _check_database(database, clients, outputs, wanted, targets, result)
    return result


_clock: Callable[[], float] = time.monotonic
"""The clock of :func:`check_until_settled`; tests replace it with :func:`_sleep`."""


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _ignore(*_: object) -> None:
    return None


def pending_work(result: CheckResult) -> int:
    """Return the submissions a pass left for a later pass to finalize.

    They are running, met a retryable provider error, are being submitted
    by another run or have a document another run holds. Failed, unresolved
    and filtered-out submissions end the wait.
    """
    return result.stats.pending


type _Key = tuple[Path, str]


@dataclass
class WaitLedger:
    """What the passes of a wait saw, kept for the totals at its end.

    *checks* holds the submission results of every pass in order, *refused*
    the databases any pass refused and *failed* the submissions any pass
    reported failed. *polls* counts the passes that ended; *started* is the
    clock reading when the wait began.
    """

    checks: list[SubmissionCheck] = field(default_factory=list)
    refused: dict[Path, str] = field(default_factory=dict)
    failed: set[_Key] = field(default_factory=set)
    polls: int = 0
    started: float | None = None

    def record(self, result: CheckResult) -> None:
        """Keep what one pass reported, also a pass that raised."""
        self.checks.extend(result.submissions)
        self.refused.update(result.refused)
        self.failed.update(
            (check.database, check.submission_id)
            for check in result.submissions
            if check.outcome == FAILED
        )

    def elapsed(self) -> float:
        """Return the seconds since the wait began (0 before)."""
        return 0.0 if self.started is None else _clock() - self.started


_STORED_OUTCOMES: Final = {
    "finalized": FINALIZED_OUTCOME,
    "abandoned": ABANDONED,
    "failed": FAILED,
    "cancelled": SKIPPED,
}


def _stored_status(
    database: Path, submission_id: str, readers: dict[Path, Store | None]
) -> str | None:
    """Return the submission's status in the store now; None if unreadable."""
    if database not in readers:
        try:
            readers[database] = read_store(database.parent)
        except StoreError:
            readers[database] = None
    reader = readers[database]
    if reader is None:
        return None
    try:
        return reader.submission(submission_id).status
    except (KeyError, StoreError, sqlite3.Error):
        return None


def _final_outcome(status: str | None, last: str, failed: bool) -> str:
    if status in _STORED_OUTCOMES:
        return _STORED_OUTCOMES[status]
    if failed:
        return FAILED
    if status == UNRESOLVED:
        return UNRESOLVED
    if status in ("submitting", "submitted"):
        return PENDING
    return last


def final_totals(ledger: WaitLedger) -> CheckStats:
    """Count every submission the wait saw by its state in the store now.

    A submission a later pass no longer listed counts too. One that a pass
    reported failed stays failed unless the store shows it finalized,
    abandoned or cancelled; an unreadable store keeps the last outcome.
    """
    latest: dict[_Key, str] = {}
    for check in ledger.checks:
        latest[(check.database, check.submission_id)] = check.outcome
    readers: dict[Path, Store | None] = {}
    stats = CheckStats()
    for (database, submission_id), outcome in latest.items():
        status = _stored_status(database, submission_id, readers)
        failed = (database, submission_id) in ledger.failed
        stats.count(_final_outcome(status, outcome, failed))
    return stats


def _retryable(exc: BaseException) -> bool:
    """Whether a pass that raised *exc* counts as pending and the wait goes on."""
    if isinstance(exc, (CheckTargetError, StoreError)):
        return False
    return classify_error(exc).retryable


async def check_until_settled(  # noqa: PLR0913 - the check's options plus the hooks
    directory: Path,
    settings: Settings,
    limits: WaitLimits,
    ledger: WaitLedger,
    *,
    formats: Sequence[str] | None = None,
    batch_ids: Sequence[str] = (),
    adopt: str | None = None,
    abandon: str | None = None,
    args: argparse.Namespace | None = None,
    on_pass: Callable[[int, CheckResult], None] = _ignore,
    on_wait: Callable[[WaitTick], None] = _ignore,
    on_retry: Callable[[int, Exception], None] = _ignore,
) -> PollResult[CheckResult]:
    """Repeat :func:`check_submissions` until no batch work is pending.

    Each pass rescans *directory* with one set of batch clients, closed when
    the wait ends on any path; *adopt* and *abandon* apply to the first pass
    only. *ledger* records every pass, also one that raises, for
    :func:`final_totals`. *on_pass* receives each pass that returned, and
    the hooks of :func:`~chronotranscriber.common.polling.poll_until_settled`
    the ticks and the passes that failed with a retryable error.

    Raises:
        CheckTargetError: *adopt* or *abandon* fits nothing in the first pass.
        StoreError: A database could not be written.
    """
    clients = BatchClients(settings, ROLE, args=args)

    async def run_pass(number: int) -> CheckResult:
        result = CheckResult()
        first = number == 1
        try:
            await check_submissions(
                directory,
                settings,
                formats=formats,
                batch_ids=batch_ids,
                adopt=adopt if first else None,
                abandon=abandon if first else None,
                args=args,
                clients=clients,
                result=result,
            )
        except Exception as exc:
            if _retryable(exc):
                ledger.polls = number
            raise
        finally:
            ledger.record(result)
        ledger.polls = number
        on_pass(number, result)
        return result

    ledger.started = _clock()
    try:
        return await poll_until_settled(
            run_pass,
            pending_work,
            limits,
            on_wait=on_wait,
            on_retry=on_retry,
            retryable=_retryable,
            clock=_clock,
            sleep=_sleep,
        )
    finally:
        clients.close()
