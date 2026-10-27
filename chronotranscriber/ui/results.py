"""Build the guided UI's result screens of batch, repair, render and drop.

Each function returns a rich renderable with the facts the CLI prints for
the same command: one row per submission, batch, document or database, the
warnings and errors, and the totals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from chronotranscriber.batch.cancel import CancelReport
from chronotranscriber.batch.check import (
    ABANDONED,
    UNRESOLVED,
    CheckResult,
    SubmissionCheck,
    failed_units,
)
from chronotranscriber.batch.finalize import FAILED, FINALIZED_OUTCOME
from chronotranscriber.batch.repair import RepairResult
from chronotranscriber.output.drop import DropOutcome
from chronotranscriber.output.render import DocumentRender
from chronotranscriber.spec import RepairSpec

__all__ = [
    "cancel_result",
    "check_result",
    "drop_result",
    "render_result",
    "repair_result",
]

_OUTCOME_STYLES = {
    FINALIZED_OUTCOME: "green",
    FAILED: "red",
    UNRESOLVED: "yellow",
    ABANDONED: "yellow",
}


def _database_label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _lines(
    messages: Iterable[str], prefix: str = "", style: str | None = None
) -> list[Text]:
    return [Text(f"{prefix}{message}", style=style or "") for message in messages]


def _refused(refused: dict[Path, str]) -> list[Text]:
    return [
        Text(f"Refused {path}: {reason}", style="red")
        for path, reason in refused.items()
    ]


def _submission_notes(check: SubmissionCheck) -> list[Text]:
    label = f"{_database_label(check.database)} {check.submission_id}"
    notes: list[Text] = []
    if check.outputs:
        names = ", ".join(path.name for path in check.outputs)
        notes.append(Text(f"{label}: outputs {names}"))
    if check.errors:
        notes.append(
            Text(
                f"{label}: {check.errors} page(s) failed in the batch; repair "
                "them with 'chronotranscriber repair'.",
                style="yellow",
            )
        )
    notes += _lines(check.warnings, f"{label}: warning: ", "yellow")
    for listing in check.candidates:
        created = listing.created_at.isoformat() if listing.created_at else "unknown"
        count = "unknown" if listing.request_count is None else listing.request_count
        status = listing.provider_status or listing.status.value
        notes.append(
            Text(
                f"{label}: candidate {listing.batch_id}: created {created}, "
                f"{count} request(s), {status}"
            )
        )
    return notes


def check_result(result: CheckResult) -> RenderableType:
    """Return the outcome of a batch check: submissions, refusals and totals."""
    table = Table(title="Batch check", title_justify="left")
    for header in ("Database", "Submission", "Provider", "Batch", "Outcome"):
        table.add_column(header)
    table.add_column("Detail")
    notes: list[Text] = []
    for check in result.submissions:
        table.add_row(
            Text(_database_label(check.database)),
            Text(check.submission_id),
            check.provider,
            Text(check.batch_id or "-"),
            Text(check.outcome, style=_OUTCOME_STYLES.get(check.outcome, "")),
            Text(check.detail or ""),
        )
        notes += _submission_notes(check)
    stats = result.stats
    failed_pages = failed_units(result.submissions)
    pages = f"; {failed_pages} page(s) failed in the batch" if failed_pages else ""
    totals = Text(
        f"Batch check: {stats.finalized} finalized, {stats.pending} pending, "
        f"{stats.failed} failed, {stats.unresolved} unresolved, "
        f"{stats.abandoned} abandoned{pages}."
    )
    if stats.unresolved:
        notes.append(
            Text(
                "Settle an unresolved submission with --adopt BATCH_ID or "
                "--abandon SUBMISSION_ID.",
                style="yellow",
            )
        )
    return Group(table, *notes, *_refused(result.refused), totals)


def cancel_result(report: CancelReport) -> RenderableType:
    """Return the outcome of a batch cancel: listing, batches and totals."""
    parts: list[RenderableType] = []
    listing = report.listing
    if listing is not None:
        parts += [
            Text(f"{provider}: key variable not set; not listed.")
            for provider in listing.skipped
        ]
        parts += [
            Text(f"{provider}: listing failed: {error}", style="red")
            for provider, error in listing.errors.items()
        ]
        parts.append(
            Text(f"Found {len(listing.batches)} unfinished batch(es) to cancel.")
        )
    parts += _refused(report.refused)
    parts += _lines(report.skipped, "Not cancelled, no provider batch: ")
    table = Table(title="Batch cancellation", title_justify="left")
    for header in ("Batch", "Provider", "Submission", "Result"):
        table.add_column(header)
    for outcome in report.outcomes:
        result = (
            Text("cancelled", style="green")
            if outcome.cancelled
            else Text(f"failed: {outcome.error}", style="red")
        )
        table.add_row(
            Text(outcome.batch_id),
            outcome.provider,
            Text(outcome.submission_id or "-"),
            result,
        )
    parts.append(table)
    parts.append(Text(f"Cancelled: {report.cancelled}; failed: {report.failed}."))
    return Group(*parts)


def repair_result(spec: RepairSpec, result: RepairResult) -> RenderableType:
    """Return the outcome of a repair of *spec*'s input."""
    if result.message is not None and not result.selected:
        return Text(result.message)
    causes = Text(f"Targeting failure types: {', '.join(spec.selection.causes)}")
    several = result.documents > 1
    if result.dry_run:
        documents = f" of {result.documents} documents" if several else ""
        return Group(
            causes,
            Text(f"[DRY RUN] {result.selected} page(s){documents} would be repaired."),
        )
    subject = f"{result.documents} documents" if several else spec.input.name
    table = Table(title=escape(f"Repair of {subject}"), title_justify="left")
    table.add_column("Mode")
    for header in ("Selected", "Repaired", "Unrepaired"):
        table.add_column(header, justify="right")
    table.add_row(
        result.mode,
        str(result.selected),
        str(result.repaired),
        Text(str(result.failed), style="red" if result.failed else ""),
    )
    parts: list[RenderableType] = [causes, table]
    if result.message is not None:
        parts.append(Text(result.message, style="yellow"))
    parts += [Text(f"Stored in: {database}") for database in result.databases]
    parts += _lines((str(path) for path in result.outputs), "Rendered: ")
    return Group(*parts)


def render_result(results: Sequence[DocumentRender], dry_run: bool) -> RenderableType:
    """Return what a render did, or would do, with each document's files."""
    prefix = "[DRY RUN] " if dry_run else ""
    verb = "would write" if dry_run else "wrote"
    table = Table(title=f"{prefix}Render", title_justify="left")
    table.add_column("Document")
    for header in ("Written", "Unchanged", "Kept", "Stale"):
        table.add_column(header, justify="right")
    table.add_column("Skipped")
    notes: list[Text] = []
    for result in results:
        label = f"{result.database.parent.name}/{result.name}"
        table.add_row(
            Text(label),
            str(len(result.written)),
            str(len(result.unchanged)),
            str(len(result.kept)),
            Text(str(len(result.stale)), style="red" if result.stale else ""),
            Text(result.skipped or "", style="red"),
        )
        notes += _lines((str(path) for path in result.written), f"{label}: {verb} ")
        notes += [
            Text(
                f"{label}: {path} was changed outside the tool and is left as it "
                "is; render with --force to replace it.",
                style="yellow",
            )
            for path in result.kept
        ]
        notes += _lines(
            (w for w in result.warnings if "changed outside the tool" not in w),
            f"{label}: ",
            "yellow",
        )
    written = sum(len(result.written) for result in results)
    unchanged = sum(len(result.unchanged) for result in results)
    kept = sum(len(result.kept) for result in results)
    totals = Text(
        f"{prefix}Render of {len(results)} document(s): {verb} {written} "
        f"file(s), {unchanged} unchanged, {kept} left as edited."
    )
    return Group(table, *notes, totals)


_LEVEL_STYLES = {"info": "", "warning": "yellow", "error": "red", "success": "green"}


def drop_result(outcome: DropOutcome) -> RenderableType:
    """Return what a drop did, or would do: documents, refusals and files."""
    return Group(
        *(
            Text(message, style=_LEVEL_STYLES[level])
            for level, message in outcome.lines()
        )
    )
