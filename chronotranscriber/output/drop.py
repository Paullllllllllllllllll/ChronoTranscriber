"""Drop documents from a store, so that the next run processes them anew.

:func:`find_database` names the database an input path holds: the
``chronotranscriber.sqlite`` itself or the one directly inside an output
folder. :func:`drop_documents` chooses stored documents by ``--select``,
``--files`` or ``--all`` and deletes them, with their rows and the rendered
files that still match their recorded hash, through
:meth:`~chronotranscriber.common.store.Store.drop`. The ``drop`` subcommand
and the wizard's drop task report the outcome; nothing here prints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chronotranscriber.common.store import (
    DocumentSelectionError,
    DropReport,
    select_documents,
)
from chronotranscriber.output.store import DATABASE_NAME, open_store, read_store

__all__ = [
    "KEPT_FILE_NOTE",
    "DropInputError",
    "DropOutcome",
    "drop_documents",
    "find_database",
]

KEPT_FILE_NOTE = (
    "A kept file at a dropped item's output path makes the next transcribe run "
    "skip that item as unstored; move the file away to transcribe the item again."
)


class DropInputError(ValueError):
    """The input path names no store."""


@dataclass(frozen=True)
class DropOutcome:
    """The database a drop worked on, its report and the selection warnings."""

    database: Path
    report: DropReport
    warnings: tuple[str, ...] = ()

    def lines(self) -> list[tuple[str, str]]:
        """Return the report as ``(level, message)`` lines, the totals last.

        The levels are ``info``, ``warning``, ``error`` and ``success``.
        """
        report = self.report
        prefix = "[DRY RUN] " if report.dry_run else ""
        drop_verb = "would drop" if report.dry_run else "dropped"
        delete_verb = "would delete" if report.dry_run else "deleted"
        lines = [("warning", warning) for warning in self.warnings]
        lines += [("info", f"{prefix}{drop_verb} {name}") for name in report.dropped]
        for refusal in report.refused:
            label = refusal.name if refusal.name is not None else f"id {refusal.doc_id}"
            lines.append(("error", f"refused {label}: {refusal.reason}"))
        lines += [("info", f"{prefix}{delete_verb} {p}") for p in report.deleted_files]
        lines += [
            ("warning", f"kept {kept.path}: {kept.reason}")
            for kept in report.kept_files
        ]
        if any(kept.path.exists() for kept in report.kept_files):
            lines.append(("warning", KEPT_FILE_NOTE))
        level = "error" if report.refused else "info" if report.dry_run else "success"
        lines.append(
            (
                level,
                f"{prefix}Drop from {self.database}: {drop_verb} "
                f"{len(report.dropped)} document(s), {len(report.refused)} refused; "
                f"{delete_verb} {len(report.deleted_files)} file(s), "
                f"{len(report.kept_files)} kept.",
            )
        )
        return lines


def find_database(path: Path) -> Path:
    """Return the database *path* names: the file itself or the one inside it.

    Raises:
        DropInputError: *path* is missing, a file of another name, or a
            folder without the database.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise DropInputError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.name != DATABASE_NAME:
            raise DropInputError(f"{path} is not a {DATABASE_NAME} file")
        return path
    database = path / DATABASE_NAME
    if not database.is_file():
        raise DropInputError(f"No {DATABASE_NAME} in {path}")
    return database


async def drop_documents(
    database: Path,
    *,
    select: str | None = None,
    files: Sequence[str] = (),
    select_all: bool = False,
    dry_run: bool = False,
) -> DropOutcome:
    """Drop the documents of *database* the selection chooses.

    A dry run reads the store only. A real drop records a ``drop`` run that
    ends with exit code 1 when a document was refused.

    Raises:
        DropInputError: The database holds no store.
        DocumentSelectionError: The selection is malformed or chooses no
            stored document.
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    reader = read_store(root)
    if reader is None:
        raise DropInputError(f"{database} holds no store")
    selection = select_documents(
        reader.documents(), select=select, files=files, select_all=select_all
    )
    if not selection.documents:
        raise DocumentSelectionError(f"{database} holds no document")
    doc_ids = [document.doc_id for document in selection.documents]
    if dry_run:
        report = await reader.drop(doc_ids, dry_run=True)
        return DropOutcome(database, report, selection.warnings)
    async with open_store(root) as store:
        await store.begin_run(
            "drop",
            spec={
                "select": {"value": select, "source": "flag"},
                "files": {"value": list(files), "source": "flag"},
                "all": {"value": select_all, "source": "flag"},
            },
        )
        failed = True
        try:
            report = await store.drop(doc_ids)
            failed = bool(report.refused)
        finally:
            await store.end_run(1 if failed else 0)
    return DropOutcome(database, report, selection.warnings)
