"""The store subcommands: render the views and drop documents.

``render`` exits 0 when every file holds its rendering or was left alone
because it was edited, 1 when a file could not be replaced or a document is
held by another run, 2 for a missing input or a refused database (one that
is not a store, newer or unreadable). ``drop`` exits 0 when every selected
document was dropped (or, in a dry run, would be), 1 when a document was
refused, 2 for a malformed or empty selection, no store at the input or a
refused database. Each writes one JSON summary line with ``--json``; with
``--dry-run`` neither changes anything.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from chronotranscriber.cli.command import (
    Summary,
    drop_fields,
    guarded,
    render_fields,
    setup_logging,
)
from chronotranscriber.cli.console import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.store import (
    DocumentSelectionError,
    VersionlessDatabaseError,
)
from chronotranscriber.output.drop import (
    DropInputError,
    DropOutcome,
    drop_documents,
    find_database,
)
from chronotranscriber.output.render import (
    DocumentRender,
    find_databases,
    preview_database,
    render_database,
)
from chronotranscriber.output.store import DATABASE_NAME, read_store
from chronotranscriber.spec import (
    DROP_OPTIONS,
    RENDER_OPTIONS,
    DropSpec,
    RenderSpec,
    SpecError,
    resolve_drop,
    resolve_render,
)

__all__ = ["execute_drop", "execute_render"]


def _usage_error(summary: Summary, error: SpecError) -> int:
    print_error(f"Invalid arguments: {error}")
    return summary.emit(ExitCode.USAGE)


def _report_render(results: list[DocumentRender], dry_run: bool) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    verb = "would write" if dry_run else "wrote"
    for result in results:
        label = f"{result.database.parent.name}/{result.name}"
        if result.skipped is not None:
            print_error(f"{label}: {result.skipped}")
        for path in result.written:
            print_info(f"{prefix}{label}: {verb} {path}")
        for path in result.kept:
            print_warning(
                f"{label}: {path} was changed outside the tool and is left as it "
                "is; render with --force to replace it."
            )
        for warning in result.warnings:
            if "changed outside the tool" not in warning:
                print_warning(f"{label}: {warning}")
    written = sum(len(result.written) for result in results)
    unchanged = sum(len(result.unchanged) for result in results)
    kept = sum(len(result.kept) for result in results)
    print_info(
        f"{prefix}Render of {len(results)} document(s): {verb} {written} "
        f"file(s), {unchanged} unchanged, {kept} left as edited."
    )


def _execute_render(spec: RenderSpec, summary: Summary) -> int:
    setup_logging()
    path = spec.input.expanduser().resolve()
    if not path.exists():
        print_error(f"Invalid arguments: Input path does not exist: {path}")
        return summary.emit(ExitCode.USAGE)
    if path.is_file() and path.name != DATABASE_NAME:
        print_error(f"Invalid arguments: {path} is not a {DATABASE_NAME} file")
        return summary.emit(ExitCode.USAGE)
    databases = find_databases(path)
    for database in databases:
        try:
            read_store(database.parent)
        except VersionlessDatabaseError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(ExitCode.USAGE)
    print_header("RENDER", str(path))
    results: list[DocumentRender] = []
    for database in databases:
        if spec.dry_run:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        else:
            results.extend(
                asyncio.run(
                    render_database(database, formats=spec.formats, force=spec.force)
                )
            )
    _report_render(results, spec.dry_run)
    failed = any(result.failed for result in results)
    return summary.emit(
        ExitCode.FAILURE if failed else ExitCode.OK,
        **render_fields(
            databases=len(databases),
            documents=len(results),
            written=sum(len(result.written) for result in results),
            unchanged=sum(len(result.unchanged) for result in results),
            kept=sum(len(result.kept) for result in results),
            stale=sum(len(result.stale) for result in results),
            skipped=sum(result.skipped is not None for result in results),
            outputs=[str(path) for result in results for path in result.written],
        ),
    )


def execute_render(args: argparse.Namespace) -> int:
    """Run the parsed render options and return the exit code."""
    flags = RENDER_OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        render_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    try:
        spec = resolve_render(flags).spec
    except SpecError as exc:
        return _usage_error(summary, exc)
    return guarded(summary, lambda: _execute_render(spec, summary))


_PRINTERS = {
    "info": print_info,
    "warning": print_warning,
    "error": print_error,
    "success": print_success,
}


def _drop_summary_fields(outcome: DropOutcome) -> dict[str, object]:
    fields = outcome.report.as_dict()
    return drop_fields(
        dropped=fields["dropped"],
        refused=fields["refused"],
        deleted_files=fields["deleted_files"],
        kept_files=fields["kept_files"],
    )


def _execute_drop(spec: DropSpec, summary: Summary) -> int:
    setup_logging()
    try:
        database = find_database(spec.input)
    except DropInputError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    print_header("DROP", str(database))
    try:
        outcome = asyncio.run(
            drop_documents(
                database,
                select=spec.select,
                files=spec.files,
                select_all=spec.all,
                dry_run=spec.dry_run,
            )
        )
    except VersionlessDatabaseError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    except (DropInputError, DocumentSelectionError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    for level, message in outcome.lines():
        _PRINTERS[level](message)
    code = ExitCode.FAILURE if outcome.report.refused else ExitCode.OK
    return summary.emit(code, **_drop_summary_fields(outcome))


def execute_drop(args: argparse.Namespace) -> int:
    """Run the parsed drop options and return the exit code."""
    flags = DROP_OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        drop_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    try:
        spec = resolve_drop(flags).spec
    except SpecError as exc:
        return _usage_error(summary, exc)
    return guarded(summary, lambda: _execute_drop(spec, summary))
