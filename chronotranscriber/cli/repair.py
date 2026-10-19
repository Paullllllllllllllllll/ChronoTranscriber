"""The repair subcommand: plan the repairs, start the extension, run them.

``--input`` names a transcription file or its input, a database, or a
folder whose databases at or below it are repaired. Exit code 0 on a clean
repair; 1 when a targeted page stays unrepaired or the repair work fails; 2
when the repair cannot be planned (missing path, a file no store records, a
refused database, ``--indices`` with several documents, unresolvable
schema, prompt or context, no key or batch API, a model without image
input) or the extension refuses it. A dry run plans without the key
variables. The extension starts on every dry run and on every run with
pages to repair.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from chronotranscriber import ext
from chronotranscriber.batch.repair import (
    RepairPlanError,
    RepairResult,
    plan_repairs,
    run_repairs,
)
from chronotranscriber.cli.command import (
    Summary,
    guarded,
    load_settings,
    repair_fields,
    setup_logging,
    warn_conditions,
)
from chronotranscriber.cli.console import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from chronotranscriber.cli.reporter import StderrReporter
from chronotranscriber.common.contract import ExitCode, exit_code_for
from chronotranscriber.settings import SettingsError, SpecError, resolve_repair
from chronotranscriber.spec import REPAIR_OPTIONS, RepairSpec

__all__ = ["execute"]


def _subject(spec: RepairSpec, result: RepairResult) -> str:
    """Name what was repaired: the input, or the number of documents."""
    if result.documents > 1:
        return f"{result.documents} documents"
    return f"'{spec.input.name}'"


def _report(spec: RepairSpec, result: RepairResult) -> None:
    """Write the repair's outcome."""
    if result.message is not None and not result.selected:
        print_info(result.message)
        return
    print_info(f"Targeting failure types: {', '.join(spec.selection.causes)}")
    if result.dry_run:
        documents = f" of {result.documents} documents" if result.documents > 1 else ""
        print_info(f"[DRY RUN] {result.selected} page(s){documents} would be repaired.")
        return
    if result.message is not None:
        print_warning(result.message)
    print_header("REPAIR COMPLETE", "")
    subject = _subject(spec, result)
    if result.failed:
        print_warning(
            f"Repair of {subject} ({result.mode}) left {result.failed} of "
            f"{result.selected} page(s) unrepaired."
        )
    else:
        print_success(
            f"Repaired {result.repaired} page(s) of {subject} ({result.mode})."
        )
    for database in result.databases:
        print_info(f"Stored in: {database}")
    for output in result.outputs:
        print_info(f"Rendered: {output}")


def _execute(args: argparse.Namespace, summary: Summary) -> int:
    flags = REPAIR_OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        spec = resolve_repair(settings, flags).spec
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    if not spec.run.dry_run:
        setup_logging(settings)
    warn_conditions(REPAIR_OPTIONS, flags, settings, spec)
    print_header("REPAIR TRANSCRIPTIONS", "")
    try:
        repairs = plan_repairs(spec, settings, args=args)
    except RepairPlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code)
    for warning in repairs.warnings:
        print_warning(warning)
    if repairs.dry_run or repairs.to_repair:
        try:
            ext.startup(spec, settings=settings, args=args)
        except SettingsError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(ExitCode.USAGE)
    if repairs.dry_run or not repairs.to_repair:
        result = repairs.outcome()
    else:
        result = asyncio.run(run_repairs(repairs, StderrReporter()))
    _report(spec, result)
    return summary.emit(
        exit_code_for(failed=result.failed),
        **repair_fields(
            repaired=result.repaired, failed=result.failed, usage=result.usage
        ),
    )


def execute(args: argparse.Namespace) -> int:
    """Run the parsed repair options and return the exit code."""
    flags = REPAIR_OPTIONS.values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        repair_fields(),
        dry_run=bool(flags.get("dry_run")),
    )
    return guarded(summary, lambda: _execute(args, summary))
