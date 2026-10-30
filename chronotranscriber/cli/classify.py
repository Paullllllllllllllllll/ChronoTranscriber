"""The classify subcommands: ``classify pages`` and ``classify documents``.

Exit codes: 0 when every item was classified (asked or reused), or in a dry
run planned; 1 when a request failed or an item was refused because its
stored document records another input; 2 for usage and configuration
errors (no classifier, a classifier of another unit, no items, an unset key
variable); 130 on Ctrl+C or Ctrl+Break. With ``--json`` one summary line
goes to stdout on every exit.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from chronotranscriber import classify, ext
from chronotranscriber.cli.command import (
    Summary,
    classify_fields,
    guarded,
    load_settings,
    setup_logging,
    warn_conditions,
)
from chronotranscriber.cli.console import print_error, print_info, print_success
from chronotranscriber.cli.reporter import StderrReporter
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.run import PlanError
from chronotranscriber.settings import SettingsError
from chronotranscriber.spec import CLASSIFY_OPTIONS, SpecError, resolve_classify

__all__ = ["execute_documents", "execute_pages"]


def _plan_fields(run_plan: classify.ClassifyPlan) -> dict[str, Any]:
    """Return the fields a dry run adds: the items, the refused ones, the spec."""
    return {
        "to_classify": [
            {
                "name": item.name,
                "type": item.type,
                "document": item.document,
                "units": item.units,
                "requests": item.requests,
            }
            for item in run_plan.items
        ],
        "refused": [
            {"name": item.name, "reason": item.reason} for item in run_plan.refused
        ],
        "spec": run_plan.effective_spec(),
    }


def _report_plan(run_plan: classify.ClassifyPlan, reporter: StderrReporter) -> None:
    for message in run_plan.warnings:
        reporter.warning(message)
    for item in run_plan.refused:
        reporter.warning(item.reason)
    classifier = run_plan.classifier
    print_info(
        f"Classifier {classifier.name} v{classifier.version} "
        f"({run_plan.classifier_path}); unit {classifier.unit}"
    )
    if not run_plan.dry_run:
        return
    for item in run_plan.items:
        if item.note is not None:
            print_info(f"[DRY RUN]   would classify: {item.name} ({item.note})")
        else:
            print_info(
                f"[DRY RUN]   would classify: {item.name}: {item.units} unit(s) in "
                f"at most {item.requests} request(s)"
            )
    print_info(
        f"[DRY RUN] {len(run_plan.items)} item(s) would be classified, "
        f"{len(run_plan.refused)} refused."
    )


def _report_result(result: classify.ClassifyResult) -> None:
    if result.failed or result.refused:
        print_error(
            f"Classification finished with {len(result.failed)} failed and "
            f"{len(result.refused)} refused item(s) of "
            f"{result.items + len(result.refused)}."
        )
    else:
        print_success("Classification complete!")
    print_info(
        f"Units: {result.units_asked} asked, {result.units_reused} reused, "
        f"{result.units_failed} failed."
    )
    for database in result.databases:
        print_info(f"Database: {database}")


def _run(
    unit: str, args: argparse.Namespace, flags: dict[str, Any], summary: Summary
) -> int:
    table = CLASSIFY_OPTIONS[unit]
    setup_logging()
    try:
        settings = load_settings(flags)
        resolution = resolve_classify(unit, flags, settings.defaults)
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    spec = resolution.spec
    if not spec.dry_run:
        setup_logging(settings)
    warn_conditions(table, flags, settings, spec)
    try:
        run_plan = classify.plan(spec, settings, sources=resolution.sources, args=args)
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code, items_total=exc.items_total)
    name = run_plan.classifier.name
    planned = classify_fields(
        unit=unit,
        classifier=name,
        items=len(run_plan.items),
        refused=len(run_plan.refused),
    )
    if run_plan.dry_run or run_plan.items:
        try:
            ext.startup(spec, settings=settings, args=args)
        except SettingsError as exc:
            print_error(f"Invalid arguments: {exc}")
            return summary.emit(ExitCode.USAGE, **planned)
    reporter = StderrReporter()
    _report_plan(run_plan, reporter)
    if run_plan.dry_run:
        return summary.emit(ExitCode.OK, **planned, **_plan_fields(run_plan))
    try:
        result = asyncio.run(classify.run(run_plan, reporter))
    except PlanError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(exc.exit_code, **planned)
    _report_result(result)
    return summary.emit(
        result.exit_code,
        **classify_fields(
            unit=unit,
            classifier=name,
            items=result.items,
            units_asked=result.units_asked,
            units_reused=result.units_reused,
            units_failed=result.units_failed,
            refused=len(result.refused),
            failed=len(result.failed),
            usage=result.usage,
        ),
    )


def _execute(unit: str, args: argparse.Namespace) -> int:
    flags = CLASSIFY_OPTIONS[unit].values(args)
    summary = Summary(
        sys.stdout if flags.get("json") else None,
        classify_fields(unit=unit),
        dry_run=bool(flags.get("dry_run")),
    )
    return guarded(summary, lambda: _run(unit, args, flags, summary))


def execute_pages(args: argparse.Namespace) -> int:
    """Run the parsed ``classify pages`` options and return the exit code."""
    return _execute("page", args)


def execute_documents(args: argparse.Namespace) -> int:
    """Run the parsed ``classify documents`` options and return the exit code."""
    return _execute("document", args)
