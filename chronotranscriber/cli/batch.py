"""The batch subcommands: check and finalize, or cancel, batch submissions.

``batch check`` exits 1 when a submission failed or waits unresolved for
``--adopt`` or ``--abandon``, and 2 when a database below the folder is
refused (versionless, newer or unreadable) or ``--adopt`` or ``--abandon``
names nothing to act on. ``batch cancel`` exits 1 when a cancellation or a
provider listing failed, and 2 for a refused database. The batch check JSON
summary counts the submissions by outcome and carries ``submitted``,
``complete`` (the documents rendered with no unit left), ``failed_pages``
(the pages finalized as failed in the batch, which also exit 1),
``outputs`` (the rendered files) and ``usage``; it adds ``unresolved``
(the submissions waiting unresolved), and ``adopted`` or ``abandoned`` when
those options are given; a parse error's line carries only the common
check fields, ``failed_pages`` and ``unresolved``.

``batch check --wait`` checks again every ``--poll-minutes`` until no batch
work is pending, and also exits 1 when ``--wait-hours`` pass first. Its
counts come from the store's state of every submission the passes saw, and
its summary adds ``polls``, ``waited_seconds`` and ``stopped`` (``settled``,
``timeout``, ``interrupted`` or ``error``; null before any pass).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Coroutine, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chronotranscriber import ext
from chronotranscriber.batch.cancel import (
    CancelReport,
    cancel_batches,
    cancel_submissions,
    cancel_unfinished,
)
from chronotranscriber.batch.check import (
    ABANDONED,
    UNRESOLVED,
    CheckResult,
    CheckStats,
    CheckTargetError,
    SubmissionCheck,
    WaitLedger,
    check_outputs,
    check_until_settled,
    check_usage,
    failed_units,
    final_totals,
)
from chronotranscriber.batch.check import check_submissions as check_batches
from chronotranscriber.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    PENDING,
    SKIPPED,
)
from chronotranscriber.cli.command import (
    Summary,
    cancel_fields,
    check_fields,
    guarded,
    load_settings,
    setup_logging,
)
from chronotranscriber.cli.console import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from chronotranscriber.cli.heartbeat import BatchHeartbeat
from chronotranscriber.common.contract import (
    USAGE_KEY,
    ExitCode,
    exit_code_for,
    usage_field,
)
from chronotranscriber.common.polling import Stop, WaitLimits, WaitTick
from chronotranscriber.common.retry import classify_error
from chronotranscriber.common.store import StoreError
from chronotranscriber.settings import (
    Settings,
    SettingsError,
    SpecError,
    resolve_batch_cancel,
    resolve_batch_check,
)
from chronotranscriber.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    BatchCheckSpec,
    CommandSpec,
)

__all__ = ["check_batches", "execute_cancel", "execute_check"]


def _label(check: SubmissionCheck) -> str:
    batch = f" batch {check.batch_id}" if check.batch_id else ""
    return (
        f"{check.database.parent.name}/{check.database.name}: submission "
        f"{check.submission_id} ({check.provider}{batch})"
    )


def _report_unresolved(check: SubmissionCheck) -> None:
    print_warning(f"{_label(check)} is unresolved: {check.detail}")
    for listing in check.candidates:
        created = listing.created_at.isoformat() if listing.created_at else "unknown"
        count = "unknown" if listing.request_count is None else listing.request_count
        print_info(
            f"  candidate {listing.batch_id}: created {created}, {count} "
            f"request(s), {listing.provider_status or listing.status.value}"
        )


def _report_submission(check: SubmissionCheck) -> None:
    label = _label(check)
    if check.outcome == FINALIZED_OUTCOME:
        outputs = ", ".join(path.name for path in check.outputs) or "no file"
        print_success(f"Finalized {label} -> {outputs}")
        if check.detail:
            print_info(f"{label}: {check.detail}")
        for warning in check.warnings:
            print_warning(warning)
        if check.errors:
            print_warning(
                f"{check.errors} page(s) of {label} failed in the batch; repair "
                "them with 'chronotranscriber repair'."
            )
    elif check.outcome == PENDING:
        print_info(f"{label}: not finished yet ({check.detail})")
    elif check.outcome == FAILED:
        print_error(f"{label}: {check.detail}")
    elif check.outcome == UNRESOLVED:
        _report_unresolved(check)
    elif check.outcome == ABANDONED:
        print_info(f"Abandoned {label}: {check.detail}")
    elif check.outcome == SKIPPED:
        print_info(f"{label}: {check.detail}")


def _report_totals(stats: CheckStats, failed_pages: int) -> None:
    pages = f"; {failed_pages} page(s) failed in the batch" if failed_pages else ""
    print_info(
        f"Batch check: {stats.finalized} finalized, {stats.pending} pending, "
        f"{stats.failed} failed, {stats.unresolved} unresolved{pages}."
    )


def _report_check(
    result: CheckResult, *, pending: bool = True, totals: bool = True
) -> None:
    """Write one line per submission, the refused databases and the totals.

    Without *pending* the submissions not finished yet are left out.
    """
    for path, reason in result.refused.items():
        print_error(f"{path}: {reason}")
    for check in result.submissions:
        if pending or check.outcome != PENDING:
            _report_submission(check)
    if totals:
        _report_totals(result.stats, failed_units(result.submissions))


def _check_options(spec: BatchCheckSpec) -> dict[str, Any]:
    """Return the keyword options of the check beyond formats and batch IDs."""
    options: dict[str, Any] = {}
    if spec.adopt is not None:
        options["adopt"] = spec.adopt
    if spec.abandon is not None:
        options["abandon"] = spec.abandon
    return options


def _check_base(flags: Mapping[str, Any]) -> dict[str, Any]:
    """Return the check summary's fields with zero counts.

    ``adopted`` and ``abandoned`` appear when their option is given, the
    wait fields with ``--wait``.
    """
    base = {**check_fields(), "failed_pages": 0, "unresolved": 0}
    for option, key in (("adopt", "adopted"), ("abandon", "abandoned")):
        if flags.get(option):
            base[key] = False
    if flags.get("wait"):
        base.update(polls=0, waited_seconds=0, stopped=None)
    return base


def _target_fields(
    spec: BatchCheckSpec, checks: Sequence[SubmissionCheck], stats: CheckStats
) -> dict[str, Any]:
    fields: dict[str, Any] = {"unresolved": stats.unresolved}
    if spec.adopt is not None:
        fields["adopted"] = any(
            check.batch_id == spec.adopt and check.outcome != FAILED for check in checks
        )
    if spec.abandon is not None:
        fields["abandoned"] = any(
            check.submission_id == spec.abandon and check.outcome == ABANDONED
            for check in checks
        )
    return fields


def _startup(
    spec: CommandSpec,
    settings: Settings,
    args: argparse.Namespace,
    summary: Summary,
    directory: Path | None,
) -> int | None:
    """Check the input folder and run the extension startup.

    Returns the usage exit code if either refuses.
    """
    if directory is not None and not directory.is_dir():
        print_error(f"Invalid arguments: Input folder does not exist: {directory}")
        return summary.emit(ExitCode.USAGE)
    try:
        ext.startup(spec, settings=settings, args=args)
    except SettingsError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    return None


def _execute_check(args: argparse.Namespace, summary: Summary) -> int:
    flags = BATCH_CHECK_OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        spec = resolve_batch_check(settings, flags).spec
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    setup_logging(settings)
    directory = spec.directory.expanduser().resolve()
    refused = _startup(spec, settings, args, summary, directory)
    if refused is not None:
        return refused
    print_info(f"Scanning '{directory}' for batch submissions...")
    if spec.wait:
        return _execute_wait(spec, settings, args, summary, directory)
    try:
        result = asyncio.run(
            check_batches(
                directory,
                settings,
                formats=spec.formats,
                batch_ids=spec.batch_ids,
                args=args,
                **_check_options(spec),
            )
        )
    except CheckTargetError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    _report_check(result)
    stats = result.stats
    if result.refused:
        code = ExitCode.USAGE
    else:
        code = ExitCode.FAILURE if result.had_failure else ExitCode.OK
    return summary.emit(
        code,
        finalized=stats.finalized,
        pending=stats.pending,
        failed=stats.failed,
        **_work_fields(result.submissions),
        **_target_fields(spec, result.submissions, stats),
    )


def _work_fields(checks: Sequence[SubmissionCheck]) -> dict[str, Any]:
    """Return the documents, failed pages, files and tokens the finalized
    *checks* produced."""
    return {
        "complete": sum(check.complete for check in checks),
        "failed_pages": failed_units(checks),
        "outputs": [str(path) for path in check_outputs(checks)],
        USAGE_KEY: usage_field(check_usage(checks)),
    }


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


class _WaitReport:
    """The stderr lines of a wait: each pass, the waits and failed passes.

    The first pass prints the full report; later ones leave out the
    submissions not finished yet and the totals. A pass that failed with a
    retryable error is reported with the time of the next check. With
    *beat*, each sleep also writes the launcher's heartbeat file.
    """

    def __init__(self, beat: BatchHeartbeat | None = None) -> None:
        self._failed: tuple[int, Exception] | None = None
        self._beat = beat

    def on_pass(self, number: int, result: CheckResult) -> None:
        first = number == 1
        _report_check(result, pending=first, totals=first)
        if self._beat is not None:
            self._beat.on_pass(result)

    def on_retry(self, number: int, exc: Exception) -> None:
        self._failed = (number, exc)

    def on_wait(self, tick: WaitTick) -> None:
        if self._beat is not None:
            self._beat.on_wait(tick)
        at = (datetime.now() + timedelta(seconds=tick.next_in)).strftime("%H:%M")
        if self._failed is not None:
            self._warn_failed(f"; checking again at {at}")
            return
        print_info(
            f"Waiting for batches: {tick.pending} pending; next check at {at} "
            f"(check {tick.passes + 1}; {tick.elapsed / 3600:.2f} of "
            f"{tick.timeout / 3600:g} h)."
        )

    def flush(self) -> None:
        """Report a failed last pass that no wait followed."""
        if self._failed is not None:
            self._warn_failed("")

    def _warn_failed(self, then: str) -> None:
        assert self._failed is not None
        number, exc = self._failed
        self._failed = None
        print_warning(f"Check {number} failed ({classify_error(exc)}): {exc}{then}.")


def _end_wait(
    spec: BatchCheckSpec,
    ledger: WaitLedger,
    summary: Summary,
    code: int,
    stopped: str,
    stats: CheckStats | None = None,
) -> int:
    """Write the end line and the totals from the store; emit the summary."""
    stats = stats if stats is not None else final_totals(ledger)
    elapsed = ledger.elapsed()
    print_info(
        f"Stopped after {_plural(ledger.polls, 'check')} and "
        f"{elapsed / 3600:.2f} h: {stopped}."
    )
    _report_totals(stats, failed_units(ledger.checks))
    return summary.emit(
        code,
        finalized=stats.finalized,
        pending=stats.pending,
        failed=stats.failed,
        **_work_fields(ledger.checks),
        **_target_fields(spec, ledger.checks, stats),
        polls=ledger.polls,
        waited_seconds=int(elapsed),
        stopped=stopped,
    )


def _execute_wait(
    spec: BatchCheckSpec,
    settings: Settings,
    args: argparse.Namespace,
    summary: Summary,
    directory: Path,
) -> int:
    """Check until no batch work is pending or ``--wait-hours`` have passed.

    An interrupt or an error out of a pass emits the summary over the
    passes so far and propagates to :func:`guarded`.
    """
    ledger = WaitLedger()
    report = _WaitReport(BatchHeartbeat.from_env())
    limits = WaitLimits(spec.poll_minutes * 60, spec.wait_hours * 3600)
    try:
        polled = asyncio.run(
            check_until_settled(
                directory,
                settings,
                limits,
                ledger,
                formats=spec.formats,
                batch_ids=spec.batch_ids,
                args=args,
                on_pass=report.on_pass,
                on_wait=report.on_wait,
                on_retry=report.on_retry,
                **_check_options(spec),
            )
        )
    except CheckTargetError as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(
            ExitCode.USAGE, polls=ledger.polls, waited_seconds=0, stopped=None
        )
    except KeyboardInterrupt:
        _end_wait(spec, ledger, summary, ExitCode.INTERRUPTED, "interrupted")
        raise
    except StoreError as exc:
        _end_wait(spec, ledger, summary, exc.exit_code, "error")
        raise
    except Exception:
        _end_wait(spec, ledger, summary, ExitCode.FAILURE, "error")
        raise
    report.flush()
    if polled.stop is Stop.TIMEOUT and polled.last is not None:
        for check in polled.last.submissions:
            if check.outcome == PENDING:
                _report_submission(check)
    stats = final_totals(ledger)
    if ledger.refused:
        code = ExitCode.USAGE
    elif (
        polled.stop is Stop.TIMEOUT
        or stats.had_failure
        or failed_units(ledger.checks) > 0
    ):
        code = ExitCode.FAILURE
    else:
        code = ExitCode.OK
    return _end_wait(spec, ledger, summary, code, str(polled.stop), stats)


def execute_check(args: argparse.Namespace) -> int:
    """Run the parsed batch check options and return the exit code."""
    flags = BATCH_CHECK_OPTIONS.values(args)
    summary = Summary(sys.stdout if flags.get("json") else None, _check_base(flags))
    return guarded(summary, lambda: _execute_check(args, summary))


def _report_cancel(report: CancelReport) -> None:
    listing = report.listing
    if listing is not None:
        for provider in listing.skipped:
            print_info(f"{provider}: key variable not set; not listed.")
        for provider, error in listing.errors.items():
            print_error(f"{provider}: listing failed: {error}")
        print_info(f"Found {len(listing.batches)} unfinished batch(es) to cancel.")
    for path, reason in report.refused.items():
        print_error(f"{path}: {reason}")
    for skipped in report.skipped:
        print_info(f"Not cancelled, no provider batch: {skipped}")
    for outcome in report.outcomes:
        if outcome.cancelled:
            print_success(f"Cancelled: {outcome.batch_id} ({outcome.provider})")
        else:
            print_error(f"Failed to cancel {outcome.batch_id}: {outcome.error}")
    print_header("CANCELLATION COMPLETE", "")
    print_success(f"Successfully cancelled: {report.cancelled}")
    if report.failed:
        print_error(f"Failed: {report.failed}")


def _cancel(
    directory: Path | None,
    batch_ids: tuple[str, ...],
    settings: Settings,
    args: argparse.Namespace,
) -> Coroutine[Any, Any, CancelReport]:
    if directory is not None:
        print_info(f"Cancelling the open batch submissions below '{directory}'...")
        return cancel_submissions(directory, settings, args=args)
    if batch_ids:
        print_info(f"Targeting {len(batch_ids)} specific batch(es)...")
        return cancel_batches(batch_ids, settings, args=args)
    print_info("Listing the unfinished batches of every provider...")
    return cancel_unfinished(settings, args=args)


def _execute_cancel(args: argparse.Namespace, summary: Summary) -> int:
    flags = BATCH_CANCEL_OPTIONS.values(args)
    setup_logging()
    try:
        settings = load_settings(flags)
        spec = resolve_batch_cancel(settings, flags).spec
    except (SettingsError, SpecError) as exc:
        print_error(f"Invalid arguments: {exc}")
        return summary.emit(ExitCode.USAGE)
    setup_logging(settings)
    directory = None
    if spec.directory is not None:
        directory = spec.directory.expanduser().resolve()
    refused = _startup(spec, settings, args, summary, directory)
    if refused is not None:
        return refused
    print_header("BATCH CANCELLATION", "")
    report = asyncio.run(_cancel(directory, spec.batch_ids, settings, args))
    _report_cancel(report)
    code = ExitCode.USAGE if report.refused else exit_code_for(failed=report.failed)
    return summary.emit(code, cancelled=report.cancelled, failed=report.failed)


def execute_cancel(args: argparse.Namespace) -> int:
    """Run the parsed batch cancel options and return the exit code."""
    flags = BATCH_CANCEL_OPTIONS.values(args)
    summary = Summary(sys.stdout if flags.get("json") else None, cancel_fields())
    return guarded(summary, lambda: _execute_cancel(args, summary))
