"""The new-transcription task of the guided UI: steps, review and run.

:func:`flow` assembles the task for :func:`chronotranscriber.ui.session.build_wizard`.
Its run resolves the answers, plans, starts the extension and runs the plan
under the rich progress display. Exit codes are those of
``chronotranscriber transcribe``: 0 when every item completed or nothing was
left, 1 when an item failed or another run leases one, 2 for usage and
configuration errors.
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from chronotranscriber import ext
from chronotranscriber import run as core
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.wizard import State
from chronotranscriber.settings import SettingsError, resolve
from chronotranscriber.spec import OPTIONS, Resolution, RunSpec
from chronotranscriber.ui.progress import RichProgress, plan_table, summary_table
from chronotranscriber.ui.review import PROGRAM, build_review, equivalent_command
from chronotranscriber.ui.session import TASKS, RunRequest, Session, TaskFlow, plural
from chronotranscriber.ui.wizard import build_steps

__all__ = ["flow"]


def _say(console: Console, text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


def _resolve(state: State, session: Session) -> Resolution[RunSpec]:
    return resolve(session.settings, state.answers)


def _nothing_left(run_plan: core.RunPlan) -> str:
    text = f"Nothing to do: {plural(len(run_plan.skipped), 'item')} already complete"
    left_out = run_plan.not_processed - len(run_plan.skipped)
    if left_out:
        text += f", {left_out} left out"
    return text + "."


def _run(request: RunRequest) -> int:
    session, console = request.session, request.console
    settings = session.settings
    try:
        resolved = resolve(settings, dict(request.answers))
    except ValueError as exc:
        _say(console, f"Error: {exc}")
        return int(ExitCode.USAGE)
    try:
        run_plan = core.plan(
            resolved.spec, settings, sources=resolved.sources, args=request.args
        )
    except core.PlanError as exc:
        _say(console, f"Error: {exc}")
        return int(exc.exit_code)
    idle = not (run_plan.to_process or run_plan.render_only or run_plan.refused)
    if not run_plan.dry_run and idle:
        _say(console, _nothing_left(run_plan))
        return int(ExitCode.FAILURE if run_plan.leased else ExitCode.OK)
    if run_plan.dry_run or run_plan.to_process:
        try:
            ext.startup(run_plan.spec, settings=settings, args=request.args)
        except SettingsError as exc:
            _say(console, f"Error: {exc}")
            return int(ExitCode.USAGE)
    if run_plan.dry_run:
        for warning in run_plan.warnings:
            _say(console, f"Warning: {warning}")
        console.print(plan_table(run_plan))
        return int(ExitCode.OK)
    try:
        with RichProgress(console) as progress:
            for warning in run_plan.warnings:
                progress.warning(warning)
            result = asyncio.run(core.run(run_plan, progress))
    except core.PlanError as exc:
        _say(console, f"Error: {exc}")
        return int(exc.exit_code)
    console.print(summary_table(result, progress))
    return int(result.exit_code)


def flow() -> TaskFlow:
    """Return the new-transcription task."""
    return TaskFlow(
        name="transcribe",
        title=TASKS["transcribe"],
        program=PROGRAM,
        table=OPTIONS,
        steps=build_steps,
        review=build_review,
        command=equivalent_command,
        resolve=_resolve,
        run=_run,
    )
