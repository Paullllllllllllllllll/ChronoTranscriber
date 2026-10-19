"""The repair task of the guided UI: choose the failed pages and repair them.

The steps follow the repair options: the transcription (a file, its input,
a database or a folder), the failure types and page indices, the model, the
response, the page images and the run mode. The review previews the plan
(the pages to repair per document and the warnings) and withholds Run when
the repair cannot be planned. The run mirrors ``chronotranscriber repair``:
plan, the extension's startup on dry runs and when pages are to repair,
then the repairs with rich progress.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Final

from rich.text import Text

from chronotranscriber import ext
from chronotranscriber.batch.repair import (
    RepairPlanError,
    RepairSet,
    batch_refusal,
    plan_repairs,
    run_repairs,
)
from chronotranscriber.common.batch import supports_batch
from chronotranscriber.common.contract import ExitCode, exit_code_for
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    Choice,
    Line,
    Nav,
    Review,
    State,
    Step,
    StepContext,
)
from chronotranscriber.settings import SettingsError, resolve_repair
from chronotranscriber.spec import (
    REPAIR_CAUSES,
    REPAIR_OPTIONS,
    RepairSpec,
    Resolution,
    to_argv,
)
from chronotranscriber.ui.progress import RichProgress
from chronotranscriber.ui.questions import (
    IMAGE_OPTIONS,
    MODEL_OPTIONS,
    RESPONSE_OPTIONS,
    ImageQuestions,
    ModelQuestions,
    Question,
    ResponseQuestions,
    images_line,
    model_line,
    resolution_of,
    response_line,
)
from chronotranscriber.ui.results import repair_result
from chronotranscriber.ui.session import (
    TASKS,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
)

__all__ = ["CAUSE_FLAGS", "PROGRAM", "cause_flags", "flow", "selected_causes"]

PROGRAM: Final = ("chronotranscriber", "repair")
CAUSE_FLAGS: Final = ("errors_only", "not_possible", "no_text", "all_failures")
_FAILURE_OPTIONS: Final = (*CAUSE_FLAGS, "indices")
_RUN_OPTIONS: Final = ("batch", "dry_run")

_CAUSE_TITLES: Mapping[str, str] = {
    "api_error": "Transcription errors",
    "not_possible": "Transcription not possible",
    "no_text": "No transcribable text",
}
_CAUSE_TEXTS: Mapping[str, str] = {
    "api_error": "errors",
    "not_possible": "not possible",
    "no_text": "no text",
}


def selected_causes(state: State) -> tuple[str, ...]:
    """Return the failure causes the state's cause flags select."""
    if state.value("all_failures"):
        return REPAIR_CAUSES
    wanted = {name for name in ("not_possible", "no_text") if state.value(name)}
    if state.value("errors_only") or not wanted:
        wanted.add("api_error")
    return tuple(cause for cause in REPAIR_CAUSES if cause in wanted)


def cause_flags(causes: Sequence[str]) -> dict[str, bool]:
    """Return the cause flags that spell *causes* on the command line."""
    wanted = set(causes)
    if wanted == set(REPAIR_CAUSES):
        return {"all_failures": True}
    flags: dict[str, bool] = {
        name: True for name in ("not_possible", "no_text") if name in wanted
    }
    if "api_error" in wanted and flags:
        flags["errors_only"] = True
    return flags


def _existing(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


def _ask_input(ctx: StepContext) -> Nav:
    current = ctx.state.value("input")
    answer = ctx.prompter.path(
        "Transcription file, its input, a database or a folder (subfolders included):",
        default="" if current is None else str(absolute(current)),
        validate=_existing,
    )
    if answer is BACK:
        return BACK
    ctx.state.set("input", absolute(answer))
    return DONE


def _ask_failures(ctx: StepContext) -> Nav:
    state = ctx.state

    def causes() -> Nav:
        while True:
            answer = ctx.prompter.checkbox(
                "Failure types to repair:",
                [Choice(cause, _CAUSE_TITLES[cause]) for cause in REPAIR_CAUSES],
                checked=selected_causes(state),
            )
            if answer is BACK:
                return BACK
            if answer:
                break
            ctx.prompter.show("  choose at least one failure type")
        for name in CAUSE_FLAGS:
            state.clear(name)
        for name, value in cause_flags(answer).items():
            state.set(name, value)
        return DONE

    def indices() -> Nav:
        return ctx.text_option(
            "indices",
            "Page indices among them, one document only (0,5,12 or 3-7; "
            "empty = every one):",
            empty=None,
        )

    return ctx.sequence([causes, indices])


def _batch_available(state: State, session: Session) -> bool:
    try:
        spec = resolution_of(state, session).spec
    except ValueError:
        return False
    return (
        supports_batch(spec.model.provider)
        and batch_refusal(spec, settings=session.settings, args=session.args) is None
    )


def _ask_run(ctx: StepContext, session: Session) -> Nav:
    state = ctx.state

    def mode() -> Nav | None:
        if not _batch_available(state, session):
            state.set("batch", False)
            return None
        return ctx.select_option(
            "batch",
            "Run mode:",
            [
                Choice(False, "Synchronous", "the pages are sent now"),
                Choice(True, "Batch", "through the provider's batch API"),
            ],
        )

    def dry_run() -> Nav:
        return ctx.select_option(
            "dry_run",
            "Dry run:",
            [
                Choice(False, "No, repair the pages"),
                Choice(True, "Yes, only report what would be repaired"),
            ],
        )

    questions: list[Question] = [mode, dry_run]
    return ctx.sequence(questions)


def _steps(session: Session) -> list[Step]:
    return [
        Step("transcription", "Transcription", ask=_ask_input, sets=("input",)),
        Step("failures", "Failures", ask=_ask_failures, sets=_FAILURE_OPTIONS),
        Step(
            "model",
            "Model",
            ask=lambda ctx: ctx.sequence(ModelQuestions(ctx, session).questions()),
            sets=MODEL_OPTIONS,
        ),
        Step(
            "response",
            "Response",
            ask=lambda ctx: ctx.sequence(ResponseQuestions(ctx, session).questions()),
            sets=RESPONSE_OPTIONS,
        ),
        Step(
            "images",
            "Page images",
            ask=lambda ctx: ctx.sequence(ImageQuestions(ctx, session).questions()),
            sets=IMAGE_OPTIONS,
        ),
        Step(
            "run",
            "Run",
            ask=lambda ctx: _ask_run(ctx, session),
            sets=_RUN_OPTIONS,
        ),
    ]


def _resolve(state: State, session: Session) -> Resolution[RepairSpec]:
    return resolve_repair(session.settings, state.answers)


def _failures_line(spec: RepairSpec) -> Line:
    selection = spec.selection
    causes = ", ".join(_CAUSE_TEXTS[cause] for cause in selection.causes)
    if selection.indices is None:
        pages = "every selected failure"
    else:
        pages = "indices " + ",".join(str(index) for index in selection.indices)
    return Line("failures", [causes, pages])


def _run_line(spec: RepairSpec) -> Line:
    parts = ["batch" if spec.run.batch else "sync"]
    if spec.run.dry_run:
        parts.append("dry run")
    return Line("run", parts)


PLAN_LIMIT: Final = 20


def _plan_notes(repairs: RepairSet) -> list[str]:
    notes: list[str] = []
    if repairs.message is not None:
        notes.append(repairs.message)
    entries = []
    for plan in repairs.plans:
        if plan.document is None:
            continue
        pages = ", ".join(str(index) for index in plan.failures)
        entries.append(
            f"Pages to repair: {len(plan.failures)} (indices {pages}) of "
            f"'{plan.document.name}' in {plan.document.database}"
        )
    notes += entries[:PLAN_LIMIT]
    if len(entries) > PLAN_LIMIT:
        notes.append(f"... and {len(entries) - PLAN_LIMIT} more documents")
    notes += [f"Warning: {warning}" for warning in repairs.warnings]
    return notes


def _review(state: State, session: Session) -> Review:
    try:
        resolution = _resolve(state, session)
    except ValueError as exc:
        return Review([], withheld=str(exc))
    spec, sources = resolution.spec, resolution.sources
    lines = [
        Line("transcription", [str(absolute(spec.input))]),
        _failures_line(spec),
        model_line(spec.model, sources, session),
        response_line(spec.model, spec.prompt, sources, session),
        images_line(spec.model, spec.images, sources),
        _run_line(spec),
    ]
    try:
        repairs = plan_repairs(spec, session.settings, args=session.args)
    except RepairPlanError as exc:
        return Review(lines, withheld=str(exc))
    return Review(lines, notes=_plan_notes(repairs))


def _command(state: State, session: Session) -> list[str]:
    spec = _resolve(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    return command_words(
        PROGRAM, argv, session, REPAIR_OPTIONS, provider=spec.model.provider
    )


def _error(request: RunRequest, message: str) -> None:
    request.console.print(Text(message, style="red"), soft_wrap=True)


def _run(request: RunRequest) -> int:
    settings = request.session.settings
    flags: dict[str, Any] = dict(request.answers)
    try:
        spec = resolve_repair(settings, flags).spec
    except ValueError as exc:
        _error(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    try:
        repairs = plan_repairs(spec, settings, args=request.args)
    except RepairPlanError as exc:
        _error(request, f"Error: {exc}")
        return int(exc.exit_code)
    for warning in repairs.warnings:
        request.console.print(Text(f"Warning: {warning}", style="yellow"))
    if repairs.dry_run or repairs.to_repair:
        try:
            ext.startup(spec, settings=settings, args=request.args)
        except SettingsError as exc:
            _error(request, f"Error: {exc}")
            return int(ExitCode.USAGE)
    if repairs.dry_run or not repairs.to_repair:
        result = repairs.outcome()
    else:
        with RichProgress(request.console) as progress:
            result = asyncio.run(run_repairs(repairs, progress))
    request.console.print(repair_result(spec, result))
    return int(exit_code_for(failed=result.failed))


def flow() -> TaskFlow:
    """Return the repair task."""
    return TaskFlow(
        name="repair",
        title=TASKS["repair"],
        program=PROGRAM,
        table=REPAIR_OPTIONS,
        steps=_steps,
        review=_review,
        command=_command,
        resolve=_resolve,
        run=_run,
    )
