"""Review screen of a new transcription: summary lines, plan preview, command.

The review resolves the answers like ``chronotranscriber transcribe`` would,
marks values taken from the settings defaults and previews the plan with
:func:`chronotranscriber.run.plan` (no API calls, no writes). Run is withheld
while the plan fails (the batch API, the model's image input) and, except
for a dry run, while an API key variable the run needs is unset; the review
names the missing variable. The equivalent command spells out every value
that differs from the code default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from chronotranscriber import ext
from chronotranscriber import run as core
from chronotranscriber.common.options import Source
from chronotranscriber.common.wizard import (
    Line,
    Part,
    Review,
    ReviewAction,
    State,
    access_line,
)
from chronotranscriber.spec import OPTIONS, RunSpec, to_argv
from chronotranscriber.ui.questions import (
    images_line,
    model_line,
    resolution_of,
    response_line,
)
from chronotranscriber.ui.session import (
    Session,
    absolute,
    command_words,
    inventory_of,
    plural,
)
from chronotranscriber.ui.wizard import unit_word

__all__ = [
    "FORCE_ACTION",
    "PLAN_LIMIT",
    "PROGRAM",
    "build_review",
    "equivalent_command",
    "plan_notes",
]

PROGRAM: Final = ("chronotranscriber", "transcribe")
FORCE_ACTION: Final = "force"
PLAN_LIMIT: Final = 20
DATABASE_NAME: Final = "chronotranscriber.sqlite"

_Mark = Callable[[str, str], Part]

_ENGINES = {
    "tesseract": "Tesseract (local OCR)",
    "native": "native text (text layer or ebook text)",
}
_STATE_TEXTS = {"none": "new"}


def equivalent_command(state: State, session: Session) -> list[str]:
    """Return the words of the ``chronotranscriber transcribe`` command.

    Raises:
        ValueError: The answers do not resolve, or a settings default cannot
            be undone on the command line.
    """
    spec = resolution_of(state, session).spec
    argv = to_argv(spec, session.settings.defaults)
    return command_words(PROGRAM, argv, session, OPTIONS, provider=spec.model.provider)


def _input_line(state: State, spec: RunSpec, mark: _Mark) -> Line:
    path = absolute(spec.input.path)
    inventory = inventory_of(state, path)
    parts = [Part(f"{path}  ({inventory.describe()})")]
    if spec.input.files:
        parts.append(Part(f"files {', '.join(spec.input.files)}"))
    if spec.input.select:
        parts.append(mark("input.select", f"select {spec.input.select}"))
    if spec.input.all:
        parts.append(mark("input.all", "all items"))
    if spec.input.recursive:
        parts.append(mark("input.recursive", "recursive"))
    return Line("input", parts)


def _method_line(spec: RunSpec, mark: _Mark) -> Line:
    parts = [mark("input.type", f"type {spec.input.type}")]
    if spec.input.type == "auto":
        parts.append(
            mark("method.name", f"{spec.method.name} for scans and image folders")
        )
        layer = spec.method.searchable_pdfs == "text"
        searchable = "text layer" if layer else spec.method.name
        parts.append(
            mark("method.searchable_pdfs", f"{searchable} for searchable PDFs")
        )
    else:
        parts.append(mark("method.name", spec.method.name))
    return Line("method", parts)


def _output_line(spec: RunSpec, mark: _Mark) -> Line:
    output = spec.output
    if output.directory is None:
        parts = [mark("output.directory", "beside input")]
    else:
        parts = [
            mark("output.directory", str(absolute(output.directory))),
            mark("output.layout", f"layout {output.layout}"),
        ]
    formats = ", ".join(output.formats)
    parts.append(mark("output.formats", f"formats {formats} (+ {DATABASE_NAME})"))
    return Line("output", parts)


def _engine_line(
    spec: RunSpec, session: Session, sources: Mapping[str, Source]
) -> Line:
    method = spec.method.name
    if method == "llm":
        return model_line(spec.model, sources, session)
    if method == "audio-api":
        return model_line(spec.model, sources, session, audio=True)
    if method == "whisper":
        whisper = spec.audio.whisper_model
        marked = sources.get("audio.whisper_model") is Source.SETTINGS
        return Line("engine", [Part(f"Whisper {whisper}", marked)])
    return Line("engine", [Part(_ENGINES.get(method, method))])


def _access_line(spec: RunSpec, session: Session) -> Line | None:
    """Return how the model calls of the run authenticate.

    A batch run names the batch calls and, with the sync fallback, the
    synchronous calls of the fallback; audio API calls use an audio client.
    """
    model = spec.model

    def describe(kind: str) -> Callable[[], str]:
        return lambda: ext.describe_access(
            model.provider,
            model.endpoint,
            kind=kind,
            settings=session.settings,
            args=session.args,
        )

    if spec.method.name == "audio-api":
        routes = [("audio", describe("audio"))]
    elif spec.run.batch:
        routes = [("batch", describe("batch"))]
        if spec.run.sync_fallback:
            routes.append(("sync fallback", describe("chat")))
    else:
        routes = [("sync", describe("chat"))]
    return access_line(routes)


def _ocr_line(spec: RunSpec, mark: _Mark) -> Line:
    return Line(
        "ocr",
        [
            mark("ocr.config", f"config {spec.ocr.config}"),
            mark("ocr.preprocess", f"preprocess {spec.ocr.preprocess}"),
        ],
    )


def _audio_line(spec: RunSpec, mark: _Mark) -> Line:
    language = spec.audio.language
    return Line(
        "audio",
        [
            mark(
                "audio.language",
                "language detected" if language is None else f"language {language}",
            ),
            mark("audio.chunk_seconds", f"chunks of {spec.audio.chunk_seconds} s"),
        ],
    )


def _pages_line(state: State, spec: RunSpec) -> Line:
    unit = unit_word(state)
    pages = spec.input.pages
    return Line("pages", [Part(f"all {unit}" if pages is None else f"{unit} {pages}")])


def _run_line(spec: RunSpec, mark: _Mark) -> Line:
    run = spec.run
    batch = run.batch and spec.method.name == "llm"
    if batch:
        mode = "batch, sync fallback" if run.sync_fallback else "batch"
        parts = [mark("run.batch", mode)]
    else:
        parts = [Part("sync")]
    parts.append(mark("run.force", "force" if run.force else "resume"))
    if run.retry_errors:
        parts.append(mark("run.retry_errors", "retry errors"))
    if not batch or run.sync_fallback:
        parts.append(mark("run.concurrency", f"concurrency {run.concurrency}"))
    if run.keep_working_files:
        parts.append(mark("run.keep_working_files", "keep working files"))
    if run.dry_run:
        parts.append(mark("run.dry_run", "dry run"))
    return Line("run", parts)


def plan_notes(plan: core.RunPlan) -> list[str]:
    """Return the plan preview: counts, one line per item, warnings."""
    left_out = (*plan.unstored, *plan.blocked, *plan.refused)
    head = f"Plan: {len(plan.to_process)} to process, {len(plan.skipped)} complete"
    if left_out:
        head += f", {len(left_out)} left out"
    entries = [
        f"  {item.name}: {_STATE_TEXTS.get(item.state, item.state)} ({item.method})"
        for item in plan.to_process
    ]
    entries += [
        f"  {item.name}: complete "
        f"({'rendered again' if item.resume.render else 'skipped'})"
        for item in plan.skipped
    ]
    entries += [f"  {item.name}: {item.state} ({item.reason})" for item in left_out]
    lines = [head, *entries[:PLAN_LIMIT]]
    if len(entries) > PLAN_LIMIT:
        lines.append(f"  ... and {len(entries) - PLAN_LIMIT} more")
    lines += [f"Warning: {warning}" for warning in plan.warnings]
    return lines


def _force(state: State) -> None:
    state.set("force", True)


def build_review(state: State, session: Session) -> Review:
    """Return the review screen of the answers in *state*."""
    try:
        resolved = resolution_of(state, session)
    except ValueError as exc:
        return Review(lines=(), withheld=str(exc))
    spec, sources = resolved.spec, resolved.sources

    def mark(target: str, text: str) -> Part:
        return Part(text, sources.get(target) is Source.SETTINGS)

    method = spec.method.name
    lines = [
        _input_line(state, spec, mark),
        _method_line(spec, mark),
        _output_line(spec, mark),
        _engine_line(spec, session, sources),
    ]
    if method in ("llm", "audio-api"):
        access = _access_line(spec, session)
        if access is not None:
            lines.append(access)
    if method == "llm":
        lines += [
            response_line(spec.model, spec.prompt, sources, session),
            images_line(spec.model, spec.images, sources),
        ]
    elif method == "tesseract":
        lines.append(_ocr_line(spec, mark))
    elif method in ("whisper", "audio-api"):
        lines.append(_audio_line(spec, mark))
    lines += [_pages_line(state, spec), _run_line(spec, mark)]

    notes: list[str] = []
    actions: list[ReviewAction] = []
    withheld: str | None = None
    try:
        plan = core.plan(
            spec, session.settings, sources=sources, args=session.args, check_keys=False
        )
    except core.PlanError as exc:
        notes.append(f"Plan error: {exc}")
        withheld = "the plan failed; edit a step"
    else:
        notes += plan_notes(plan)
        if plan.missing_keys and not spec.run.dry_run:
            notes += [
                f"Missing API key: {key.variable} is not set (provider "
                f"{key.provider}; {key.needs})"
                for key in plan.missing_keys
            ]
            names = ", ".join(dict.fromkeys(key.variable for key in plan.missing_keys))
            withheld = f"set {names} in the environment"
        if plan.skipped and not spec.run.force:
            actions.append(
                ReviewAction(
                    FORCE_ACTION,
                    f"Transcribe {plural(len(plan.skipped), 'complete item')} "
                    "again (--force)",
                    _force,
                )
            )
    if spec.run.dry_run:
        notes.append("Dry run: Run shows the plan only, with no API calls or writes.")
    return Review(lines=lines, notes=notes, actions=actions, withheld=withheld)
