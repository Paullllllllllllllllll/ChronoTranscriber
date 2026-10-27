"""The render and drop tasks of the guided UI.

:func:`render_flow` re-renders the views of the tool's databases at an
input path; :func:`drop_flow` deletes chosen documents from a store. Their
runs report like ``chronotranscriber render`` and ``drop`` and return the
same exit codes.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.text import Text

from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.selection import is_numeric_selection, select_items
from chronotranscriber.common.store import (
    DocumentSelectionError,
    StoreError,
    VersionlessDatabaseError,
)
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
    OUTPUT_FORMATS,
    RENDER_OPTIONS,
    DropSpec,
    RenderSpec,
    Resolution,
    SpecError,
    resolve_drop,
    resolve_render,
    to_argv,
)
from chronotranscriber.ui.results import drop_result, render_result
from chronotranscriber.ui.session import (
    TASKS,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
    plural,
)

__all__ = ["drop_flow", "render_flow"]

logger = logging.getLogger(__name__)

RENDER_PROGRAM: Final = ("chronotranscriber", "render")
DROP_PROGRAM: Final = ("chronotranscriber", "drop")

_KEEP = "keep"
_CHOOSE = "choose"


def _path_text(text: str) -> str | None:
    return "enter a path" if not text.strip() else None


def _input_step(message: str, validate: Callable[[str], str | None]) -> Step:
    def ask(ctx: StepContext) -> Nav:
        current = ctx.state.value("input")
        answer = ctx.prompter.path(
            message,
            default="" if current is None else str(absolute(current)),
            validate=validate,
        )
        if answer is BACK:
            return BACK
        ctx.state.set("input", absolute(answer))
        return DONE

    return Step("input", "Input", ask=ask, sets=("input",))


def _flag_step(
    name: str, title: str, message: str, off: Choice[bool], on: Choice[bool]
) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return ctx.select_option(name, message, [off, on])

    return Step(name, title, ask=ask, sets=(name,))


def _dry_run_step(write: str) -> Step:
    return _flag_step(
        "dry_run",
        "Dry run",
        "Dry run:",
        Choice(False, "No", write),
        Choice(True, "Yes", "show the plan, write nothing"),
    )


def _formats_step() -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state
        chosen: dict[str, str] = {}

        def mode() -> Nav:
            current = _KEEP if state.value("formats") is None else _CHOOSE
            answer = ctx.prompter.select(
                "Formats:",
                [
                    Choice(_KEEP, "As rendered before", "each document's formats"),
                    Choice(_CHOOSE, "Choose formats"),
                ],
                default=current,
            )
            if answer is BACK:
                return BACK
            chosen["mode"] = answer
            if answer == _KEEP:
                state.set("formats", None)
            return DONE

        def formats() -> Nav | None:
            if chosen.get("mode") != _CHOOSE:
                return None
            checked = state.value("formats") or OUTPUT_FORMATS[:1]
            while True:
                answer = ctx.prompter.checkbox(
                    "Formats to render:",
                    [Choice(fmt, fmt) for fmt in OUTPUT_FORMATS],
                    checked=tuple(checked),
                )
                if answer is BACK:
                    return BACK
                if answer:
                    state.set("formats", tuple(answer))
                    return DONE
                ctx.prompter.show("  choose at least one format")

        return StepContext.sequence([mode, formats])

    return Step("formats", "Formats", ask=ask, sets=("formats",))


def _command(
    program: tuple[str, ...],
    resolve: Callable[[State, Session], Resolution[Any]],
) -> Callable[[State, Session], list[str]]:
    def command(state: State, session: Session) -> list[str]:
        spec = resolve(state, session).spec
        return command_words(program, to_argv(spec), session, state.table)

    return command


def _error(console: Console, message: str) -> None:
    console.print(Text(message, style="red"))


def _guarded(console: Console, body: Callable[[], int]) -> int:
    """Run *body* with the exit codes of the CLI for refusals and crashes."""
    try:
        return body()
    except StoreError as exc:
        _error(console, f"Error:{exc}")
        return exc.exit_code
    except KeyboardInterrupt:
        _error(console, "Interrupted by the user.")
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        _error(console, f"Unexpected error: {exc}")
        return ExitCode.FAILURE


def _resolve_render(state: State, session: Session) -> Resolution[RenderSpec]:
    return resolve_render(state.answers)


def _render_refusal(path: Path) -> str | None:
    """Return why render cannot run on *path*, or None."""
    if not path.exists():
        return f"Input path does not exist: {path}"
    if path.is_file() and path.name != DATABASE_NAME:
        return f"{path} is not a {DATABASE_NAME} file"
    return None


@dataclass(frozen=True)
class _RenderPreview:
    databases: list[Path]
    results: list[DocumentRender]
    refused: str | None = None


def _preview_render(path: Path, spec: RenderSpec) -> _RenderPreview:
    databases = find_databases(path)
    results: list[DocumentRender] = []
    for database in databases:
        try:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        except (StoreError, sqlite3.Error, OSError) as exc:
            return _RenderPreview(databases, results, refused=str(exc))
    return _RenderPreview(databases, results)


def _render_review(state: State, session: Session) -> Review:
    try:
        spec = _resolve_render(state, session).spec
    except SpecError as exc:
        return Review([Line("input", [str(state.value("input"))])], withheld=str(exc))
    path = absolute(spec.input)
    formats = ", ".join(spec.formats) if spec.formats else "as rendered before"
    lines = [
        Line("formats", [formats]),
        Line(
            "force",
            [
                "replace files changed outside the tool"
                if spec.force
                else "keep files changed outside the tool"
            ],
        ),
        Line("run", ["dry run" if spec.dry_run else "write the files"]),
    ]
    refusal = _render_refusal(path)
    if refusal is not None:
        return Review([Line("input", [str(path)]), *lines], withheld=refusal)
    preview = _preview_render(path, spec)
    found = (
        f"{plural(len(preview.databases), 'database')}, "
        f"{plural(len(preview.results), 'document')}"
    )
    lines.insert(0, Line("input", [str(path), found]))
    if preview.refused is not None:
        return Review(lines, withheld=preview.refused)
    written = sum(len(result.written) for result in preview.results)
    unchanged = sum(len(result.unchanged) for result in preview.results)
    kept = sum(len(result.kept) for result in preview.results)
    notes = [
        f"Would write {plural(written, 'file')}, {unchanged} unchanged, "
        f"{kept} left as edited."
    ]
    return Review(lines, notes=notes)


def _execute_render(spec: RenderSpec, console: Console) -> int:
    path = spec.input.expanduser().resolve()
    if not path.exists():
        _error(console, f"Error:Input path does not exist: {path}")
        return ExitCode.USAGE
    if path.is_file() and path.name != DATABASE_NAME:
        _error(console, f"Error:{path} is not a {DATABASE_NAME} file")
        return ExitCode.USAGE
    databases = find_databases(path)
    for database in databases:
        try:
            read_store(database.parent)
        except VersionlessDatabaseError as exc:
            _error(console, f"Error:{exc}")
            return ExitCode.USAGE
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
    console.print(render_result(results, spec.dry_run))
    failed = any(result.failed for result in results)
    return ExitCode.FAILURE if failed else ExitCode.OK


def _run_render(request: RunRequest) -> int:
    try:
        spec = resolve_render(dict(request.answers)).spec
    except SpecError as exc:
        _error(request.console, f"Error:{exc}")
        return ExitCode.USAGE
    return _guarded(request.console, lambda: _execute_render(spec, request.console))


def render_flow() -> TaskFlow:
    """Return the render task: input, formats, force and dry run."""

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(f"Input (folder or {DATABASE_NAME}):", _path_text),
            _formats_step(),
            _flag_step(
                "force",
                "Force",
                "Files changed outside the tool:",
                Choice(False, "Keep them"),
                Choice(True, "Replace them"),
            ),
            _dry_run_step("render the files"),
        ]

    return TaskFlow(
        name="render",
        title=TASKS["render"],
        program=RENDER_PROGRAM,
        table=RENDER_OPTIONS,
        steps=steps,
        review=_render_review,
        command=_command(RENDER_PROGRAM, _resolve_render),
        resolve=_resolve_render,
        run=_run_render,
    )


_DROP_NAMES = "drop_names"
_DOCUMENT_OPTIONS: Final = ("select", "files", "all")
_ALL = "all"
_RANGE = "range"
_SEARCH = "search"
_LIST = "list"
_LISTED = 30
"""The numbered documents shown before a number or range is asked."""


def _store_path(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    try:
        find_database(absolute(text))
    except DropInputError as exc:
        return str(exc)
    return None


def _stored_names(state: State) -> list[str]:
    """Return the names of the documents stored at the input, in name order.

    The names are kept per input path; an unreadable store has none.
    """
    current = state.value("input")
    if current is None:
        return []
    path = absolute(current)
    stored = state.data.get(_DROP_NAMES)
    if isinstance(stored, tuple) and stored[0] == path:
        return list(stored[1])
    names: list[str] = []
    try:
        reader = read_store(find_database(path).parent)
        if reader is not None:
            names = sorted(document.name for document in reader.documents())
    except (DropInputError, StoreError, sqlite3.Error, OSError) as exc:
        logger.debug("No stored documents at %s: %s", path, exc)
    state.data[_DROP_NAMES] = (path, tuple(names))
    return names


def _document_mode(state: State) -> str:
    if state.value("files"):
        return _LIST
    chosen = state.value("select")
    if chosen:
        return _RANGE if is_numeric_selection(str(chosen)) else _SEARCH
    return _ALL


def _clear_documents(state: State) -> None:
    for name in _DOCUMENT_OPTIONS:
        state.clear(name)


class _DocumentQuestions:
    """Which stored documents to drop.

    The choice is all, by number or range, by name search, or from a
    checklist of their names (``--files``).
    """

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.how: str | None = None

    def ask(self) -> Nav:
        return StepContext.sequence([self.which, self.numbers, self.search, self.pick])

    def which(self) -> Nav:
        names = _stored_names(self.state)
        choices = [Choice(_ALL, f"All {len(names)}")]
        if names:
            choices += [
                Choice(_RANGE, "By number or range", "such as 1-3 or 2,5"),
                Choice(_SEARCH, "By name search", "names that contain a text"),
                Choice(_LIST, "Choose from the list"),
            ]
        current = _document_mode(self.state)
        answer = self.ctx.prompter.select(
            "Which documents?",
            choices,
            default=current if any(c.value == current for c in choices) else _ALL,
        )
        if answer is BACK:
            return BACK
        self.how = answer
        if answer == _ALL:
            _clear_documents(self.state)
            self.state.set("all", True)
        return DONE

    def _set_select(self, expression: str) -> None:
        _clear_documents(self.state)
        self.state.set("select", expression)

    def numbers(self) -> Nav | None:
        if self.how != _RANGE:
            return None
        names = _stored_names(self.state)
        for index, name in enumerate(names[:_LISTED], start=1):
            self.ctx.prompter.show(f"  {index:>4}  {name}")
        if len(names) > _LISTED:
            self.ctx.prompter.show(f"  ... and {len(names) - _LISTED} more")

        def validate(text: str) -> str | None:
            if not is_numeric_selection(text):
                return "enter numbers or ranges, such as 1-3 or 2,5"
            result = select_items(names, text)
            if result.unmatched:
                return f"out of range: {', '.join(result.unmatched)} (1-{len(names)})"
            return None

        current = self.state.value("select")
        answer = self.ctx.prompter.text(
            f"Documents (1-{len(names)}):",
            default=str(current) if _document_mode(self.state) == _RANGE else "",
            validate=validate,
        )
        if answer is BACK:
            return BACK
        self._set_select(answer.strip())
        return DONE

    def search(self) -> Nav | None:
        if self.how != _SEARCH:
            return None
        names = _stored_names(self.state)

        def validate(text: str) -> str | None:
            if not text.strip():
                return "enter a part of a name"
            if is_numeric_selection(text):
                return "a number selects by position; choose By number or range"
            if not select_items(names, text).indices:
                return "no document name contains this text"
            return None

        current = self.state.value("select")
        answer = self.ctx.prompter.text(
            "Name contains:",
            default=str(current) if _document_mode(self.state) == _SEARCH else "",
            validate=validate,
        )
        if answer is BACK:
            return BACK
        self._set_select(answer.strip())
        matched = select_items(names, answer.strip()).pick(names)
        self.ctx.prompter.show(
            f"  {plural(len(matched), 'document')}: {', '.join(matched)}"
        )
        return DONE

    def pick(self) -> Nav | None:
        if self.how != _LIST:
            return None
        names = _stored_names(self.state)
        current = [name for name in self.state.value("files") or () if name in names]
        while True:
            answer = self.ctx.prompter.checkbox(
                "Documents:", [Choice(name, name) for name in names], checked=current
            )
            if answer is BACK:
                return BACK
            if answer:
                _clear_documents(self.state)
                self.state.set("files", tuple(answer))
                return DONE
            self.ctx.prompter.show("  select at least one document")


def _documents_step() -> Step:
    def ask(ctx: StepContext) -> Nav:
        return _DocumentQuestions(ctx).ask()

    return Step("documents", "Documents", ask=ask, sets=_DOCUMENT_OPTIONS)


def _resolve_drop(state: State, session: Session) -> Resolution[DropSpec]:
    return resolve_drop(state.answers)


def _selection_text(spec: DropSpec) -> str:
    if spec.all:
        return "all"
    if spec.files:
        return ", ".join(spec.files)
    return f"--select {spec.select}"


def _preview_drop(spec: DropSpec) -> DropOutcome | str:
    """Return the dry run of *spec*, or why it cannot run."""
    try:
        database = find_database(absolute(spec.input))
        return asyncio.run(
            drop_documents(
                database,
                select=spec.select,
                files=spec.files,
                select_all=spec.all,
                dry_run=True,
            )
        )
    except (DropInputError, DocumentSelectionError, StoreError) as exc:
        return str(exc)
    except (sqlite3.Error, OSError) as exc:
        return f"the store could not be read: {exc}"


def _drop_review(state: State, session: Session) -> tuple[Review, DropOutcome | None]:
    """Return the review and the dry run it shows, None when it cannot run."""
    try:
        spec = _resolve_drop(state, session).spec
    except SpecError as exc:
        line = Line("input", [str(state.value("input"))])
        return Review([line], withheld=str(exc)), None
    lines = [
        Line("input", [str(absolute(spec.input))]),
        Line("documents", [_selection_text(spec)]),
        Line("run", ["dry run" if spec.dry_run else "drop the documents"]),
    ]
    preview = _preview_drop(spec)
    if isinstance(preview, str):
        return Review(lines, withheld=preview), None
    report = preview.report
    notes = [*preview.warnings]
    if report.dropped:
        notes.append(f"Would drop: {', '.join(report.dropped)}")
    notes += [
        f"Refused {refusal.name or refusal.doc_id}: {refusal.reason}"
        for refusal in report.refused
    ]
    notes.append(
        f"Would delete {plural(len(report.deleted_files), 'file')}, keep "
        f"{len(report.kept_files)}."
    )
    notes += [f"Kept {kept.path}: {kept.reason}" for kept in report.kept_files]
    return Review(lines, notes=notes), preview


def _selected(outcome: DropOutcome) -> tuple[str, ...]:
    """Return the names of the documents a drop selected, dropped or refused."""
    report = outcome.report
    names = [*report.dropped, *(r.name for r in report.refused if r.name)]
    return tuple(sorted(names))


def _execute_drop(
    spec: DropSpec, console: Console, reviewed: tuple[str, ...] | None
) -> int:
    """Drop what *spec* selects.

    A real drop first checks that the selection still names the *reviewed*
    documents.
    """
    if not spec.dry_run:
        current = _preview_drop(spec)
        if isinstance(current, DropOutcome) and _selected(current) != reviewed:
            _error(
                console,
                "Error:The store changed since the review; the selection now "
                f"names {', '.join(_selected(current)) or 'nothing'}. Nothing "
                "was dropped; review the task again.",
            )
            return ExitCode.FAILURE
    try:
        database = find_database(spec.input)
        outcome = asyncio.run(
            drop_documents(
                database,
                select=spec.select,
                files=spec.files,
                select_all=spec.all,
                dry_run=spec.dry_run,
            )
        )
    except (DropInputError, DocumentSelectionError) as exc:
        _error(console, f"Error:{exc}")
        return ExitCode.USAGE
    console.print(drop_result(outcome))
    return ExitCode.FAILURE if outcome.report.refused else ExitCode.OK


def drop_flow() -> TaskFlow:
    """Return the drop task: store, documents and dry run.

    Run drops only the documents the last review listed: a store that
    changed in between makes it drop nothing.
    """
    reviewed: dict[str, tuple[str, ...]] = {}

    def review(state: State, session: Session) -> Review:
        shown, preview = _drop_review(state, session)
        reviewed.clear()
        if preview is not None:
            reviewed["names"] = _selected(preview)
        return shown

    def run(request: RunRequest) -> int:
        try:
            spec = resolve_drop(dict(request.answers)).spec
        except SpecError as exc:
            _error(request.console, f"Error:{exc}")
            return ExitCode.USAGE
        names = reviewed.get("names")
        return _guarded(
            request.console, lambda: _execute_drop(spec, request.console, names)
        )

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(f"Store (output folder or {DATABASE_NAME}):", _store_path),
            _documents_step(),
            _dry_run_step("drop the documents"),
        ]

    return TaskFlow(
        name="drop",
        title=TASKS["drop"],
        program=DROP_PROGRAM,
        table=DROP_OPTIONS,
        steps=steps,
        review=review,
        command=_command(DROP_PROGRAM, _resolve_drop),
        resolve=_resolve_drop,
        run=run,
    )
