"""What a guided session shares: settings, tasks, the task-flow contract.

Every wizard task (a new transcription, a repair, batch check and cancel,
render, drop) is a :class:`TaskFlow`. :func:`build_wizard` puts the task
step first, so the user can switch tasks, then the task's own steps.
:class:`Inventory` holds what an input path holds.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import fitz

from chronotranscriber import ext
from chronotranscriber import run as core
from chronotranscriber.common.options import OptionTable
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    Choice,
    Nav,
    Prompter,
    Review,
    State,
    Step,
    StepContext,
    Wizard,
)
from chronotranscriber.settings import Settings
from chronotranscriber.sources import SourceItem, has_images, source_type
from chronotranscriber.sources.images import count_images
from chronotranscriber.spec import OPTIONS, Resolution, SpecError, resolve_spec

if TYPE_CHECKING:
    from rich.console import Console

__all__ = [
    "INVENTORY",
    "TASKS",
    "TASK_STEP",
    "TITLE",
    "Inventory",
    "RunRequest",
    "Session",
    "TaskFlow",
    "TaskSwitch",
    "absolute",
    "build_wizard",
    "can_reset",
    "command_words",
    "has_settings_default",
    "inventory_of",
    "new_state",
    "plural",
    "take_inventory",
    "task_step",
]

logger = logging.getLogger(__name__)

TITLE: Final = "ChronoTranscriber"
INVENTORY: Final = "inventory"
TASK_STEP: Final = "task"

TASKS: Mapping[str, str] = MappingProxyType(
    {
        "transcribe": "new transcription",
        "repair": "repair a transcription",
        "batch-check": "check batches",
        "batch-cancel": "cancel batches",
        "render": "render files again",
        "drop": "drop documents from the store",
    }
)
"""The wizard tasks in menu order, by name, with their titles."""


@dataclass(frozen=True)
class Session:
    """The settings of a guided session and where they came from.

    ``settings_path`` is the ``--settings`` file the ui was started with;
    ``environ`` is where the review looks up API key variables; ``args``
    holds the parsed ``ui`` arguments the wizard passes to the extension.
    """

    settings: Settings
    settings_path: Path | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    args: argparse.Namespace | None = None


def new_state(session: Session, table: OptionTable) -> State:
    """Return an empty answer state over *table* and its settings defaults."""
    defaults = {
        name: value
        for name, value in session.settings.defaults.items()
        if name in table and table[name].settable
    }
    return State(table, defaults)


_QUOTES = ('"', "'")


def absolute(path: Path | str) -> Path:
    """Make *path* absolute against the current directory.

    One matching pair of surrounding quotes is removed, as Explorer's
    "Copy as path" adds them, and ``~`` is expanded.
    """
    text = str(path).strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] == text[0]:
        text = text[1:-1]
    return Path(os.path.abspath(os.path.expanduser(text)))


def plural(count: int, word: str, many: str | None = None) -> str:
    """Return ``1 page`` or ``2 pages``."""
    return f"{count} {word if count == 1 else (many or word + 's')}"


def has_settings_default(state: State, name: str) -> bool:
    """Whether the settings defaults give option *name* a value."""
    current: str | None = name
    while current is not None:
        if current in state.defaults:
            return True
        current = state.table[current].fallback
    return False


def can_reset(state: State, name: str) -> bool:
    """Whether *name* may be reset to its unset code default.

    An unset value has no command-line spelling, so it cannot override a
    settings default; the wizard offers the reset only without one.
    """
    return not has_settings_default(state, name)


_OPENAI_ONLY: Final = "--service-tier"
_OFFLINE_COMMANDS: Final = frozenset({("render",), ("drop",)})
"""The commands that call no provider and so take no extension options."""


def _without_flag(argv: Sequence[str], flag: str) -> list[str]:
    """Return *argv* without *flag*, in both its spellings, and its value."""
    words: list[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
        elif word == flag:
            skip = True
        elif not word.startswith(f"{flag}="):
            words.append(word)
    return words


def command_words(
    program: Sequence[str],
    argv: Sequence[str],
    session: Session,
    table: OptionTable,
    *,
    provider: str | None = None,
) -> list[str]:
    """Return the full command: *program*, *argv*, settings and extension words.

    With a *provider* other than OpenAI, ``--service-tier`` is left out, as
    the review leaves out the tier. ``--settings`` is added when the ui was
    started with one and *table* has the option. The extension's own options
    given to the ui follow (:func:`chronotranscriber.ext.extra_argv`) where
    the command registers them: on the commands that call providers.
    """
    if provider is not None and provider != "openai":
        argv = _without_flag(argv, _OPENAI_ONLY)
    words = [*program, *argv]
    if session.settings_path is not None and "settings" in table:
        words += ["--settings", str(absolute(session.settings_path))]
    if tuple(program[1:]) not in _OFFLINE_COMMANDS:
        words += ext.extra_argv(session.args)
    return words


class TaskSwitch(Exception):
    """The user chose another task on the task step."""

    def __init__(self, task: str) -> None:
        super().__init__(task)
        self.task = task


def task_step(current: str) -> Step:
    """Return the task step: DONE for *current*, TaskSwitch for another task."""

    def ask(ctx: StepContext) -> Nav:
        answer = ctx.prompter.select(
            "Task:",
            [Choice(name, title) for name, title in TASKS.items()],
            default=current,
        )
        if answer is BACK:
            return BACK
        if answer != current:
            raise TaskSwitch(answer)
        return DONE

    return Step(TASK_STEP, "Task", ask=ask)


class _TaskWizard(Wizard):
    """A wizard whose task step counts as answered on its first visit.

    It serves a task the user chose on an earlier wizard's task step: the
    first pass skips the task step without showing it, and Back from the
    task's first step returns to it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_chosen = True

    def _ask(self, step: Step) -> Nav:
        if self._task_chosen and step.name == TASK_STEP:
            self._task_chosen = False
            self.visited.add(step.name)
            return DONE
        return super()._ask(step)


@dataclass(frozen=True)
class RunRequest:
    """What a task runs with after the wizard: answers, console and arguments.

    ``args`` holds the parsed ``ui`` arguments for the extension.
    """

    session: Session
    answers: Mapping[str, Any]
    console: Console
    args: argparse.Namespace


@dataclass(frozen=True)
class TaskFlow:
    """One wizard task: its steps, review, command, spec and run.

    ``program`` holds the leading command words, such as
    ``("chronotranscriber", "batch", "check")``; ``steps`` returns the task's
    steps without the task step; ``command`` returns the full equivalent
    command, program first; ``run`` returns the exit code.
    """

    name: str
    title: str
    program: tuple[str, ...]
    table: OptionTable
    steps: Callable[[Session], Sequence[Step]]
    review: Callable[[State, Session], Review]
    command: Callable[[State, Session], list[str]]
    resolve: Callable[[State, Session], Resolution[Any]]
    run: Callable[[RunRequest], int]


def build_wizard(
    flow: TaskFlow,
    session: Session,
    prompter: Prompter,
    *,
    width: int = 80,
    chosen: bool = False,
) -> Wizard:
    """Return the wizard of *flow*: the task step, then the task's steps.

    The extension steps follow only for the transcribe options. *chosen*
    means the user chose the task on an earlier wizard's task step: the
    wizard then starts at the task's first step, and Back from it returns
    to the task step.
    """
    steps = [task_step(flow.name), *flow.steps(session)]
    if flow.table is OPTIONS:
        steps += ext.wizard_steps()
    wizard = _TaskWizard if chosen else Wizard
    return wizard(
        new_state(session, flow.table),
        prompter,
        steps,
        title=TITLE,
        task=flow.title,
        review=lambda state: flow.review(state, session),
        command=lambda state: flow.command(state, session),
        width=width,
    )


_KINDS: tuple[tuple[str, str, str | None], ...] = (
    ("pdf", "PDF", "page"),
    ("images", "image folder", "image"),
    ("epub", "EPUB", None),
    ("mobi", "MOBI", None),
    ("audio", "audio file", None),
)
_ORDER = tuple(kind for kind, _label, _unit in _KINDS)
_LABELS = {kind: (label, unit) for kind, label, unit in _KINDS}
_PROBE_METHODS = {"audio": "whisper"}


def _page_count(path: Path) -> int | None:
    try:
        with fitz.open(path) as document:
            return int(document.page_count)
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF has no count
        logger.debug("Could not count the pages of %s: %s", path, exc)
        return None


def _count(item: SourceItem) -> int | None:
    if item.type == "pdf":
        return _page_count(item.path)
    if item.type == "images":
        try:
            return count_images(item.path)
        except OSError:
            return None
    return None


@dataclass(frozen=True)
class Inventory:
    """The items found at an input path, with page or image counts.

    PDFs count pages and image folders images; other items have no count.
    """

    path: Path
    items: tuple[SourceItem, ...]
    counts: tuple[int | None, ...]
    recursive: bool = False

    @property
    def names(self) -> list[str]:
        """The item names."""
        return [item.name for item in self.items]

    def describe(self) -> str:
        """Return ``1 PDF, 212 pages; 1 EPUB`` or ``nothing to process``."""
        parts: list[str] = []
        for kind, label, unit in _KINDS:
            counts = [
                count
                for item, count in zip(self.items, self.counts, strict=True)
                if item.type == kind
            ]
            if not counts:
                continue
            text = plural(len(counts), label)
            if unit is not None:
                total = sum(count or 0 for count in counts)
                unknown = " or more" if any(count is None for count in counts) else ""
                text += f", {plural(total, unit)}{unknown}"
            parts.append(text)
        return "; ".join(parts) if parts else "nothing to process"

    def label(self, index: int) -> str:
        """Return one numbered list line for the item at 0-based *index*."""
        item, count = self.items[index], self.counts[index]
        detail, unit = _LABELS.get(item.type, (item.type, None))
        if unit is not None and count is not None:
            detail += f", {plural(count, unit)}"
        return f"{index + 1:>4}  {item.name}  ({detail})"


def _probe(path: Path, kind: str, recursive: bool) -> list[SourceItem]:
    """Return what a run of type *kind* would discover at *path*."""
    flags: dict[str, Any] = {"input": path, "type": kind}
    if kind in _PROBE_METHODS:
        flags["method"] = _PROBE_METHODS[kind]
    if kind != "auto":
        flags["recursive"] = recursive
    try:
        return core.discover(resolve_spec(flags).spec)
    except (core.PlanError, SpecError):
        return []


def _discover(path: Path, recursive: bool) -> list[SourceItem]:
    if path.is_file():
        kind = source_type(path)
        return [SourceItem(path, kind)] if kind is not None else []
    if not path.is_dir():
        return []
    if recursive:
        found = [
            item
            for kind in _ORDER
            for item in _probe(path, kind, True)
            if kind != "images" or has_images(item.path)
        ]
    else:
        found = [*_probe(path, "auto", False), *_probe(path, "audio", False)]
    if has_images(path) and not any(item.path == path for item in found):
        found.append(SourceItem(path, "images"))
    unique = {(item.path, item.type): item for item in found}
    return sorted(unique.values(), key=lambda item: _ORDER.index(item.type))


def take_inventory(path: Path, *, recursive: bool = False) -> Inventory:
    """Discover the items at *path* and count their pages or images.

    A folder is probed like auto mode (PDFs, image folders, EPUB and MOBI
    files) plus its audio files; *recursive* searches the subfolders as
    ``--recursive`` does for each type. A folder that holds images itself
    counts as one image folder.
    """
    items = tuple(_discover(path, recursive))
    return Inventory(path, items, tuple(_count(item) for item in items), recursive)


def inventory_of(state: State, path: Path) -> Inventory:
    """Return the stored inventory of *path*, taking it when missing.

    The search is recursive when the state's ``recursive`` option is on.
    """
    recursive = bool("recursive" in state.table and state.value("recursive"))
    stored = state.data.get(INVENTORY)
    if (
        isinstance(stored, Inventory)
        and stored.path == path
        and stored.recursive == recursive
    ):
        return stored
    inventory = take_inventory(path, recursive=recursive)
    state.data[INVENTORY] = inventory
    return inventory
