"""Core entry points: discover items, plan a run, run it.

:func:`plan` turns a resolved :class:`~chronotranscriber.spec.RunSpec` and
the machine settings into a :class:`RunPlan` without API calls or writes: it
finds the items, applies ``--all``, ``--select`` or ``--files`` (a folder
with several items needs one of them), routes each item to a method (auto
mode decides per item), classifies resume from the store of each output
root, resolves the prompt inputs of LLM items and checks the key variable of
every provider the run needs, the provider's batch API for ``--batch`` and
the model's image input for LLM items. A dry run plans without the key
variables and lists the missing ones as warnings. :func:`run` processes the
plan's items on one event loop and reports through
:class:`~chronotranscriber.events.RunEvents`. Neither prints, prompts or
exits: a plan that cannot run raises a :class:`PlanError` that carries its
exit code.

Every output root (the output folder, or each input's folder when writing
beside the input) holds one ``chronotranscriber.sqlite``, the store of the
run's documents. Resume reads only the store: an item whose rendered output
exists without units in the store is ``unstored`` and skipped, an item
leased by another live run or waiting for a batch is ``blocked`` and
skipped, and an item whose stored pages used other image settings is
``refused``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chronotranscriber import ext
from chronotranscriber.batch import submit as batch_submit
from chronotranscriber.batch.repair import batch_refusal
from chronotranscriber.common.batch import supports_batch
from chronotranscriber.common.context import ContextError
from chronotranscriber.common.contract import ExitCode, exit_code_for
from chronotranscriber.common.decisions import dialect_for
from chronotranscriber.common.options import Source
from chronotranscriber.common.selection import (
    PageRange,
    parse_page_range,
    select_items,
)
from chronotranscriber.common.spec_view import resolved_view
from chronotranscriber.common.sqlite import DEFAULT_BUSY_TIMEOUT, DEFAULT_WARN_INTERVAL
from chronotranscriber.common.store import (
    Lease,
    LeaseKeeper,
    LeaseLostError,
    Store,
    StoreError,
    UnitRow,
    WriterOptions,
    bounded_error,
    file_sha256,
)
from chronotranscriber.common.usage import Usage
from chronotranscriber.events import (
    ItemFinished,
    ItemStarted,
    NullEvents,
    RunEvents,
)
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.capabilities import ensure_image_support
from chronotranscriber.llm.options import (
    model_capabilities,
    output_limit,
    response_format_for,
)
from chronotranscriber.llm.prompts import (
    PLAIN_PROMPT_PATH,
    SCHEMA_PROMPT_PATH,
    fallback_context,
    item_context,
    item_context_image,
    resolve_schema,
    system_prompt_path,
)
from chronotranscriber.methods import (
    VALID_PAIRS,
    Item,
    Job,
    NothingSelectedError,
    SubmitOutcome,
    Transcriber,
)
from chronotranscriber.methods.llm import image_record
from chronotranscriber.methods.units import DocumentHandle
from chronotranscriber.output.paths import item_paths, natural_sort_key
from chronotranscriber.output.render import format_path, render_item, text_formats
from chronotranscriber.output.store import (
    Classification,
    ItemState,
    classify,
    database_path,
    document_name,
    lease_reason,
    open_store,
)
from chronotranscriber.settings import Settings, SettingsError
from chronotranscriber.sources import (
    DiscoveryError,
    SourceItem,
    discover_items,
    source_type,
)
from chronotranscriber.sources.audio import chunk_settings, describe_chunks
from chronotranscriber.sources.audio.ffmpeg import configure_ffmpeg_executables
from chronotranscriber.sources.auto import decide, file_item, scan
from chronotranscriber.sources.images import resolve_image_settings
from chronotranscriber.spec import InputSpec, RunSpec, effective_spec

__all__ = [
    "BUSY_TIMEOUT",
    "BUSY_WARN_INTERVAL",
    "AmbiguousSelectionError",
    "ConfigurationError",
    "InputError",
    "ItemResult",
    "MissingKey",
    "NoItemsError",
    "PlanError",
    "PlannedItem",
    "RunPlan",
    "RunResult",
    "SelectionError",
    "StoreRefusedError",
    "choose_items",
    "discover",
    "item_location",
    "plan",
    "run",
    "selection_order",
]

logger = logging.getLogger(__name__)

_TRANSCRIBERS: Mapping[str, tuple[str, str]] = {
    "llm": ("chronotranscriber.methods.llm", "LlmTranscriber"),
    "tesseract": ("chronotranscriber.methods.tesseract", "TesseractTranscriber"),
    "native": ("chronotranscriber.methods.native", "NativeTranscriber"),
    "whisper": ("chronotranscriber.methods.whisper", "WhisperTranscriber"),
    "audio-api": ("chronotranscriber.methods.audio_api", "AudioApiTranscriber"),
}
"""The module and class of each method's transcriber, imported when a run
uses the method; :data:`VALID_PAIRS` routes the sources."""

_KEYED_METHODS = ("llm", "audio-api")
"""Methods that call a provider and so need its key variable."""

_EBOOKS = ("epub", "mobi")
"""Source types that never take the mirror layout."""

BUSY_TIMEOUT: float = DEFAULT_BUSY_TIMEOUT
"""Seconds a store write or read waits on a busy database before it backs off."""

BUSY_WARN_INTERVAL: float = DEFAULT_WARN_INTERVAL
"""Seconds between two warnings about the same busy database."""


class PlanError(Exception):
    """The run cannot be planned.

    *items_total* is the number of items the run would have processed, for
    the run summary; ``exit_code`` is the process exit code of the error.
    """

    exit_code: ExitCode = ExitCode.USAGE

    def __init__(self, message: str, *, items_total: int = 0) -> None:
        super().__init__(message)
        self.items_total = items_total


class ConfigurationError(PlanError):
    """The spec and settings cannot serve the run (exit 2)."""


class StoreRefusedError(ConfigurationError):
    """An output root's database is not a store, newer or unreadable (exit 2).

    The database is left as it is.
    """


class InputError(PlanError):
    """The input or output path cannot serve the run (exit 2)."""


class NoItemsError(PlanError):
    """The input holds nothing to process (exit 2)."""


class SelectionError(PlanError):
    """``--select`` matched no item (exit 2)."""


class AmbiguousSelectionError(PlanError):
    """Several items without ``--all``, ``--select`` or ``--files`` (exit 2)."""


@dataclass(frozen=True)
class MissingKey:
    """An API key variable the run needs that is not set.

    *needs* names what needs it, such as ``--batch`` or ``--method llm``.
    """

    variable: str
    provider: str
    needs: str

    @property
    def message(self) -> str:
        """The plan error of a run that needs the variable."""
        return (
            f"The API key variable {self.variable} of provider {self.provider} "
            f"is not set; {self.needs} needs it."
        )

    @property
    def warning(self) -> str:
        """The dry-run warning about the variable."""
        return (
            f"The API key variable {self.variable} of provider {self.provider} "
            f"is not set; the run itself needs it ({self.needs})."
        )


@dataclass(frozen=True)
class PlannedItem:
    """One discovered item: its method, resume state, store row and prompt.

    *resume* is the item's classification from the store of *root*, where
    the item is the document *document*; *output* is its rendered output
    path of the first format. *context* and *context_image* are resolved for
    LLM items; *chunks* describes the chunk count of a recording in a dry
    run.
    """

    source: SourceItem
    method: str
    resume: Classification
    root: Path
    document: str
    output: Path
    context: str | None = None
    context_image: Path | None = None
    chunks: str | None = None

    @property
    def path(self) -> Path:
        """The item's input path."""
        return self.source.path

    @property
    def name(self) -> str:
        """The item's file or folder name."""
        return self.source.name

    @property
    def type(self) -> str:
        """The item's source type."""
        return self.source.type

    @property
    def state(self) -> str:
        """The resume state: none, partial, complete, unstored, blocked or refused."""
        return self.resume.state.value

    @property
    def reason(self) -> str:
        """Why the item has its resume state."""
        return self.resume.reason


@dataclass(frozen=True)
class RunPlan:
    """What a run will do.

    *output_dir* None writes beside each input. *discovered* holds the items
    found and selected, in discovery order. *to_process* is in processing
    order (in auto mode grouped by method), *skipped* holds the complete
    items in discovery order; a complete item whose rendered files are
    missing or stale is rendered again without model calls. *unstored* items
    have outputs without store rows, *blocked* items are leased by another
    run or wait for a batch, *refused* items were stored with other image
    settings; none of them is processed. *warnings* are notes for the user.
    *sources* maps spec fields to where their values came from.
    *missing_keys* lists the key variables a dry run (or a plan that does
    not check them) found unset.
    """

    spec: RunSpec
    input_path: Path
    output_dir: Path | None
    discovered: tuple[SourceItem, ...]
    to_process: tuple[PlannedItem, ...]
    skipped: tuple[PlannedItem, ...]
    warnings: tuple[str, ...] = ()
    settings: Settings = field(default_factory=Settings)
    sources: Mapping[str, Source] = field(default_factory=dict)
    schema_path: Path | None = None
    pages: PageRange | None = None
    unstored: tuple[PlannedItem, ...] = ()
    blocked: tuple[PlannedItem, ...] = ()
    refused: tuple[PlannedItem, ...] = ()
    missing_keys: tuple[MissingKey, ...] = ()

    @property
    def auto(self) -> bool:
        """Whether the method is decided per item."""
        return self.spec.input.type == "auto"

    @property
    def batch(self) -> bool:
        """Whether LLM items are submitted to the provider's batch API.

        In auto mode this holds when ``--batch`` is set and items are routed
        to the LLM method.
        """
        return self.spec.run.batch and "llm" in self.methods

    @property
    def force(self) -> bool:
        """Whether existing outputs are reprocessed instead of resumed."""
        return self.spec.run.force

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.run.dry_run

    @property
    def methods(self) -> tuple[str, ...]:
        """The methods the run uses, checked for access at plan time.

        A run of one method names it even with nothing to process; in auto
        mode only the methods of the items to process count. The run opens
        them before its first item, and none when nothing is to process.
        """
        if self.auto:
            return tuple(dict.fromkeys(item.method for item in self.to_process))
        return (self.spec.method.name,)

    @property
    def render_only(self) -> tuple[PlannedItem, ...]:
        """The complete items whose rendered files are rendered again."""
        return tuple(item for item in self.skipped if item.resume.render)

    @property
    def not_processed(self) -> int:
        """The number of items skipped as complete, unstored or blocked."""
        return len(self.skipped) + len(self.unstored) + len(self.blocked)

    @property
    def leased(self) -> tuple[PlannedItem, ...]:
        """The blocked items another run's lease holds, not a batch."""
        return tuple(item for item in self.blocked if item.resume.leased_by)

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return every field's value and source as the store records them.

        The shape is ``{field: {"value": ..., "source": ...}}``.
        """
        return effective_spec(self.spec, self.sources)

    def resolved_spec(self) -> dict[str, dict[str, Any]]:
        """Return :meth:`effective_spec` as the dry run shows it.

        For a run of the LLM method the unset output limit, response format
        and image options carry the values the requests use, and a provider
        other than OpenAI shows no service tier
        (:func:`~chronotranscriber.common.spec_view.resolved_view`).
        """
        view = self.effective_spec()
        if not self.auto and self.spec.method.name != "llm":
            return view
        model = self.spec.model
        try:
            caps = model_capabilities(model, self.settings)
            images, _family, _section = resolve_image_settings(
                model.provider,
                model.model,
                self.spec.images,
                self.settings.images.original_max_side_px,
            )
        except (ValueError, SettingsError):
            return view
        return resolved_view(
            view,
            model="model",
            provider=model.provider,
            limit=output_limit(model, caps),
            response_format=response_format_for(model, caps),
            images=images,
        )


def _absolute(path: Path) -> Path:
    """Expand ``~`` and resolve *path* against the current directory."""
    return path.expanduser().resolve()


def _input_path(spec: RunSpec) -> Path:
    path = _absolute(spec.input.path)
    if not path.exists():
        raise InputError(f"Input path does not exist: {path}")
    if not (path.is_file() or path.is_dir()):
        raise InputError(f"Input path is neither a file nor directory: {path}")
    return path


def _output_dir(spec: RunSpec) -> Path | None:
    if spec.output.directory is None:
        return None
    path = _absolute(spec.output.directory)
    if path.exists() and not path.is_dir():
        raise InputError(f"Output path exists but is not a directory: {path}")
    return path


def _discover(spec: RunSpec, input_path: Path) -> tuple[list[SourceItem], list[str]]:
    """Return the items at *input_path* and notes about skipped files."""
    if spec.input.type != "auto":
        try:
            items = discover_items(
                input_path,
                spec.input.type,
                files=spec.input.files,
                recursive=spec.input.recursive,
            )
        except DiscoveryError as exc:
            raise InputError(str(exc)) from exc
        if not items:
            raise NoItemsError(f"No items found to process in: {input_path}")
        return items, []

    warnings: list[str] = []
    if input_path.is_file():
        single = file_item(input_path)
        items = [single] if single is not None else []
    else:
        found = scan(input_path)
        items = list(found.items)
        if found.loose_images:
            logger.warning(
                "Skipped %d loose image file(s) in %s: unsupported in auto mode.",
                found.loose_images,
                input_path,
            )
            warnings.append(
                "loose image files are not supported in auto mode; place them "
                "in a subfolder to process the folder."
            )
    if not items:
        raise NoItemsError(_auto_empty_message(input_path))
    return items, warnings


_AUDIO_HINT = " Audio recordings need --type audio."


def _auto_empty_message(input_path: Path) -> str:
    """Explain why auto mode found nothing to process at *input_path*."""
    if input_path.is_file():
        message = (
            f"Auto mode cannot take {input_path}: it takes PDF, EPUB and MOBI files."
        )
        if source_type(input_path) == "audio":
            message += _AUDIO_HINT
        return message
    message = f"No PDFs, image folders, EPUB or MOBI files found in: {input_path}."
    if any(
        entry.is_file() and source_type(entry) == "audio"
        for entry in input_path.iterdir()
    ):
        message += _AUDIO_HINT
    return message


def discover(spec: RunSpec) -> list[SourceItem]:
    """Return the items the run of *spec* finds at its input.

    The items come before ``--all`` or ``--select`` choose among them
    (``--files`` does apply), in discovery order; :func:`selection_order`
    gives the order ``--select`` numbers them in.

    Raises:
        PlanError: The input is missing or holds nothing to process.
    """
    return _discover(spec, _input_path(spec))[0]


def selection_order(items: Sequence[SourceItem]) -> list[SourceItem]:
    """Return *items* in the order ``--select`` numbers them.

    The order is by path, with numbers in names compared by value.
    """
    return sorted(items, key=lambda item: natural_sort_key(item.path.as_posix()))


def choose_items(
    items: list[SourceItem], selection: InputSpec, input_path: Path
) -> tuple[list[SourceItem], list[str]]:
    """Apply ``--all``, ``--select`` or ``--files``; one item needs none.

    ``--select`` numbers the items from 1 in :func:`selection_order` and
    otherwise searches their names; its out-of-range parts are warnings. The
    chosen items keep their discovery order.

    Raises:
        SelectionError: ``--select`` matched no item.
        AmbiguousSelectionError: Several items and no selection.
    """
    if selection.all or selection.files:
        return items, []
    if selection.select:
        ordered = selection_order(items)
        result = select_items([item.name for item in ordered], selection.select)
        warnings = []
        if result.unmatched:
            warnings.append(
                "Ignored out-of-range or invalid --select part(s): "
                f"{', '.join(result.unmatched)} (valid: 1-{len(items)})"
            )
        chosen = {id(item) for item in result.pick(ordered)}
        picked = [item for item in items if id(item) in chosen]
        if not picked:
            raise SelectionError(
                f"No item of {input_path} matches --select {selection.select!r}"
            )
        return picked, warnings
    if len(items) == 1:
        return items, []
    if selection.type == "auto":
        # plan() refuses --files in auto mode, so the hint names only the
        # two selections that work here.
        raise AmbiguousSelectionError(
            f"{len(items)} items found in {input_path}, but neither --all nor "
            "--select was given. Pass --all to process every item, or --select "
            "to choose them by number, range or name search."
        )
    raise AmbiguousSelectionError(
        f"{len(items)} items found in {input_path}, but none of --all, --select "
        "or --files was given. Pass --all to process every item, --select to "
        "choose them by number, range or name search, or --files to name them."
    )


def _route(item: SourceItem, spec: RunSpec) -> str:
    """Return the method of *item*: decided per item in auto mode."""
    if spec.input.type == "auto":
        method = decide(item, spec.method.name, spec.method.searchable_pdfs)
    else:
        method = spec.method.name
    if method not in VALID_PAIRS.get(item.type, ()):
        raise ConfigurationError(
            f"{item.type} items take --method "
            f"{' or '.join(VALID_PAIRS.get(item.type, ()))}, not {method}"
        )
    return method


def _by_method(items: list[PlannedItem]) -> list[PlannedItem]:
    """Group *items* by method, in the order the methods first appear."""
    methods = dict.fromkeys(item.method for item in items)
    order = {method: rank for rank, method in enumerate(methods)}
    return sorted(items, key=lambda item: order[item.method])


def _uses_llm(spec: RunSpec) -> bool:
    return spec.method.name == "llm" and spec.input.type in ("auto", "pdf", "images")


def _with_prompt(item: PlannedItem, spec: RunSpec, fallback: str | None) -> PlannedItem:
    """Return *item* with its context and context image resolved.

    *fallback* is the text of ``--fallback-context``, if any.
    """
    if item.method != "llm":
        return item
    try:
        context = item_context(item.path, spec.prompt.context, fallback)
        image = item_context_image(item.path, spec.prompt.context_image)
    except ContextError as exc:
        raise ConfigurationError(str(exc)) from exc
    return replace(item, context=context, context_image=image)


def _key_kind(spec: RunSpec, method: str) -> str:
    """Return the client kind (:data:`ext.CLIENT_KINDS`) of a keyed method."""
    if method == "audio-api":
        return "audio"
    return "batch" if spec.run.batch else "chat"


def _missing_keys(
    spec: RunSpec,
    settings: Settings,
    methods: tuple[str, ...],
    total: int,
    args: argparse.Namespace | None = None,
) -> list[MissingKey]:
    """Return the keyed methods none of whose key variables is set.

    The extension names the variables of each method's provider.

    Raises:
        ConfigurationError: The settings name no variable for the provider.
    """
    model = spec.model
    missing: list[MissingKey] = []
    for method in methods:
        if method not in _KEYED_METHODS:
            continue
        try:
            variables = ext.key_variables(
                model.provider,
                model.endpoint,
                kind=_key_kind(spec, method),
                settings=settings,
                args=args,
            )
        except SettingsError as exc:
            raise ConfigurationError(str(exc), items_total=total) from exc
        if variables and not any(os.environ.get(name) for name in variables):
            needs = "--batch" if spec.run.batch and method == "llm" else method
            if needs == method:
                needs = f"--method {method}"
            missing.append(MissingKey(variables[0], model.provider, needs))
    if "llm" in methods and spec.decision.mode != "off":
        dialect = dialect_for(spec.decision.model, spec.decision.provider)
        try:
            variables = ext.key_variables(
                dialect, kind="decision", settings=settings, args=args
            )
        except SettingsError as exc:
            raise ConfigurationError(str(exc), items_total=total) from exc
        if variables and not any(os.environ.get(name) for name in variables):
            needs = f"--decisions {spec.decision.mode}"
            missing.append(MissingKey(variables[0], dialect, needs))
    return missing


def _check_batch(
    spec: RunSpec,
    settings: Settings,
    args: argparse.Namespace | None,
    total: int,
) -> None:
    """Require a provider with a batch API and the extension's consent.

    Raises:
        ConfigurationError: The provider has no batch API or the extension
            refuses batch submission.
    """
    provider = spec.model.provider
    if not supports_batch(provider):
        raise ConfigurationError(
            f"--batch needs a provider with a batch API (openai, anthropic, "
            f"google), not {provider}",
            items_total=total,
        )
    refusal = batch_refusal(spec, settings=settings, args=args)
    if refusal is not None:
        raise ConfigurationError(refusal, items_total=total)


def _check_images(spec: RunSpec, settings: Settings, total: int) -> None:
    """Require a model that accepts the page images of LLM items.

    Raises:
        ConfigurationError: The model does not accept image input
            (:class:`~chronotranscriber.llm.capabilities.CapabilityError`) or
            its custom endpoint is undefined.
    """
    try:
        caps = model_capabilities(spec.model, settings)
        ensure_image_support(spec.model.model, caps)
    except ValueError as exc:
        raise ConfigurationError(str(exc), items_total=total) from exc


def _chunk_notes(
    items: list[PlannedItem], spec: RunSpec, settings: Settings
) -> list[PlannedItem]:
    """Return *items* with the chunk counts of their recordings."""
    if not any(item.type == "audio" for item in items):
        return items
    configure_ffmpeg_executables(settings.tools.ffmpeg_cmd, settings.tools.ffprobe_cmd)
    chunking = chunk_settings(spec, settings)
    return [
        replace(item, chunks=describe_chunks(item.path, item.method, chunking))
        if item.type == "audio"
        else item
        for item in items
    ]


def item_location(
    item: SourceItem,
    *,
    output_dir: Path | None,
    layout: str,
    input_root: Path,
    output_format: str = "txt",
) -> tuple[Path, str, Path]:
    """Return the output root, document name and output path of *item*.

    *output_dir* None writes beside the input; the document name does not
    depend on *output_format*, the suffix of the returned output path.
    """
    output = item_paths(
        item.path,
        output_dir=output_dir,
        layout=layout,
        input_root=input_root,
        output_format=output_format,
        mirror=item.type not in _EBOOKS,
    ).output
    root = output_dir if output_dir is not None else item.path.parent
    return root, document_name(output, root), output


@dataclass(frozen=True)
class _Classifier:
    """Classifies the items of one plan from the stores of their roots.

    *image* is the run's image-settings record.
    """

    spec: RunSpec
    output_dir: Path | None
    input_root: Path
    image: Mapping[str, Any] | None

    def locate(self, item: SourceItem) -> tuple[Path, str, Path]:
        """Return the output root, document name and output path of *item*."""
        return item_location(
            item,
            output_dir=self.output_dir,
            layout=self.spec.output.layout,
            input_root=self.input_root,
            output_format=text_formats(self.spec.output.formats)[0],
        )

    def planned(self, item: SourceItem, method: str) -> PlannedItem:
        """Return *item* planned with its resume state; write nothing.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        root, name, output = self.locate(item)
        formats = text_formats(self.spec.output.formats)
        outputs = [format_path(output, fmt) for fmt in formats]
        keyed = method in _KEYED_METHODS

        try:
            state = classify(
                root,
                name,
                outputs=outputs,
                image_settings=self.image if method == "llm" else None,
                model=self.spec.model.model if keyed else None,
                retry_errors=self.spec.run.retry_errors,
            )
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        if self.spec.run.force and state.state is not ItemState.BLOCKED:
            state = Classification(
                ItemState.NONE, name, state.doc_id, reason="overwrite mode"
            )
        return PlannedItem(item, method, state, root, name, output)


def _image(
    spec: RunSpec, settings: Settings, items: Sequence[SourceItem]
) -> Mapping[str, Any] | None:
    """Return the run's image-settings record when an item may take the LLM."""
    if not _uses_llm(spec) or not items:
        return None
    try:
        return image_record(spec, settings.images.original_max_side_px)
    except ValueError:
        return None


def _check_shared(items: Sequence[PlannedItem]) -> None:
    """Raise when two items would write the same output and store document.

    Document names compare case-insensitively, as the rendered files do on
    Windows and macOS.

    Raises:
        ConfigurationError: Two items share an output root and document name.
    """
    claimed: dict[tuple[Path, str], Path] = {}
    for item in items:
        key = (item.root, item.document.casefold())
        first = claimed.setdefault(key, item.path)
        if first != item.path:
            raise ConfigurationError(
                f"{first} and {item.path} would share the output {item.output} "
                "and its store document. Set --output DIR to keep them apart, or "
                "take one of them with --select."
            )


def plan(
    spec: RunSpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
    args: argparse.Namespace | None = None,
    check_keys: bool = True,
) -> RunPlan:
    """Discover, select, route and classify the items of a run; write nothing.

    *sources* maps spec fields to where their values came from; the plan
    reports them with the effective spec. *args* holds the parsed command
    line, passed to the extension's batch consent and key variables. A run
    whose key variables are unset raises unless it is a dry run or
    *check_keys* is False; the plan then lists them in ``missing_keys``, a
    dry run also as warnings.

    Raises:
        PlanError: The run cannot proceed; the subclass names the reason and
            ``exit_code`` the exit code. A database that is not a store, or
            is newer or unreadable, raises :class:`StoreRefusedError`.
    """
    if spec.input.type == "auto" and spec.input.files:
        raise ConfigurationError(
            "--files selects items of one input type; auto mode would plan "
            "every file in the folder. Set --type, or use --select or --all."
        )
    output_dir = _output_dir(spec)
    input_path = _input_path(spec)
    logger.info("Input=%s, Output=%s", input_path, output_dir or "beside input")
    found, warnings = _discover(spec, input_path)
    discovered, notes = choose_items(found, spec.input, input_path)
    warnings += notes

    image = _image(spec, settings, discovered)
    classifier = _Classifier(spec, output_dir, input_path, image)
    groups: dict[ItemState, list[PlannedItem]] = {state: [] for state in ItemState}
    planned_items = [
        classifier.planned(item, _route(item, spec)) for item in discovered
    ]
    _check_shared(planned_items)
    for planned in planned_items:
        groups[planned.resume.state].append(planned)
    to_process = _in_order(groups, discovered)
    if spec.input.type == "auto":
        to_process = _by_method(to_process)
    pages = parse_page_range(spec.input.pages) if spec.input.pages else None
    warnings += _nothing_selected(pages, groups[ItemState.COMPLETE])
    logger.info(
        "%d item(s) to process, %d already complete",
        len(to_process),
        len(groups[ItemState.COMPLETE]),
    )

    schema_path: Path | None = None
    if _uses_llm(spec):
        try:
            schema_path = resolve_schema(spec.prompt.schema)[1]
            system_prompt_path(spec.prompt.prompt)
        except (ValueError, FileNotFoundError) as exc:
            raise ConfigurationError(str(exc)) from exc
        fallback: str | None = None
        if spec.prompt.context is None and to_process:
            fallback, note = fallback_context(spec.prompt.fallback_context)
            if note is not None:
                warnings.append(note)
        to_process = [_with_prompt(item, spec, fallback) for item in to_process]

    run_plan = RunPlan(
        spec=spec,
        input_path=input_path,
        output_dir=output_dir,
        discovered=tuple(discovered),
        to_process=tuple(to_process),
        skipped=tuple(groups[ItemState.COMPLETE]),
        warnings=tuple(warnings),
        settings=settings,
        sources=dict(sources or {}),
        schema_path=schema_path,
        pages=pages,
        unstored=tuple(groups[ItemState.UNSTORED]),
        blocked=tuple(groups[ItemState.BLOCKED]),
        refused=tuple(groups[ItemState.REFUSED]),
    )
    total = len(to_process)
    if run_plan.batch:
        _check_batch(spec, settings, args, total)
    missing = _missing_keys(spec, settings, run_plan.methods, total, args)
    if missing and check_keys and not spec.run.dry_run:
        raise ConfigurationError(missing[0].message, items_total=total)
    if "llm" in run_plan.methods:
        _check_images(spec, settings, total)
    if missing:
        unset = tuple(key.warning for key in missing) if spec.run.dry_run else ()
        run_plan = replace(
            run_plan, missing_keys=tuple(missing), warnings=run_plan.warnings + unset
        )
    if spec.run.dry_run:
        chunked = _chunk_notes(to_process, spec, settings)
        run_plan = replace(run_plan, to_process=tuple(chunked))
    return run_plan


def _nothing_selected(
    pages: PageRange | None, complete: Sequence[PlannedItem]
) -> list[str]:
    """Return a warning for each complete item of which *pages* selects no unit.

    The units of an EPUB or MOBI document that ``--pages`` counts are its
    sections, unit 0 being the metadata header.
    """
    if pages is None:
        return []
    warnings = []
    for planned in complete:
        total = planned.resume.total - (1 if planned.type in ("epub", "mobi") else 0)
        if total > 0 and not pages.resolve(total):
            warnings.append(
                f"Page range '{pages.describe()}' selects none of the {total} "
                f"units of '{planned.name}', which is complete. Skipping."
            )
    return warnings


def _in_order(
    groups: Mapping[ItemState, list[PlannedItem]], discovered: Sequence[SourceItem]
) -> list[PlannedItem]:
    """Return the items to process (none or partial) in discovery order."""
    chosen = {
        id(item.source): item
        for state in (ItemState.NONE, ItemState.PARTIAL)
        for item in groups[state]
    }
    return [chosen[id(item)] for item in discovered if id(item) in chosen]


@dataclass(frozen=True)
class ItemResult:
    """The outcome of one item.

    *success* is False when the item raised or was refused; *error* then
    says why. *outputs* are the absolute paths of its rendered files.
    *submitted* means its pages went to a batch, which ``batch check``
    finalizes. *partial* marks a successful ``--pages`` item whose document
    is still short of complete; it counts neither complete nor failed.
    *failures* are the failed units of a failed item's document, with their
    error types and errors.
    """

    name: str
    success: bool
    outputs: tuple[str, ...] = ()
    error: str | None = None
    submitted: bool = False
    partial: bool = False
    failures: tuple[UnitRow, ...] = ()


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run.

    *usage* holds the tokens of the run's model calls per role, where the
    responses reported them. *databases* are the database files written.
    *blocked* names the items not processed because another run held them
    or a batch holds their pages, and *blocked_reasons* gives the reason of
    each, in the same order, a lease with its expiry; two items of the same
    name in different folders keep their own reasons. *leased* names those
    of them that another run's lease held; they make the run exit 1, while
    items waiting for a batch are submitted work. *unselected* names those
    of which ``--pages`` selects no unit. *rendered* are the files rendered
    again for complete items.
    """

    items: tuple[ItemResult, ...]
    skipped: tuple[str, ...] = ()
    seconds: float = 0.0
    usage: Mapping[str, Usage] = field(default_factory=dict)
    databases: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    rendered: tuple[str, ...] = ()
    unselected: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    leased: tuple[str, ...] = ()

    @property
    def total_usage(self) -> Usage:
        """The tokens of all roles together."""
        total = Usage()
        for usage in self.usage.values():
            total = total + usage
        return total

    @property
    def complete(self) -> int:
        """The number of items that finished complete.

        Batch submissions and items ``--pages`` left partial are left out.
        """
        return sum(
            1
            for item in self.items
            if item.success and not item.submitted and not item.partial
        )

    @property
    def partial(self) -> int:
        """The number of items whose documents ``--pages`` left partial."""
        return sum(1 for item in self.items if item.success and item.partial)

    @property
    def submitted(self) -> int:
        """The number of items whose pages went to a batch."""
        return sum(1 for item in self.items if item.success and item.submitted)

    @property
    def failed(self) -> int:
        """The number of items that failed."""
        return sum(1 for item in self.items if not item.success)

    @property
    def exit_code(self) -> ExitCode:
        """1 when an item failed or another run's lease held one, else 0."""
        return exit_code_for(failed=self.failed + len(self.leased))

    @property
    def outputs(self) -> list[str]:
        """The paths written: files rendered again, item files, then databases."""
        items = [path for item in self.items for path in item.outputs]
        return [*self.rendered, *items, *self.databases]


def _writer_options(events: RunEvents) -> WriterOptions:
    """Return the writer options of a run: busy waits warn through *events*."""

    def on_busy(path: Path, waited: float) -> None:
        events.warning(
            f"The database {path} is busy (another program holds it open); "
            f"waited {waited:.0f} s. Results are kept in memory and written "
            "once it is free."
        )

    return WriterOptions(
        busy_timeout=BUSY_TIMEOUT, warn_interval=BUSY_WARN_INTERVAL, on_busy=on_busy
    )


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s for the store: %s", path, exc)
        return None


class _Stores:
    """The stores of a run: one per output root, opened on first use.

    Opening a store stamps the run, stores the prompt and schema texts of
    LLM runs, marks submissions left ``submitting`` as unresolved and starts
    the heartbeat that renews the run's leases.
    """

    def __init__(self, run_plan: RunPlan, events: RunEvents) -> None:
        self._plan = run_plan
        self._events = events
        self._stores: dict[Path, Store] = {}
        self._keepers: list[LeaseKeeper] = []

    @property
    def databases(self) -> tuple[str, ...]:
        """The database files of the opened stores, in opening order."""
        return tuple(str(store.path.resolve()) for store in self._stores.values())

    async def get(self, root: Path) -> Store:
        """Return the open store of output root *root*.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        key = root.resolve()
        store = self._stores.get(key)
        if store is not None:
            return store
        key.mkdir(parents=True, exist_ok=True)
        store = open_store(key, writer=_writer_options(self._events))
        try:
            await store.open()
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        self._stores[key] = store
        await store.begin_run("transcribe", spec=self._plan.effective_spec())
        await self._artifacts(store)
        marked = await store.mark_stale_submissions()
        if marked:
            self._events.warning(
                f"{len(marked)} batch submission(s) in {database_path(key)} were "
                "left without a provider batch id and are now unresolved; their "
                "pages stay blocked. Run 'chronotranscriber batch check' to "
                "reconcile them."
            )
        keeper = store.heartbeat(on_lost=self._lost)
        keeper.start()
        self._keepers.append(keeper)
        return store

    def _lost(self, lease: Lease) -> None:
        self._events.warning(
            f"The lease on document {lease.doc_id} expired; its remaining units "
            "are not stored by this run."
        )

    async def _artifacts(self, store: Store) -> None:
        """Store the prompt and schema texts of an LLM run."""
        spec = self._plan.spec
        if "llm" not in self._plan.methods:
            return
        plain = spec.model.response_format == "text"
        prompt = spec.prompt.prompt
        default = PLAIN_PROMPT_PATH if plain else SCHEMA_PROMPT_PATH
        texts = {
            "prompt": _read_text(_absolute(prompt) if prompt else default),
            "schema": None if plain else _read_text(self._plan.schema_path),
        }
        for kind, text in texts.items():
            if text is not None:
                await store.store_artifact(kind, text)

    async def close(self, exit_code: int) -> None:
        """Stop the heartbeats, record the end of the run and close the stores."""
        for keeper in self._keepers:
            await keeper.stop()
        for store in self._stores.values():
            try:
                if store.run_id is not None:
                    await store.end_run(exit_code)
            except sqlite3.Error as exc:
                logger.warning("Could not record the run end in %s: %s", store, exc)
            try:
                await store.close()
            except sqlite3.Error as exc:
                logger.warning("Could not close %s: %s", store.path, exc)


async def _note(
    store: Store, doc_id: int | None, message: str, events: RunEvents
) -> None:
    events.warning(message)
    try:
        await store.add_note(message, doc_id=doc_id)
    except sqlite3.Error as exc:
        logger.warning("Could not record a note in %s: %s", store.path, exc)


def _input_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return file_sha256(path)
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return None


@dataclass(frozen=True)
class _Blocked:
    """An item whose document another run leased after planning."""

    reason: str


@dataclass
class _ItemRun:
    """One item of a run: its store document, lease, method and render.

    *held* is the reason the item could not be leased, once :meth:`lease`
    found another run holding it.
    """

    job: Job
    planned: PlannedItem
    stores: _Stores
    held: str | None = None

    async def lease(self) -> tuple[Store, int, Lease] | None:
        """Ensure the item's document and lease it; None when another holds it."""
        planned = self.planned
        store = await self.stores.get(planned.root)
        doc_id = await store.ensure_document(
            planned.document,
            input_path=planned.path,
            input_sha256=_input_sha256(planned.path),
            kind=planned.type,
        )
        lease = await store.acquire(doc_id)
        if lease is None:
            state = None
            with contextlib.suppress(StoreError, sqlite3.Error):
                state = store.resume_state(doc_id)
            self.held = lease_reason(
                state.lease_run_id if state is not None else None,
                state.lease_until if state is not None else None,
            )
            await _note(
                store,
                doc_id,
                f"Skipping '{planned.name}': another run holds its document "
                f"{planned.document} in {store.path} ({self.held}).",
                self.job.events,
            )
            return None
        return store, doc_id, lease

    async def render(
        self, store: Store, doc_id: int, lease: Lease, *, force: bool
    ) -> tuple[str, ...]:
        """Render the item's formats; return the files that hold the rendering."""
        report = await render_item(
            store, doc_id, self.job.formats, self.planned.output, lease, force=force
        )
        for message in report.warnings:
            self.job.events.warning(message)
        current = {*report.written, *report.unchanged}
        targets = [format_path(self.planned.output, name) for name in self.job.formats]
        return tuple(str(path) for path in targets if path in current)

    async def render_or_note(
        self, store: Store, doc_id: int, lease: Lease, *, force: bool
    ) -> tuple[str, ...]:
        """Render like :meth:`render`; note a lost lease or a file error instead."""
        name = self.planned.name
        try:
            return await self.render(store, doc_id, lease, force=force)
        except LeaseLostError as exc:
            message = f"'{name}' was not rendered: {exc}"
        except OSError as exc:
            target = exc.filename or ", ".join(
                str(format_path(self.planned.output, fmt)) for fmt in self.job.formats
            )
            message = (
                f"'{name}': {target} was not rendered ({exc}); its units are "
                "stored and a later render or run replaces the file."
            )
        await _note(store, doc_id, message, self.job.events)
        return ()

    async def transcribe(
        self,
        store: Store,
        doc_id: int,
        lease: Lease,
        transcriber: Transcriber,
        pages: PageRange | None,
    ) -> tuple[tuple[str, ...], str | None]:
        """Run the method, then render; return the outputs and the error."""
        planned, job = self.planned, self.job
        spec = job.spec
        if planned.resume.note:
            await _note(store, doc_id, planned.resume.note, job.events)
        if planned.context:
            await store.store_artifact("context", planned.context)
        handle = DocumentHandle(
            store,
            doc_id,
            planned.document,
            lease,
            model=spec.model.model,
            provider=spec.model.provider,
        )
        item = Item(
            path=planned.path,
            source=planned.type,
            method=planned.method,
            pages=pages,
            context=planned.context,
            context_image=planned.context_image,
            document=handle,
        )
        returned: tuple[str, ...] = ()
        error: str | None = None
        try:
            returned = tuple(str(path) for path in await transcriber.transcribe(item))
        except LeaseLostError as exc:
            error = f"its lease was lost: {exc}"
            await _note(store, doc_id, f"'{planned.name}': {error}", job.events)
            return returned, error
        except NothingSelectedError as exc:
            await _note(store, doc_id, str(exc), job.events)
            raise
        except Exception as exc:
            logger.exception("Failed to process item %s: %s", planned.name, exc)
            error = bounded_error(str(exc)) or type(exc).__name__
        force = spec.run.force or planned.resume.state is ItemState.NONE
        rendered = await self.render_or_note(store, doc_id, lease, force=force)
        outputs = tuple(dict.fromkeys((*returned, *rendered)))
        return outputs, error


async def _run_item(
    job: Job,
    transcriber: Transcriber,
    planned: PlannedItem,
    index: int,
    total: int,
    pages: PageRange | None,
    stores: _Stores,
) -> ItemResult | _Blocked:
    """Process one item; :class:`_Blocked` when another run holds its document.

    Raises:
        NothingSelectedError: ``--pages`` selects no unit of the item.
    """
    events = job.events
    item_run = _ItemRun(job, planned, stores)
    leased = await item_run.lease()
    if leased is None:
        return _Blocked(item_run.held or "another run holds its document")
    store, doc_id, lease = leased
    events.item_started(ItemStarted(planned.name, planned.type, index, total))
    try:
        outputs, error = await item_run.transcribe(
            store, doc_id, lease, transcriber, pages
        )
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    success = error is None
    submitted = (
        success
        and job.batch
        and planned.method == "llm"
        and bool(store.blocked_units(doc_id))
    )
    partial = (
        success
        and not submitted
        and pages is not None
        and _short_of_complete(store, doc_id)
    )
    events.item_finished(
        ItemFinished(
            planned.name,
            index,
            total,
            success,
            outputs,
            None,
            error,
            submitted,
            partial,
        )
    )
    failures = () if success else _failed_units(store, doc_id)
    return ItemResult(
        planned.name, success, outputs, error, submitted, partial, failures
    )


def _failed_units(store: Store, doc_id: int) -> tuple[UnitRow, ...]:
    """Return the failed units of the document *doc_id*; none when unreadable."""
    with contextlib.suppress(StoreError, sqlite3.Error):
        return tuple(store.failed_units(doc_id=doc_id))
    return ()


def _short_of_complete(store: Store, doc_id: int) -> bool:
    """Whether the document *doc_id* is still short of complete."""
    with contextlib.suppress(StoreError, sqlite3.Error):
        document = store.document(doc_id)
        return document is not None and document.status != "complete"
    return False


async def _render_only(
    job: Job, planned: PlannedItem, stores: _Stores
) -> tuple[str, ...] | _Blocked:
    """Render a complete item whose files are missing or stale; no model call.

    Return the files that hold the rendering, or :class:`_Blocked` when
    another run holds the item's document.
    """
    item_run = _ItemRun(job, planned, stores)
    leased = await item_run.lease()
    if leased is None:
        return _Blocked(item_run.held or "another run holds its document")
    store, doc_id, lease = leased
    try:
        return await item_run.render_or_note(store, doc_id, lease, force=False)
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)


@dataclass
class _Held:
    """The blocked items of a run, their reasons and those leased elsewhere.

    It starts from the plan's blocked items; *reasons* holds the reason of
    each blocked item, in the order of *blocked*; *complete* holds the ids
    of complete items whose render found their document leased.
    """

    blocked: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    leased: list[str] = field(default_factory=list)
    complete: set[int] = field(default_factory=set)

    @classmethod
    def of(cls, run_plan: RunPlan) -> _Held:
        held = cls()
        for item in run_plan.blocked:
            held.blocked.append(item.name)
            held.reasons.append(item.reason)
            if item.resume.leased_by:
                held.leased.append(item.name)
        return held

    def add(self, planned: PlannedItem, outcome: _Blocked) -> None:
        """Record *planned* as held by another run's lease."""
        self.blocked.append(planned.name)
        self.leased.append(planned.name)
        self.reasons.append(outcome.reason)


async def _render_again(
    job: Job, run_plan: RunPlan, stores: _Stores, held: _Held
) -> list[str]:
    """Render the plan's complete items whose files are missing or stale.

    Return the files rendered; an item another run holds goes to *held*.
    """
    rendered: list[str] = []
    for planned in run_plan.render_only:
        outcome = await _render_only(job, planned, stores)
        if isinstance(outcome, _Blocked):
            held.complete.add(id(planned))
            held.add(planned, outcome)
        else:
            rendered += outcome
    return rendered


async def _note_blocked(job: Job, run_plan: RunPlan, stores: _Stores) -> None:
    """Record a note on the document of each blocked item."""
    for planned in run_plan.blocked:
        if planned.resume.doc_id is None:
            continue
        store = await stores.get(planned.root)
        await _note(
            store,
            planned.resume.doc_id,
            f"Skipping '{planned.name}': {planned.reason}.",
            job.events,
        )


@dataclass(frozen=True)
class _Unbuildable:
    """The outcome of a submission whose requests could not be built."""

    detail: str
    status: str = "refused"
    submission_ids: tuple[str, ...] = ()


async def _submit_batch(
    payloads: Sequence[Any],
    *,
    document: DocumentHandle,
    spec: RunSpec,
    settings: Settings,
    schema_path: Path | None,
    context: str | None = None,
    context_image: Path | None = None,
) -> SubmitOutcome:
    """Submit one document's pages through the batch package.

    A submission whose requests cannot be built is reported as refused.
    """
    try:
        return await batch_submit.submit_batch(
            list(payloads),
            document=document,
            spec=spec,
            settings=settings,
            schema_path=schema_path,
            context=context,
            context_image=context_image,
        )
    except batch_submit.BatchSubmissionError as exc:
        return _Unbuildable(str(exc) or type(exc).__name__)


async def _open_methods(
    run_plan: RunPlan, job: Job, stack: contextlib.AsyncExitStack
) -> dict[str, Transcriber]:
    """Open the transcriber of every method the run uses; none without items.

    Raises:
        ConfigurationError: A method's client cannot be built.
    """
    transcribers: dict[str, Transcriber] = {}
    if not run_plan.to_process:
        return transcribers
    for method in run_plan.methods:
        module, name = _TRANSCRIBERS[method]
        factory: Callable[[Job], Transcriber] = getattr(
            importlib.import_module(module), name
        )
        transcriber = factory(job)
        try:
            await transcriber.open()
        except Exception as exc:
            raise ConfigurationError(
                f"Cannot build the {method} client: {exc}",
                items_total=len(run_plan.to_process),
            ) from exc
        stack.push_async_callback(transcriber.close)
        transcribers[method] = transcriber
    return transcribers


async def run(run_plan: RunPlan, events: RunEvents | None = None) -> RunResult:
    """Process the plan's items one after another on the running event loop.

    The run owns one :class:`~chronotranscriber.llm.caller.CallEnv` and opens
    the transcriber of every method it uses before any item starts or any
    file is written; with nothing to process it opens none. It opens one
    store per output root, stamped with the run. Each item's document is
    leased while its method writes units and while the run core renders its
    formats from the store, also after a failed unit; complete items with
    missing files are rendered without model calls. An item whose document
    another run holds, or of which ``--pages`` selects no unit, is skipped
    with a note; an item another run's lease holds, also a complete one to
    render again, counts as blocked and the run records exit code 1 like a
    failure. A failed item is logged and
    counted; the run continues with the next one. Refused items count as
    failed. Cancellation propagates.

    Raises:
        ValueError: The plan is a dry run.
        ConfigurationError: A method's client cannot be built; nothing has
            been written. :class:`StoreRefusedError` when a database became
            unusable after planning.
    """
    if run_plan.dry_run:
        raise ValueError("a dry-run plan is reported, not run")
    sink: RunEvents = events if events is not None else NullEvents()
    env = CallEnv.create(run_plan.settings, sink)
    started = time.monotonic()
    results: list[ItemResult] = []
    rendered: list[str] = []
    held = _Held.of(run_plan)
    unselected: list[str] = []
    tools = run_plan.settings.tools
    if tools.tesseract_cmd:
        # pytesseract (with pandas) is imported only when there is a command.
        from chronotranscriber.images.tesseract_runtime import use_tesseract_command

        use_tesseract_command(tools.tesseract_cmd)
    configure_ffmpeg_executables(tools.ffmpeg_cmd, tools.ffprobe_cmd)
    job = Job(
        spec=run_plan.spec,
        settings=run_plan.settings,
        events=sink,
        env=env,
        output_dir=run_plan.output_dir,
        input_root=run_plan.input_path,
        schema_path=run_plan.schema_path,
        batch=run_plan.batch,
        submit=_submit_batch if run_plan.batch else None,
    )
    stores = _Stores(run_plan, sink)
    exit_code: int = ExitCode.FAILURE
    try:
        async with contextlib.AsyncExitStack() as stack:
            transcribers = await _open_methods(run_plan, job, stack)
            if run_plan.output_dir is not None:
                run_plan.output_dir.mkdir(parents=True, exist_ok=True)
            await _note_blocked(job, run_plan, stores)
            rendered = await _render_again(job, run_plan, stores, held)
            total = len(run_plan.to_process)
            for index, planned in enumerate(run_plan.to_process, start=1):
                try:
                    result = await _run_item(
                        job,
                        transcribers[planned.method],
                        planned,
                        index,
                        total,
                        run_plan.pages,
                        stores,
                    )
                except NothingSelectedError:
                    unselected.append(planned.name)
                    continue
                if isinstance(result, _Blocked):
                    held.add(planned, result)
                else:
                    results.append(result)
            for planned in run_plan.refused:
                sink.warning(planned.reason)
                results.append(ItemResult(planned.name, False, (), planned.reason))
        failed = any(not result.success for result in results)
        exit_code = ExitCode.FAILURE if failed or held.leased else ExitCode.OK
    except (asyncio.CancelledError, KeyboardInterrupt):
        exit_code = ExitCode.INTERRUPTED
        raise
    finally:
        await stores.close(exit_code)
    return RunResult(
        items=tuple(results),
        skipped=tuple(
            item.name for item in run_plan.skipped if id(item) not in held.complete
        ),
        seconds=time.monotonic() - started,
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=stores.databases,
        blocked=tuple(held.blocked),
        rendered=tuple(rendered),
        unselected=tuple(unselected),
        blocked_reasons=tuple(held.reasons),
        leased=tuple(held.leased),
    )
