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
skipped, and an item whose stored pages used other image settings, or whose
done pages another schema, is ``refused``. Each run records the SHA-256 of
the schema its LLM requests send in its ``runs.spec`` record
(:meth:`RunPlan.run_record`).
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
from collections.abc import AsyncIterable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from chronotranscriber.common.discovery import duplicated, ordered, relative_name
from chronotranscriber.common.discovery import scan as scan_folder
from chronotranscriber.common.options import Source
from chronotranscriber.common.retry import RunStopped
from chronotranscriber.common.scheduler import ItemTicket, RunScheduler
from chronotranscriber.common.selection import (
    PageRange,
    parse_page_range,
    select_items,
)
from chronotranscriber.common.spec_view import resolved_view
from chronotranscriber.common.sqlite import (
    DEFAULT_BUSY_TIMEOUT,
    DEFAULT_WARN_INTERVAL,
    read,
)
from chronotranscriber.common.store import (
    DocumentRow,
    Lease,
    LeaseKeeper,
    LeaseLostError,
    Store,
    StoreError,
    UnitRow,
    WriterOptions,
    bounded_error,
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
    read_schema_text,
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
from chronotranscriber.methods.units import SOLO, DocumentHandle, Lane
from chronotranscriber.output.paths import item_paths
from chronotranscriber.output.render import format_path, render_item, text_formats
from chronotranscriber.output.store import (
    SCHEMA_ENTRY,
    Classification,
    ItemState,
    StoreReader,
    changed_input,
    database_path,
    document_name,
    identity_write,
    lease_reason,
    open_store,
    schema_entry,
    schema_refusal,
    store_reader,
)
from chronotranscriber.settings import Settings, SettingsError
from chronotranscriber.sources import (
    DiscoveryError,
    SourceItem,
    find_items,
    source_type,
)
from chronotranscriber.sources.audio import chunk_settings, describe_chunks
from chronotranscriber.sources.audio.ffmpeg import configure_ffmpeg_executables
from chronotranscriber.sources.auto import decide
from chronotranscriber.sources.auto import find as find_auto
from chronotranscriber.sources.identity import (
    Snapshot,
    input_identity,
    input_snapshot,
    reordered,
)
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
    "RunStop",
    "SelectionError",
    "StoreRefusedError",
    "choose_items",
    "discover",
    "find_input",
    "item_location",
    "open_items_effect",
    "open_window",
    "plan",
    "run",
    "selection_names",
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

_HASH_WORKERS = 8
"""Inputs hashed at once while planning; the threads overlap the file reads."""

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
    run. *input_sha256* is the input identity computed at planning
    (:func:`~chronotranscriber.sources.identity.input_identity`) and
    *snapshot* the input's file sizes and times at that moment.
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
    input_sha256: str | None = None
    snapshot: Snapshot = ()

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
    found and selected, in the natural order of their paths, the order
    *to_process* keeps; *skipped* holds the complete items in that order; a
    complete item whose rendered files are missing or stale is rendered
    again without model calls. *unstored* items
    have outputs without store rows, *blocked* items are leased by another
    run or wait for a batch, *refused* items were stored with other image
    settings; none of them is processed. *warnings* are notes for the user.
    *sources* maps spec fields to where their values came from.
    *missing_keys* lists the key variables a dry run (or a plan that does
    not check them) found unset. *schema_text* is the text of the schema
    file the LLM requests send, read once at planning, and *schema_sha256*
    its SHA-256; both are None for a run in the text response format.
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
    schema_text: str | None = None
    schema_sha256: str | None = None
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

    @property
    def schema_identity(self) -> str | None:
        """The schema hash the run records, None when no item sends the schema."""
        return self.schema_sha256 if "llm" in self.methods else None

    def run_record(self) -> dict[str, dict[str, Any]]:
        """Return the record the store keeps of the run in ``runs.spec``.

        It is :meth:`effective_spec` with the ``schema_sha256`` entry when
        LLM items send a schema: the ``artifacts.sha256`` of its text.
        """
        record = self.effective_spec()
        if self.schema_identity is not None:
            record[SCHEMA_ENTRY] = schema_entry(self.schema_identity)
        return record

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


def find_input(
    selection: InputSpec, input_path: Path, output_dir: Path | None = None
) -> tuple[list[SourceItem], list[str]]:
    """Return the items at *input_path* and notes about what was left out.

    Auto mode takes the input file, or the PDFs, image folders, EPUB and
    MOBI files of the folder; another type the items of that type. The
    search leaves out the run's output folder *output_dir*, the working
    folders and the sidecars (:func:`~chronotranscriber.sources.scan_input`).
    The items come in the natural order of their paths below the input.

    Raises:
        InputError: The input cannot serve the type, or ``--files`` names
            entries that are not items.
    """
    try:
        if selection.type == "auto":
            found = find_auto(
                input_path,
                files=selection.files,
                recursive=selection.recursive,
                output_dir=output_dir,
            )
        else:
            found = find_items(
                input_path,
                selection.type,
                files=selection.files,
                recursive=selection.recursive,
                output_dir=output_dir,
            )
    except DiscoveryError as exc:
        raise InputError(str(exc)) from exc
    warnings: list[str] = []
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
    for folder, parent in found.nested:
        warnings.append(
            f"Skipped the image folder '{relative_name(folder, input_path)}': it "
            f"lies inside the image folder '{relative_name(parent, input_path)}'. "
            "Pass it as --input to process it."
        )
    if found.working:
        more = len(found.working) - 1
        warnings.append(
            f"Skipped the folder '{found.working[0]}'"
            + (f" and {more} more" if more else "")
            + ": folders of this name hold working files kept by "
            "--keep-working-files. Pass one as --input to process its files."
        )
    return list(found.items), warnings


def _discover(spec: RunSpec, input_path: Path) -> tuple[list[SourceItem], list[str]]:
    """Return the items at *input_path* and notes about skipped files."""
    items, warnings = find_input(spec.input, input_path, _output_dir(spec))
    if not items:
        if spec.input.type == "auto":
            raise NoItemsError(_auto_empty_message(input_path))
        raise NoItemsError(f"No items found to process in: {input_path}")
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
    if any(source_type(path) == "audio" for path in scan_folder(input_path)):
        message += _AUDIO_HINT
    return message


def discover(spec: RunSpec) -> list[SourceItem]:
    """Return the items the run of *spec* finds at its input.

    The items come before ``--all`` or ``--select`` choose among them
    (``--files`` does apply), in the natural order of their paths below the
    input, the order a run processes them in and ``--select`` numbers them.

    Raises:
        PlanError: The input is missing or holds nothing to process.
    """
    return _discover(spec, _input_path(spec))[0]


def selection_names(items: Sequence[SourceItem], input_path: Path) -> list[str]:
    """Return the names ``--select`` searches: each path below *input_path*."""
    return [relative_name(item.path, input_path) for item in items]


def choose_items(
    items: list[SourceItem], selection: InputSpec, input_path: Path
) -> tuple[list[SourceItem], list[str]]:
    """Apply ``--all``, ``--select`` or ``--files``; one item needs none.

    The items are taken in the natural order of their paths below
    *input_path*. ``--select`` numbers them from 1 in that order and
    otherwise searches their :func:`selection_names`; its out-of-range parts
    are warnings.

    Raises:
        SelectionError: ``--select`` matched no item.
        AmbiguousSelectionError: Several items and no selection.
    """
    items = ordered(items, input_path, lambda item: item.path)
    if selection.all or selection.files:
        return items, []
    if selection.select:
        result = select_items(selection_names(items, input_path), selection.select)
        warnings = []
        if result.unmatched:
            warnings.append(
                "Ignored out-of-range or invalid --select part(s): "
                f"{', '.join(result.unmatched)} (valid: 1-{len(items)})"
            )
        picked = result.pick(items)
        if not picked:
            raise SelectionError(
                f"No item of {input_path} matches --select {selection.select!r}"
            )
        return picked, warnings
    if len(items) == 1:
        return items, []
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

    *image* is the run's image-settings record and *schema_sha256* the hash
    of the schema its LLM requests send.
    """

    spec: RunSpec
    output_dir: Path | None
    input_root: Path
    image: Mapping[str, Any] | None
    schema_sha256: str | None = None

    def locate(self, item: SourceItem) -> tuple[Path, str, Path]:
        """Return the output root, document name and output path of *item*."""
        return item_location(
            item,
            output_dir=self.output_dir,
            layout=self.spec.output.layout,
            input_root=self.input_root,
            output_format=text_formats(self.spec.output.formats)[0],
        )

    def plan_items(self, routed: Sequence[tuple[SourceItem, str]]) -> list[PlannedItem]:
        """Return each ``(item, method)`` of *routed* planned; write nothing.

        The inputs are hashed in a thread pool first; then the items of each
        output root are classified through one reader of its store, the
        roots in the order of their first item.

        Raises:
            StoreRefusedError: A root's database is refused.
        """
        located = [self.locate(item) for item, _ in routed]
        with ThreadPoolExecutor(_HASH_WORKERS) as pool:
            identities = list(pool.map(_identify, [item.path for item, _ in routed]))
        by_root: dict[Path, list[int]] = {}
        for index, (root, _, _) in enumerate(located):
            by_root.setdefault(root, []).append(index)
        planned: dict[int, PlannedItem] = {}
        for root, indices in by_root.items():
            try:
                with store_reader(root) as reader:
                    for index in indices:
                        item, method = routed[index]
                        planned[index] = self.planned(
                            reader, item, method, located[index], identities[index]
                        )
            except StoreError as exc:
                raise StoreRefusedError(str(exc)) from exc
        return [planned[index] for index in range(len(routed))]

    def planned(
        self,
        reader: StoreReader,
        item: SourceItem,
        method: str,
        location: tuple[Path, str, Path],
        identity: _Identity,
    ) -> PlannedItem:
        """Return *item* planned with its resume state; write nothing.

        *location* is the item's :meth:`locate` and *identity* its input
        identity with its snapshot; an input whose identity cannot be
        computed is refused, also under --force, so that no document records
        it as unknown.
        """
        root, name, output = location
        formats = text_formats(self.spec.output.formats)
        outputs = [format_path(output, fmt) for fmt in formats]
        keyed = method in _KEYED_METHODS
        snapshot, sha256, error = identity
        unreadable: str | None = None
        if error is not None:
            unreadable = (
                f"Cannot compute the input identity of '{name}' ({error}). Skipping it."
            )

        state = reader.classify(
            name,
            outputs=outputs,
            image_settings=self.image if method == "llm" else None,
            model=self.spec.model.model if keyed else None,
            retry_errors=self.spec.run.retry_errors,
            input_sha256=sha256,
            schema_sha256=self.schema_sha256 if method == "llm" else None,
        )
        if item.type == "images" and unreadable is None:
            moved = _misplaced_pages(reader, name, item.path, state)
            if moved is not None:
                state = Classification(
                    ItemState.REFUSED, name, state.doc_id, reason=moved
                )
        if self.spec.run.force and state.state is not ItemState.BLOCKED:
            state = Classification(
                ItemState.NONE, name, state.doc_id, reason="overwrite mode"
            )
        if unreadable is not None and state.state is not ItemState.BLOCKED:
            state = Classification(
                ItemState.REFUSED, name, state.doc_id, reason=unreadable
            )
        return PlannedItem(
            item,
            method,
            state,
            root,
            name,
            output,
            input_sha256=sha256,
            snapshot=snapshot,
        )


type _Identity = tuple[Snapshot, str | None, OSError | None]
"""An input's snapshot, its identity and the error that left it unknown."""


def _identify(path: Path) -> _Identity:
    """Return the snapshot and identity of the input at *path*, snapshot first."""
    snapshot: Snapshot = ()
    try:
        snapshot = input_snapshot(path)
        return snapshot, input_identity(path), None
    except OSError as exc:
        return snapshot, None, exc


def _misplaced_pages(
    reader: StoreReader, name: str, folder: Path, state: Classification
) -> str | None:
    """Return why a stored folder document's pages no longer fit its images.

    A partial or complete document is checked page by page
    (:func:`~chronotranscriber.sources.identity.reordered`), whether it
    records an identity or not.
    """
    if state.doc_id is None or state.state not in (
        ItemState.PARTIAL,
        ItemState.COMPLETE,
    ):
        return None
    labels = reader.page_labels(state.doc_id)
    try:
        return reordered(name, folder, labels)
    except OSError as exc:
        return f"Cannot list the images of '{name}' ({exc}). Skipping it."


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


def _schema(
    spec: RunSpec, settings: Settings
) -> tuple[Path | None, str | None, str | None]:
    """Return the schema file of an LLM run, its text and the text's SHA-256.

    The schema is found in the settings' folders, and the system prompt
    file is checked. A run in the text response format sends no schema:
    its text and hash are None. A run without the LLM method returns
    three Nones.

    Raises:
        ConfigurationError: The schema or prompt is not found, two user
            folders hold the schema, or its file cannot be read.
    """
    if not _uses_llm(spec):
        return None, None, None
    try:
        path = resolve_schema(spec.prompt.schema, settings.schemas_dirs)[1]
        system_prompt_path(spec.prompt.prompt)
        if spec.model.response_format == "text":
            return path, None, None
        text, sha256 = read_schema_text(path)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    return path, text, sha256


def _check_shared(
    items: Sequence[SourceItem],
    locate: Callable[[SourceItem], tuple[Path, str, Path]],
) -> None:
    """Raise when two items would write the same output and store document.

    *locate* gives an item's output root, document name and output path.
    Document names compare case-insensitively, as the rendered files do on
    Windows and macOS. The error names both inputs of every such pair.

    Raises:
        ConfigurationError: Two items share an output root and document name.
    """
    located = [(item, *locate(item)) for item in items]
    pairs = duplicated(located, key=lambda entry: (entry[1], entry[2].casefold()))
    if pairs:
        clashes = "; ".join(
            f"{earlier[0].path} and {later[0].path} would share the output "
            f"{later[3]} and its store document"
            for later, earlier in pairs
        )
        raise ConfigurationError(
            f"{clashes}. Set --output DIR to keep them apart, or take one of "
            "them with --select."
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
    output_dir = _output_dir(spec)
    input_path = _input_path(spec)
    logger.info("Input=%s, Output=%s", input_path, output_dir or "beside input")
    found, warnings = _discover(spec, input_path)
    discovered, notes = choose_items(found, spec.input, input_path)
    warnings += notes
    schema_path, schema_text, schema_sha256 = _schema(spec, settings)

    image = _image(spec, settings, discovered)
    classifier = _Classifier(spec, output_dir, input_path, image, schema_sha256)
    groups: dict[ItemState, list[PlannedItem]] = {state: [] for state in ItemState}
    routed = [(item, _route(item, spec)) for item in discovered]
    _check_shared(discovered, classifier.locate)
    planned_items = classifier.plan_items(routed)
    for planned in planned_items:
        groups[planned.resume.state].append(planned)
    to_process = _in_order(groups, discovered)
    pages = parse_page_range(spec.input.pages) if spec.input.pages else None
    warnings += _nothing_selected(pages, groups[ItemState.COMPLETE])
    logger.info(
        "%d item(s) to process, %d already complete",
        len(to_process),
        len(groups[ItemState.COMPLETE]),
    )

    if _uses_llm(spec):
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
        schema_text=schema_text,
        schema_sha256=schema_sha256,
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
class RunStop:
    """Why a run stopped before its last item, and what it left to do.

    *reason* names the failed call whose ``STOP`` decision ended the run;
    *item* is the item in work, *others* the further items open at that
    moment that the stop closed, *units_left* their units still pending and
    *outputs* the files they were rendered to; *items_left* counts the items
    not started.
    """

    reason: str
    item: str
    units_left: int = 0
    items_left: int = 0
    outputs: tuple[str, ...] = ()
    others: tuple[str, ...] = ()

    @property
    def message(self) -> str:
        """The error line of the stopped run."""
        if self.others:
            names = ", ".join(f"'{name}'" for name in (self.item, *self.others))
            subject = f"the open items {names}"
        else:
            subject = f"'{self.item}'"
        return (
            f"The run stopped: the provider serves no further calls ({self.reason}). "
            f"{self.units_left} pending unit(s) of {subject} and "
            f"{self.items_left} further item(s) were left unprocessed; a rerun "
            "resumes them."
        )

    @classmethod
    def merged(cls, reason: str, stops: Sequence[RunStop], items_left: int) -> RunStop:
        """Return one stop for the items *stops* closed, in item order."""
        first, *rest = stops
        return cls(
            reason,
            first.item,
            units_left=sum(stop.units_left for stop in stops),
            items_left=items_left,
            outputs=tuple(path for stop in stops for path in stop.outputs),
            others=tuple(stop.item for stop in rest),
        )


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
    again for complete items. *stop* says why the run stopped before its
    last item; the item in work is in none of the other fields.
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
    stop: RunStop | None = None

    @property
    def stopped(self) -> str | None:
        """The reason the run stopped, or None when it was not stopped."""
        return self.stop.reason if self.stop is not None else None

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
        """3 when the run stopped, 1 when an item failed or was leased, else 0."""
        if self.stop is not None:
            return ExitCode.STOPPED
        return exit_code_for(failed=self.failed + len(self.leased))

    @property
    def outputs(self) -> list[str]:
        """The paths written: files rendered again, item files, then databases."""
        items = [path for item in self.items for path in item.outputs]
        stopped = list(self.stop.outputs) if self.stop is not None else []
        return [*self.rendered, *items, *stopped, *self.databases]


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
    the heartbeat that renews the run's leases. The items open at once share
    the store of their root; *finishing* lets one item at a time render its
    files.
    """

    def __init__(self, run_plan: RunPlan, events: RunEvents) -> None:
        self._plan = run_plan
        self._events = events
        self._stores: dict[Path, Store] = {}
        self._keepers: list[LeaseKeeper] = []
        self._opening = asyncio.Lock()
        self.finishing = asyncio.Lock()

    @property
    def databases(self) -> tuple[str, ...]:
        """The database files of the opened stores, in opening order."""
        return tuple(str(store.path.resolve()) for store in self._stores.values())

    async def get(self, root: Path) -> Store:
        """Return the open store of output root *root*, opening it once.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        key = root.resolve()
        # A store enters the map before its run stamp is written: wait for that.
        async with self._opening:
            store = self._stores.get(key)
            if store is not None:
                return store
            return await self._open(key)

    async def _open(self, key: Path) -> Store:
        key.mkdir(parents=True, exist_ok=True)
        store = open_store(key, writer=_writer_options(self._events))
        try:
            await store.open()
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        self._stores[key] = store
        await store.begin_run("transcribe", spec=self._plan.run_record())
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

    @property
    def schema_sha256(self) -> str | None:
        """The schema hash of the run's LLM requests, as the run records it."""
        return self._plan.schema_identity

    async def _artifacts(self, store: Store) -> None:
        """Store the prompt and schema texts of an LLM run.

        The schema text is the one read at planning, whose hash the run
        record holds.
        """
        spec = self._plan.spec
        if "llm" not in self._plan.methods:
            return
        plain = spec.model.response_format == "text"
        prompt = spec.prompt.prompt
        default = PLAIN_PROMPT_PATH if plain else SCHEMA_PROMPT_PATH
        texts = {
            "prompt": _read_text(_absolute(prompt) if prompt else default),
            "schema": self._plan.schema_text,
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


@dataclass(frozen=True)
class _Blocked:
    """An item whose document another run leased after planning."""

    reason: str


def _pending_units(store: Store, doc_id: int) -> int:
    """Return the pending units of the document *doc_id*; 0 when unreadable."""
    with contextlib.suppress(StoreError, sqlite3.Error):
        document = store.document(doc_id)
        return document.units_pending if document is not None else 0
    return 0


def _changed_during_run(planned: PlannedItem) -> str | None:
    """Return why *planned*'s input no longer has its planned identity, or None.

    The input is hashed again only when its snapshot changed.
    """
    try:
        if input_snapshot(planned.path) == planned.snapshot:
            return None
        current = input_identity(planned.path)
    except OSError as exc:
        return (
            f"Cannot compute the input identity of '{planned.document}' ({exc}). "
            "Skipping it."
        )
    if current == planned.input_sha256:
        return None
    return (
        f"Input changed for '{planned.document}' during the run (planned "
        f"{str(planned.input_sha256)[:12]}..., now {current[:12]}...). Skipping it."
    )


@dataclass
class _ItemRun:
    """One item of a run: its store document, lease, method and render.

    *held* is the reason the item could not be leased, once :meth:`lease`
    found another run holding it; *refused* the reason its input refused it.
    *replacement* is the identity a --force run records with the units'
    reset, when the stored units belong to another input. *stopped* is set
    when the provider stopped the run during the item.
    """

    job: Job
    planned: PlannedItem
    stores: _Stores
    held: str | None = None
    refused: str | None = None
    replacement: str | None = None
    stopped: RunStopped | None = None

    def _refusal(self, stored: DocumentRow | None) -> str | None:
        """Return why the stored document refuses the item's input, or None.

        A stored identity that differs from the planned one refuses it when
        the document has units, unless --force.
        """
        planned = self.planned
        if (
            planned.input_sha256 is None
            or stored is None
            or stored.input_sha256 is None
            or stored.input_sha256 == planned.input_sha256
            or stored.units_total == 0
            or self.job.force
        ):
            return None
        return changed_input(
            planned.document, stored.input_sha256, planned.input_sha256
        )

    async def _settle_identity(
        self, store: Store, doc_id: int, lease: Lease
    ) -> str | None:
        """Check and record the input identity under *lease*; return a refusal.

        The leased document is read again: an input that changed since
        planning or a stored identity that differs refuses the item (unless
        --force, which records the planned one with the units' reset, see
        :attr:`replacement`); a document without one takes the planned one,
        with a note when it already has units.
        """
        planned = self.planned
        refused = _changed_during_run(planned)
        stored = store.document(doc_id)
        if refused is None:
            refused = self._refusal(stored)
        if refused is not None or planned.input_sha256 is None or stored is None:
            return refused
        if stored.input_sha256 == planned.input_sha256:
            return None
        if stored.units_total > 0 and self.job.force:
            self.replacement = planned.input_sha256
            return None
        await store.write(lease, identity_write(doc_id, planned.input_sha256))
        if stored.units_total > 0:
            message = (
                f"Recorded the input identity of '{planned.document}' for the "
                "first time."
            )
            logger.info(message)
            with contextlib.suppress(sqlite3.Error):
                await store.add_note(message, doc_id=doc_id, level="info")
        return None

    def _schema_refusal(self, store: Store, doc_id: int) -> str | None:
        """Return why the leased document refuses the run's schema, or None.

        Another run may have written pages since planning, so the check of
        :func:`~chronotranscriber.output.store.classify` is repeated: an LLM
        item with units to write (pending, or failed under --retry-errors)
        is refused when a done page records another schema hash. --force
        replaces every page and is not checked.
        """
        current = self.stores.schema_sha256
        if current is None or self.planned.method != "llm" or self.job.force:
            return None
        state = store.resume_state(doc_id)
        if state is None or not (
            state.pending or (self.job.retry_errors and state.failed)
        ):
            return None
        name = self.planned.document
        return read(
            store.path,
            lambda connection: schema_refusal(connection, doc_id, name, current),
        )

    async def lease(self, *, writes: bool = True) -> tuple[Store, int, Lease] | None:
        """Ensure the item's document and lease it; None when it is not leased.

        A stored identity that differs, or an input that changed since
        planning, refuses the item before the document row is written; the
        row never takes an identity there. Under the
        lease the input and the stored identity are checked again and the
        identity recorded (:meth:`_settle_identity`); an item that *writes*
        units is checked against the run's schema (:meth:`_schema_refusal`).
        :attr:`refused` says why a refused item was not leased.
        """
        planned = self.planned
        store = await self.stores.get(planned.root)
        stored = store.document(planned.document)
        self.refused = self._refusal(stored) or _changed_during_run(planned)
        if self.refused is not None:
            await _note(
                store,
                stored.doc_id if stored is not None else None,
                self.refused,
                self.job.events,
            )
            return None
        doc_id = await store.ensure_document(
            planned.document,
            input_path=planned.path,
            input_sha256=None,
            kind=planned.type,
        )
        lease = await store.acquire(doc_id)
        if lease is not None:
            try:
                self.refused = await self._settle_identity(store, doc_id, lease)
                if self.refused is None and writes:
                    self.refused = self._schema_refusal(store, doc_id)
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    await store.release(lease)
                raise
            if self.refused is not None:
                with contextlib.suppress(sqlite3.Error):
                    await store.release(lease)
                await _note(store, doc_id, self.refused, self.job.events)
                return None
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
        """Render like :meth:`render`; note a lost lease or a file error instead.

        One item at a time renders: the others wait for :attr:`_Stores.finishing`.
        """
        name = self.planned.name
        try:
            async with self.stores.finishing:
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
        lane: Lane = SOLO,
    ) -> tuple[tuple[str, ...], str | None]:
        """Run the method, then render; return the outputs and the error.

        The method's calls draw on *lane*. A stopped run (:attr:`stopped`)
        renders the units stored so far and returns no error.
        """
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
            identity=self.replacement,
        )
        item = Item(
            path=planned.path,
            source=planned.type,
            method=planned.method,
            pages=pages,
            context=planned.context,
            context_image=planned.context_image,
            document=handle,
            lane=lane,
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
        except RunStopped as exc:
            self.stopped = exc
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
    lane: Lane = SOLO,
) -> ItemResult | _Blocked | RunStop:
    """Process one item; :class:`_Blocked` when another run holds its document.

    The item's calls draw on *lane*. A :class:`RunStop` is returned when the
    provider stopped the run during the item; its lease is released and no
    item event reports it finished.

    Raises:
        NothingSelectedError: ``--pages`` selects no unit of the item.
    """
    events = job.events
    item_run = _ItemRun(job, planned, stores)
    leased = await item_run.lease()
    if leased is None and item_run.refused is not None:
        return ItemResult(planned.name, False, (), item_run.refused)
    if leased is None:
        return _Blocked(item_run.held or "another run holds its document")
    store, doc_id, lease = leased
    reason = job.env.stop.reason
    if reason is not None:
        # Another open item stopped the run while this one waited for its
        # lease: it sends nothing, so --force must not reset what it stored.
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
        return RunStop(reason, planned.name, units_left=_pending_units(store, doc_id))
    document, root = planned.document, str(store.path.parent)
    events.item_started(
        ItemStarted(
            planned.name, planned.type, index, total, document=document, root=root
        )
    )
    try:
        outputs, error = await item_run.transcribe(
            store, doc_id, lease, transcriber, pages, lane
        )
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    if item_run.stopped is not None:
        return RunStop(
            job.env.stop.reason or str(item_run.stopped),
            planned.name,
            units_left=_pending_units(store, doc_id),
            outputs=outputs,
        )
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
            document=document,
            root=root,
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
) -> tuple[str, ...] | _Blocked | ItemResult:
    """Render a complete item whose files are missing or stale; no model call.

    Return the files that hold the rendering, :class:`_Blocked` when
    another run holds the item's document, or a failed :class:`ItemResult`
    when its input changed since planning.
    """
    item_run = _ItemRun(job, planned, stores)
    leased = await item_run.lease(writes=False)
    if leased is None and item_run.refused is not None:
        return ItemResult(planned.name, False, (), item_run.refused)
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
    job: Job,
    run_plan: RunPlan,
    stores: _Stores,
    held: _Held,
    failed: list[ItemResult],
) -> list[str]:
    """Render the plan's complete items whose files are missing or stale.

    Return the files rendered; an item another run holds goes to *held*,
    one whose input changed since planning to *failed*.
    """
    rendered: list[str] = []
    for planned in run_plan.render_only:
        outcome = await _render_only(job, planned, stores)
        if isinstance(outcome, _Blocked):
            held.complete.add(id(planned))
            held.add(planned, outcome)
        elif isinstance(outcome, ItemResult):
            held.complete.add(id(planned))
            failed.append(outcome)
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
    payloads: AsyncIterable[Any],
    *,
    document: DocumentHandle,
    spec: RunSpec,
    settings: Settings,
    schema_path: Path | None,
    context: str | None = None,
    context_image: Path | None = None,
    schema_text: str | None = None,
) -> SubmitOutcome:
    """Submit one document's pages through the batch package as they render.

    A submission whose requests cannot be built is reported as refused.
    """
    try:
        return await batch_submit.submit_batch(
            payloads,
            document=document,
            spec=spec,
            settings=settings,
            schema_path=schema_path,
            context=context,
            context_image=context_image,
            schema_text=schema_text,
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


async def _process_items(  # noqa: PLR0913 - the run's shared state
    job: Job,
    run_plan: RunPlan,
    transcribers: Mapping[str, Transcriber],
    stores: _Stores,
    held: _Held,
    results: list[ItemResult],
    unselected: list[str],
) -> RunStop | None:
    """Process the plan's items; return the stop that ended them early.

    The run scheduler opens the items in order, up to :func:`open_window` at
    once, and their units share :func:`_limit` request slots. Once all have
    ended, each item's outcome goes, in item order, to *results*, *held* or
    *unselected*. A stopped run opens no further item; the stop counts the
    units the open items left and the items not opened.
    """
    items = list(enumerate(run_plan.to_process, start=1))
    total = len(items)
    scheduler = RunScheduler(
        _limit(run_plan), window=open_window(run_plan), stop=job.env.stop
    )

    async def process(entry: tuple[int, PlannedItem], ticket: ItemTicket) -> _Outcome:
        index, planned = entry
        try:
            return await _run_item(
                job,
                transcribers[planned.method],
                planned,
                index,
                total,
                run_plan.pages,
                stores,
                Lane(scheduler, ticket),
            )
        except NothingSelectedError:
            return _Unselected()

    outcomes = await scheduler.run(items, process)
    stops: list[RunStop] = []
    for (_index, planned), outcome in zip(items, outcomes, strict=False):
        if isinstance(outcome, RunStop):
            stops.append(outcome)
        elif isinstance(outcome, _Blocked):
            held.add(planned, outcome)
        elif isinstance(outcome, _Unselected):
            unselected.append(planned.name)
        else:
            results.append(outcome)
    left = total - len(outcomes)
    reason = job.env.stop.reason
    if stops:
        return RunStop.merged(reason or stops[0].reason, stops, left)
    # An item that fails otherwise while the run stops, as by a lost lease,
    # returns that failure; the stop then counts only the items not opened.
    if reason is not None and left and outcomes:
        return RunStop(reason, items[len(outcomes) - 1][1].name, items_left=left)
    return None


@dataclass(frozen=True)
class _Unselected:
    """The outcome of an item of which ``--pages`` selects no unit."""


type _Outcome = ItemResult | _Blocked | RunStop | _Unselected
"""What one item of a run comes to."""


def _limit(run_plan: RunPlan) -> int:
    """Return the requests the open items of a run may have in flight at once.

    Recordings take the ``concurrency.audio`` setting, other items
    ``--concurrency``.
    """
    if run_plan.spec.input.type == "audio":
        return run_plan.settings.concurrency.audio
    return run_plan.spec.run.concurrency


def open_items_effect(run_plan: RunPlan) -> str | None:
    """Return why ``--open-items`` has no effect on *run_plan*, or None.

    The option shares the requests of a provider's synchronous calls among
    several items; a run without such calls, a ``--batch`` run and a run of
    one item open one item at a time.
    """
    if not any(method in _KEYED_METHODS for method in run_plan.methods):
        return "the run calls no provider"
    if run_plan.batch:
        return "--batch submits the pages to the batch API"
    if len(run_plan.to_process) == 1:
        return "the run has one item to process"
    return None


def open_window(run_plan: RunPlan) -> int:
    """Return how many items of the run are open at once (``--open-items``)."""
    if open_items_effect(run_plan) is not None:
        return 1
    return run_plan.spec.run.open_items


async def run(run_plan: RunPlan, events: RunEvents | None = None) -> RunResult:
    """Process the plan's items on the running event loop.

    The items open in order, ``--open-items`` at once where the option has
    an effect (:func:`open_items_effect`), else one after another; the
    requests of all open items share ``--concurrency`` slots
    (``concurrency.audio`` for recordings). Their events interleave, and
    the results, the stop and the exit code are taken in item order.
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
    failed. Cancellation propagates: the open items are cancelled and their
    leases released before the stores close. When the provider stops the
    run (:class:`~chronotranscriber.common.retry.RunStopped`), each open
    item is rendered from the units stored so far and released, no further
    item starts, and :attr:`RunResult.stop` says what was left; the run
    records exit code 3.

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
        schema_text=run_plan.schema_text,
        batch=run_plan.batch,
        submit=_submit_batch if run_plan.batch else None,
    )
    stores = _Stores(run_plan, sink)
    exit_code: int = ExitCode.FAILURE
    stop: RunStop | None = None
    try:
        async with contextlib.AsyncExitStack() as stack:
            transcribers = await _open_methods(run_plan, job, stack)
            if run_plan.output_dir is not None:
                run_plan.output_dir.mkdir(parents=True, exist_ok=True)
            await _note_blocked(job, run_plan, stores)
            rendered = await _render_again(job, run_plan, stores, held, results)
            stop = await _process_items(
                job, run_plan, transcribers, stores, held, results, unselected
            )
            for planned in run_plan.refused:
                sink.warning(planned.reason)
                results.append(ItemResult(planned.name, False, (), planned.reason))
        failed = any(not result.success for result in results)
        exit_code = ExitCode.FAILURE if failed or held.leased else ExitCode.OK
        if stop is not None:
            exit_code = ExitCode.STOPPED
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
        stop=stop,
    )
