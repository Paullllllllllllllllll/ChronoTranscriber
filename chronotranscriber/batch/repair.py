"""Repair: transcribe the failed pages of stored documents again.

:func:`plan_repairs` takes ``--input`` as a file, a database or a folder.
A file is found in the nearest ``chronotranscriber.sqlite`` at or above it
whose outputs registry lists it, else in the nearest one that records it as
a document's input (:func:`plan_repair` plans that one document). A
``chronotranscriber.sqlite`` names every document it holds; a folder that
no database records as an input names every document of the databases at
or below it. For each document the plan selects the page units of the
chosen failure causes (a transcription error is a failed unit;
``not_possible`` and ``no_text`` are their outcomes; ``--skipped`` selects
the pages the decision pre-pass skipped), narrowed by
``--indices`` (page indices; one document only), leaves out the units an
open or unresolved batch submission holds, picks the image settings the
pages were rendered with, resolves the schema, prompt and context and checks
provider access; a dry run plans without the key variables. It writes
nothing and raises :class:`RepairPlanError` (exit 2) for usage and
configuration errors, a database the store refuses included.

:func:`run_repair` leases the document, renders the selected pages again
from the recorded source and transcribes them: synchronously through
:func:`~chronotranscriber.methods.units.run_unit_stream`, after resetting
the units (their responses stay), or through the provider's batch API with
:func:`~chronotranscriber.batch.submit.submit_units`, a wait on the
submissions and :func:`~chronotranscriber.batch.finalize.finalize_submission`.
Every unit is written in a transaction that checks the lease; the
document's files are then rendered again from the store.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chronotranscriber import ext
from chronotranscriber.batch.bodies import build_plan
from chronotranscriber.batch.clients import BatchClients
from chronotranscriber.batch.finalize import (
    FINALIZED_OUTCOME,
    delete_provider_files,
    finalize_submission,
)
from chronotranscriber.batch.submit import (
    BatchSubmissionError,
    UnitRequest,
    submit_units,
)
from chronotranscriber.batch.wait import wait_for_submissions
from chronotranscriber.common.batch import BatchRequest, supports_batch
from chronotranscriber.common.contract import ExitCode, exit_code_for
from chronotranscriber.common.sqlite import read
from chronotranscriber.common.store import (
    LeaseLostError,
    Store,
    StoreError,
    Submission,
    UnitResult,
    UnitRow,
    WriterOptions,
)
from chronotranscriber.common.usage import Usage, UsageTotals
from chronotranscriber.events import NullEvents, RunEvents
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.capabilities import ensure_image_support
from chronotranscriber.llm.options import model_capabilities
from chronotranscriber.llm.prompts import (
    ContextError,
    fallback_context,
    item_context,
    item_context_image,
    resolve_schema,
    system_prompt_path,
)
from chronotranscriber.llm.transcription import PageTranscriber
from chronotranscriber.methods.pages import transcribe_payload
from chronotranscriber.methods.units import (
    DocumentHandle,
    UnitsFailedError,
    run_unit_stream,
)
from chronotranscriber.output.render import TEXT_FORMATS, render_item
from chronotranscriber.output.store import (
    DATABASE_NAME,
    KIND_PAGE,
    changed_input,
    database_path,
    identity_write,
    open_store,
    read_store,
)
from chronotranscriber.settings import Settings, SettingsError
from chronotranscriber.sources.audio.ffmpeg import configure_ffmpeg_executables
from chronotranscriber.sources.identity import input_identity, reordered
from chronotranscriber.sources.images import (
    PageImage,
    detect_model_type,
    folder_image_name,
    list_folder_images,
    load_image,
    resolve_image_settings,
)
from chronotranscriber.sources.pdf import render_pdf_page
from chronotranscriber.spec import (
    DEFAULT_CONCURRENCY,
    ImageSpec,
    ModelSpec,
    RepairSpec,
    RunSpec,
    effective_spec,
)

__all__ = [
    "CAUSE_OUTCOMES",
    "ImageSettings",
    "RepairDocument",
    "RepairPlan",
    "RepairPlanError",
    "RepairResult",
    "RepairSet",
    "batch_refusal",
    "locate_document",
    "page_image",
    "plan_repair",
    "plan_repairs",
    "repair_image_settings",
    "repair_transcription",
    "run_repair",
    "run_repairs",
]

logger = logging.getLogger(__name__)

ROLE = "repair"
LABEL = "batch-repair"

CAUSE_OUTCOMES: Mapping[str, str] = {
    "api_error": "failed",
    "not_possible": "not_possible",
    "no_text": "no_text",
    "skipped": "skipped_no_text",
}
"""The unit outcome each failure cause of ``--errors-only`` and its kin selects."""


class RepairPlanError(Exception):
    """The repair cannot be planned: a usage or configuration error (exit 2)."""

    exit_code: ExitCode = ExitCode.USAGE


@dataclass(frozen=True)
class RepairDocument:
    """The stored document a repair works on.

    *root* is the folder of its database; *output* the path its files are
    rendered to, with the suffix of each of *formats*.
    """

    root: Path
    doc_id: int
    name: str
    input_path: Path | None
    image_settings: Mapping[str, Any] | None
    output: Path
    formats: tuple[str, ...]
    lease_run_id: str | None = None
    lease_until: float | None = None
    input_sha256: str | None = None

    @property
    def database(self) -> Path:
        """Return the database file of the document."""
        return database_path(self.root)


@dataclass(frozen=True)
class ImageSettings:
    """The image settings the repaired pages are rendered with.

    *recorded* tells whether they are the document's recorded settings; only
    then is their detail sent with each page.
    """

    config: Mapping[str, Any]
    model_type: str
    recorded: bool

    @property
    def detail(self) -> str | None:
        """Return the resolved image detail."""
        value = self.config.get("resolved_detail")
        return str(value) if value else None

    def request_options(self) -> dict[str, str]:
        """Return the per-page options of a synchronous request."""
        if not self.recorded:
            return {}
        options: dict[str, str] = {}
        if self.detail:
            options["image_detail"] = self.detail
        resolution = self.config.get("media_resolution")
        if resolution:
            options["media_resolution"] = str(resolution)
        return options


@dataclass
class RepairResult:
    """The outcome of one repair.

    *selected* counts the units targeted; *repaired* those transcribed with
    text afterwards and *failed* the others. *database* is the database
    written (the first, when several were) and *databases* every one;
    *outputs* are the files that hold the new rendering; *documents* counts
    the documents with pages to repair; *message* explains a repair that
    ended before any page was sent. *usage* holds the tokens of the repair
    calls, or of the finalized repair batches, under the ``repair`` role.
    """

    repaired: int = 0
    failed: int = 0
    selected: int = 0
    mode: str = "sync"
    dry_run: bool = False
    database: Path | None = None
    outputs: tuple[Path, ...] = ()
    message: str | None = None
    documents: int = 0
    databases: tuple[Path, ...] = ()
    usage: dict[str, Usage] = field(default_factory=dict)


@dataclass
class RepairPlan:
    """What a repair will do, decided before anything is written.

    *units* are the selected page units; *message* explains a plan with
    nothing to repair. *refused* marks a plan whose input changed since
    transcription: its units stay unrepaired. *identity* is the input
    identity computed at planning, checked again under the lease and
    recorded there when the document records none. *warnings* are for the front
    end to show. *args* holds the parsed command line, if any.
    """

    spec: RepairSpec
    settings: Settings
    document: RepairDocument | None = None
    units: list[UnitRow] = field(default_factory=list)
    images: ImageSettings | None = None
    schema_path: Path | None = None
    prompt_path: Path | None = None
    context: str | None = None
    context_image: Path | None = None
    message: str | None = None
    warnings: tuple[str, ...] = ()
    args: argparse.Namespace | None = None
    refused: bool = False
    identity: str | None = None

    @property
    def failures(self) -> list[int]:
        """Return the indices of the selected units."""
        return [unit.idx for unit in self.units]

    @property
    def mode(self) -> str:
        """``batch`` or ``sync``."""
        return "batch" if self.spec.run.batch else "sync"

    @property
    def dry_run(self) -> bool:
        """Whether the repair only reports what it would do."""
        return self.spec.run.dry_run

    @property
    def to_repair(self) -> bool:
        """Whether any unit is selected for repair."""
        return self.message is None and bool(self.units)

    def outcome(self) -> RepairResult:
        """Return the result of the plan before any page is sent."""
        return RepairResult(
            failed=len(self.units) if self.refused else 0,
            selected=len(self.units) if self.to_repair else 0,
            mode=self.mode,
            dry_run=self.dry_run and self.to_repair,
            message=self.message,
            documents=1 if self.to_repair else 0,
        )


NOTHING_TO_REPAIR = "No failed pages detected matching your criteria."
REFUSED_REPAIR = "No page was repaired: the input changed since it was transcribed."


@dataclass
class RepairSet:
    """The repairs of one ``repair`` command: one plan per document.

    *plans* hold the documents with pages to repair; *message* explains a
    set with none. *refused* counts the selected pages of documents whose
    input changed since transcription; they count as failed. *warnings* are
    for the front end to show.
    """

    spec: RepairSpec
    plans: list[RepairPlan] = field(default_factory=list)
    message: str | None = None
    warnings: tuple[str, ...] = ()
    refused: int = 0

    @property
    def mode(self) -> str:
        """``batch`` or ``sync``."""
        return "batch" if self.spec.run.batch else "sync"

    @property
    def dry_run(self) -> bool:
        """Whether the repair only reports what it would do."""
        return self.spec.run.dry_run

    @property
    def to_repair(self) -> bool:
        """Whether any page of any document is selected."""
        return any(plan.to_repair for plan in self.plans)

    @property
    def selected(self) -> int:
        """The number of pages selected across the documents."""
        return sum(len(plan.units) for plan in self.plans if plan.to_repair)

    def outcome(self) -> RepairResult:
        """Return the result of the set before any page is sent."""
        return RepairResult(
            failed=self.refused,
            selected=self.selected,
            mode=self.mode,
            dry_run=self.dry_run and self.to_repair,
            message=None if self.to_repair else self.message,
            documents=sum(1 for plan in self.plans if plan.to_repair),
        )


def _path_key(path: Path, root: Path) -> str:
    """Return *path* as the outputs registry of *root*'s database keys it."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _roots_above(path: Path) -> Iterator[Path]:
    """Yield the folders at or above *path* that hold the tool's database."""
    start = path if path.is_dir() else path.parent
    for folder in (start, *start.parents):
        if database_path(folder).is_file():
            yield folder


def _decoded(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


def _registered(store: Store, root: Path, doc_id: int) -> list[tuple[str, Path]]:
    """Return the document's registered text outputs as ``(format, path)``."""
    found = []
    for row in store.outputs(doc_id):
        if row.format not in TEXT_FORMATS:
            continue
        path = Path(row.path)
        found.append((row.format, path if path.is_absolute() else root / path))
    return sorted(found, key=lambda entry: TEXT_FORMATS.index(entry[0]))


def _document(
    store: Store, root: Path, doc_id: int, output: Path | None
) -> RepairDocument | None:
    row = store.document(doc_id)
    if row is None:
        return None
    registered = _registered(store, root, doc_id)
    formats = tuple(dict.fromkeys(fmt for fmt, _path in registered)) or ("txt",)
    if output is None:
        output = registered[0][1] if registered else root / f"{row.name}.txt"
    return RepairDocument(
        root=root,
        doc_id=row.doc_id,
        name=row.name,
        input_path=Path(row.input_path) if row.input_path else None,
        image_settings=_decoded(row.image_settings),
        output=output,
        formats=formats,
        lease_run_id=row.lease_run_id,
        lease_until=row.lease_until,
        input_sha256=row.input_sha256,
    )


def _by_output(store: Store, root: Path, path: Path) -> RepairDocument | None:
    key = _path_key(path, root)
    row = read(
        store.path,
        lambda connection: connection.execute(
            "SELECT doc_id FROM outputs WHERE path = ? ORDER BY doc_id", (key,)
        ).fetchone(),
    )
    if row is None:
        return None
    return _document(store, root, int(row[0]), path)


def _by_input(store: Store, root: Path, path: Path) -> RepairDocument | None:
    resolved = path.resolve()
    for row in store.documents():
        if row.input_path and Path(row.input_path).resolve() == resolved:
            return _document(store, root, row.doc_id, None)
    return None


def locate_document(path: Path) -> RepairDocument | None:
    """Return the stored document of *path*, or None; never write.

    The nearest database at or above *path* whose outputs registry lists
    *path* wins; else the nearest one with a document whose input *path* is.

    Raises:
        RepairPlanError: A database on the way is refused by the store
            (versionless, newer or unreadable).
    """
    stores: list[tuple[Path, Store]] = []
    for root in _roots_above(path):
        try:
            store = read_store(root)
        except StoreError as exc:
            raise RepairPlanError(str(exc)) from exc
        if store is None:
            continue
        found = _by_output(store, root, path)
        if found is not None:
            return found
        stores.append((root, store))
    for root, store in stores:
        found = _by_input(store, root, path)
        if found is not None:
            return found
    return None


def _selected(unit: UnitRow, causes: Iterable[str]) -> bool:
    if unit.status == "pending":
        return False
    for cause in causes:
        if cause == "api_error" and unit.status == "failed":
            return True
        if unit.outcome == CAUSE_OUTCOMES[cause]:
            return True
    return False


def _select(
    spec: RepairSpec, units: Sequence[UnitRow], blocked: set[Any]
) -> tuple[list[UnitRow], str | None, list[str]]:
    """Return the selected page units, or why none is, and the warnings."""
    causes = spec.selection.causes
    failures = [unit for unit in units if _selected(unit, causes)]
    warnings: list[str] = []
    others = [unit for unit in failures if unit.kind != KIND_PAGE]
    if others:
        kinds = ", ".join(sorted({unit.kind for unit in others}))
        warnings.append(
            f"{len(others)} failed unit(s) of kind {kinds} are not pages and are "
            "left out; transcribe the item again with --retry-errors."
        )
    failures = [unit for unit in failures if unit.kind == KIND_PAGE]
    if not failures:
        return [], NOTHING_TO_REPAIR, warnings
    if spec.selection.indices is not None:
        wanted = set(spec.selection.indices)
        failures = [unit for unit in failures if unit.idx in wanted]
        if not failures:
            return (
                [],
                "None of the specified indices matched detected failures.",
                warnings,
            )
    held = [unit for unit in failures if unit.key in blocked]
    if held:
        text = (
            f"{len(held)} selected page(s) await an open or unresolved batch "
            "submission; run 'chronotranscriber batch check' to finalize or "
            "settle it before repairing them."
        )
        failures = [unit for unit in failures if unit.key not in blocked]
        if not failures:
            return [], text, warnings
        warnings.append(text + " They are left out.")
    return failures, None, warnings


def repair_image_settings(
    document: RepairDocument,
    model: ModelSpec,
    images: ImageSpec | None,
    original_max_side_px: int | None = None,
) -> tuple[ImageSettings, str | None]:
    """Return the image settings of the repaired pages and a warning, if any.

    The document's recorded settings apply when they were made for the
    repair's model; otherwise the current settings of the model do, with
    the ``images`` setting *original_max_side_px*.

    Raises:
        ValueError: A current image setting is invalid.
    """
    recorded = document.image_settings
    if recorded is not None and recorded.get("model_name", model.model) == model.model:
        config = {key: value for key, value in recorded.items() if key != "model_type"}
        model_type = str(
            recorded.get("model_type") or detect_model_type(model.provider, model.model)
        )
        return ImageSettings(config, model_type, True), None
    if recorded is None:
        warning = (
            f"No recorded image settings for '{document.name}'; using the current ones."
        )
    else:
        warning = (
            f"Repair model {model.model} differs from the recorded model "
            f"{recorded.get('model_name')}; using the current image settings."
        )
    logger.warning(warning)
    config, model_type, _section = resolve_image_settings(
        model.provider, model.model, images, original_max_side_px
    )
    return ImageSettings(config, model_type, False), warning


def _folder_image(folder: Path, unit: UnitRow) -> Path:
    """Return the folder image of *unit*: by its name, else by its position."""
    files = list_folder_images(folder)
    names = {unit.label, unit.source_index}
    for path in files:
        if folder_image_name(path) in names or path.name in names:
            return path
    if 0 <= unit.idx < len(files):
        return files[unit.idx]
    raise FileNotFoundError(f"No image of page {unit.idx} in {folder}")


def page_image(
    document: RepairDocument, unit: UnitRow, images: ImageSettings
) -> PageImage:
    """Render the page of *unit* again from the document's recorded source.

    A page that cannot be rendered carries ``render_error``, so it is
    recorded as a transcription error.
    """
    label = unit.label or f"page_{unit.idx + 1:04d}"
    config = dict(images.config)
    source = document.input_path
    try:
        if source is None or not source.exists():
            raise FileNotFoundError(f"Recorded source missing: {source}")
        if source.is_dir():
            page = load_image(
                _folder_image(source, unit), unit.idx, config, images.model_type
            )
        elif source.suffix.lower() == ".pdf":
            index = unit.source_index
            page_index = index if isinstance(index, int) else unit.idx
            page = replace(
                render_pdf_page(source, page_index, config, images.model_type),
                index=unit.idx,
            )
        else:
            raise ValueError(f"Pages of {source.name} cannot be rendered again")
    except Exception as exc:  # noqa: BLE001 - a failed page is recorded
        logger.error("Re-render failed for %s of '%s': %s", label, document.name, exc)
        return PageImage(
            index=unit.idx,
            image_name=label,
            source_file=str(source or ""),
            render_error=str(exc),
            image_provenance={"render_error": str(exc)},
        )
    return replace(page, image_name=label)


def batch_refusal(
    spec: RunSpec | RepairSpec,
    *,
    settings: Settings,
    args: argparse.Namespace | None = None,
) -> str | None:
    """Return why the extension refuses batch submission, or None.

    The reason is the message of the
    :class:`~chronotranscriber.settings.SettingsError` that
    :func:`chronotranscriber.ext.batch_allowed` raises, else a generic one
    when it returns False.
    """
    try:
        allowed = ext.batch_allowed(spec, settings=settings, args=args)
    except SettingsError as exc:
        return str(exc)
    if allowed:
        return None
    task = "repair" if isinstance(spec, RepairSpec) else "run"
    return f"Batch submission is not available for this {task}."


def _check_access(
    spec: RepairSpec, settings: Settings, args: argparse.Namespace | None
) -> list[str]:
    """Require the model's key variable, image input and, for batch, a batch API.

    A dry run plans without the key variable; the returned warnings name
    an unset one.

    Raises:
        RepairPlanError: The repair cannot reach its provider, or the model
            does not accept the page images it resends.
    """
    warnings: list[str] = []
    model = spec.model
    if spec.run.batch:
        if not supports_batch(model.provider):
            raise RepairPlanError(
                f"Batch repair needs a provider with a batch API (openai, "
                f"anthropic, google), not {model.provider}"
            )
        refusal = batch_refusal(spec, settings=settings, args=args)
        if refusal is not None:
            raise RepairPlanError(refusal)
    try:
        variables = ext.key_variables(
            model.provider,
            model.endpoint,
            kind="batch" if spec.run.batch else "chat",
            settings=settings,
            args=args,
        )
    except SettingsError as exc:
        raise RepairPlanError(str(exc)) from exc
    if variables and not any(os.environ.get(name) for name in variables):
        unset = f"The API key variable {variables[0]} of provider {model.provider}"
        if not spec.run.dry_run:
            raise RepairPlanError(f"{unset} is not set; repair needs it.")
        warnings.append(f"{unset} is not set; the repair itself needs it.")
    try:
        ensure_image_support(model.model, model_capabilities(model, settings))
    except ValueError as exc:
        raise RepairPlanError(str(exc)) from exc
    return warnings


def _fallback(spec: RepairSpec) -> tuple[str | None, list[str]]:
    """Return the ``--fallback-context`` text under ``--context auto`` and warnings."""
    if spec.prompt.context is not None:
        return None, []
    text, warning = fallback_context(spec.prompt.fallback_context)
    return text, [] if warning is None else [warning]


def _context(
    spec: RepairSpec, document: RepairDocument, path: Path, fallback: str | None
) -> tuple[str | None, Path | None]:
    """Return the context text and image as transcription resolved them.

    Raises:
        RepairPlanError: The ``--context`` file or the ``--context-image``
            cannot be read.
    """
    anchor = document.input_path if document.input_path is not None else path
    try:
        return (
            item_context(anchor, spec.prompt.context, fallback),
            item_context_image(anchor, spec.prompt.context_image),
        )
    except ContextError as exc:
        raise RepairPlanError(str(exc)) from exc


def _lease_warning(document: RepairDocument, now: float) -> str | None:
    holder, until = document.lease_run_id, document.lease_until
    if holder is None or until is None or until <= now:
        return None
    return (
        f"'{document.name}' is leased by run {holder}; the repair is refused "
        "while that run holds it."
    )


def _input_refusal(
    document: RepairDocument, units: Sequence[UnitRow]
) -> tuple[str | None, str | None]:
    """Return why *document*'s input refuses a repair, and its current identity.

    The repair renders pages from the input, so an input whose identity
    differs from the recorded one, or cannot be computed, refuses it, and so
    does a folder whose stored page *units* no longer fit its images. A
    document without a recorded identity is otherwise trusted.
    """
    path = document.input_path
    if path is None:
        return None, None
    try:
        current = input_identity(path)
        moved = None
        if path.is_dir() and document.input_sha256 in (None, current):
            labels = {unit.idx: unit.label for unit in units if unit.kind == KIND_PAGE}
            moved = reordered(document.name, path, labels)
    except OSError as exc:
        return (
            f"Cannot compute the input identity of '{document.name}' ({exc}). "
            "Skipping it."
        ), None
    if moved is not None:
        return moved, None
    if document.input_sha256 is None or current == document.input_sha256:
        return None, current
    return changed_input(document.name, document.input_sha256, current), None


def _leased_refusal(store: Store, plan: RepairPlan) -> str | None:
    """Return why the leased document no longer fits the plan's input, or None.

    The stored identity must still be the one planned on, and the input must
    still have the identity computed at planning.
    """
    document = plan.document
    assert document is not None
    row = store.document(document.doc_id)
    stored = row.input_sha256 if row is not None else None
    if stored != document.input_sha256:
        return (
            f"The stored input identity of '{document.name}' changed since the "
            "repair was planned. Skipping it."
        )
    if plan.identity is None or document.input_path is None:
        return None
    try:
        current = input_identity(document.input_path)
    except OSError as exc:
        return (
            f"Cannot compute the input identity of '{document.name}' ({exc}). "
            "Skipping it."
        )
    if current == plan.identity:
        return None
    return (
        f"Input changed for '{document.name}' during the repair (planned "
        f"{plan.identity[:12]}..., now {current[:12]}...). Skipping it."
    )


def _plan_document(
    spec: RepairSpec,
    settings: Settings,
    document: RepairDocument,
    path: Path,
    *,
    fallback: str | None,
    args: argparse.Namespace | None,
) -> RepairPlan:
    """Select the failed pages of *document* and resolve what resends them.

    *path* anchors the context lookup of a document without a recorded
    input; *fallback* is the ``--fallback-context`` text. Provider access is
    the caller's to check.

    Raises:
        RepairPlanError: The database is refused, or the schema, prompt,
            context or image settings cannot be resolved.
    """
    try:
        store = read_store(document.root)
    except StoreError as exc:
        raise RepairPlanError(str(exc)) from exc
    if store is None:
        raise RepairPlanError(f"{document.database} holds no document.")
    units = store.units(document.doc_id)
    blocked = set(store.blocked_units(document.doc_id))
    selected, message, warnings = _select(spec, units, blocked)
    plan = RepairPlan(spec, settings, document, selected, message=message, args=args)
    leased = _lease_warning(document, time.time())
    if leased is not None:
        warnings.append(leased)
    if message is not None:
        plan.warnings = tuple(warnings)
        return plan
    refusal, plan.identity = _input_refusal(document, units)
    if refusal is not None:
        plan.message, plan.refused = refusal, True
        plan.warnings = tuple(warnings)
        return plan
    try:
        images, warning = repair_image_settings(
            document,
            spec.model,
            spec.images,
            settings.images.original_max_side_px,
        )
    except ValueError as exc:
        raise RepairPlanError(str(exc)) from exc
    if warning is not None:
        warnings.append(warning)
    plan.images = images
    plan.warnings = tuple(warnings)
    try:
        plan.schema_path = resolve_schema(spec.prompt.schema)[1]
        plan.prompt_path = system_prompt_path(spec.prompt.prompt)
    except (ValueError, FileNotFoundError) as exc:
        raise RepairPlanError(str(exc)) from exc
    plan.context, plan.context_image = _context(spec, document, path, fallback)
    return plan


def _input_path(spec: RepairSpec) -> Path:
    path = spec.input.expanduser().resolve()
    if not path.exists():
        raise RepairPlanError(f"Input path does not exist: {path}")
    return path


def _unrecorded(path: Path) -> RepairPlanError:
    return RepairPlanError(
        f"{path} is not recorded in a {DATABASE_NAME} at or above it. Repair "
        "takes a file the tool rendered or the input it transcribed, a "
        f"{DATABASE_NAME}, or a folder with databases at or below it."
    )


def plan_repair(
    spec: RepairSpec, settings: Settings, *, args: argparse.Namespace | None = None
) -> RepairPlan:
    """Find the one document of a file, select its failed pages, check the repair.

    Writes nothing. Schema, prompt, context, image settings and provider
    access are checked only when a page is selected, dry runs included; a
    dry run plans without the key variable. *args* holds the parsed command
    line, passed to the extension's batch consent and key variables and
    kept on the plan for its batch clients.

    Raises:
        RepairPlanError: The input is missing or not recorded in a store,
            its database is refused, the schema, prompt, context or image
            settings cannot be resolved, or the provider cannot be reached
            (no key, no batch API).
    """
    path = _input_path(spec)
    document = locate_document(path)
    if document is None:
        raise _unrecorded(path)
    fallback, notes = _fallback(spec)
    plan = _plan_document(spec, settings, document, path, fallback=fallback, args=args)
    if plan.to_repair:
        notes += _check_access(spec, settings, args)
        plan.warnings += tuple(notes)
    return plan


def _database_documents(root: Path) -> list[RepairDocument]:
    """Return every document of the database in *root*.

    Raises:
        RepairPlanError: The store refuses the database.
    """
    try:
        store = read_store(root)
    except StoreError as exc:
        raise RepairPlanError(str(exc)) from exc
    if store is None:
        return []
    found = (_document(store, root, row.doc_id, None) for row in store.documents())
    return [document for document in found if document is not None]


def _documents(path: Path) -> list[tuple[RepairDocument, Path]]:
    """Return the documents *path* names, each with its context anchor.

    A database names its documents; a folder that no database at or above
    it records as an input names those of every database at or below it;
    any other path names its one recorded document.

    Raises:
        RepairPlanError: A database is refused.
    """
    if path.is_file() and path.name == DATABASE_NAME:
        roots = [path.parent]
    else:
        document = locate_document(path)
        if document is not None:
            return [(document, path)]
        if not path.is_dir():
            return []
        roots = [found.parent for found in sorted(path.rglob(DATABASE_NAME))]
    return [
        (document, document.input_path or document.root)
        for root in roots
        for document in _database_documents(root)
    ]


def plan_repairs(
    spec: RepairSpec, settings: Settings, *, args: argparse.Namespace | None = None
) -> RepairSet:
    """Plan the repair of every document ``--input`` names; write nothing.

    ``--input`` is a file, a ``chronotranscriber.sqlite`` or a folder (see
    :func:`_documents`). Each document is planned like :func:`plan_repair`;
    provider access is checked once, when a page is selected.

    Raises:
        RepairPlanError: The input is missing or names no recorded document,
            ``--indices`` meets several documents, a database is refused,
            the prompt inputs cannot be resolved, or the provider cannot be
            reached.
    """
    path = _input_path(spec)
    documents = _documents(path)
    if not documents:
        if path.is_dir() or path.name == DATABASE_NAME:
            return RepairSet(spec, message=f"No stored document at or below {path}.")
        raise _unrecorded(path)
    if spec.selection.indices is not None and len(documents) > 1:
        raise RepairPlanError(
            f"--indices selects pages of one document, but {path} holds "
            f"{len(documents)}; name the transcription file instead."
        )
    fallback, warnings = _fallback(spec)
    several = len(documents) > 1
    plans: list[RepairPlan] = []
    refused = 0
    for document, anchor in documents:
        plan = _plan_document(
            spec, settings, document, anchor, fallback=fallback, args=args
        )
        prefix = f"'{document.name}': " if several else ""
        warnings += [prefix + warning for warning in plan.warnings]
        if plan.refused:
            refused += len(plan.units)
            warnings.append(str(plan.message))
        elif plan.to_repair:
            plans.append(plan)
        elif not several and plan.message is not None:
            return RepairSet(spec, [], plan.message, tuple(warnings))
    if not plans:
        message = REFUSED_REPAIR if refused else NOTHING_TO_REPAIR
        return RepairSet(spec, [], message, tuple(warnings), refused)
    warnings += _check_access(spec, settings, args)
    return RepairSet(spec, plans, None, tuple(warnings), refused)


class _DetailedTranscriber:
    """Sends every page with the recorded image detail of the repair."""

    def __init__(self, inner: PageTranscriber, options: Mapping[str, str]) -> None:
        self.inner = inner
        self.options = dict(options)

    async def transcribe_image_from_base64(
        self, image_base64: str, mime_type: str, *, label: str | None = None
    ) -> dict[str, Any]:
        return await self.inner.transcribe_image_from_base64(
            image_base64, mime_type, label=label, **self.options
        )


async def _stream(pages: Sequence[PageImage]) -> AsyncIterator[PageImage]:
    for page in pages:
        yield page


@dataclass
class _Session:
    """One repair of one leased document; *usage* sums its tokens by role."""

    plan: RepairPlan
    store: Store
    handle: DocumentHandle
    events: RunEvents
    clients: BatchClients | None = None
    usage: UsageTotals = field(default_factory=UsageTotals)

    @property
    def images(self) -> ImageSettings:
        assert self.plan.images is not None
        return self.plan.images

    @property
    def idxs(self) -> list[int]:
        return [unit.idx for unit in self.plan.units]

    def warn(self, message: str) -> None:
        logger.warning(message)
        self.events.warning(message)

    async def sync(self, pages: Sequence[PageImage]) -> bool:
        """Transcribe *pages* one call each; return True (units were written)."""
        plan = self.plan
        await self.handle.reset(self.idxs)
        env = CallEnv.create(plan.settings, self.events)
        transcriber = PageTranscriber(
            plan.spec.model,
            env,
            schema_path=plan.schema_path,
            prompt_path=plan.prompt_path,
            image_detail=plan.spec.images.detail,
            context=plan.context,
            role=ROLE,
        )
        transcriber.update_context_image(plan.context_image)
        options = self.images.request_options()
        client: Any = (
            _DetailedTranscriber(transcriber, options) if options else transcriber
        )
        try:
            await run_unit_stream(
                self.handle,
                _stream(pages),
                client,
                transcribe_payload,
                concurrency=DEFAULT_CONCURRENCY,
                delay=plan.settings.concurrency.delay_between_tasks,
            )
        except UnitsFailedError as exc:
            logger.info("Repair of '%s': %s", self.handle.name, exc)
        finally:
            await transcriber.close()
            for role in env.usage.roles:
                self.usage.add(role, env.usage.get(role))
        return True

    async def _record_pages(self, pages: Sequence[PageImage]) -> None:
        """Return the submitted units to pending with their new provenance."""
        handle = self.handle
        await handle.reset(self.idxs)
        for page in pages:
            provenance = page.provenance()
            if not provenance:
                continue
            await self.store.write_unit(
                handle.lease,
                UnitResult(
                    kind=handle.kind,
                    idx=page.index,
                    status="pending",
                    model=handle.model,
                    provider=handle.provider,
                    label=page.image_name,
                    source_index=(
                        page.page_index
                        if isinstance(page.page_index, int)
                        else page.image_name
                    ),
                    provenance=provenance,
                    attempts=0,
                ),
            )

    async def batch(self, pages: Sequence[PageImage]) -> list[Submission] | None:
        """Submit *pages*, wait for the batches and finalize them.

        Returns the finalized submissions, or None when no unit was written:
        a refused submission leaves the units as they were, an unresolved or
        unfinished one leaves them to ``batch check``.
        """
        plan, handle = self.plan, self.handle
        images = self.images
        clients = self.clients = BatchClients(plan.settings, ROLE, args=plan.args)
        try:
            body = build_plan(
                plan.spec.model,
                plan.settings,
                schema_path=plan.schema_path,
                prompt_path=plan.prompt_path,
                context=plan.context,
                context_image=plan.context_image,
                detail=images.detail,
            )
        except (OSError, ValueError) as exc:
            self.warn(f"Cannot build the repair batch requests: {exc}")
            return None
        detail = images.detail if images.recorded else None
        units = [
            UnitRequest(
                handle.key(page.index),
                None
                if page.render_error is not None
                else BatchRequest(
                    custom_id=f"page-{page.index}",
                    image_base64=page.base64,
                    mime_type=page.mime_type,
                    image_detail=detail,
                    order_index=page.index,
                ),
            )
            for page in pages
        ]
        try:
            outcome = await submit_units(
                self.store, units, plan=body, clients=clients, label=LABEL
            )
        except BatchSubmissionError as exc:
            self.warn(f"Failed to submit the repair batch: {exc}")
            return None
        if outcome.status == "refused":
            self.warn(
                "The repair batch was refused "
                f"({outcome.detail or 'no detail'}); the pages are left as they were."
            )
            return None
        if outcome.status == "unresolved":
            self.warn(
                f"The repair batch is unresolved ({outcome.detail or 'no detail'}); "
                "its pages stay blocked until 'chronotranscriber batch check' "
                "settles it."
            )
            return None
        await self._record_pages(pages)
        waited = await wait_for_submissions(
            self.store, outcome.submission_ids, clients, self.events
        )
        ids = ", ".join(outcome.submission_ids)
        if waited.error is not None or waited.running:
            reason = (
                f"their status could not be read ({waited.error})"
                if waited.error is not None
                else "they did not finish within the timeout"
            )
            self.warn(
                f"The repair batches were not finalized: {reason}. Submissions "
                f"{ids} stay in the store; run 'chronotranscriber batch check' to "
                "finalize them."
            )
            return None
        finalized: list[Submission] = []
        for submission_id in outcome.submission_ids:
            submission = self.store.submission(submission_id)
            result = await finalize_submission(self.store, submission, clients)
            if result.outcome == FINALIZED_OUTCOME:
                finalized.append(submission)
                self.usage.add(ROLE, result.usage)
            else:
                self.warn(
                    f"Repair submission {submission_id} not finalized "
                    f"({result.outcome}): {result.detail}"
                )
        return finalized or None

    async def render(self) -> tuple[Path, ...]:
        document = self.plan.document
        assert document is not None
        report = await render_item(
            self.store,
            document.doc_id,
            document.formats,
            document.output,
            self.handle.lease,
        )
        for message in report.warnings:
            self.warn(message)
        return (*report.written, *report.unchanged)

    def count(self, result: RepairResult) -> None:
        wanted = set(self.idxs)
        repaired = sum(
            1
            for unit in self.store.units(self.handle.doc_id, kind=self.handle.kind)
            if unit.idx in wanted and unit.status == "done" and unit.outcome == "ok"
        )
        result.repaired = repaired
        result.failed = len(wanted) - repaired


def _writer_options(events: RunEvents) -> WriterOptions:
    def on_busy(path: Path, waited: float) -> None:
        events.warning(
            f"The database {path} is busy (another program holds it open); "
            f"waited {waited:.0f} s. Results are kept in memory and written "
            "once it is free."
        )

    return WriterOptions(on_busy=on_busy)


async def _repair(
    store: Store, plan: RepairPlan, result: RepairResult, events: RunEvents
) -> None:
    document = plan.document
    assert document is not None
    for submission_id in await store.mark_stale_submissions():
        events.warning(
            f"Submission {submission_id} in {store.path} was left without a "
            "provider batch id; it is unresolved. Run 'chronotranscriber batch "
            "check' to reconcile it."
        )
    lease = await store.acquire(document.doc_id)
    if lease is None:
        message = (
            f"Skipping '{document.name}': another run holds its document in "
            f"{store.path}; repair it once that run has ended."
        )
        events.warning(message)
        await store.add_note(message, doc_id=document.doc_id)
        result.message = message
        result.failed = result.selected
        return
    refusal = _leased_refusal(store, plan)
    if refusal is not None:
        events.warning(refusal)
        await store.add_note(refusal, doc_id=document.doc_id)
        result.message = refusal
        result.selected = result.failed = len(plan.units)
        return
    if document.input_sha256 is None and plan.identity is not None:
        await store.write(lease, identity_write(document.doc_id, plan.identity))
        await store.add_note(
            f"Recorded the input identity of '{document.name}' for the first time.",
            doc_id=document.doc_id,
            level="info",
        )
    blocked = set(store.blocked_units(document.doc_id))
    units = [unit for unit in plan.units if unit.key not in blocked]
    if len(units) < len(plan.units):
        events.warning(
            f"{len(plan.units) - len(units)} page(s) were submitted to a batch "
            "since the repair was planned and are left out; run "
            "'chronotranscriber batch check'."
        )
    session = _Session(
        replace(plan, units=units),
        store,
        DocumentHandle(
            store,
            document.doc_id,
            document.name,
            lease,
            model=plan.spec.model.model,
            provider=plan.spec.model.provider,
        ),
        events,
    )
    result.selected = len(plan.units)
    if not units:
        result.failed = result.selected
        return
    assert plan.images is not None
    pages = [page_image(document, unit, plan.images) for unit in units]
    async with store.heartbeat():
        try:
            if plan.spec.run.batch:
                finalized = await session.batch(pages)
                written = finalized is not None
            else:
                finalized = None
                written = await session.sync(pages)
            if written:
                result.outputs = await session.render()
            if session.clients is not None:
                for submission in finalized or ():
                    await delete_provider_files(session.clients, submission)
        except LeaseLostError as exc:
            session.warn(
                f"The lease on '{document.name}' was lost; the repair stopped: {exc}"
            )
    result.usage = {role: session.usage.get(role) for role in session.usage.roles}
    session.count(result)
    result.failed += result.selected - len(units)


async def run_repair(plan: RepairPlan, events: RunEvents | None = None) -> RepairResult:
    """Repair the units *plan* selected; a plan with nothing to do returns its result.

    Errors raised here are failures of the repair, not usage errors.
    """
    result = plan.outcome()
    document = plan.document
    if document is None or not plan.to_repair or plan.dry_run:
        return result
    tools = plan.settings.tools
    configure_ffmpeg_executables(tools.ffmpeg_cmd, tools.ffprobe_cmd)
    sink = events if events is not None else NullEvents()
    result.database = document.database
    result.databases = (document.database,)
    async with open_store(document.root, writer=_writer_options(sink)) as store:
        await store.begin_run("repair", spec=effective_spec(plan.spec))
        exit_code = int(ExitCode.FAILURE)
        try:
            await _repair(store, plan, result, sink)
            exit_code = int(exit_code_for(failed=result.failed))
        finally:
            try:
                await store.end_run(exit_code)
            except sqlite3.Error as exc:
                logger.warning("Could not record the run end in %s: %s", store, exc)
    return result


async def run_repairs(
    repairs: RepairSet, events: RunEvents | None = None
) -> RepairResult:
    """Repair the documents of *repairs* one after another; sum their results.

    A set with nothing to do, or a dry run, returns its outcome.
    """
    result = repairs.outcome()
    if not repairs.to_repair or repairs.dry_run:
        return result
    result.selected = 0
    databases: list[Path] = []
    outputs: list[Path] = []
    usage = UsageTotals()
    for plan in repairs.plans:
        done = await run_repair(plan, events)
        result.repaired += done.repaired
        result.failed += done.failed
        result.selected += done.selected
        databases.extend(done.databases)
        outputs.extend(done.outputs)
        for role, tokens in done.usage.items():
            usage.add(role, tokens)
        result.usage = {role: usage.get(role) for role in usage.roles}
        if done.message is not None and result.message is None:
            result.message = done.message
    result.databases = tuple(dict.fromkeys(databases))
    result.database = result.databases[0] if result.databases else None
    result.outputs = tuple(outputs)
    return result


async def repair_transcription(
    spec: RepairSpec, settings: Settings, events: RunEvents | None = None
) -> RepairResult:
    """Plan and run the repair of the document of *spec*.

    Raises:
        RepairPlanError: The repair cannot be planned (see :func:`plan_repair`).
    """
    plan = plan_repair(spec, settings)
    sink = events if events is not None else NullEvents()
    for warning in plan.warnings:
        sink.warning(warning)
    return await run_repair(plan, sink)
