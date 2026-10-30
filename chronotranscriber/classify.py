"""The classify commands: a classifier's answers about pages or documents.

:func:`plan` turns a :class:`~chronotranscriber.spec.ClassifySpec` and the
settings into a :class:`ClassifyPlan` without API calls or writes: it
resolves the classifier (a path, a file stem in ``classifiers/`` or a
classifier name there), finds and selects the PDFs and image folders of the
input in ``transcribe``'s discovery order (EPUB and MOBI items it selects
are skipped with a warning), and names each item's store document: the name
of a stored document whose recorded input path is the item, else the name
``transcribe`` computes. It records the input identity of each item (a
file's SHA-256, an image folder's digest of its sorted image names and
their SHA-256s) and checks the decision model's key variable. An item whose
document the store holds with another input SHA-256 is refused, as the
store would refuse its rows.

:func:`run` renders the selected pages through the page pipeline (the image
settings of the default transcription model), converts them with
:func:`~chronotranscriber.common.images.decision_jpeg` at
``decisions.image_max_px`` and asks the decision model through the shared
runner under the role ``classify``. ``classify pages`` sends every selected
page as one unit, grouped into requests of the classifier's
``per_request``; its ordinal is the page's 0-based position in the item,
the ``order_index`` every transcription method gives the page in
``page_text``, and its locator the page-image name, for information only.
``classify documents`` sends the classifier's selection (the first pages)
as one unit per document, one request each, since a request asks about one
document; a page of it that fails to render fails the document unit
unasked. Answers are appended to the store's
``classifications`` component without a lease and without a document row;
answers stored for the same request (fingerprint, input, content and
classifier unchanged) are reused and appended again under the run, unless
``--force`` asks again. Classification records answers only: it writes no
transcription unit and gates nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import httpx

from chronotranscriber import ext
from chronotranscriber.common.classifiers import (
    Classifier,
    load_classifier,
    resolve_classifier,
)
from chronotranscriber.common.classify_store import (
    ChangedInputError,
    DocumentChangedError,
    RowContext,
    answer_rows,
    record_answers,
    republish_answers,
    reusable_answers,
)
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.decision_runner import (
    DEFAULT_IMAGE_BUDGET,
    Request,
    Unit,
    plan_requests,
)
from chronotranscriber.common.decisions import (
    MAX_QUESTIONS,
    Image,
    Json,
    MalformedAnswer,
    Part,
    Text,
    dialect_for,
    request_fingerprint,
)
from chronotranscriber.common.images import decision_jpeg
from chronotranscriber.common.options import Source
from chronotranscriber.common.selection import parse_page_range
from chronotranscriber.common.sqlite import read
from chronotranscriber.common.store import Store, StoreError, WriterOptions, file_sha256
from chronotranscriber.common.usage import Usage
from chronotranscriber.events import ItemFinished, ItemStarted, NullEvents, RunEvents
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.decide import DecisionModel
from chronotranscriber.llm.prompts import PROJECT_ROOT
from chronotranscriber.output.store import (
    image_settings_record,
    open_store,
    read_store,
)
from chronotranscriber.run import (
    BUSY_TIMEOUT,
    BUSY_WARN_INTERVAL,
    ConfigurationError,
    InputError,
    NoItemsError,
    StoreRefusedError,
    choose_items,
    item_location,
)
from chronotranscriber.settings import Settings, SettingsError
from chronotranscriber.sources import DiscoveryError, SourceItem, discover_items
from chronotranscriber.sources.auto import file_item, scan
from chronotranscriber.sources.images import (
    FolderPages,
    list_folder_images,
    resolve_image_settings,
    stream_pages,
)
from chronotranscriber.sources.pdf import PdfPages, PDFProcessor
from chronotranscriber.spec import (
    DEFAULT_MODEL,
    ClassifySpec,
    ImageSpec,
    effective_spec,
)

__all__ = [
    "CLASSIFIERS_DIR",
    "DOCUMENT_LOCATOR",
    "ROLE",
    "ClassifyItem",
    "ClassifyPlan",
    "ClassifyResult",
    "classifier_names",
    "plan",
    "run",
]

logger = logging.getLogger(__name__)

CLASSIFIERS_DIR: Final = PROJECT_ROOT / "classifiers"
"""The folder whose classifiers ``--classifier`` finds by name."""

ROLE: Final = "classify"
"""The usage role of the classify calls."""

DOCUMENT_LOCATOR: Final = "document"
"""The locator of a document unit."""

_CLASSIFIED_TYPES: Final = ("pdf", "images")

_ASK_ERRORS: Final = (httpx.HTTPError, MalformedAnswer, OSError, ValueError)
"""Errors of one request that fail its units and leave the item's others."""


def classifier_names() -> list[str]:
    """Return the names of the classifiers in ``classifiers/`` that load."""
    names: list[str] = []
    if CLASSIFIERS_DIR.is_dir():
        for path in sorted(CLASSIFIERS_DIR.glob("*.json")):
            try:
                names.append(load_classifier(path).name)
            except (OSError, ValueError):
                continue
    return sorted(names)


@dataclass(frozen=True)
class ClassifyItem:
    """One item to classify: its store document and the input's SHA-256.

    *input_sha256* is a file's SHA-256 or an image folder's digest; *doc_id*
    the stored document of that name at planning, if any. *reason* says why
    a refused item is left out. In a dry run *units* and *requests* count
    what the item would send; *note* says why they could not be counted.
    """

    source: SourceItem
    root: Path
    document: str
    input_sha256: str | None
    doc_id: int | None = None
    reason: str = ""
    units: int | None = None
    requests: int | None = None
    note: str | None = None

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
        """The item's source type, pdf or images."""
        return self.source.type


@dataclass(frozen=True)
class ClassifyPlan:
    """What a classify run will ask.

    *items* are the items to classify in discovery order, *refused* those
    whose stored document records another input. *fingerprint* is the
    request fingerprint every answer row carries; *image* the page-image
    settings, model family and section the pages are rendered with.
    """

    spec: ClassifySpec
    settings: Settings
    classifier: Classifier
    classifier_path: Path
    input_path: Path
    output_dir: Path | None
    items: tuple[ClassifyItem, ...]
    refused: tuple[ClassifyItem, ...]
    fingerprint: str
    image: tuple[dict[str, Any], str, str]
    warnings: tuple[str, ...] = ()
    sources: Mapping[str, Source] = field(default_factory=dict)
    missing_key: str | None = None

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.dry_run

    @property
    def dialect(self) -> str:
        """The decision API dialect of the run."""
        return dialect_for(self.spec.classify.model, self.spec.classify.provider)

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return every field's value and source as the store records them."""
        return effective_spec(self.spec, self.sources)


@dataclass(frozen=True)
class ClassifyResult:
    """The outcome of a classify run.

    *units_asked* counts the units answered by a request of this run,
    *units_reused* those whose stored answers it kept, *units_failed* those
    left without answers. *failed* and *refused* name items with the reason.
    """

    items: int = 0
    units_asked: int = 0
    units_reused: int = 0
    units_failed: int = 0
    failed: tuple[tuple[str, str], ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    usage: Mapping[str, Usage] = field(default_factory=dict)
    databases: tuple[str, ...] = ()
    seconds: float = 0.0

    @property
    def exit_code(self) -> ExitCode:
        """1 when an item failed or was refused, else 0."""
        return ExitCode.FAILURE if self.failed or self.refused else ExitCode.OK


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _classifier(spec: ClassifySpec) -> tuple[Path, Classifier]:
    """Return the file and the classifier of *spec*, checked against its unit.

    Raises:
        ConfigurationError: The classifier is not found, does not load, has
            another unit, or selects more pages than a request carries.
    """
    try:
        path = resolve_classifier(spec.classify.classifier, CLASSIFIERS_DIR)
        classifier = load_classifier(path)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    if classifier.unit != spec.unit:
        command = "pages" if spec.unit == "page" else "documents"
        raise ConfigurationError(
            f"classifier {classifier.name!r} ({path}) asks about unit "
            f"{classifier.unit}; classify {command} takes unit {spec.unit}"
        )
    selection = classifier.selection
    if selection is not None and selection.first_pages > DEFAULT_IMAGE_BUDGET:
        raise ConfigurationError(
            f"classifier {classifier.name!r} selects {selection.first_pages} pages; "
            f"a request carries at most {DEFAULT_IMAGE_BUDGET} images"
        )
    return path, classifier


def _discover(spec: ClassifySpec, input_path: Path) -> list[SourceItem]:
    """Return the items at *input_path* in transcribe's discovery order.

    Auto mode keeps EPUB and MOBI items, so ``--select`` numbers the items
    as ``transcribe`` does; :func:`_supported` sets them aside afterwards.

    Raises:
        InputError: The input cannot serve the type, or ``--files`` names
            missing entries.
        NoItemsError: The input holds no item.
    """
    kind = spec.input.type
    if kind != "auto":
        try:
            items = discover_items(
                input_path,
                kind,
                files=spec.input.files,
                recursive=spec.input.recursive,
            )
        except DiscoveryError as exc:
            raise InputError(str(exc)) from exc
    elif input_path.is_file():
        single = file_item(input_path)
        items = [single] if single is not None else []
    else:
        items = list(scan(input_path).items)
    if not items:
        raise NoItemsError(
            f"No PDFs or image folders found to classify in: {input_path}"
        )
    return items


def _supported(
    chosen: Sequence[SourceItem], input_path: Path
) -> tuple[list[SourceItem], list[str]]:
    """Return the chosen PDFs and image folders, and a warning per other item.

    Raises:
        NoItemsError: No chosen item is a PDF or an image folder.
    """
    kept = [item for item in chosen if item.type in _CLASSIFIED_TYPES]
    warnings = [
        f"Skipping '{item.name}': classify takes PDFs and image folders, not "
        f"{item.type} items."
        for item in chosen
        if item.type not in _CLASSIFIED_TYPES
    ]
    if not kept:
        raise NoItemsError(
            f"No PDF or image folder among the selected items of {input_path}"
        )
    return kept, warnings


def _input_sha256(item: SourceItem) -> str | None:
    """Return the input identity: a file's SHA-256, or a folder's image digest.

    A folder's digest hashes its supported images' file names, sorted, each
    with its SHA-256, so any added, removed or changed image changes it.
    """
    try:
        if item.path.is_file():
            return file_sha256(item.path)
        if item.type == "images" and item.path.is_dir():
            images = sorted(list_folder_images(item.path), key=lambda p: p.name)
            listing = [[path.name, file_sha256(path)] for path in images]
            text = json.dumps(listing, ensure_ascii=False)
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except OSError as exc:
        logger.warning("Could not hash %s: %s", item.path, exc)
    return None


def _same_path(recorded: str | None, path: Path) -> bool:
    if recorded is None:
        return False
    try:
        return Path(recorded).resolve() == path.resolve()
    except (OSError, ValueError):
        return False


def _stored(
    item: ClassifyItem, stores: dict[Path, Store | None]
) -> tuple[str, int | None]:
    """Return the document name to write under and its stored ``doc_id``.

    A stored document whose recorded input path is the item's input gives
    the name (the computed one when it is among several); otherwise the
    computed name stands.

    Raises:
        StoreRefusedError: The root's database is refused.
    """
    if item.root not in stores:
        try:
            stores[item.root] = read_store(item.root)
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
    store = stores[item.root]
    if store is None:
        return item.document, None
    documents = store.documents()
    matches = [doc for doc in documents if _same_path(doc.input_path, item.path)]
    names = {doc.name: doc for doc in matches}
    if item.document not in names and len(matches) == 1:
        return matches[0].name, matches[0].doc_id
    known = names.get(item.document) or next(
        (doc for doc in documents if doc.name == item.document), None
    )
    return item.document, known.doc_id if known is not None else None


def _refusal(item: ClassifyItem, stores: dict[Path, Store | None]) -> str:
    """Return why the store refuses *item*'s rows, or ''."""
    store = stores.get(item.root)
    stored = store.document(item.document) if store is not None else None
    if stored is None or stored.input_sha256 is None:
        return ""
    if stored.input_sha256 == item.input_sha256:
        return ""
    return (
        f"'{item.name}': the store holds document {item.document} with input "
        f"SHA-256 {stored.input_sha256[:12]}..., the file has "
        f"{str(item.input_sha256)[:12]}...; transcribe the changed file first"
    )


def _check_shared(items: Sequence[ClassifyItem]) -> None:
    """Raise when two items would share one store document.

    Raises:
        ConfigurationError: Two items share an output root and document name.
    """
    claimed: dict[tuple[Path, str], Path] = {}
    for item in items:
        first = claimed.setdefault((item.root, item.document.casefold()), item.path)
        if first != item.path:
            raise ConfigurationError(
                f"{first} and {item.path} would share the store document "
                f"{item.document}. Set --output DIR to keep them apart, or take "
                "one of them with --select."
            )


def _missing_key(
    spec: ClassifySpec, settings: Settings, args: argparse.Namespace | None
) -> str | None:
    """Return the decision model's unset key variable, or None.

    Raises:
        ConfigurationError: The settings name no variable for the dialect.
    """
    dialect = dialect_for(spec.classify.model, spec.classify.provider)
    try:
        variables = ext.key_variables(
            dialect, kind="decision", settings=settings, args=args
        )
    except SettingsError as exc:
        raise ConfigurationError(str(exc)) from exc
    if variables and not any(os.environ.get(name) for name in variables):
        return (
            f"The API key variable {variables[0]} of provider {dialect} is not "
            f"set; classify needs it."
        )
    return None


def _image(settings: Settings) -> tuple[dict[str, Any], str, str]:
    """Return the page-image settings, family and section pages are rendered with."""
    return resolve_image_settings(
        "openai", DEFAULT_MODEL, ImageSpec(), settings.images.original_max_side_px
    )


def _fingerprint(
    spec: ClassifySpec,
    classifier: Classifier,
    settings: Settings,
    image: tuple[dict[str, Any], str, str],
) -> str:
    """Return the request fingerprint of the run's answer rows."""
    return request_fingerprint(
        tool="chronotranscriber",
        command="classify",
        unit=spec.unit,
        dialect=dialect_for(spec.classify.model, spec.classify.provider),
        model=spec.classify.model,
        endpoint=spec.classify.endpoint,
        image_max_px=settings.decisions.image_max_px,
        per_request=classifier.per_request,
        context="document",
        upstream=image_settings_record(image[0], image[1]),
    )


def _page_indices(
    item: SourceItem, spec: ClassifySpec, classifier: Classifier
) -> tuple[list[int], int]:
    """Return the 0-based pages of *item* to send and the item's page count.

    ``classify pages`` sends the ``--pages`` selection (all by default),
    ``classify documents`` the classifier's first pages.

    Raises:
        Exception: The PDF cannot be opened.
    """
    if item.type == "images":
        total = len(list_folder_images(item.path))
    else:
        total = PDFProcessor(item.path).page_count()
    if classifier.selection is not None:
        return list(range(min(classifier.selection.first_pages, total))), total
    if spec.input.pages:
        return list(parse_page_range(spec.input.pages).resolve(total)), total
    return list(range(total)), total


def _group_size(classifier: Classifier) -> int:
    """Return the most page units one request of *classifier* carries."""
    per_unit = len(classifier.questions)
    return max(
        1, min(classifier.per_request, MAX_QUESTIONS // per_unit, DEFAULT_IMAGE_BUDGET)
    )


def _counted(
    item: ClassifyItem, spec: ClassifySpec, classifier: Classifier
) -> ClassifyItem:
    """Return *item* with the units and requests a dry run reports."""
    try:
        pages, _total = _page_indices(item.source, spec, classifier)
    except Exception as exc:  # noqa: BLE001 - the dry run names the item
        return replace(item, note=f"its pages cannot be counted ({exc})")
    if spec.unit == "document":
        units = 1 if pages else 0
        return replace(item, units=units, requests=units)
    return replace(
        item, units=len(pages), requests=math.ceil(len(pages) / _group_size(classifier))
    )


def plan(
    spec: ClassifySpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
    args: argparse.Namespace | None = None,
    check_keys: bool = True,
) -> ClassifyPlan:
    """Resolve the classifier, find, select and name the items; write nothing.

    A run whose decision key variable is unset raises unless it is a dry
    run or *check_keys* is False; the plan then names it in
    ``missing_key``, a dry run also as a warning.

    Raises:
        PlanError: The run cannot proceed; ``exit_code`` is its exit code.
    """
    path, classifier = _classifier(spec)
    output_dir = None
    if spec.output.directory is not None:
        output_dir = _absolute(spec.output.directory)
        if output_dir.exists() and not output_dir.is_dir():
            raise InputError(f"Output path exists but is not a directory: {output_dir}")
    input_path = _absolute(spec.input.path)
    if not input_path.exists():
        raise InputError(f"Input path does not exist: {input_path}")
    found = _discover(spec, input_path)
    chosen, warnings = choose_items(found, spec.input, input_path)
    supported, skipped = _supported(chosen, input_path)
    warnings += skipped
    stores: dict[Path, Store | None] = {}
    items: list[ClassifyItem] = []
    for source in supported:
        root, name, _output = item_location(
            source,
            output_dir=output_dir,
            layout=spec.output.layout,
            input_root=input_path,
        )
        item = ClassifyItem(source, root, name, _input_sha256(source))
        name, doc_id = _stored(item, stores)
        items.append(replace(item, document=name, doc_id=doc_id))
    _check_shared(items)
    checked = [replace(item, reason=_refusal(item, stores)) for item in items]
    to_classify = [item for item in checked if not item.reason]
    refused = [item for item in checked if item.reason]
    if spec.dry_run:
        to_classify = [_counted(item, spec, classifier) for item in to_classify]
    missing = _missing_key(spec, settings, args)
    if missing is not None and check_keys and not spec.dry_run:
        raise ConfigurationError(missing, items_total=len(to_classify))
    if missing is not None and spec.dry_run:
        warnings.append(missing)
    image = _image(settings)
    return ClassifyPlan(
        spec=spec,
        settings=settings,
        classifier=classifier,
        classifier_path=path,
        input_path=input_path,
        output_dir=output_dir,
        items=tuple(to_classify),
        refused=tuple(refused),
        fingerprint=_fingerprint(spec, classifier, settings, image),
        image=image,
        warnings=tuple(warnings),
        sources=dict(sources or {}),
        missing_key=missing,
    )


class _Stores:
    """The stores of a classify run: one per output root, opened on first use."""

    def __init__(self, run_plan: ClassifyPlan, events: RunEvents) -> None:
        self._plan = run_plan
        self._events = events
        self._stores: dict[Path, Store] = {}

    @property
    def databases(self) -> tuple[str, ...]:
        """The database files of the opened stores, in opening order."""
        return tuple(str(store.path.resolve()) for store in self._stores.values())

    def _writer(self) -> WriterOptions:
        events = self._events

        def on_busy(path: Path, waited: float) -> None:
            events.warning(
                f"The database {path} is busy (another program holds it open); "
                f"waited {waited:.0f} s. Answers are written once it is free."
            )

        return WriterOptions(
            busy_timeout=BUSY_TIMEOUT, warn_interval=BUSY_WARN_INTERVAL, on_busy=on_busy
        )

    async def get(self, root: Path) -> Store:
        """Return the open store of output root *root*, stamped with the run.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        key = root.resolve()
        store = self._stores.get(key)
        if store is not None:
            return store
        key.mkdir(parents=True, exist_ok=True)
        store = open_store(key, writer=self._writer())
        try:
            await store.open()
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        self._stores[key] = store
        await store.begin_run("classify", spec=self._plan.effective_spec())
        return store

    async def close(self, exit_code: int) -> None:
        """Record the end of the run and close the stores."""
        for store in self._stores.values():
            try:
                if store.run_id is not None:
                    await store.end_run(exit_code)
                await store.close()
            except sqlite3.Error as exc:
                logger.warning("Could not close %s: %s", store.path, exc)


@dataclass
class _Tally:
    """The counts of one item."""

    asked: int = 0
    reused: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class _ItemClassifier:
    """Ask a classifier about the units of the plan's items."""

    def __init__(
        self, run_plan: ClassifyPlan, model: DecisionModel, events: RunEvents
    ) -> None:
        self.plan = run_plan
        self.classifier = run_plan.classifier
        self.model = model
        self.events = events
        self.image_max_px = run_plan.settings.decisions.image_max_px

    async def _pages(
        self, item: ClassifyItem, indices: list[int], tally: _Tally
    ) -> AsyncIterator[tuple[int, str, bytes]]:
        """Yield ``(index, image name, decision JPEG)`` of each page that renders."""
        img_cfg, model_type, section = self.plan.image
        pages_type = FolderPages if item.type == "images" else PdfPages
        source = pages_type(
            item.source.path, dict(img_cfg), model_type, section=section
        )
        async for page in stream_pages(source, indices):
            data: bytes | None = None
            error = page.render_error
            if error is None and page.base64:
                try:
                    data = decision_jpeg(
                        base64.b64decode(page.base64), self.image_max_px
                    )
                except OSError as exc:
                    error = str(exc)
            if data is None:
                tally.failed += 1
                tally.errors.append(f"{page.image_name}: {error or 'no image'}")
                continue
            yield page.index, page.image_name, data

    async def _page_units(
        self, item: ClassifyItem, indices: list[int], tally: _Tally
    ) -> AsyncIterator[list[Unit]]:
        """Yield the page units of *item* in groups of one request each."""
        size = _group_size(self.classifier)
        group: list[Unit] = []
        async for index, name, data in self._pages(item, indices, tally):
            group.append(Unit(name, index, (Image(data),)))
            if len(group) == size:
                yield group
                group = []
        if group:
            yield group

    async def _document_unit(
        self, item: ClassifyItem, indices: list[int], tally: _Tally
    ) -> list[Unit]:
        """Return the document unit of *item*: its selected pages, labelled.

        A page that fails to render fails the whole unit, counted once, so a
        document is never judged from part of its selection.
        """
        pages = _Tally()
        parts: list[Part] = []
        async for index, _name, data in self._pages(item, indices, pages):
            parts += [Text(f"Page {index + 1}:"), Image(data)]
        if pages.failed or not parts:
            tally.failed += 1
            tally.errors.extend(pages.errors or ["no page rendered"])
            return []
        return [Unit(DOCUMENT_LOCATOR, 0, tuple(parts))]

    async def classify(self, item: ClassifyItem, store: Store) -> _Tally:
        """Ask about the units of *item*, reusing stored answers.

        A ``--pages`` range that selects none of the item's pages fails it.

        Raises:
            ChangedInputError: The store came to hold the document with
                another input SHA-256, or (:class:`DocumentChangedError`)
                dropped or replaced it since planning.
        """
        tally = _Tally()
        try:
            indices, total = _page_indices(item.source, self.plan.spec, self.classifier)
        except Exception as exc:  # noqa: BLE001 - one unreadable item fails alone
            tally.errors.append(f"its pages cannot be read ({exc})")
            return tally
        if not total:
            self.events.warning(f"'{item.name}' has no page to classify.")
            return tally
        if not indices:
            tally.errors.append(
                f"page range '{self.plan.spec.input.pages}' selects none of its "
                f"{total} pages"
            )
            return tally
        context = RowContext(
            document=item.document,
            fingerprint=self.plan.fingerprint,
            input_sha256=item.input_sha256,
            run_id=store.run_id,
            seen_doc_id=item.doc_id,
        )
        shared: tuple[Part, ...] = (Json({"document": item.document}),)
        try:
            if self.plan.spec.unit == "document":
                unit = await self._document_unit(item, indices, tally)
                for request in plan_requests(
                    self.classifier, item.document, unit, context=shared
                ):
                    await self._settle(request, store, context, tally)
            else:
                async for units in self._page_units(item, indices, tally):
                    for request in plan_requests(
                        self.classifier, item.document, units, context=shared
                    ):
                        await self._settle(request, store, context, tally)
        except RuntimeError as exc:  # too many pages failed to render
            tally.errors.append(str(exc))
        return tally

    def _reusable(
        self, request: Request, store: Store, context: RowContext
    ) -> list[dict[str, dict[str, Any]]] | None:
        """Return each unit's stored answers when all of them are reusable."""
        classifier = self.classifier

        def lookup(connection: sqlite3.Connection) -> list[dict[str, dict[str, Any]]]:
            return [
                reusable_answers(
                    connection,
                    context,
                    unit=classifier.unit,
                    locator=unit.locator,
                    classifier=classifier.name,
                    content_sha256=unit.content_sha256,
                    request_sha256=request.sha256,
                )
                for unit in request.units
            ]

        names = {question.name for question in classifier.questions}
        found = read(store.path, lookup, busy_timeout=BUSY_TIMEOUT)
        return found if all(set(answers) == names for answers in found) else None

    async def _settle(
        self, request: Request, store: Store, context: RowContext, tally: _Tally
    ) -> None:
        """Reuse the stored answers of *request*, or ask and record them."""
        units = request.units
        if not self.plan.spec.classify.force:
            reused = self._reusable(request, store, context)
            if reused is not None:
                for unit, answers in zip(units, reused, strict=True):
                    await republish_answers(
                        store,
                        answers,
                        run_id=store.run_id,
                        locator=unit.locator,
                        ordinal=unit.ordinal,
                        seen_doc_id=context.seen_doc_id,
                    )
                tally.reused += len(units)
                return
        label = (
            f"{request.document} {self.classifier.name} "
            f"{units[0].locator}..{units[-1].locator}"
        )
        try:
            result = await self.model.runner.ask(request, label=label)
        except _ASK_ERRORS as exc:
            tally.failed += len(units)
            tally.errors.append(f"{label}: {exc}")
            return
        rows = answer_rows(result, self.classifier, context, store.clock())
        await record_answers(store, rows, seen_doc_id=context.seen_doc_id)
        tally.asked += len(units)


@dataclass
class _Outcomes:
    """The counts and the failed and refused items of a run."""

    items: int = 0
    asked: int = 0
    reused: int = 0
    failed_units: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)


async def _run_items(
    run_plan: ClassifyPlan,
    model: DecisionModel,
    stores: _Stores,
    events: RunEvents,
    outcomes: _Outcomes,
) -> None:
    worker = _ItemClassifier(run_plan, model, events)
    total = len(run_plan.items)
    for index, item in enumerate(run_plan.items, start=1):
        events.item_started(ItemStarted(item.name, item.type, index, total))
        store = await stores.get(item.root)
        outcomes.items += 1
        try:
            tally = await worker.classify(item, store)
        except DocumentChangedError as exc:
            outcomes.failed.append((item.name, str(exc)))
            events.item_finished(
                ItemFinished(item.name, index, total, False, error=str(exc))
            )
            continue
        except ChangedInputError as exc:
            outcomes.refused.append((item.name, str(exc)))
            events.item_finished(
                ItemFinished(item.name, index, total, False, error=str(exc))
            )
            continue
        outcomes.asked += tally.asked
        outcomes.reused += tally.reused
        outcomes.failed_units += tally.failed
        error = "; ".join(tally.errors) if tally.errors else None
        if error is not None:
            outcomes.failed.append((item.name, error))
        events.item_finished(
            ItemFinished(item.name, index, total, error is None, error=error)
        )


async def run(
    run_plan: ClassifyPlan, events: RunEvents | None = None
) -> ClassifyResult:
    """Ask the plan's classifier about its items, one request at a time.

    Each output root's store is opened on first use and stamped with the
    run. A request that fails after its retries leaves its units without
    answers and fails the item, whose other requests still run; an item
    whose stored document came to record another input is refused.
    Cancellation propagates.

    Raises:
        ValueError: The plan is a dry run.
        ConfigurationError: The decision client cannot be built.
    """
    if run_plan.dry_run:
        raise ValueError("a dry-run plan is reported, not run")
    sink: RunEvents = events if events is not None else NullEvents()
    env = CallEnv.create(run_plan.settings, sink)
    started = time.monotonic()
    outcomes = _Outcomes(
        refused=[(item.name, item.reason) for item in run_plan.refused]
    )
    stores = _Stores(run_plan, sink)
    exit_code: int = ExitCode.FAILURE
    model: DecisionModel | None = None
    try:
        if run_plan.items:
            classify = run_plan.spec.classify
            try:
                model = DecisionModel(
                    classify.model, classify.provider, classify.endpoint, env, role=ROLE
                )
            except (OSError, ValueError) as exc:
                raise ConfigurationError(
                    f"Cannot build the decision client: {exc}",
                    items_total=len(run_plan.items),
                ) from exc
            await _run_items(run_plan, model, stores, sink, outcomes)
        failed = outcomes.failed or outcomes.refused
        exit_code = ExitCode.FAILURE if failed else ExitCode.OK
    except (asyncio.CancelledError, KeyboardInterrupt):
        exit_code = ExitCode.INTERRUPTED
        raise
    finally:
        await stores.close(exit_code)
        if model is not None:
            await model.aclose()
    return ClassifyResult(
        items=outcomes.items,
        units_asked=outcomes.asked,
        units_reused=outcomes.reused,
        units_failed=outcomes.failed_units,
        failed=tuple(outcomes.failed),
        refused=tuple(outcomes.refused),
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=stores.databases,
        seconds=time.monotonic() - started,
    )
