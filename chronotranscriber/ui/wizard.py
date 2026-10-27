"""The steps of a new transcription in the guided UI.

Steps in order: input, type and method (with the items to process), output
and formats, model, response format (with the prompt and context), page
images, Tesseract or audio options, pages, and the run options with batch
or sync. Each :class:`Step`
names the options it may set and applies only to the methods that use
them. Answers act like command-line flags over the settings defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from chronotranscriber import run as core
from chronotranscriber.batch.repair import batch_refusal
from chronotranscriber.common.batch import supports_batch
from chronotranscriber.common.selection import (
    is_numeric_selection,
    parse_page_range,
    select_items,
)
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    Choice,
    Nav,
    State,
    Step,
    StepContext,
)
from chronotranscriber.sources import SourceItem, auto
from chronotranscriber.spec import (
    METHODS_BY_TYPE,
    OCR_PRESETS,
    OPTIONS,
    OUTPUT_FORMATS,
    SOURCE_TYPES,
    WHISPER_MODELS,
    SpecError,
    resolve_spec,
)
from chronotranscriber.ui.questions import (
    IMAGE_OPTIONS,
    MODEL_OPTIONS,
    RESPONSE_OPTIONS,
    ImageQuestions,
    ModelQuestions,
    Question,
    ResponseQuestions,
    resolution_of,
)
from chronotranscriber.ui.session import (
    Inventory,
    Session,
    absolute,
    has_settings_default,
    inventory_of,
    plural,
    take_inventory,
)

__all__ = [
    "AUDIO_OPTIONS",
    "ITEM_OPTIONS",
    "METHOD_TITLES",
    "OCR_OPTIONS",
    "RUN_FLAGS",
    "RUN_OPTIONS",
    "STEP_NAMES",
    "TYPE_TITLES",
    "batch_available",
    "build_steps",
    "file_names",
    "offered_types",
    "run_items",
    "unit_word",
]

STEP_NAMES: Final = (
    "input",
    "method",
    "output",
    "model",
    "response",
    "images",
    "ocr",
    "audio",
    "pages",
    "run",
)
ITEM_OPTIONS: Final = ("files", "select", "all")
OCR_OPTIONS: Final = ("ocr_config", "ocr_preprocess")
AUDIO_OPTIONS: Final = ("whisper_model", "language", "audio_chunk_seconds")
RUN_FLAGS: Final = ("retry_errors", "force", "keep_working_files", "dry_run")
RUN_OPTIONS: Final = ("batch", "sync_fallback", *RUN_FLAGS, "concurrency")

TYPE_TITLES: Final = {
    "auto": "Automatic (each file by its type)",
    "pdf": "PDF",
    "images": "Image folders",
    "epub": "EPUB",
    "mobi": "MOBI",
    "audio": "Audio",
}
METHOD_TITLES: Final = {
    "llm": "LLM (vision model)",
    "tesseract": "Tesseract (local OCR)",
    "native": "Native text (text layer or ebook text)",
    "whisper": "Whisper (local speech-to-text)",
    "audio-api": "Audio API (OpenAI or Google)",
}
_UNITS = {
    "auto": "pages or images",
    "pdf": "pages",
    "images": "images",
    "epub": "sections",
    "mobi": "sections",
    "audio": "audio chunks",
}
_RUN_FLAG_TITLES = {
    "retry_errors": "Retry units whose earlier result is an error",
    "force": "Transcribe complete items again (--force)",
    "keep_working_files": "Keep working files (preprocessed images, audio chunks)",
    "dry_run": "Dry run: show the plan, no API calls or writes",
}
_WORKING_FILE_METHODS = frozenset({"tesseract", "whisper", "audio-api"})
_MODEL_METHODS = frozenset({"llm", "audio-api"})
_AUDIO_METHODS = frozenset({"whisper", "audio-api"})


def _method(state: State) -> str:
    return str(state.value("method"))


def _input(state: State) -> Path | None:
    value = state.value("input")
    return None if value is None else absolute(value)


def unit_word(state: State) -> str:
    """Return what ``--pages`` counts for the chosen type and method."""
    if _method(state) in _AUDIO_METHODS:
        return _UNITS["audio"]
    return _UNITS.get(str(state.value("type")), "pages")


def _existing_input(text: str) -> str | None:
    if not text.strip():
        return "enter a file or a folder"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


class _InputQuestions:
    """The questions of the input step: path, subfolders, a check."""

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state

    def questions(self) -> list[Question]:
        return [self.path, self.recursive, self.found]

    def _show_found(self, location: Path) -> Inventory:
        inventory = inventory_of(self.state, location)
        self.ctx.prompter.show(f"  found {inventory.describe()}")
        return inventory

    def path(self) -> Nav:
        """Ask the input.

        An input with nothing to process, subfolders included, is asked
        again in place.
        """
        state = self.state
        current = _input(state)
        while True:
            answer = self.ctx.prompter.path(
                "Input (file or folder):",
                default="" if current is None else str(current),
                validate=_existing_input,
            )
            if answer is BACK:
                return BACK
            location = absolute(answer)
            state.set("input", location)
            if not location.is_dir():
                state.clear("recursive")
            if inventory_of(state, location).items or _holds_items(location):
                self._show_found(location)
                return DONE
            self.ctx.prompter.show("  nothing to process; choose another input")
            current = location

    def recursive(self) -> Nav | None:
        location = _input(self.state)
        if location is None or not location.is_dir():
            return None
        before = bool(self.state.value("recursive"))
        result = self.ctx.select_option(
            "recursive",
            "Search the subfolders too?",
            [
                Choice(False, "No, this folder only"),
                Choice(True, "Yes, every subfolder"),
            ],
        )
        if result is DONE and bool(self.state.value("recursive")) != before:
            self._show_found(location)
        return result

    def found(self) -> Nav | None:
        location = _input(self.state)
        if location is None or inventory_of(self.state, location).items:
            return None
        self.ctx.prompter.show("  nothing to process here; search the subfolders")
        return BACK


def _holds_items(location: Path) -> bool:
    """Whether a folder holds items in its subfolders (a file holds none)."""
    return location.is_dir() and bool(take_inventory(location, recursive=True).items)


def _ask_input(ctx: StepContext) -> Nav:
    return StepContext.sequence(_InputQuestions(ctx).questions())


def _auto_finds_items(location: Path) -> bool:
    if location.is_dir():
        return bool(auto.scan(location).items)
    return auto.file_item(location) is not None


def offered_types(state: State) -> list[str]:
    """Return the types the input can be run as: auto, then those it holds.

    Auto is offered only when auto mode finds an item in the input (a PDF,
    EPUB or MOBI file, or such files and image folders directly in a folder)
    and the subfolder search is off, since auto mode does not search it.
    """
    location = _input(state)
    if location is None:
        return ["auto"]
    inventory = inventory_of(state, location)
    held = {item.type for item in inventory.items}
    types = [kind for kind in SOURCE_TYPES if kind != "auto" and kind in held]
    if not state.value("recursive") and _auto_finds_items(location):
        types.insert(0, "auto")
    return types


def _of_type(inventory: Inventory, kind: str) -> list[SourceItem]:
    return [
        item
        for item in inventory.items
        if item.type == kind and item.path != inventory.path
    ]


def _relative(item: SourceItem, root: Path) -> str:
    try:
        return item.path.relative_to(root).as_posix()
    except ValueError:
        return item.name


def run_items(state: State) -> list[SourceItem]:
    """Return the items the run finds, before any selection.

    The chosen input, type and subfolder search decide the items; they come
    in the order ``--select`` numbers them
    (:func:`chronotranscriber.run.selection_order`).
    """
    location = _input(state)
    if location is None:
        return []
    kind = str(state.value("type"))
    flags: dict[str, Any] = {"input": location, "type": kind}
    if kind != "auto":
        flags["recursive"] = bool(state.value("recursive"))
    try:
        return core.selection_order(core.discover(resolve_spec(flags).spec))
    except (core.PlanError, SpecError):
        return []


def file_names(state: State) -> tuple[list[str], bool]:
    """Return the ``--files`` names of the chosen type and whether they are needed.

    A folder run of image folders without the subfolder search processes the
    input folder itself, so its image subfolders must be named.
    """
    location = _input(state)
    kind = str(state.value("type"))
    if location is None or kind == "auto" or not location.is_dir():
        return [], False
    inventory = inventory_of(state, location)
    names = [_relative(item, location) for item in _of_type(inventory, kind)]
    needed = kind == "images" and bool(names) and not state.value("recursive")
    return names, needed


_ALL = "all"
_RANGE = "range"
_SEARCH = "search"
_LIST = "list"
_LISTED = 30
"""The numbered items shown before a number or range is asked."""


def _item_mode(state: State) -> str:
    """Return how the state's answers select the items."""
    if state.value("files"):
        return _LIST
    chosen = state.value("select")
    if chosen:
        return _RANGE if is_numeric_selection(str(chosen)) else _SEARCH
    return _ALL


def _clear_items(state: State) -> None:
    for name in ITEM_OPTIONS:
        state.clear(name)


class _MethodQuestions:
    """The questions of the type and method step, with the items to process.

    A folder with several items asks ``Which items?``: all of them, by
    number or range, by name search, or from a checklist. The checklist
    names ``--files`` for one type, and numbers ``--select`` in auto mode.
    """

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.how: str | None = None

    def questions(self) -> list[Question]:
        return [
            self.kind,
            self.method,
            self.searchable,
            self.which,
            self.numbers,
            self.search,
            self.pick,
        ]

    def kind(self) -> Nav:
        state = self.state
        location = _input(state)
        inventory = inventory_of(state, location) if location is not None else None
        before = state.value("type")
        choices = []
        for kind in offered_types(state):
            note = None
            if inventory is not None and kind != "auto":
                count = sum(1 for item in inventory.items if item.type == kind)
                note = plural(count, "item")
            choices.append(Choice(kind, TYPE_TITLES[kind], note))
        result = self.ctx.select_option("type", "Input type:", choices)
        if result is DONE and state.value("type") != before:
            _clear_items(state)
        if result is DONE and state.value("type") == "auto":
            state.clear("files")
        return result

    def method(self) -> Nav:
        kind = str(self.state.value("type"))
        message = "Method:"
        if kind == "auto":
            message = "Method for scanned PDFs and image folders:"
        choices = [Choice(name, METHOD_TITLES[name]) for name in METHODS_BY_TYPE[kind]]
        return self.ctx.select_option("method", message, choices)

    def searchable(self) -> Nav | None:
        if self.state.value("type") != "auto":
            return None
        return self.ctx.select_option(
            "searchable_pdfs",
            "PDFs with a text layer:",
            [
                Choice("ocr", "Transcribe them with the method"),
                Choice("text", "Take their text layer"),
            ],
        )

    def which(self) -> Nav | None:
        """Ask which items: all, by number or range, by name, or a checklist.

        Image subfolders that only ``--files`` reaches offer all or the
        checklist.
        """
        state = self.state
        names, needed = file_names(state)
        count = len(names) if needed else len(run_items(state))
        if count < 2 and not needed:
            _clear_items(state)
            return None
        choices = [Choice(_ALL, f"All {count}")]
        if not needed:
            choices += [
                Choice(_RANGE, "By number or range", "such as 1-3 or 2,5"),
                Choice(_SEARCH, "By name search", "names that contain a text"),
            ]
        choices.append(Choice(_LIST, "Choose from the list"))
        current = _item_mode(state)
        answer = self.ctx.prompter.select(
            "Which items?",
            choices,
            default=current if any(c.value == current for c in choices) else _ALL,
        )
        if answer is BACK:
            return BACK
        self.how = answer
        if answer == _ALL:
            _clear_items(state)
            if needed:
                state.set("files", tuple(names))
            else:
                state.set("all", True)
        return DONE

    def _set_select(self, expression: str) -> None:
        _clear_items(self.state)
        self.state.set("select", expression)

    def numbers(self) -> Nav | None:
        """Ask item numbers or ranges after showing the numbered items."""
        if self.how != _RANGE:
            return None
        items = run_items(self.state)
        names = [item.name for item in items]
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
            f"Items (1-{len(names)}):",
            default=str(current) if _item_mode(self.state) == _RANGE else "",
            validate=validate,
        )
        if answer is BACK:
            return BACK
        self._set_select(answer.strip())
        return DONE

    def search(self) -> Nav | None:
        """Ask a text the names of the items to process contain."""
        if self.how != _SEARCH:
            return None
        names = [item.name for item in run_items(self.state)]

        def validate(text: str) -> str | None:
            if not text.strip():
                return "enter a part of a name"
            if is_numeric_selection(text):
                return "a number selects by position; choose By number or range"
            if not select_items(names, text).indices:
                return "no item name contains this text"
            return None

        current = self.state.value("select")
        answer = self.ctx.prompter.text(
            "Name contains:",
            default=str(current) if _item_mode(self.state) == _SEARCH else "",
            validate=validate,
        )
        if answer is BACK:
            return BACK
        self._set_select(answer.strip())
        matched = select_items(names, answer.strip()).pick(names)
        self.ctx.prompter.show(
            f"  {plural(len(matched), 'item')}: {', '.join(matched)}"
        )
        return DONE

    def pick(self) -> Nav | None:
        """Ask the items from a checklist: ``--files`` names, auto mode numbers."""
        if self.how != _LIST:
            return None
        state = self.state
        if state.value("type") == "auto":
            return self._pick_numbers()
        names, _needed = file_names(state)
        current = [name for name in state.value("files") or () if name in names]
        while True:
            answer = self.ctx.prompter.checkbox(
                "Files:", [Choice(name, name) for name in names], checked=current
            )
            if answer is BACK:
                return BACK
            if answer:
                _clear_items(state)
                state.set("files", tuple(answer))
                return DONE
            self.ctx.prompter.show("  select at least one item")

    def _pick_numbers(self) -> Nav:
        names = [item.name for item in run_items(self.state)]
        chosen = self.state.value("select")
        checked: list[int] = []
        if chosen and is_numeric_selection(str(chosen)):
            checked = [index + 1 for index in select_items(names, str(chosen)).indices]
        choices = [
            Choice(index, f"{index:>4}  {name}")
            for index, name in enumerate(names, start=1)
        ]
        while True:
            answer = self.ctx.prompter.checkbox("Items:", choices, checked=checked)
            if answer is BACK:
                return BACK
            if answer:
                self._set_select(",".join(str(index) for index in sorted(answer)))
                return DONE
            self.ctx.prompter.show("  select at least one item")


def _ask_method(ctx: StepContext) -> Nav:
    return StepContext.sequence(_MethodQuestions(ctx).questions())


def _folder(text: str) -> str | None:
    if not text.strip():
        return "enter a folder"
    if absolute(text).is_file():
        return "this is a file; enter a folder"
    return None


_FORMAT_TITLES = {"txt": "Plain text (.txt)", "md": "Markdown (.md)"}


def _ask_output(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def where() -> Nav:
        folder = state.value("output") is not None
        answer = ctx.prompter.select(
            "Where should the outputs go?",
            [
                Choice("beside", "Beside each input"),
                Choice("folder", "Into one folder", "created when missing"),
            ],
            default="folder" if folder else "beside",
            marker=state.marker("output") if folder else None,
        )
        if answer is BACK:
            return BACK
        chosen["where"] = answer
        if answer == "folder":
            state.clear("beside_input")
        elif has_settings_default(state, "output"):
            state.set("beside_input", True)
        else:
            state.clear("output")
            state.clear("beside_input")
        return DONE

    def folder() -> Nav | None:
        if chosen.get("where") != "folder":
            return None
        current = state.value("output")
        answer = ctx.prompter.path(
            "Output folder:",
            default="" if current is None else str(current),
            validate=_folder,
            only_directories=True,
            marker=state.marker("output"),
        )
        if answer is BACK:
            return BACK
        state.set("output", absolute(answer))
        return DONE

    def layout() -> Nav | None:
        if chosen.get("where") != "folder":
            return None
        return ctx.select_option(
            "layout",
            "Folder layout:",
            [
                Choice("hash", "One folder per item", "hash-suffixed names"),
                Choice("mirror", "Mirror the input folders"),
            ],
        )

    def formats() -> Nav:
        choices = [Choice(name, _FORMAT_TITLES[name]) for name in OUTPUT_FORMATS]
        while True:
            answer = ctx.prompter.checkbox(
                "Output formats (chronotranscriber.sqlite is always written):",
                choices,
                checked=tuple(state.value("formats") or ()),
                marker=state.marker("formats"),
            )
            if answer is BACK:
                return BACK
            if answer:
                state.set("formats", OPTIONS["formats"].convert(list(answer)))
                return DONE
            ctx.prompter.show("  select at least one format")

    return StepContext.sequence([where, folder, layout, formats])


def _uses_llm(state: State) -> bool:
    return _method(state) == "llm"


def _model_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(ModelQuestions(ctx, session).questions())

    return Step(
        "model",
        "Model",
        ask=ask,
        sets=MODEL_OPTIONS,
        applies=lambda state: _method(state) in _MODEL_METHODS,
    )


def _response_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(ResponseQuestions(ctx, session).questions())

    return Step(
        "response",
        "Response format",
        ask=ask,
        sets=RESPONSE_OPTIONS,
        applies=_uses_llm,
    )


def _images_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(ImageQuestions(ctx, session).questions())

    return Step("images", "Page images", ask=ask, sets=IMAGE_OPTIONS, applies=_uses_llm)


def _ask_ocr(ctx: StepContext) -> Nav:
    def config() -> Nav:
        return ctx.text_option("ocr_config", "Tesseract configuration:")

    def preprocess() -> Nav:
        return ctx.select_option(
            "ocr_preprocess",
            "Image preprocessing:",
            [Choice(name, name) for name in OCR_PRESETS],
        )

    return StepContext.sequence([config, preprocess])


def _ask_audio(ctx: StepContext) -> Nav:
    state = ctx.state

    def whisper() -> Nav | None:
        if _method(state) != "whisper":
            return None
        names = list(WHISPER_MODELS)
        current = state.value("whisper_model")
        if current not in names:
            names.insert(0, current)
        return ctx.select_option(
            "whisper_model",
            "Whisper model:",
            [Choice(name, name) for name in names],
        )

    def language() -> Nav:
        return ctx.text_option(
            "language",
            "Spoken language (ISO 639-1 code, empty = detect):",
            empty=None,
        )

    def chunk() -> Nav:
        return ctx.text_option("audio_chunk_seconds", "Chunk length in seconds:")

    return StepContext.sequence([whisper, language, chunk])


def _page_range(text: str) -> str | None:
    try:
        parse_page_range(text)
    except ValueError as exc:
        return str(exc)
    return None


def _ask_pages(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def mode() -> Nav:
        unit = unit_word(state)
        answer = ctx.prompter.select(
            f"Which {unit}?",
            [
                Choice("all", f"All {unit}"),
                Choice(
                    "range",
                    "A range",
                    "such as 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8",
                ),
            ],
            default="range" if state.value("pages") else "all",
        )
        if answer is BACK:
            return BACK
        chosen["mode"] = answer
        if answer == "all":
            state.clear("pages")
        return DONE

    def text() -> Nav | None:
        if chosen.get("mode") != "range":
            return None
        answer = ctx.prompter.text(
            "Range:", default=state.value("pages") or "", validate=_page_range
        )
        if answer is BACK:
            return BACK
        state.set("pages", answer.strip())
        return DONE

    return StepContext.sequence([mode, text])


def batch_available(state: State, session: Session) -> bool:
    """Whether the run may use the batch API: LLM, provider and extension."""
    if not _uses_llm(state):
        return False
    try:
        spec = resolution_of(state, session).spec
    except ValueError:
        return False
    return (
        supports_batch(spec.model.provider)
        and batch_refusal(spec, settings=session.settings, args=session.args) is None
    )


def _run_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state

        def submission() -> Nav | None:
            if not batch_available(state, session):
                state.set("batch", False)
                state.clear("sync_fallback")
                return None
            return ctx.select_option(
                "batch",
                "Submission:",
                [
                    Choice(False, "Synchronous", "results now"),
                    Choice(True, "Batch API", "collected later with batch check"),
                ],
            )

        def fallback() -> Nav | None:
            if not state.value("batch"):
                return None
            return ctx.select_option(
                "sync_fallback",
                "If a batch submission fails:",
                [
                    Choice(False, "Stop with an error"),
                    Choice(True, "Transcribe synchronously"),
                ],
            )

        def flags() -> Nav:
            names = [
                name
                for name in RUN_FLAGS
                if name != "keep_working_files"
                or _method(state) in _WORKING_FILE_METHODS
            ]
            choices = [
                Choice(name, _RUN_FLAG_TITLES[name], state.marker(name))
                for name in names
            ]
            answer = ctx.prompter.checkbox(
                "Run options:",
                choices,
                checked=[name for name in names if state.value(name)],
            )
            if answer is BACK:
                return BACK
            for name in names:
                state.set(name, name in answer)
            return DONE

        def concurrency() -> Nav | None:
            if state.value("batch") and not state.value("sync_fallback"):
                return None
            return ctx.text_option("concurrency", "Parallel requests:")

        return StepContext.sequence([submission, fallback, flags, concurrency])

    return Step("run", "Run options", ask=ask, sets=RUN_OPTIONS)


def build_steps(session: Session) -> list[Step]:
    """Return the transcribe steps in order, task and extension steps excluded."""
    return [
        Step("input", "Input", ask=_ask_input, sets=("input", "recursive")),
        Step(
            "method",
            "Type and method",
            ask=_ask_method,
            sets=("type", "method", "searchable_pdfs", *ITEM_OPTIONS),
        ),
        Step(
            "output",
            "Output and formats",
            ask=_ask_output,
            sets=("output", "beside_input", "layout", "formats"),
        ),
        _model_step(session),
        _response_step(session),
        _images_step(session),
        Step(
            "ocr",
            "Tesseract options",
            ask=_ask_ocr,
            sets=OCR_OPTIONS,
            applies=lambda state: _method(state) == "tesseract",
        ),
        Step(
            "audio",
            "Audio options",
            ask=_ask_audio,
            sets=AUDIO_OPTIONS,
            applies=lambda state: _method(state) in _AUDIO_METHODS,
        ),
        Step("pages", "Pages", ask=_ask_pages, sets=("pages",)),
        _run_step(session),
    ]
