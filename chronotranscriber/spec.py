"""The run specifications and the option tables that build them.

``OPTIONS`` holds the transcribe options: the CLI parser, the settings
``defaults`` block and the equivalent command line derive from it. The
repair table reuses its schema, prompt, context, model and image options
under the same names, so a settings default of those options applies there
too. Image options left unset take the provider's recommended value
(:func:`recommended_images`).

A folder with several items needs ``--all``, ``--select`` or ``--files``;
the three are alternatives (see :func:`chronotranscriber.run.plan`).

``--type`` and ``--method`` must form a valid pair (:data:`METHODS_BY_TYPE`);
a type that cannot take ``llm`` (EPUB, MOBI, audio) takes its first method
unless ``--method`` names one as a flag; a settings default does not count.
A type that takes ``llm`` never swaps an invalid method for another. In auto
mode the method serves scanned PDFs and image folders; audio needs
``--type audio``.

A run that calls a model refuses a provider set at a higher precedence level
than a model of another provider. List options given more than once
accumulate (:class:`~chronotranscriber.common.options.Option`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

from chronotranscriber.common.image_presets import (
    DetailCheck,
    DetailRules,
    ImagePresets,
    anthropic_preset,
    custom_preset,
    google_preset,
    openai_preset,
)
from chronotranscriber.common.options import (
    Condition,
    Kind,
    Option,
    OptionTable,
    Resolved,
    Source,
)
from chronotranscriber.common.polling import DEFAULT_POLL_MINUTES, wait_options
from chronotranscriber.common.selection import parse_page_range
from chronotranscriber.common.standard_options import (
    DRY_RUN_HELP,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    JSON_HELP,
    REASONING_EFFORTS,
    RESPONSE_FORMATS,
    SERVICE_TIERS,
    VERBOSITIES,
    Dpi,
    abandon_option,
    adopt_option,
    batch_cancel_options,
    batch_ids_option,
    batch_option,
    check_input_option,
    concurrency_option,
    dry_run_option,
    flag_option,
    force_option,
    image_options,
    json_option,
    model_options,
    nonempty_text,
    output_options,
    positive_int,
    render_options,
    settings_option,
)
from chronotranscriber.providers import (
    PROVIDERS,
    api_model_id,
    implied_provider,
    infer_provider,
    split_provider_prefix,
)

__all__ = [
    "AUDIO_API_PROVIDERS",
    "BATCH_CANCEL_OPTIONS",
    "BATCH_CHECK_OPTIONS",
    "CONTEXT_AUTO",
    "CONTEXT_NONE",
    "DEFAULT_AUDIO_MODEL",
    "DEFAULT_FORMATS",
    "DEFAULT_GOOGLE_AUDIO_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_SCHEMA",
    "DEFAULT_WAIT_HOURS",
    "DROP_OPTIONS",
    "IMAGE_DETAILS",
    "IMAGE_FORMATS",
    "IMAGE_PRESETS",
    "LAYOUTS",
    "METHODS",
    "METHODS_BY_TYPE",
    "MODEL_SUGGESTIONS",
    "OCR_PRESETS",
    "OPTIONS",
    "OUTPUT_FORMATS",
    "PROVIDERS",
    "REASONING_EFFORTS",
    "RENDER_OPTIONS",
    "REPAIR_CAUSES",
    "REPAIR_OPTIONS",
    "RESPONSE_FORMATS",
    "SEARCHABLE_PDF_MODES",
    "SERVICE_TIERS",
    "SOURCE_TYPES",
    "VERBOSITIES",
    "AudioSpec",
    "BatchCancelSpec",
    "BatchCheckSpec",
    "CommandSpec",
    "ContextChoice",
    "Dpi",
    "DropSpec",
    "ImageChoice",
    "ImageSpec",
    "InputSpec",
    "MethodSpec",
    "ModelSpec",
    "OcrSpec",
    "OutputSpec",
    "PromptSpec",
    "RenderSpec",
    "RepairControl",
    "RepairSelection",
    "RepairSpec",
    "Resolution",
    "RunControl",
    "RunSpec",
    "SpecError",
    "effective_spec",
    "inapplicable",
    "model_suggestions",
    "parse",
    "parse_batch_cancel",
    "parse_batch_check",
    "parse_drop",
    "parse_render",
    "parse_repair",
    "recommended_images",
    "resolve_batch_cancel",
    "resolve_batch_check",
    "resolve_drop",
    "resolve_render",
    "resolve_repair",
    "resolve_spec",
    "to_argv",
]

SOURCE_TYPES = ("auto", "pdf", "images", "epub", "mobi", "audio")
METHODS = ("llm", "tesseract", "native", "whisper", "audio-api")
METHODS_BY_TYPE: Mapping[str, tuple[str, ...]] = {
    "auto": ("llm", "tesseract"),
    "pdf": ("llm", "tesseract", "native"),
    "images": ("llm", "tesseract"),
    "epub": ("native",),
    "mobi": ("native",),
    "audio": ("whisper", "audio-api"),
}
SEARCHABLE_PDF_MODES = ("ocr", "text")
LAYOUTS = ("hash", "mirror")
OUTPUT_FORMATS = ("txt", "md")
DEFAULT_FORMATS = ("txt",)
DEFAULT_WAIT_HOURS = 50.0
"""Hours a ``batch check --wait`` waits: the longest provider batch window
(Gemini, 48 hours) plus margin."""
OCR_PRESETS = ("default", "sauvola", "none")
AUDIO_API_PROVIDERS = ("openai", "google")
REPAIR_CAUSES = ("api_error", "not_possible", "no_text")

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_AUDIO_MODEL = "gpt-transcribe"
DEFAULT_GOOGLE_AUDIO_MODEL = "gemini-3.6-flash"
DEFAULT_SCHEMA = "markdown_transcription_schema"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_VERBOSITY = "medium"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_SERVICE_TIER = "default"
DEFAULT_OCR_CONFIG = "--oem 3 --psm 6"
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_AUDIO_CHUNK_SECONDS = 600
DEFAULT_CONCURRENCY = 16

CONTEXT_AUTO = "auto"
CONTEXT_NONE: Final = "none"
PROMPT_DEFAULT = "default"
LANGUAGE_AUTO = "auto"

MODEL_SUGGESTIONS: Mapping[str, tuple[str, ...]] = {
    "openai": (
        DEFAULT_MODEL,
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.5",
    ),
    "anthropic": ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
    "google": ("gemini-3.7-flash", "gemini-3.1-pro", "gemini-3.5-flash"),
    "openrouter": (
        "openai/gpt-5.6-luna",
        "anthropic/claude-sonnet-5-5",
        "google/gemini-3.1-pro",
    ),
}
AUDIO_MODEL_SUGGESTIONS: Mapping[str, tuple[str, ...]] = {
    "openai": (
        DEFAULT_AUDIO_MODEL,
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "whisper-1",
    ),
    "google": (DEFAULT_GOOGLE_AUDIO_MODEL,),
}
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")

type ContextChoice = str | None
"""Context text or a path to a text file, :data:`CONTEXT_NONE`, or None for
the automatic lookup."""

type ImageChoice = Path | Literal["none"] | None
"""A context image file, :data:`CONTEXT_NONE`, or None for the automatic
lookup."""


class SpecError(ValueError):
    """The options do not form a valid run specification."""


@dataclass(frozen=True)
class InputSpec:
    """The input path, its type and the selection of items and pages.

    ``files`` names items exactly, ``select`` picks them by number, range or
    name search, and ``all`` takes every item; a folder with several items
    needs one of them.
    """

    path: Path
    type: str
    files: tuple[str, ...]
    recursive: bool
    pages: str | None
    select: str | None = None
    all: bool = False


@dataclass(frozen=True)
class MethodSpec:
    """The transcription method and how auto mode treats searchable PDFs."""

    name: str
    searchable_pdfs: str


@dataclass(frozen=True)
class OutputSpec:
    """Where and what to write; ``directory`` None means beside the input.

    ``formats`` are the rendered text formats; the database is always written.
    """

    directory: Path | None
    layout: str
    formats: tuple[str, ...]


@dataclass(frozen=True)
class PromptSpec:
    """Schema, system prompt and context of LLM calls.

    ``prompt`` None takes the prompt of the response format. ``context`` and
    ``context_image`` None look up the item's sidecar files;
    :data:`CONTEXT_NONE` sends no context. ``context`` is otherwise context
    text or a path to a text file. ``fallback_context`` (text or a path;
    None for none) applies under the automatic lookup to an item without a
    sidecar.
    """

    schema: str
    prompt: Path | None
    context: ContextChoice
    context_image: ImageChoice
    fallback_context: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """The model of LLM and remote audio calls.

    ``provider`` is a provider name once resolved, and ``model`` carries no
    ``provider:`` prefix. ``endpoint`` names a custom endpoint from the
    settings. ``max_output_tokens`` None takes the model's output limit.
    ``response_format`` None means schema when the model supports it, else
    json.
    """

    provider: str
    model: str
    endpoint: str | None
    reasoning_effort: str
    verbosity: str
    max_output_tokens: int | None
    temperature: float
    top_p: float
    service_tier: str
    response_format: str | None


@dataclass(frozen=True)
class ImageSpec:
    """Page image settings; None takes the provider's recommended value."""

    dpi: Dpi | None = None
    detail: str | None = None
    format: str | None = None
    jpeg_quality: int | None = None
    grayscale: bool | None = None

    def over(self, base: ImageSpec) -> ImageSpec:
        """Return these values with unset fields taken from ``base``."""
        return ImageSpec(
            *(
                getattr(self, f.name)
                if getattr(self, f.name) is not None
                else getattr(base, f.name)
                for f in fields(self)
            )
        )


@dataclass(frozen=True)
class OcrSpec:
    """Tesseract configuration string and preprocessing preset."""

    config: str
    preprocess: str


@dataclass(frozen=True)
class AudioSpec:
    """Audio options; ``language`` None detects the language."""

    whisper_model: str
    language: str | None
    chunk_seconds: int


@dataclass(frozen=True)
class RunControl:
    """Batch, resume, concurrency, working-file and agent-contract switches.

    ``keep_working_files`` keeps Tesseract's preprocessed images and the
    audio chunks, which are deleted after each item otherwise.
    """

    batch: bool
    sync_fallback: bool
    retry_errors: bool
    force: bool
    concurrency: int
    keep_working_files: bool
    dry_run: bool
    json: bool


@dataclass(frozen=True)
class RunSpec:
    """Everything one transcribe run needs besides the machine settings."""

    input: InputSpec
    method: MethodSpec
    output: OutputSpec
    prompt: PromptSpec
    model: ModelSpec
    images: ImageSpec
    ocr: OcrSpec
    audio: AudioSpec
    run: RunControl


@dataclass(frozen=True)
class RepairSelection:
    """The failure causes to repair and an optional filter of page indices."""

    causes: tuple[str, ...]
    indices: tuple[int, ...] | None


@dataclass(frozen=True)
class RepairControl:
    """Batch and agent-contract switches of a repair."""

    batch: bool
    dry_run: bool
    json: bool


@dataclass(frozen=True)
class RepairSpec:
    """The options of one repair.

    *input* is a transcription file or its input, a folder whose databases
    at or below it are searched, or a database.
    """

    input: Path
    selection: RepairSelection
    prompt: PromptSpec
    model: ModelSpec
    images: ImageSpec
    run: RepairControl


@dataclass(frozen=True)
class BatchCheckSpec:
    """Check and finalize batch submissions found below ``directory``.

    ``formats`` None renders each document in the formats of the run that
    submitted it. ``adopt`` names a provider batch to record for an
    unresolved submission; ``abandon`` names an unresolved or submitted
    submission whose units return to pending. With ``wait`` the check repeats every
    ``poll_minutes`` until no batch work is pending or ``wait_hours`` have
    passed.
    """

    directory: Path
    batch_ids: tuple[str, ...]
    formats: tuple[str, ...] | None
    json: bool
    adopt: str | None = None
    abandon: str | None = None
    wait: bool = False
    poll_minutes: float = DEFAULT_POLL_MINUTES
    wait_hours: float = DEFAULT_WAIT_HOURS


@dataclass(frozen=True)
class BatchCancelSpec:
    """Cancel batches.

    The cancel covers the named batches, every unfinished batch with ``all``,
    or the open submissions of the databases below ``directory``.
    """

    batch_ids: tuple[str, ...]
    all: bool
    json: bool
    directory: Path | None = None


@dataclass(frozen=True)
class RenderSpec:
    """The options of ``render``; ``formats`` None renders the registered ones."""

    input: Path
    formats: tuple[str, ...] | None = None
    force: bool = False
    dry_run: bool = False
    json: bool = False


@dataclass(frozen=True)
class DropSpec:
    """The options of ``drop``.

    They name the store and exactly one selection of its documents
    (``select``, ``files`` or ``all``).
    """

    input: Path
    select: str | None = None
    files: tuple[str, ...] = ()
    all: bool = False
    dry_run: bool = False
    json: bool = False


type CommandSpec = (
    RunSpec | RepairSpec | BatchCheckSpec | BatchCancelSpec | RenderSpec | DropSpec
)


_DETAIL_RULES = DetailRules(
    checks={
        "openai": DetailCheck(
            kept=frozenset({"low", "high"}), original=True, fallback=None
        ),
        "openrouter": DetailCheck(
            kept=frozenset({"low", "high"}),
            fallback=None,
            needs_detail_support=True,
        ),
    },
)

IMAGE_PRESETS = ImagePresets(
    families={
        "openai": openai_preset(),
        "anthropic": anthropic_preset(),
        "google": google_preset(),
        "custom": custom_preset(),
    },
    detail=_DETAIL_RULES,
)
"""The image presets of each model family: the values the wizard recommends
and the request sections of ``sources.images``. Every family sends colour
JPEG pages at quality 95; DPI and detail differ per family. OpenAI keeps
low, high and (where the model takes it) original, and OpenRouter models
that take an image detail keep low and high; any other detail becomes the
model's default detail."""

_RECOMMENDED_IMAGES: Mapping[str, ImageSpec] = {
    family: ImageSpec(**asdict(preset.recommended))
    for family, preset in IMAGE_PRESETS.families.items()
}


def recommended_images(family: str) -> ImageSpec:
    """Return the recommended image settings of a model family.

    ``family`` is openai, anthropic, google or custom; OpenRouter models use
    the family of the underlying model, unknown families the custom values.
    """
    return _RECOMMENDED_IMAGES.get(family, _RECOMMENDED_IMAGES["custom"])


def model_suggestions(
    provider: str | None = None, *, audio: bool = False
) -> tuple[str, ...]:
    """Return suggested model names of *provider*, or of every provider.

    ``audio`` lists the models of the audio-api method. Suggestions are not
    enforced; a custom endpoint has none.
    """
    table = AUDIO_MODEL_SUGGESTIONS if audio else MODEL_SUGGESTIONS
    if provider is None:
        names = (name for names in table.values() for name in names)
        return tuple(dict.fromkeys(names))
    return table.get(provider, ())


def _pages(value: str) -> str:
    parse_page_range(value)
    return value.strip()


def _context(value: str) -> ContextChoice:
    keyword = value.strip().lower()
    if keyword == CONTEXT_AUTO:
        return None
    if keyword == CONTEXT_NONE:
        return CONTEXT_NONE
    return nonempty_text(value)


def _fallback_context(value: str) -> str | None:
    if value.strip().lower() == CONTEXT_NONE:
        return None
    return nonempty_text(value)


def _context_image(value: str) -> ImageChoice:
    keyword = value.strip().lower()
    if keyword == CONTEXT_AUTO:
        return None
    if keyword == CONTEXT_NONE:
        return CONTEXT_NONE
    return Path(nonempty_text(value))


def _prompt(value: str) -> Path | None:
    if value.strip().lower() == PROMPT_DEFAULT:
        return None
    return Path(nonempty_text(value))


def _language(value: str) -> str | None:
    if value.strip().lower() == LANGUAGE_AUTO:
        return None
    return nonempty_text(value).strip()


def _parse_indices(value: str) -> tuple[int, ...]:
    """Parse 0-based indices such as ``0,5,12`` or ``3-7``."""
    indices: set[int] = set()
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        if part.startswith("-"):
            raise ValueError(f"invalid index {part!r}: indices are not negative")
        start, separator, end = part.partition("-")
        try:
            low = int(start)
            high = int(end) if separator else low
        except ValueError:
            raise ValueError(f"invalid index or range {part!r}") from None
        if low > high:
            raise ValueError(f"range start exceeds its end in {part!r}")
        indices.update(range(low, high + 1))
    if not indices:
        raise ValueError("no indices given")
    return tuple(sorted(indices))


def _format_indices(indices: Iterable[int]) -> str:
    """Return sorted indices with consecutive runs written as ranges."""
    runs: list[list[int]] = []
    for index in sorted(set(indices)):
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _indices(value: str) -> str:
    return _format_indices(_parse_indices(value))


_LLM = Condition("method", frozenset({"llm"}))
_TESSERACT = Condition("method", frozenset({"tesseract"}))
_WHISPER = Condition("method", frozenset({"whisper"}))
_AUDIO = Condition("method", frozenset({"whisper", "audio-api"}))
_TYPED = Condition("type", frozenset(SOURCE_TYPES) - {"auto"})


def _prompt_options() -> list[Option]:
    return [
        Option(
            "schema",
            "--schema",
            "transcription schema: a name found in schemas/ or a path",
            group="prompt",
            type=nonempty_text,
            metavar="NAME|PATH",
            default=DEFAULT_SCHEMA,
            target="prompt.schema",
        ),
        Option(
            "prompt",
            "--prompt",
            "system prompt file, or default (the prompt of the response format)",
            group="prompt",
            type=_prompt,
            metavar="PATH|default",
            default_help=PROMPT_DEFAULT,
            target="prompt.prompt",
        ),
        Option(
            "context",
            "--context",
            (
                "transcription context: topics as text or a path to a text file "
                "(overrides sidecars), auto (the item's sidecar) or none (no "
                "context)"
            ),
            group="prompt",
            type=_context,
            metavar="TEXT|PATH|auto|none",
            default_help=CONTEXT_AUTO,
            target="prompt.context",
        ),
        Option(
            "fallback_context",
            "--fallback-context",
            (
                "transcription context for items without a sidecar under "
                "--context auto: topics as text, a path to a text file, or none"
            ),
            group="prompt",
            type=_fallback_context,
            metavar="TEXT|PATH|none",
            default_help=CONTEXT_NONE,
            target="prompt.fallback_context",
        ),
        Option(
            "context_image",
            "--context-image",
            (
                "image sent with every page: an image file, auto (the "
                "item's sidecar image) or none"
            ),
            group="prompt",
            type=_context_image,
            metavar="PATH|auto|none",
            default_help=CONTEXT_AUTO,
            target="prompt.context_image",
        ),
    ]


def _model_options() -> list[Option]:
    return model_options(
        providers=PROVIDERS,
        suggestions=model_suggestions,
        defaults={
            "model": DEFAULT_MODEL,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "verbosity": DEFAULT_VERBOSITY,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "service_tier": DEFAULT_SERVICE_TIER,
        },
        response_formats=RESPONSE_FORMATS,
        default_help={
            "model": (
                f"{DEFAULT_MODEL}; {DEFAULT_AUDIO_MODEL} for --method audio-api, "
                f"{DEFAULT_GOOGLE_AUDIO_MODEL} with --provider google"
            ),
        },
    )


def _engine_options() -> list[Option]:
    return [
        Option(
            "ocr_config",
            "--ocr-config",
            "Tesseract configuration string",
            group="ocr",
            type=nonempty_text,
            metavar="STR",
            default=DEFAULT_OCR_CONFIG,
            condition=_TESSERACT,
            target="ocr.config",
        ),
        Option(
            "ocr_preprocess",
            "--ocr-preprocess",
            "Tesseract image preprocessing preset",
            group="ocr",
            choices=OCR_PRESETS,
            default="default",
            condition=_TESSERACT,
            target="ocr.preprocess",
        ),
        Option(
            "whisper_model",
            "--whisper-model",
            "faster-whisper model size",
            group="audio",
            type=nonempty_text,
            metavar="SIZE",
            choice_provider=lambda: WHISPER_MODELS,
            default=DEFAULT_WHISPER_MODEL,
            condition=_WHISPER,
            target="audio.whisper_model",
        ),
        Option(
            "language",
            "--language",
            "spoken language as an ISO 639-1 code, or auto",
            group="audio",
            type=_language,
            metavar="CODE|auto",
            default_help=LANGUAGE_AUTO,
            condition=_AUDIO,
            target="audio.language",
        ),
        Option(
            "audio_chunk_seconds",
            "--audio-chunk-seconds",
            "length of the audio chunks sent per request, in seconds",
            group="audio",
            type=positive_int,
            metavar="N",
            default=DEFAULT_AUDIO_CHUNK_SECONDS,
            condition=_AUDIO,
            target="audio.chunk_seconds",
        ),
    ]


_GROUP_TITLES = {
    "input": "input",
    "method": "method",
    "output": "output",
    "prompt": "schema, prompt and context",
    "model": "model",
    "images": "images",
    "ocr": "Tesseract",
    "audio": "audio",
    "selection": "failures to repair",
    "run": "run control",
}


def _input_options() -> list[Option]:
    return [
        Option(
            "input",
            "--input",
            "PDF, EPUB, MOBI or audio file, image folder, or a folder of these",
            group="input",
            type=Path,
            metavar="PATH",
            target="input.path",
            settable=False,
        ),
        Option(
            "type",
            "--type",
            (
                "input type; auto takes the PDFs, image folders, EPUB and MOBI "
                "files of the input; audio needs --type audio"
            ),
            group="input",
            choices=SOURCE_TYPES,
            default="auto",
            target="input.type",
        ),
        Option(
            "files",
            "--files",
            "comma-separated names of files or folders inside the input folder",
            group="input",
            kind=Kind.LIST,
            type=nonempty_text,
            metavar="LIST",
            default=(),
            condition=_TYPED,
            exclusive="selection",
            target="input.files",
            settable=False,
        ),
        Option(
            "select",
            "--select",
            "select items by number (1,3), range (1-5) or name search",
            group="input",
            type=nonempty_text,
            metavar="EXPR",
            exclusive="selection",
            target="input.select",
        ),
        Option(
            "all",
            "--all",
            "process every discovered item",
            group="input",
            kind=Kind.FLAG,
            default=False,
            exclusive="selection",
            target="input.all",
        ),
        Option(
            "recursive",
            "--recursive",
            "search the subfolders of the input folder",
            group="input",
            kind=Kind.SWITCH,
            default=False,
            condition=_TYPED,
            target="input.recursive",
        ),
        Option(
            "pages",
            "--pages",
            (
                "pages, images or audio chunks: 5, first:5, last:5, 3-7, 3-, "
                "-7 or 1,3,5-8"
            ),
            group="input",
            type=_pages,
            metavar="RANGE",
            default_help="all",
            target="input.pages",
            settable=False,
        ),
        Option(
            "method",
            "--method",
            (
                "transcription method; in auto mode it serves scanned PDFs and "
                "image folders; EPUB and MOBI take native and audio takes "
                "whisper unless --method names another"
            ),
            group="method",
            choices=METHODS,
            default="llm",
            target="method.name",
        ),
        Option(
            "searchable_pdfs",
            "--searchable-pdfs",
            (
                "PDFs with a text layer in auto mode: ocr transcribes them "
                "with the method, text takes the text layer"
            ),
            group="method",
            choices=SEARCHABLE_PDF_MODES,
            default="ocr",
            condition=Condition("type", frozenset({"auto"})),
            target="method.searchable_pdfs",
        ),
    ]


def _output_options() -> list[Option]:
    return [
        *output_options(),
        Option(
            "layout",
            "--layout",
            (
                "output folder layout: hash (one hash-suffixed folder per "
                "item) or mirror (the input folder tree)"
            ),
            group="output",
            choices=LAYOUTS,
            default="hash",
            target="output.layout",
        ),
        Option(
            "formats",
            "--formats",
            "comma-separated output formats",
            group="output",
            kind=Kind.LIST,
            choices=OUTPUT_FORMATS,
            default=DEFAULT_FORMATS,
            metavar="LIST",
            target="output.formats",
        ),
    ]


def _run_options() -> list[Option]:
    return [
        batch_option("submit the pages to the provider's batch API", condition=_LLM),
        Option(
            "sync_fallback",
            "--sync-fallback",
            "transcribe synchronously when a batch submission fails",
            group="run",
            kind=Kind.SWITCH,
            default=False,
            condition=_LLM,
            target="run.sync_fallback",
        ),
        Option(
            "retry_errors",
            "--retry-errors",
            (
                "on resume, transcribe pages again whose earlier result is an "
                "error; --no-retry-errors keeps those results"
            ),
            group="run",
            kind=Kind.SWITCH,
            default=True,
            target="run.retry_errors",
        ),
        force_option(
            "transcribe items again from the start: every stored unit of the "
            "item is reset, also outside --pages; stored responses are kept"
        ),
        concurrency_option(DEFAULT_CONCURRENCY),
        Option(
            "keep_working_files",
            "--keep-working-files",
            (
                "keep Tesseract's preprocessed page images and the audio chunks "
                "in the item's working folder"
            ),
            group="run",
            kind=Kind.SWITCH,
            default=False,
            target="run.keep_working_files",
        ),
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ]


OPTIONS = OptionTable(
    [
        *_input_options(),
        *_output_options(),
        *_prompt_options(),
        *_model_options(),
        *image_options(),
        *_engine_options(),
        *_run_options(),
    ],
    group_titles=_GROUP_TITLES,
)

_REPAIR_SHARED = tuple(
    option.name for option in (*_prompt_options(), *_model_options(), *image_options())
)


def _repair_flag(name: str, help: str) -> Option:
    return flag_option(name, help, group="selection")


def _repair_shared(name: str) -> Option:
    """Return the transcribe option *name* for the repair table."""
    if name == "model":
        return replace(OPTIONS[name], default_help=DEFAULT_MODEL)
    return OPTIONS[name]


REPAIR_OPTIONS = OptionTable(
    [
        Option(
            "input",
            "--input",
            (
                "a transcription file or the input it was transcribed from, a "
                "chronotranscriber.sqlite, or a folder whose databases at or "
                "below it are repaired"
            ),
            group="input",
            type=Path,
            metavar="PATH",
            default=Path("."),
            default_help="the current directory",
            target="input",
            settable=False,
        ),
        _repair_flag(
            "errors_only",
            "repair transcription errors; the default when no other cause is given",
        ),
        _repair_flag("not_possible", "repair pages marked transcription not possible"),
        _repair_flag("no_text", "repair pages marked no transcribable text"),
        _repair_flag("all_failures", "repair every failure cause"),
        Option(
            "indices",
            "--indices",
            "0-based page indices (order in the item) among the selected "
            "failures of one document: 0,5,12 or 3-7",
            group="selection",
            type=_indices,
            metavar="LIST",
            default_help="every selected failure",
            settable=False,
        ),
        *(_repair_shared(name) for name in _REPAIR_SHARED),
        batch_option("submit the repairs to the provider's batch API"),
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ],
    group_titles=_GROUP_TITLES,
)


BATCH_CHECK_OPTIONS = OptionTable(
    [
        check_input_option(),
        batch_ids_option(
            "comma-separated batch IDs to check", default_help="every batch"
        ),
        adopt_option(),
        abandon_option(
            "give up an unresolved or submitted submission: its pages return to "
            "pending and its uploaded files are deleted; a submitted batch is "
            "not cancelled"
        ),
        replace(
            OPTIONS["formats"],
            default=None,
            default_help="the formats of the run that submitted the batch",
            target="formats",
            settable=False,
        ),
        *wait_options(group="batch", default_hours=DEFAULT_WAIT_HOURS),
        json_option("json"),
        settings_option(),
    ],
    group_titles={**_GROUP_TITLES, "batch": "batches"},
)

BATCH_CANCEL_OPTIONS = OptionTable(
    batch_cancel_options(),
    group_titles={**_GROUP_TITLES, "batch": "batches"},
)

RENDER_OPTIONS = OptionTable(
    render_options(
        input_help="folder or chronotranscriber.sqlite to render",
        formats=OUTPUT_FORMATS,
        dry_run_help=DRY_RUN_HELP,
        json_help=JSON_HELP,
    ),
    group_titles=_GROUP_TITLES,
)

DROP_OPTIONS = OptionTable(
    [
        Option(
            "input",
            "--input",
            "output folder holding chronotranscriber.sqlite, or the database itself",
            group="input",
            type=Path,
            metavar="PATH",
            target="input",
            settable=False,
        ),
        Option(
            "select",
            "--select",
            "select stored documents by number (1,3), range (1-5) or name search",
            group="documents",
            type=nonempty_text,
            metavar="EXPR",
            exclusive="selection",
            target="select",
            settable=False,
        ),
        Option(
            "files",
            "--files",
            "comma-separated names of stored documents or of their input files",
            group="documents",
            kind=Kind.LIST,
            type=nonempty_text,
            metavar="LIST",
            default=(),
            exclusive="selection",
            target="files",
            settable=False,
        ),
        flag_option(
            "all",
            "drop every document of the store",
            group="documents",
            target="all",
            exclusive="selection",
        ),
        dry_run_option("dry_run"),
        json_option("json"),
    ],
    group_titles={**_GROUP_TITLES, "documents": "documents to drop"},
)


def _parse_table(table: OptionTable, argv: Sequence[str]) -> dict[str, Any]:
    """Parse *argv* with *table*; argparse exits with status 2 on errors."""
    return table.parse(argv)


@dataclass(frozen=True)
class Resolution[S]:
    """The effective spec and the source of each field (dotted path)."""

    spec: S
    sources: Mapping[str, Source]


def effective_spec(
    spec: CommandSpec, sources: Mapping[str, Source] | None = None
) -> dict[str, dict[str, Any]]:
    """Return every field of *spec* (dotted path) with its value and source.

    Values are JSON values; fields missing from *sources* were given by the
    caller and count as flags.
    """
    known = sources or {}
    return {
        path: {
            "value": _jsonable(value),
            "source": known.get(path, Source.FLAG).value,
        }
        for path, value in _flatten(spec).items()
    }


def inapplicable(spec: RunSpec, given: Iterable[str]) -> list[str]:
    """Return the *given* option names whose condition fails for *spec*.

    Such options do not affect the run; a front end reports them.
    """
    flat = _flatten(spec)
    values = {
        option.name: flat[option.target]
        for option in OPTIONS
        if option.target is not None
    }
    wanted = set(given)
    return [name for name in OPTIONS.inapplicable(values) if name in wanted]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if is_dataclass(value):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def _part(values: Mapping[str, Any], name: str) -> dict[str, Any]:
    prefix = name + "."
    return {
        key.removeprefix(prefix): value
        for key, value in values.items()
        if key.startswith(prefix)
    }


def _applicable(
    table: OptionTable, defaults: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return the settings defaults that set settable options of *table*."""
    return {
        name: value
        for name, value in (defaults or {}).items()
        if name in table and table[name].settable
    }


def _resolve_table(
    table: OptionTable, flags: Mapping[str, Any], defaults: Mapping[str, Any] | None
) -> dict[str, Resolved]:
    unknown = set(flags) - set(table.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    return table.resolve(flags, _applicable(table, defaults))


def _settle_method(resolved: dict[str, Resolved]) -> None:
    """Check the type and method pair.

    A type that cannot take ``llm`` takes its first method unless a flag
    names the method.
    """
    kind = resolved["type"]
    method = resolved["method"]
    allowed = METHODS_BY_TYPE[kind.value]
    if method.value in allowed:
        return
    if "llm" not in allowed and method.source is not Source.FLAG:
        resolved["method"] = Resolved(allowed[0], kind.source)
        return
    raise SpecError(
        f"--type {kind.value} takes --method {' or '.join(allowed)}, not {method.value}"
    )


def _settle_endpoint(resolved: dict[str, Resolved]) -> None:
    """Make a named endpoint imply provider custom, by precedence."""
    provider = resolved["provider"]
    endpoint = resolved["endpoint"]
    if endpoint.value is not None and provider.value != "custom":
        if endpoint.source.rank >= provider.source.rank:
            resolved["provider"] = Resolved("custom", endpoint.source)
        else:
            resolved["endpoint"] = Resolved(None, provider.source)


_MODEL_METHODS = frozenset({"llm", "audio-api"})
"""The methods that call a model; the others ignore the provider."""

_LEVELS: Mapping[Source, str] = {
    Source.FLAG: "a flag",
    Source.SETTINGS: "the settings defaults",
    Source.DEFAULT: "the code default",
}


def _check_model_level(resolved: Mapping[str, Resolved]) -> None:
    """Refuse a provider set above a model that is not one of its models.

    A custom provider needs a model at its endpoint's level or above; any
    other provider refuses a lower-level model that implies another provider.
    """
    provider = resolved["provider"]
    model = resolved["model"]
    if provider.value == "custom":
        endpoint = resolved["endpoint"]
        if model.source.rank < endpoint.source.rank:
            raise SpecError(
                f"--endpoint {endpoint.value} (from {_LEVELS[endpoint.source]}) "
                f"needs a model of that endpoint, but the model {model.value!r} "
                f"comes from {_LEVELS[model.source]}; name it with --model NAME"
            )
        return
    if provider.source.rank <= model.source.rank:
        return
    implied = implied_provider(model.value)
    if implied is not None and implied != provider.value:
        raise SpecError(
            f"--provider {provider.value} (from {_LEVELS[provider.source]}) "
            f"cannot run the model {model.value!r} of provider {implied} "
            f"(from {_LEVELS[model.source]}); name a model of provider "
            f"{provider.value} with --model NAME"
        )


def _settle_provider(resolved: dict[str, Resolved]) -> None:
    """Settle the provider from the model name and strip a provider prefix.

    An unset provider is inferred and takes the model's source. A provider
    set below the model's level yields to the provider the name implies; a
    prefix contradicting a provider set at the model's level or above, or a
    custom provider, is an error. The model name then becomes the id the
    provider's API knows (:func:`~chronotranscriber.providers.api_model_id`).
    """
    _settle_endpoint(resolved)
    provider = resolved["provider"]
    model = resolved["model"]
    prefixed, name = split_provider_prefix(model.value)
    if prefixed is not None:
        resolved["model"] = Resolved(name, model.source)
    if provider.value is None:
        inferred = prefixed or infer_provider(name)
        resolved["provider"] = Resolved(inferred, model.source)
    elif prefixed is not None and prefixed != provider.value:
        if provider.value == "custom" or provider.source.rank >= model.source.rank:
            raise SpecError(
                f"model {model.value!r} names provider {prefixed}, "
                f"but the provider is {provider.value}"
            )
        resolved["provider"] = Resolved(prefixed, model.source)
    elif provider.value != "custom" and provider.source.rank < model.source.rank:
        implied = implied_provider(name)
        if implied is not None and implied != provider.value:
            resolved["provider"] = Resolved(implied, model.source)
    if resolved["provider"].value == "custom" and resolved["endpoint"].value is None:
        raise SpecError(
            "provider custom needs --endpoint NAME naming an endpoint from the settings"
        )
    model = resolved["model"]
    api_id = api_model_id(resolved["provider"].value, model.value)
    if api_id != model.value:
        resolved["model"] = Resolved(api_id, model.source)


def _model_spec(values: Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(**_part(values, "model"))


def _prompt_spec(values: Mapping[str, Any]) -> PromptSpec:
    return PromptSpec(**_part(values, "prompt"))


def resolve_spec(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[RunSpec]:
    """Resolve transcribe options: flag, then settings default, then code default.

    ``flags`` are converted values keyed by option name, as returned by
    ``OPTIONS.parse``; ``defaults`` is a checked settings defaults block. The
    audio-api method takes :data:`DEFAULT_AUDIO_MODEL` when no flag or
    settings default names a model, and :data:`DEFAULT_GOOGLE_AUDIO_MODEL`
    when provider google is set above the model's level.

    Raises:
        SpecError: No input path, no output format, a type and method that
            do not fit, a custom provider without an endpoint, a model prefix
            that contradicts the provider, a provider set above a model of
            another provider in a run that calls a model, or an audio-api
            provider other than openai or google.
    """
    resolved = _resolve_table(OPTIONS, flags, defaults)
    _settle_method(resolved)
    method = resolved["method"].value
    if method == "audio-api":
        model, provider = resolved["model"], resolved["provider"]
        if provider.value == "google" and provider.source.rank > model.source.rank:
            resolved["model"] = Resolved(DEFAULT_GOOGLE_AUDIO_MODEL, Source.DEFAULT)
        elif model.source is Source.DEFAULT:
            resolved["model"] = Resolved(DEFAULT_AUDIO_MODEL, Source.DEFAULT)
    _settle_provider(resolved)
    if method == "audio-api" and resolved["provider"].value not in AUDIO_API_PROVIDERS:
        raise SpecError(
            f"--method audio-api runs on provider {' or '.join(AUDIO_API_PROVIDERS)}"
            f", not {resolved['provider'].value}"
        )
    if method in _MODEL_METHODS:
        _check_model_level(resolved)
    values, sources = OPTIONS.by_targets(resolved)
    if values["input.path"] is None:
        raise SpecError("an input path is required (--input PATH)")
    if not values["output.formats"]:
        raise SpecError("--formats needs at least one format")
    spec = RunSpec(
        input=InputSpec(**_part(values, "input")),
        method=MethodSpec(**_part(values, "method")),
        output=OutputSpec(**_part(values, "output")),
        prompt=_prompt_spec(values),
        model=_model_spec(values),
        images=ImageSpec(**_part(values, "images")),
        ocr=OcrSpec(**_part(values, "ocr")),
        audio=AudioSpec(**_part(values, "audio")),
        run=RunControl(**_part(values, "run")),
    )
    return Resolution(spec, sources)


def _repair_causes(flags: Mapping[str, Any]) -> tuple[str, ...]:
    if flags.get("all_failures"):
        return REPAIR_CAUSES
    causes = {name for name in ("not_possible", "no_text") if flags.get(name)}
    if flags.get("errors_only") or not causes:
        causes.add("api_error")
    return tuple(cause for cause in REPAIR_CAUSES if cause in causes)


def resolve_repair(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[RepairSpec]:
    """Resolve repair options; settings defaults apply to the shared options.

    Without a cause flag, transcription errors are repaired. ``--indices``
    narrows the selected failures to the given page indices.

    Raises:
        SpecError: A custom provider without an endpoint, a model prefix
            that contradicts the provider, or a provider set above a model
            of another provider.
    """
    resolved = _resolve_table(REPAIR_OPTIONS, flags, defaults)
    _settle_provider(resolved)
    _check_model_level(resolved)
    values, sources = REPAIR_OPTIONS.by_targets(resolved)
    cause_flags = ("errors_only", "not_possible", "no_text", "all_failures")
    indices = flags.get("indices")
    selection = RepairSelection(
        causes=_repair_causes(flags),
        indices=None if indices is None else _parse_indices(indices),
    )
    sources = {
        **sources,
        "selection.causes": _given_source(flags, cause_flags),
        "selection.indices": _given_source(flags, ("indices",)),
    }
    spec = RepairSpec(
        input=values["input"],
        selection=selection,
        prompt=_prompt_spec(values),
        model=_model_spec(values),
        images=ImageSpec(**_part(values, "images")),
        run=RepairControl(**_part(values, "run")),
    )
    return Resolution(spec, sources)


def _given_source(flags: Mapping[str, Any], names: Sequence[str]) -> Source:
    return Source.FLAG if any(name in flags for name in names) else Source.DEFAULT


def resolve_batch_check(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution[BatchCheckSpec]:
    """Resolve batch check options; no option takes a settings default.

    Raises:
        SpecError: No output format, or ``--poll-minutes`` or
            ``--wait-hours`` without ``--wait``.
    """
    resolved = _resolve_table(BATCH_CHECK_OPTIONS, flags, defaults)
    values, sources = BATCH_CHECK_OPTIONS.by_targets(resolved)
    if values["formats"] == ():
        raise SpecError("--formats needs at least one format")
    if not values["wait"]:
        for name in ("poll_minutes", "wait_hours"):
            if name in flags:
                flag = BATCH_CHECK_OPTIONS[name].flag
                raise SpecError(f"{flag} needs --wait")
    return Resolution(BatchCheckSpec(**values), sources)


def resolve_batch_cancel(flags: Mapping[str, Any]) -> Resolution[BatchCancelSpec]:
    """Resolve batch cancel options.

    Raises:
        SpecError: None of batch IDs, ``--all`` and ``--input`` is given.
    """
    resolved = _resolve_table(BATCH_CANCEL_OPTIONS, flags, None)
    values, sources = BATCH_CANCEL_OPTIONS.by_targets(resolved)
    if not values["batch_ids"] and not values["all"] and values["directory"] is None:
        raise SpecError("batch cancel needs --batch-ids IDS or --all, or --input DIR")
    return Resolution(BatchCancelSpec(**values), sources)


def resolve_render(flags: Mapping[str, Any]) -> Resolution[RenderSpec]:
    """Resolve render options; no option takes a settings default.

    Raises:
        SpecError: An empty ``--formats``.
    """
    resolved = _resolve_table(RENDER_OPTIONS, flags, None)
    values, sources = RENDER_OPTIONS.by_targets(resolved)
    if values["formats"] == ():
        raise SpecError("--formats needs at least one format")
    return Resolution(RenderSpec(**values), sources)


def resolve_drop(flags: Mapping[str, Any]) -> Resolution[DropSpec]:
    """Resolve drop options; no option takes a settings default.

    Raises:
        SpecError: No input path, or not exactly one of ``--select``,
            ``--files`` and ``--all``.
    """
    resolved = _resolve_table(DROP_OPTIONS, flags, None)
    values, sources = DROP_OPTIONS.by_targets(resolved)
    if values["input"] is None:
        raise SpecError("an input path is required (--input PATH)")
    chosen = (values["select"] is not None, bool(values["files"]), values["all"])
    if sum(chosen) != 1:
        raise SpecError("give exactly one of --select, --files and --all")
    return Resolution(DropSpec(**values), sources)


def parse(argv: Sequence[str], defaults: Mapping[str, Any] | None = None) -> RunSpec:
    """Parse transcribe arguments; argparse exits with status 2 on errors."""
    return resolve_spec(_parse_table(OPTIONS, argv), defaults).spec


def parse_repair(
    argv: Sequence[str], defaults: Mapping[str, Any] | None = None
) -> RepairSpec:
    """Parse repair arguments; argparse exits with status 2 on errors."""
    return resolve_repair(_parse_table(REPAIR_OPTIONS, argv), defaults).spec


def parse_batch_check(
    argv: Sequence[str], defaults: Mapping[str, Any] | None = None
) -> BatchCheckSpec:
    """Parse batch check arguments; argparse exits with status 2 on errors."""
    return resolve_batch_check(_parse_table(BATCH_CHECK_OPTIONS, argv), defaults).spec


def parse_batch_cancel(argv: Sequence[str]) -> BatchCancelSpec:
    """Parse batch cancel arguments; argparse exits with status 2 on errors."""
    return resolve_batch_cancel(_parse_table(BATCH_CANCEL_OPTIONS, argv)).spec


def parse_render(argv: Sequence[str]) -> RenderSpec:
    """Parse render arguments; argparse exits with status 2 on errors."""
    return resolve_render(_parse_table(RENDER_OPTIONS, argv)).spec


def parse_drop(argv: Sequence[str]) -> DropSpec:
    """Parse drop arguments; argparse exits with status 2 on errors."""
    return resolve_drop(_parse_table(DROP_OPTIONS, argv)).spec


_UNSET = object()

# Unset values with a spelling; other options cannot be reset to None.
_UNSET_SPELLINGS = {
    "context": CONTEXT_AUTO,
    "fallback_context": CONTEXT_NONE,
    "context_image": CONTEXT_AUTO,
    "prompt": PROMPT_DEFAULT,
    "language": LANGUAGE_AUTO,
}


def _spell(given: dict[str, Any], option: Option, value: Any) -> None:
    """Add *option* with *value* to *given*, or raise when it has no spelling."""
    if option.name == "output" and value is None:
        given["beside_input"] = True
    elif option.name in _UNSET_SPELLINGS or (
        value is not None and (option.kind is not Kind.FLAG or value)
    ):
        given[option.name] = value
    else:
        raise SpecError(
            f"the settings default of {option.name} cannot be undone on the "
            f"command line: {option.flag} has no spelling for {value!r}"
        )


def _alternative_given(
    table: OptionTable, option: Option, given: Mapping[str, Any]
) -> bool:
    """Whether an alternative of *option* is given, which resets *option*."""
    return option.exclusive is not None and any(
        other.exclusive == option.exclusive and other.name in given for other in table
    )


def _spell_values(
    table: OptionTable,
    flat: Mapping[str, Any],
    defaults: Mapping[str, Any],
    code_default: Callable[[Option], Any] = lambda option: option.default,
) -> dict[str, Any]:
    """Spell every value that differs from its code or its settings default.

    A given input is always spelled; the provider is left to
    :func:`_spell_provider`.
    """
    targeted = [
        option
        for option in table
        if option.target is not None and option.name != "provider"
    ]
    given: dict[str, Any] = {}
    for option in targeted:
        value = flat[option.target or ""]
        given_input = option.name == "input" and value is not None
        if given_input or value != code_default(option):
            _spell(given, option, value)
    for option in targeted:
        if (
            option.name in given
            or option.name == "endpoint"
            or _alternative_given(table, option, given)
        ):
            continue
        value = flat[option.target or ""]
        setting = defaults.get(option.name, _UNSET)
        if setting is not _UNSET and setting != value:
            _spell(given, option, value)
    return given


def _spell_provider(
    given: dict[str, Any],
    model: ModelSpec,
    defaults: Mapping[str, Any],
    resolve: Callable[[Mapping[str, Any]], ModelSpec],
    *,
    calls_model: bool,
) -> None:
    """Spell the provider unless the model name and the settings yield it.

    When the run *calls_model* and the spelled provider would outrank an
    unspelled model, the model is spelled too.
    """
    if model.provider == infer_provider(model.model):
        if not defaults:
            return
        if _yields(given, model, resolve):
            return
    given["provider"] = model.provider
    if calls_model and "model" not in given and not _yields(given, model, resolve):
        given["model"] = model.model


def _yields(
    given: Mapping[str, Any],
    model: ModelSpec,
    resolve: Callable[[Mapping[str, Any]], ModelSpec],
) -> bool:
    """Whether *given* resolves to the provider, endpoint and model of *model*."""
    try:
        resolved = resolve(given)
    except SpecError:
        return False
    return (resolved.provider, resolved.endpoint, resolved.model) == (
        model.provider,
        model.endpoint,
        model.model,
    )


def _with_keywords(given: dict[str, Any]) -> dict[str, Any]:
    for name, keyword in _UNSET_SPELLINGS.items():
        if name in given and given[name] is None:
            given[name] = keyword
    return given


def _cause_flags(causes: Sequence[str]) -> dict[str, bool]:
    wanted = set(causes)
    if wanted == set(REPAIR_CAUSES):
        return {"all_failures": True}
    given: dict[str, bool] = {
        name: True for name in ("not_possible", "no_text") if name in wanted
    }
    if "api_error" in wanted and given:
        given["errors_only"] = True
    return given


def to_argv(spec: CommandSpec, defaults: Mapping[str, Any] | None = None) -> list[str]:
    """Return arguments that reproduce ``spec`` with and without ``defaults``.

    ``defaults`` is a checked settings defaults block. Every value that
    differs from its code default is spelled out, and so is every value the
    defaults would change; a provider the model name implies is omitted
    unless the defaults would change it. A value that begins with ``-`` is
    written as ``--flag=value``.

    Raises:
        SpecError: The defaults change a value that has no spelling, such
            as an unset ``--top-p``.
    """
    if isinstance(spec, RunSpec):
        return _transcribe_argv(spec, _applicable(OPTIONS, defaults))
    if isinstance(spec, RepairSpec):
        return _repair_argv(spec, _applicable(REPAIR_OPTIONS, defaults))
    table = _TABLES[type(spec)]
    given = _spell_values(table, _flatten(spec), _applicable(table, defaults))
    return _argv(table, given)


_TABLES: Mapping[type, OptionTable] = {
    BatchCheckSpec: BATCH_CHECK_OPTIONS,
    BatchCancelSpec: BATCH_CANCEL_OPTIONS,
    RenderSpec: RENDER_OPTIONS,
    DropSpec: DROP_OPTIONS,
}


def _argv(table: OptionTable, given: Mapping[str, Any]) -> list[str]:
    """Return the tokens of *given* in table order; see :func:`to_argv`."""
    unknown = set(given) - set(table.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    tokens: list[str] = []
    for option in table:
        if option.name not in given:
            continue
        tokens.extend(option.format(given[option.name]))
    return tokens


def _transcribe_argv(spec: RunSpec, defaults: Mapping[str, Any]) -> list[str]:
    def code_default(option: Option) -> Any:
        if option.name == "model" and spec.method.name == "audio-api":
            return DEFAULT_AUDIO_MODEL
        return option.default

    given = _spell_values(OPTIONS, _flatten(spec), defaults, code_default)
    _spell_provider(
        given,
        spec.model,
        defaults,
        lambda trial: resolve_spec(trial, defaults).spec.model,
        calls_model=spec.method.name in _MODEL_METHODS,
    )
    return _argv(OPTIONS, _with_keywords(given))


def _repair_argv(spec: RepairSpec, defaults: Mapping[str, Any]) -> list[str]:
    given = _spell_values(REPAIR_OPTIONS, _flatten(spec), defaults)
    given.update(_cause_flags(spec.selection.causes))
    if spec.selection.indices is not None:
        given["indices"] = _format_indices(spec.selection.indices)
    _spell_provider(
        given,
        spec.model,
        defaults,
        lambda trial: resolve_repair(trial, defaults).spec.model,
        calls_model=True,
    )
    return _argv(REPAIR_OPTIONS, _with_keywords(given))
