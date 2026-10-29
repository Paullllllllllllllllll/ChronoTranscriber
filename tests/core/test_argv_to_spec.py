"""Hand-written command lines parsed into the run spec, as the CLI parses them.

Each row parses one subcommand's arguments with the CLI parser, resolves them
with the spec module and compares the spec and the source of every field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber.cli.parser import UsageError, build_parser
from chronotranscriber.common.options import OptionTable, Source
from chronotranscriber.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    CONTEXT_NONE,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_GOOGLE_AUDIO_MODEL,
    FAILURE_CAUSES,
    OPTIONS,
    RENDER_OPTIONS,
    REPAIR_OPTIONS,
    AudioSpec,
    BatchCancelSpec,
    BatchCheckSpec,
    CommandSpec,
    ImageSpec,
    InputSpec,
    MethodSpec,
    ModelSpec,
    OcrSpec,
    OutputSpec,
    PromptSpec,
    RenderSpec,
    RepairControl,
    RepairSelection,
    RepairSpec,
    Resolution,
    RunControl,
    RunSpec,
    SpecError,
    resolve_batch_cancel,
    resolve_batch_check,
    resolve_render,
    resolve_repair,
    resolve_spec,
)

FLAG = Source.FLAG
SETTINGS = Source.SETTINGS
DEFAULT = Source.DEFAULT

CODE_MODEL = ModelSpec(
    provider="openai",
    model="gpt-6-luna",
    endpoint=None,
    reasoning_effort="xhigh",
    verbosity="medium",
    max_output_tokens=None,
    temperature=1.0,
    top_p=1.0,
    service_tier="default",
    response_format=None,
)
CODE_PROMPT = PromptSpec(
    schema="markdown_transcription_schema",
    prompt=None,
    context=None,
    context_image=None,
)
CODE_DEFAULTS = RunSpec(
    input=InputSpec(
        path=Path("in.pdf"), type="auto", files=(), recursive=False, pages=None
    ),
    method=MethodSpec(name="llm", searchable_pdfs="ocr"),
    output=OutputSpec(directory=None, layout="hash", formats=("txt",)),
    prompt=CODE_PROMPT,
    model=CODE_MODEL,
    images=ImageSpec(
        dpi=None, detail=None, format=None, jpeg_quality=None, grayscale=None
    ),
    ocr=OcrSpec(config="--oem 3 --psm 6", preprocess="default"),
    audio=AudioSpec(whisper_model="large-v3", language=None, chunk_seconds=600),
    run=RunControl(
        batch=False,
        sync_fallback=False,
        retry_errors=True,
        force=False,
        concurrency=16,
        keep_working_files=False,
        dry_run=False,
        json=False,
    ),
)
REPAIR_DEFAULTS = RepairSpec(
    input=Path("d.txt"),
    selection=RepairSelection(causes=("api_error",), indices=None),
    prompt=CODE_PROMPT,
    model=CODE_MODEL,
    images=ImageSpec(),
    run=RepairControl(batch=False, dry_run=False, json=False),
)
CHECK_DEFAULTS = BatchCheckSpec(
    directory=Path("."), batch_ids=(), formats=None, json=False
)
UNSELECTED = {"input.select": None, "input.all": False}
"""The alternatives a given ``--files``, ``--select`` or ``--all`` resets."""
CANCEL_DEFAULTS = BatchCancelSpec(batch_ids=(), all=False, json=False)
RENDER_DEFAULTS = RenderSpec(input=Path("."))

type Resolver = Callable[[Mapping[str, Any], Mapping[str, Any]], Resolution[Any]]


@dataclass(frozen=True)
class Command:
    """A subcommand: its words, option table, resolver and base command line."""

    words: tuple[str, ...]
    table: OptionTable
    resolve: Resolver
    base_argv: tuple[str, ...]
    base: CommandSpec
    base_fields: tuple[str, ...]


COMMANDS: dict[str, Command] = {
    "transcribe": Command(
        ("transcribe",),
        OPTIONS,
        resolve_spec,
        ("--input", "in.pdf"),
        CODE_DEFAULTS,
        ("input.path",),
    ),
    "repair": Command(
        ("repair",),
        REPAIR_OPTIONS,
        resolve_repair,
        ("--input", "d.txt"),
        REPAIR_DEFAULTS,
        ("input",),
    ),
    "batch check": Command(
        ("batch", "check"),
        BATCH_CHECK_OPTIONS,
        resolve_batch_check,
        (),
        CHECK_DEFAULTS,
        (),
    ),
    "batch cancel": Command(
        ("batch", "cancel"),
        BATCH_CANCEL_OPTIONS,
        lambda flags, _: resolve_batch_cancel(flags),
        (),
        CANCEL_DEFAULTS,
        (),
    ),
    "render": Command(
        ("render",),
        RENDER_OPTIONS,
        lambda flags, _: resolve_render(flags),
        (),
        RENDER_DEFAULTS,
        (),
    ),
}


def _flags(command: Command, argv: tuple[str, ...]) -> dict[str, Any]:
    """Parse *argv* with the CLI parser and return the given options by name."""
    namespace = build_parser().parse_args([*command.words, *argv])
    return command.table.values(namespace)


def _resolve(
    command: Command, argv: tuple[str, ...], defaults: Mapping[str, Any]
) -> Resolution[Any]:
    checked = OPTIONS.check_defaults(defaults)
    return command.resolve(_flags(command, argv), checked)


def _with(spec: Any, changes: Mapping[str, Any]) -> Any:
    """Return *spec* with the dotted-path fields of *changes* replaced."""
    for path, value in changes.items():
        spec = _set(spec, path.split("."), value)
    return spec


def _set(obj: Any, names: list[str], value: Any) -> Any:
    head, *rest = names
    if rest:
        part = getattr(obj, head)
        assert is_dataclass(part)
        value = _set(part, rest, value)
    return replace(obj, **{head: value})


@dataclass(frozen=True)
class Row:
    """One command line and the spec fields it sets.

    ``sets`` take the source FLAG unless ``sources`` names another one; a
    given option resets its alternatives, which take its source.
    """

    command: str
    option: str
    argv: tuple[str, ...]
    sets: Mapping[str, Any]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Source] = field(default_factory=dict)


def _row(
    option: str,
    *words: str,
    sets: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
    sources: Mapping[str, Source] | None = None,
    command: str = "transcribe",
) -> Row:
    argv = (*COMMANDS[command].base_argv, *words)
    return Row(command, option, argv, sets, defaults or {}, sources or {})


def _model(**changes: Any) -> dict[str, Any]:
    return {f"model.{name}": value for name, value in changes.items()}


ROWS = [
    Row(
        "transcribe",
        "input",
        ("--input", "a b/c.pdf"),
        {"input.path": Path("a b/c.pdf")},
    ),
    _row("type", "--type", "pdf", sets={"input.type": "pdf"}),
    _row("type", "--type", "images", sets={"input.type": "images"}),
    _row(
        "type",
        "--type",
        "epub",
        sets={"input.type": "epub", "method.name": "native"},
    ),
    _row(
        "type",
        "--type",
        "mobi",
        sets={"input.type": "mobi", "method.name": "native"},
    ),
    _row(
        "type",
        "--type",
        "audio",
        sets={"input.type": "audio", "method.name": "whisper"},
    ),
    _row(
        "type",
        "--type",
        "audio",
        sets={"input.type": "audio", "method.name": "whisper"},
        defaults={"method": "tesseract"},
    ),
    _row(
        "files",
        "--type",
        "pdf",
        "--files",
        "a.pdf",
        "--files",
        "b c.pdf,a.pdf,scans",
        sets={
            "input.type": "pdf",
            "input.files": ("a.pdf", "b c.pdf", "scans"),
            **UNSELECTED,
        },
    ),
    _row(
        "select",
        "--select",
        "1-3,ledger",
        sets={"input.select": "1-3,ledger", "input.files": (), "input.all": False},
    ),
    _row(
        "all",
        "--all",
        sets={"input.all": True, "input.files": (), "input.select": None},
    ),
    _row(
        "select",
        "--select",
        "2",
        sets={"input.select": "2", "input.files": (), "input.all": False},
        defaults={"all": True},
    ),
    _row(
        "recursive",
        "--type",
        "images",
        "--recursive",
        sets={"input.type": "images", "input.recursive": True},
    ),
    _row(
        "recursive",
        "--no-recursive",
        sets={"input.recursive": False},
        defaults={"recursive": True},
    ),
    _row("pages", "--pages", "first:5", sets={"input.pages": "first:5"}),
    _row("pages", "--pages", "1,3,5-8", sets={"input.pages": "1,3,5-8"}),
    _row("pages", "--pages=-7", sets={"input.pages": "-7"}),
    _row("method", "--method", "tesseract", sets={"method.name": "tesseract"}),
    _row(
        "method",
        "--type",
        "pdf",
        "--method",
        "native",
        sets={"input.type": "pdf", "method.name": "native"},
    ),
    _row(
        "method",
        "--type",
        "audio",
        "--method",
        "whisper",
        sets={"input.type": "audio", "method.name": "whisper"},
    ),
    _row(
        "method",
        "--type",
        "audio",
        "--method",
        "audio-api",
        sets={
            "input.type": "audio",
            "method.name": "audio-api",
            **_model(model=DEFAULT_AUDIO_MODEL),
        },
        sources={"model.model": DEFAULT},
    ),
    _row(
        "method",
        "--type",
        "audio",
        "--method",
        "audio-api",
        "--provider",
        "google",
        sets={
            "input.type": "audio",
            "method.name": "audio-api",
            **_model(provider="google", model=DEFAULT_GOOGLE_AUDIO_MODEL),
        },
        sources={"model.model": DEFAULT},
    ),
    _row(
        "searchable_pdfs",
        "--searchable-pdfs",
        "text",
        sets={"method.searchable_pdfs": "text"},
    ),
    _row("output", "--output", "x y", sets={"output.directory": Path("x y")}),
    _row(
        "beside_input",
        "--beside-input",
        sets={"output.directory": None},
        defaults={"output": "out"},
    ),
    _row("layout", "--layout", "mirror", sets={"output.layout": "mirror"}),
    _row("formats", "--formats", "md", sets={"output.formats": ("md",)}),
    _row("formats", "--formats", "md,txt", sets={"output.formats": ("txt", "md")}),
    _row(
        "formats",
        "--formats",
        "md",
        "--formats",
        "txt",
        sets={"output.formats": ("txt", "md")},
    ),
    _row(
        "schema",
        "--schema",
        "plain_text_transcription_schema",
        sets={"prompt.schema": "plain_text_transcription_schema"},
    ),
    _row("schema", "--schema=-s.json", sets={"prompt.schema": "-s.json"}),
    _row("prompt", "--prompt", "p q.txt", sets={"prompt.prompt": Path("p q.txt")}),
    _row(
        "prompt",
        "--prompt",
        "default",
        sets={"prompt.prompt": None},
        defaults={"prompt": "p.txt"},
    ),
    _row("context", "--context", "ctx.txt", sets={"prompt.context": "ctx.txt"}),
    _row(
        "context",
        "--context",
        "Basel ledgers, 1780s",
        sets={"prompt.context": "Basel ledgers, 1780s"},
    ),
    _row("context", "--context", "none", sets={"prompt.context": CONTEXT_NONE}),
    _row(
        "fallback_context",
        "--fallback-context",
        "notes.txt",
        sets={"prompt.fallback_context": "notes.txt"},
    ),
    _row(
        "fallback_context",
        "--fallback-context",
        "none",
        sets={"prompt.fallback_context": None},
        defaults={"fallback_context": "context/transcr_context.txt"},
    ),
    _row(
        "context",
        "--context",
        "auto",
        sets={"prompt.context": None},
        defaults={"context": "ctx.txt"},
    ),
    _row(
        "context_image",
        "--context-image",
        "page.png",
        sets={"prompt.context_image": Path("page.png")},
    ),
    _row(
        "context_image",
        "--context-image",
        "none",
        sets={"prompt.context_image": CONTEXT_NONE},
    ),
    _row(
        "provider",
        "--provider",
        "anthropic",
        "--method",
        "tesseract",
        sets={"method.name": "tesseract", **_model(provider="anthropic")},
    ),
    _row(
        "model",
        "--model",
        "gpt-5.6-terra",
        sets=_model(model="gpt-5.6-terra", provider="openai"),
    ),
    _row(
        "model",
        "--model",
        "claude-sonnet-5-5",
        sets=_model(model="claude-sonnet-5-5", provider="anthropic"),
    ),
    _row(
        "model",
        "--model",
        "openrouter:anthropic/claude-sonnet-5-5",
        sets=_model(model="anthropic/claude-sonnet-5-5", provider="openrouter"),
    ),
    _row(
        "model",
        "--provider",
        "google",
        "--model",
        "gemini-3.1-pro",
        sets=_model(model="gemini-3.1-pro", provider="google"),
    ),
    _row(
        "endpoint",
        "--endpoint",
        "local",
        "--model",
        "qwen-vl",
        sets=_model(endpoint="local", provider="custom", model="qwen-vl"),
    ),
    _row(
        "reasoning_effort",
        "--reasoning-effort",
        "low",
        sets=_model(reasoning_effort="low"),
    ),
    _row("verbosity", "--verbosity", "high", sets=_model(verbosity="high")),
    _row(
        "max_output_tokens",
        "--max-output-tokens",
        "8000",
        sets=_model(max_output_tokens=8000),
    ),
    _row("temperature", "--temperature", "0.4", sets=_model(temperature=0.4)),
    _row("top_p", "--top-p", "0.9", sets=_model(top_p=0.9)),
    _row(
        "service_tier",
        "--service-tier",
        "priority",
        sets=_model(service_tier="priority"),
    ),
    _row("decisions", "--decisions", "shadow", sets={"decision.mode": "shadow"}),
    _row(
        "decision_model",
        "--decisions",
        "shadow",
        "--decision-model",
        "gpt-6-luna",
        sets={"decision.mode": "shadow", "decision.model": "gpt-6-luna"},
    ),
    _row(
        "decision_provider",
        "--decisions",
        "apply",
        "--decision-provider",
        "openai",
        sets={"decision.mode": "apply", "decision.provider": "openai"},
    ),
    _row(
        "decision_endpoint",
        "--decisions",
        "shadow",
        "--decision-endpoint",
        "https://decide.example/v1",
        sets={
            "decision.mode": "shadow",
            "decision.endpoint": "https://decide.example/v1",
        },
    ),
    _row(
        "response_format",
        "--response-format",
        "text",
        sets=_model(response_format="text"),
    ),
    _row("dpi", "--dpi", "native", sets={"images.dpi": "native"}),
    _row("dpi", "--dpi", "300", sets={"images.dpi": 300}),
    _row("image_detail", "--image-detail", "high", sets={"images.detail": "high"}),
    _row("image_format", "--image-format", "png", sets={"images.format": "png"}),
    _row("jpeg_quality", "--jpeg-quality", "80", sets={"images.jpeg_quality": 80}),
    _row("grayscale", "--grayscale", sets={"images.grayscale": True}),
    _row("grayscale", "--no-grayscale", sets={"images.grayscale": False}),
    _row(
        "ocr_config",
        "--method",
        "tesseract",
        "--ocr-config=--psm 4 -l deu",
        sets={"method.name": "tesseract", "ocr.config": "--psm 4 -l deu"},
    ),
    _row(
        "ocr_preprocess",
        "--method",
        "tesseract",
        "--ocr-preprocess",
        "sauvola",
        sets={"method.name": "tesseract", "ocr.preprocess": "sauvola"},
    ),
    _row(
        "whisper_model",
        "--type",
        "audio",
        "--whisper-model",
        "small",
        sets={
            "input.type": "audio",
            "method.name": "whisper",
            "audio.whisper_model": "small",
        },
    ),
    _row(
        "language",
        "--type",
        "audio",
        "--language",
        "de",
        sets={"input.type": "audio", "method.name": "whisper", "audio.language": "de"},
    ),
    _row(
        "language",
        "--language",
        "auto",
        sets={"audio.language": None},
        defaults={"language": "fr"},
    ),
    _row(
        "audio_chunk_seconds",
        "--type",
        "audio",
        "--audio-chunk-seconds",
        "120",
        sets={
            "input.type": "audio",
            "method.name": "whisper",
            "audio.chunk_seconds": 120,
        },
    ),
    _row("batch", "--batch", sets={"run.batch": True}),
    _row(
        "batch",
        "--no-batch",
        sets={"run.batch": False},
        defaults={"batch": True},
    ),
    _row(
        "sync_fallback",
        "--batch",
        "--sync-fallback",
        sets={"run.batch": True, "run.sync_fallback": True},
    ),
    _row("retry_errors", "--retry-errors", sets={"run.retry_errors": True}),
    _row("retry_errors", "--no-retry-errors", sets={"run.retry_errors": False}),
    _row("force", "--force", sets={"run.force": True}),
    _row("concurrency", "--concurrency", "4", sets={"run.concurrency": 4}),
    _row(
        "keep_working_files",
        "--keep-working-files",
        sets={"run.keep_working_files": True},
    ),
    _row("dry_run", "--dry-run", sets={"run.dry_run": True}),
    _row("json", "--json", sets={"run.json": True}),
    _row("settings", "--settings", "other.yaml", sets={}),
    _row(
        "errors_only",
        "--errors-only",
        sets={"selection.causes": ("api_error",)},
        command="repair",
    ),
    _row(
        "not_possible",
        "--not-possible",
        sets={"selection.causes": ("not_possible",)},
        command="repair",
    ),
    _row(
        "no_text",
        "--errors-only",
        "--no-text",
        sets={"selection.causes": ("api_error", "no_text")},
        command="repair",
    ),
    _row(
        "all_failures",
        "--all-failures",
        sets={"selection.causes": FAILURE_CAUSES},
        command="repair",
    ),
    _row(
        "skipped",
        "--skipped",
        sets={"selection.causes": ("skipped",)},
        command="repair",
    ),
    _row(
        "indices",
        "--indices",
        "12,0,3-5",
        sets={"selection.indices": (0, 3, 4, 5, 12)},
        command="repair",
    ),
    _row(
        "model",
        "--model",
        "claude-sonnet-5-5",
        "--context",
        "none",
        "--dpi",
        "200",
        sets={
            **_model(model="claude-sonnet-5-5", provider="anthropic"),
            "prompt.context": CONTEXT_NONE,
            "images.dpi": 200,
        },
        command="repair",
    ),
    _row(
        "batch",
        "--batch",
        "--dry-run",
        "--json",
        sets={"run.batch": True, "run.dry_run": True, "run.json": True},
        defaults={"concurrency": 4},
        command="repair",
    ),
    _row(
        "batch",
        "--no-batch",
        sets={"run.batch": False},
        defaults={"batch": True},
        command="repair",
    ),
    _row(
        "input",
        "--input",
        "out dir",
        sets={"directory": Path("out dir")},
        command="batch check",
    ),
    _row(
        "batch_ids",
        "--batch-ids",
        "b1,b2",
        "--batch-ids",
        "b3",
        sets={"batch_ids": ("b1", "b2", "b3")},
        command="batch check",
    ),
    _row(
        "formats",
        "--formats",
        "md",
        sets={"formats": ("md",)},
        defaults={"formats": ["txt", "md"]},
        command="batch check",
    ),
    _row(
        "adopt",
        "--adopt",
        "msgbatch_1",
        "--json",
        sets={"adopt": "msgbatch_1", "abandon": None, "json": True},
        command="batch check",
    ),
    _row(
        "abandon",
        "--abandon=-s1",
        sets={"abandon": "-s1", "adopt": None},
        command="batch check",
    ),
    _row(
        "batch_ids",
        "--batch-ids=-b1,b2",
        sets={"batch_ids": ("-b1", "b2"), "all": False, "directory": None},
        command="batch cancel",
    ),
    _row(
        "all",
        "--all",
        "--json",
        sets={"all": True, "batch_ids": (), "directory": None, "json": True},
        command="batch cancel",
    ),
    _row(
        "input",
        "--input",
        "jobs",
        sets={"directory": Path("jobs"), "batch_ids": (), "all": False},
        command="batch cancel",
    ),
    _row(
        "input",
        "--input",
        "out/chronotranscriber.sqlite",
        sets={"input": Path("out/chronotranscriber.sqlite")},
        command="render",
    ),
    _row(
        "formats",
        "--formats",
        "md",
        "--formats",
        "txt",
        "--force",
        "--dry-run",
        "--json",
        sets={
            "formats": ("txt", "md"),
            "force": True,
            "dry_run": True,
            "json": True,
        },
        defaults={"formats": ["md"]},
        command="render",
    ),
]


def test_every_transcribe_option_has_a_row() -> None:
    options = {row.option for row in ROWS if row.command == "transcribe"}
    assert options == set(OPTIONS.names)


@pytest.mark.parametrize("command", list(COMMANDS))
def test_every_subcommand_has_a_row(command: str) -> None:
    assert any(row.command == command for row in ROWS)


@pytest.mark.parametrize(
    "row",
    ROWS,
    ids=[f"{row.command}:{row.option}:{' '.join(row.argv)}" for row in ROWS],
)
def test_a_command_line_sets_its_fields(row: Row) -> None:
    command = COMMANDS[row.command]
    assert row.option in _flags(command, row.argv)

    resolution = _resolve(command, row.argv, row.defaults)

    base = command.base
    if row.option == "input" and row.command == "transcribe":
        base = _with(base, {"input.path": Path(row.argv[1])})
    assert resolution.spec == _with(base, row.sets)
    flagged = {*command.base_fields, *row.sets}
    expected = {
        path: row.sources.get(path, FLAG if path in flagged else DEFAULT)
        for path in resolution.sources
    }
    assert dict(resolution.sources) == expected


def test_transcribe_with_the_code_defaults() -> None:
    resolution = _resolve(COMMANDS["transcribe"], ("--input", "in.pdf"), {})
    assert resolution.spec == CODE_DEFAULTS
    assert {path for path, s in resolution.sources.items() if s is FLAG} == {
        "input.path"
    }


def test_the_settings_option_sets_the_settings_path() -> None:
    flags = _flags(COMMANDS["transcribe"], ("--input", "x", "--settings", "o.yaml"))
    assert flags["settings"] == Path("o.yaml")


COMMAND_LINES: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        (
            "--input",
            "scans",
            "--type",
            "images",
            "--files",
            "a,b",
            "--recursive",
            "--model",
            "claude-sonnet-5-5",
            "--formats",
            "txt,md",
            "--layout",
            "mirror",
            "--context",
            "none",
            "--batch",
            "--sync-fallback",
            "--dry-run",
            "--json",
        ),
        {
            "input.path": Path("scans"),
            "input.type": "images",
            "input.files": ("a", "b"),
            **UNSELECTED,
            "input.recursive": True,
            **_model(model="claude-sonnet-5-5", provider="anthropic"),
            "output.formats": ("txt", "md"),
            "output.layout": "mirror",
            "prompt.context": CONTEXT_NONE,
            "run.batch": True,
            "run.sync_fallback": True,
            "run.dry_run": True,
            "run.json": True,
        },
    ),
    (
        (
            "--input",
            "talks",
            "--type",
            "audio",
            "--method",
            "audio-api",
            "--model",
            "gpt-4o-transcribe",
            "--language",
            "de",
            "--audio-chunk-seconds",
            "300",
            "--pages",
            "last:2",
            "--output",
            "out",
            "--retry-errors",
            "--keep-working-files",
            "--concurrency",
            "2",
        ),
        {
            "input.path": Path("talks"),
            "input.type": "audio",
            "input.pages": "last:2",
            "method.name": "audio-api",
            **_model(model="gpt-4o-transcribe"),
            "audio.language": "de",
            "audio.chunk_seconds": 300,
            "output.directory": Path("out"),
            "run.retry_errors": True,
            "run.keep_working_files": True,
            "run.concurrency": 2,
        },
    ),
    (
        (
            "--input",
            "book.pdf",
            "--type",
            "pdf",
            "--method",
            "tesseract",
            "--ocr-config=-l deu --psm 4",
            "--ocr-preprocess",
            "none",
            "--formats",
            "md",
            "--force",
        ),
        {
            "input.path": Path("book.pdf"),
            "input.type": "pdf",
            "method.name": "tesseract",
            "ocr.config": "-l deu --psm 4",
            "ocr.preprocess": "none",
            "output.formats": ("md",),
            "run.force": True,
        },
    ),
]


@pytest.mark.parametrize(
    ("argv", "changes"), COMMAND_LINES, ids=[" ".join(c[0]) for c in COMMAND_LINES]
)
def test_a_full_command_line_gives_the_written_spec(
    argv: tuple[str, ...], changes: dict[str, Any]
) -> None:
    resolution = _resolve(COMMANDS["transcribe"], argv, {})
    assert resolution.spec == _with(CODE_DEFAULTS, changes)


PRECEDENCE: list[tuple[tuple[str, ...], dict[str, Any], str, Any, Source]] = [
    ((), {}, "run.concurrency", 16, DEFAULT),
    ((), {"concurrency": 4}, "run.concurrency", 4, SETTINGS),
    (("--concurrency", "8"), {"concurrency": 4}, "run.concurrency", 8, FLAG),
    ((), {"formats": ["md"]}, "output.formats", ("md",), SETTINGS),
    (("--formats", "txt"), {"formats": ["md"]}, "output.formats", ("txt",), FLAG),
    ((), {"output": "out"}, "output.directory", Path("out"), SETTINGS),
    (("--beside-input",), {"output": "out"}, "output.directory", None, FLAG),
    ((), {"model": "claude-sonnet-5-5"}, "model.provider", "anthropic", SETTINGS),
    (
        ("--model", "gpt-5.6-terra"),
        {"provider": "anthropic"},
        "model.provider",
        "openai",
        FLAG,
    ),
    ((), {"endpoint": "local", "model": "q"}, "model.provider", "custom", SETTINGS),
    (
        ("--provider", "openai"),
        {"endpoint": "local"},
        "model.endpoint",
        None,
        FLAG,
    ),
    ((), {"method": "tesseract"}, "method.name", "tesseract", SETTINGS),
    (("--type", "epub"), {"method": "tesseract"}, "method.name", "native", FLAG),
    ((), {"type": "audio"}, "method.name", "whisper", SETTINGS),
    ((), {"batch": True}, "run.batch", True, SETTINGS),
    (("--no-batch",), {"batch": True}, "run.batch", False, FLAG),
    ((), {"dpi": 300}, "images.dpi", 300, SETTINGS),
    (("--dpi", "native"), {"dpi": 300}, "images.dpi", "native", FLAG),
    ((), {"context": "ctx.txt"}, "prompt.context", "ctx.txt", SETTINGS),
    ((), {"fallback_context": "f.txt"}, "prompt.fallback_context", "f.txt", SETTINGS),
    ((), {"all": True}, "input.all", True, SETTINGS),
    (("--select", "1"), {"all": True}, "input.all", False, FLAG),
    ((), {"language": "de"}, "audio.language", "de", SETTINGS),
    ((), {"reasoning_effort": "low"}, "model.reasoning_effort", "low", SETTINGS),
    (
        ("--reasoning-effort", "xhigh"),
        {"reasoning_effort": "low"},
        "model.reasoning_effort",
        "xhigh",
        FLAG,
    ),
]


@pytest.mark.parametrize(("argv", "defaults", "path", "value", "source"), PRECEDENCE)
def test_precedence_of_flag_settings_default_and_code_default(
    argv: tuple[str, ...],
    defaults: dict[str, Any],
    path: str,
    value: Any,
    source: Source,
) -> None:
    resolution = _resolve(
        COMMANDS["transcribe"], ("--input", "in.pdf", *argv), defaults
    )
    obj: Any = resolution.spec
    for name in path.split("."):
        obj = getattr(obj, name)
    assert obj == value
    assert resolution.sources[path] is source


def test_repair_keeps_a_shared_settings_default_and_ignores_the_others() -> None:
    defaults = {"model": "gpt-5.6-terra", "batch": True, "concurrency": 4}
    resolution = _resolve(COMMANDS["repair"], ("--input", "d.txt"), defaults)
    assert resolution.spec.model.model == "gpt-5.6-terra"
    assert resolution.sources["model.model"] is SETTINGS
    assert resolution.spec.run.batch is True
    assert resolution.sources["run.batch"] is SETTINGS
    assert "run.concurrency" not in resolution.sources


def test_repair_takes_the_current_directory_by_default() -> None:
    resolution = _resolve(COMMANDS["repair"], (), {})
    assert resolution.spec.input == Path(".")


def test_batch_check_takes_no_settings_formats() -> None:
    resolution = _resolve(COMMANDS["batch check"], (), {"formats": ["md"]})
    assert resolution.spec.formats is None


SPEC_ERRORS: list[tuple[tuple[str, ...], dict[str, Any], str]] = [
    (("--provider", "anthropic"), {}, "cannot run the model 'gpt-6-luna'"),
    (
        ("--provider", "google"),
        {"model": "claude-sonnet-5-5"},
        "cannot run the model 'claude-sonnet-5-5'",
    ),
    (("--type", "epub", "--method", "llm"), {}, "--type epub takes --method native"),
    (("--type", "pdf", "--method", "whisper"), {}, "--type pdf takes"),
    (("--type", "audio", "--method", "llm"), {}, "whisper or audio-api, not llm"),
    (("--type", "images", "--method", "native"), {}, "--type images takes"),
]


@pytest.mark.parametrize(("argv", "defaults", "message"), SPEC_ERRORS)
def test_an_invalid_combination_is_a_spec_error(
    argv: tuple[str, ...], defaults: dict[str, Any], message: str
) -> None:
    with pytest.raises(SpecError, match=message):
        _resolve(COMMANDS["transcribe"], ("--input", "in.pdf", *argv), defaults)


@pytest.mark.parametrize(
    "argv",
    [
        ("--provider", "anthropic"),
        ("--provider", "google", "--method", "llm"),
        ("--type", "epub", "--method", "llm"),
        ("--type", "audio", "--method", "tesseract"),
    ],
)
def test_an_invalid_combination_exits_2(
    argv: tuple[str, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = str(tmp_path / "in.pdf")
    code = entry.main(["transcribe", "--input", source, *argv, "--dry-run", "--json"])
    assert code == 2
    assert "Invalid arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ("transcribe", "--input", "x", "--type", "pdfs"),
        ("transcribe", "--input", "x", "--method", "gpt"),
        ("transcribe", "--input", "x", "--output", "o", "--beside-input"),
        ("repair", "--input", "d.txt", "--indices", "5-2"),
        ("batch", "check", "--adopt", "b1", "--abandon", "s1"),
        ("batch", "cancel", "--all", "--input", "jobs"),
        ("render", "--formats", "json"),
    ],
)
def test_a_parse_error_exits_2(argv: tuple[str, ...]) -> None:
    with pytest.raises(UsageError):
        build_parser().parse_args(list(argv))
    assert entry.main(list(argv)) == 2
