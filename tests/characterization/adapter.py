"""The single entry point through which characterization tests run the tool.

``run_tool`` keeps the keyword names of the commands the cases were recorded
with and translates them, and the config overrides a case names, into the
argv of ``chronotranscriber`` and a temporary settings file.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import io
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from tests.characterization.fakes import CUSTOM_KEY_ENV, DUMMY_KEYS, FakeLLM

Command = Literal["transcribe", "repair", "check_batches", "cancel_batches"]
ResponseFormat = Literal["schema", "json", "text"]

CUSTOM_MODEL = "org/fake-model"
CUSTOM_BASE_URL = "https://fake-endpoint.invalid/v1/"
CUSTOM_ENDPOINT = "fake"

CONFIG_FILES = (
    "model_config",
    "paths_config",
    "concurrency_config",
    "image_processing_config",
    "audio_config",
    "api_keys_config",
)
"""Override groups ``run_tool`` translates; ``config`` keys name them."""

_SUBCOMMANDS: dict[str, tuple[str, ...]] = {
    "transcribe": ("transcribe",),
    "repair": ("repair",),
    "check_batches": ("batch", "check"),
    "cancel_batches": ("batch", "cancel"),
}
_TYPES = {
    "pdfs": "pdf",
    "images": "images",
    "epubs": "epub",
    "mobis": "mobi",
    "audio": "audio",
}
_METHODS = {
    "gpt": "llm",
    "native": "native",
    "tesseract": "tesseract",
    "audio-api": "audio-api",
    "whisper": "whisper",
}
_VERBOSITIES = {"concise": "low", "medium": "medium", "verbose": "high"}
_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"

# Keywords each command accepts; the cancel command's force and the check
# command's no_diagnostics have no effect on the current CLI.
_KEYWORDS: dict[str, frozenset[str]] = {
    "transcribe": frozenset(
        {
            "input",
            "output",
            "type",
            "method",
            "auto",
            "sync_fallback",
            "schema",
            "context",
            "fallback_context",
            "context_image",
            "model",
            "provider",
            "reasoning_effort",
            "model_verbosity",
            "max_output_tokens",
            "service_tier",
            "resume",
            "force",
            "files",
            "recursive",
            "output_mode",
            "pages",
            "dry_run",
            "retry_errors",
            "batch",
            "output_format",
            "response_format",
            "endpoint",
            "keep_working_files",
        }
    ),
    "repair": frozenset(
        {
            "transcription",
            "errors_only",
            "not_possible",
            "no_text",
            "all_failures",
            "indices",
            "batch",
            "model",
            "provider",
        }
    ),
    "check_batches": frozenset({"directory", "no_diagnostics", "output_format"}),
    "cancel_batches": frozenset({"batch_ids", "force"}),
}


@dataclass(frozen=True)
class ItemPaths:
    """Where the tool writes one input item.

    ``folder`` is the item's folder (hash-suffixed unless mirrored), ``output``
    the final transcription with its real extension and ``database`` the
    store of the item's output root.
    """

    folder: Path
    output: Path
    database: Path


@dataclass
class RunResult:
    """Outcome of one ``run_tool`` call.

    ``calls`` and ``models`` are copies of the LLM traffic of this run;
    ``created`` lists the files the run created in ``scan_dirs``. Console
    messages are on ``stderr``; ``stdout`` holds only the JSON line.
    """

    command: str
    argv: list[str]
    exit_code: int
    summary: dict[str, Any] | None
    stdout: str
    stderr: str
    root: Path
    config_dir: Path
    output_dir: Path | None
    scan_dirs: tuple[Path, ...]
    created: list[Path]
    llm: FakeLLM
    calls: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    output_format: str = "txt"
    layout: str = "hash"
    input_root: Path | None = None
    beside_input: bool = False

    def paths(self, item: Path, output_dir: Path | None = None) -> ItemPaths:
        """Return the paths the tool writes for input *item* in this run.

        *output_dir* defaults to the run's output folder.
        """
        target = output_dir if output_dir is not None else self.output_dir
        if target is None and not self.beside_input:
            raise ValueError("this run has no output folder")
        return item_paths(
            item,
            output_dir=target if target is not None else item.parent,
            output_format=self.output_format,
            layout=self.layout,
            input_root=self.input_root,
            beside_input=self.beside_input,
        )

    def output(self, item: Path) -> Path:
        """Return the final transcription path of input *item*."""
        return self.paths(item).output


def item_paths(
    item: Path,
    *,
    output_dir: Path,
    output_format: str = "txt",
    layout: str = "hash",
    input_root: Path | None = None,
    beside_input: bool = False,
) -> ItemPaths:
    """Return the paths the tool derives for *item*, from its own naming code.

    A folder is an image folder; a file is named by its stem. The final
    output is named with ``.txt`` and then given the real extension, except
    in the mirror layout, which names it with the real extension directly.
    """
    from chronotranscriber.output.paths import (
        create_safe_directory_name,
        create_safe_filename,
        mirror_output_path,
    )
    from chronotranscriber.output.store import database_path

    item = item.resolve()
    output_dir = output_dir.resolve()
    root = input_root.resolve() if input_root is not None else None
    is_folder = item.is_dir()
    stem = item.name if is_folder else item.stem
    ext = f".{output_format}"
    ebook = not is_folder and item.suffix.lower() in {
        ".epub",
        ".mobi",
        ".azw",
        ".azw3",
        ".kfx",
    }

    def named(directory: Path) -> Path:
        return (directory / create_safe_filename(stem, ".txt", directory)).with_suffix(
            ext
        )

    if beside_input:
        folder = item.parent / create_safe_directory_name(stem)
        return ItemPaths(folder, named(item.parent), database_path(item.parent))
    database = database_path(output_dir)
    if layout == "mirror" and root is not None and not ebook:
        folder = mirror_output_path(
            item if is_folder else item.parent, root, output_dir
        )
        output = folder / create_safe_filename(stem, ext, folder)
        return ItemPaths(folder, output, database)
    key = _relative_key(item, root)
    folder = output_dir / create_safe_directory_name(key or stem)
    return ItemPaths(folder, named(folder), database)


def _relative_key(item: Path, input_root: Path | None) -> str | None:
    if input_root is None:
        return None
    try:
        relative = str(item.relative_to(input_root))
    except ValueError:
        return None
    return None if relative in (".", "") else relative


def custom_endpoint(
    response_format: ResponseFormat, model: str = CUSTOM_MODEL
) -> dict[str, Any]:
    """Return a ``model_config`` override that routes to the fake custom endpoint.

    *response_format* picks the endpoint mode: ``schema`` enforces the schema
    through structured output, ``json`` prompts for the schema, ``text``
    prompts for plain text.
    """
    return {
        "transcription_model": {
            "provider": "custom",
            "name": model,
            "custom_endpoint": {
                "base_url": CUSTOM_BASE_URL,
                "api_key_env_var": CUSTOM_KEY_ENV,
                "capabilities": {
                    "supports_vision": True,
                    "supports_structured_output": response_format == "schema",
                },
                "use_plain_text_prompt": response_format == "text",
            },
        }
    }


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return *base* with *override* merged in; mappings merge key by key."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ============================================================================
# Translation
# ============================================================================
class _Invocation:
    """Option values and settings collected from the keywords and overrides."""

    def __init__(self) -> None:
        self.flags: dict[str, Any] = {}
        self.settings: dict[str, Any] = {}

    def flag(self, name: str, value: Any) -> None:
        if name in self.flags and self.flags[name] != value:
            raise ValueError(
                f"conflicting values for --{name.replace('_', '-')}: "
                f"{self.flags[name]!r} and {value!r}"
            )
        self.flags[name] = value

    def setting(self, *path_and_value: Any) -> None:
        *path, value = path_and_value
        node = self.settings
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value


def _unknown(group: str, keys: Sequence[str]) -> None:
    if keys:
        raise ValueError(f"run_tool cannot translate {group} override(s) {keys}")


def _model_override(run: _Invocation, data: Mapping[str, Any]) -> None:
    model = dict(data.get("transcription_model") or {})
    _unknown("model_config", sorted(set(data) - {"transcription_model"}))
    if model.get("provider") != "custom":
        raise ValueError("only custom-endpoint model_config overrides translate")
    endpoint = dict(model.pop("custom_endpoint"))
    capabilities = dict(endpoint.get("capabilities") or {})
    structured = bool(capabilities.get("supports_structured_output"))
    plain = bool(endpoint.get("use_plain_text_prompt"))
    run.setting(
        "endpoints",
        CUSTOM_ENDPOINT,
        {
            "base_url": endpoint["base_url"],
            "api_key_env": endpoint["api_key_env_var"],
            "supports_vision": bool(capabilities.get("supports_vision", True)),
            "supports_schema": structured,
        },
    )
    run.flag("endpoint", CUSTOM_ENDPOINT)
    run.flag("model", model.pop("name"))
    run.flag("response_format", "schema" if structured else "text" if plain else "json")
    _unknown("model_config", sorted(set(model) - {"provider"}))


def _general_override(run: _Invocation, general: Mapping[str, Any]) -> bool:
    """Translate ``paths_config.general``; return whether outputs go beside."""
    values = dict(general)
    beside = bool(values.pop("input_paths_is_output_path", False))
    if values.pop("resume_mode", "skip") == "overwrite":
        run.flag("force", True)
    if values.pop("keep_preprocessed_images", False):
        run.flag("keep_working_files", True)
    if not values.pop("auto_mode_pdf_use_ocr_for_scanned", True):
        raise ValueError("auto mode always transcribes scanned PDFs with the method")
    if not values.pop("auto_mode_pdf_use_ocr_for_searchable", True):
        run.flag("searchable_pdfs", "text")
    methods = {
        values.pop(key, "gpt")
        for key in ("auto_mode_pdf_ocr_method", "auto_mode_image_ocr_method")
    }
    if methods != {"gpt"}:
        if len(methods) != 1:
            raise ValueError(f"auto mode takes one method, not {sorted(methods)}")
        run.flag("method", _METHODS[methods.pop()])
    _unknown("paths_config.general", sorted(values))
    return beside


def _concurrency_override(run: _Invocation, data: Mapping[str, Any]) -> None:
    concurrency = dict(data.get("concurrency") or {})
    _unknown("concurrency_config", sorted(set(data) - {"concurrency"}))
    transcription = dict(concurrency.pop("transcription", None) or {})
    _unknown("concurrency_config.concurrency", sorted(concurrency))
    retry = dict(transcription.pop("retry", None) or {})
    names = {
        "attempts": "max_attempts",
        "timeout_attempts": "timeout_attempts",
        "validation_attempts": "validation_attempts",
        "min_input_tokens": "min_input_tokens",
    }
    for key in [key for key in retry if key in names]:
        run.setting("retry", names[key], retry.pop(key))
    _unknown("concurrency retry", sorted(retry))
    if "concurrency_limit" in transcription:
        run.flag("concurrency", transcription.pop("concurrency_limit"))
    if "delay_between_tasks" in transcription:
        delay = transcription.pop("delay_between_tasks")
        run.setting("concurrency", "delay_between_tasks", delay)
    if "request_timeout" in transcription:
        run.setting("timeouts", "request", transcription.pop("request_timeout"))
    _unknown("concurrency transcription", sorted(transcription))


def _image_override(run: _Invocation, data: Mapping[str, Any]) -> None:
    tesseract = dict(data.get("tesseract_image_processing") or {})
    others = sorted(set(data) - {"tesseract_image_processing"})
    _unknown("image_processing_config", others)
    ocr = dict(tesseract.pop("ocr", None) or {})
    _unknown("tesseract_image_processing", sorted(tesseract))
    if "tesseract_cmd" in ocr:
        run.setting("tools", "tesseract_cmd", ocr.pop("tesseract_cmd"))
    if "tesseract_config" in ocr:
        run.flag("ocr_config", ocr.pop("tesseract_config"))
    _unknown("tesseract ocr", sorted(ocr))


def _audio_override(run: _Invocation, data: Mapping[str, Any]) -> None:
    values = copy.deepcopy(dict(data))
    ffmpeg = values.pop("ffmpeg", None) or {}
    for key in ("ffmpeg_cmd", "ffprobe_cmd"):
        if key in ffmpeg:
            run.setting("tools", key, ffmpeg.pop(key))
    _unknown("audio ffmpeg", sorted(ffmpeg))

    chunking = values.pop("chunking", None) or {}
    if "target_seconds" in chunking:
        run.flag("audio_chunk_seconds", chunking.pop("target_seconds"))
    for key in list(chunking):
        run.setting("audio_chunking", key, chunking.pop(key))

    _audio_whisper(run, values.pop("local_whisper", None) or {})
    _audio_remote(run, values.pop("audio_transcription", None) or {})
    _unknown("audio_config", sorted(values))


def _audio_whisper(run: _Invocation, whisper: dict[str, Any]) -> None:
    for key in ("beam_size", "vad_filter", "device", "compute_type"):
        if key in whisper:
            run.setting("whisper", key, whisper.pop(key))
    if "model_size" in whisper:
        run.flag("whisper_model", whisper.pop("model_size"))
    if "language" in whisper:
        run.flag("language", whisper.pop("language"))
    _unknown("local_whisper", sorted(whisper))


def _audio_remote(run: _Invocation, remote: dict[str, Any]) -> None:
    if "provider" in remote:
        run.flag("provider", remote.pop("provider"))
    openai = remote.pop("openai", None) or {}
    for key in ("prompt", "languages", "keywords"):
        if key in openai:
            run.setting("audio_api", key, openai.pop(key))
    if "language" in openai:
        run.flag("language", openai.pop("language"))
    if "model" in openai:
        run.flag("model", openai.pop("model"))
    google = remote.pop("google", None) or {}
    if "language_hint" in google:
        run.flag("language", google.pop("language_hint"))
    if "model" in google:
        run.flag("model", google.pop("model"))
    _unknown("audio_transcription", sorted({*remote, *openai, *google}))


def _translate_config(
    run: _Invocation, config: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Translate config overrides; return whether outputs go beside the input."""
    _unknown("config", sorted(set(config) - set(CONFIG_FILES)))
    beside = False
    for name, data in config.items():
        if name == "model_config":
            _model_override(run, data)
        elif name == "paths_config":
            _unknown("paths_config", sorted(set(data) - {"general"}))
            beside = _general_override(run, data.get("general") or {})
        elif name == "concurrency_config":
            _concurrency_override(run, data)
        elif name == "image_processing_config":
            _image_override(run, data)
        elif name == "audio_config":
            _audio_override(run, data)
        elif name == "api_keys_config":
            for provider, variable in data.items():
                run.setting("api_keys", provider, variable)
    return beside


def _copy_given(
    run: _Invocation, keywords: Mapping[str, Any], names: Sequence[str]
) -> None:
    """Pass each named keyword that is set on as the flag of the same name."""
    for name in names:
        if keywords.get(name) is not None:
            run.flag(name, keywords[name])


def _copy_switches(
    run: _Invocation, keywords: Mapping[str, Any], names: Sequence[str]
) -> None:
    """Raise each named switch whose keyword is true."""
    for name in names:
        if keywords.get(name):
            run.flag(name, True)


def _transcribe_flags(
    run: _Invocation, keywords: Mapping[str, Any], beside: bool
) -> None:
    if keywords.get("input") is not None:
        run.flag("input", keywords["input"])
    if beside:
        run.flag("beside_input", True)
    elif keywords.get("output") is not None:
        run.flag("output", keywords["output"])
    if keywords.get("auto"):
        run.flag("type", "auto")
    elif keywords.get("type") is not None:
        run.flag("type", _TYPES[keywords["type"]])
    if keywords.get("method") is not None:
        run.flag("method", _METHODS[keywords["method"]])
    if keywords.get("model_verbosity") is not None:
        run.flag("verbosity", _VERBOSITIES[keywords["model_verbosity"]])
    if keywords.get("output_format") is not None:
        run.flag("formats", (keywords["output_format"],))
    if keywords.get("output_mode") is not None:
        run.flag("layout", keywords["output_mode"])
    if keywords.get("files") is not None:
        run.flag("files", tuple(keywords["files"]))
    else:
        # A case that names no files covers every item of the folder.
        run.flag("all", True)
    if not keywords.get("retry_errors"):
        # A case keeps error pages on a rerun unless it asks to retry them.
        run.flag("retry_errors", False)
    _copy_given(
        run,
        keywords,
        (
            "schema",
            "context",
            "fallback_context",
            "context_image",
            "model",
            "provider",
            "reasoning_effort",
            "max_output_tokens",
            "service_tier",
            "pages",
            "response_format",
            "endpoint",
        ),
    )
    _copy_switches(
        run,
        keywords,
        (
            "sync_fallback",
            "force",
            "recursive",
            "dry_run",
            "retry_errors",
            "batch",
            "keep_working_files",
        ),
    )


def _command_flags(
    command: str, run: _Invocation, keywords: Mapping[str, Any], beside: bool
) -> None:
    if command == "transcribe":
        _transcribe_flags(run, keywords, beside)
    elif command == "repair":
        if keywords.get("transcription") is not None:
            run.flag("input", keywords["transcription"])
        _copy_given(run, keywords, ("indices", "model", "provider"))
        _copy_switches(
            run,
            keywords,
            ("errors_only", "not_possible", "no_text", "all_failures", "batch"),
        )
    elif command == "check_batches":
        if keywords.get("directory") is not None:
            run.flag("input", keywords["directory"])
        if keywords.get("output_format") is not None:
            run.flag("formats", (keywords["output_format"],))
    else:
        batch_ids = keywords.get("batch_ids")
        if batch_ids:
            run.flag("batch_ids", tuple(batch_ids))
        else:
            run.flag("all", True)


def build_argv(
    command: Command,
    keywords: Mapping[str, Any],
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return the argv and the settings data that run *command* with *keywords*.

    *keywords* are ``run_tool`` keywords; *config* the overrides a case
    names. The argv lacks ``--json`` and ``--settings``.

    Raises:
        TypeError: A keyword the command does not take.
        ValueError: An override with no flag or settings value.
    """
    if command not in _SUBCOMMANDS:
        raise ValueError(f"unknown command {command!r}")
    given = {
        name: value
        for name, value in keywords.items()
        if value is not None and value is not False
    }
    unknown = sorted(set(given) - _KEYWORDS[command])
    if unknown:
        raise TypeError(f"{command} has no flag for {unknown}")
    run = _Invocation()
    beside = _translate_config(run, config or {})
    _command_flags(command, run, given, beside)
    from chronotranscriber.spec import (
        BATCH_CANCEL_OPTIONS,
        BATCH_CHECK_OPTIONS,
        OPTIONS,
        REPAIR_OPTIONS,
    )

    table = {
        "transcribe": OPTIONS,
        "repair": REPAIR_OPTIONS,
        "check_batches": BATCH_CHECK_OPTIONS,
        "cancel_batches": BATCH_CANCEL_OPTIONS,
    }[command]
    return [*_SUBCOMMANDS[command], *table.to_argv(run.flags)], run.settings


def write_settings(folder: Path, data: Mapping[str, Any]) -> Path:
    """Write *data* as the settings file of a run into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "settings.yaml"
    path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return path


# ============================================================================
# Running
# ============================================================================
def _files_under(dirs: Sequence[Path]) -> set[Path]:
    found: set[Path] = set()
    for directory in dirs:
        if directory.is_dir():
            found.update(path for path in directory.rglob("*") if path.is_file())
    return found


def _last_json_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _no_sleep(_seconds: float) -> None:
    """Stand-in for ``time.sleep`` that returns at once."""


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Stand-in for ``asyncio.sleep`` that only yields to the event loop.

    The store's lease heartbeat keeps its real interval, so it does not spin.
    """
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _ASYNC_SLEEP(delay, result)
    await _ASYNC_SLEEP(0)
    return result


def _absolute(value: Any, root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def run_tool(
    command: Command,
    *,
    llm: FakeLLM,
    root: Path,
    config_dir: Path | None = None,
    config: Mapping[str, Mapping[str, Any]] | None = None,
    env: Mapping[str, str | None] | None = None,
    # transcribe
    input: Path | str | None = None,
    output: Path | str | None = None,
    type: str | None = None,
    method: str | None = None,
    auto: bool = False,
    sync_fallback: bool = False,
    schema: str | None = None,
    context: Path | str | None = None,
    fallback_context: Path | str | None = None,
    context_image: Path | str | None = None,
    reasoning_effort: str | None = None,
    model_verbosity: str | None = None,
    max_output_tokens: int | None = None,
    service_tier: str | None = None,
    response_format: ResponseFormat | None = None,
    endpoint: str | None = None,
    resume: bool = False,
    force: bool = False,
    files: Sequence[str] | None = None,
    recursive: bool = False,
    output_mode: str | None = None,
    pages: str | None = None,
    dry_run: bool = False,
    retry_errors: bool = False,
    keep_working_files: bool = False,
    # transcribe and repair
    model: str | None = None,
    provider: str | None = None,
    batch: bool = False,
    # transcribe and check_batches
    output_format: str | None = None,
    # repair
    transcription: Path | str | None = None,
    errors_only: bool = False,
    not_possible: bool = False,
    no_text: bool = False,
    all_failures: bool = False,
    indices: str | None = None,
    # check_batches
    directory: Path | str | None = None,
    no_diagnostics: bool = False,
    # cancel_batches
    batch_ids: Sequence[str] | None = None,
) -> RunResult:
    """Run one ChronoTranscriber command in process, as a fresh CLI call would.

    This is the only function that knows how the tool is invoked. The
    keywords keep the names of the commands the cases were recorded with
    (``type="pdfs"``, ``method="gpt"``, ``transcription=``); :func:`build_argv`
    translates them into ``chronotranscriber`` arguments, and a keyword the
    command lacks raises ``TypeError``. ``--json`` and ``--settings`` are
    always passed. Paths stay as given, and the working directory is *root*,
    so relative paths resolve against it.

    *config* maps override groups (``CONFIG_FILES``) to the values a case
    changes; each translates to a flag or a settings value, written with the
    adapter's settings into *config_dir* (``root.parent / "config"`` by
    default), and an override without one raises ``ValueError``. The dummy
    provider keys in ``DUMMY_KEYS`` are set, then *env* (a None value
    removes a variable). During the run ``time.sleep`` returns at once and
    ``asyncio.sleep`` only yields; neither shapes the outputs. The fake LLM
    must already be installed (see ``conftest.fake_llm``); its records are
    reset first.
    """
    keywords = {
        name: value
        for name, value in locals().items()
        if name not in {"command", "llm", "root", "config_dir", "config", "env"}
    }
    command_argv, settings_data = build_argv(command, keywords, config)
    configs_dir = config_dir or root.parent / "config"
    settings_path = write_settings(configs_dir, settings_data)
    argv = [*command_argv, "--json", "--settings", str(settings_path)]
    beside_input = "--beside-input" in command_argv

    out_dir = _absolute(output if command == "transcribe" else directory, root)
    anchor_value = {"transcribe": input, "repair": transcription}.get(command)
    anchor = _absolute(anchor_value, root)
    scan = [path for path in (anchor.parent if anchor else None, out_dir) if path]
    scan_dirs = tuple(dict.fromkeys(scan or [root]))
    before = _files_under(scan_dirs)

    llm.reset_records()
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        for name, value in {**DUMMY_KEYS, **dict(env or {})}.items():
            if value is None:
                mp.delenv(name, raising=False)
            else:
                mp.setenv(name, value)
        mp.chdir(root)
        mp.setattr(time, "sleep", _no_sleep)
        mp.setattr(asyncio, "sleep", _yield_only)
        from chronotranscriber import __main__ as entry

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code: Any = entry.main(argv)
            except SystemExit as exc:
                code = exc.code
    exit_code = code if isinstance(code, int) else (0 if code is None else 1)

    created = sorted(_files_under(scan_dirs) - before)
    return RunResult(
        command=command,
        argv=argv,
        exit_code=exit_code,
        summary=_last_json_line(stdout.getvalue()),
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        root=root,
        config_dir=configs_dir,
        output_dir=out_dir,
        scan_dirs=scan_dirs,
        created=created,
        llm=llm,
        calls=llm.sorted_calls(),
        models=llm.models,
        output_format=output_format or "txt",
        layout=output_mode or "hash",
        input_root=anchor if command == "transcribe" else None,
        beside_input=beside_input,
    )


RUN_TOOL_KEYWORDS: Mapping[str, frozenset[str]] = _KEYWORDS
"""Keywords of ``run_tool`` that each command accepts."""

__all__ = [
    "CONFIG_FILES",
    "CUSTOM_BASE_URL",
    "CUSTOM_ENDPOINT",
    "CUSTOM_MODEL",
    "RUN_TOOL_KEYWORDS",
    "ItemPaths",
    "ResponseFormat",
    "RunResult",
    "build_argv",
    "custom_endpoint",
    "deep_merge",
    "item_paths",
    "run_tool",
    "write_settings",
]
