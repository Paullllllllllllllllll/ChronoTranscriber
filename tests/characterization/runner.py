"""Run the ``chronotranscriber`` CLI in process for characterization tests.

``run_cli`` takes a subcommand, its options by their names in the option
tables of ``chronotranscriber.spec`` and a settings mapping in the layout of
``settings.yaml``; it returns a ``RunResult`` with everything the run produced.
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

from chronotranscriber.common.options import Kind, OptionTable
from tests.characterization.fakes import CUSTOM_KEY_ENV, DUMMY_KEYS, FakeLLM

Command = Literal["transcribe", "repair", "batch check", "batch cancel"]
ResponseFormat = Literal["schema", "json", "text"]

CUSTOM_MODEL = "org/fake-model"
CUSTOM_BASE_URL = "https://fake-endpoint.invalid/v1/"
CUSTOM_ENDPOINT = "fake"

RESERVED_OPTIONS = frozenset({"json", "settings"})
"""Options ``run_cli`` always passes itself."""

_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


def option_table(command: Command) -> OptionTable:
    """Return the option table of *command*."""
    from chronotranscriber.spec import (
        BATCH_CANCEL_OPTIONS,
        BATCH_CHECK_OPTIONS,
        OPTIONS,
        REPAIR_OPTIONS,
    )

    tables = {
        "transcribe": OPTIONS,
        "repair": REPAIR_OPTIONS,
        "batch check": BATCH_CHECK_OPTIONS,
        "batch cancel": BATCH_CANCEL_OPTIONS,
    }
    if command not in tables:
        raise ValueError(f"unknown command {command!r}")
    return tables[command]


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
    """Outcome of one ``run_cli`` call.

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
    settings_dir: Path
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


def custom_endpoint_settings(response_format: ResponseFormat) -> dict[str, Any]:
    """Return the ``endpoints`` settings of the fake custom endpoint.

    The endpoint enforces the schema through structured output only when
    *response_format* is ``schema``.
    """
    return {
        "endpoints": {
            CUSTOM_ENDPOINT: {
                "base_url": CUSTOM_BASE_URL,
                "api_key_env": CUSTOM_KEY_ENV,
                "supports_vision": True,
                "supports_schema": response_format == "schema",
            }
        }
    }


def custom_endpoint_options(
    response_format: ResponseFormat, model: str = CUSTOM_MODEL
) -> dict[str, Any]:
    """Return the options that route a run to the fake custom endpoint.

    *response_format* picks the endpoint mode: ``schema`` enforces the schema
    through structured output, ``json`` prompts for the schema, ``text``
    prompts for plain text.
    """
    return {
        "endpoint": CUSTOM_ENDPOINT,
        "model": model,
        "response_format": response_format,
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


def build_argv(command: Command, options: Mapping[str, Any]) -> list[str]:
    """Return the argv that runs *command* with *options*.

    *options* are option names of the command's table; a None or false
    value is left out, and a string given to a list option is a one-item
    list. Two defaults differ from the CLI: ``transcribe`` covers every item
    of the input folder (``--all``) unless ``files`` or ``select`` is given
    and keeps error pages on a rerun (``--no-retry-errors``) unless
    ``retry_errors`` is given; ``batch cancel`` cancels every unfinished
    batch (``--all``) unless ``batch_ids`` is given. The argv lacks
    ``--json`` and ``--settings``.

    Raises:
        TypeError: An option the command does not take, or a reserved one.
    """
    table = option_table(command)
    given = {
        name: value
        for name, value in options.items()
        if value is not None and value is not False
    }
    unknown = sorted((set(given) - set(table.names)) | (set(given) & RESERVED_OPTIONS))
    if unknown:
        raise TypeError(f"{command} has no option for {unknown}")
    for name, value in given.items():
        if table[name].kind is Kind.LIST and isinstance(value, str):
            given[name] = (value,)
    if command == "transcribe":
        if not {"files", "select"} & set(given):
            given.setdefault("all", True)
        given.setdefault("retry_errors", False)
    elif command == "batch cancel":
        if "batch_ids" not in given:
            given.setdefault("all", True)
    return [*command.split(), *table.to_argv(given)]


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


def run_cli(
    command: Command,
    *,
    llm: FakeLLM,
    root: Path,
    settings: Mapping[str, Any] | None = None,
    settings_dir: Path | None = None,
    env: Mapping[str, str | None] | None = None,
    **options: Any,
) -> RunResult:
    """Run one ``chronotranscriber`` command in process, as a fresh CLI call would.

    :func:`build_argv` turns *command* and *options* into the argv; ``--json``
    and ``--settings`` are always passed. *settings* is written as
    ``settings.yaml`` into *settings_dir* (``root.parent / "config"`` by
    default). Paths stay as given, and the working directory is *root*, so
    relative paths resolve against it. The dummy provider keys in
    ``DUMMY_KEYS`` are set, then *env* (a None value removes a variable).
    During the run ``time.sleep`` returns at once and ``asyncio.sleep`` only
    yields; neither shapes the outputs. The fake LLM must already be
    installed (see ``conftest.fake_llm``); its records are reset first.
    """
    command_argv = build_argv(command, options)
    folder = settings_dir or root.parent / "config"
    settings_path = write_settings(folder, settings or {})
    argv = [*command_argv, "--json", "--settings", str(settings_path)]
    beside_input = "--beside-input" in command_argv

    out_option = {"transcribe": "output", "batch check": "input"}.get(command)
    out_dir = _absolute(options.get(out_option) if out_option else None, root)
    anchor_value = options.get("input") if command in ("transcribe", "repair") else None
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

    formats = options.get("formats")
    if isinstance(formats, str):
        formats = (formats,)
    created = sorted(_files_under(scan_dirs) - before)
    return RunResult(
        command=command,
        argv=argv,
        exit_code=exit_code,
        summary=_last_json_line(stdout.getvalue()),
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        root=root,
        settings_dir=folder,
        output_dir=out_dir,
        scan_dirs=scan_dirs,
        created=created,
        llm=llm,
        calls=llm.sorted_calls(),
        models=llm.models,
        output_format=formats[0] if formats else "txt",
        layout=options.get("layout") or "hash",
        input_root=anchor if command == "transcribe" else None,
        beside_input=beside_input,
    )


__all__ = [
    "CUSTOM_BASE_URL",
    "CUSTOM_ENDPOINT",
    "CUSTOM_MODEL",
    "RESERVED_OPTIONS",
    "Command",
    "ItemPaths",
    "ResponseFormat",
    "RunResult",
    "build_argv",
    "custom_endpoint_options",
    "custom_endpoint_settings",
    "deep_merge",
    "item_paths",
    "option_table",
    "run_cli",
    "write_settings",
]
