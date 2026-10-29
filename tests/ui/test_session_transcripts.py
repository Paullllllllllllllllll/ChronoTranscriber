"""Recorded guided sessions, run to the end: LLM PDF, Tesseract images, audio.

Each scripted session goes through the real wizard and app, goes Back at
least once and runs from the review screen. The settings file holds a
defaults block, so the review marks values taken from it. Providers, the
Tesseract engine and the audio runtimes are the characterization fakes; the
image preprocessing runs for real. The golden ``golden/session_<name>.txt``
holds the prompter transcript followed by the run screen and the summary,
with temp paths and durations replaced by placeholders.
``CT_UPDATE_GOLDEN=1`` rewrites it.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import io
import os
import re
import shlex
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from chronotranscriber import ext
from chronotranscriber.cli.parser import build_parser
from chronotranscriber.common.testing import KEEP, ScriptedPrompter, WriteGuard
from chronotranscriber.common.wizard import BACK, command_display
from chronotranscriber.llm import prompts
from chronotranscriber.settings import Settings, load, resolve
from chronotranscriber.spec import OPTIONS, RunSpec
from chronotranscriber.ui import app
from tests.characterization.fakes import DUMMY_KEYS, PROVIDER_CLASSES, FakeLLM
from tests.characterization.fakes_audio import FakeAudio
from tests.characterization.fakes_ocr import FakeTesseract, block_real_tesseract
from tests.characterization.golden import Normalizer
from tests.characterization.inputs import make_image_folder, make_scanned_pdf
from tests.characterization.inputs_audio import make_wav
from tests.characterization.seams import replace

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
UPDATE_ENV = "CT_UPDATE_GOLDEN"
WIDTH = 100
PUBLIC_SCHEMAS = (
    "markdown_transcription_schema.json",
    "plain_text_transcription_schema.json",
    "plain_text_transcription_with_markers_schema.json",
)

_COMMAND_HEADER = "Equivalent command:"
_DURATION_RE = re.compile(r"(Run time: |Waiting )\d+(?:\.\d+)?s\b")
_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Stand-in for ``asyncio.sleep`` that only yields to the event loop.

    The store's lease heartbeat keeps its real interval, so it does not spin.
    """
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _ASYNC_SLEEP(delay, result)
    await _ASYNC_SLEEP(0)
    return result


@dataclasses.dataclass(frozen=True)
class Case:
    """One recorded session: its input, fakes, settings defaults and answers."""

    name: str
    defaults: Mapping[str, Any]
    make_input: Callable[[Path], Path]
    install: Callable[[pytest.MonkeyPatch, Path], None]
    answers: Callable[[Path, Path], list[Any]]


def _install_llm(monkeypatch: pytest.MonkeyPatch, _root: Path) -> None:
    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))


def _install_tesseract(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    block_real_tesseract(monkeypatch, root)
    FakeTesseract().install(monkeypatch)


def _install_audio(monkeypatch: pytest.MonkeyPatch, _root: Path) -> None:
    FakeAudio().install(monkeypatch)


def _llm_pdf_answers(source: Path, out: Path) -> list[Any]:
    return [
        KEEP,  # task
        str(source),
        "pdf",
        "llm",
        "folder",
        str(out),
        BACK,  # from the layout back to the output folder
        KEEP,  # output folder
        KEEP,  # layout
        ["txt", "md"],
        *[KEEP] * 4,  # provider, model, reasoning, model parameters
        *[KEEP] * 4,  # response format, schema, prompt, context
        KEEP,  # items without a sidecar
        KEEP,  # context image
        KEEP,  # page images
        KEEP,  # decision pre-pass
        KEEP,  # pages
        KEEP,  # submission
        KEEP,  # run options
        KEEP,  # concurrency
        "run",
    ]


def _tesseract_answers(source: Path, out: Path) -> list[Any]:
    return [
        KEEP,  # task
        str(source),
        KEEP,  # subfolders
        "images",
        "tesseract",
        "folder",
        str(out),
        KEEP,  # layout
        KEEP,  # formats
        BACK,  # from the Tesseract configuration back to the output step
        KEEP,  # output choice
        KEEP,  # output folder
        KEEP,  # layout
        ["txt", "md"],
        KEEP,  # Tesseract configuration
        KEEP,  # preprocessing
        KEEP,  # pages
        KEEP,  # run options
        KEEP,  # concurrency
        "run",
    ]


def _audio_answers(source: Path, out: Path) -> list[Any]:
    return [
        KEEP,  # task
        str(source),
        "audio",
        "whisper",
        "folder",
        str(out),
        KEEP,  # layout
        ["txt"],
        BACK,  # from the Whisper model back to the output step
        KEEP,  # output choice
        KEEP,  # output folder
        KEEP,  # layout
        ["txt", "md"],
        KEEP,  # Whisper model
        KEEP,  # language
        KEEP,  # chunk length
        KEEP,  # pages
        KEEP,  # run options
        KEEP,  # concurrency
        "run",
    ]


CASES = (
    Case(
        "llm-pdf",
        {"model": "gpt-5.6-terra", "service_tier": "priority"},
        lambda root: make_scanned_pdf(root / "inputs" / "book.pdf", pages=3),
        _install_llm,
        _llm_pdf_answers,
    ),
    Case(
        "tesseract-images",
        {"ocr_preprocess": "sauvola", "formats": ["md"]},
        lambda root: make_image_folder(root / "inputs" / "scans", pages=3),
        _install_tesseract,
        _tesseract_answers,
    ),
    Case(
        "audio",
        {"whisper_model": "small", "language": "de"},
        lambda root: make_wav(root / "inputs" / "interview.wav", seconds=3),
        _install_audio,
        _audio_answers,
    ),
)


def _public_schemas(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Show the wizard only the public schemas, whatever else ``schemas/`` holds."""
    folder.mkdir(parents=True)
    for name in PUBLIC_SCHEMAS:
        shutil.copyfile(prompts.SCHEMAS_DIR / name, folder / name)
    monkeypatch.setattr(prompts, "SCHEMAS_DIR", folder)


def _settings_file(folder: Path, defaults: Mapping[str, Any]) -> Path:
    """Write a settings file with a defaults block."""
    path = folder / "settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"defaults": dict(defaults)}
    path.write_text(yaml.safe_dump(data), encoding="utf-8", newline="\n")
    return path


def _command_block(lines: Sequence[str]) -> tuple[int, int]:
    """Return the slice of the last printed equivalent command."""
    start = len(lines) - 1 - lines[::-1].index(_COMMAND_HEADER) + 1
    end = start
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return start, end


_POWERSHELL_WORD = re.compile(r"'((?:[^']|'')*)'|(\S+)")


def _split(line: str) -> list[str]:
    """Split a command line printed for the current platform into words.

    On Windows the words are quoted for PowerShell: bare, or in single quotes
    with each inner single quote doubled.
    """
    if os.name != "nt":
        return shlex.split(line)
    return [
        bare if quoted is None else quoted.replace("''", "'")
        for quoted, bare in (
            (match.group(1), match.group(2))
            for match in _POWERSHELL_WORD.finditer(line)
        )
    ]


def _spec_of(words: Sequence[str]) -> RunSpec:
    """Parse and resolve a printed command as ``chronotranscriber`` would."""
    assert list(words[:2]) == ["chronotranscriber", "transcribe"]
    flags = OPTIONS.values(build_parser().parse_args(list(words[1:])))
    settings = load(flags.get("settings"), blocks=ext.settings_blocks())
    return resolve(settings, flags).spec


def _normalize(lines: list[str], words: Sequence[str], console: str, root: Path) -> str:
    """Replace the temp path and durations; re-wrap the command.

    The command is wrapped again from its normalized words, quoted for
    Windows, so its line breaks do not depend on the temp path or the OS.
    """
    norm = Normalizer({root: "<tmp>"})
    start, end = _command_block(lines)
    shown = command_display(
        [norm.text(word) for word in words], width=WIDTH, windows=True
    )
    lines = [*lines[:start], *shown, *lines[end:]]
    text = norm.text("\n".join(lines) + "\n" + console)
    text = _DURATION_RE.sub(r"\1<duration>", text)
    return "".join(line.rstrip() + "\n" for line in text.splitlines())


def _check_golden(name: str, actual: str, write_guard: WriteGuard) -> None:
    golden = GOLDEN_DIR / f"session_{name}.txt"
    if os.environ.get(UPDATE_ENV) == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual, encoding="utf-8", newline="\n")
        write_guard.consume(golden.parent)
        return
    if not golden.is_file():
        pytest.fail(f"No golden {golden.name}; run with {UPDATE_ENV}=1 to create.")
    expected = golden.read_bytes().decode("utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{golden.name}",
                tofile="session",
            )
        )
        pytest.fail(
            f"Session transcript changed (set {UPDATE_ENV}=1 to accept):\n{diff}"
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_recorded_session(
    case: Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_guard: WriteGuard,
) -> None:
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(asyncio, "sleep", _yield_only)
    case.install(monkeypatch, tmp_path)
    _public_schemas(tmp_path / "schemas", monkeypatch)
    source = case.make_input(tmp_path)
    output = tmp_path / "out"
    settings = _settings_file(tmp_path / "config", case.defaults)
    started: list[RunSpec] = []
    real_startup = ext.startup

    def startup(spec: RunSpec, *, settings: Settings, args: argparse.Namespace) -> None:
        started.append(spec)
        real_startup(spec, settings=settings, args=args)

    monkeypatch.setattr(ext, "startup", startup)
    monkeypatch.chdir(tmp_path)

    console = Console(
        file=io.StringIO(),
        width=WIDTH,
        color_system=None,
        force_terminal=False,
        legacy_windows=False,
        soft_wrap=True,
    )
    prompter = ScriptedPrompter(case.answers(source, output))
    with prompter:
        code = app.run_app(
            settings, prompter=prompter, console=console, require_tty=False
        )
    screen = console.file
    assert isinstance(screen, io.StringIO)
    assert code == 0, prompter.transcript + screen.getvalue()

    lines = prompter.transcript.splitlines()
    start, end = _command_block(lines)
    words = _split(" ".join(line.strip() for line in lines[start:end]))
    assert lines[start:end] == command_display(words, width=WIDTH)
    assert len(started) == 1
    assert _spec_of(words) == started[0]

    actual = _normalize(lines, words, screen.getvalue(), tmp_path)
    _check_golden(case.name, actual, write_guard)
