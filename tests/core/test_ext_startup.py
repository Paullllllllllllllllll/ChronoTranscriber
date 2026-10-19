"""The extension seam as the CLI calls it after planning.

``ext.startup`` receives the spec, the settings and the parsed arguments,
dry runs included, and is not called when nothing is left to process; a
SettingsError it raises is a configuration error (exit 2, one JSON line).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.cli.transcribe as transcribe_cli
from chronotranscriber import __main__ as entry
from chronotranscriber import ext
from chronotranscriber.settings import Settings, SettingsError
from chronotranscriber.spec import RunSpec

MakePdf = Callable[..., Path]


class _Startup:
    """Stand-in for ``ext.startup`` that records its calls and may raise."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[RunSpec, Settings, argparse.Namespace]] = []

    def __call__(
        self, spec: RunSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        self.calls.append((spec, settings, args))
        if self.error is not None:
            raise self.error


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    def make(name: str = "doc.pdf") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
        document.save(path)
        document.close()
        return path

    return make


def _install(monkeypatch: pytest.MonkeyPatch, startup: _Startup) -> _Startup:
    monkeypatch.setattr(ext, "startup", startup)
    return startup


def _no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("run must not be called")

    monkeypatch.setattr(transcribe_cli, "run", run)


def _native(pdf: Path, out: Path, *extra: str) -> list[str]:
    return [
        "transcribe",
        "--input",
        str(pdf),
        "--type",
        "pdf",
        "--method",
        "native",
        "--output",
        str(out),
        *extra,
    ]


def test_the_cli_passes_the_spec_the_settings_and_the_parsed_arguments(
    make_pdf: MakePdf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = _install(monkeypatch, _Startup())
    pdf = make_pdf()

    assert entry.main(_native(pdf, tmp_path / "out")) == 0

    ((spec, settings, args),) = startup.calls
    assert isinstance(spec, RunSpec)
    assert spec.input.path == pdf
    assert isinstance(settings, Settings)
    assert args.command == "transcribe"


def test_startup_is_called_once_for_a_dry_run(
    make_pdf: MakePdf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = _install(monkeypatch, _Startup())
    _no_run(monkeypatch)

    assert entry.main(_native(make_pdf(), tmp_path / "out", "--dry-run")) == 0

    ((spec, _settings, _args),) = startup.calls
    assert spec.run.dry_run


def test_startup_is_not_called_when_nothing_is_left_to_process(
    make_pdf: MakePdf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf()
    assert entry.main(_native(pdf, tmp_path / "out")) == 0
    startup = _install(monkeypatch, _Startup())

    assert entry.main(_native(pdf, tmp_path / "out")) == 0

    assert startup.calls == []


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_settings_error_from_startup_exits_2_with_one_json_line(
    make_pdf: MakePdf,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    _install(monkeypatch, _Startup(SettingsError("extension block is invalid")))
    _no_run(monkeypatch)
    out = tmp_path / "out"
    extra = ("--json", "--dry-run") if dry_run else ("--json",)

    assert entry.main(_native(make_pdf(), out, *extra)) == 2

    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["items_complete"] == 0
    assert json.loads(line)["dry_run"] is dry_run
    assert "extension block is invalid" in captured.err
    assert not out.exists()
