"""The repair command: plan, then startup, then the repair work, and exit codes.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_repair_command.py -q
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.cli.repair as repair_cli
from chronotranscriber import __main__ as entry
from chronotranscriber import ext
from chronotranscriber.batch.repair import RepairResult, RepairSet
from chronotranscriber.settings import Settings, SettingsError, load
from chronotranscriber.spec import CommandSpec
from tests.batch.test_repair_store import store_document
from tests.cli.conftest import json_lines

ZERO_USAGE = {
    "total": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
}


def _pages(repairs: RepairSet) -> int:
    return sum(len(plan.failures) for plan in repairs.plans)


class Timeline:
    """Records the plan, startup and run steps of one repair command."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, refuse: bool = False
    ) -> None:
        self.steps: list[str] = []
        self.refuse = refuse
        real_plan = repair_cli.plan_repairs

        def plan(spec: Any, settings: Settings, **kwargs: Any) -> RepairSet:
            repairs = real_plan(spec, settings, **kwargs)
            self.steps.append(f"plan of {_pages(repairs)} page(s)")
            return repairs

        async def run(repairs: RepairSet, events: Any) -> RepairResult:
            self.steps.append("run")
            return RepairResult(repaired=_pages(repairs))

        monkeypatch.setattr(repair_cli, "plan_repairs", plan)
        monkeypatch.setattr(repair_cli, "run_repairs", run)
        monkeypatch.setattr(ext, "startup", self.startup)

    def startup(
        self, spec: CommandSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        assert isinstance(settings, Settings)
        assert args.command == "repair"
        self.steps.append("startup")
        if self.refuse:
            raise SettingsError("extension refused the repair")


def transcription(tmp_path: Path, *, clean: bool = False) -> Path:
    """Store a document with one failed page (none if *clean*); return its file."""
    outcomes = ["ok", "ok"] if clean else ["ok", "failed"]
    return asyncio.run(store_document(tmp_path, outcomes)).output


def repair(path: Path, *extra: str) -> int:
    return entry.main(["repair", "--input", str(path), "--json", *extra])


def one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    (payload,) = json_lines(captured.out)
    assert len(captured.out.splitlines()) == 1
    return payload, captured.err


def contents(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_a_dry_run_without_the_key_warns_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    timeline = Timeline(monkeypatch)
    path = transcription(tmp_path)
    before = contents(tmp_path)

    assert repair(path, "--dry-run") == 0

    payload, err = one_line(capsys)
    assert payload == {
        "dry_run": True,
        "command": "repair",
        "repaired": 0,
        "failed": 0,
        "usage": ZERO_USAGE,
    }
    assert "[WARNING]" in err and "OPENAI_API_KEY" in err
    assert "[DRY RUN] 1 page(s) would be repaired." in err
    assert contents(tmp_path) == before
    assert timeline.steps == ["plan of 1 page(s)", "startup"]


@pytest.mark.usefixtures("key")
def test_startup_runs_after_the_failure_scan_and_before_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch)

    assert repair(transcription(tmp_path)) == 0

    assert timeline.steps == ["plan of 1 page(s)", "startup", "run"]
    payload, _err = one_line(capsys)
    assert (payload["repaired"], payload["failed"]) == (1, 0)


@pytest.mark.usefixtures("key")
def test_a_dry_run_calls_startup_but_not_the_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch)

    assert repair(transcription(tmp_path), "--dry-run") == 0

    assert timeline.steps == ["plan of 1 page(s)", "startup"]
    assert "[DRY RUN] 1 page(s) would be repaired." in capsys.readouterr().err
    assert not load().log_file().parent.exists()


def test_a_clean_file_does_not_call_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    timeline = Timeline(monkeypatch)

    assert repair(transcription(tmp_path, clean=True)) == 0

    assert timeline.steps == ["plan of 0 page(s)"]
    payload, err = one_line(capsys)
    assert (payload["repaired"], payload["failed"]) == (0, 0)
    assert "No failed pages detected" in err


def test_a_dry_run_of_a_clean_file_calls_startup_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    timeline = Timeline(monkeypatch)
    path = transcription(tmp_path, clean=True)
    before = contents(tmp_path)

    assert repair(path, "--dry-run") == 0

    assert contents(tmp_path) == before
    assert timeline.steps == ["plan of 0 page(s)", "startup"]
    payload, err = one_line(capsys)
    assert (payload["repaired"], payload["failed"]) == (0, 0)
    assert "No failed pages detected" in err


def test_a_plan_error_exits_2_before_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    timeline = Timeline(monkeypatch)

    assert repair(transcription(tmp_path)) == 2

    payload, err = one_line(capsys)
    assert payload["command"] == "repair"
    assert "Invalid arguments" in err
    assert timeline.steps == []
    assert not (tmp_path / "out" / "repairs").exists()


def test_a_versionless_database_exits_2_as_not_a_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = Timeline(monkeypatch)
    path = tmp_path / "doc.txt"
    path.write_text("p0.jpg: [transcription error]\n", encoding="utf-8")
    connection = sqlite3.connect(tmp_path / "chronotranscriber.sqlite")
    try:
        with connection:
            connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE pages (name TEXT, idx INTEGER)")
    finally:
        connection.close()
    before = contents(tmp_path)

    assert repair(path) == 2

    payload, err = one_line(capsys)
    assert payload["command"] == "repair"
    assert "not a chronotranscriber store" in " ".join(err.split())
    assert "migrate" not in err
    assert contents(tmp_path) == before
    assert timeline.steps == []


@pytest.mark.usefixtures("key")
def test_an_error_of_the_repair_work_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    Timeline(monkeypatch)

    async def broken(repairs: RepairSet, events: Any) -> RepairResult:
        raise ValueError("page image unreadable")

    monkeypatch.setattr(repair_cli, "run_repairs", broken)

    assert repair(transcription(tmp_path)) == 1

    payload, err = one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "repair",
        "repaired": 0,
        "failed": 0,
        "usage": ZERO_USAGE,
    }
    assert "Unexpected error: page image unreadable" in err
    assert "Invalid arguments" not in err


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize("dry_run", [False, True])
def test_a_settings_error_from_startup_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    timeline = Timeline(monkeypatch, refuse=True)
    extra = ("--dry-run",) if dry_run else ()

    assert repair(transcription(tmp_path), *extra) == 2

    payload, err = one_line(capsys)
    assert payload["dry_run"] is dry_run
    assert "extension refused the repair" in err
    assert timeline.steps == ["plan of 1 page(s)", "startup"]
