"""The agent contract of every subcommand: exit codes and the JSON line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.cli.batch as batch_cli
import chronotranscriber.cli.repair as repair_cli
import chronotranscriber.cli.transcribe as transcribe_cli
from chronotranscriber import __main__ as entry
from chronotranscriber import ext
from chronotranscriber.batch.cancel import CancelOutcome, CancelReport
from chronotranscriber.batch.check import CheckResult, CheckStats
from chronotranscriber.batch.repair import RepairResult
from chronotranscriber.common.heartbeat import ENV_VAR
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.settings import Settings, load
from chronotranscriber.spec import CommandSpec
from tests.batch.test_repair_store import store_document
from tests.cli.conftest import MakePdf, json_lines

TRANSCRIBE_KEYS = {
    "dry_run",
    "items_total",
    "items_complete",
    "items_failed",
    "items_skipped",
    "items_blocked",
    "items_submitted",
    "outputs",
    "usage",
}
PLAN_KEYS = {"to_process", "skipped", "unstored", "blocked", "refused", "spec"}
ZERO_USAGE = {
    "total": {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
}


def _transcribe(*argv: str) -> int:
    return entry.main(["transcribe", *argv])


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _usage_error(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(make_pdf()), "--concurrency", "zero"]


def _unknown_flag(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(make_pdf()), "--bogus"]


def _missing_settings(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(make_pdf()), "--settings", str(tmp_path / "nope.yaml")]


def _unknown_endpoint(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(make_pdf()), "--endpoint", "missing"]


def _missing_input(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--input", str(tmp_path / "missing.pdf"), "--type", "pdf"]


def _no_input(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return ["--method", "native"]


def _dry_run(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    return [
        "--input",
        str(make_pdf()),
        "--method",
        "tesseract",
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
    ]


def _native(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    pdf = make_pdf(pages=2)
    return [
        "--input",
        str(pdf),
        "--type",
        "pdf",
        "--method",
        "native",
        "--output",
        str(tmp_path / "out"),
    ]


@pytest.mark.parametrize(
    ("setup", "code", "total"),
    [
        pytest.param(_usage_error, 2, 0, id="usage_error"),
        pytest.param(_unknown_flag, 2, 0, id="unknown_flag"),
        pytest.param(_missing_settings, 2, 0, id="missing_settings"),
        pytest.param(_unknown_endpoint, 2, 0, id="unknown_endpoint"),
        pytest.param(_missing_input, 2, 0, id="missing_input"),
        pytest.param(_no_input, 2, 0, id="no_input"),
        pytest.param(_dry_run, 0, 1, id="dry_run"),
        pytest.param(_native, 0, 1, id="native_run"),
    ],
)
def test_every_exit_writes_one_json_line(
    tmp_path: Path,
    make_pdf: MakePdf,
    capsys: pytest.CaptureFixture[str],
    setup: Callable[[Path, MakePdf], list[str]],
    code: int,
    total: int,
) -> None:
    argv = setup(tmp_path, make_pdf)

    assert _transcribe(*argv, "--json") == code

    payload, err = _one_line(capsys)
    dry_run = "--dry-run" in argv
    assert set(payload) == TRANSCRIBE_KEYS | (PLAN_KEYS if dry_run else set())
    assert payload["dry_run"] is dry_run
    assert payload["items_total"] == total
    if code:
        assert "[ERROR]" in err


def test_a_run_reports_its_outputs_and_the_database(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _transcribe(*_native(tmp_path, make_pdf), "--json") == 0

    payload, _err = _one_line(capsys)
    database = tmp_path / "out" / "chronotranscriber.sqlite"
    assert (payload["items_complete"], payload["items_skipped"]) == (1, 0)
    assert payload["outputs"][-1] == str(database.resolve())
    assert [Path(path).suffix for path in payload["outputs"]] == [".txt", ".sqlite"]
    assert payload["usage"] == ZERO_USAGE


def test_a_dry_run_reports_the_plan_and_the_effective_spec(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _transcribe(*_dry_run(tmp_path, make_pdf), "--json") == 0

    payload, _err = _one_line(capsys)
    (item,) = payload["to_process"]
    assert (item["method"], item["state"]) == ("tesseract", "none")
    assert payload["skipped"] == []
    assert payload["spec"]["method.name"] == {"value": "tesseract", "source": "flag"}
    assert payload["spec"]["output.formats"] == {"value": ["txt"], "source": "default"}


def test_a_usage_error_writes_the_usage_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _transcribe("--service-tier", "bogus", "--json") == 2
    payload, err = _one_line(capsys)
    assert set(payload) == TRANSCRIBE_KEYS
    assert "usage: chronotranscriber transcribe" in err
    assert "'bogus' is not one of" in err


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_interrupt_and_crash_write_one_json_line(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(transcribe_cli, "plan", _raise)

    assert _transcribe("--input", str(make_pdf()), "--json") == code
    payload, err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "items_total": 0,
        "items_complete": 0,
        "items_failed": 0,
        "items_skipped": 0,
        "items_blocked": 0,
        "items_submitted": 0,
        "outputs": [],
        "usage": ZERO_USAGE,
    }
    assert "[ERROR]" in err


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_an_interrupted_run_reports_the_tokens_spent(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    async def spend(*_args: Any, **_kwargs: Any) -> Any:
        CallEnv.create().record("transcription", Usage(5, 1, 6))
        raise failure

    monkeypatch.setattr(transcribe_cli, "run", spend)

    assert _transcribe(*_native(tmp_path, make_pdf), "--json") == code
    payload, _err = _one_line(capsys)
    assert payload["usage"]["transcription"]["total_tokens"] == 6
    assert payload["usage"]["total"]["total_tokens"] == 6


def test_a_blocked_item_exits_1_and_is_named_with_its_lease(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _native(tmp_path, make_pdf)
    assert _transcribe(*argv) == 0
    capsys.readouterr()
    connection = sqlite3.connect(tmp_path / "out" / "chronotranscriber.sqlite")
    try:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'other-run', lease_until = ?",
            (time.time() + 600,),
        )
        connection.commit()
    finally:
        connection.close()

    assert _transcribe(*argv, "--json") == 1

    payload, err = _one_line(capsys)
    assert (payload["items_blocked"], payload["items_skipped"]) == (1, 1)
    assert payload["items_failed"] == 0
    assert "Not processed (blocked): 'doc.pdf': leased by run other-run until" in err
    assert "Processing incomplete: 1 item(s) leased by another run" in err
    assert "Processing complete!" not in err


def test_an_item_pages_left_partial_counts_neither_complete_nor_failed(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _native(tmp_path, make_pdf)

    assert _transcribe(*argv, "--pages", "1", "--json") == 0
    payload, err = _one_line(capsys)
    assert (payload["items_complete"], payload["items_failed"]) == (0, 0)
    assert "1 item(s) left partial by --pages." in err

    assert _transcribe(*argv, "--json") == 0
    payload, err = _one_line(capsys)
    assert (payload["items_complete"], payload["items_failed"]) == (1, 0)
    assert "left partial" not in err


def test_blocked_items_of_one_name_keep_their_own_reasons(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    for folder in ("a", "b"):
        make_pdf(f"in/{folder}/doc.pdf")
    argv = [
        "--input",
        str(tmp_path / "in"),
        "--type",
        "pdf",
        "--method",
        "native",
        "--recursive",
        "--all",
        "--output",
        str(tmp_path / "out"),
    ]
    assert _transcribe(*argv) == 0
    capsys.readouterr()
    connection = sqlite3.connect(tmp_path / "out" / "chronotranscriber.sqlite")
    try:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'run-' || doc_id, lease_until = ?",
            (time.time() + 600,),
        )
        connection.commit()
    finally:
        connection.close()

    assert _transcribe(*argv, "--json") == 1

    payload, err = _one_line(capsys)
    assert payload["items_blocked"] == 2
    for run_id in ("run-1", "run-2"):
        assert f"'doc.pdf': leased by run {run_id} until" in err


def test_without_json_stdout_stays_empty(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _transcribe(*_native(tmp_path, make_pdf)) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Processing complete" in captured.err


def test_a_run_leaves_its_last_item_finished_in_the_heartbeat_file(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "job.heartbeat.json"
    monkeypatch.setenv(ENV_VAR, str(path))

    assert _transcribe(*_native(tmp_path, make_pdf)) == 0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["unit_label"] == "pages processed, item 1/1"
    assert data["units_done"] == data["units_total"]
    assert data["eta_seconds"] == 0


def test_a_dry_run_writes_nothing(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf = make_pdf("in/doc.pdf")
    out = tmp_path / "out"

    argv = ["--input", str(pdf), "--type", "pdf", "--output", str(out), "--dry-run"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert _transcribe(*argv) == 0
    assert not out.exists()
    assert sorted(path.name for path in pdf.parent.iterdir()) == ["doc.pdf"]
    assert "[DRY RUN] 1 discovered" in capsys.readouterr().err


@pytest.mark.parametrize("dry_run", [True, False])
def test_only_a_run_that_is_not_dry_opens_the_log_file(
    tmp_path: Path, make_pdf: MakePdf, dry_run: bool
) -> None:
    extra = ["--dry-run"] if dry_run else []
    assert _transcribe(*_native(tmp_path, make_pdf), *extra) == 0
    assert load().log_file().parent.is_dir() is not dry_run


def test_relative_paths_resolve_against_the_current_directory(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_pdf("work/doc.pdf")
    monkeypatch.chdir(tmp_path / "work")

    code = _transcribe(
        "--input", "doc.pdf", "--type", "pdf", "--method", "native", "--output", "out"
    )

    assert code == 0
    assert list((tmp_path / "work" / "out").rglob("doc.txt"))


class _Timeline:
    """Records ext.startup calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[CommandSpec] = []
        monkeypatch.setattr(ext, "startup", self.startup)

    def startup(
        self, spec: CommandSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        assert isinstance(settings, Settings)
        assert isinstance(args, argparse.Namespace)
        self.calls.append(spec)


def test_startup_runs_before_processing(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _Timeline(monkeypatch)
    order: list[str] = []
    real = transcribe_cli.run

    async def process(*args: Any, **kwargs: Any) -> Any:
        order.append(f"process after {len(timeline.calls)} startup call(s)")
        return await real(*args, **kwargs)

    monkeypatch.setattr(transcribe_cli, "run", process)

    assert _transcribe(*_native(tmp_path, make_pdf)) == 0
    assert order == ["process after 1 startup call(s)"]


@pytest.mark.parametrize(("failed", "code"), [(0, 0), (1, 1)])
def test_repair_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed: int,
    code: int,
) -> None:
    async def repair(repairs: Any, events: Any) -> RepairResult:
        return RepairResult(
            repaired=3,
            failed=failed,
            selected=3 + failed,
            usage={"repair": Usage(5, 1, 6)},
        )

    monkeypatch.setattr(repair_cli, "run_repairs", repair)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    path = asyncio.run(store_document(tmp_path, ["ok", "failed"])).output

    argv = ["repair", "--input", str(path), "--json"]
    assert entry.main(argv) == code
    payload, _err = _one_line(capsys)
    tokens = Usage(5, 1, 6).as_dict()
    assert payload == {
        "dry_run": False,
        "command": "repair",
        "repaired": 3,
        "failed": failed,
        "usage": {"repair": tokens, "total": tokens},
    }


def test_repair_of_a_missing_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["repair", "--input", str(tmp_path / "missing.txt"), "--json"]
    assert entry.main(argv) == 2
    payload, err = _one_line(capsys)
    assert payload["command"] == "repair"
    assert "does not exist" in err


@pytest.mark.parametrize(
    ("stats", "code"),
    [
        (CheckStats(finalized=2, pending=1, failed=0), 0),
        (CheckStats(finalized=0, pending=0, failed=2), 1),
    ],
)
def test_batch_check_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stats: CheckStats,
    code: int,
) -> None:
    seen: list[tuple[Path, tuple[str, ...]]] = []

    async def check(
        directory: Path, settings: Any, *, formats: Any, batch_ids: Any, args: Any
    ) -> CheckResult:
        seen.append((directory, tuple(formats)))
        return CheckResult(stats=stats)

    monkeypatch.setattr(batch_cli, "check_batches", check)

    argv = ["batch", "check", "--input", str(tmp_path), "--formats", "md", "--json"]
    assert entry.main(argv) == code
    payload, _err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "check_batches",
        "finalized": stats.finalized,
        "pending": stats.pending,
        "failed": stats.failed,
        "unresolved": 0,
        "submitted": 0,
        "complete": 0,
        "failed_pages": 0,
        "outputs": [],
        "usage": ZERO_USAGE,
    }
    assert seen == [(tmp_path.resolve(), ("md",))]


def test_batch_cancel_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def cancel(batch_ids: Any, settings: Any, *, args: Any) -> CancelReport:
        return CancelReport(
            outcomes=[
                CancelOutcome(batch_id, "openai", batch_id == "b1")
                for batch_id in batch_ids
            ]
        )

    monkeypatch.setattr(batch_cli, "cancel_batches", cancel)

    assert entry.main(["batch", "cancel", "--batch-ids", "b1,b2", "--json"]) == 1
    payload, _err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "cancel_batches",
        "cancelled": 1,
        "failed": 1,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["batch", "cancel", "--json"],
        ["batch", "cancel", "--all", "--batch-ids", "b1", "--json"],
    ],
)
def test_batch_cancel_needs_one_target(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main(argv) == 2
    payload, _err = _one_line(capsys)
    assert payload == {
        "dry_run": False,
        "command": "cancel_batches",
        "cancelled": 0,
        "failed": 0,
    }
