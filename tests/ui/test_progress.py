"""Run screen: progress events, summary table and plan table."""

from __future__ import annotations

import dataclasses
import io
import re
import sys
from pathlib import Path

from rich.console import Console, RenderableType

import chronotranscriber.run as core
from chronotranscriber.common.store import UnitKey, UnitRow
from chronotranscriber.common.usage import Usage
from chronotranscriber.events import (
    ItemFinished,
    ItemStarted,
    PageDone,
    RunEvents,
    Waiting,
)
from chronotranscriber.output.store import Classification, ItemState
from chronotranscriber.sources import SourceItem
from chronotranscriber.ui.progress import RichProgress, plan_table, summary_table
from tests.conftest import make_spec

ESC = "\x1b"


def _console() -> Console:
    return Console(
        file=io.StringIO(), width=100, color_system=None, force_terminal=False
    )


def _output(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def _render(renderable: RenderableType) -> str:
    console = _console()
    console.print(renderable)
    return _output(console)


def _drive(events: RunEvents) -> None:
    events.item_started(ItemStarted("alpha.pdf", "pdf", 1, 3))
    events.page_done(PageDone("alpha.pdf", 2, "ok", 1, 2, already_complete=2))
    events.waiting(Waiting("rate limit", 30.0))
    events.page_done(PageDone("alpha.pdf", 3, "ok", 2, 2, already_complete=2))
    events.item_finished(
        ItemFinished("alpha.pdf", 1, 3, True, ("/out/alpha.txt", "/out/alpha.md"))
    )
    events.item_started(ItemStarted("scans", "images", 2, 3))
    events.warning("model refused image 1\nretrying later")
    events.page_done(PageDone("scans", 0, "failed", 1, 2, unit="image"))
    events.page_done(PageDone("scans", 1, "ok", 2, 2, unit="image"))
    events.item_finished(ItemFinished("scans", 2, 3, False, error="1 image failed"))
    events.item_started(ItemStarted("talk.mp3", "audio", 3, 3))
    events.page_done(PageDone("talk.mp3", 0, "deferred", 1, 1, unit="chunk"))
    events.item_finished(ItemFinished("talk.mp3", 3, 3, True))


def test_non_terminal_console_prints_plain_lines() -> None:
    console = _console()
    with RichProgress(console) as progress:
        assert not progress.live
        _drive(progress)
    output = _output(console)
    assert ESC not in output
    lines = output.splitlines()
    assert "[1/3] alpha.pdf (PDF)" in lines
    assert "[2/3] scans (image folder)" in lines
    assert "[3/3] talk.mp3 (audio)" in lines
    assert "Waiting 30s: rate limit" in lines
    assert "Warning: model refused image 1" in lines
    assert "Warning: retrying later" in lines
    assert (
        "[1/3] alpha.pdf: complete; 2 page(s) ok, 0 failed, 0 deferred, "
        "2 from an earlier run"
    ) in lines
    assert "    outputs: /out/alpha.txt, /out/alpha.md" in lines
    assert "[2/3] scans: failed; 1 image(s) ok, 1 failed, 0 deferred" in lines
    assert "    error: 1 image failed" in lines
    assert "[3/3] talk.mp3: complete; 0 chunk(s) ok, 0 failed, 1 deferred" in lines


def test_item_without_units_prints_its_verdict() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("gamma.epub", "epub", 1, 1))
        progress.item_finished(ItemFinished("gamma.epub", 1, 1, False))
    lines = _output(console).splitlines()
    assert "[1/1] gamma.epub (EPUB)" in lines
    assert "[1/1] gamma.epub: failed" in lines


def test_submitted_item_is_named_and_the_batch_check_with_it() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("alpha.pdf", "pdf", 1, 1))
        progress.item_finished(ItemFinished("alpha.pdf", 1, 1, True, submitted=True))
    assert "[1/1] alpha.pdf: submitted" in _output(console).splitlines()
    result = core.RunResult(items=(core.ItemResult("alpha.pdf", True, submitted=True),))
    output = _render(summary_table(result, progress))
    assert _row(output, "alpha.pdf")[1] == "submitted"
    assert "0 complete, 1 submitted, 0 failed" in output
    assert (
        "Batch: 1 item(s) submitted; run the batch check task, or "
        "'chronotranscriber batch check', to collect the results."
    ) in " ".join(output.split())


def test_partial_item_line_agrees_with_the_summary() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("alpha.pdf", "pdf", 1, 1))
        progress.page_done(PageDone("alpha.pdf", 0, "ok", 1, 1))
        progress.item_finished(ItemFinished("alpha.pdf", 1, 1, True, partial=True))
    lines = _output(console).splitlines()
    assert "[1/1] alpha.pdf: partial; 1 page(s) ok, 0 failed, 0 deferred" in lines
    result = core.RunResult(items=(core.ItemResult("alpha.pdf", True, partial=True),))
    assert _row(_render(summary_table(result, progress)), "alpha.pdf")[1] == "partial"


def test_terminal_console_shows_bars_and_redirects_stderr() -> None:
    console = Console(
        file=io.StringIO(), width=100, force_terminal=True, color_system=None
    )
    with RichProgress(console) as progress:
        assert progress.live
        progress.item_started(ItemStarted("talk.mp3", "audio", 1, 1))
        progress.page_done(
            PageDone("talk.mp3", 0, "ok", 1, 3, already_complete=2, unit="chunk")
        )
        progress.waiting(Waiting("rate limit", 12.0))
        screen = _render(progress._render())
        assert "Waiting 12s: rate limit" in screen
        assert "chunks, +2 from an earlier run" in screen
        sys.stderr.write("a log record\n")
        progress.item_finished(ItemFinished("talk.mp3", 1, 1, True))
    assert not progress.live
    output = _output(console)
    assert "a log record" in output
    assert "talk.mp3: complete; 1 chunk(s) ok" in output
    assert "Items" in output


def _result() -> core.RunResult:
    return core.RunResult(
        items=(
            core.ItemResult("alpha.pdf", True, ("/out/alpha.txt",)),
            core.ItemResult("scans", False, error="1 image failed"),
            core.ItemResult("gamma.epub", True, ("/out/gamma.txt",)),
        ),
        skipped=("delta.pdf",),
        seconds=12.34,
        usage={
            "transcription": Usage(1200, 300, 1500, 100, 0),
            "repair": Usage(800, 200, 1000, 0, 50),
        },
        databases=("/out/chronotranscriber.sqlite",),
        blocked=("epsilon.pdf",),
    )


def _row(output: str, first: str) -> list[str]:
    for line in output.splitlines():
        cells = [cell.strip() for cell in re.split(r"[│┃]", line.strip("│┃ "))]
        if cells and cells[0] == first:
            return cells
    raise AssertionError(f"no row starting with {first!r} in:\n{output}")


def test_summary_table_lists_items_units_outputs_and_usage() -> None:
    progress = RichProgress(_console())
    with progress:
        _drive(progress)
    output = _render(summary_table(_result(), progress))
    assert ESC not in output
    assert _row(output, "alpha.pdf") == ["alpha.pdf", "complete", "page", "2", "0", "0"]
    assert _row(output, "scans") == [
        "scans",
        "failed",
        "image",
        "1",
        "1",
        "0",
        "1 image failed",
    ]
    assert _row(output, "gamma.epub") == ["gamma.epub", "complete", "-", "-", "-", "-"]
    assert "2 complete, 1 failed" in output
    assert "Outputs: /out/alpha.txt, /out/gamma.txt" in output
    assert "Skipped (already complete): delta.pdf" in output
    assert "Blocked (held by another run): epsilon.pdf" in output
    assert "Databases written: /out/chronotranscriber.sqlite" in output
    assert "Run time: 12.3s" in output
    assert _row(output, "transcription") == [
        "transcription",
        "1,200",
        "300",
        "100",
        "0",
        "1,500",
    ]
    assert _row(output, "repair")[-1] == "1,000"
    totals = [
        line for line in output.splitlines() if line.strip("│ ").startswith("Total")
    ]
    assert len(totals) == 2
    assert _row(totals[0], "Total")[3:6] == ["3", "1", "0"]
    assert "2,000" in totals[1] and "2,500" in totals[1]


def test_summary_table_without_progress_usage_or_skips() -> None:
    result = core.RunResult(items=(core.ItemResult("alpha.pdf", True),))
    output = _render(summary_table(result))
    assert _row(output, "alpha.pdf")[2:6] == ["-", "-", "-", "-"]
    assert "Token usage: none reported" in output
    assert "Skipped (already complete): none" in output
    assert "Blocked (held by another run): none" in output
    assert "Outputs: none" in output


def _failed_unit(idx: int, error_type: str | None, error: str | None) -> UnitRow:
    return UnitRow(
        doc_id=1,
        kind="page",
        idx=idx,
        status="failed",
        outcome="failed",
        source_index=idx,
        label=None,
        content=None,
        tool="chronotranscriber",
        model="m",
        provider="google",
        run_id="r1",
        attempts=1,
        usage=None,
        error=error,
        error_type=error_type,
        provenance=None,
        response_id=None,
        stale=0,
        updated_at=None,
    )


def test_summary_names_each_failed_unit_with_its_error_type_unclipped() -> None:
    long_name = "glasse_art_of_cookery_2p.pdf-909ebc59/glasse_art_of_cookery_2p"
    error = f"1 of 2 page(s) failed to transcribe for '{long_name}'"
    filtered = "Response stopped by the provider's content filter: finish_reason=X"
    result = core.RunResult(
        items=(
            core.ItemResult(
                "scans",
                False,
                error=error,
                failures=(
                    _failed_unit(1, "content_filter", filtered),
                    _failed_unit(4, None, None),
                ),
            ),
        )
    )
    progress = RichProgress(_console())
    with progress:
        progress.page_done(PageDone("scans", 1, "failed", 1, 2, unit="image"))
    output = _render(summary_table(result, progress))
    flat = " ".join(output.split())
    assert "…" not in output
    assert "Failed units:" in output
    assert f"scans: image 2: content_filter: {filtered}" in flat
    assert "scans: image 5: error" in flat
    error_cells = [
        cells[-2].strip()
        for line in output.splitlines()
        if len(cells := line.split("│")) > 2
    ]
    assert long_name in "".join(error_cells)


def test_summary_table_shows_names_and_errors_literally() -> None:
    result = core.RunResult(
        items=(core.ItemResult("[red]draft.pdf", False, error="bad [/x] tag"),)
    )
    output = _render(summary_table(result))
    assert "[red]draft.pdf" in output
    assert "bad [/x] tag" in output


def test_plan_table_shows_names_and_reasons_literally(tmp_path: Path) -> None:
    run_plan = dataclasses.replace(
        _plan(tmp_path),
        to_process=(
            _planned(
                tmp_path,
                "[bold]a.pdf",
                "pdf",
                "llm",
                Classification(ItemState.PARTIAL, "[bold]a.pdf", note="[/i] m2"),
            ),
        ),
        refused=(
            _planned(
                tmp_path,
                "[red]b.pdf",
                "pdf",
                "llm",
                Classification(ItemState.REFUSED, "[red]b.pdf", reason="dpi [/x]"),
            ),
        ),
    )
    output = _render(plan_table(run_plan))
    assert "[bold]a.pdf" in output
    assert "[/i] m2" in output
    assert "[red]b.pdf" in output
    assert "dpi [/x]" in output


def test_progress_descriptions_show_bracketed_names_literally() -> None:
    console = Console(
        file=io.StringIO(), width=100, force_terminal=True, color_system=None
    )
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("[red]scan", "images", 1, 1))
        assert "[red]scan" in _render(progress._render())


def _planned(
    tmp_path: Path,
    name: str,
    kind: str,
    method: str,
    resume: Classification,
    chunks: str | None = None,
) -> core.PlannedItem:
    return core.PlannedItem(
        SourceItem(tmp_path / name, kind),
        method,
        resume,
        tmp_path,
        name,
        tmp_path / f"{name}.txt",
        chunks=chunks,
    )


def _plan(tmp_path: Path, *, batch: bool = False) -> core.RunPlan:
    spec = make_spec(tmp_path, type="auto")
    spec = dataclasses.replace(spec, run=dataclasses.replace(spec.run, batch=batch))
    partial = Classification(
        ItemState.PARTIAL,
        "alpha.pdf",
        units=(UnitKey(1, "page", 3), UnitKey(1, "page", 4)),
        note="m2",
    )
    return core.RunPlan(
        spec=spec,
        input_path=tmp_path,
        output_dir=None,
        discovered=(),
        to_process=(
            _planned(tmp_path, "alpha.pdf", "pdf", "llm", partial),
            _planned(
                tmp_path,
                "talk.mp3",
                "audio",
                "whisper",
                Classification(ItemState.NONE, "talk.mp3"),
                chunks="3 chunk(s) of 600 s",
            ),
        ),
        skipped=(
            _planned(
                tmp_path,
                "done.pdf",
                "pdf",
                "native",
                Classification(ItemState.COMPLETE, "done.pdf"),
            ),
            _planned(
                tmp_path,
                "stale.epub",
                "epub",
                "native",
                Classification(ItemState.COMPLETE, "stale.epub", render=True),
            ),
        ),
        unstored=(
            _planned(
                tmp_path,
                "old.pdf",
                "pdf",
                "tesseract",
                Classification(ItemState.UNSTORED, "old.pdf", reason="no store rows"),
            ),
        ),
        blocked=(
            _planned(
                tmp_path,
                "held",
                "images",
                "llm",
                Classification(ItemState.BLOCKED, "held", reason="leased by run r1"),
            ),
        ),
        refused=(
            _planned(
                tmp_path,
                "other.pdf",
                "pdf",
                "llm",
                Classification(ItemState.REFUSED, "other.pdf", reason="dpi 300"),
            ),
        ),
    )


def test_plan_table_shows_methods_resume_state_and_left_out_items(
    tmp_path: Path,
) -> None:
    output = _render(plan_table(_plan(tmp_path)))
    assert ESC not in output
    assert "Plan: 2 to process, 2 already complete, 3 left out" in output
    assert _row(output, "1") == [
        "1",
        "alpha.pdf",
        "PDF",
        "llm",
        "partial",
        "2 unit(s) to process; m2",
    ]
    assert _row(output, "2") == [
        "2",
        "talk.mp3",
        "audio",
        "whisper",
        "none",
        "3 chunk(s) of 600 s",
    ]
    lines = {
        cells[1]: cells[2:]
        for line in output.splitlines()
        if (cells := [c.strip() for c in re.split(r"[│┃]", line.strip("│┃ "))])
        and cells[0] == "-"
    }
    assert lines["done.pdf"] == ["PDF", "native", "complete (skip)"]
    assert lines["stale.epub"] == ["EPUB", "native", "complete (render again)"]
    assert lines["old.pdf"] == ["PDF", "tesseract", "unstored", "no store rows"]
    assert lines["held"] == ["image folder", "llm", "blocked", "leased by run r1"]
    assert lines["other.pdf"] == ["PDF", "llm", "refused", "dpi 300"]
    assert "Batch:" not in output


def test_plan_table_names_the_batch(tmp_path: Path) -> None:
    run_plan = _plan(tmp_path, batch=True)
    output = _render(plan_table(run_plan))
    provider = run_plan.spec.model.provider
    assert (
        f"Batch: 1 LLM item(s) will be submitted to the {provider} batch API"
    ) in output
