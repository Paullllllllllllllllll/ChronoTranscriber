"""The LLM method's streaming contract on the store.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/methods/test_llm_store.py -q

A scripted chat model answers every page (``tests.llm.helpers``), so no
request leaves the machine. The oracles read the unit rows through ``mode=ro``
connections and the rendered ``.txt`` the run core writes after the method.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest
from PIL import Image

import chronotranscriber.methods.llm as llm_method
import chronotranscriber.run as core
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.output.store import DATABASE_NAME
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import Script, StatusError
from tests.recording_events import RecordingEvents


def _page(text: str) -> dict[str, Any]:
    return {
        "image_analysis": "Print.",
        "transcription": text,
        "no_transcribable_text": False,
        "transcription_not_possible": False,
    }


PAGES = [_page(f"page {i} text") for i in range(4)]


def _pdf(path: Path, pages: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Page {index + 1}")
    document.save(path)
    document.close()
    return path


def _run(
    source: Path, out: Path, **flags: Any
) -> tuple[core.RunResult, RecordingEvents]:
    flags.setdefault("type", "pdf")
    flags.setdefault("concurrency", 1)
    spec = make_spec(source, output=out, **flags)
    run_plan = core.plan(spec, make_settings(rate_limits=((100_000, 1),)))
    events = RecordingEvents()
    return asyncio.run(core.run(run_plan, events)), events


def _units(out: Path) -> list[dict[str, Any]]:
    return read_rows(out / DATABASE_NAME, "units", order_by=["idx"])


def _rendered(out: Path, name: str = "doc.txt") -> str:
    (path,) = out.rglob(name)
    return path.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_a_clean_run_commits_every_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = helpers.install(monkeypatch, PAGES)
    out = tmp_path / "out"

    result, _ = _run(_pdf(tmp_path / "in" / "doc.pdf"), out)

    assert (result.complete, result.failed) == (1, 0)
    assert len(script.requests) == 4
    rows = _units(out)
    assert [(row["idx"], row["status"]) for row in rows] == [
        (i, "done") for i in range(4)
    ]
    text = _rendered(out)
    for i in range(4):
        assert f"page {i} text" in text


def test_a_failing_page_fails_the_item_after_the_others_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers.install(monkeypatch, [PAGES[0], StatusError(400), PAGES[2], PAGES[3]])
    out = tmp_path / "out"

    result, _ = _run(_pdf(tmp_path / "in" / "doc.pdf"), out)

    ((name, success, error),) = [
        (item.name, item.success, item.error) for item in result.items
    ]
    assert (name, success) == ("doc.pdf", False)
    assert error is not None and "1 of 4" in error
    assert [row["status"] for row in _units(out)] == [
        "done",
        "failed",
        "done",
        "done",
    ]
    lines = _rendered(out).splitlines()
    assert lines[1].startswith("page_0002_pre_processed.jpg: [transcription error")
    assert [lines[0], lines[2], lines[3]] == [f"page {i} text" for i in (0, 2, 3)]


def test_a_rerun_calls_the_model_only_for_the_missing_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = helpers.install(monkeypatch, PAGES)
    source = _pdf(tmp_path / "in" / "doc.pdf")
    out = tmp_path / "out"
    _run(source, out, pages="1-2")
    assert [row["status"] for row in _units(out)] == ["done"] * 2 + ["pending"] * 2
    first = len(script.requests)

    result, _ = _run(source, out)

    assert result.complete == 1
    assert (first, len(script.requests)) == (2, 4)
    assert [row["status"] for row in _units(out)] == ["done"] * 4
    text = _rendered(out)
    assert text.count("text\n") == 4


def _folder(path: Path, count: int = 2) -> Path:
    path.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (40, 60), "white").save(path / f"img{index + 1}.png")
    return path


def _failing_stream(*_args: Any) -> AsyncIterator[Any]:
    async def gen() -> AsyncIterator[Any]:
        raise RuntimeError("incompatible page source")
        yield  # pragma: no cover

    return gen()


def test_an_error_while_streaming_an_image_folder_fails_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script: Script = helpers.install(monkeypatch, PAGES)
    monkeypatch.setattr(llm_method, "stream_pages", _failing_stream)
    out = tmp_path / "out"

    result, _ = _run(_folder(tmp_path / "in" / "scans"), out, type="images")

    ((item,),) = [result.items]
    assert (item.name, item.success) == ("scans", False)
    assert item.error is not None and "incompatible page source" in item.error
    assert script.requests == []
    assert [row["status"] for row in _units(out)] == ["pending", "pending"]


def test_the_streaming_error_propagates_from_the_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers.install(monkeypatch, PAGES)
    monkeypatch.setattr(llm_method, "stream_pages", _failing_stream)
    folder = _folder(tmp_path / "in" / "scans")
    captured: list[BaseException] = []
    original = llm_method.LlmTranscriber.transcribe

    async def spy(self: Any, item: Any) -> Any:
        try:
            return await original(self, item)
        except BaseException as exc:
            captured.append(exc)
            raise

    monkeypatch.setattr(llm_method.LlmTranscriber, "transcribe", spy)

    _run(folder, tmp_path / "out", type="images")

    assert [type(exc) for exc in captured] == [RuntimeError]
