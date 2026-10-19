"""Model calls of a whole run: attempt contexts and per-role token totals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.run as core
from chronotranscriber.common.testing import FakeRequest
from chronotranscriber.common.usage import Usage
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import RecordingExt, StatusError

RECOVERED = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
ANSWER = Usage(input_tokens=100, output_tokens=20, total_tokens=120)
PAGE: dict[str, Any] = {
    "image_analysis": "One line of print.",
    "transcription": "Page text",
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}
_REAL_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


def _pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Page {index + 1}")
    document.save(path)
    document.close()
    return path


def _responder() -> Callable[[FakeRequest], Any]:
    failed: list[bool] = []

    def answer(request: FakeRequest) -> Any:
        if not failed:
            failed.append(True)
            return StatusError(408, RECOVERED)
        return PAGE

    return answer


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Only yield, except in the lease heartbeat, which keeps its interval."""
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _REAL_SLEEP(delay, result)
    await _REAL_SLEEP(0)
    return result


@pytest.fixture
def scripted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[core.RunResult, RecordingExt]:
    """Run a two-page PDF; the first transcription attempt fails."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(asyncio, "sleep", _yield_only)
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [_responder()])
    settings = make_settings(rate_limits=((100_000, 1),))
    spec = make_spec(
        _pdf(tmp_path / "in" / "doc.pdf", pages=2),
        type="pdf",
        output=tmp_path / "out",
        concurrency=1,
    )
    result = asyncio.run(core.run(core.plan(spec, settings)))
    return result, recorder


def test_every_model_call_runs_in_an_attempt_context(
    scripted_run: tuple[core.RunResult, RecordingExt],
) -> None:
    result, recorder = scripted_run

    assert result.complete == 1
    assert recorder.roles == ["transcription"] * 3
    assert all(attempt.released for attempt in recorder.attempts)
    assert [type(error) for error in recorder.errors] == [StatusError]
    assert recorder.commits == [ANSWER] * 2
    failed = next(a for a in recorder.attempts if a.errors)
    assert failed.identity.role == "transcription"
    assert failed.recovered == Usage(8, 2, 10)


def test_the_run_result_holds_token_totals_per_role(
    scripted_run: tuple[core.RunResult, RecordingExt],
) -> None:
    result, _ = scripted_run

    assert result.usage == {"transcription": ANSWER + ANSWER + Usage(8, 2, 10)}
    assert result.total_usage.total_tokens == 250
