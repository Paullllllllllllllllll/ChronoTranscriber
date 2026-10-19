"""A missing engine or an unsplittable recording fails the item, not the run."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fitz  # PyMuPDF
import pytest

import chronotranscriber.methods.audio_api as audio_api
import chronotranscriber.methods.tesseract as tesseract_method
import chronotranscriber.methods.whisper as whisper_module
import chronotranscriber.run as core
import chronotranscriber.sources.audio as audio_sources
from chronotranscriber import __main__ as entry
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.images.tesseract_runtime import TESSERACT_MISSING
from chronotranscriber.methods.whisper import FASTER_WHISPER_HINT
from chronotranscriber.output.store import database_path
from chronotranscriber.settings import AudioChunking
from chronotranscriber.spec import to_argv
from tests.conftest import StubAudioBackend, make_settings, make_spec
from tests.recording_events import RecordingEvents


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_sources, "is_ffmpeg_available", lambda: False)


def _outputs(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix in (".txt", ".md", ".json", ".jsonl")
    ]


def _stored_statuses(folder: Path) -> list[str]:
    """Return the status of every unit the store holds for the run."""
    db = database_path(folder)
    if not db.is_file():
        return []
    return [str(row["status"]) for row in read_rows(db, "units", order_by=["idx"])]


def test_missing_faster_whisper_fails_the_run_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_audio_file: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(whisper_module, "is_faster_whisper_available", lambda: False)

    def _no_model(*_args: object) -> str:
        raise AssertionError("the Whisper model was loaded")

    monkeypatch.setattr(whisper_module.WhisperTranscriber, "transcribe_file", _no_model)
    out = tmp_path / "out"
    spec = make_spec(sample_audio_file, type="audio", method="whisper", output=out)

    code = entry.main(["transcribe", *to_argv(spec), "--json"])

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    summary = json.loads(lines[-1])
    assert code == 1
    assert (summary["items_complete"], summary["items_failed"]) == (0, 1)
    assert FASTER_WHISPER_HINT in captured.err
    assert _outputs(out) == []
    assert _stored_statuses(out) == ["pending"]


def test_an_oversized_recording_without_ffmpeg_fails_without_a_call(
    monkeypatch: pytest.MonkeyPatch, sample_audio_file: Path, tmp_path: Path
) -> None:
    backend = StubAudioBackend()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(audio_api, "open_audio_backend", lambda *_args: backend)
    settings = make_settings(audio_chunking=AudioChunking(max_request_bytes=100))
    out = tmp_path / "out"
    spec = make_spec(sample_audio_file, type="audio", method="audio-api", output=out)
    events = RecordingEvents()

    result = asyncio.run(core.run(core.plan(spec, settings), events))

    assert (result.complete, result.failed) == (0, 1)
    (finished,) = events.of("item_finished")
    assert finished.error is not None and "FFmpeg is unavailable" in finished.error
    assert backend.calls == []
    assert _outputs(out) == []
    assert _stored_statuses(out) == []


def test_native_on_a_scanned_pdf_without_tesseract_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tesseract_method, "ensure_tesseract_available", lambda: False)
    scan = tmp_path / "in" / "scan.pdf"
    scan.parent.mkdir()
    document = fitz.open()
    document.new_page(width=200, height=300)
    document.save(scan)
    document.close()
    out = tmp_path / "out"
    events = RecordingEvents()

    run_plan = core.plan(
        make_spec(scan, type="pdf", method="native", output=out), make_settings()
    )
    result = asyncio.run(core.run(run_plan, events))

    assert (result.complete, result.failed) == (0, 1)
    (finished,) = events.of("item_finished")
    assert finished.error == TESSERACT_MISSING
    assert not list(out.rglob("preprocessed_images_tesseract"))
    assert _outputs(out) == []
    assert _stored_statuses(out) == []
