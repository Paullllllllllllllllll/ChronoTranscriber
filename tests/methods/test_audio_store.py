"""The audio methods write chunk units and response rows into the store.

Run on Windows with:

    uv --directory <repo> run pytest tests/methods/test_audio_store.py -q

FFmpeg is stubbed: the probe reports 90 s and ``cut_segment`` writes a small
MP3-like file and records its path. The audio-API backend is either a fake
or the real ``OpenAIAudioBackend`` behind an ``httpx.MockTransport``
(``tests.llm.network``), so no request leaves the machine. Oracle: the chunk
units exist as ``pending`` before the first call; a changed chunk plan
returns them to ``pending`` with a warning, deletes those past it and keeps
the responses; ``--force`` deletes the units of another method; a rerun
skips the done chunks; chunk files exist only in the transient working
directory; each audio-API attempt is a ``responses`` row with the body
verbatim (``raw_kind`` ``http``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.methods.whisper as whisper_module
import chronotranscriber.sources.audio as audio_sources
import chronotranscriber.sources.audio.stream as stream_module
from chronotranscriber.common.selection import parse_page_range
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.common.store import PlannedUnit, Store
from chronotranscriber.methods import Item, Job
from chronotranscriber.methods.audio_api import (
    AudioApiTranscriber,
    OpenAIAudioBackend,
    build_response,
    error_response,
)
from chronotranscriber.methods.recording import CHUNK_FOLDER, SIGNATURE_KEY
from chronotranscriber.methods.units import DocumentHandle, UnitsFailedError
from chronotranscriber.methods.whisper import WhisperTranscriber
from chronotranscriber.output import store as output_store
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.settings import Concurrency, RetrySettings, Settings
from chronotranscriber.sources.audio import chunk_settings, plan_recording
from chronotranscriber.sources.audio.chunker import chunk_plan_signature
from tests.audio.test_audio_capture import AUDIO, TRANSCRIPT
from tests.conftest import make_job, make_settings, make_spec
from tests.llm.conftest import Sleeps
from tests.llm.network import MockNetwork
from tests.recording_events import RecordingEvents

DURATION = 90.0


class FakeBackend:
    """An audio backend that reads the store's units at its first call.

    Chunks whose index is in *fail* get an error response.
    """

    provider_name = "openai"
    model = "fake-transcribe"

    def __init__(self, db: Path, fail: set[int] | None = None) -> None:
        self.db = db
        self.fail = fail or set()
        self.calls: list[int] = []
        self.units_at_first_call: list[dict[str, Any]] | None = None

    async def transcribe_audio_chunk(self, payload: Any) -> dict[str, Any]:
        if self.units_at_first_call is None:
            self.units_at_first_call = read_rows(self.db, "units", order_by=["idx"])
        self.calls.append(payload.index)
        if payload.index in self.fail:
            return error_response("boom", provider="openai", model=self.model)
        return build_response(
            output_text=f"chunk {payload.index}", provider="openai", model=self.model
        )

    async def close(self) -> None:
        """Nothing to close."""


@pytest.fixture
def cuts(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Stub FFmpeg; return the path of every chunk file cut."""
    made: list[Path] = []
    monkeypatch.setattr(audio_sources, "is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(audio_sources, "probe_duration_seconds", lambda _p: DURATION)

    def cut(source: Path, dst: Path, **_kwargs: Any) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(AUDIO)
        made.append(dst)

    monkeypatch.setattr(stream_module, "cut_segment", cut)
    return made


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "store"
    folder.mkdir()
    return folder


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    path = inbox / "rec.mp3"
    path.write_bytes(AUDIO)
    return path


def _settings(**changes: Any) -> Settings:
    return make_settings(
        concurrency=dataclasses.replace(Concurrency(), audio=1),
        retry=RetrySettings(max_attempts=3),
        rate_limits=((100_000, 1),),
        **changes,
    )


def _job(
    recording: Path,
    *,
    method: str = "audio-api",
    output_dir: Path | None = None,
    settings: Settings | None = None,
    **flags: Any,
) -> Job:
    flags.setdefault("audio_chunk_seconds", 30)
    spec = make_spec(recording, type="audio", method=method, **flags)
    job: Job = make_job(
        spec,
        output_dir=output_dir,
        input_root=recording.parent,
        settings=settings or _settings(),
        events=RecordingEvents(),
    )
    return job


def _chunk_count(job: Job, recording: Path) -> int:
    settings = chunk_settings(job.spec, job.settings)
    return len(
        plan_recording(recording, "audio-api", settings, page_range_active=False).specs
    )


async def _run(
    root: Path,
    transcriber: Any,
    recording: Path,
    method: str,
    pages: str | None = None,
) -> None:
    """Run one recording as the run core would, with a fresh run of the store."""
    selection = parse_page_range(pages) if pages else None
    async with open_store(root) as store:
        handle = await _document(store, recording)
        try:
            item = Item(recording, "audio", method, selection, document=handle)
            assert await transcriber.transcribe(item) == []
        finally:
            await store.release(handle.lease)


async def _document(store: Store, recording: Path) -> DocumentHandle:
    await store.begin_run("transcribe")
    doc_id = await store.ensure_document(recording.stem, input_path=recording)
    lease = await store.acquire(doc_id)
    assert lease is not None
    return DocumentHandle(store, doc_id, recording.stem, lease, "run-model", "openai")


def _api(job: Job, backend: Any) -> AudioApiTranscriber:
    transcriber = AudioApiTranscriber(job)
    transcriber._client = backend
    return transcriber


def _units(root: Path) -> list[dict[str, Any]]:
    return read_rows(database_path(root), "units", order_by=["idx"])


def _responses(root: Path) -> list[dict[str, Any]]:
    return read_rows(database_path(root), "responses", order_by=["response_id"])


def _record(root: Path) -> dict[str, Any]:
    (row,) = read_rows(database_path(root), "documents")
    record: dict[str, Any] = json.loads(row["image_settings"])
    return record


def _warnings(job: Job) -> list[str]:
    events = job.events
    assert isinstance(events, RecordingEvents)
    return [str(message) for message in events.of("warning")]


class TestChunkUnits:
    async def test_every_chunk_is_pending_before_the_first_call(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        job = _job(recording)
        backend = FakeBackend(database_path(root))
        count = _chunk_count(job, recording)
        assert count >= 3

        await _run(root, _api(job, backend), recording, "audio-api")

        first = backend.units_at_first_call
        assert first is not None
        assert [(unit["kind"], unit["idx"]) for unit in first] == [
            ("chunk", index) for index in range(count)
        ]
        assert {unit["status"] for unit in first} == {"pending"}
        assert first[0]["label"] == "rec_chunk_0001.mp3"
        rows = _units(root)
        assert [unit["status"] for unit in rows] == ["done"] * count
        assert [unit["content"] for unit in rows] == [
            f"chunk {index}" for index in range(count)
        ]
        assert (rows[0]["provider"], rows[0]["model"]) == ("openai", "fake-transcribe")

    async def test_the_document_records_the_plan_signature(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        job = _job(recording)
        await _run(
            root, _api(job, FakeBackend(database_path(root))), recording, "audio-api"
        )

        settings = chunk_settings(job.spec, job.settings)
        specs = plan_recording(
            recording, "audio-api", settings, page_range_active=False
        ).specs
        record = _record(root)
        assert record[SIGNATURE_KEY] == chunk_plan_signature(specs)
        assert (record["provider"], record["model"]) == ("openai", "fake-transcribe")
        assert record["chunk_count"] == len(specs)

    async def test_a_rerun_skips_done_chunks_and_retries_failed_ones(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        job = _job(recording)
        backend = FakeBackend(database_path(root), fail={1})
        count = _chunk_count(job, recording)

        with pytest.raises(UnitsFailedError, match=f"1 of {count}"):
            await _run(root, _api(job, backend), recording, "audio-api")
        assert [unit["status"] for unit in _units(root)][:2] == ["done", "failed"]

        again = FakeBackend(database_path(root))
        kept = _job(recording, retry_errors=False)
        await _run(root, _api(kept, again), recording, "audio-api")
        assert again.calls == []

        retry = _job(recording)
        await _run(root, _api(retry, again), recording, "audio-api")
        assert again.calls == [1]
        assert [unit["status"] for unit in _units(root)] == ["done"] * count

    async def test_a_page_range_leaves_the_other_chunks_pending(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        job = _job(recording)
        backend = FakeBackend(database_path(root))

        await _run(root, _api(job, backend), recording, "audio-api", pages="2-2")

        assert backend.calls == [1]
        statuses = [unit["status"] for unit in _units(root)]
        assert statuses[1] == "done"
        assert statuses.count("pending") == len(statuses) - 1

    async def test_a_changed_plan_resets_the_chunks_with_a_warning(
        self,
        recording: Path,
        root: Path,
        cuts: list[Path],
        network: MockNetwork,
        sleeps: Sleeps,
    ) -> None:
        network.answers = [(200, TRANSCRIPT)]
        coarse = _job(recording, audio_chunk_seconds=60)
        fine = _job(recording, audio_chunk_seconds=30)
        before, after = _chunk_count(coarse, recording), _chunk_count(fine, recording)
        assert after > before

        await _run(root, _api(coarse, _openai(coarse)), recording, "audio-api")
        assert len(_responses(root)) == before

        await _run(root, _api(fine, _openai(fine)), recording, "audio-api")

        (warning,) = [w for w in _warnings(fine) if "chunk plan" in w]
        assert "changed since the last run" in warning
        rows = _units(root)
        responses = _responses(root)
        assert len(rows) == after
        assert len(responses) == before + after
        assert {unit["status"] for unit in rows} == {"done"}
        assert {unit["response_id"] for unit in rows} == {
            row["response_id"] for row in responses[before:]
        }

    async def test_a_shrunk_plan_deletes_the_chunks_past_it(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        fine = _job(recording, audio_chunk_seconds=30)
        coarse = _job(recording, audio_chunk_seconds=60)
        before, after = _chunk_count(fine, recording), _chunk_count(coarse, recording)
        assert after < before
        await _run(
            root, _api(fine, FakeBackend(database_path(root))), recording, "audio-api"
        )

        backend = FakeBackend(database_path(root))
        await _run(root, _api(coarse, backend), recording, "audio-api")

        assert backend.calls == list(range(after))
        rows = _units(root)
        assert [unit["idx"] for unit in rows] == list(range(after))
        assert {unit["status"] for unit in rows} == {"done"}
        (document,) = read_rows(database_path(root), "documents")
        assert (document["status"], document["units_total"]) == ("complete", after)

    async def test_force_deletes_the_units_of_another_method(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        async with open_store(root) as store:
            handle = await _document(store, recording)
            native = dataclasses.replace(handle, kind="text")
            await native.plan([PlannedUnit(0)])
            await native.write(0, status="done", content="native text")
            await store.release(handle.lease)
        job = _job(recording, force=True)

        await _run(
            root, _api(job, FakeBackend(database_path(root))), recording, "audio-api"
        )

        rows = _units(root)
        assert {unit["kind"] for unit in rows} == {"chunk"}
        assert len(rows) == _chunk_count(job, recording)

    async def test_an_interrupted_forced_shorter_plan_keeps_the_old_chunks(
        self,
        recording: Path,
        root: Path,
        cuts: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fine = _job(recording, audio_chunk_seconds=30)
        coarse = _job(recording, audio_chunk_seconds=60, force=True)
        assert _chunk_count(coarse, recording) < _chunk_count(fine, recording)
        await _run(
            root, _api(fine, FakeBackend(database_path(root))), recording, "audio-api"
        )
        before = (_units(root), _record(root))

        def interrupted(connection: Any, doc_id: int | None = None) -> None:
            raise RuntimeError("interrupted")

        monkeypatch.setattr(output_store, "refresh_documents", interrupted)
        backend = FakeBackend(database_path(root))
        with pytest.raises(RuntimeError, match="interrupted"):
            await _run(root, _api(coarse, backend), recording, "audio-api")

        assert backend.calls == []
        assert (_units(root), _record(root)) == before

    async def test_the_same_plan_resets_nothing(
        self, recording: Path, root: Path, cuts: list[Path]
    ) -> None:
        job = _job(recording)
        await _run(
            root, _api(job, FakeBackend(database_path(root))), recording, "audio-api"
        )
        rerun = _job(recording)
        await _run(
            root, _api(rerun, FakeBackend(database_path(root))), recording, "audio-api"
        )

        assert not [w for w in _warnings(rerun) if "chunk plan" in w]
        assert {unit["status"] for unit in _units(root)} == {"done"}


class TestWorkingFiles:
    async def test_chunk_files_live_only_in_the_transient_working_directory(
        self,
        tmp_path: Path,
        recording: Path,
        root: Path,
        cuts: list[Path],
    ) -> None:
        job = _job(recording, output_dir=tmp_path / "out")

        await _run(
            root, _api(job, FakeBackend(database_path(root))), recording, "audio-api"
        )

        assert cuts
        for path in cuts:
            assert not path.exists()
            assert path.parent.name == CHUNK_FOLDER
            assert not {recording.parent, tmp_path / "out", root} & set(path.parents)
        assert {path.name for path in recording.parent.iterdir()} == {"rec.mp3"}
        assert not (tmp_path / "out").exists()
        assert list(tmp_path.rglob("*.jsonl")) == []

    async def test_keep_working_files_keeps_the_chunks(
        self,
        tmp_path: Path,
        recording: Path,
        root: Path,
        cuts: list[Path],
    ) -> None:
        job = _job(recording, output_dir=tmp_path / "out", keep_working_files=True)

        await _run(
            root, _api(job, FakeBackend(database_path(root))), recording, "audio-api"
        )

        kept = job.locate(recording).folder / CHUNK_FOLDER
        assert sorted(cuts) == sorted(kept.iterdir())

    async def test_a_whole_file_plan_needs_no_working_directory(
        self,
        tmp_path: Path,
        recording: Path,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(whisper_module, "is_faster_whisper_available", lambda: True)
        monkeypatch.setattr(
            WhisperTranscriber, "transcribe_file", lambda _self, _path: "local text"
        )
        job = _job(
            recording,
            method="whisper",
            output_dir=tmp_path / "out",
            keep_working_files=True,
        )

        await _run(root, WhisperTranscriber(job), recording, "whisper")

        (unit,) = _units(root)
        assert (unit["label"], unit["content"]) == ("rec_chunk_0001.mp3", "local text")
        assert unit["provider"] == "faster-whisper"
        assert unit["model"] == WhisperTranscriber(job).options.model
        assert not (tmp_path / "out").exists()


class TestWhisperResume:
    async def test_done_chunks_need_no_faster_whisper(
        self,
        tmp_path: Path,
        recording: Path,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(whisper_module, "is_faster_whisper_available", lambda: True)
        monkeypatch.setattr(
            WhisperTranscriber, "transcribe_file", lambda _self, _path: "kept text"
        )
        job = _job(recording, method="whisper")
        await _run(root, WhisperTranscriber(job), recording, "whisper")
        monkeypatch.setattr(
            whisper_module, "is_faster_whisper_available", lambda: False
        )

        await _run(root, WhisperTranscriber(job), recording, "whisper")

        (unit,) = _units(root)
        assert unit["content"] == "kept text"

    async def test_missing_faster_whisper_fails_with_chunks_to_do(
        self,
        recording: Path,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            whisper_module, "is_faster_whisper_available", lambda: False
        )
        job = _job(recording, method="whisper")

        with pytest.raises(whisper_module.WhisperMissingError):
            await _run(root, WhisperTranscriber(job), recording, "whisper")

        assert [unit["status"] for unit in _units(root)] == ["pending"]


class _StoredRows:
    """The response rows, read through a ``mode=ro`` connection."""

    def __init__(self, db: Path) -> None:
        self.db = db

    @property
    def records(self) -> list[dict[str, Any]]:
        if not self.db.exists():
            return []
        return read_rows(self.db, "responses", order_by=["response_id"])


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, root: Path) -> MockNetwork:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    view: Any = _StoredRows(database_path(root))
    return MockNetwork(view).install(monkeypatch)


def _openai(job: Job) -> OpenAIAudioBackend:
    return OpenAIAudioBackend("gpt-transcribe", env=job.env)


class TestAudioApiResponses:
    async def test_each_attempt_is_a_response_row_with_the_http_body(
        self,
        recording: Path,
        root: Path,
        cuts: list[Path],
        network: MockNetwork,
        sleeps: Sleeps,
    ) -> None:
        server_error = (
            b'{"error": {"message": "The server had an error.", '
            b'"type": "server_error", "param": null, "code": null}}'
        )
        network.answers = [(500, server_error), (200, TRANSCRIPT)]
        job = _job(recording)
        count = _chunk_count(job, recording)

        await _run(root, _api(job, _openai(job)), recording, "audio-api")

        rows = _responses(root)
        assert len(rows) == count + 1
        assert [row["raw_kind"] for row in rows] == ["http"] * len(rows)
        assert str(rows[0]["raw"]).encode("utf-8") == server_error
        assert all(str(row["raw"]).encode("utf-8") == TRANSCRIPT for row in rows[1:])
        assert [row["http_status"] for row in rows] == [500] + [200] * count
        units = _units(root)
        assert [(row["kind"], row["idx"]) for row in rows] == [("chunk", 0)] + [
            ("chunk", index) for index in range(count)
        ]
        assert units[0]["response_id"] == rows[1]["response_id"]
        assert units[0]["attempts"] == 2
        assert units[0]["content"] == "Grüße aus  Zürich"
        assert network.violations == []
