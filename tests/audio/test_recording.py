"""Orchestration tests for the audio methods.

Covers one recording through the audio methods into the store (the missing
local runtime, the client guard, the chunk-plan signature, a recording that
cannot be cut), the chunk settings of a run, item discovery and spec
validation, all with mocked runtimes, so neither ffmpeg nor faster-whisper is
required.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.methods.whisper as whisper_module
import chronotranscriber.sources.audio as audio_sources
from chronotranscriber.cli.command import condition_warnings
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.methods import Item
from chronotranscriber.methods.audio_api import AudioApiTranscriber, build_response
from chronotranscriber.methods.units import DocumentHandle
from chronotranscriber.methods.whisper import WhisperMissingError, WhisperTranscriber
from chronotranscriber.output.paths import ItemPaths, item_paths
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.settings import AudioChunking
from chronotranscriber.sources import discover_items
from chronotranscriber.sources.audio import (
    RecordingSplitError,
    chunk_settings,
    plan_recording,
)
from chronotranscriber.sources.audio.constants import (
    GEMINI_INLINE_LIMIT_BYTES,
    OPENAI_UPLOAD_LIMIT_BYTES,
)
from chronotranscriber.sources.audio.ffmpeg import configure_ffmpeg_executables
from chronotranscriber.spec import OPTIONS, RunSpec, SpecError
from tests.conftest import StubAudioBackend, make_job, make_settings, make_spec


class _StubAudioClient:
    """Audio backend stand-in returning one canned transcript."""

    provider_name = "stub"
    model = "stub-audio-model"

    def __init__(self, text: str = "spoken words") -> None:
        self.text = text
        self.calls: list[Any] = []

    async def transcribe_audio_chunk(self, payload: Any) -> dict[str, Any]:
        self.calls.append(payload)
        return build_response(
            output_text=self.text, provider=self.provider_name, model=self.model
        )


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Keep the chunk planning off ffmpeg."""
    configure_ffmpeg_executables()
    monkeypatch.setattr(audio_sources, "is_ffmpeg_available", lambda: False)
    yield
    configure_ffmpeg_executables()


def _spec(recording: Path, method: str = "audio-api") -> RunSpec:
    return make_spec(recording, type="audio", method=method)


def _paths(recording: Path, output_dir: Path) -> ItemPaths:
    return item_paths(
        recording, output_dir=output_dir, layout="hash", input_root=recording
    )


async def _transcribe(
    transcriber_cls: type[Any],
    recording: Path,
    output_dir: Path,
    client: Any = None,
    method: str = "audio-api",
    **job_fields: Any,
) -> list[Path]:
    """Transcribe *recording* into the store of *output_dir*, as the run core does."""
    spec = _spec(recording, method)
    job = make_job(spec, output_dir=output_dir, input_root=recording, **job_fields)
    transcriber = transcriber_cls(job)
    if client is not None:
        transcriber._client = client
    output_dir.mkdir(parents=True, exist_ok=True)
    async with open_store(output_dir) as store:
        await store.begin_run("transcribe")
        doc_id = await store.ensure_document(recording.stem, input_path=recording)
        lease = await store.acquire(doc_id)
        assert lease is not None
        handle = DocumentHandle(store, doc_id, recording.stem, lease, "m", "p")
        try:
            item = Item(recording, "audio", method, document=handle)
            return list(await transcriber.transcribe(item))
        finally:
            await store.release(lease)


def _units(output_dir: Path) -> list[dict[str, Any]]:
    return read_rows(database_path(output_dir), "units", order_by=["idx"])


@pytest.mark.unit
class TestAudioRecording:
    async def test_the_transcript_is_stored_and_no_file_written(
        self, sample_audio_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "audio_out"
        client = _StubAudioClient("hello there")

        written = await _transcribe(AudioApiTranscriber, sample_audio_file, out, client)

        assert written == []
        assert not _paths(sample_audio_file, out).output.exists()
        (unit,) = _units(out)
        assert (unit["kind"], unit["status"], unit["content"]) == (
            "chunk",
            "done",
            "hello there",
        )
        assert len(client.calls) == 1

    async def test_a_whole_file_run_leaves_no_item_folder(
        self, sample_audio_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "audio_out"
        await _transcribe(
            AudioApiTranscriber, sample_audio_file, out, _StubAudioClient()
        )
        assert not _paths(sample_audio_file, out).folder.exists()

    async def test_the_document_records_the_plan_signature_and_the_backend(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        mock_audio_backend: StubAudioBackend,
    ) -> None:
        out = tmp_path / "audio_out"
        await _transcribe(
            AudioApiTranscriber, sample_audio_file, out, mock_audio_backend
        )
        (document,) = read_rows(database_path(out), "documents")
        record = json.loads(document["image_settings"])
        assert record["chunk_plan_signature"].startswith("1x0s-")
        assert (record["provider"], record["model"]) == ("openai", "stub-audio-model")
        assert len(mock_audio_backend.calls) == 1

    async def test_missing_faster_whisper_fails_the_item(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_audio_file: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            whisper_module, "is_faster_whisper_available", lambda: False
        )
        out = tmp_path / "audio_out"

        with pytest.raises(WhisperMissingError, match="faster-whisper is not"):
            await _transcribe(
                WhisperTranscriber, sample_audio_file, out, method="whisper"
            )

        assert not _paths(sample_audio_file, out).output.exists()

    async def test_resumed_chunks_are_written_without_faster_whisper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_audio_file: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(whisper_module, "is_faster_whisper_available", lambda: True)
        monkeypatch.setattr(
            WhisperTranscriber, "transcribe_file", lambda _self, _path: "kept text"
        )
        out = tmp_path / "audio_out"
        await _transcribe(WhisperTranscriber, sample_audio_file, out, method="whisper")
        monkeypatch.setattr(
            whisper_module, "is_faster_whisper_available", lambda: False
        )

        written = await _transcribe(
            WhisperTranscriber, sample_audio_file, out, method="whisper"
        )

        assert written == []
        (unit,) = _units(out)
        assert (unit["status"], unit["content"]) == ("done", "kept text")

    async def test_local_whisper_runs_when_the_runtime_is_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_audio_file: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(whisper_module, "is_faster_whisper_available", lambda: True)
        monkeypatch.setattr(
            WhisperTranscriber,
            "transcribe_file",
            lambda _self, _path: "offline transcript",
        )
        out = tmp_path / "audio_out"

        await _transcribe(WhisperTranscriber, sample_audio_file, out, method="whisper")

        (unit,) = _units(out)
        assert unit["content"] == "offline transcript"
        assert unit["provider"] == "faster-whisper"

    async def test_audio_api_without_a_client_raises(
        self, sample_audio_file: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="No audio transcriber"):
            await _transcribe(
                AudioApiTranscriber, sample_audio_file, tmp_path / "audio_out"
            )

    async def test_an_oversized_recording_without_ffmpeg_fails(
        self, sample_audio_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "audio_out"
        client = _StubAudioClient()
        settings = make_settings(audio_chunking=AudioChunking(max_request_bytes=100))

        with pytest.raises(RecordingSplitError, match="FFmpeg is unavailable"):
            await _transcribe(
                AudioApiTranscriber, sample_audio_file, out, client, settings=settings
            )

        assert client.calls == []
        assert not _paths(sample_audio_file, out).output.exists()

    async def test_an_oversized_recording_without_a_duration_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_audio_file: Path,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(audio_sources, "is_ffmpeg_available", lambda: True)
        monkeypatch.setattr(audio_sources, "probe_duration_seconds", lambda _p: None)
        client = _StubAudioClient()
        settings = make_settings(audio_chunking=AudioChunking(max_request_bytes=100))

        with pytest.raises(RecordingSplitError, match="Could not determine"):
            await _transcribe(
                AudioApiTranscriber,
                sample_audio_file,
                tmp_path / "audio_out",
                client,
                settings=settings,
            )

        assert client.calls == []


@pytest.mark.unit
class TestAudioSettings:
    @pytest.mark.parametrize(
        ("provider", "method", "limit"),
        [
            ("openai", "audio-api", OPENAI_UPLOAD_LIMIT_BYTES),
            ("google", "audio-api", GEMINI_INLINE_LIMIT_BYTES),
            ("google", "whisper", OPENAI_UPLOAD_LIMIT_BYTES),
        ],
    )
    def test_chunk_settings_pick_the_provider_ceiling(
        self, provider: str, method: str, limit: int
    ) -> None:
        flags: dict[str, Any] = {"type": "audio", "method": method}
        if method == "audio-api":
            flags["provider"] = provider
        settings = chunk_settings(make_spec(**flags), make_settings())
        assert settings.max_request_bytes == limit

    def test_chunk_settings_take_the_flag_and_the_settings(self) -> None:
        spec = make_spec(type="audio", method="whisper", audio_chunk_seconds=90)
        chunking = AudioChunking(
            max_request_bytes=1000,
            overlap_seconds=2.5,
            chunk_format="wav16",
            sample_rate=8000,
            mono=False,
            apply_to_local=True,
        )
        settings = chunk_settings(spec, make_settings(audio_chunking=chunking))
        assert (settings.target_seconds, settings.max_request_bytes) == (90, 1000)
        assert (settings.overlap_seconds, settings.chunk_format) == (2.5, "wav16")
        assert (settings.sample_rate, settings.mono) == (8000, False)
        assert settings.apply_to_local is True

    def test_a_local_run_sends_the_file_whole_without_probing(
        self, sample_audio_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_probe(_path: Path) -> float:
            raise AssertionError("probed")

        monkeypatch.setattr(audio_sources, "probe_duration_seconds", _no_probe)
        settings = chunk_settings(
            make_spec(type="audio", method="whisper"), make_settings()
        )
        plan = plan_recording(
            sample_audio_file, "whisper", settings, page_range_active=False
        )
        assert plan.planned is False
        assert len(plan.specs) == 1


@pytest.mark.unit
class TestAudioFileCollection:
    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "recordings"
        directory.mkdir()
        for name in ("a.mp3", "b.wav", "c.m4a", "d.flac", "notes.txt", "scan.pdf"):
            (directory / name).write_bytes(b"x")
        nested = directory / "nested"
        nested.mkdir()
        (nested / "e.ogg").write_bytes(b"x")
        return directory

    def _collect(self, path: Path, **flags: Any) -> list[Path]:
        return [item.path for item in discover_items(path, "audio", **flags)]

    def test_audio_extensions_are_collected(self, audio_dir: Path) -> None:
        found = self._collect(audio_dir)
        assert {p.name for p in found} == {"a.mp3", "b.wav", "c.m4a", "d.flac"}

    def test_recursive_collection_descends(self, audio_dir: Path) -> None:
        found = self._collect(audio_dir, recursive=True)
        assert "e.ogg" in {p.name for p in found}

    def test_a_single_audio_file_input_is_returned_as_is(self, audio_dir: Path) -> None:
        target = audio_dir / "a.mp3"
        assert self._collect(target) == [target]

    def test_explicit_file_names_are_honored(self, audio_dir: Path) -> None:
        found = self._collect(audio_dir, files=("b.wav",))
        assert found == [audio_dir / "b.wav"]

    @pytest.mark.parametrize("recursive", [False, True])
    def test_upper_case_extensions_are_collected_on_a_case_sensitive_filesystem(
        self, audio_dir: Path, monkeypatch: pytest.MonkeyPatch, recursive: bool
    ) -> None:
        # Path.glob matches case-sensitively by default on POSIX; emulate it here.
        posix_glob = Path.glob

        def glob(
            self: Path, pattern: Any, *, case_sensitive: Any = None, **kw: Any
        ) -> Any:
            sensitive = True if case_sensitive is None else case_sensitive
            return posix_glob(self, pattern, case_sensitive=sensitive, **kw)

        monkeypatch.setattr(Path, "glob", glob)
        (audio_dir / "REC001.MP3").write_bytes(b"x")
        found = self._collect(audio_dir, recursive=recursive)
        assert "REC001.MP3" in {p.name for p in found}
        assert len(found) == len(set(found))


@pytest.mark.unit
class TestAudioSpec:
    def test_audio_takes_whisper_without_a_method(self) -> None:
        assert make_spec(type="audio").method.name == "whisper"

    @pytest.mark.parametrize("method", ["tesseract", "llm"])
    def test_a_document_method_on_audio_is_refused(self, method: str) -> None:
        with pytest.raises(SpecError, match="whisper or audio-api"):
            make_spec(type="audio", method=method)

    @pytest.mark.parametrize("method", ["whisper", "audio-api"])
    def test_an_audio_method_on_a_document_type_is_refused(self, method: str) -> None:
        with pytest.raises(SpecError, match="--type pdf takes"):
            make_spec(type="pdf", method=method)

    def test_batch_on_audio_warns(self) -> None:
        flags = {"type": "audio", "method": "audio-api", "batch": True}
        (warning,) = condition_warnings(OPTIONS, flags, {}, make_spec(**flags))
        assert warning.startswith("--batch applies only for method llm")

    @pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
    def test_a_non_audio_provider_is_refused(self, provider: str) -> None:
        with pytest.raises(SpecError, match="openai or google"):
            make_spec(type="audio", method="audio-api", provider=provider)
