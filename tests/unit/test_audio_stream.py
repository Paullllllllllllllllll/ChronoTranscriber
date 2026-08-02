"""Tests for modules.audio.audio_stream: chunk naming, resume, and streaming."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from modules.audio.audio_stream import (
    AudioChunkPayload,
    audio_chunk_name,
    compute_audio_skip_indices,
    parse_audio_chunk_index,
    stream_audio_chunks,
)
from modules.audio.chunker import ChunkSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write *records* as JSONL and return the path."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _fake_cut(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    """Replace cut_segment with a recorder that writes a dummy chunk file."""

    def _cut(
        src: Path,
        dst: Path,
        *,
        start: float,
        duration: float | None,
        chunk_format: str,
        mono: bool,
        sample_rate: int,
    ) -> None:
        calls.append(
            {
                "src": src,
                "dst": dst,
                "start": start,
                "duration": duration,
                "chunk_format": chunk_format,
                "mono": mono,
                "sample_rate": sample_rate,
            }
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f"chunk@{start}".encode())

    monkeypatch.setattr("modules.audio.audio_stream.cut_segment", _cut)


def _forbid_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any cut_segment call an outright test failure."""

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cut_segment must not be called for a whole-file plan")

    monkeypatch.setattr("modules.audio.audio_stream.cut_segment", _boom)


async def _collect(source: Any) -> list[AudioChunkPayload]:
    """Drain an async payload iterator into a list."""
    return [payload async for payload in source]


# ---------------------------------------------------------------------------
# Chunk naming
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioChunkName:
    def test_first_chunk_is_one_based_in_the_name(self) -> None:
        assert audio_chunk_name(Path("rec.wav"), 0, ".mp3") == "rec_chunk_0001.mp3"

    def test_name_is_zero_padded_to_four_digits(self) -> None:
        assert audio_chunk_name(Path("rec.wav"), 8, ".mp3") == "rec_chunk_0009.mp3"

    def test_suffix_is_lowercased(self) -> None:
        assert audio_chunk_name(Path("rec.wav"), 0, ".MP3") == "rec_chunk_0001.mp3"

    def test_stem_case_is_preserved(self) -> None:
        assert audio_chunk_name(Path("REC.WAV"), 0, ".wav") == "REC_chunk_0001.wav"

    def test_index_beyond_9999_widens_the_field(self) -> None:
        assert audio_chunk_name(Path("rec.wav"), 9999, ".mp3") == "rec_chunk_10000.mp3"


@pytest.mark.unit
class TestParseAudioChunkIndex:
    @pytest.mark.parametrize("index", [0, 1, 42, 9998, 9999, 12345])
    def test_round_trip(self, index: int) -> None:
        name = audio_chunk_name(Path("recording.m4a"), index, ".mp3")
        assert parse_audio_chunk_index(name) == index

    def test_round_trip_beyond_9999(self) -> None:
        name = audio_chunk_name(Path("rec.wav"), 10_000, ".wav")
        assert name == "rec_chunk_10001.wav"
        assert parse_audio_chunk_index(name) == 10_000

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert parse_audio_chunk_index("  rec_chunk_0003.mp3 \n") == 2

    def test_uppercase_suffix_still_parses(self) -> None:
        assert parse_audio_chunk_index("rec_chunk_0003.MP3") == 2

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "page_0001.png",
            "rec_chunk_12.mp3",
            "rec_chunk_0001",
            "rec_chunk_abcd.mp3",
            "rec.mp3",
        ],
    )
    def test_non_chunk_names_return_none(self, name: str) -> None:
        assert parse_audio_chunk_index(name) is None


# ---------------------------------------------------------------------------
# Resume skip set
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeAudioSkipIndices:
    @pytest.fixture
    def jsonl_path(self, tmp_path: Path) -> Path:
        return _write_jsonl(
            tmp_path / "temp.jsonl",
            [
                {"resume_format": {"version": 1}},
                {
                    "image_name": "rec_chunk_0001.mp3",
                    "text_chunk": "first",
                    "order_index": 0,
                },
                {
                    "image_name": "rec_chunk_0002.mp3",
                    "text_chunk": "[transcription error: rec_chunk_0002.mp3]",
                    "order_index": 1,
                },
                {
                    "image_name": "rec_chunk_0003.mp3",
                    "text_chunk": "third",
                    "order_index": 2,
                },
                # A non-chunk name (image workflow leftovers) must be ignored.
                {
                    "image_name": "page_0001_pre_processed.jpg",
                    "text_chunk": "unrelated",
                    "order_index": 3,
                },
            ],
        )

    def test_all_recorded_chunks_are_skipped_by_default(self, jsonl_path: Path) -> None:
        assert compute_audio_skip_indices(jsonl_path) == {0, 1, 2}

    def test_exclude_errors_drops_the_error_placeholder_chunk(
        self, jsonl_path: Path
    ) -> None:
        assert compute_audio_skip_indices(jsonl_path, exclude_errors=True) == {0, 2}

    def test_missing_file_yields_an_empty_skip_set(self, tmp_path: Path) -> None:
        assert compute_audio_skip_indices(tmp_path / "absent.jsonl") == set()

    def test_empty_file_yields_an_empty_skip_set(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert compute_audio_skip_indices(path) == set()

    def test_metadata_only_file_yields_an_empty_skip_set(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "meta.jsonl",
            [
                {"resume_format": {"version": 1}},
                {"file_provenance": {"chunk_plan_signature": "6x600s-deadbeef"}},
            ],
        )
        assert compute_audio_skip_indices(path) == set()

    def test_later_record_wins_when_excluding_errors(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "retry.jsonl",
            [
                {
                    "image_name": "rec_chunk_0001.mp3",
                    "text_chunk": "[transcription error: rec_chunk_0001.mp3]",
                    "order_index": 0,
                },
                {
                    "image_name": "rec_chunk_0001.mp3",
                    "text_chunk": "recovered",
                    "order_index": 0,
                },
            ],
        )
        assert compute_audio_skip_indices(path, exclude_errors=True) == {0}


# ---------------------------------------------------------------------------
# AudioChunkPayload
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioChunkPayload:
    @pytest.fixture
    def payload(self, tmp_path: Path) -> AudioChunkPayload:
        chunk = tmp_path / "rec_chunk_0002.mp3"
        chunk.write_bytes(b"audio-bytes")
        return AudioChunkPayload(
            index=1,
            image_name=chunk.name,
            path=chunk,
            mime_type="audio/mp3",
            source_file=str(tmp_path / "rec.wav"),
            start_seconds=600.0,
            duration_seconds=600.0,
            byte_size=len(b"audio-bytes"),
            sha256=hashlib.sha256(b"audio-bytes").hexdigest(),
        )

    async def test_read_bytes(self, payload: AudioChunkPayload) -> None:
        assert await payload.read_bytes() == b"audio-bytes"

    async def test_read_base64(self, payload: AudioChunkPayload) -> None:
        expected = base64.b64encode(b"audio-bytes").decode("utf-8")
        assert await payload.read_base64() == expected

    def test_provenance_keys(self, payload: AudioChunkPayload) -> None:
        provenance = payload.provenance()
        assert set(provenance) == {
            "sha256",
            "byte_size",
            "source_file",
            "chunk_index",
            "start_seconds",
            "duration_seconds",
            "is_whole_file",
        }

    def test_provenance_values(self, payload: AudioChunkPayload) -> None:
        provenance = payload.provenance()
        assert provenance["chunk_index"] == 1
        assert provenance["start_seconds"] == 600.0
        assert provenance["duration_seconds"] == 600.0
        assert provenance["byte_size"] == len(b"audio-bytes")
        assert provenance["sha256"] == hashlib.sha256(b"audio-bytes").hexdigest()
        assert provenance["is_whole_file"] is False

    def test_page_index_defaults_to_none(self, payload: AudioChunkPayload) -> None:
        assert payload.page_index is None


# ---------------------------------------------------------------------------
# stream_audio_chunks: whole-file plan
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamWholeFile:
    @pytest.fixture
    def whole_file_specs(self) -> list[ChunkSpec]:
        return [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]

    async def test_yields_one_payload_pointing_at_the_source(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        whole_file_specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _forbid_cut(monkeypatch)
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=whole_file_specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload.path == sample_audio_file
        assert payload.is_whole_file is True
        assert payload.index == 0
        assert payload.image_name == "sample_recording_chunk_0001.wav"
        assert payload.mime_type == "audio/wav"
        assert payload.source_file == str(sample_audio_file)

    async def test_whole_file_payload_carries_the_real_size_and_digest(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        whole_file_specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _forbid_cut(monkeypatch)
        raw = sample_audio_file.read_bytes()
        (payload,) = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=whole_file_specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert payload.byte_size == len(raw)
        assert payload.sha256 == hashlib.sha256(raw).hexdigest()

    async def test_whole_file_plan_creates_no_work_directory(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        whole_file_specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _forbid_cut(monkeypatch)
        work_dir = tmp_path / "work"
        await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=whole_file_specs,
                work_dir=work_dir,
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert not work_dir.exists()

    async def test_skipping_chunk_zero_yields_nothing(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        whole_file_specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _forbid_cut(monkeypatch)
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=whole_file_specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
                skip_indices={0},
            )
        )
        assert payloads == []


# ---------------------------------------------------------------------------
# stream_audio_chunks: multi-chunk plan
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamMultiChunk:
    @pytest.fixture
    def specs(self) -> list[ChunkSpec]:
        return [
            ChunkSpec(index=0, start_seconds=0.0, duration_seconds=600.0),
            ChunkSpec(index=1, start_seconds=600.0, duration_seconds=600.0),
            ChunkSpec(index=2, start_seconds=1200.0, duration_seconds=None),
        ]

    async def test_cuts_and_yields_one_payload_per_spec(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []
        _fake_cut(monkeypatch, calls)
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert len(calls) == 3
        assert [p.index for p in payloads] == [0, 1, 2]
        assert [p.image_name for p in payloads] == [
            "sample_recording_chunk_0001.mp3",
            "sample_recording_chunk_0002.mp3",
            "sample_recording_chunk_0003.mp3",
        ]

    async def test_cut_arguments_mirror_the_specs(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []
        _fake_cut(monkeypatch, calls)
        await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="wav16",
                mono=False,
                sample_rate=22050,
            )
        )
        assert [c["start"] for c in calls] == [0.0, 600.0, 1200.0]
        assert [c["duration"] for c in calls] == [600.0, 600.0, None]
        assert {c["chunk_format"] for c in calls} == {"wav16"}
        assert {c["mono"] for c in calls} == {False}
        assert {c["sample_rate"] for c in calls} == {22050}

    async def test_payload_metadata_is_derived_from_the_cut_file(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cut(monkeypatch, [])
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        first = payloads[0]
        raw = first.path.read_bytes()
        assert first.is_whole_file is False
        assert first.byte_size == len(raw)
        assert first.sha256 == hashlib.sha256(raw).hexdigest()
        assert first.mime_type == "audio/mp3"
        assert first.start_seconds == 0.0
        assert first.duration_seconds == 600.0

    async def test_last_spec_reports_zero_duration_when_open_ended(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cut(monkeypatch, [])
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert payloads[-1].duration_seconds == 0.0

    async def test_skipped_indices_are_neither_cut_nor_yielded(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []
        _fake_cut(monkeypatch, calls)
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
                skip_indices={1},
            )
        )
        assert [p.index for p in payloads] == [0, 2]
        assert [c["start"] for c in calls] == [0.0, 1200.0]

    async def test_wav16_chunks_get_a_wav_suffix_and_mime(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cut(monkeypatch, [])
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="wav16",
                mono=True,
                sample_rate=16000,
            )
        )
        assert payloads[0].image_name.endswith(".wav")
        assert payloads[0].mime_type == "audio/wav"

    async def test_copy_format_keeps_the_source_suffix(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cut(monkeypatch, [])
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="copy",
                mono=True,
                sample_rate=16000,
            )
        )
        assert payloads[0].image_name.endswith(".wav")

    async def test_work_directory_is_created_and_holds_the_chunks(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cut(monkeypatch, [])
        work_dir = tmp_path / "work" / "audio_chunks"
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=work_dir,
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert work_dir.is_dir()
        assert all(p.path.parent == work_dir for p in payloads)

    async def test_a_failing_cut_skips_only_that_chunk(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _cut(src: Path, dst: Path, *, start: float, **_kwargs: Any) -> None:
            if start == 600.0:
                raise RuntimeError("ffmpeg failed")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"ok")

        monkeypatch.setattr("modules.audio.audio_stream.cut_segment", _cut)
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        assert [p.index for p in payloads] == [0, 2]

    async def test_unknown_suffix_falls_back_to_a_generic_mime(
        self,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "rec.xyz"
        source.write_bytes(b"raw")
        _fake_cut(monkeypatch, [])
        payloads = await _collect(
            stream_audio_chunks(
                source,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="copy",
                mono=True,
                sample_rate=16000,
            )
        )
        assert payloads[0].mime_type == "application/octet-stream"

    async def test_chunk_names_round_trip_through_the_resume_parser(
        self,
        sample_audio_file: Path,
        tmp_path: Path,
        specs: list[ChunkSpec],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_cut(monkeypatch, [])
        payloads = await _collect(
            stream_audio_chunks(
                sample_audio_file,
                specs=specs,
                work_dir=tmp_path / "work",
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        )
        for payload in payloads:
            assert parse_audio_chunk_index(payload.image_name) == payload.index
