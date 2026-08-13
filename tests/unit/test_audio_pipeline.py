"""Tests for the audio handlers and the streaming pipeline's method/handler seam.

Covers ``transcribe_audio_payload``, ``transcribe_audio_payload_whisper``, and
``run_streaming_transcription_pipeline`` driven with the audio method — plus a
regression guard that the image path's defaults are unchanged.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from modules.audio.backends.base import build_legacy_response
from modules.audio.constants import NO_TRANSCRIBABLE_TEXT
from modules.transcribe.pipeline import (
    PageTranscriptionError,
    run_streaming_transcription_pipeline,
    transcribe_audio_payload,
    transcribe_audio_payload_whisper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_payload(index: int, source: str = "rec.wav", **extra: Any) -> Any:
    """Lightweight stand-in for an AudioChunkPayload."""
    name = f"rec_chunk_{index + 1:04d}.mp3"
    return SimpleNamespace(
        index=index,
        image_name=name,
        path=Path(f"/tmp/{name}"),
        mime_type="audio/mp3",
        source_file=source,
        page_index=None,
        provenance=lambda: {"chunk_index": index},
        **extra,
    )


class _StubAudioTranscriber:
    """AudioTranscriber-shaped facade over a canned response sequence."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[Any] = []

    async def transcribe_audio_chunk(self, payload: Any) -> dict[str, Any]:
        self.calls.append(payload)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item


def _ok(text: str) -> dict[str, Any]:
    return build_legacy_response(
        output_text=text, provider="stub", model="stub-audio-model"
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _transcription_records(path: Path) -> list[dict[str, Any]]:
    return [
        r
        for r in _read_records(path)
        if r.get("image_name") and r.get("text_chunk") is not None
    ]


# ---------------------------------------------------------------------------
# transcribe_audio_payload
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscribeAudioPayload:
    async def test_success_returns_the_five_tuple(self) -> None:
        payload = _chunk_payload(2)
        transcriber = _StubAudioTranscriber([_ok("Spoken words.")])
        result = await transcribe_audio_payload(payload, transcriber)
        assert len(result) == 5
        assert result[0] is payload
        assert result[1] == "rec_chunk_0003.mp3"
        assert result[2] == "Spoken words."
        assert result[3]["output_text"] == "Spoken words."
        assert result[4] == 2

    async def test_the_payload_is_handed_to_the_transcriber(self) -> None:
        payload = _chunk_payload(0)
        transcriber = _StubAudioTranscriber([_ok("text")])
        await transcribe_audio_payload(payload, transcriber)
        assert transcriber.calls == [payload]

    async def test_empty_transcript_arrives_as_the_shared_sentinel(self) -> None:
        transcriber = _StubAudioTranscriber([_ok("")])
        result = await transcribe_audio_payload(_chunk_payload(0), transcriber)
        assert result[2] == NO_TRANSCRIBABLE_TEXT

    async def test_backend_error_dict_becomes_an_error_placeholder(self) -> None:
        error = build_legacy_response(
            output_text="", provider="stub", model="m", error="503 upstream"
        )
        transcriber = _StubAudioTranscriber([error])
        result = await transcribe_audio_payload(_chunk_payload(0), transcriber)
        assert result[2].startswith("[transcription error")

    async def test_extraction_failure_becomes_a_named_error_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_result: Any, _name: str = "") -> str:
            raise ValueError("unparseable")

        monkeypatch.setattr(
            "modules.transcribe.pipeline.extract_transcribed_text", _boom
        )
        transcriber = _StubAudioTranscriber([_ok("text")])
        result = await transcribe_audio_payload(_chunk_payload(1), transcriber)
        assert result[2] == "[transcription error: rec_chunk_0002.mp3]"
        assert result[3] is not None

    async def test_backend_exception_returns_an_error_tuple(self) -> None:
        transcriber = _StubAudioTranscriber([RuntimeError("connection reset")])
        result = await transcribe_audio_payload(_chunk_payload(4), transcriber)
        assert result[2] == "[transcription error: rec_chunk_0005.mp3]"
        assert result[3] is None
        assert result[4] == 4


# ---------------------------------------------------------------------------
# transcribe_audio_payload_whisper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscribeAudioPayloadWhisper:
    async def test_transcript_is_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file",
            lambda path, cfg: "local transcript",
        )
        result = await transcribe_audio_payload_whisper(_chunk_payload(0), {})
        assert result[2] == "local transcript"
        assert result[3] is None

    async def test_none_becomes_an_error_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file",
            lambda path, cfg: None,
        )
        result = await transcribe_audio_payload_whisper(_chunk_payload(2), {})
        assert result[2] == "[transcription error: rec_chunk_0003.mp3]"

    async def test_the_sentinel_passes_through_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file",
            lambda path, cfg: NO_TRANSCRIBABLE_TEXT,
        )
        result = await transcribe_audio_payload_whisper(_chunk_payload(0), {})
        assert result[2] == NO_TRANSCRIBABLE_TEXT

    async def test_an_exception_becomes_an_error_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_path: Path, _cfg: dict[str, Any]) -> str:
            raise RuntimeError("model crashed")

        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file", _boom
        )
        result = await transcribe_audio_payload_whisper(_chunk_payload(0), {})
        assert result[2].startswith("[transcription error")

    async def test_path_and_config_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _capture(path: Path, cfg: dict[str, Any]) -> str:
            seen["path"] = path
            seen["cfg"] = cfg
            return "ok"

        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file", _capture
        )
        payload = _chunk_payload(0)
        await transcribe_audio_payload_whisper(payload, {"beam_size": 1})
        assert seen["path"] == payload.path
        assert seen["cfg"] == {"beam_size": 1}

    async def test_inference_runs_off_the_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker_thread: list[int] = []

        def _capture(_path: Path, _cfg: dict[str, Any]) -> str:
            worker_thread.append(threading.get_ident())
            return "ok"

        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file", _capture
        )
        await transcribe_audio_payload_whisper(_chunk_payload(0), {})
        assert worker_thread and worker_thread[0] != threading.get_ident()


# ---------------------------------------------------------------------------
# run_streaming_transcription_pipeline driven with the audio method
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingPipelineAudioMode:
    @pytest.fixture
    def paths(self, tmp_path: Path) -> tuple[Path, Path]:
        jsonl_path = tmp_path / "rec.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        return jsonl_path, tmp_path / "rec.txt"

    async def _run(
        self,
        paths: tuple[Path, Path],
        responses: list[Any],
        *,
        count: int = 3,
    ) -> _StubAudioTranscriber:
        jsonl_path, output_path = paths
        transcriber = _StubAudioTranscriber(responses)

        async def payload_source() -> Any:
            for index in range(count):
                yield _chunk_payload(index)

        await run_streaming_transcription_pipeline(
            payload_source(),
            transcriber,
            jsonl_path,
            output_path,
            "rec.wav",
            {"concurrency": {"transcription": {"concurrency_limit": 2}}},
            {},
            method="audio-api",
            handler=transcribe_audio_payload,
        )
        return transcriber

    async def test_records_carry_the_audio_method(
        self, paths: tuple[Path, Path]
    ) -> None:
        await self._run(paths, [_ok("one"), _ok("two"), _ok("three")])
        records = _transcription_records(paths[0])
        assert len(records) == 3
        assert {r["method"] for r in records} == {"audio-api"}

    async def test_order_indices_follow_the_chunk_indices(
        self, paths: tuple[Path, Path]
    ) -> None:
        await self._run(paths, [_ok("one"), _ok("two"), _ok("three")])
        records = _transcription_records(paths[0])
        assert sorted(r["order_index"] for r in records) == [0, 1, 2]

    async def test_text_chunks_are_persisted(self, paths: tuple[Path, Path]) -> None:
        await self._run(paths, [_ok("one"), _ok("two"), _ok("three")])
        by_index = {
            r["order_index"]: r["text_chunk"] for r in _transcription_records(paths[0])
        }
        assert by_index == {0: "one", 1: "two", 2: "three"}

    async def test_records_carry_chunk_provenance(
        self, paths: tuple[Path, Path]
    ) -> None:
        await self._run(paths, [_ok("one"), _ok("two"), _ok("three")])
        records = _transcription_records(paths[0])
        assert all(r["pre_processed_image"] is None for r in records)
        assert all(r["source_file"] == "rec.wav" for r in records)
        assert all("chunk_index" in r["image_provenance"] for r in records)

    async def test_output_is_regenerated_from_the_jsonl_in_order(
        self, paths: tuple[Path, Path]
    ) -> None:
        await self._run(paths, [_ok("alpha"), _ok("beta"), _ok("gamma")])
        text = paths[1].read_text(encoding="utf-8")
        assert text.index("alpha") < text.index("beta") < text.index("gamma")

    async def test_a_resume_marker_is_written(self, paths: tuple[Path, Path]) -> None:
        await self._run(paths, [_ok("one"), _ok("two"), _ok("three")])
        assert any("resume_format" in r for r in _read_records(paths[0]))

    async def test_file_provenance_is_written_first(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "rec.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        transcriber = _StubAudioTranscriber([_ok("one")])

        async def payload_source() -> Any:
            yield _chunk_payload(0)

        await run_streaming_transcription_pipeline(
            payload_source(),
            transcriber,
            jsonl_path,
            tmp_path / "rec.txt",
            "rec.wav",
            {"concurrency": {"transcription": {"concurrency_limit": 1}}},
            {},
            file_provenance={"file_provenance": {"chunk_plan_signature": "1x0s-abc"}},
            method="audio-api",
            handler=transcribe_audio_payload,
        )
        provenance = [r for r in _read_records(jsonl_path) if "file_provenance" in r]
        assert provenance[0]["file_provenance"]["chunk_plan_signature"] == "1x0s-abc"

    async def test_no_text_chunk_is_not_a_failure(
        self, paths: tuple[Path, Path]
    ) -> None:
        # An empty transcript is a legitimate silent chunk, not an API failure:
        # it must be recorded and must NOT raise PageTranscriptionError.
        await self._run(paths, [_ok("one"), _ok(""), _ok("three")])
        texts = {r["text_chunk"] for r in _transcription_records(paths[0])}
        assert NO_TRANSCRIBABLE_TEXT in texts

    async def test_no_text_chunk_reaches_the_final_output(
        self, paths: tuple[Path, Path]
    ) -> None:
        await self._run(paths, [_ok("one"), _ok(""), _ok("three")])
        assert NO_TRANSCRIBABLE_TEXT in paths[1].read_text(encoding="utf-8")

    async def test_a_failed_chunk_raises_after_the_output_is_written(
        self, paths: tuple[Path, Path]
    ) -> None:
        jsonl_path, output_path = paths
        with pytest.raises(PageTranscriptionError) as excinfo:
            await self._run(
                paths,
                [_ok("one"), RuntimeError("API 500"), _ok("three")],
            )
        assert excinfo.value.failed_pages == 1
        assert output_path.exists()
        assert len(_transcription_records(jsonl_path)) == 3

    async def test_every_payload_reaches_the_transcriber(
        self, paths: tuple[Path, Path]
    ) -> None:
        transcriber = await self._run(paths, [_ok("t")], count=5)
        assert len(transcriber.calls) == 5


# ---------------------------------------------------------------------------
# The image path's defaults must be untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingPipelineImageDefaults:
    async def test_default_method_and_handler_stay_on_the_image_path(
        self, tmp_path: Path
    ) -> None:
        jsonl_path = tmp_path / "doc.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "doc.txt"

        payloads = [
            SimpleNamespace(
                index=i,
                image_name=f"page_{i + 1:04d}_pre_processed.jpg",
                base64="Zm9v",
                mime_type="image/jpeg",
                source_file="doc.pdf",
                page_index=i,
                provenance=lambda: {"sha256": "x"},
            )
            for i in range(2)
        ]

        class _ImageTranscriber:
            def __init__(self) -> None:
                self.calls = 0

            async def transcribe_image_from_base64(
                self, _b64: str, _mime: str, **kwargs: Any
            ) -> dict[str, Any]:
                self.calls += 1
                return {"output_text": f"page text {self.calls}"}

        async def payload_source() -> Any:
            for payload in payloads:
                yield payload

        transcriber = _ImageTranscriber()
        await run_streaming_transcription_pipeline(
            payload_source(),
            transcriber,
            jsonl_path,
            output_path,
            "doc.pdf",
            {"concurrency": {"transcription": {"concurrency_limit": 1}}},
            {},
        )

        records = _transcription_records(jsonl_path)
        assert len(records) == 2
        assert {r["method"] for r in records} == {"gpt"}
        assert transcriber.calls == 2
