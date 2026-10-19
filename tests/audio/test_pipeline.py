"""Tests for the audio chunk handler ``transcribe_audio_payload``.

The unit pipeline that drives it is covered in ``tests/methods``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronotranscriber.methods.audio_api import build_response
from chronotranscriber.methods.pages import transcribe_audio_payload
from chronotranscriber.sources.audio.constants import NO_TRANSCRIBABLE_TEXT


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
    """Audio backend stand-in over a canned response sequence."""

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
    return build_response(output_text=text, provider="stub", model="stub-audio-model")


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

    async def test_empty_transcript_arrives_as_the_shared_sentinel(self) -> None:
        transcriber = _StubAudioTranscriber([_ok("")])
        result = await transcribe_audio_payload(_chunk_payload(0), transcriber)
        assert result[2] == NO_TRANSCRIBABLE_TEXT

    async def test_backend_error_dict_becomes_an_error_placeholder(self) -> None:
        error = build_response(
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
            "chronotranscriber.methods.pages.extract_transcribed_text", _boom
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
