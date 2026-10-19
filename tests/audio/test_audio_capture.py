"""Audio attempts are recorded with the body that came over HTTP.

``OpenAIAudioBackend`` runs on the real ``AsyncOpenAI`` client and
``GoogleAudioBackend`` on the chat model of the client factory; a
``MockTransport`` answers behind their httpx clients
(``tests.llm.network``), so no request leaves the machine.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.capture import use_recorder
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.methods.audio_api import (
    GoogleAudioBackend,
    OpenAIAudioBackend,
)
from chronotranscriber.settings import RetrySettings
from chronotranscriber.sources.audio.constants import NO_TRANSCRIBABLE_TEXT
from chronotranscriber.sources.audio.stream import AudioChunkPayload
from tests.conftest import make_settings
from tests.llm.network import MockNetwork, Recorder, bodies

AUDIO = b"ID3\x04\x00chunk-bytes" * 64

TRANSCRIPT = (
    b'{"text": "Gr\xc3\xbc\xc3\x9fe aus  Z\xc3\xbcrich",  "usage": {"type": "tokens",'
    b' "input_tokens": 120, "output_tokens": 9, "total_tokens": 129}}\n'
)
MALFORMED = b'{"text": "Gr\xc3\xbc\xc3\x9fe aus Z\xc3\xbc'
REFUSAL = (
    b'{"error": {"message": "This request was rejected by the content policy.",'
    b' "type": "invalid_request_error", "param": null,'
    b' "code": "content_policy_violation"}}'
)
SERVER_ERROR = (
    b'{"error": {"message": "The server had an error.", "type": "server_error",'
    b' "param": null, "code": null}}'
)
GEMINI = (
    b'{"candidates": [{"content": {"parts": [{"text": "Gr\xc3\xbc\xc3\x9fe"}],'
    b' "role": "model"}, "finishReason": "STOP", "index": 0}],'
    b' "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 4,'
    b' "totalTokenCount": 304}, "modelVersion": "gemini-3-flash"}'
)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> MockNetwork:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-audio")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    real_sleep = asyncio.sleep

    async def no_wait(delay: float, result: Any = None) -> Any:
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    return MockNetwork(recorder).install(monkeypatch)


def _env() -> CallEnv:
    return CallEnv.create(
        make_settings(retry=RetrySettings(max_attempts=3), rate_limits=((100_000, 1),))
    )


def _payload(tmp_path: Path) -> AudioChunkPayload:
    chunk = tmp_path / "rec_chunk_0001.mp3"
    chunk.write_bytes(AUDIO)
    return AudioChunkPayload(
        index=0,
        image_name=chunk.name,
        path=chunk,
        mime_type="audio/mp3",
        source_file=str(tmp_path / "rec.mp3"),
        byte_size=len(AUDIO),
        sha256=hashlib.sha256(AUDIO).hexdigest(),
    )


def _transcribe(backend: Any, tmp_path: Path, recorder: Recorder) -> dict[str, Any]:
    async def main() -> dict[str, Any]:
        try:
            with use_recorder(recorder):
                result: dict[str, Any] = await backend.transcribe_audio_chunk(
                    _payload(tmp_path)
                )
                return result
        finally:
            await backend.close()

    return asyncio.run(main())


class TestOpenAI:
    """The oracle cases on the transcription endpoint."""

    def test_a_success_is_one_record_with_the_verbatim_body(
        self, network: MockNetwork, recorder: Recorder, tmp_path: Path
    ) -> None:
        network.answers = [(200, TRANSCRIPT)]
        backend = OpenAIAudioBackend("gpt-transcribe", env=_env())

        response = _transcribe(backend, tmp_path, recorder)

        assert response["output_text"] == "Grüße aus  Zürich"
        (record,) = recorder.records
        assert bodies(recorder.records) == [TRANSCRIPT]
        assert (record.provider, record.model, record.role) == (
            "openai",
            "gpt-transcribe",
            "audio",
        )
        assert (record.raw_kind, record.http_status, record.attempt) == ("http", 200, 1)
        assert record.label == "rec_chunk_0001.mp3"
        assert record.usage is not None and record.usage["total_tokens"] == 129
        assert record.request["model"] == "gpt-transcribe"
        assert record.request["file"][1] == f"<omitted {len(AUDIO)} bytes>"
        assert network.violations == []

    def test_a_malformed_output_is_one_failed_record(
        self, network: MockNetwork, recorder: Recorder, tmp_path: Path
    ) -> None:
        """The audio ladder does not retry a body it cannot parse."""
        network.answers = [(200, MALFORMED)]
        backend = OpenAIAudioBackend("gpt-transcribe", env=_env())

        response = _transcribe(backend, tmp_path, recorder)

        assert "error" in response
        (record,) = recorder.records
        assert bodies(recorder.records) == [MALFORMED]
        assert (record.http_status, record.error_type) == (200, "JSONDecodeError")

    def test_a_refusal_is_one_record(
        self, network: MockNetwork, recorder: Recorder, tmp_path: Path
    ) -> None:
        network.answers = [(400, REFUSAL)]
        backend = OpenAIAudioBackend("gpt-transcribe", env=_env())

        response = _transcribe(backend, tmp_path, recorder)

        assert "content policy" in response["error"]
        (record,) = recorder.records
        assert bodies(recorder.records) == [REFUSAL]
        assert (record.http_status, record.error_type) == (400, "BadRequestError")

    def test_a_500_then_a_200_is_two_records_stored_before_the_retry(
        self, network: MockNetwork, recorder: Recorder, tmp_path: Path
    ) -> None:
        network.answers = [(500, SERVER_ERROR), (200, TRANSCRIPT)]
        backend = OpenAIAudioBackend("gpt-transcribe", env=_env())

        response = _transcribe(backend, tmp_path, recorder)

        assert response["output_text"] == "Grüße aus  Zürich"
        assert bodies(recorder.records) == [SERVER_ERROR, TRANSCRIPT]
        assert [r.http_status for r in recorder.records] == [500, 200]
        assert [r.attempt for r in recorder.records] == [1, 2]
        assert network.violations == []
        assert network.scoped == [True, True]

    def test_without_a_recorder_nothing_is_captured(
        self, network: MockNetwork, recorder: Recorder, tmp_path: Path
    ) -> None:
        network.answers = [(200, b'{"text": ""}')]
        backend = OpenAIAudioBackend("gpt-transcribe", env=_env())

        async def main() -> dict[str, Any]:
            result: dict[str, Any] = await backend.transcribe_audio_chunk(
                _payload(tmp_path)
            )
            await backend.close()
            return result

        response = asyncio.run(main())

        assert response["output_text"] == NO_TRANSCRIBABLE_TEXT
        assert recorder.records == []
        assert network.scoped == [False]


def test_the_google_backend_records_through_the_caller(
    network: MockNetwork, recorder: Recorder, tmp_path: Path
) -> None:
    network.answers = [(500, SERVER_ERROR), (200, GEMINI)]
    backend = GoogleAudioBackend("gemini-3-flash", env=_env())

    response = _transcribe(backend, tmp_path, recorder)

    assert response["output_text"] == "Grüße"
    assert bodies(recorder.records) == [SERVER_ERROR, GEMINI]
    assert [(r.role, r.http_status) for r in recorder.records] == [
        ("audio", 500),
        ("audio", 200),
    ]
    request = str(recorder.records[1].request)
    assert "<omitted base64" in request
    assert base64.urlsafe_b64encode(AUDIO).decode()[:64] not in request
    assert network.violations == []
