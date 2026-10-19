"""Tests for the audio-API method's OpenAI and Google backends.

``AsyncOpenAI`` and the Google backend's chat model are replaced with fakes
and retry sleeps return at once, so nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from chronotranscriber import ext
from chronotranscriber.common.retry import Decision
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.capabilities import CapabilityError
from chronotranscriber.methods import audio_api
from chronotranscriber.methods.audio_api import (
    GOOGLE_AUDIO_PROMPT,
    GoogleAudioBackend,
    OpenAIAudioBackend,
    open_audio_backend,
)
from chronotranscriber.settings import AudioApi, RetrySettings, Timeouts
from chronotranscriber.sources.audio.constants import (
    GEMINI_INLINE_LIMIT_BYTES,
    NO_TRANSCRIBABLE_TEXT,
)
from chronotranscriber.sources.audio.stream import AudioChunkPayload
from chronotranscriber.spec import DEFAULT_GOOGLE_AUDIO_MODEL
from tests.conftest import make_settings, make_spec
from tests.llm import helpers


class _FakeUsage:
    def __init__(self, usage_type: str, **fields: int) -> None:
        self.type = usage_type
        for name, value in fields.items():
            setattr(self, name, value)


def _tokens(input_tokens: int, output_tokens: int, total_tokens: int) -> _FakeUsage:
    return _FakeUsage(
        "tokens",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


class _FakeResponse:
    def __init__(self, text: str = "", usage: Any = None) -> None:
        self.text = text
        self.usage = usage


class _FakeTranscriptions:
    def __init__(self, client: _FakeAsyncOpenAI) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        self._client.calls.append(kwargs)
        outcomes = _FakeAsyncOpenAI.outcome_queue
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeAsyncOpenAI:
    """Fake ``AsyncOpenAI`` capturing construction kwargs and API calls."""

    instances: list[_FakeAsyncOpenAI] = []
    outcome_queue: list[Any] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.audio = type("_Audio", (), {})()
        self.audio.transcriptions = _FakeTranscriptions(self)
        _FakeAsyncOpenAI.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class _HttpError(Exception):
    """Minimal SDK-shaped error carrying an HTTP status, body and headers."""

    def __init__(
        self,
        message: str,
        status_code: int,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.headers = headers


@pytest.fixture(autouse=True)
def _isolate_backends(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fake the SDK client; record retry sleeps."""
    _FakeAsyncOpenAI.instances = []
    _FakeAsyncOpenAI.outcome_queue = [_FakeResponse("default transcript")]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-audio")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.setattr(audio_api, "AsyncOpenAI", _FakeAsyncOpenAI)

    slept: list[float] = []

    async def _sleep(delay: float, result: Any = None) -> Any:
        slept.append(delay)
        return result

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return slept


def _env(**changes: Any) -> CallEnv:
    changes.setdefault("retry", RetrySettings(max_attempts=3))
    return CallEnv.create(make_settings(**changes))


def _openai(
    model: str = "gpt-transcribe",
    *,
    env: CallEnv | None = None,
    language: str | None = None,
    **extras: Any,
) -> OpenAIAudioBackend:
    return OpenAIAudioBackend(
        model, env=env or _env(), language=language, extras=AudioApi(**extras)
    )


def _google(
    model: str = DEFAULT_GOOGLE_AUDIO_MODEL,
    *,
    env: CallEnv | None = None,
    language: str | None = None,
) -> GoogleAudioBackend:
    return GoogleAudioBackend(model, env=env or _env(), language=language)


def _payload(
    tmp_path: Path,
    *,
    name: str = "rec_chunk_0001.mp3",
    data: bytes = b"chunk-bytes",
    byte_size: int | None = None,
    index: int = 0,
) -> AudioChunkPayload:
    chunk = tmp_path / name
    chunk.write_bytes(data)
    return AudioChunkPayload(
        index=index,
        image_name=name,
        path=chunk,
        mime_type="audio/mp3",
        source_file=str(tmp_path / "rec.mp3"),
        byte_size=len(data) if byte_size is None else byte_size,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _queue(*outcomes: Any) -> None:
    """Queue the outcomes of the next ``create`` calls; the last one repeats."""
    _FakeAsyncOpenAI.outcome_queue = list(outcomes)


def _calls() -> list[dict[str, Any]]:
    return [call for client in _FakeAsyncOpenAI.instances for call in client.calls]


@pytest.mark.unit
class TestOpenAudioBackend:
    def test_the_openai_default_model(self) -> None:
        spec = make_spec(type="audio", method="audio-api")
        backend = open_audio_backend(spec, _env())
        assert isinstance(backend, OpenAIAudioBackend)
        assert backend.model == "gpt-transcribe"

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [({}, DEFAULT_GOOGLE_AUDIO_MODEL), ({"model": "gemini-3-pro"}, "gemini-3-pro")],
        ids=["default-model", "named-model"],
    )
    def test_google_takes_its_default_or_the_named_model(
        self, monkeypatch: pytest.MonkeyPatch, flags: dict[str, str], expected: str
    ) -> None:
        _gemini(monkeypatch)
        spec = make_spec(type="audio", method="audio-api", provider="google", **flags)
        assert spec.model.model == expected
        backend = open_audio_backend(spec, _env())
        assert isinstance(backend, GoogleAudioBackend)
        assert backend.model == expected

    def test_the_language_and_settings_extras_reach_openai(self) -> None:
        spec = make_spec(
            type="audio", method="audio-api", model="whisper-1", language="de"
        )
        env = _env(audio_api=AudioApi(prompt="Grain prices."))
        backend = open_audio_backend(spec, env)
        assert isinstance(backend, OpenAIAudioBackend)
        assert (backend.model, backend.language) == ("whisper-1", "de")
        assert backend.extras.prompt == "Grain prices."


@pytest.mark.unit
class TestOpenAIConstruction:
    def test_the_client_is_built_at_once_without_sdk_retries(self) -> None:
        _openai()
        kwargs = _FakeAsyncOpenAI.instances[-1].init_kwargs
        assert kwargs["max_retries"] == 0
        assert kwargs["api_key"] == "sk-test-audio"

    def test_the_timeout_is_per_phase(self) -> None:
        import httpx

        _openai(env=_env(timeouts=Timeouts(request=600.0)))
        timeout = _FakeAsyncOpenAI.instances[-1].init_kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert (timeout.read, timeout.connect, timeout.write, timeout.pool) == (
            600.0,
            10.0,
            30.0,
            30.0,
        )

    def test_a_non_audio_model_is_refused(self) -> None:
        with pytest.raises(CapabilityError):
            _openai("gpt-5")

    def test_an_unset_key_variable_fails_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY")
        with pytest.raises(OSError, match="OPENAI_API_KEY"):
            _openai()


@pytest.mark.unit
class TestOpenAIRequestKwargs:
    async def test_gpt_transcribe_sends_lists_via_extra_body(
        self, tmp_path: Path
    ) -> None:
        backend = _openai(languages=("de", "fr"), keywords=("Zurich",), language="it")
        await backend.transcribe_audio_chunk(_payload(tmp_path))
        kwargs = _calls()[0]
        assert kwargs["extra_body"] == {
            "languages": ["de", "fr"],
            "keywords": ["Zurich"],
        }
        assert "language" not in kwargs

    async def test_gpt_transcribe_sends_the_run_language_without_a_list(
        self, tmp_path: Path
    ) -> None:
        await _openai(language="de").transcribe_audio_chunk(_payload(tmp_path))
        kwargs = _calls()[0]
        assert kwargs["extra_body"] == {"languages": ["de"]}
        assert "language" not in kwargs

    async def test_gpt_transcribe_omits_empty_lists(self, tmp_path: Path) -> None:
        await _openai(languages=("", "")).transcribe_audio_chunk(_payload(tmp_path))
        assert "extra_body" not in _calls()[0]

    @pytest.mark.parametrize("model", ["whisper-1", "gpt-4o-transcribe"])
    async def test_singular_models_use_the_scalar_language(
        self, tmp_path: Path, model: str
    ) -> None:
        backend = _openai(model, language="de", languages=("fr",))
        await backend.transcribe_audio_chunk(_payload(tmp_path))
        kwargs = _calls()[0]
        assert kwargs["language"] == "de"
        assert "extra_body" not in kwargs

    async def test_singular_models_omit_an_unset_language(self, tmp_path: Path) -> None:
        await _openai("whisper-1").transcribe_audio_chunk(_payload(tmp_path))
        assert "language" not in _calls()[0]

    async def test_the_file_tuple_carries_name_bytes_and_mime(
        self, tmp_path: Path
    ) -> None:
        payload = _payload(tmp_path)
        await _openai().transcribe_audio_chunk(payload)
        assert _calls()[0]["file"] == (payload.image_name, b"chunk-bytes", "audio/mp3")

    async def test_the_prompt_is_sent_when_set(self, tmp_path: Path) -> None:
        await _openai(prompt=" Culinary terms. ").transcribe_audio_chunk(
            _payload(tmp_path)
        )
        assert _calls()[0]["prompt"] == "Culinary terms."

    async def test_a_blank_prompt_is_omitted(self, tmp_path: Path) -> None:
        await _openai(prompt="   ").transcribe_audio_chunk(_payload(tmp_path))
        assert "prompt" not in _calls()[0]

    def test_a_long_whisper1_prompt_is_truncated_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _openai("whisper-1", prompt="a" * (224 * 4 + 500))
        with caplog.at_level("WARNING"):
            assert len(backend.resolve_prompt()) == 224 * 4
        assert "whisper-1 caps the prompt" in caplog.text

    def test_other_models_keep_a_long_prompt(self) -> None:
        prompt = "c" * (224 * 4 + 500)
        assert _openai(prompt=prompt).resolve_prompt() == prompt


@pytest.mark.unit
class TestOpenAIResponses:
    async def test_a_successful_transcript(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("  Some spoken words.  "))
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert result["output_text"] == "Some spoken words."
        assert "error" not in result
        assert result["metadata"] == {
            "model": "gpt-transcribe",
            "provider": "openai",
            "chunk_index": 0,
        }

    @pytest.mark.parametrize("text", ["", "   \n\t "])
    async def test_an_empty_transcript_collapses_to_the_sentinel(
        self, tmp_path: Path, text: str
    ) -> None:
        _queue(_FakeResponse(text))
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert result["output_text"] == NO_TRANSCRIBABLE_TEXT

    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            (_tokens(120, 30, 150), (120, 30, 150)),
            (_tokens(7, 3, 0), (7, 3, 10)),
            (_FakeUsage("duration", seconds=42), (0, 0, 0)),
            (None, (0, 0, 0)),
        ],
    )
    async def test_token_usage_is_reported_and_recorded(
        self, tmp_path: Path, usage: Any, expected: tuple[int, int, int]
    ) -> None:
        _queue(_FakeResponse("text", usage))
        env = _env()
        result = await _openai(env=env).transcribe_audio_chunk(_payload(tmp_path))
        reported = result["usage"]
        assert (
            reported["input_tokens"],
            reported["output_tokens"],
            reported["total_tokens"],
        ) == expected
        assert env.usage.get("audio") == Usage(*expected)

    async def test_a_disallowed_extension_is_refused_before_the_api(
        self, tmp_path: Path
    ) -> None:
        result = await _openai().transcribe_audio_chunk(
            _payload(tmp_path, name="rec_chunk_0001.flac")
        )
        assert ".flac" in result["error"]
        assert result["output_text"] == ""
        assert _calls() == []

    async def test_an_api_failure_is_reported_in_the_error_key(
        self, tmp_path: Path
    ) -> None:
        _queue(_HttpError("bad request", status_code=400))
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert "bad request" in result["error"]
        assert result["output_text"] == ""
        assert result["metadata"]["chunk_index"] == 0

    async def test_a_read_failure_is_reported_rather_than_raised(
        self, tmp_path: Path
    ) -> None:
        payload = _payload(tmp_path)
        payload.path.unlink()
        result = await _openai().transcribe_audio_chunk(payload)
        assert "error" in result


@pytest.mark.unit
class TestOpenAIRetries:
    @pytest.mark.parametrize("status", [429, 500, 503])
    async def test_transient_statuses_are_retried(
        self, tmp_path: Path, status: int
    ) -> None:
        _queue(_HttpError("try again", status_code=status), _FakeResponse("recovered"))
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert result["output_text"] == "recovered"
        assert len(_calls()) == 2

    async def test_a_retry_honors_the_retry_after_header(
        self, tmp_path: Path, _isolate_backends: list[float]
    ) -> None:
        _queue(
            _HttpError("slow down", status_code=429, headers={"retry-after": "7"}),
            _FakeResponse("after the floor"),
        )
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert result["output_text"] == "after the floor"
        assert _isolate_backends[0] == 7.0

    async def test_retries_stop_at_the_attempt_limit(self, tmp_path: Path) -> None:
        _queue(_HttpError("still down", status_code=503))
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert "error" in result
        assert len(_calls()) == 3

    async def test_a_client_error_is_not_retried(self, tmp_path: Path) -> None:
        _queue(_HttpError("unauthorized", status_code=401))
        await _openai().transcribe_audio_chunk(_payload(tmp_path))
        assert len(_calls()) == 1


class _RecordingAttempt(ext.Attempt[Any]):
    """An attempt that logs entry, commits and errors."""

    def __init__(
        self, real: ext.Attempt[Any], log: list[Any], decision: Decision | None
    ) -> None:
        super().__init__(real.identity, real.estimate, real._build_client)
        self.log = log
        self.decision = decision

    async def __aenter__(self) -> _RecordingAttempt:
        self.log.append("enter")
        await super().__aenter__()
        return self

    def commit(self, usage: Usage | None) -> None:
        self.log.append(("commit", usage))
        super().commit(usage)

    def on_error(self, exc: BaseException) -> Decision:
        self.log.append(("on_error", type(exc).__name__))
        decided = super().on_error(exc)
        return self.decision or decided


@pytest.fixture
def recording_ext(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Decision | None], list[_RecordingAttempt]]:
    """Return an installer of a recording ``ext.attempt``."""

    def install(decision: Decision | None = None) -> list[_RecordingAttempt]:
        attempts: list[_RecordingAttempt] = []
        original = ext.attempt

        def attempt(*args: Any, **kwargs: Any) -> _RecordingAttempt:
            recorded = _RecordingAttempt(original(*args, **kwargs), [], decision)
            attempts.append(recorded)
            return recorded

        monkeypatch.setattr(ext, "attempt", attempt)
        return attempts

    return install


def _entered(attempts: list[_RecordingAttempt]) -> list[_RecordingAttempt]:
    return [attempt for attempt in attempts if "enter" in attempt.log]


@pytest.mark.unit
class TestOpenAIAttempts:
    async def test_one_attempt_per_request_with_a_commit(
        self,
        tmp_path: Path,
        recording_ext: Callable[[Decision | None], list[_RecordingAttempt]],
    ) -> None:
        attempts = recording_ext(None)
        _queue(_FakeResponse("text", _tokens(5, 2, 7)))
        env = _env()
        await _openai(env=env).transcribe_audio_chunk(_payload(tmp_path))

        (call,) = _entered(attempts)
        assert call.log == ["enter", ("commit", Usage(5, 2, 7))]
        assert call.released
        assert call.identity == ext.Identity(
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            role="audio",
            model="gpt-transcribe",
            kind="audio",
        )
        assert env.usage.get("audio") == Usage(5, 2, 7)

    async def test_a_retried_request_counts_one_attempt_per_try(
        self,
        tmp_path: Path,
        recording_ext: Callable[[Decision | None], list[_RecordingAttempt]],
    ) -> None:
        attempts = recording_ext(None)
        _queue(_HttpError("busy", status_code=503), _FakeResponse("second try"))
        await _openai().transcribe_audio_chunk(_payload(tmp_path))

        entered = _entered(attempts)
        assert len(entered) == len(_calls()) == 2
        assert entered[0].log == ["enter", ("on_error", "_HttpError")]
        assert entered[1].log[0] == "enter"
        assert entered[1].log[1][0] == "commit"
        assert all(attempt.released for attempt in entered)

    async def test_the_on_error_decision_ends_the_retries(
        self,
        tmp_path: Path,
        recording_ext: Callable[[Decision | None], list[_RecordingAttempt]],
    ) -> None:
        attempts = recording_ext(Decision.FAIL)
        _queue(_HttpError("busy", status_code=503), _FakeResponse("never"))
        result = await _openai().transcribe_audio_chunk(_payload(tmp_path))

        assert "busy" in result["error"]
        assert len(_entered(attempts)) == len(_calls()) == 1

    async def test_usage_recovered_from_a_failure_is_recorded(
        self,
        tmp_path: Path,
        recording_ext: Callable[[Decision | None], list[_RecordingAttempt]],
    ) -> None:
        recording_ext(None)
        body = {"usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}}
        _queue(_HttpError("bad", status_code=400, body=body))
        env = _env()
        await _openai(env=env).transcribe_audio_chunk(_payload(tmp_path))
        assert env.usage.get("audio").total_tokens == 5

    async def test_close_disposes_the_client(self) -> None:
        backend = _openai()
        await backend.close()
        assert _FakeAsyncOpenAI.instances[-1].closed is True


def _gemini(monkeypatch: pytest.MonkeyPatch, *answers: Any) -> helpers.Script:
    """Answer the Google backend's chat model from *answers* (the last repeats)."""
    return helpers.install(monkeypatch, list(answers or ["stub transcript"]))


def _file_block(request: Any) -> dict[str, Any]:
    (block,) = [b for b in request.messages[1].content if b.get("type") == "file"]
    return dict(block)


@pytest.mark.unit
class TestGoogle:
    def test_the_chat_model_takes_the_audio_knobs(self) -> None:
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as cls:
            _google("gemini-3-flash")
        kwargs = dict(cls.call_args.kwargs)
        assert set(kwargs.pop("client_args")) == {"event_hooks", "transport"}
        assert kwargs == {
            "google_api_key": "google-test",
            "model": "gemini-3-flash",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "max_tokens": 32768,
            "timeout": None,
            "max_retries": 0,
        }

    def test_a_non_audio_model_is_refused(self) -> None:
        with pytest.raises(CapabilityError):
            _google("gemma-4-31b-it")

    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            ("German", f"{GOOGLE_AUDIO_PROMPT}\nThe recording is in German."),
            (None, GOOGLE_AUDIO_PROMPT),
        ],
        ids=["with-language", "default"],
    )
    async def test_the_prompt_carries_the_language_when_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        language: str | None,
        expected: str,
    ) -> None:
        script = _gemini(monkeypatch)
        await _google(language=language).transcribe_audio_chunk(_payload(tmp_path))
        assert script.requests[0].messages[0].content == expected

    async def test_audio_is_sent_base64_encoded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = _gemini(monkeypatch)
        await _google().transcribe_audio_chunk(_payload(tmp_path, data=b"abcdef"))
        (request,) = script.requests
        block = _file_block(request)
        encoded = base64.b64encode(b"abcdef").decode("utf-8")
        assert encoded in (block.get("base64"), block.get("data"))
        assert block["mime_type"] == "audio/mp3"
        assert request.kwargs == {"max_output_tokens": 32768}
        assert request.structured is None

    async def test_a_disallowed_extension_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = _gemini(monkeypatch)
        result = await _google().transcribe_audio_chunk(
            _payload(tmp_path, name="rec_chunk_0004.webm", index=3)
        )
        assert ".webm" in result["error"]
        assert result["metadata"]["chunk_index"] == 3
        assert script.requests == []

    @pytest.mark.parametrize(
        ("size", "sent"),
        [(GEMINI_INLINE_LIMIT_BYTES + 1, False), (GEMINI_INLINE_LIMIT_BYTES, True)],
    )
    async def test_the_inline_limit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: int, sent: bool
    ) -> None:
        script = _gemini(monkeypatch)
        result = await _google().transcribe_audio_chunk(
            _payload(tmp_path, byte_size=size)
        )
        assert ("error" not in result) is sent
        assert len(script.requests) == int(sent)

    async def test_a_failed_call_becomes_an_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _gemini(monkeypatch, _HttpError("invalid audio", 400))
        result = await _google().transcribe_audio_chunk(_payload(tmp_path))
        assert result["error"] == "invalid audio"
        assert result["output_text"] == ""
        assert result["metadata"]["chunk_index"] == 0

    async def test_a_transient_error_is_retried(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = _gemini(monkeypatch, _HttpError("overloaded", 503), "Spoken words.")
        result = await _google().transcribe_audio_chunk(_payload(tmp_path))
        assert len(script.requests) == 2
        assert result["output_text"] == "Spoken words."

    async def test_content_usage_and_metadata_are_mapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        answer = helpers.message(
            "Spoken words.",
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 40,
                "total_tokens": 240,
                "input_token_details": {"cache_read": 64},
            },
            response_metadata={"finish_reason": "STOP"},
        )
        _gemini(monkeypatch, answer)
        env = _env()
        response = await _google(env=env).transcribe_audio_chunk(_payload(tmp_path))
        assert response["output_text"] == "Spoken words."
        assert response["usage"] == Usage(200, 40, 240, cached_tokens=64).as_dict()
        assert response["metadata"]["finish_reason"] == "STOP"
        assert response["metadata"]["model"] == DEFAULT_GOOGLE_AUDIO_MODEL
        assert env.usage.get("audio") == Usage(200, 40, 240, cached_tokens=64)

    async def test_an_empty_transcript_is_the_no_text_placeholder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _gemini(monkeypatch, "")
        response = await _google().transcribe_audio_chunk(_payload(tmp_path))
        assert response["output_text"] == NO_TRANSCRIBABLE_TEXT
        assert "error" not in response

    async def test_transcription_not_possible_reaches_the_shared_placeholder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from chronotranscriber.llm.response_parsing import (
            detect_transcription_cause,
            extract_transcribed_text,
        )

        content = json.dumps(
            {
                "transcription": "",
                "no_transcribable_text": False,
                "transcription_not_possible": True,
            }
        )
        _gemini(monkeypatch, content)
        response = await _google().transcribe_audio_chunk(_payload(tmp_path))
        assert response["parsed"]["transcription_not_possible"] is True
        text = extract_transcribed_text(response, "rec_chunk_0001.mp3")
        assert detect_transcription_cause(text) == "not_possible"

    async def test_close_releases_the_clients_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _gemini(monkeypatch)
        backend = _google()
        await backend.close()
        await backend.close()
        assert backend._caller._clients == {}
