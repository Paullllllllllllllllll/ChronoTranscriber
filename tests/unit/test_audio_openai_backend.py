"""Tests for modules.audio.backends.openai_audio.OpenAIAudioBackend.

The ``AsyncOpenAI`` class is replaced with a fake, the shared rate limiter and
the daily token ledger are stubbed, and the retry backoff is flattened, so
nothing here touches the network, the real ledger, or a real config file.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from modules.audio import retry as audio_retry
from modules.audio.audio_stream import AudioChunkPayload
from modules.audio.backends import openai_audio
from modules.audio.backends.openai_audio import (
    DEFAULT_AUDIO_REQUEST_TIMEOUT_S,
    DEFAULT_OPENAI_AUDIO_MODEL,
    OpenAIAudioBackend,
)
from modules.audio.constants import NO_TRANSCRIBABLE_TEXT
from modules.config.capabilities import CapabilityError

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, usage_type: str, **fields: int) -> None:
        self.type = usage_type
        for name, value in fields.items():
            setattr(self, name, value)


class _FakeResponse:
    def __init__(self, text: str = "", usage: Any = None) -> None:
        self.text = text
        self.usage = usage


class _FakeTranscriptions:
    def __init__(self, client: _FakeAsyncOpenAI) -> None:
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        self._client.calls.append(kwargs)
        outcome = (
            self._client.outcomes.pop(0)
            if len(self._client.outcomes) > 1
            else self._client.outcomes[0]
        )
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
        self.outcomes: list[Any] = list(_FakeAsyncOpenAI.outcome_queue) or [
            _FakeResponse("default transcript")
        ]
        self.audio = type("_Audio", (), {})()
        self.audio.transcriptions = _FakeTranscriptions(self)
        _FakeAsyncOpenAI.instances.append(self)

    async def close(self) -> None:
        self.closed = True


class _RecordingTracker:
    """Stand-in for the daily token tracker; records commits in memory."""

    def __init__(self) -> None:
        self.commits: list[dict[str, Any]] = []

    def add_tokens(
        self, total: int, *, provider: str, key_env: str | None, model: str
    ) -> None:
        self.commits.append(
            {
                "total": total,
                "provider": provider,
                "key_env": key_env,
                "model": model,
            }
        )


class _StubLimiter:
    def __init__(self) -> None:
        self.errors: list[bool] = []
        self.successes = 0

    def report_error(self, is_rate_limit: bool = False) -> None:
        self.errors.append(is_rate_limit)

    def report_success(self) -> None:
        self.successes += 1


class _HttpError(Exception):
    """Minimal SDK-shaped error carrying an HTTP status, body, and headers."""

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_openai_audio_backend(
    monkeypatch: pytest.MonkeyPatch, no_api_key_remap: None
) -> Generator[_RecordingTracker]:
    """Stub every out-of-process seam the backend and its retry policy touch."""
    _FakeAsyncOpenAI.instances = []
    _FakeAsyncOpenAI.outcome_queue = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-audio")
    monkeypatch.setattr(openai_audio, "AsyncOpenAI", _FakeAsyncOpenAI)

    # Flat backoff and a fixed attempt budget: no config read, no real sleeping.
    monkeypatch.setattr(audio_retry, "_BACKOFF_INITIAL_S", 0)
    monkeypatch.setattr(audio_retry, "_BACKOFF_MAX_S", 0)
    monkeypatch.setattr(audio_retry, "load_max_retries", lambda: 3)

    from modules.infra import rate_limit

    async def _await_capacity(_limiter: Any) -> float:
        return 0.0

    monkeypatch.setattr(
        rate_limit, "get_shared_rate_limiter", lambda _p: _StubLimiter()
    )
    monkeypatch.setattr(rate_limit, "await_capacity", _await_capacity)

    tracker = _RecordingTracker()
    from modules.infra import token_budget

    monkeypatch.setattr(token_budget, "get_token_tracker", lambda: tracker)
    return tracker


@pytest.fixture
def token_tracker(
    _isolate_openai_audio_backend: _RecordingTracker,
) -> _RecordingTracker:
    """The in-memory tracker the backend commits to."""
    return _isolate_openai_audio_backend


def _audio_config(**openai_settings: Any) -> dict[str, Any]:
    return {"audio_transcription": {"openai": openai_settings}}


def _payload(tmp_path: Path, name: str = "rec_chunk_0001.mp3") -> AudioChunkPayload:
    chunk = tmp_path / name
    chunk.write_bytes(b"chunk-bytes")
    return AudioChunkPayload(
        index=0,
        image_name=name,
        path=chunk,
        mime_type="audio/mp3",
        source_file=str(tmp_path / "rec.mp3"),
        byte_size=len(b"chunk-bytes"),
        sha256=hashlib.sha256(b"chunk-bytes").hexdigest(),
    )


def _backend(**openai_settings: Any) -> OpenAIAudioBackend:
    return OpenAIAudioBackend(_audio_config(**openai_settings), {})


def _queue(*outcomes: Any) -> None:
    """Queue the outcomes the next fake client's ``create`` will produce."""
    _FakeAsyncOpenAI.outcome_queue = list(outcomes)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstruction:
    def test_defaults(self) -> None:
        backend = _backend()
        assert backend.provider_name == "openai"
        assert backend.model == DEFAULT_OPENAI_AUDIO_MODEL

    def test_configured_model_is_used(self) -> None:
        assert _backend(model="whisper-1").model == "whisper-1"

    def test_client_is_built_with_retries_disabled(self) -> None:
        _backend()
        assert _FakeAsyncOpenAI.instances[-1].init_kwargs["max_retries"] == 0

    def test_default_request_timeout(self) -> None:
        _backend()
        assert (
            _FakeAsyncOpenAI.instances[-1].init_kwargs["timeout"]
            == DEFAULT_AUDIO_REQUEST_TIMEOUT_S
        )

    def test_configured_request_timeout_is_honored(self) -> None:
        OpenAIAudioBackend(
            _audio_config(),
            {"concurrency": {"transcription": {"request_timeout": 42}}},
        )
        assert _FakeAsyncOpenAI.instances[-1].init_kwargs["timeout"] == 42.0

    def test_malformed_timeout_falls_back_to_the_default(self) -> None:
        OpenAIAudioBackend(
            _audio_config(),
            {"concurrency": {"transcription": {"request_timeout": "nonsense"}}},
        )
        assert (
            _FakeAsyncOpenAI.instances[-1].init_kwargs["timeout"]
            == DEFAULT_AUDIO_REQUEST_TIMEOUT_S
        )

    def test_non_audio_model_is_refused(self) -> None:
        with pytest.raises(CapabilityError):
            _backend(model="gpt-5")

    def test_key_env_is_resolved_from_the_default_mapping(self) -> None:
        assert _backend()._key_env == "OPENAI_API_KEY"


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequestKwargs:
    async def test_gpt_transcribe_sends_lists_via_extra_body(
        self, tmp_path: Path
    ) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(
            model="gpt-transcribe", languages=["de", "fr"], keywords=["Zurich"]
        )
        await backend.transcribe_chunk(_payload(tmp_path))
        kwargs = _FakeAsyncOpenAI.instances[-1].calls[0]
        assert kwargs["extra_body"] == {
            "languages": ["de", "fr"],
            "keywords": ["Zurich"],
        }
        assert "languages" not in kwargs
        assert "keywords" not in kwargs
        assert "language" not in kwargs

    async def test_gpt_transcribe_omits_extra_body_when_both_lists_are_empty(
        self, tmp_path: Path
    ) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(model="gpt-transcribe", languages=[], keywords=[])
        await backend.transcribe_chunk(_payload(tmp_path))
        assert "extra_body" not in _FakeAsyncOpenAI.instances[-1].calls[0]

    async def test_gpt_transcribe_drops_blank_list_entries(
        self, tmp_path: Path
    ) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(model="gpt-transcribe", languages=["de", "", None])
        await backend.transcribe_chunk(_payload(tmp_path))
        kwargs = _FakeAsyncOpenAI.instances[-1].calls[0]
        assert kwargs["extra_body"]["languages"] == ["de"]

    async def test_gpt_transcribe_ignores_the_scalar_language(
        self, tmp_path: Path
    ) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(model="gpt-transcribe", language="de")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert "language" not in _FakeAsyncOpenAI.instances[-1].calls[0]

    @pytest.mark.parametrize("model", ["whisper-1", "gpt-4o-transcribe"])
    async def test_singular_models_use_the_scalar_language(
        self, tmp_path: Path, model: str
    ) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(model=model, language="de", languages=["fr"])
        await backend.transcribe_chunk(_payload(tmp_path))
        kwargs = _FakeAsyncOpenAI.instances[-1].calls[0]
        assert kwargs["language"] == "de"
        assert "extra_body" not in kwargs

    @pytest.mark.parametrize("model", ["whisper-1", "gpt-4o-mini-transcribe"])
    async def test_singular_models_omit_language_when_unset(
        self, tmp_path: Path, model: str
    ) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(model=model, language="")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert "language" not in _FakeAsyncOpenAI.instances[-1].calls[0]

    async def test_file_tuple_carries_name_bytes_and_mime(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend()
        payload = _payload(tmp_path)
        await backend.transcribe_chunk(payload)
        name, data, mime = _FakeAsyncOpenAI.instances[-1].calls[0]["file"]
        assert name == payload.image_name
        assert data == b"chunk-bytes"
        assert mime == "audio/mp3"

    async def test_temperature_is_forwarded_when_set(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(temperature=0.25)
        await backend.transcribe_chunk(_payload(tmp_path))
        assert _FakeAsyncOpenAI.instances[-1].calls[0]["temperature"] == 0.25

    async def test_temperature_is_omitted_when_null(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(temperature=None)
        await backend.transcribe_chunk(_payload(tmp_path))
        assert "temperature" not in _FakeAsyncOpenAI.instances[-1].calls[0]

    async def test_prompt_is_forwarded_when_set(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(prompt="Historic culinary vocabulary.")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert (
            _FakeAsyncOpenAI.instances[-1].calls[0]["prompt"]
            == "Historic culinary vocabulary."
        )

    async def test_blank_prompt_is_omitted(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("text"))
        backend = _backend(prompt="   ")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert "prompt" not in _FakeAsyncOpenAI.instances[-1].calls[0]


@pytest.mark.unit
class TestWhisper1PromptCap:
    def test_long_prompt_is_truncated_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        long_prompt = "a" * (224 * 4 + 500)
        backend = _backend(model="whisper-1", prompt=long_prompt)
        with caplog.at_level("WARNING"):
            resolved = backend._resolve_prompt()
        assert len(resolved) == 224 * 4
        assert "whisper-1 caps the prompt" in caplog.text

    def test_prompt_at_the_cap_is_untouched(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        prompt = "b" * (224 * 4)
        backend = _backend(model="whisper-1", prompt=prompt)
        with caplog.at_level("WARNING"):
            assert backend._resolve_prompt() == prompt
        assert "caps the prompt" not in caplog.text

    def test_other_models_are_not_truncated(self) -> None:
        long_prompt = "c" * (224 * 4 + 500)
        backend = _backend(model="gpt-transcribe", prompt=long_prompt)
        assert backend._resolve_prompt() == long_prompt


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResponseMapping:
    async def test_successful_transcript(self, tmp_path: Path) -> None:
        _queue(_FakeResponse("  Some spoken words.  "))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == "Some spoken words."
        assert "error" not in result
        assert result["metadata"]["provider"] == "openai"
        assert result["metadata"]["chunk_index"] == 0

    @pytest.mark.parametrize("text", ["", "   \n\t "])
    async def test_empty_transcript_collapses_to_the_sentinel(
        self, tmp_path: Path, text: str
    ) -> None:
        _queue(_FakeResponse(text))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == NO_TRANSCRIBABLE_TEXT
        assert "error" not in result

    async def test_usage_triple_is_reported_in_the_response(
        self, tmp_path: Path
    ) -> None:
        usage = _FakeUsage(
            "tokens", input_tokens=120, output_tokens=30, total_tokens=150
        )
        _queue(_FakeResponse("text", usage))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["usage"] == {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUsageAccounting:
    async def test_token_usage_is_committed_with_the_full_stamp(
        self, tmp_path: Path, token_tracker: _RecordingTracker
    ) -> None:
        usage = _FakeUsage(
            "tokens", input_tokens=100, output_tokens=20, total_tokens=120
        )
        _queue(_FakeResponse("text", usage))
        await _backend(model="gpt-transcribe").transcribe_chunk(_payload(tmp_path))
        assert token_tracker.commits == [
            {
                "total": 120,
                "provider": "openai",
                "key_env": "OPENAI_API_KEY",
                "model": "gpt-transcribe",
            }
        ]

    async def test_missing_total_is_derived_from_the_parts(
        self, tmp_path: Path, token_tracker: _RecordingTracker
    ) -> None:
        usage = _FakeUsage("tokens", input_tokens=7, output_tokens=3, total_tokens=0)
        _queue(_FakeResponse("text", usage))
        await _backend().transcribe_chunk(_payload(tmp_path))
        assert token_tracker.commits[0]["total"] == 10

    async def test_duration_billing_commits_nothing(
        self, tmp_path: Path, token_tracker: _RecordingTracker
    ) -> None:
        _queue(_FakeResponse("text", _FakeUsage("duration", seconds=42)))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert token_tracker.commits == []
        assert result["usage"]["total_tokens"] == 0

    async def test_absent_usage_commits_nothing(
        self, tmp_path: Path, token_tracker: _RecordingTracker
    ) -> None:
        _queue(_FakeResponse("text", None))
        await _backend().transcribe_chunk(_payload(tmp_path))
        assert token_tracker.commits == []

    async def test_zero_tokens_commit_nothing(
        self, tmp_path: Path, token_tracker: _RecordingTracker
    ) -> None:
        usage = _FakeUsage("tokens", input_tokens=0, output_tokens=0, total_tokens=0)
        _queue(_FakeResponse("text", usage))
        await _backend().transcribe_chunk(_payload(tmp_path))
        assert token_tracker.commits == []

    async def test_a_failing_tracker_does_not_break_the_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.infra import token_budget

        def _boom() -> Any:
            raise RuntimeError("ledger unavailable")

        monkeypatch.setattr(token_budget, "get_token_tracker", _boom)
        usage = _FakeUsage("tokens", input_tokens=1, output_tokens=1, total_tokens=2)
        _queue(_FakeResponse("text", usage))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == "text"


# ---------------------------------------------------------------------------
# Guard rails and failures
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGuardsAndFailures:
    async def test_disallowed_extension_short_circuits_before_the_api(
        self, tmp_path: Path
    ) -> None:
        backend = _backend()
        payload = _payload(tmp_path, name="rec_chunk_0001.flac")
        result = await backend.transcribe_chunk(payload)
        assert "error" in result
        assert ".flac" in result["error"]
        assert result["output_text"] == ""
        assert _FakeAsyncOpenAI.instances[-1].calls == []

    async def test_api_failure_is_reported_through_the_error_key(
        self, tmp_path: Path
    ) -> None:
        _queue(_HttpError("bad request", status_code=400))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert "bad request" in result["error"]
        assert result["output_text"] == ""
        assert result["metadata"]["chunk_index"] == 0

    async def test_a_read_failure_is_reported_rather_than_raised(
        self, tmp_path: Path
    ) -> None:
        payload = _payload(tmp_path)
        payload.path.unlink()
        result = await _backend().transcribe_chunk(payload)
        assert "error" in result


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetryPolicy:
    async def test_a_quota_carrying_429_is_retried_like_any_other(
        self, tmp_path: Path
    ) -> None:
        """Classification is status-code-first: the error body does not gate it."""
        _queue(
            _HttpError(
                "You exceeded your current quota",
                status_code=429,
                body={"error": {"type": "insufficient_quota"}},
            ),
            _FakeResponse("recovered after the 429"),
        )
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == "recovered after the 429"
        assert len(_FakeAsyncOpenAI.instances[-1].calls) == 2

    async def test_a_429_retry_honors_the_retry_after_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server-sent Retry-After raises the computed backoff to its floor."""
        slept: list[float] = []

        async def _record_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(audio_retry, "_BACKOFF_MAX_S", 120)
        monkeypatch.setattr(asyncio, "sleep", _record_sleep)
        _queue(
            _HttpError("slow down", status_code=429, headers={"retry-after": "7"}),
            _FakeResponse("after the floor"),
        )
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == "after the floor"
        assert slept == [7.0]

    async def test_plain_500_is_retried_until_it_succeeds(self, tmp_path: Path) -> None:
        _queue(
            _HttpError("upstream boom", status_code=500),
            _FakeResponse("recovered text"),
        )
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == "recovered text"
        assert len(_FakeAsyncOpenAI.instances[-1].calls) == 2

    async def test_plain_429_is_retried(self, tmp_path: Path) -> None:
        _queue(
            _HttpError("slow down", status_code=429),
            _FakeResponse("after backoff"),
        )
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert result["output_text"] == "after backoff"
        assert len(_FakeAsyncOpenAI.instances[-1].calls) == 2

    async def test_retries_stop_at_the_attempt_budget(self, tmp_path: Path) -> None:
        _queue(_HttpError("still down", status_code=503))
        result = await _backend().transcribe_chunk(_payload(tmp_path))
        assert "error" in result
        assert len(_FakeAsyncOpenAI.instances[-1].calls) == 3

    async def test_non_transient_status_is_not_retried(self, tmp_path: Path) -> None:
        _queue(_HttpError("unauthorized", status_code=401))
        await _backend().transcribe_chunk(_payload(tmp_path))
        assert len(_FakeAsyncOpenAI.instances[-1].calls) == 1


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycle:
    async def test_close_disposes_the_active_client(self) -> None:
        backend = _backend()
        await backend.close()
        assert _FakeAsyncOpenAI.instances[-1].closed is True

    async def test_close_disposes_retired_clients_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()
        original = _FakeAsyncOpenAI.instances[-1]
        monkeypatch.setenv("OPENAI_API_KEY_2", "sk-second")
        monkeypatch.setattr(
            openai_audio, "resolve_api_key_env_var", lambda _p: "OPENAI_API_KEY_2"
        )
        backend.rekey()
        assert backend._key_env == "OPENAI_API_KEY_2"
        await backend.close()
        assert original.closed is True
        assert _FakeAsyncOpenAI.instances[-1].closed is True

    def test_rekey_is_a_noop_when_the_mapping_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()
        before = len(_FakeAsyncOpenAI.instances)
        monkeypatch.setattr(
            openai_audio, "resolve_api_key_env_var", lambda _p: "OPENAI_API_KEY"
        )
        backend.rekey()
        assert len(_FakeAsyncOpenAI.instances) == before

    def test_rekey_is_skipped_when_the_new_variable_is_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()
        monkeypatch.delenv("OPENAI_API_KEY_9", raising=False)
        monkeypatch.setattr(
            openai_audio, "resolve_api_key_env_var", lambda _p: "OPENAI_API_KEY_9"
        )
        backend.rekey()
        assert backend._key_env == "OPENAI_API_KEY"

    async def test_close_swallows_client_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()

        async def _boom() -> None:
            raise RuntimeError("already closed")

        monkeypatch.setattr(_FakeAsyncOpenAI.instances[-1], "close", _boom)
        await backend.close()
