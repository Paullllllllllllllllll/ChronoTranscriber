"""Tests for modules.audio.backends.google_audio.GoogleAudioBackend.

The underlying ``GoogleProvider`` is replaced with a stub, so nothing here
constructs an SDK client or reaches the network.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from modules.audio.audio_stream import AudioChunkPayload
from modules.audio.backends.google_audio import (
    DEFAULT_GOOGLE_AUDIO_MODEL,
    DEFAULT_GOOGLE_PROMPT,
    GoogleAudioBackend,
)
from modules.audio.constants import GEMINI_INLINE_LIMIT_BYTES, NO_TRANSCRIBABLE_TEXT
from modules.config.capabilities import CapabilityError
from modules.llm.providers.base import TranscriptionResult

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubGoogleProvider:
    """Records every transcription request and returns a canned result."""

    instances: list[_StubGoogleProvider] = []
    next_result: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.result: Any = _StubGoogleProvider.next_result or TranscriptionResult(
            content="stub transcript"
        )
        _StubGoogleProvider.instances.append(self)

    async def transcribe_audio_from_base64(
        self,
        audio_base64: str,
        mime_type: str,
        *,
        system_prompt: str,
        max_output_tokens: int | None = None,
    ) -> Any:
        self.calls.append(
            {
                "audio_base64": audio_base64,
                "mime_type": mime_type,
                "system_prompt": system_prompt,
                "max_output_tokens": max_output_tokens,
            }
        )
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _stub_google_provider(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Route ``get_provider`` to the stub for the whole module."""
    _StubGoogleProvider.instances = []
    _StubGoogleProvider.next_result = None
    from modules.llm.providers import factory

    monkeypatch.setattr(
        factory, "get_provider", lambda **kwargs: _StubGoogleProvider(**kwargs)
    )
    yield
    _StubGoogleProvider.next_result = None


def _audio_config(**google_settings: Any) -> dict[str, Any]:
    return {"audio_transcription": {"google": google_settings}}


def _backend(**google_settings: Any) -> GoogleAudioBackend:
    return GoogleAudioBackend(_audio_config(**google_settings), {})


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


def _provider() -> _StubGoogleProvider:
    return _StubGoogleProvider.instances[-1]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstruction:
    def test_defaults(self) -> None:
        backend = _backend()
        assert backend.provider_name == "google"
        assert backend.model == DEFAULT_GOOGLE_AUDIO_MODEL

    def test_configured_model_is_used(self) -> None:
        assert _backend(model="gemini-3-pro").model == "gemini-3-pro"

    def test_provider_is_built_for_google_with_the_configured_knobs(self) -> None:
        _backend(model="gemini-3-flash", temperature=0.5, max_output_tokens=1024)
        kwargs = _provider().init_kwargs
        assert kwargs["provider"] == "google"
        assert kwargs["model"] == "gemini-3-flash"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 1024

    @pytest.mark.parametrize("raw", [None, "", 0])
    def test_falsy_max_output_tokens_becomes_none(self, raw: Any) -> None:
        backend = _backend(max_output_tokens=raw)
        assert backend._max_output_tokens is None

    def test_missing_temperature_is_not_forced(self) -> None:
        _backend()
        assert _provider().init_kwargs["temperature"] is None

    def test_non_audio_model_is_refused(self) -> None:
        with pytest.raises(CapabilityError):
            _backend(model="gemma-4-31b-it")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSystemPrompt:
    async def test_default_prompt_is_sent_when_unconfigured(
        self, tmp_path: Path
    ) -> None:
        backend = _backend()
        await backend.transcribe_chunk(_payload(tmp_path))
        assert _provider().calls[0]["system_prompt"] == DEFAULT_GOOGLE_PROMPT

    async def test_configured_prompt_replaces_the_default(self, tmp_path: Path) -> None:
        backend = _backend(prompt="  Transcribe the recipe dictation.  ")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert (
            _provider().calls[0]["system_prompt"] == "Transcribe the recipe dictation."
        )

    async def test_language_hint_is_appended(self, tmp_path: Path) -> None:
        backend = _backend(prompt="Base prompt.", language_hint="German")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert (
            _provider().calls[0]["system_prompt"]
            == "Base prompt.\nThe recording is in German."
        )

    async def test_language_hint_is_appended_to_the_default_prompt(
        self, tmp_path: Path
    ) -> None:
        backend = _backend(language_hint="French")
        await backend.transcribe_chunk(_payload(tmp_path))
        prompt = _provider().calls[0]["system_prompt"]
        assert prompt.startswith(DEFAULT_GOOGLE_PROMPT)
        assert prompt.endswith("The recording is in French.")

    async def test_blank_language_hint_adds_nothing(self, tmp_path: Path) -> None:
        backend = _backend(prompt="Base prompt.", language_hint="   ")
        await backend.transcribe_chunk(_payload(tmp_path))
        assert _provider().calls[0]["system_prompt"] == "Base prompt."


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRequestPayload:
    async def test_audio_is_sent_base64_encoded_with_its_mime_type(
        self, tmp_path: Path
    ) -> None:
        backend = _backend()
        await backend.transcribe_chunk(_payload(tmp_path, data=b"abcdef"))
        call = _provider().calls[0]
        assert call["audio_base64"] == base64.b64encode(b"abcdef").decode("utf-8")
        assert call["mime_type"] == "audio/mp3"

    async def test_max_output_tokens_is_forwarded(self, tmp_path: Path) -> None:
        backend = _backend(max_output_tokens=2048)
        await backend.transcribe_chunk(_payload(tmp_path))
        assert _provider().calls[0]["max_output_tokens"] == 2048


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGuards:
    async def test_disallowed_extension_short_circuits_before_the_provider(
        self, tmp_path: Path
    ) -> None:
        backend = _backend()
        result = await backend.transcribe_chunk(
            _payload(tmp_path, name="rec_chunk_0001.webm")
        )
        assert "error" in result
        assert ".webm" in result["error"]
        assert result["output_text"] == ""
        assert _provider().calls == []

    async def test_oversize_payload_short_circuits_before_the_provider(
        self, tmp_path: Path
    ) -> None:
        backend = _backend()
        payload = _payload(tmp_path, byte_size=GEMINI_INLINE_LIMIT_BYTES + 1)
        result = await backend.transcribe_chunk(payload)
        assert "error" in result
        assert "inline" in result["error"]
        assert _provider().calls == []

    async def test_payload_exactly_at_the_inline_limit_is_sent(
        self, tmp_path: Path
    ) -> None:
        backend = _backend()
        payload = _payload(tmp_path, byte_size=GEMINI_INLINE_LIMIT_BYTES)
        result = await backend.transcribe_chunk(payload)
        assert "error" not in result
        assert len(_provider().calls) == 1

    async def test_error_metadata_carries_the_chunk_index(self, tmp_path: Path) -> None:
        backend = _backend()
        result = await backend.transcribe_chunk(
            _payload(tmp_path, name="rec_chunk_0004.webm", index=3)
        )
        assert result["metadata"]["chunk_index"] == 3

    async def test_provider_exception_becomes_an_error_dict(
        self, tmp_path: Path
    ) -> None:
        _StubGoogleProvider.next_result = RuntimeError("gemini exploded")
        backend = _backend()
        result = await backend.transcribe_chunk(_payload(tmp_path))
        assert "gemini exploded" in result["error"]
        assert result["output_text"] == ""


# ---------------------------------------------------------------------------
# TranscriptionResult -> legacy dict
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLegacyDictMapping:
    async def _run(self, tmp_path: Path, result: TranscriptionResult) -> dict[str, Any]:
        _StubGoogleProvider.next_result = result
        backend = _backend()
        return await backend.transcribe_chunk(_payload(tmp_path))

    async def test_content_and_usage_are_mapped(self, tmp_path: Path) -> None:
        response = await self._run(
            tmp_path,
            TranscriptionResult(
                content="Spoken words.",
                input_tokens=200,
                output_tokens=40,
                total_tokens=240,
            ),
        )
        assert response["output_text"] == "Spoken words."
        assert response["usage"] == {
            "input_tokens": 200,
            "output_tokens": 40,
            "total_tokens": 240,
        }

    async def test_metadata_merges_the_raw_response(self, tmp_path: Path) -> None:
        response = await self._run(
            tmp_path,
            TranscriptionResult(content="text", raw_response={"finish_reason": "STOP"}),
        )
        assert response["metadata"]["finish_reason"] == "STOP"
        assert response["metadata"]["provider"] == "google"
        assert response["metadata"]["model"] == DEFAULT_GOOGLE_AUDIO_MODEL
        assert response["metadata"]["chunk_index"] == 0

    async def test_cache_tokens_are_added_only_when_positive(
        self, tmp_path: Path
    ) -> None:
        response = await self._run(
            tmp_path,
            TranscriptionResult(
                content="text", cached_input_tokens=64, cache_creation_tokens=0
            ),
        )
        assert response["usage"]["cached_input_tokens"] == 64
        assert "cache_creation_tokens" not in response["usage"]

    async def test_no_cache_keys_when_both_are_zero(self, tmp_path: Path) -> None:
        response = await self._run(tmp_path, TranscriptionResult(content="text"))
        assert "cached_input_tokens" not in response["usage"]
        assert "cache_creation_tokens" not in response["usage"]

    @pytest.mark.parametrize("content", ["", "   \n "])
    async def test_empty_content_collapses_to_the_sentinel(
        self, tmp_path: Path, content: str
    ) -> None:
        response = await self._run(tmp_path, TranscriptionResult(content=content))
        assert response["output_text"] == NO_TRANSCRIBABLE_TEXT
        assert "error" not in response

    async def test_provider_error_empties_the_output_text(self, tmp_path: Path) -> None:
        response = await self._run(
            tmp_path, TranscriptionResult(content="", error="429 rate limited")
        )
        assert response["error"] == "429 rate limited"
        assert response["output_text"] == ""

    async def test_parsed_output_is_surfaced(self, tmp_path: Path) -> None:
        payload = json.dumps(
            {
                "transcription": "",
                "no_transcribable_text": False,
                "transcription_not_possible": True,
            }
        )
        response = await self._run(tmp_path, TranscriptionResult(content=payload))
        assert response["parsed"]["transcription_not_possible"] is True

    async def test_transcription_not_possible_reaches_the_shared_placeholder(
        self, tmp_path: Path
    ) -> None:
        from modules.llm.response_parsing import (
            detect_transcription_cause,
            extract_transcribed_text,
        )

        payload = json.dumps(
            {
                "transcription": "",
                "no_transcribable_text": False,
                "transcription_not_possible": True,
            }
        )
        response = await self._run(tmp_path, TranscriptionResult(content=payload))
        text = extract_transcribed_text(response, "rec_chunk_0001.mp3")
        assert detect_transcription_cause(text) == "not_possible"

    async def test_a_plain_transcript_extracts_unchanged(self, tmp_path: Path) -> None:
        from modules.llm.response_parsing import extract_transcribed_text

        response = await self._run(
            tmp_path, TranscriptionResult(content="Plain spoken words.")
        )
        assert (
            extract_transcribed_text(response, "rec_chunk_0001.mp3")
            == "Plain spoken words."
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycle:
    async def test_close_disposes_the_provider(self) -> None:
        backend = _backend()
        await backend.close()
        assert _provider().closed is True

    async def test_close_swallows_provider_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _backend()

        async def _boom() -> None:
            raise RuntimeError("already disposed")

        monkeypatch.setattr(_provider(), "close", _boom)
        await backend.close()

    def test_rekey_is_a_noop(self) -> None:
        assert _backend().rekey() is None
