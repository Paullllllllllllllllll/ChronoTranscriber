"""Fakes for the external runtimes and SDKs of the audio workflow.

``FakeAudio`` stands in for every external piece the audio pipeline touches:
the FFmpeg runtime functions (availability, duration probe, chunk cutting),
the ``faster_whisper`` package with its ``WhisperModel`` class, the
``AsyncOpenAI`` client of the OpenAI audio backend, and the
``ChatGoogleGenerativeAI`` class behind the Google audio backend. The real
chunk planning, streaming, retry, response parsing and output writing all run.

The fake FFmpeg cuts a chunk by slicing WAV frames, so every chunk file holds
WAV data whatever its suffix. A request is matched to the second its audio
starts at (see ``inputs_audio.decode_wav``), never by call order, so scripts
and records do not depend on concurrency.
"""

from __future__ import annotations

import base64
import binascii
import copy
import importlib.machinery
import sys
import threading
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from chronotranscriber.common.testing import FakeRequest, ScriptedChatModel
from tests.characterization.fakes import USAGE, redact
from tests.characterization.inputs_audio import (
    WavInfo,
    decode_wav,
    read_frames,
    wav_bytes,
)
from tests.characterization.seams import replace, replace_tool_function

OPENAI_USAGE: dict[str, Any] = {
    "type": "tokens",
    "input_tokens": 120,
    "output_tokens": 40,
    "total_tokens": 160,
}

FFMPEG_FUNCTIONS: tuple[str, ...] = (
    "is_ffmpeg_available",
    "is_ffprobe_available",
    "probe_duration_seconds",
    "cut_segment",
)
"""Names of the tool's FFmpeg runtime functions the fake replaces."""

OPENAI_CLIENT = ("openai", "AsyncOpenAI")
GOOGLE_CLIENT = ("langchain_google_genai", "ChatGoogleGenerativeAI")


def default_transcript(start: int, seconds: float) -> str:
    """Return the default transcript of the chunk starting at second *start*.

    The text carries a run of six internal spaces, which the outputs keep
    because transcripts are written as returned.
    """
    end = start + seconds
    return (
        f"Recording from second {start} to {end:g}:      the council read the "
        f"grain accounts aloud. The clerk noted the price of rye for {1700 + start}."
    )


@dataclass
class ChunkScript:
    """Scripted answers for the chunk that starts at one second.

    ``replies`` maps an attempt number (counted per chunk and run from 1) to
    an exception to raise or a text to return; ``always`` answers every
    attempt without an entry there. ``text`` replaces the default transcript.
    """

    text: str | None = None
    replies: dict[int, BaseException | str] = field(default_factory=dict)
    always: BaseException | str | None = None


def _chunk_record(info: WavInfo) -> dict[str, Any]:
    return {"start_second": info.start_second, "seconds": info.seconds}


class FakeAudio:
    """Audio fakes with their scripts and records.

    ``ffmpeg`` lists the probe and cut calls in call order, ``models`` one
    entry per constructed client or model, ``calls`` one digest per request.
    ``scripts`` maps a chunk's start second to its ``ChunkScript``.
    """

    def __init__(self) -> None:
        self.scripts: dict[int, ChunkScript] = {}
        self.ffmpeg: list[dict[str, Any]] = []
        self.constructions: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self._attempts: dict[int, int] = {}
        self._lock = threading.Lock()

    # -- scripting -------------------------------------------------------------
    def script(self, start_second: int, **changes: Any) -> ChunkScript:
        """Return the script for the chunk at *start_second*, applying *changes*."""
        with self._lock:
            script = self.scripts.setdefault(start_second, ChunkScript())
        for name, value in changes.items():
            if not hasattr(script, name):
                raise AttributeError(f"ChunkScript has no field {name!r}")
            setattr(script, name, value)
        return script

    def reset_records(self) -> None:
        """Forget records and attempt counters."""
        with self._lock:
            self.ffmpeg.clear()
            self.constructions.clear()
            self.requests.clear()
            self._attempts.clear()

    @property
    def models(self) -> list[dict[str, Any]]:
        """Return copies of the construction records."""
        with self._lock:
            return copy.deepcopy(self.constructions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return the request digests ordered by chunk and attempt."""

        def key(call: dict[str, Any]) -> tuple[int, int]:
            chunk = call.get("chunk") or {}
            return int(chunk.get("start_second", -1)), int(call["attempt"])

        with self._lock:
            return sorted(copy.deepcopy(self.requests), key=key)

    # -- shared answering ------------------------------------------------------
    def _construct(self, provider: str, constructor: str, kwargs: Any) -> None:
        with self._lock:
            self.constructions.append(
                {
                    "provider": provider,
                    "constructor": constructor,
                    "kwargs": redact(kwargs),
                }
            )

    def _answer(self, info: WavInfo, digest: dict[str, Any]) -> str:
        """Record *digest* for the chunk in *info*; return or raise its answer."""
        start = info.start_second
        with self._lock:
            attempt = self._attempts.get(start, 0) + 1
            self._attempts[start] = attempt
            script = self.scripts.get(start, ChunkScript())
            self.requests.append(
                {**digest, "chunk": _chunk_record(info), "attempt": attempt}
            )
        reply = script.replies.get(attempt, script.always)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, str):
            return reply
        if script.text is not None:
            return script.text
        return default_transcript(start, info.seconds)

    # -- FFmpeg ----------------------------------------------------------------
    @staticmethod
    def ffmpeg_available() -> bool:
        """Stand-in for the FFmpeg and ffprobe availability checks."""
        return True

    def probe_duration_seconds(self, path: Path) -> float | None:
        """Stand-in for ``probe_duration_seconds``: read the WAV header."""
        seconds = decode_wav(Path(path).read_bytes()).seconds
        with self._lock:
            self.ffmpeg.append(
                {"call": "probe", "file": Path(path).name, "seconds": seconds}
            )
        return seconds

    def cut_segment(
        self,
        src: Path,
        dst: Path,
        *,
        start: float,
        duration: float | None,
        chunk_format: str,
        mono: bool,
        sample_rate: int,
    ) -> None:
        """Stand-in for ``cut_segment``: write the frames of the span to *dst*."""
        frames, frame_rate = read_frames(Path(src).read_bytes())
        width = 2
        first = round(start * frame_rate) * width
        last = (
            len(frames)
            if duration is None
            else first + round(duration * frame_rate) * width
        )
        data = wav_bytes(frames[first:last], frame_rate)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        with self._lock:
            self.ffmpeg.append(
                {
                    "call": "cut",
                    "source": Path(src).name,
                    "target": Path(dst).name,
                    "start": start,
                    "duration": duration,
                    "chunk_format": chunk_format,
                    "mono": mono,
                    "sample_rate": sample_rate,
                    "bytes": len(data),
                }
            )

    # -- faster-whisper --------------------------------------------------------
    def whisper_module(self) -> types.ModuleType:
        """Return a stand-in ``faster_whisper`` module."""
        controller = self

        class WhisperModel:
            """Stand-in for ``faster_whisper.WhisperModel``; never downloads."""

            def __init__(self, model_size_or_path: str, **kwargs: Any) -> None:
                controller._construct(
                    "faster-whisper",
                    "WhisperModel",
                    {"model_size_or_path": model_size_or_path, **kwargs},
                )

            def transcribe(
                self, audio: Any, **kwargs: Any
            ) -> tuple[Iterator[Any], Any]:
                return controller.whisper_transcribe(audio, kwargs)

        module = types.ModuleType("faster_whisper")
        module.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", None)
        module.WhisperModel = WhisperModel  # type: ignore[attr-defined]
        return module

    def whisper_transcribe(
        self, audio: Any, kwargs: Mapping[str, Any]
    ) -> tuple[Iterator[Any], Any]:
        """Answer one ``WhisperModel.transcribe`` call with two segments."""
        path = Path(str(audio))
        data = path.read_bytes()
        info = decode_wav(data)
        text = self._answer(
            info,
            {
                "provider": "faster-whisper",
                "file": path.name,
                "bytes": len(data),
                "kwargs": redact(dict(kwargs)),
            },
        )
        head, _, tail = text.partition(". ")
        pieces = [f" {head}. " if tail else f" {head} ", tail]
        segments = [types.SimpleNamespace(text=piece) for piece in pieces if piece]
        details = types.SimpleNamespace(language="en", duration=info.seconds)
        return iter(segments), details

    # -- OpenAI ----------------------------------------------------------------
    def openai_client(self) -> Callable[..., Any]:
        """Return a stand-in for the ``AsyncOpenAI`` class."""
        controller = self

        class AsyncOpenAI:
            """Stand-in for ``openai.AsyncOpenAI`` with the audio endpoint only."""

            def __init__(self, **kwargs: Any) -> None:
                controller._construct("openai", "AsyncOpenAI", kwargs)
                self.audio = types.SimpleNamespace(
                    transcriptions=types.SimpleNamespace(create=self._create)
                )

            async def _create(self, **kwargs: Any) -> Any:
                return controller.openai_transcribe(kwargs)

            async def close(self) -> None:
                return None

        return AsyncOpenAI

    def openai_transcribe(self, kwargs: Mapping[str, Any]) -> Any:
        """Answer one ``audio.transcriptions.create`` call."""
        name, data, mime = kwargs["file"]
        info = decode_wav(data)
        digest = {
            "provider": "openai",
            "file": {"name": name, "bytes": len(data), "mime": mime},
            "kwargs": redact({k: v for k, v in kwargs.items() if k != "file"}),
        }
        text = self._answer(info, digest)
        usage = types.SimpleNamespace(**OPENAI_USAGE)
        return types.SimpleNamespace(text=text, usage=usage)

    # -- Google ----------------------------------------------------------------
    def google_class(self) -> Callable[..., ScriptedChatModel]:
        """Return a stand-in for ``ChatGoogleGenerativeAI``."""

        def construct(*args: Any, **kwargs: Any) -> ScriptedChatModel:
            if args:
                raise TypeError("ChatGoogleGenerativeAI takes keyword arguments only")
            self._construct("google", "ChatGoogleGenerativeAI", kwargs)
            return ScriptedChatModel(
                responder=self.google_respond, recorder=self.google_record
            )

        return construct

    def google_record(self, request: FakeRequest) -> None:
        """Decode the inline audio part of a Gemini request."""
        request.notes["shapes"], request.notes["info"] = _google_shapes(
            request.messages
        )

    def google_respond(self, request: FakeRequest) -> AIMessage:
        """Answer one Gemini request with the scripted transcript."""
        info: WavInfo | None = request.notes.get("info")
        if info is None:
            raise AssertionError("Gemini audio request without an audio part")
        digest = {
            "provider": "google",
            "kwargs": redact(dict(request.kwargs)),
            "bound": redact(dict(request.bound)),
            "messages": request.notes["shapes"],
        }
        text = self._answer(info, digest)
        return AIMessage(
            content=text,
            usage_metadata=dict(USAGE),
            response_metadata={"model_name": "fake", "finish_reason": "STOP"},
        )

    # -- installation ----------------------------------------------------------
    def install(self, monkeypatch: pytest.MonkeyPatch, *, ffmpeg: bool = True) -> None:
        """Replace the external runtimes and clients with these fakes.

        With ``ffmpeg=False`` the real FFmpeg runtime stays in place, which
        then fails on the nonexistent executables the tests configure.
        """
        if ffmpeg:
            replacements: dict[str, Callable[..., Any]] = {
                "is_ffmpeg_available": self.ffmpeg_available,
                "is_ffprobe_available": self.ffmpeg_available,
                "probe_duration_seconds": self.probe_duration_seconds,
                "cut_segment": self.cut_segment,
            }
            for name in FFMPEG_FUNCTIONS:
                replace_tool_function(monkeypatch, name, replacements[name])
        monkeypatch.setitem(sys.modules, "faster_whisper", self.whisper_module())
        replace(monkeypatch, *OPENAI_CLIENT, self.openai_client())
        replace(monkeypatch, *GOOGLE_CLIENT, self.google_class())


def _google_shapes(
    messages: list[BaseMessage],
) -> tuple[list[dict[str, Any]], WavInfo | None]:
    """Return message digests and the decoded audio of a Gemini request."""
    shapes: list[dict[str, Any]] = []
    info: WavInfo | None = None
    for message in messages:
        content = message.content
        parts = [content] if isinstance(content, str) else list(content)
        blocks: list[dict[str, Any]] = []
        for block in parts:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
            elif block.get("type") == "text":
                blocks.append({"type": "text", "text": str(block.get("text", ""))})
            elif block.get("type") == "file":
                # langchain-core hands the model the standard block, which
                # carries the payload under "base64" rather than "data".
                encoded = block.get("base64", block.get("data", ""))
                try:
                    data = base64.b64decode(str(encoded))
                except (binascii.Error, ValueError):
                    data = b""
                info = decode_wav(data)
                blocks.append(
                    {
                        "type": "file",
                        "keys": sorted(block),
                        "mime": block.get("mime_type"),
                        "bytes": len(data),
                    }
                )
            else:
                blocks.append({"type": str(block.get("type"))})
        shapes.append({"role": message.type, "blocks": blocks})
    return shapes, info


__all__ = [
    "FFMPEG_FUNCTIONS",
    "OPENAI_USAGE",
    "ChunkScript",
    "FakeAudio",
    "default_transcript",
]
