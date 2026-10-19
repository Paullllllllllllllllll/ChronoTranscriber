"""Characterization of audio transcription: local Whisper and the audio API.

FFmpeg, faster-whisper and the provider SDKs are faked (``fakes_audio``); the
configured FFmpeg executables point at a path that does not exist, so a call
that escapes the fake fails instead of running an installed FFmpeg. Each
case runs twice in fresh roots and is compared with ``golden/audio/<case>/``,
which adds ``ffmpeg.json`` (probe and cut calls) to the usual snapshot and
the fake models and requests to ``requests.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import WriteGuard
from tests.characterization.fakes import FakeLLM, permanent_error, transient_error
from tests.characterization.fakes_audio import FakeAudio
from tests.characterization.golden import (
    check_files,
    compare,
    normalizer_for,
    snapshot,
)
from tests.characterization.inputs_audio import make_wav
from tests.characterization.runner import RunResult, deep_merge, run_cli

pytestmark = pytest.mark.integration

SHORT_SECONDS = 20
LONG_SECONDS = 75
SMALL_REQUEST_BYTES = 12_000
"""Below the size of the long recording, above that of the short one."""

CHUNK_STARTS = [0, 30, 60]
"""Start seconds of the chunks of the long recording (30-second minimum)."""


@dataclass
class AudioRun:
    """One ``run_cli`` call with the audio fakes' records of it."""

    result: RunResult
    ffmpeg: list[dict[str, Any]]


Build = Callable[[Path], AudioRun]


def _base_settings(root: Path) -> dict[str, Any]:
    missing = root / "no-ffmpeg"
    return {
        "tools": {
            "ffmpeg_cmd": str(missing / "ffmpeg.exe"),
            "ffprobe_cmd": str(missing / "ffprobe.exe"),
        }
    }


def transcribe_audio(
    llm: FakeLLM,
    audio: FakeAudio,
    root: Path,
    *,
    input: Path,
    settings: Mapping[str, Any] | None = None,
    **options: Any,
) -> AudioRun:
    """Run ``transcribe --type audio`` on *input* into ``root/out``."""
    audio.reset_records()
    result = run_cli(
        "transcribe",
        llm=llm,
        root=root,
        settings=deep_merge(_base_settings(root), settings or {}),
        input=input,
        output=root / "out",
        type="audio",
        **options,
    )
    result.models = [*result.models, *audio.models]
    result.calls = [*result.calls, *audio.calls]
    return AudioRun(result=result, ffmpeg=list(audio.ffmpeg))


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def dry_run_lines(result: RunResult) -> list[str]:
    """Return the ``[DRY RUN]`` console lines of a run."""
    return [line.strip() for line in result.stderr.splitlines() if "[DRY RUN]" in line]


def audio_snapshot(run: AudioRun, *, stdout: bool = False) -> dict[str, str]:
    """Return the golden files of *run*; *stdout* adds its ``[DRY RUN]`` lines.

    The lines are kept in ``stdout.txt``, the console stream they were first
    recorded from.
    """
    files = snapshot(run.result)
    norm = normalizer_for(run.result)
    files["ffmpeg.json"] = _dumps(norm.data(run.ffmpeg))
    if stdout:
        lines = [norm.text(line) for line in dry_run_lines(run.result)]
        files["stdout.txt"] = "".join(f"{line}\n" for line in lines)
    return files


def check_case(
    case: str,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
    build: Build,
    *,
    stdout: bool = False,
) -> AudioRun:
    """Run *build* in two fresh roots, require equal snapshots, check the golden."""
    first = audio_snapshot(build(case_root("run1")), stdout=stdout)
    run = build(case_root("run2"))
    actual = audio_snapshot(run, stdout=stdout)
    problems = compare(first, actual)
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)
    check_files(f"audio/{case}", actual, on_write=write_guard.consume)
    return run


def _short(root: Path) -> Path:
    return make_wav(root / "audio" / "talk.wav", SHORT_SECONDS)


def _long(root: Path) -> Path:
    return make_wav(root / "audio" / "talk.wav", LONG_SECONDS)


def _text(run: AudioRun) -> str:
    output = run.result.output(run.result.root / "audio" / "talk.wav")
    assert output in run.result.created
    return output.read_text(encoding="utf-8")


def _starts(run: AudioRun) -> list[int]:
    return [call["chunk"]["start_second"] for call in run.result.calls]


_OVERSIZED = {"audio_chunking": {"max_request_bytes": SMALL_REQUEST_BYTES}}


# ============================================================================
# Local Whisper
# ============================================================================
def test_whisper_single_request(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    settings = {"whisper": {"beam_size": 3}}
    run = check_case(
        "whisper_single_request",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_short(root),
            method="whisper",
            language="de",
            settings=settings,
        ),
    )

    assert run.result.exit_code == 0
    assert run.ffmpeg == []
    assert [model["constructor"] for model in run.result.models] == ["WhisperModel"]
    kwargs = run.result.models[0]["kwargs"]
    assert kwargs["model_size_or_path"] == "large-v3"
    assert run.result.calls[0]["kwargs"]["beam_size"] == 3
    assert run.result.calls[0]["file"] == "talk.wav"
    assert "Recording from second 0 to 20" in _text(run)


def test_whisper_chunked_local(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    settings = {"audio_chunking": {"apply_to_local": True}}
    run = check_case(
        "whisper_chunked_local",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_long(root),
            method="whisper",
            audio_chunk_seconds=30,
            settings=settings,
        ),
    )

    assert run.result.exit_code == 0
    assert [call["call"] for call in run.ffmpeg] == ["probe", "cut", "cut", "cut"]
    assert _starts(run) == CHUNK_STARTS
    assert len(run.result.models) == 1
    text = _text(run)
    assert all(f"Recording from second {start} " in text for start in CHUNK_STARTS)


# ============================================================================
# Audio API
# ============================================================================
def test_openai_single_request(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    settings = {
        "audio_api": {
            "prompt": "Minutes of the grain market.",
            "languages": ["de", "fr"],
            "keywords": ["Roggen"],
        }
    }
    run = check_case(
        "openai_single_request",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_short(root),
            method="audio-api",
            settings=settings,
        ),
    )

    assert run.result.exit_code == 0
    assert [call["call"] for call in run.ffmpeg] == ["probe"]
    call = run.result.calls[0]
    assert call["file"]["name"] == "talk_chunk_0001.wav"
    assert call["kwargs"]["model"] == "gpt-transcribe"
    assert call["kwargs"]["extra_body"] == {
        "languages": ["de", "fr"],
        "keywords": ["Roggen"],
    }
    assert "Recording from second 0 to 20" in _text(run)


def test_openai_oversized_chunked(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    run = check_case(
        "openai_oversized_chunked",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_long(root),
            method="audio-api",
            settings=_OVERSIZED,
        ),
    )

    assert run.result.exit_code == 0
    assert _starts(run) == CHUNK_STARTS
    names = [call["file"]["name"] for call in run.result.calls]
    assert names == [f"talk_chunk_{n:04d}.mp3" for n in (1, 2, 3)]


def test_google_single_request(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    run = check_case(
        "google_single_request",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_short(root),
            method="audio-api",
            provider="google",
            language="German",
        ),
    )

    assert run.result.exit_code == 0
    assert [model["constructor"] for model in run.result.models] == [
        "ChatGoogleGenerativeAI"
    ]
    assert run.result.models[0]["kwargs"]["model"] == "gemini-3.6-flash"
    system = run.result.calls[0]["messages"][0]["blocks"][0]["text"]
    assert system.endswith("The recording is in German.")
    assert "Recording from second 0 to 20" in _text(run)


def test_provider_and_model_options_route_to_the_audio_api(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    run = check_case(
        "provider_and_model_overrides",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_short(root),
            method="audio-api",
            provider="google",
            model="gemini-3.5-flash",
        ),
    )

    assert run.result.exit_code == 0
    assert [model["provider"] for model in run.result.models] == ["google"]
    assert run.result.models[0]["kwargs"]["model"] == "gemini-3.5-flash"


def test_model_override_keeps_configured_provider(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    run = check_case(
        "model_override_openai",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_short(root),
            method="audio-api",
            model="whisper-1",
            language="de",
        ),
    )

    assert run.result.exit_code == 0
    kwargs = run.result.calls[0]["kwargs"]
    assert (kwargs["model"], kwargs["language"]) == ("whisper-1", "de")
    assert "extra_body" not in kwargs


# ============================================================================
# Errors
# ============================================================================
def test_transient_chunk_error_is_retried(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> AudioRun:
        fake_audio.script(30, replies={1: transient_error()})
        return transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_long(root),
            method="audio-api",
            settings=_OVERSIZED,
        )

    run = check_case("transient_chunk_error", case_root, write_guard, build)

    assert run.result.exit_code == 0
    attempts = [(c["chunk"]["start_second"], c["attempt"]) for c in run.result.calls]
    assert attempts == [(0, 1), (30, 1), (30, 2), (60, 1)]
    assert "[transcription error" not in _text(run)


def test_permanent_chunk_error_fails_the_item(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> AudioRun:
        fake_audio.script(30, always=permanent_error())
        return transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_long(root),
            method="audio-api",
            settings=_OVERSIZED,
        )

    run = check_case("permanent_chunk_error", case_root, write_guard, build)

    assert run.result.exit_code == 1
    assert run.result.summary is not None
    assert run.result.summary["items_failed"] == 1
    attempts = [(c["chunk"]["start_second"], c["attempt"]) for c in run.result.calls]
    assert attempts == [(0, 1), (30, 1), (60, 1)]


# ============================================================================
# Dry run and page ranges
# ============================================================================
def test_dry_run_reports_chunk_counts(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> AudioRun:
        folder = root / "audio"
        make_wav(folder / "brief.wav", SHORT_SECONDS)
        make_wav(folder / "session.wav", LONG_SECONDS)
        return transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=folder,
            method="audio-api",
            dry_run=True,
            settings=_OVERSIZED,
        )

    run = check_case("dry_run_chunk_counts", case_root, write_guard, build, stdout=True)

    assert run.result.exit_code == 0
    assert run.result.summary is not None
    assert run.result.summary["dry_run"] is True
    assert run.result.summary["items_total"] == 2
    assert run.result.calls == []
    lines = dry_run_lines(run.result)
    assert any("brief.wav (1 chunk)" in line for line in lines)
    assert any("session.wav (3 chunk(s))" in line for line in lines)


def test_pages_select_audio_chunks(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    run = check_case(
        "pages_select_chunks",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=_long(root),
            method="audio-api",
            pages="2-3",
            settings=_OVERSIZED,
        ),
    )

    assert run.result.exit_code == 0
    assert _starts(run) == [30, 60]
    assert [call["target"] for call in run.ffmpeg if call["call"] == "cut"] == [
        "talk_chunk_0002.mp3",
        "talk_chunk_0003.mp3",
    ]
