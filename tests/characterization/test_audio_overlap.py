"""Characterization of chunk overlap for an oversized recording.

The 75-second recording exceeds the request cap, so the audio API path cuts
it into 30-second chunks; with ``audio_chunking.overlap_seconds`` above 0 each
chunk after the first starts that many seconds before the previous one ends.
``ffmpeg.json`` in ``golden/audio/openai_oversized_overlap/`` records the cut
boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from chronotranscriber.common.testing import WriteGuard
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_audio import FakeAudio
from tests.characterization.inputs_audio import make_wav
from tests.characterization.test_audio import (
    LONG_SECONDS,
    SMALL_REQUEST_BYTES,
    check_case,
    transcribe_audio,
)

pytestmark = pytest.mark.integration

OVERLAP_SECONDS = 5
OVERLAP_STARTS = [0, 25, 50]
"""Start seconds of the chunks: 30-second chunks that advance by 25 seconds."""


def test_oversized_recording_with_overlap(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    settings = {
        "audio_chunking": {
            "max_request_bytes": SMALL_REQUEST_BYTES,
            "overlap_seconds": OVERLAP_SECONDS,
        }
    }
    run = check_case(
        "openai_oversized_overlap",
        case_root,
        write_guard,
        lambda root: transcribe_audio(
            fake_llm,
            fake_audio,
            root,
            input=make_wav(root / "audio" / "talk.wav", LONG_SECONDS),
            method="audio-api",
            settings=settings,
        ),
    )

    assert run.result.exit_code == 0
    cuts = [(call["start"], call["duration"]) for call in run.ffmpeg if "start" in call]
    assert cuts == [(0.0, 30.0), (25.0, 30.0), (50.0, None)]
    starts = [call["chunk"]["start_second"] for call in run.result.calls]
    assert starts == OVERLAP_STARTS
