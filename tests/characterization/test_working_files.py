"""``--keep-working-files``: Tesseract images and audio chunks stay only with it.

A Tesseract run over an image folder and a chunked local Whisper run each
go once with and once without the option; the files the run leaves behind
are compared, not golden snapshots.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_audio import FakeAudio
from tests.characterization.fakes_ocr import (
    FakeTesseract,
    block_real_tesseract,
    tesseract_cmd_settings,
)
from tests.characterization.inputs import make_image_folder
from tests.characterization.inputs_audio import make_wav
from tests.characterization.runner import RunResult, run_cli

pytestmark = pytest.mark.integration

PAGES = 3


def _working(result: RunResult, marker: str) -> list[Path]:
    """Return the files below the run's root whose name holds *marker*."""
    return sorted(path for path in result.root.rglob("*") if marker in path.name)


@pytest.mark.parametrize("keep", [False, True])
def test_tesseract_keeps_its_images_only_when_asked(
    keep: bool,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    case_root: Callable[[str], Path],
) -> None:
    FakeTesseract().install(monkeypatch)
    root = case_root()
    missing = block_real_tesseract(monkeypatch, root)
    folder = make_image_folder(root / "scans", PAGES)

    result = run_cli(
        "transcribe",
        llm=fake_llm,
        root=root,
        settings=tesseract_cmd_settings(missing),
        input=folder,
        output=root / "out",
        type="images",
        method="tesseract",
        keep_working_files=keep,
    )

    assert result.exit_code == 0
    assert len(_working(result, "_tess_preprocessed")) == (PAGES if keep else 0)


@pytest.mark.parametrize("keep", [False, True])
def test_whisper_keeps_its_chunks_only_when_asked(
    keep: bool,
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    case_root: Callable[[str], Path],
) -> None:
    root = case_root()
    missing = root / "no-ffmpeg"
    settings = {
        "tools": {
            "ffmpeg_cmd": str(missing / "ffmpeg.exe"),
            "ffprobe_cmd": str(missing / "ffprobe.exe"),
        },
        "audio_chunking": {"apply_to_local": True},
    }

    result = run_cli(
        "transcribe",
        llm=fake_llm,
        root=root,
        settings=settings,
        input=make_wav(root / "audio" / "talk.wav", 75),
        output=root / "out",
        type="audio",
        method="whisper",
        audio_chunk_seconds=30,
        keep_working_files=keep,
    )

    assert result.exit_code == 0
    assert [call["call"] for call in fake_audio.ffmpeg].count("cut") == 3
    assert len(_working(result, "_chunk_")) == (3 if keep else 0)
