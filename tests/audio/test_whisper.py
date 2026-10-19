"""Tests for the Whisper method's local faster-whisper runtime.

``faster-whisper`` is an optional extra; every test here fakes it, so the
suite never requires the package to be installed.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chronotranscriber.methods import whisper as whisper_module
from chronotranscriber.methods.whisper import (
    WhisperMissingError,
    WhisperTranscriber,
    is_faster_whisper_available,
)
from chronotranscriber.settings import Whisper
from chronotranscriber.sources.audio.constants import NO_TRANSCRIBABLE_TEXT
from tests.conftest import make_job, make_settings, make_spec


class _FakeWhisperModel:
    """Stand-in for ``faster_whisper.WhisperModel``."""

    instances: list[_FakeWhisperModel] = []

    def __init__(self, model_ref: str, **kwargs: Any) -> None:
        self.model_ref = model_ref
        self.kwargs = kwargs
        self.transcribe_calls: list[dict[str, Any]] = []
        self.segments: list[Any] = []
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
        self.transcribe_calls.append({"path": path, **kwargs})
        return iter(self.segments), SimpleNamespace(language="en")


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> type[_FakeWhisperModel]:
    """Inject a stub ``faster_whisper`` module into ``sys.modules``."""
    _FakeWhisperModel.instances = []
    module = ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return _FakeWhisperModel


def _transcriber(
    tmp_path: Path,
    *,
    whisper: Whisper | None = None,
    events: Any = None,
    **flags: Any,
) -> WhisperTranscriber:
    spec = make_spec(tmp_path / "rec.wav", type="audio", method="whisper", **flags)
    job = make_job(
        spec,
        output_dir=tmp_path / "out",
        input_root=tmp_path,
        settings=make_settings(whisper=whisper or Whisper()),
        events=events,
    )
    return WhisperTranscriber(job)


def _segment(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _chunk(tmp_path: Path, index: int = 0) -> SimpleNamespace:
    name = f"rec_chunk_{index + 1:04d}.wav"
    path = tmp_path / name
    path.write_bytes(b"RIFF")
    return SimpleNamespace(index=index, image_name=name, path=path)


@pytest.mark.unit
class TestAvailability:
    @pytest.mark.parametrize(("spec", "expected"), [(object(), True), (None, False)])
    def test_availability_follows_the_spec_lookup(
        self, monkeypatch: pytest.MonkeyPatch, spec: object, expected: bool
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: spec)
        assert is_faster_whisper_available() is expected

    @pytest.mark.parametrize("error", [ImportError("boom"), ValueError("bad name")])
    def test_unavailable_when_the_lookup_raises(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def _find_spec(_name: str) -> Any:
            raise error

        monkeypatch.setattr("importlib.util.find_spec", _find_spec)
        assert is_faster_whisper_available() is False


@pytest.mark.unit
class TestOptions:
    def test_the_whisper_model_flag_and_language_apply(self, tmp_path: Path) -> None:
        options = _transcriber(tmp_path, whisper_model="tiny", language="de").options
        assert (options.model, options.language) == ("tiny", "de")

    def test_model_path_overrides_the_model_size(self, tmp_path: Path) -> None:
        folder = tmp_path / "models" / "custom"
        whisper = Whisper(model_path=folder)
        options = _transcriber(tmp_path, whisper=whisper, whisper_model="small").options
        assert options.model == str(folder)

    def test_explicit_device_and_compute_type_are_kept(self, tmp_path: Path) -> None:
        whisper = Whisper(device="cuda", compute_type="float16")
        options = _transcriber(tmp_path, whisper=whisper).options
        assert (options.device, options.compute_type) == ("cuda", "float16")


@pytest.mark.unit
class TestModelLoading:
    def test_the_model_is_loaded_once_per_run(
        self, tmp_path: Path, fake_whisper: type[_FakeWhisperModel]
    ) -> None:
        transcriber = _transcriber(tmp_path, whisper_model="tiny")
        first = transcriber.load_model()
        second = transcriber.load_model()
        assert first is second
        assert len(fake_whisper.instances) == 1
        assert first.model_ref == "tiny"
        assert first.kwargs == {"device": "auto", "compute_type": "default"}

    def test_download_root_is_forwarded_to_the_library(
        self, tmp_path: Path, fake_whisper: type[_FakeWhisperModel]
    ) -> None:
        whisper = Whisper(download_root=tmp_path / "cache")
        model = _transcriber(tmp_path, whisper=whisper).load_model()
        assert model.kwargs["download_root"] == str(tmp_path / "cache")

    def test_a_new_run_loads_its_own_model(
        self, tmp_path: Path, fake_whisper: type[_FakeWhisperModel]
    ) -> None:
        first = _transcriber(tmp_path).load_model()
        second = _transcriber(tmp_path).load_model()
        assert first is not second

    def test_missing_dependency_raises_an_actionable_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        with pytest.raises(RuntimeError, match="uv sync --extra audio"):
            _transcriber(tmp_path).load_model()


@pytest.mark.unit
class TestTranscribeFile:
    def _patch_model(
        self, monkeypatch: pytest.MonkeyPatch, transcriber: WhisperTranscriber
    ) -> _FakeWhisperModel:
        model = _FakeWhisperModel("tiny")
        monkeypatch.setattr(transcriber, "load_model", lambda: model)
        return model

    @pytest.mark.parametrize(
        ("pieces", "expected"),
        [
            (["Hello  ", " world", "again"], "Hello world again"),
            (["  ", "text", ""], "text"),
            ([None, "only"], "only"),
            ([], NO_TRANSCRIBABLE_TEXT),
            (["   ", "\n\t"], NO_TRANSCRIBABLE_TEXT),
        ],
    )
    def test_segments_are_joined_with_single_spaces(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        pieces: list[str | None],
        expected: str,
    ) -> None:
        transcriber = _transcriber(tmp_path)
        model = self._patch_model(monkeypatch, transcriber)
        model.segments = [_segment(piece) for piece in pieces]
        assert transcriber.transcribe_file(_chunk(tmp_path).path) == expected

    def test_transcribe_arguments_come_from_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        whisper = Whisper(beam_size=3, vad_filter=False)
        transcriber = _transcriber(tmp_path, whisper=whisper, language="de")
        model = self._patch_model(monkeypatch, transcriber)
        path = _chunk(tmp_path).path
        transcriber.transcribe_file(path)
        assert model.transcribe_calls[0] == {
            "path": str(path),
            "beam_size": 3,
            "vad_filter": False,
            "language": "de",
        }

    def test_auto_language_is_sent_as_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)
        model = self._patch_model(monkeypatch, transcriber)
        transcriber.transcribe_file(_chunk(tmp_path).path)
        assert model.transcribe_calls[0]["language"] is None

    def test_model_load_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)

        def _boom() -> Any:
            raise RuntimeError("faster-whisper is not installed")

        monkeypatch.setattr(transcriber, "load_model", _boom)
        assert transcriber.transcribe_file(_chunk(tmp_path).path) is None

    def test_inference_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)
        model = self._patch_model(monkeypatch, transcriber)

        def _boom(_path: str, **_kwargs: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

        model.transcribe = _boom  # type: ignore[method-assign]
        assert transcriber.transcribe_file(_chunk(tmp_path).path) is None


@pytest.mark.unit
class TestTranscribePayload:
    async def test_the_transcript_is_returned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)
        monkeypatch.setattr(transcriber, "transcribe_file", lambda _p: "local text")
        payload = _chunk(tmp_path, 2)
        result = await transcriber.transcribe_payload(payload, None)
        assert result == (payload, payload.image_name, "local text", None, 2)

    async def test_none_becomes_an_error_placeholder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)
        monkeypatch.setattr(transcriber, "transcribe_file", lambda _p: None)
        result = await transcriber.transcribe_payload(_chunk(tmp_path, 2), None)
        assert result[2] == "[transcription error: rec_chunk_0003.wav]"

    async def test_an_exception_becomes_an_error_placeholder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)

        def _boom(_path: Path) -> str:
            raise RuntimeError("model crashed")

        monkeypatch.setattr(transcriber, "transcribe_file", _boom)
        result = await transcriber.transcribe_payload(_chunk(tmp_path), None)
        assert result[2].startswith("[transcription error")

    async def test_inference_runs_off_the_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)
        threads: list[int] = []

        def _capture(_path: Path) -> str:
            threads.append(threading.get_ident())
            return "ok"

        monkeypatch.setattr(transcriber, "transcribe_file", _capture)
        await transcriber.transcribe_payload(_chunk(tmp_path), None)
        assert threads and threads[0] != threading.get_ident()

    def test_a_missing_runtime_raises_with_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        transcriber = _transcriber(tmp_path)
        monkeypatch.setattr(
            whisper_module, "is_faster_whisper_available", lambda: False
        )
        with pytest.raises(WhisperMissingError, match="uv sync --extra audio"):
            transcriber._handler()


@pytest.mark.unit
class TestNoTranscribableTextSentinel:
    def test_sentinel_is_byte_identical_to_the_tesseract_placeholder(
        self, tmp_path: Path
    ) -> None:
        """The Tesseract path must emit exactly this string for empty OCR."""
        from chronotranscriber.images.tesseract_runtime import perform_ocr

        img_path = tmp_path / "blank.png"
        img_path.write_bytes(b"")

        fake_image = MagicMock()
        fake_image.__enter__ = MagicMock(return_value=fake_image)
        fake_image.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "chronotranscriber.images.tesseract_runtime.Image.open",
                return_value=fake_image,
            ),
            patch(
                "chronotranscriber.images.tesseract_runtime.pytesseract.image_to_string",
                return_value="   \n  ",
            ),
        ):
            result = perform_ocr(img_path, "--oem 3 --psm 6")

        assert result == NO_TRANSCRIBABLE_TEXT

    def test_sentinel_is_classified_as_no_text_not_an_error(self) -> None:
        from chronotranscriber.llm.response_parsing import detect_transcription_cause

        assert detect_transcription_cause(NO_TRANSCRIBABLE_TEXT) == "no_text"
