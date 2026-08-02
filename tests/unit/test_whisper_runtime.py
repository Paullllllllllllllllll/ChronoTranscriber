"""Tests for modules.audio.whisper_runtime.

``faster-whisper`` is an optional extra; every test here fakes it, so the
suite never requires the package to be installed.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from modules.audio import whisper_runtime
from modules.audio.constants import NO_TRANSCRIBABLE_TEXT
from modules.audio.whisper_runtime import (
    _resolve_runtime,
    clear_model_cache,
    ensure_faster_whisper_available,
    get_whisper_model,
    is_faster_whisper_available,
    transcribe_file,
)


@pytest.fixture(autouse=True)
def _clear_whisper_model_cache() -> Generator[None]:
    """Keep the process-wide model cache out of the tests' way."""
    clear_model_cache()
    yield
    clear_model_cache()


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


def _install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a stub ``faster_whisper`` module into ``sys.modules``."""
    _FakeWhisperModel.instances = []
    module = ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)


def _segment(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAvailability:
    def test_available_when_the_spec_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "importlib.util.find_spec", lambda name: object() if name else None
        )
        assert is_faster_whisper_available() is True

    def test_unavailable_when_the_spec_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert is_faster_whisper_available() is False

    @pytest.mark.parametrize("error", [ImportError("boom"), ValueError("bad name")])
    def test_unavailable_when_the_lookup_raises(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def _find_spec(_name: str) -> Any:
            raise error

        monkeypatch.setattr("importlib.util.find_spec", _find_spec)
        assert is_faster_whisper_available() is False

    def test_ensure_returns_true_when_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            whisper_runtime, "is_faster_whisper_available", lambda: True
        )
        assert ensure_faster_whisper_available() is True

    def test_ensure_prints_an_install_hint_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            whisper_runtime, "is_faster_whisper_available", lambda: False
        )
        assert ensure_faster_whisper_available() is False
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "faster-whisper is not installed" in combined
        assert "uv sync --extra audio" in combined


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveRuntime:
    def test_defaults(self) -> None:
        assert _resolve_runtime({}) == ("large-v3", "auto", "default", None)

    def test_model_path_overrides_model_size(self) -> None:
        model_ref, *_ = _resolve_runtime(
            {"model_size": "small", "model_path": "/models/custom"}
        )
        assert model_ref == "/models/custom"

    def test_auto_compute_type_maps_to_the_library_sentinel(self) -> None:
        _ref, _device, compute_type, _root = _resolve_runtime({"compute_type": "auto"})
        assert compute_type == "default"

    def test_empty_compute_type_maps_to_the_library_sentinel(self) -> None:
        _ref, _device, compute_type, _root = _resolve_runtime({"compute_type": ""})
        assert compute_type == "default"

    def test_explicit_device_and_compute_type_are_lowercased(self) -> None:
        _ref, device, compute_type, _root = _resolve_runtime(
            {"device": " CUDA ", "compute_type": " Float16 "}
        )
        assert device == "cuda"
        assert compute_type == "float16"

    def test_blank_download_root_becomes_none(self) -> None:
        assert _resolve_runtime({"download_root": "  "})[3] is None

    def test_download_root_is_preserved(self) -> None:
        assert _resolve_runtime({"download_root": "/cache"})[3] == "/cache"


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelCache:
    def test_same_config_returns_the_cached_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        cfg = {"model_size": "tiny", "device": "cpu", "compute_type": "int8"}
        first = get_whisper_model(cfg)
        second = get_whisper_model(dict(cfg))
        assert first is second
        assert len(_FakeWhisperModel.instances) == 1

    def test_a_different_device_loads_a_new_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        first = get_whisper_model({"model_size": "tiny", "device": "cpu"})
        second = get_whisper_model({"model_size": "tiny", "device": "cuda"})
        assert first is not second
        assert len(_FakeWhisperModel.instances) == 2

    def test_a_different_model_size_loads_a_new_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        first = get_whisper_model({"model_size": "tiny"})
        second = get_whisper_model({"model_size": "large-v3"})
        assert first is not second

    def test_cache_key_ignores_the_download_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        first = get_whisper_model({"model_size": "tiny", "download_root": "/a"})
        second = get_whisper_model({"model_size": "tiny", "download_root": "/b"})
        assert first is second

    def test_download_root_is_forwarded_to_the_library(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        model = get_whisper_model({"model_size": "tiny", "download_root": "/cache"})
        assert model.kwargs["download_root"] == "/cache"

    def test_download_root_is_omitted_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        model = get_whisper_model({"model_size": "tiny"})
        assert "download_root" not in model.kwargs
        assert model.kwargs == {"device": "auto", "compute_type": "default"}

    def test_clear_model_cache_forces_a_reload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_faster_whisper(monkeypatch)
        first = get_whisper_model({"model_size": "tiny"})
        clear_model_cache()
        second = get_whisper_model({"model_size": "tiny"})
        assert first is not second

    def test_missing_dependency_raises_an_actionable_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        with pytest.raises(RuntimeError, match="uv sync --extra audio"):
            get_whisper_model({"model_size": "tiny"})


# ---------------------------------------------------------------------------
# transcribe_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTranscribeFile:
    @pytest.fixture
    def audio(self, tmp_path: Path) -> Path:
        path = tmp_path / "chunk.wav"
        path.write_bytes(b"RIFF")
        return path

    def _patch_model(
        self, monkeypatch: pytest.MonkeyPatch, segments: list[Any]
    ) -> _FakeWhisperModel:
        model = _FakeWhisperModel("tiny")
        model.segments = segments
        monkeypatch.setattr(whisper_runtime, "get_whisper_model", lambda _cfg: model)
        return model

    def test_segments_are_joined_with_single_spaces(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        self._patch_model(
            monkeypatch, [_segment("Hello  "), _segment(" world"), _segment("again")]
        )
        assert transcribe_file(audio, {}) == "Hello world again"

    def test_blank_segments_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        self._patch_model(monkeypatch, [_segment("  "), _segment("text"), _segment("")])
        assert transcribe_file(audio, {}) == "text"

    def test_a_none_segment_text_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        self._patch_model(monkeypatch, [_segment(None), _segment("only")])
        assert transcribe_file(audio, {}) == "only"

    def test_no_segments_yields_the_shared_sentinel(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        self._patch_model(monkeypatch, [])
        assert transcribe_file(audio, {}) == NO_TRANSCRIBABLE_TEXT

    def test_whitespace_only_transcript_yields_the_shared_sentinel(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        self._patch_model(monkeypatch, [_segment("   "), _segment("\n\t")])
        assert transcribe_file(audio, {}) == NO_TRANSCRIBABLE_TEXT

    def test_transcribe_arguments_come_from_the_config(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        model = self._patch_model(monkeypatch, [_segment("x")])
        transcribe_file(audio, {"beam_size": 3, "vad_filter": False, "language": "de"})
        call = model.transcribe_calls[0]
        assert call["path"] == str(audio)
        assert call["beam_size"] == 3
        assert call["vad_filter"] is False
        assert call["language"] == "de"

    def test_blank_language_is_sent_as_none_for_auto_detection(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        model = self._patch_model(monkeypatch, [_segment("x")])
        transcribe_file(audio, {"language": "  "})
        assert model.transcribe_calls[0]["language"] is None

    def test_default_beam_size_and_vad_filter(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        model = self._patch_model(monkeypatch, [_segment("x")])
        transcribe_file(audio, {})
        call = model.transcribe_calls[0]
        assert call["beam_size"] == 5
        assert call["vad_filter"] is True

    def test_model_load_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        def _boom(_cfg: dict[str, Any]) -> Any:
            raise RuntimeError("faster-whisper is not installed")

        monkeypatch.setattr(whisper_runtime, "get_whisper_model", _boom)
        assert transcribe_file(audio, {}) is None

    def test_inference_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        model = _FakeWhisperModel("tiny")

        def _boom(_path: str, **_kwargs: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

        model.transcribe = _boom  # type: ignore[method-assign]
        monkeypatch.setattr(whisper_runtime, "get_whisper_model", lambda _cfg: model)
        assert transcribe_file(audio, {}) is None
