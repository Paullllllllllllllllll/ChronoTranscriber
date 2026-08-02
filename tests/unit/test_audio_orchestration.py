"""Orchestration tests for the audio workflow.

Covers the WorkflowManager's audio branch (batch exclusion, the missing local
runtime, the method guard), the CLI-to-config translation, and item-level
resume — all with mocked runtimes, so neither ffmpeg nor faster-whisper is
required and no real API, ledger, or config file is touched.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from modules.audio.backends.base import build_legacy_response
from modules.audio.paths import prepare_audio_output
from modules.transcribe import config_builder
from modules.transcribe import manager as manager_module
from modules.transcribe.config_builder import (
    AUDIO_CLI_PROVIDERS,
    AUDIO_METHODS,
    _collect_files_for_type,
    _resolve_audio_config_from_cli,
    create_config_from_cli_args,
)
from modules.transcribe.manager import WorkflowManager
from modules.transcribe.resume import ProcessingState, ResumeChecker
from modules.transcribe.user_config import UserConfiguration

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _StubTracker:
    """Token tracker stand-in: admits everything, records nothing on disk."""

    def __init__(self) -> None:
        self.reservations = 0

    def try_reserve(self, *_args: Any, **_kwargs: Any) -> int:
        self.reservations += 1
        return 1

    def release(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def commit(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def add_tokens(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _StubAudioTranscriber:
    """AudioTranscriber-shaped facade returning one canned transcript."""

    provider_name = "stub"
    model = "stub-audio-model"

    def __init__(self, text: str = "spoken words") -> None:
        self.text = text
        self.calls: list[Any] = []

    async def transcribe_audio_chunk(self, payload: Any) -> dict[str, Any]:
        self.calls.append(payload)
        return build_legacy_response(
            output_text=self.text, provider=self.provider_name, model=self.model
        )

    def rekey(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolate_audio_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[_StubTracker]:
    """Keep the manager off ffmpeg, the real ledger, and the batch API."""
    from modules.audio.ffmpeg_runtime import configure_ffmpeg_executables

    configure_ffmpeg_executables({})
    tracker = _StubTracker()
    monkeypatch.setattr(manager_module, "get_token_tracker", lambda: tracker)
    monkeypatch.setattr(manager_module, "is_ffmpeg_available", lambda: False)
    monkeypatch.setattr(manager_module, "ensure_ffmpeg_available", lambda: False)
    yield tracker
    configure_ffmpeg_executables({})


def _user_config(**overrides: Any) -> UserConfiguration:
    config = UserConfiguration()
    config.processing_type = "audio"
    config.transcription_method = "audio-api"
    config.use_batch_processing = False
    config.selected_items = []
    config.process_all = False
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _manager(
    user_config: UserConfiguration,
    paths_config: dict[str, Any],
    audio_config: dict[str, Any],
) -> WorkflowManager:
    return WorkflowManager(
        user_config,
        paths_config,
        {"transcription_model": {"provider": "openai", "name": "gpt-4o"}},
        {"concurrency": {"transcription": {"concurrency_limit": 2}}},
        {"tesseract_image_processing": {"ocr": {}}, "postprocessing": {}},
        audio_config=audio_config,
    )


# ---------------------------------------------------------------------------
# WorkflowManager: batch exclusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioBatchExclusion:
    async def test_batch_flag_never_reaches_the_batch_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("submit_batch must never run for audio")

        monkeypatch.setattr(manager_module, "submit_batch", _forbidden)

        user_config = _user_config(use_batch_processing=True)
        workflow = _manager(
            user_config, mock_paths_config_with_audio, mock_audio_config
        )
        transcriber = _StubAudioTranscriber()

        await workflow.process_single_audio(sample_audio_file, transcriber)

        assert len(transcriber.calls) == 1
        output = capsys.readouterr().out
        assert "no batch API" in output

    async def test_the_transcript_is_written_synchronously(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        user_config = _user_config(use_batch_processing=True)
        workflow = _manager(
            user_config, mock_paths_config_with_audio, mock_audio_config
        )
        await workflow.process_single_audio(
            sample_audio_file, _StubAudioTranscriber("hello there")
        )

        _parent, output_txt, _jsonl = prepare_audio_output(
            sample_audio_file,
            output_dir=workflow.audio_output_dir,
            input_paths_is_output_path=False,
            output_mode="hash",
            input_root=None,
        )
        assert output_txt.exists()
        assert "hello there" in output_txt.read_text(encoding="utf-8")

    async def test_a_whole_file_run_leaves_no_chunk_directory(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        await workflow.process_single_audio(sample_audio_file, _StubAudioTranscriber())
        parent, _txt, _jsonl = prepare_audio_output(
            sample_audio_file,
            output_dir=workflow.audio_output_dir,
            input_paths_is_output_path=False,
            output_mode="hash",
            input_root=None,
        )
        assert not (parent / "audio_chunks").exists()

    async def test_the_jsonl_records_the_chunk_plan_signature(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        import json

        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        await workflow.process_single_audio(sample_audio_file, _StubAudioTranscriber())
        _parent, _txt, jsonl_path = prepare_audio_output(
            sample_audio_file,
            output_dir=workflow.audio_output_dir,
            input_paths_is_output_path=False,
            output_mode="hash",
            input_root=None,
        )
        signatures = [
            json.loads(line)["file_provenance"]["chunk_plan_signature"]
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "file_provenance" in line
        ]
        assert signatures and signatures[0].startswith("1x0s-")


# ---------------------------------------------------------------------------
# WorkflowManager: missing local runtime and the method guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioRuntimeGuards:
    async def test_missing_faster_whisper_skips_the_item_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        monkeypatch.setattr(
            manager_module, "ensure_faster_whisper_available", lambda: False
        )
        workflow = _manager(
            _user_config(transcription_method="whisper"),
            mock_paths_config_with_audio,
            mock_audio_config,
        )
        await workflow.process_single_audio(sample_audio_file, None)

        _parent, output_txt, _jsonl = prepare_audio_output(
            sample_audio_file,
            output_dir=workflow.audio_output_dir,
            input_paths_is_output_path=False,
            output_mode="hash",
            input_root=None,
        )
        assert not output_txt.exists()

    async def test_local_whisper_runs_when_the_runtime_is_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        monkeypatch.setattr(
            manager_module, "ensure_faster_whisper_available", lambda: True
        )
        monkeypatch.setattr(
            "modules.transcribe.pipeline.whisper_transcribe_file",
            lambda path, cfg: "offline transcript",
        )
        workflow = _manager(
            _user_config(transcription_method="whisper"),
            mock_paths_config_with_audio,
            mock_audio_config,
        )
        await workflow.process_single_audio(sample_audio_file, None)

        _parent, output_txt, _jsonl = prepare_audio_output(
            sample_audio_file,
            output_dir=workflow.audio_output_dir,
            input_paths_is_output_path=False,
            output_mode="hash",
            input_root=None,
        )
        assert "offline transcript" in output_txt.read_text(encoding="utf-8")

    @pytest.mark.parametrize("method", ["gpt", "tesseract", "native", ""])
    async def test_a_non_audio_method_is_refused(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
        method: str,
    ) -> None:
        workflow = _manager(
            _user_config(transcription_method=method),
            mock_paths_config_with_audio,
            mock_audio_config,
        )
        with pytest.raises(ValueError, match="audio-api"):
            await workflow.process_single_audio(
                sample_audio_file, _StubAudioTranscriber()
            )

    async def test_the_method_guard_counts_the_item_as_failed(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        user_config = _user_config(
            transcription_method="gpt",
            selected_items=[sample_audio_file],
            resume_mode="overwrite",
        )
        workflow = _manager(
            user_config, mock_paths_config_with_audio, mock_audio_config
        )
        summary = await workflow.process_selected_items(_StubAudioTranscriber())
        assert summary.failed == 1
        assert summary.processed == 0

    async def test_audio_api_without_a_transcriber_raises(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        sample_audio_file: Path,
    ) -> None:
        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        with pytest.raises(ValueError, match="No audio transcriber"):
            await workflow.process_single_audio(sample_audio_file, None)


@pytest.mark.unit
class TestAudioManagerWiring:
    def test_audio_output_dir_comes_from_the_paths_config(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        assert workflow.audio_output_dir == tmp_path / "audio_out"

    def test_the_audio_output_dir_is_not_created_eagerly(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
    ) -> None:
        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        assert not workflow.audio_output_dir.exists()

    def test_the_audio_postprocessing_profile_is_picked_up(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
    ) -> None:
        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        assert (
            workflow.audio_postprocessing_config == mock_audio_config["postprocessing"]
        )

    def test_a_missing_audio_postprocessing_section_falls_back_to_none(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
    ) -> None:
        audio_config = deepcopy(mock_audio_config)
        audio_config.pop("postprocessing")
        workflow = _manager(_user_config(), mock_paths_config_with_audio, audio_config)
        assert workflow.audio_postprocessing_config is None

    def test_the_audio_concurrency_override_is_applied_without_mutation(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
    ) -> None:
        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        effective = workflow._audio_concurrency_config()
        assert effective["concurrency"]["transcription"]["concurrency_limit"] == 2
        assert (
            workflow.concurrency_config["concurrency"]["transcription"][
                "concurrency_limit"
            ]
            == 2
        )

    def test_chunk_settings_pick_the_provider_ceiling(
        self,
        mock_paths_config_with_audio: dict[str, Any],
        mock_audio_config: dict[str, Any],
    ) -> None:
        from modules.audio.constants import (
            GEMINI_INLINE_LIMIT_BYTES,
            OPENAI_UPLOAD_LIMIT_BYTES,
        )

        workflow = _manager(
            _user_config(), mock_paths_config_with_audio, mock_audio_config
        )
        assert (
            workflow._audio_chunk_settings().max_request_bytes
            == OPENAI_UPLOAD_LIMIT_BYTES
        )

        google_config = deepcopy(mock_audio_config)
        google_config["audio_transcription"]["provider"] = "google"
        google_workflow = _manager(
            _user_config(), mock_paths_config_with_audio, google_config
        )
        assert (
            google_workflow._audio_chunk_settings().max_request_bytes
            == GEMINI_INLINE_LIMIT_BYTES
        )


# ---------------------------------------------------------------------------
# config_builder: file collection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioFileCollection:
    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "recordings"
        directory.mkdir()
        for name in ("a.mp3", "b.wav", "c.m4a", "d.flac", "notes.txt", "scan.pdf"):
            (directory / name).write_bytes(b"x")
        nested = directory / "nested"
        nested.mkdir()
        (nested / "e.ogg").write_bytes(b"x")
        return directory

    def test_audio_extensions_are_collected(self, audio_dir: Path) -> None:
        args = Namespace(files=None, recursive=False)
        found = _collect_files_for_type(audio_dir, "audio", args)
        assert {p.name for p in found} == {"a.mp3", "b.wav", "c.m4a", "d.flac"}

    def test_non_audio_files_are_ignored(self, audio_dir: Path) -> None:
        args = Namespace(files=None, recursive=False)
        found = _collect_files_for_type(audio_dir, "audio", args)
        assert not any(p.suffix in {".txt", ".pdf"} for p in found)

    def test_recursive_collection_descends(self, audio_dir: Path) -> None:
        args = Namespace(files=None, recursive=True)
        found = _collect_files_for_type(audio_dir, "audio", args)
        assert "e.ogg" in {p.name for p in found}

    def test_a_single_audio_file_input_is_returned_as_is(self, audio_dir: Path) -> None:
        args = Namespace(files=None, recursive=False)
        target = audio_dir / "a.mp3"
        assert _collect_files_for_type(target, "audio", args) == [target]

    def test_explicit_file_names_are_honored(self, audio_dir: Path) -> None:
        args = Namespace(files=["b.wav"], recursive=False)
        found = _collect_files_for_type(audio_dir, "audio", args)
        assert found == [audio_dir / "b.wav"]


# ---------------------------------------------------------------------------
# config_builder: CLI validation
# ---------------------------------------------------------------------------


def _cli_args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = {
        "input": None,
        "output": None,
        "type": "audio",
        "method": None,
        "auto": False,
        "batch": False,
        "schema": None,
        "context": None,
        "context_image": None,
        "model": None,
        "provider": None,
        "reasoning_effort": None,
        "model_verbosity": None,
        "max_output_tokens": None,
        "resume": None,
        "force": None,
        "files": None,
        "recursive": False,
        "output_format": None,
        "pages": None,
        "retry_errors": False,
        "output_mode": None,
        "sync_fallback": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _paths_config() -> dict[str, Any]:
    return {"general": {}, "file_paths": {}}


@pytest.fixture
def audio_input_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "audio_in"
    directory.mkdir()
    (directory / "rec.mp3").write_bytes(b"x")
    return directory


@pytest.fixture
def audio_output_root(tmp_path: Path) -> Path:
    directory = tmp_path / "audio_out"
    directory.mkdir()
    return directory


@pytest.mark.unit
class TestAudioCliValidation:
    def test_default_method_for_audio_is_the_remote_api(
        self, audio_input_dir: Path, audio_output_root: Path
    ) -> None:
        config = create_config_from_cli_args(
            _cli_args(), audio_input_dir, audio_output_root, _paths_config()
        )
        assert config.transcription_method == "audio-api"
        assert config.processing_type == "audio"

    def test_explicit_whisper_method_is_accepted(
        self, audio_input_dir: Path, audio_output_root: Path
    ) -> None:
        config = create_config_from_cli_args(
            _cli_args(method="whisper"),
            audio_input_dir,
            audio_output_root,
            _paths_config(),
        )
        assert config.transcription_method == "whisper"

    def test_audio_with_tesseract_is_refused(
        self, audio_input_dir: Path, audio_output_root: Path
    ) -> None:
        with pytest.raises(ValueError, match="audio-api or whisper"):
            create_config_from_cli_args(
                _cli_args(method="tesseract"),
                audio_input_dir,
                audio_output_root,
                _paths_config(),
            )

    def test_audio_with_gpt_is_refused(
        self, audio_input_dir: Path, audio_output_root: Path
    ) -> None:
        with pytest.raises(ValueError, match="audio-api or whisper"):
            create_config_from_cli_args(
                _cli_args(method="gpt"),
                audio_input_dir,
                audio_output_root,
                _paths_config(),
            )

    @pytest.mark.parametrize("method", list(AUDIO_METHODS))
    def test_an_audio_method_on_a_document_type_is_refused(
        self, audio_input_dir: Path, audio_output_root: Path, method: str
    ) -> None:
        with pytest.raises(ValueError, match="requires --type audio"):
            create_config_from_cli_args(
                _cli_args(type="pdfs", method=method),
                audio_input_dir,
                audio_output_root,
                _paths_config(),
            )

    def test_batch_is_forced_off_with_a_warning(
        self,
        audio_input_dir: Path,
        audio_output_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = create_config_from_cli_args(
            _cli_args(batch=True),
            audio_input_dir,
            audio_output_root,
            _paths_config(),
        )
        assert config.use_batch_processing is False
        assert "ignoring --batch" in capsys.readouterr().out

    def test_selected_items_are_the_collected_recordings(
        self, audio_input_dir: Path, audio_output_root: Path
    ) -> None:
        config = create_config_from_cli_args(
            _cli_args(), audio_input_dir, audio_output_root, _paths_config()
        )
        assert [p.name for p in config.selected_items] == ["rec.mp3"]

    def test_an_empty_audio_directory_is_refused(
        self, tmp_path: Path, audio_output_root: Path
    ) -> None:
        empty = tmp_path / "empty_in"
        empty.mkdir()
        with pytest.raises(ValueError, match="No items found"):
            create_config_from_cli_args(
                _cli_args(), empty, audio_output_root, _paths_config()
            )


# ---------------------------------------------------------------------------
# config_builder: audio-config CLI overrides
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveAudioConfigFromCli:
    def test_no_overrides_returns_an_equal_copy(
        self, mock_audio_config: dict[str, Any]
    ) -> None:
        effective = _resolve_audio_config_from_cli(_cli_args(), mock_audio_config)
        assert effective == mock_audio_config
        assert effective is not mock_audio_config

    def test_provider_and_model_land_in_the_provider_section(
        self, mock_audio_config: dict[str, Any]
    ) -> None:
        effective = _resolve_audio_config_from_cli(
            _cli_args(provider="google", model="gemini-3-pro"), mock_audio_config
        )
        assert effective["audio_transcription"]["provider"] == "google"
        assert effective["audio_transcription"]["google"]["model"] == "gemini-3-pro"

    def test_the_original_config_is_never_mutated(
        self, mock_audio_config: dict[str, Any]
    ) -> None:
        original = deepcopy(mock_audio_config)
        _resolve_audio_config_from_cli(
            _cli_args(provider="google", model="gemini-3-pro"), mock_audio_config
        )
        assert mock_audio_config == original

    def test_model_alone_targets_the_configured_provider(
        self, mock_audio_config: dict[str, Any]
    ) -> None:
        effective = _resolve_audio_config_from_cli(
            _cli_args(model="whisper-1"), mock_audio_config
        )
        assert effective["audio_transcription"]["openai"]["model"] == "whisper-1"

    def test_provider_matching_is_case_insensitive(
        self, mock_audio_config: dict[str, Any]
    ) -> None:
        effective = _resolve_audio_config_from_cli(
            _cli_args(provider="GOOGLE"), mock_audio_config
        )
        assert effective["audio_transcription"]["provider"] == "google"

    @pytest.mark.parametrize("provider", ["anthropic", "openrouter", "nonsense"])
    def test_a_non_audio_provider_is_refused(
        self, mock_audio_config: dict[str, Any], provider: str
    ) -> None:
        with pytest.raises(ValueError, match="does not serve audio"):
            _resolve_audio_config_from_cli(
                _cli_args(provider=provider), mock_audio_config
            )

    def test_the_error_lists_the_supported_venues(
        self, mock_audio_config: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            _resolve_audio_config_from_cli(
                _cli_args(provider="anthropic"), mock_audio_config
            )
        for name in AUDIO_CLI_PROVIDERS:
            assert name in str(excinfo.value)

    def test_an_empty_config_gains_the_section(self) -> None:
        effective = _resolve_audio_config_from_cli(
            _cli_args(provider="openai", model="whisper-1"), {}
        )
        assert effective["audio_transcription"]["openai"]["model"] == "whisper-1"

    def test_chat_only_flags_are_reported_as_ignored(
        self,
        mock_audio_config: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _resolve_audio_config_from_cli(
            _cli_args(reasoning_effort="high", model_verbosity="low"),
            mock_audio_config,
        )
        output = capsys.readouterr().out
        assert "--reasoning-effort does not apply" in output
        assert "--model-verbosity does not apply" in output

    def test_applied_overrides_are_reported(
        self,
        mock_audio_config: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _resolve_audio_config_from_cli(_cli_args(provider="google"), mock_audio_config)
        assert "CLI audio overrides" in capsys.readouterr().out

    def test_audio_methods_and_providers_are_exported(self) -> None:
        assert config_builder.AUDIO_METHODS == ("audio-api", "whisper")
        assert config_builder.AUDIO_CLI_PROVIDERS == ("openai", "google")


# ---------------------------------------------------------------------------
# Item-level resume for audio
# ---------------------------------------------------------------------------


def _resume_paths_config(output_dir: Path) -> dict[str, Any]:
    text = str(output_dir)
    return {
        "general": {"input_paths_is_output_path": False},
        "file_paths": {
            "PDFs": {"input": text, "output": text},
            "Images": {"input": text, "output": text},
            "EPUBs": {"input": text, "output": text},
            "MOBIs": {"input": text, "output": text},
            "Audio": {"input": text, "output": text},
        },
    }


@pytest.mark.unit
class TestAudioResume:
    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "audio_out"
        directory.mkdir()
        return directory

    @pytest.fixture
    def recording(self, tmp_path: Path) -> Path:
        directory = tmp_path / "audio_in"
        directory.mkdir()
        path = directory / "lecture.mp3"
        path.write_bytes(b"x")
        return path

    def _checker(self, output_dir: Path, **overrides: Any) -> ResumeChecker:
        kwargs: dict[str, Any] = {
            "resume_mode": "skip",
            "paths_config": _resume_paths_config(output_dir),
            "audio_output_dir": output_dir,
        }
        kwargs.update(overrides)
        return ResumeChecker(**kwargs)

    def _prepare(
        self, recording: Path, output_dir: Path, **overrides: Any
    ) -> tuple[Path, Path, Path]:
        kwargs: dict[str, Any] = {
            "output_dir": output_dir,
            "input_paths_is_output_path": False,
            "output_mode": "hash",
            "input_root": None,
        }
        kwargs.update(overrides)
        return prepare_audio_output(recording, **kwargs)

    def test_nothing_on_disk_is_none(self, recording: Path, output_dir: Path) -> None:
        result = self._checker(output_dir).should_skip(recording, "audio")
        assert result.state is ProcessingState.NONE

    def test_an_untouched_working_folder_is_none(
        self, recording: Path, output_dir: Path
    ) -> None:
        # prepare_audio_output touches an EMPTY JSONL; that is not partial work.
        self._prepare(recording, output_dir)
        result = self._checker(output_dir).should_skip(recording, "audio")
        assert result.state is ProcessingState.NONE

    def test_a_non_empty_jsonl_alone_is_partial(
        self, recording: Path, output_dir: Path
    ) -> None:
        _parent, _txt, jsonl_path = self._prepare(recording, output_dir)
        jsonl_path.write_text('{"image_name": "x", "text_chunk": "y"}\n', "utf-8")
        result = self._checker(output_dir).should_skip(recording, "audio")
        assert result.state is ProcessingState.PARTIAL
        assert result.output_path == jsonl_path

    def test_a_non_empty_transcript_is_complete(
        self, recording: Path, output_dir: Path
    ) -> None:
        _parent, txt_path, _jsonl = self._prepare(recording, output_dir)
        txt_path.write_text("the transcript", encoding="utf-8")
        result = self._checker(output_dir).should_skip(recording, "audio")
        assert result.state is ProcessingState.COMPLETE
        assert result.output_path == txt_path

    def test_an_empty_transcript_is_not_complete(
        self, recording: Path, output_dir: Path
    ) -> None:
        _parent, txt_path, _jsonl = self._prepare(recording, output_dir)
        txt_path.write_text("", encoding="utf-8")
        result = self._checker(output_dir).should_skip(recording, "audio")
        assert result.state is not ProcessingState.COMPLETE

    def test_overwrite_mode_never_skips(
        self, recording: Path, output_dir: Path
    ) -> None:
        _parent, txt_path, _jsonl = self._prepare(recording, output_dir)
        txt_path.write_text("the transcript", encoding="utf-8")
        checker = self._checker(output_dir, resume_mode="overwrite")
        assert checker.should_skip(recording, "audio").state is ProcessingState.NONE

    def test_retry_errors_downgrades_a_placeholder_transcript_to_partial(
        self, recording: Path, output_dir: Path
    ) -> None:
        _parent, txt_path, _jsonl = self._prepare(recording, output_dir)
        txt_path.write_text(
            "rec_chunk_0001.mp3: [transcription error: rec_chunk_0001.mp3]",
            encoding="utf-8",
        )
        checker = self._checker(output_dir, retry_errors=True)
        assert checker.should_skip(recording, "audio").state is ProcessingState.PARTIAL

    def test_mirror_mode_finds_the_mirrored_transcript(
        self, recording: Path, output_dir: Path
    ) -> None:
        input_root = recording.parent
        _parent, txt_path, _jsonl = self._prepare(
            recording,
            output_dir,
            output_mode="mirror",
            input_root=input_root,
        )
        txt_path.write_text("mirrored transcript", encoding="utf-8")
        checker = self._checker(output_dir, output_mode="mirror", input_root=input_root)
        result = checker.should_skip(recording, "audio")
        assert result.state is ProcessingState.COMPLETE
        assert result.output_path == txt_path

    def test_mirror_mode_without_output_is_none(
        self, recording: Path, output_dir: Path
    ) -> None:
        checker = self._checker(
            output_dir, output_mode="mirror", input_root=recording.parent
        )
        assert checker.should_skip(recording, "audio").state is ProcessingState.NONE

    def test_the_audio_output_dir_falls_back_to_the_paths_config(
        self, output_dir: Path
    ) -> None:
        checker = ResumeChecker(
            resume_mode="skip", paths_config=_resume_paths_config(output_dir)
        )
        assert checker.audio_output_dir == output_dir

    def test_filter_items_skips_completed_recordings(
        self, recording: Path, output_dir: Path
    ) -> None:
        _parent, txt_path, _jsonl = self._prepare(recording, output_dir)
        txt_path.write_text("done", encoding="utf-8")
        remaining, skipped = self._checker(output_dir).filter_items(
            [recording], "audio"
        )
        assert remaining == []
        assert len(skipped) == 1
