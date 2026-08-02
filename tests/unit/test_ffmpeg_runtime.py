"""Tests for modules.audio.ffmpeg_runtime.

Every subprocess call is mocked: the suite never requires a real ffmpeg or
ffprobe binary on PATH.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from modules.audio import ffmpeg_runtime
from modules.audio.ffmpeg_runtime import (
    _encoder_args,
    configure_ffmpeg_executables,
    cut_segment,
    ensure_ffmpeg_available,
    get_ffmpeg_command,
    get_ffprobe_command,
    is_ffmpeg_available,
    is_ffprobe_available,
    probe_duration_seconds,
)


@pytest.fixture(autouse=True)
def _reset_ffmpeg_module_state() -> Generator[None]:
    """Clear the configured paths and the availability cache around each test."""
    configure_ffmpeg_executables({})
    yield
    configure_ffmpeg_executables({})


class _Completed:
    """Stand-in for ``subprocess.CompletedProcess`` with byte streams."""

    def __init__(
        self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _record_run(
    monkeypatch: pytest.MonkeyPatch,
    results: list[_Completed] | _Completed,
) -> list[dict[str, Any]]:
    """Patch ``subprocess.run`` in the module and record every invocation."""
    queue = list(results) if isinstance(results, list) else [results]
    calls: list[dict[str, Any]] = []

    def _run(argv: Any, **kwargs: Any) -> _Completed:
        calls.append({"argv": argv, "kwargs": kwargs})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(ffmpeg_runtime.subprocess, "run", _run)
    return calls


# ---------------------------------------------------------------------------
# Executable configuration and availability caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutableConfiguration:
    def test_defaults_resolve_to_path_lookups(self) -> None:
        assert get_ffmpeg_command() == "ffmpeg"
        assert get_ffprobe_command() == "ffprobe"

    def test_configured_paths_override_the_defaults(self, tmp_path: Path) -> None:
        ffmpeg = tmp_path / "ffmpeg.exe"
        ffprobe = tmp_path / "ffprobe.exe"
        ffmpeg.write_text("x", encoding="utf-8")
        ffprobe.write_text("x", encoding="utf-8")
        configure_ffmpeg_executables(
            {"ffmpeg": {"ffmpeg_cmd": str(ffmpeg), "ffprobe_cmd": str(ffprobe)}}
        )
        assert get_ffmpeg_command() == str(ffmpeg)
        assert get_ffprobe_command() == str(ffprobe)

    def test_whitespace_only_paths_fall_back_to_path_lookups(self) -> None:
        configure_ffmpeg_executables(
            {"ffmpeg": {"ffmpeg_cmd": "   ", "ffprobe_cmd": "\t"}}
        )
        assert get_ffmpeg_command() == "ffmpeg"
        assert get_ffprobe_command() == "ffprobe"

    def test_missing_section_is_tolerated(self) -> None:
        configure_ffmpeg_executables({"ffmpeg": None})
        assert get_ffmpeg_command() == "ffmpeg"

    def test_nonexistent_configured_path_warns_but_is_still_used(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing = tmp_path / "nowhere" / "ffmpeg.exe"
        with caplog.at_level("WARNING"):
            configure_ffmpeg_executables({"ffmpeg": {"ffmpeg_cmd": str(missing)}})
        assert get_ffmpeg_command() == str(missing)
        assert "does not exist" in caplog.text


@pytest.mark.unit
class TestAvailabilityCaching:
    def test_ffmpeg_availability_is_probed_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def _probe(command: str) -> bool:
            calls.append(command)
            return True

        monkeypatch.setattr(ffmpeg_runtime, "_probe_executable", _probe)
        assert is_ffmpeg_available() is True
        assert is_ffmpeg_available() is True
        assert calls == ["ffmpeg"]

    def test_ffprobe_availability_is_probed_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            ffmpeg_runtime,
            "_probe_executable",
            lambda command: calls.append(command) or False,
        )
        assert is_ffprobe_available() is False
        assert is_ffprobe_available() is False
        assert calls == ["ffprobe"]

    def test_reconfiguring_invalidates_the_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def _probe(command: str) -> bool:
            calls.append(command)
            return True

        monkeypatch.setattr(ffmpeg_runtime, "_probe_executable", _probe)
        assert is_ffmpeg_available() is True

        override = tmp_path / "ffmpeg.exe"
        override.write_text("x", encoding="utf-8")
        configure_ffmpeg_executables({"ffmpeg": {"ffmpeg_cmd": str(override)}})

        assert is_ffmpeg_available() is True
        assert calls == ["ffmpeg", str(override)]

    def test_probe_executable_runs_a_version_check_without_a_shell(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "ffmpeg.exe"
        exe.write_text("x", encoding="utf-8")
        configure_ffmpeg_executables({"ffmpeg": {"ffmpeg_cmd": str(exe)}})
        calls = _record_run(monkeypatch, _Completed(returncode=0))

        assert is_ffmpeg_available() is True
        assert calls[0]["argv"] == [str(exe), "-version"]
        assert "shell" not in calls[0]["kwargs"]

    def test_probe_executable_reports_unavailable_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "ffmpeg.exe"
        exe.write_text("x", encoding="utf-8")
        configure_ffmpeg_executables({"ffmpeg": {"ffmpeg_cmd": str(exe)}})

        def _run(argv: Any, **kwargs: Any) -> Any:
            raise OSError("cannot execute")

        monkeypatch.setattr(ffmpeg_runtime.subprocess, "run", _run)
        assert is_ffmpeg_available() is False

    def test_probe_executable_reports_unavailable_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "ffmpeg.exe"
        exe.write_text("x", encoding="utf-8")
        configure_ffmpeg_executables({"ffmpeg": {"ffmpeg_cmd": str(exe)}})

        def _run(argv: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

        monkeypatch.setattr(ffmpeg_runtime.subprocess, "run", _run)
        assert is_ffmpeg_available() is False

    def test_missing_binary_on_path_short_circuits_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime.shutil, "which", lambda _cmd: None)
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        assert is_ffmpeg_available() is False
        assert calls == []


@pytest.mark.unit
class TestEnsureFfmpegAvailable:
    def test_returns_true_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)
        assert ensure_ffmpeg_available() is True

    def test_prints_an_install_hint_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: False)
        assert ensure_ffmpeg_available() is False
        captured = capsys.readouterr()
        assert "FFmpeg is not installed" in captured.out + captured.err


# ---------------------------------------------------------------------------
# probe_duration_seconds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProbeDurationSeconds:
    @pytest.fixture
    def audio(self, tmp_path: Path) -> Path:
        path = tmp_path / "rec.wav"
        path.write_bytes(b"RIFF")
        return path

    def test_ffprobe_json_is_parsed(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        payload = json.dumps({"format": {"duration": "3601.25"}}).encode()
        calls = _record_run(monkeypatch, _Completed(returncode=0, stdout=payload))

        assert probe_duration_seconds(audio) == pytest.approx(3601.25)
        argv = calls[0]["argv"]
        assert argv[0] == "ffprobe"
        assert "format=duration" in argv
        assert str(audio) in argv

    def test_ffprobe_is_preferred_and_ffmpeg_is_not_invoked(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(
            ffmpeg_runtime,
            "is_ffmpeg_available",
            lambda: pytest.fail("ffmpeg fallback must not run"),
        )
        payload = json.dumps({"format": {"duration": "12.0"}}).encode()
        _record_run(monkeypatch, _Completed(returncode=0, stdout=payload))
        assert probe_duration_seconds(audio) == pytest.approx(12.0)

    def test_falls_back_to_ffmpeg_stderr_when_ffprobe_is_absent(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: False)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)
        stderr = b"  Duration: 01:02:03.50, start: 0.000000, bitrate: 128 kb/s\n"
        calls = _record_run(monkeypatch, _Completed(returncode=1, stderr=stderr))

        assert probe_duration_seconds(audio) == pytest.approx(3723.5)
        assert "-nostdin" in calls[0]["argv"]

    def test_falls_back_when_ffprobe_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)
        results = [
            _Completed(returncode=1),
            _Completed(returncode=0, stderr=b"Duration: 00:00:30.00, start: 0"),
        ]
        _record_run(monkeypatch, results)
        assert probe_duration_seconds(audio) == pytest.approx(30.0)

    def test_falls_back_when_ffprobe_emits_unparseable_json(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)
        results = [
            _Completed(returncode=0, stdout=b"not json"),
            _Completed(returncode=0, stderr=b"Duration: 00:01:00.00, start: 0"),
        ]
        _record_run(monkeypatch, results)
        assert probe_duration_seconds(audio) == pytest.approx(60.0)

    def test_zero_duration_is_treated_as_unknown(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: False)
        payload = json.dumps({"format": {"duration": "0"}}).encode()
        _record_run(monkeypatch, _Completed(returncode=0, stdout=payload))
        assert probe_duration_seconds(audio) is None

    def test_returns_none_when_neither_binary_is_available(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: False)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: False)
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        assert probe_duration_seconds(audio) is None
        assert calls == []

    def test_returns_none_when_both_probes_fail(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)
        results = [
            _Completed(returncode=1),
            _Completed(returncode=1, stderr=b"Invalid data found"),
        ]
        _record_run(monkeypatch, results)
        assert probe_duration_seconds(audio) is None

    def test_returns_none_when_the_probe_subprocess_raises(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: False)

        def _run(argv: Any, **kwargs: Any) -> Any:
            raise OSError("no such binary")

        monkeypatch.setattr(ffmpeg_runtime.subprocess, "run", _run)
        assert probe_duration_seconds(audio) is None

    def test_missing_duration_key_yields_none(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffprobe_available", lambda: True)
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: False)
        payload = json.dumps({"format": {}}).encode()
        _record_run(monkeypatch, _Completed(returncode=0, stdout=payload))
        assert probe_duration_seconds(audio) is None


# ---------------------------------------------------------------------------
# Encoder argument construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEncoderArgs:
    def test_copy_never_re_encodes(self) -> None:
        assert _encoder_args("copy", True, 16000) == ["-c:a", "copy"]

    def test_wav16_is_always_mono_pcm(self) -> None:
        assert _encoder_args("wav16", False, 16000) == [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ]

    def test_mp3_mono_downmixes(self) -> None:
        args = _encoder_args("mp3", True, 16000)
        assert args[:2] == ["-ac", "1"]
        assert "libmp3lame" in args

    def test_mp3_stereo_omits_the_channel_flag(self) -> None:
        args = _encoder_args("mp3", False, 44100)
        assert "-ac" not in args
        assert args == ["-ar", "44100", "-c:a", "libmp3lame", "-q:a", "4"]

    def test_unknown_format_warns_and_falls_back_to_mp3(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            args = _encoder_args("opus", True, 16000)
        assert "libmp3lame" in args
        assert "Unknown chunk_format" in caplog.text


# ---------------------------------------------------------------------------
# cut_segment
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCutSegment:
    @pytest.fixture
    def paths(self, tmp_path: Path) -> tuple[Path, Path]:
        src = tmp_path / "rec.wav"
        src.write_bytes(b"RIFF")
        return src, tmp_path / "out" / "rec_chunk_0001.mp3"

    @pytest.fixture(autouse=True)
    def _ffmpeg_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)

    def test_raises_when_ffmpeg_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: False)
        with pytest.raises(RuntimeError, match="FFmpeg is required"):
            cut_segment(
                src,
                dst,
                start=0.0,
                duration=None,
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )

    def test_argv_always_carries_nostdin_and_never_uses_a_shell(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=12.5,
            duration=30.0,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        argv = calls[0]["argv"]
        assert isinstance(argv, list)
        assert "-nostdin" in argv
        assert calls[0]["kwargs"].get("shell") in (None, False)

    def test_seek_precedes_the_input(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=600.0,
            duration=600.0,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        argv = calls[0]["argv"]
        assert argv.index("-ss") < argv.index("-i")
        assert argv[argv.index("-ss") + 1] == "600.000"
        assert argv[argv.index("-t") + 1] == "600.000"
        assert argv[argv.index("-i") + 1] == str(src)

    def test_open_ended_segment_omits_the_duration_flag(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        assert "-t" not in calls[0]["argv"]

    def test_video_streams_and_extra_audio_tracks_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        argv = calls[0]["argv"]
        assert "-vn" in argv
        assert argv[argv.index("-map") + 1] == "0:a:0"

    def test_mp3_mono_encoder_flags(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        argv = calls[0]["argv"]
        assert argv[-9:] == [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(dst),
        ]
        assert argv[argv.index("-ac") + 1] == "1"
        assert argv[argv.index("-ar") + 1] == "16000"
        assert argv[-1] == str(dst)

    def test_mp3_stereo_omits_the_channel_flag(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="mp3",
            mono=False,
            sample_rate=44100,
        )
        argv = calls[0]["argv"]
        assert "-ac" not in argv
        assert argv[argv.index("-ar") + 1] == "44100"

    def test_wav16_encoder_flags_and_wav_destination(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "rec.flac"
        src.write_bytes(b"fLaC")
        dst = tmp_path / "out" / "rec_chunk_0001.wav"
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="wav16",
            mono=False,
            sample_rate=16000,
        )
        argv = calls[0]["argv"]
        assert "pcm_s16le" in argv
        # wav16 is emitted mono regardless of the mono flag.
        assert argv[argv.index("-ac") + 1] == "1"
        assert argv[-1].endswith(".wav")

    def test_copy_format_carries_no_resampling_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "rec.m4a"
        src.write_bytes(b"ftyp")
        dst = tmp_path / "out" / "rec_chunk_0001.m4a"
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=5.0,
            duration=10.0,
            chunk_format="copy",
            mono=True,
            sample_rate=16000,
        )
        argv = calls[0]["argv"]
        assert argv[argv.index("-c:a") + 1] == "copy"
        assert "-ac" not in argv
        assert "-ar" not in argv

    def test_destination_parent_is_created(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        assert dst.parent.is_dir()

    def test_non_zero_exit_raises_with_the_stderr_tail(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        _record_run(
            monkeypatch,
            _Completed(returncode=1, stderr=b"Invalid argument: bogus codec"),
        )
        with pytest.raises(RuntimeError) as excinfo:
            cut_segment(
                src,
                dst,
                start=0.0,
                duration=None,
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        message = str(excinfo.value)
        assert "exit 1" in message
        assert "Invalid argument: bogus codec" in message
        assert src.name in message

    def test_stderr_tail_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        _record_run(monkeypatch, _Completed(returncode=2, stderr=b"E" * 5000 + b"TAIL"))
        with pytest.raises(RuntimeError) as excinfo:
            cut_segment(
                src,
                dst,
                start=0.0,
                duration=None,
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )
        assert "TAIL" in str(excinfo.value)
        assert len(str(excinfo.value)) < 1500

    def test_subprocess_failure_is_wrapped_in_a_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths

        def _run(argv: Any, **kwargs: Any) -> Any:
            raise OSError("exec format error")

        monkeypatch.setattr(ffmpeg_runtime.subprocess, "run", _run)
        with pytest.raises(RuntimeError, match="Could not run ffmpeg"):
            cut_segment(
                src,
                dst,
                start=0.0,
                duration=None,
                chunk_format="mp3",
                mono=True,
                sample_rate=16000,
            )

    def test_configured_executable_is_used_as_argv_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, paths: tuple[Path, Path]
    ) -> None:
        src, dst = paths
        exe = tmp_path / "custom-ffmpeg.exe"
        exe.write_text("x", encoding="utf-8")
        configure_ffmpeg_executables({"ffmpeg": {"ffmpeg_cmd": str(exe)}})
        monkeypatch.setattr(ffmpeg_runtime, "is_ffmpeg_available", lambda: True)
        calls = _record_run(monkeypatch, _Completed(returncode=0))
        cut_segment(
            src,
            dst,
            start=0.0,
            duration=None,
            chunk_format="mp3",
            mono=True,
            sample_rate=16000,
        )
        assert calls[0]["argv"][0] == str(exe)
