"""Pytest fixtures and configuration for ChronoTranscriber tests.

The session is hermetic. ``pytest_configure`` strips provider keys from the
environment and points the home directories at a temp dir. The autouse
``_hermetic`` fixture then redirects every per-user path into ``tmp_path``
(home, the default settings file and the logs folder), so a real
``config/settings.yaml`` is never read; it blocks non-loopback connections,
launches of the tool itself and of clipboard programs, fails any test that
writes outside the pytest base temp directory, and removes the log handlers
a test installed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import wave
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chronotranscriber.common import testing
from chronotranscriber.common.testing import (
    XDG_VARS,
    LaunchEntry,
    WriteGuard,
    home_env,
    is_loopback_address,
    is_secret_env_var,
)
from chronotranscriber.settings import Settings
from chronotranscriber.spec import RunSpec, resolve_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAUNCH_ENTRY = LaunchEntry(
    programs=("chronotranscriber",),
    scripts=("chronotranscriber/__main__.py",),
    modules=("chronotranscriber",),
)

_SESSION_ENV_BACKUP: dict[str, str | None] = {}

__all__ = [
    "CLIPBOARD_PROGRAMS",
    "PROJECT_ROOT",
    "ClipboardLaunchRefusedError",
    "NetworkBlockedError",
    "StubAudioBackend",
    "ToolLaunchRefusedError",
    "WriteGuard",
    "home_env",
    "is_loopback_address",
    "is_secret_env_var",
    "launches_clipboard",
    "launches_tool",
    "make_job",
    "make_settings",
    "make_spec",
]


class NetworkBlockedError(RuntimeError):
    """Raised when a test connects to a non-loopback address."""


class ToolLaunchRefusedError(RuntimeError):
    """Raised when a test launches a ChronoTranscriber entry point."""


class ClipboardLaunchRefusedError(RuntimeError):
    """Raised when a test starts a clipboard program."""


CLIPBOARD_PROGRAMS = frozenset({"clip", "pbcopy", "xclip", "xsel", "wl-copy"})
"""Programs that would overwrite the user's clipboard."""


# =============================================================================
# Settings and spec builders
# =============================================================================


def make_spec(input_path: Path | str = "doc.pdf", **flags: Any) -> RunSpec:
    """Return a transcribe spec from option values keyed by option name.

    Without ``files`` or ``select`` the spec takes every item (``--all``), as
    a folder with several items needs a selection.
    """
    if not {"files", "select", "all"} & set(flags):
        flags["all"] = True
    return resolve_spec({"input": Path(input_path), **flags}).spec


def make_settings(**changes: Any) -> Settings:
    """Return the code-default settings with *changes* applied."""
    return dataclasses.replace(Settings(), **changes)


def make_job(
    spec: RunSpec,
    *,
    output_dir: Path | None,
    input_root: Path,
    settings: Settings | None = None,
    events: Any = None,
    **changes: Any,
) -> Any:
    """Return the method job of *spec*; *changes* set further job fields."""
    from chronotranscriber.events import NullEvents
    from chronotranscriber.llm.caller import CallEnv
    from chronotranscriber.methods import Job

    values = settings if settings is not None else Settings()
    return Job(
        spec=spec,
        settings=values,
        events=events if events is not None else NullEvents(),
        env=CallEnv.create(values),
        output_dir=output_dir,
        input_root=input_root,
        **changes,
    )


# =============================================================================
# Session setup
# =============================================================================


def _session_home(config: pytest.Config) -> Path:
    factory: pytest.TempPathFactory | None = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        return factory.mktemp("session-home")
    return Path(tempfile.mkdtemp(prefix="chronotranscriber-tests-"))


def _set_session_env(name: str, value: str | None) -> None:
    _SESSION_ENV_BACKUP.setdefault(name, os.environ.get(name))
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Drop the key and heartbeat variables and redirect the home dirs."""
    for name in [name for name in os.environ if is_secret_env_var(name)]:
        _set_session_env(name, None)
    _set_session_env("CHRONO_HEARTBEAT_FILE", None)
    session_home = _session_home(config)
    env = home_env(session_home)
    xdg_names = {*XDG_VARS, *(n for n in os.environ if n.startswith("XDG_"))}
    env.update(dict.fromkeys(xdg_names, str(session_home)))
    for name, value in env.items():
        _set_session_env(name, value)
    _install_write_guard()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the environment changed in ``pytest_configure``."""
    for name, value in _SESSION_ENV_BACKUP.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _SESSION_ENV_BACKUP.clear()


# =============================================================================
# Write guard
# =============================================================================

_WRITE_GUARD = WriteGuard()
_WRITE_GUARD_INSTALLED = False


def _install_write_guard() -> None:
    global _WRITE_GUARD_INSTALLED
    if not _WRITE_GUARD_INSTALLED:
        sys.addaudithook(_WRITE_GUARD.audit)
        _WRITE_GUARD_INSTALLED = True


# =============================================================================
# Network and launch guards
# =============================================================================


def launches_tool(args: Any, executable: Any = None) -> bool:
    """Return True when a Popen argv would start a ChronoTranscriber entry point."""
    return testing.launches_entry(args, executable, LAUNCH_ENTRY)


def _program_name(program: Any) -> str:
    """Return the lower-case file name of *program* without ``.exe``."""
    name = Path(os.fsdecode(program)).name.lower()
    return name.removesuffix(".exe")


def launches_clipboard(args: Any, executable: Any = None) -> bool:
    """Return True when a Popen argv would start a clipboard program."""
    if isinstance(args, str | bytes | os.PathLike):
        words = os.fsdecode(args).split()
        first: Any = words[0].strip("\"'") if words else ""
    else:
        first = next(iter(args), "")
    programs = [first] if executable is None else [first, executable]
    return any(_program_name(program) in CLIPBOARD_PROGRAMS for program in programs)


def _guard_network_and_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_popen_init = subprocess.Popen.__init__

    def getaddrinfo(host: Any, port: Any, *a: Any, **kw: Any) -> Any:
        # asyncio's Windows proactor loop connects through ConnectEx, which
        # bypasses socket.connect; every connection still resolves its host here.
        name = host.decode() if isinstance(host, bytes) else host
        if name not in (None, "localhost") and not is_loopback_address((name, 0)):
            raise NetworkBlockedError(f"Network access blocked in tests: {host}")
        return real_getaddrinfo(host, port, *a, **kw)

    def connect(sock: socket.socket, address: Any) -> None:
        if not is_loopback_address(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if not is_loopback_address(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        return real_connect_ex(sock, address)

    def popen_init(self: subprocess.Popen[Any], args: Any, *a: Any, **kw: Any) -> None:
        executable = kw.get("executable", a[1] if len(a) > 1 else None)
        if launches_tool(args, executable):
            raise ToolLaunchRefusedError(
                f"Tests must not launch ChronoTranscriber as a subprocess: {args!r}"
            )
        if launches_clipboard(args, executable):
            raise ClipboardLaunchRefusedError(
                f"Tests must not start a clipboard program: {args!r}"
            )
        real_popen_init(self, args, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(subprocess.Popen, "__init__", popen_init)


# =============================================================================
# Logger state
# =============================================================================

_LOGGER_NAMES = ("chronotranscriber", "")


def _logger_state() -> dict[str, tuple[list[logging.Handler], int]]:
    state = {}
    for name in _LOGGER_NAMES:
        logger = logging.getLogger(name)
        state[name] = (list(logger.handlers), logger.level)
    return state


def _restore_loggers(state: dict[str, tuple[list[logging.Handler], int]]) -> None:
    """Close and drop the handlers a test added; reset the levels.

    Pytest's own capture handlers are swapped per test phase and left alone.
    An open log file would block the removal of ``tmp_path`` on Windows.
    """
    for name, (handlers, level) in state.items():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if handler in handlers or type(handler).__module__.startswith("_pytest"):
                continue
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(level)


# =============================================================================
# Per-test isolation (autouse)
# =============================================================================


@pytest.fixture(autouse=True)
def _hermetic(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[WriteGuard]:
    """Redirect per-user paths, settings and logs into ``tmp_path``."""
    import chronotranscriber.settings as settings_module

    home = tmp_path / "home"
    temp = tmp_path / "tmp"
    for directory in (home, temp):
        directory.mkdir()
    for name, value in home_env(home).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    monkeypatch.setattr(
        settings_module,
        "DEFAULT_SETTINGS_PATH",
        tmp_path / "config" / "settings.yaml",
    )
    monkeypatch.setattr(settings_module, "DEFAULT_LOGS_DIR", tmp_path / "logs")

    _guard_network_and_launches(monkeypatch)

    logger_state = _logger_state()

    _WRITE_GUARD.start([tmp_path_factory.getbasetemp()])
    try:
        yield _WRITE_GUARD
    finally:
        leaked = _WRITE_GUARD.stop()
        _restore_loggers(logger_state)
    if leaked:
        listing = "\n  ".join(leaked)
        pytest.fail(
            f"Test wrote outside the pytest temp directory:\n  {listing}",
            pytrace=False,
        )


@pytest.fixture
def write_guard(_hermetic: WriteGuard) -> WriteGuard:
    """Return the write guard so a test can consume expected violations."""
    return _hermetic


# =============================================================================
# Audio Fixtures
# =============================================================================


@pytest.fixture
def sample_audio_file(tmp_path: Path) -> Path:
    """Create a small, real WAV file (0.5 s of silence, 16 kHz mono 16-bit)."""
    audio_path = tmp_path / "sample_recording.wav"
    frames = 8000  # 0.5 s at 16 kHz
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)
    return audio_path


class StubAudioBackend:
    """Minimal :class:`~chronotranscriber.methods.audio_api.AudioBackend` stub.

    Satisfies the backend protocol with canned responses: each call pops the
    next entry from ``responses`` (an ``Exception`` instance is raised instead
    of returned) and falls back to ``default_response`` once the queue is
    empty. Every payload it saw is recorded on ``calls``.
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        provider_name: str = "stub",
        model: str = "stub-audio-model",
        default_response: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.responses: list[Any] = list(responses or [])
        self.default_response: dict[str, Any] = default_response or {
            "output_text": "stub transcript",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "metadata": {"provider": provider_name, "model": model},
        }
        self.calls: list[Any] = []
        self.closed = False

    async def transcribe_audio_chunk(self, payload: Any) -> dict[str, Any]:
        """Return (or raise) the next canned response for *payload*."""
        self.calls.append(payload)
        item = self.responses.pop(0) if self.responses else self.default_response
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        """Record that the backend was disposed."""
        self.closed = True


@pytest.fixture
def mock_audio_backend() -> StubAudioBackend:
    """Provide a configurable :class:`StubAudioBackend` instance."""
    return StubAudioBackend()


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path]:
    """Return a directory under ``tmp_path`` for test outputs."""
    out_dir = tmp_path / "temp_dir"
    out_dir.mkdir()
    yield out_dir


@pytest.fixture
def temp_input_dir(temp_dir: Path) -> Path:
    """Create a temporary input directory structure."""
    input_dir = temp_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory structure."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_image_folder(temp_input_dir: Path) -> Path:
    """Create a sample image folder with placeholder images."""
    img_folder = temp_input_dir / "test_images"
    img_folder.mkdir(parents=True, exist_ok=True)

    # Placeholder image files: empty files with image extensions.
    for i in range(3):
        (img_folder / f"page_{i + 1:03d}.png").write_bytes(b"")

    return img_folder


@pytest.fixture
def sample_pdf_file(temp_input_dir: Path) -> Path:
    """Create a placeholder PDF file for testing."""
    pdf_path = temp_input_dir / "test_document.pdf"
    # A PDF header only: not a valid PDF, but enough for path tests.
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return pdf_path


@pytest.fixture
def sample_text_file(temp_input_dir: Path) -> Path:
    """Create a sample text file for testing."""
    txt_path = temp_input_dir / "sample_transcription.txt"
    txt_path.write_text(
        "Page 1:\nThis is sample transcription text.\n\n"
        "Page 2:\n[transcription error: page_002.png]\n\n"
        "Page 3:\n[No transcribable text]\n",
        encoding="utf-8",
    )
    return txt_path


# =============================================================================
# Mock API Client Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_response() -> dict[str, Any]:
    """Provide a mock OpenAI API response."""
    return {
        "id": "resp_test123",
        "object": "response",
        "created": 1234567890,
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "transcription": "This is the transcribed text.",
                                "no_transcribable_text": False,
                                "transcription_not_possible": False,
                            }
                        ),
                    }
                ],
            }
        ],
        "output_text": json.dumps(
            {
                "transcription": "This is the transcribed text.",
                "no_transcribable_text": False,
                "transcription_not_possible": False,
            }
        ),
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    }


@pytest.fixture
def mock_openai_client(mock_openai_response: dict[str, Any]) -> MagicMock:
    """Provide a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_openai_response
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "transcription": "Test transcription",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        }
                    )
                )
            )
        ]
    )
    return mock_client


@pytest.fixture
def mock_langchain_llm() -> MagicMock:
    """Provide a mock LangChain LLM."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {
            "transcription": "LangChain transcribed text",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }
    )
    mock_response.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    mock_llm.invoke = MagicMock(return_value=mock_response)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def mock_env_no_api_keys() -> Generator[None]:
    """Mock environment with no API keys set."""
    with patch.dict(os.environ, {}, clear=True):
        env_copy = os.environ.copy()
        for key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            env_copy.pop(key, None)
        with patch.dict(os.environ, env_copy, clear=True):
            yield


@pytest.fixture
def mock_env_with_openai_key() -> Generator[None]:
    """Mock environment with OpenAI API key set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}):
        yield


# =============================================================================
# Schema Fixtures
# =============================================================================


@pytest.fixture
def sample_transcription_schema() -> dict[str, Any]:
    """Provide a sample transcription schema."""
    return {
        "type": "object",
        "properties": {
            "transcription": {
                "type": "string",
                "description": "The transcribed text from the image",
            },
            "no_transcribable_text": {
                "type": "boolean",
                "description": "True if no text could be found",
            },
            "transcription_not_possible": {
                "type": "boolean",
                "description": "True if transcription was not possible",
            },
        },
        "required": [
            "transcription",
            "no_transcribable_text",
            "transcription_not_possible",
        ],
        "additionalProperties": False,
    }


# =============================================================================
# Async Test Helpers
# =============================================================================


@pytest.fixture
def event_loop() -> Generator[Any]:
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
