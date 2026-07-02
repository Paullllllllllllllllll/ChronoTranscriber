"""Tests for modules.transcribe.dual_mode."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from unittest.mock import MagicMock, patch

import pytest

from modules.transcribe.dual_mode import (
    AsyncDualModeScript,
    DualModeScript,
    _DualModeBase,
)

# ---------------------------------------------------------------------------
# Concrete subclasses for testing the abstract bases
# ---------------------------------------------------------------------------


class _SyncScript(DualModeScript):
    """Minimal concrete DualModeScript for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactive_called = False
        self.cli_called = False
        self.cli_args: Namespace | None = None

    def create_argument_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--name", default="test")
        return parser

    def run_interactive(self) -> None:
        self.interactive_called = True

    def run_cli(self, args: Namespace) -> None:
        self.cli_called = True
        self.cli_args = args


class _AsyncScript(AsyncDualModeScript):
    """Minimal concrete AsyncDualModeScript for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactive_called = False
        self.cli_called = False
        self.cli_args: Namespace | None = None

    def create_argument_parser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--name", default="test")
        return parser

    async def run_interactive(self) -> None:
        self.interactive_called = True

    async def run_cli(self, args: Namespace) -> None:
        self.cli_called = True
        self.cli_args = args


# ---------------------------------------------------------------------------
# _DualModeBase
# ---------------------------------------------------------------------------


class TestDualModeBaseInit:
    def test_init_sets_script_name(self) -> None:
        base = _DualModeBase("my_script")
        assert base.script_name == "my_script"

    def test_init_defaults(self) -> None:
        base = _DualModeBase("x")
        assert base.config_service is None
        assert base.is_interactive is False
        assert base.paths_config == {}
        assert base.model_config == {}
        assert base.concurrency_config == {}
        assert base.image_processing_config == {}


class TestDualModeBaseInitializeConfig:
    def test_initialize_config_loads_all(self, mock_config_service: MagicMock) -> None:
        base = _DualModeBase("test")
        with patch(
            "modules.transcribe.dual_mode.get_config_service",
            return_value=mock_config_service,
        ):
            base.initialize_config()
        assert base.config_service is mock_config_service
        assert base.paths_config == mock_config_service.get_paths_config()
        assert base.model_config == mock_config_service.get_model_config()
        assert base.concurrency_config == mock_config_service.get_concurrency_config()
        assert (
            base.image_processing_config
            == mock_config_service.get_image_processing_config()
        )


class TestDualModeBaseDetectMode:
    def test_detect_mode_interactive_true(self) -> None:
        base = _DualModeBase("test")
        base.paths_config = {"general": {"interactive_mode": True}}
        assert base._detect_mode() is True

    def test_detect_mode_interactive_false(self) -> None:
        base = _DualModeBase("test")
        base.paths_config = {"general": {"interactive_mode": False}}
        assert base._detect_mode() is False

    def test_detect_mode_missing_defaults_to_true(self) -> None:
        base = _DualModeBase("test")
        base.paths_config = {}
        assert base._detect_mode() is True


class TestDualModeBaseHandleInterrupt:
    def test_handle_interrupt_exits(self) -> None:
        base = _DualModeBase("test")
        with pytest.raises(SystemExit) as exc_info:
            base._handle_interrupt()
        # CLI agent contract: user interrupt exits 130.
        assert exc_info.value.code == 130


class TestDualModeBaseTtyGuard:
    def test_interactive_without_tty_exits_2(self) -> None:
        base = _DualModeBase("test")
        with (
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
            pytest.raises(SystemExit) as exc_info,
        ):
            m_stdin.isatty.return_value = False
            base._guard_interactive_tty()
        assert exc_info.value.code == 2

    def test_interactive_with_tty_passes(self) -> None:
        base = _DualModeBase("test")
        with patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin:
            m_stdin.isatty.return_value = True
            base._guard_interactive_tty()  # no raise


class TestDualModeBaseHandleError:
    def test_handle_error_exits_with_code_1(self) -> None:
        base = _DualModeBase("test")
        with pytest.raises(SystemExit) as exc_info:
            base._handle_error(RuntimeError("boom"))
        assert exc_info.value.code == 1


class TestDualModeBasePrintOrLog:
    @patch("modules.transcribe.dual_mode.print_info")
    @patch("modules.transcribe.dual_mode.print_error")
    def test_print_or_log_error(self, mock_print_error, mock_print_info) -> None:
        base = _DualModeBase("test")
        base.print_or_log("oops", level="error")
        # The method re-imports inside the function body, so we patch at module level
        # Just verify no crash; the actual print funcs are re-imported inside.

    def test_print_or_log_info_default(self) -> None:
        base = _DualModeBase("test")
        # Should not raise
        base.print_or_log("hello")

    def test_print_or_log_warning(self) -> None:
        base = _DualModeBase("test")
        base.print_or_log("careful", level="warning")

    def test_print_or_log_success(self) -> None:
        base = _DualModeBase("test")
        base.print_or_log("done", level="success")


# ---------------------------------------------------------------------------
# DualModeScript (sync)
# ---------------------------------------------------------------------------


class TestDualModeScriptExecuteInteractive:
    def test_execute_interactive_mode(self, mock_config_service: MagicMock) -> None:
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
        ):
            m_stdin.isatty.return_value = True
            script.execute()
        assert script.interactive_called is True
        assert script.cli_called is False

    def test_execute_cli_mode(self, mock_config_service: MagicMock) -> None:
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": False}
        }
        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(sys, "argv", ["script", "--name", "hello"]),
        ):
            script.execute()
        assert script.cli_called is True
        assert script.interactive_called is False
        assert script.cli_args.name == "hello"


class TestDualModeScriptExecuteErrorHandling:
    def test_execute_handles_keyboard_interrupt(
        self, mock_config_service: MagicMock
    ) -> None:
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
            patch.object(script, "run_interactive", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit) as exc_info,
        ):
            m_stdin.isatty.return_value = True
            script.execute()
        assert exc_info.value.code == 130

    def test_execute_handles_exception(self, mock_config_service: MagicMock) -> None:
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
            patch.object(script, "run_interactive", side_effect=ValueError("bad")),
            pytest.raises(SystemExit) as exc_info,
        ):
            m_stdin.isatty.return_value = True
            script.execute()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# AsyncDualModeScript
# ---------------------------------------------------------------------------


class TestAsyncDualModeScriptExecuteInteractive:
    def test_execute_interactive_mode(self, mock_config_service: MagicMock) -> None:
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
        ):
            m_stdin.isatty.return_value = True
            script.execute()
        assert script.interactive_called is True
        assert script.cli_called is False

    def test_execute_cli_mode(self, mock_config_service: MagicMock) -> None:
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": False}
        }
        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(sys, "argv", ["script", "--name", "world"]),
        ):
            script.execute()
        assert script.cli_called is True
        assert script.cli_args.name == "world"


class TestAsyncDualModeScriptExecuteErrorHandling:
    def test_execute_handles_keyboard_interrupt(
        self, mock_config_service: MagicMock
    ) -> None:
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }

        async def raise_interrupt():
            raise KeyboardInterrupt

        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
            patch.object(script, "run_interactive", side_effect=raise_interrupt),
            pytest.raises(SystemExit) as exc_info,
        ):
            m_stdin.isatty.return_value = True
            script.execute()
        assert exc_info.value.code == 130

    def test_execute_handles_exception(self, mock_config_service: MagicMock) -> None:
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }

        async def raise_error():
            raise RuntimeError("async boom")

        with (
            patch(
                "modules.transcribe.dual_mode.get_config_service",
                return_value=mock_config_service,
            ),
            patch("modules.transcribe.dual_mode.sys.stdin") as m_stdin,
            patch.object(script, "run_interactive", side_effect=raise_error),
            pytest.raises(SystemExit) as exc_info,
        ):
            m_stdin.isatty.return_value = True
            script.execute()
        assert exc_info.value.code == 1
