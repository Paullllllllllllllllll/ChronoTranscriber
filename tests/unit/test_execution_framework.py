"""Tests for modules.core.execution_framework."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from unittest.mock import MagicMock, patch, call

import pytest

from modules.core.execution_framework import (
    _DualModeBase,
    DualModeScript,
    AsyncDualModeScript,
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
    def test_init_sets_script_name(self):
        base = _DualModeBase("my_script")
        assert base.script_name == "my_script"

    def test_init_defaults(self):
        base = _DualModeBase("x")
        assert base.config_service is None
        assert base.is_interactive is False
        assert base.paths_config == {}
        assert base.model_config == {}
        assert base.concurrency_config == {}
        assert base.image_processing_config == {}


class TestDualModeBaseInitializeConfig:
    def test_initialize_config_loads_all(self, mock_config_service):
        base = _DualModeBase("test")
        with patch(
            "modules.core.execution_framework.get_config_service",
            return_value=mock_config_service,
        ):
            base.initialize_config()
        assert base.config_service is mock_config_service
        assert base.paths_config == mock_config_service.get_paths_config()
        assert base.model_config == mock_config_service.get_model_config()
        assert base.concurrency_config == mock_config_service.get_concurrency_config()
        assert base.image_processing_config == mock_config_service.get_image_processing_config()


class TestDualModeBaseDetectMode:
    def test_detect_mode_interactive_true(self):
        base = _DualModeBase("test")
        base.paths_config = {"general": {"interactive_mode": True}}
        assert base._detect_mode() is True

    def test_detect_mode_interactive_false(self):
        base = _DualModeBase("test")
        base.paths_config = {"general": {"interactive_mode": False}}
        assert base._detect_mode() is False

    def test_detect_mode_missing_defaults_to_true(self):
        base = _DualModeBase("test")
        base.paths_config = {}
        assert base._detect_mode() is True


class TestDualModeBaseHandleInterrupt:
    def test_handle_interrupt_exits(self):
        base = _DualModeBase("test")
        with pytest.raises(SystemExit) as exc_info:
            base._handle_interrupt()
        assert exc_info.value.code == 0


class TestDualModeBaseHandleError:
    def test_handle_error_exits_with_code_1(self):
        base = _DualModeBase("test")
        with pytest.raises(SystemExit) as exc_info:
            base._handle_error(RuntimeError("boom"))
        assert exc_info.value.code == 1


class TestDualModeBasePrintOrLog:
    @patch("modules.core.execution_framework.print_info")
    @patch("modules.core.execution_framework.print_error")
    def test_print_or_log_error(self, mock_print_error, mock_print_info):
        base = _DualModeBase("test")
        base.print_or_log("oops", level="error")
        # The method re-imports inside the function body, so we patch at module level
        # Just verify no crash; the actual print funcs are re-imported inside.

    def test_print_or_log_info_default(self):
        base = _DualModeBase("test")
        # Should not raise
        base.print_or_log("hello")

    def test_print_or_log_warning(self):
        base = _DualModeBase("test")
        base.print_or_log("careful", level="warning")

    def test_print_or_log_success(self):
        base = _DualModeBase("test")
        base.print_or_log("done", level="success")


# ---------------------------------------------------------------------------
# DualModeScript (sync)
# ---------------------------------------------------------------------------

class TestDualModeScriptExecuteInteractive:
    def test_execute_interactive_mode(self, mock_config_service):
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with patch(
            "modules.core.execution_framework.get_config_service",
            return_value=mock_config_service,
        ):
            script.execute()
        assert script.interactive_called is True
        assert script.cli_called is False

    def test_execute_cli_mode(self, mock_config_service):
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": False}
        }
        with (
            patch(
                "modules.core.execution_framework.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(sys, "argv", ["script", "--name", "hello"]),
        ):
            script.execute()
        assert script.cli_called is True
        assert script.interactive_called is False
        assert script.cli_args.name == "hello"


class TestDualModeScriptExecuteErrorHandling:
    def test_execute_handles_keyboard_interrupt(self, mock_config_service):
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with (
            patch(
                "modules.core.execution_framework.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(
                script, "run_interactive", side_effect=KeyboardInterrupt
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            script.execute()
        assert exc_info.value.code == 0

    def test_execute_handles_exception(self, mock_config_service):
        script = _SyncScript("test_sync")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with (
            patch(
                "modules.core.execution_framework.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(
                script, "run_interactive", side_effect=ValueError("bad")
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            script.execute()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# AsyncDualModeScript
# ---------------------------------------------------------------------------

class TestAsyncDualModeScriptExecuteInteractive:
    def test_execute_interactive_mode(self, mock_config_service):
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }
        with patch(
            "modules.core.execution_framework.get_config_service",
            return_value=mock_config_service,
        ):
            script.execute()
        assert script.interactive_called is True
        assert script.cli_called is False

    def test_execute_cli_mode(self, mock_config_service):
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": False}
        }
        with (
            patch(
                "modules.core.execution_framework.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(sys, "argv", ["script", "--name", "world"]),
        ):
            script.execute()
        assert script.cli_called is True
        assert script.cli_args.name == "world"


class TestAsyncDualModeScriptExecuteErrorHandling:
    def test_execute_handles_keyboard_interrupt(self, mock_config_service):
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }

        async def raise_interrupt():
            raise KeyboardInterrupt

        with (
            patch(
                "modules.core.execution_framework.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(script, "run_interactive", side_effect=raise_interrupt),
            pytest.raises(SystemExit) as exc_info,
        ):
            script.execute()
        assert exc_info.value.code == 0

    def test_execute_handles_exception(self, mock_config_service):
        script = _AsyncScript("test_async")
        mock_config_service.get_paths_config.return_value = {
            "general": {"interactive_mode": True}
        }

        async def raise_error():
            raise RuntimeError("async boom")

        with (
            patch(
                "modules.core.execution_framework.get_config_service",
                return_value=mock_config_service,
            ),
            patch.object(script, "run_interactive", side_effect=raise_error),
            pytest.raises(SystemExit) as exc_info,
        ):
            script.execute()
        assert exc_info.value.code == 1
