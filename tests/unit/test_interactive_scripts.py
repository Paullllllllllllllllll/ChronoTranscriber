"""Unit tests for interactive-mode paths of the batch entry-point scripts.

Covers two verified fixes:

- ``cancel_batches.CancelBatchesScript.run_interactive`` must confirm before
  cancelling and only target non-terminal batches (previously it cancelled all
  non-terminal batches with no confirmation, unlike the CLI path).
- ``check_batches.CheckBatchesScript.run_interactive`` must honor the configured
  ``output_format`` and surface the returned stats (previously it ignored both).
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import main.cancel_batches as cancel_mod
import main.check_batches as check_mod
from modules.batch.check import BatchCheckStats


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a stub ``openai`` module so ``from openai import OpenAI`` works."""
    fake = ModuleType("openai")
    fake.OpenAI = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)


def _patch_cancel_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("print_info", "print_header", "print_error", "print_success"):
        monkeypatch.setattr(cancel_mod, name, MagicMock())
    monkeypatch.setattr(cancel_mod, "display_batch_summary", MagicMock())
    monkeypatch.setattr(cancel_mod, "display_batch_cancellation_results", MagicMock())


class TestCancelBatchesInteractive:
    @pytest.mark.unit
    def test_declining_confirmation_cancels_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_openai(monkeypatch)
        _patch_cancel_ui(monkeypatch)

        batches = [
            {"id": "batch_1", "status": "in_progress"},
            {"id": "batch_2", "status": "completed"},
        ]
        monkeypatch.setattr(
            cancel_mod, "list_all_batches", MagicMock(return_value=batches)
        )
        monkeypatch.setattr(cancel_mod, "confirm_action", MagicMock(return_value=False))
        cancel = MagicMock(return_value=True)
        monkeypatch.setattr(cancel_mod, "cancel_batch_by_id", cancel)

        cancel_mod.CancelBatchesScript().run_interactive()

        cancel.assert_not_called()

    @pytest.mark.unit
    def test_confirmation_cancels_only_non_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_openai(monkeypatch)
        _patch_cancel_ui(monkeypatch)

        batches = [
            {"id": "batch_1", "status": "in_progress"},
            {"id": "batch_2", "status": "completed"},
            {"id": "batch_3", "status": "validating"},
        ]
        monkeypatch.setattr(
            cancel_mod, "list_all_batches", MagicMock(return_value=batches)
        )
        confirm = MagicMock(return_value=True)
        monkeypatch.setattr(cancel_mod, "confirm_action", confirm)
        cancel = MagicMock(return_value=True)
        monkeypatch.setattr(cancel_mod, "cancel_batch_by_id", cancel)

        cancel_mod.CancelBatchesScript().run_interactive()

        # Confirmation asked for the two non-terminal batches only.
        confirm.assert_called_once()
        cancelled_ids = {call.args[1] for call in cancel.call_args_list}
        assert cancelled_ids == {"batch_1", "batch_3"}

    @pytest.mark.unit
    def test_no_non_terminal_batches_skips_confirmation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_openai(monkeypatch)
        _patch_cancel_ui(monkeypatch)

        batches = [{"id": "batch_2", "status": "completed"}]
        monkeypatch.setattr(
            cancel_mod, "list_all_batches", MagicMock(return_value=batches)
        )
        confirm = MagicMock(return_value=True)
        monkeypatch.setattr(cancel_mod, "confirm_action", confirm)
        cancel = MagicMock(return_value=True)
        monkeypatch.setattr(cancel_mod, "cancel_batch_by_id", cancel)

        cancel_mod.CancelBatchesScript().run_interactive()

        confirm.assert_not_called()
        cancel.assert_not_called()


class TestCheckBatchesInteractive:
    @pytest.mark.unit
    def test_passes_configured_output_format_and_prints_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stats = BatchCheckStats(finalized=2, pending=1, failed=0)
        finalize = MagicMock(return_value=stats)
        monkeypatch.setattr(check_mod, "run_batch_finalization", finalize)
        info = MagicMock()
        monkeypatch.setattr(check_mod, "print_info", info)

        script = check_mod.CheckBatchesScript()
        script.paths_config = {"general": {"output_format": "md"}}
        script.run_interactive()

        finalize.assert_called_once_with(run_diagnostics=True, output_format="md")
        summary = " ".join(str(call.args[0]) for call in info.call_args_list)
        assert "2 finalized" in summary
        assert "1 pending" in summary
        assert "0 failed" in summary

    @pytest.mark.unit
    def test_defaults_output_format_to_txt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        finalize = MagicMock(return_value=BatchCheckStats())
        monkeypatch.setattr(check_mod, "run_batch_finalization", finalize)
        monkeypatch.setattr(check_mod, "print_info", MagicMock())

        script = check_mod.CheckBatchesScript()
        script.paths_config = {}
        script.run_interactive()

        finalize.assert_called_once_with(run_diagnostics=True, output_format="txt")
