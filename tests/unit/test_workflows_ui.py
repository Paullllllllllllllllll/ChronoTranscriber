"""Unit tests for modules/ui/workflows.py.

Tests WorkflowUI display and configuration helpers.
Includes CT-4 regression tests verifying the removal of the incorrect
image_processing key access from display_processing_summary().
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.transcribe.user_config import UserConfiguration
from modules.ui.prompts import NavigationAction, PromptResult
from modules.ui.workflows import WorkflowUI


def _continue(value: Any) -> PromptResult:
    """Build a CONTINUE PromptResult carrying *value*."""
    return PromptResult(action=NavigationAction.CONTINUE, value=value)


def _back() -> PromptResult:
    """Build a BACK PromptResult."""
    return PromptResult(action=NavigationAction.BACK)


class TestDisplayProcessingSummaryConcurrencyConfig:
    """CT-4 regression: concurrency config key access in display_processing_summary.

    Before the fix, display_processing_summary() read
    concurrency_config.get("image_processing", {}), a key that does not exist
    in concurrency_config.yaml.  The row always showed the hardcoded default 24.
    The fix removes that row entirely.
    """

    @pytest.mark.unit
    def test_image_processing_key_not_accessed_in_source(self) -> None:
        """Helper must not read 'image_processing' from concurrency_config."""
        from modules.ui import workflows as wf_module

        source = inspect.getsource(wf_module.WorkflowUI._build_concurrency_config_lines)
        assert 'concurrency_config.get("image_processing"' not in source, (
            "_build_concurrency_config_lines still contains the non-existent "
            "'image_processing' key access on concurrency_config"
        )

    @pytest.mark.unit
    def test_concurrency_path_reads_from_correct_key(self) -> None:
        """Concurrency helper reads API concurrency from concurrency.transcription.*."""
        from modules.ui import workflows as wf_module

        source = inspect.getsource(wf_module.WorkflowUI._build_concurrency_config_lines)
        assert (
            'concurrency_config.get("concurrency", {}).get("transcription"' in source
        ), (
            "_build_concurrency_config_lines is not reading from the correct "
            "concurrency.transcription path"
        )

    @pytest.mark.unit
    def test_correct_concurrency_config_path_resolution(self) -> None:
        """The concurrency.transcription.* path resolves the correct config values."""
        concurrency_cfg: dict[str, Any] = {
            "concurrency": {
                "transcription": {
                    "concurrency_limit": 1500,
                    "service_tier": "flex",
                    "retry": {"attempts": 10},
                }
            },
            "daily_token_limit": {"enabled": True, "daily_tokens": 25_000_000},
        }

        trans_cfg = concurrency_cfg.get("concurrency", {}).get("transcription", {})
        assert trans_cfg.get("concurrency_limit", 5) == 1500
        assert trans_cfg.get("service_tier", "default") == "flex"
        assert trans_cfg.get("retry", {}).get("attempts", 5) == 10

    @pytest.mark.unit
    def test_empty_concurrency_config_returns_defaults(self) -> None:
        """When concurrency_config is empty, defaults are applied correctly."""
        concurrency_cfg: dict[str, Any] = {}

        trans_cfg = concurrency_cfg.get("concurrency", {}).get("transcription", {})
        assert trans_cfg.get("concurrency_limit", 5) == 5
        assert trans_cfg.get("service_tier", "default") == "default"
        assert trans_cfg.get("retry", {}).get("attempts", 5) == 5


class TestWorkflowUIOptions:
    """Tests for WorkflowUI static option helpers."""

    @pytest.mark.unit
    def test_get_processing_type_options_returns_list(self) -> None:
        """get_processing_type_options returns a non-empty list of (value, label)."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_processing_type_options()
        assert isinstance(options, list)
        assert len(options) > 0
        for value, label in options:
            assert isinstance(value, str)
            assert isinstance(label, str)

    @pytest.mark.unit
    def test_processing_type_includes_auto(self) -> None:
        """Auto mode is present in processing type options."""
        from modules.ui.workflows import WorkflowUI

        values = [v for v, _ in WorkflowUI.get_processing_type_options()]
        assert "auto" in values

    @pytest.mark.unit
    def test_processing_type_includes_pdfs_and_images(self) -> None:
        """PDFs and images are present in processing type options."""
        from modules.ui.workflows import WorkflowUI

        values = [v for v, _ in WorkflowUI.get_processing_type_options()]
        assert "pdfs" in values
        assert "images" in values

    @pytest.mark.unit
    def test_get_method_options_for_pdfs(self) -> None:
        """PDF processing type offers native, tesseract, and gpt options."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_method_options("pdfs")
        values = [v for v, _ in options]
        assert "native" in values
        assert "tesseract" in values
        assert "gpt" in values

    @pytest.mark.unit
    def test_get_method_options_for_images(self) -> None:
        """Image processing type offers tesseract and gpt options."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_method_options("images")
        values = [v for v, _ in options]
        assert "tesseract" in values
        assert "gpt" in values

    @pytest.mark.unit
    def test_get_batch_options_returns_yes_no(self) -> None:
        """Batch options include yes and no."""
        from modules.ui.workflows import WorkflowUI

        options = WorkflowUI.get_batch_options()
        values = [v for v, _ in options]
        assert "yes" in values
        assert "no" in values

    @pytest.mark.unit
    def test_processing_type_includes_mobis(self) -> None:
        """MOBI processing type is offered (wizard parity with CLI --type mobis)."""
        values = [v for v, _ in WorkflowUI.get_processing_type_options()]
        assert "mobis" in values

    @pytest.mark.unit
    def test_get_method_options_for_mobis_is_native_only(self) -> None:
        """MOBI processing offers only the native extraction method."""
        options = WorkflowUI.get_method_options("mobis")
        values = [v for v, _ in options]
        assert values == ["native"]

    @pytest.mark.unit
    def test_get_method_options_for_epubs_is_native_only(self) -> None:
        """EPUB processing offers only the native extraction method."""
        options = WorkflowUI.get_method_options("epubs")
        values = [v for v, _ in options]
        assert values == ["native"]


class TestConfigureResumeMode:
    """Tests for the resume/overwrite/retry-errors prompt (D2)."""

    @pytest.mark.unit
    def test_retry_errors_option_present_in_choices(self) -> None:
        """The prompt offers a third 'retry_errors' option."""
        config = UserConfiguration()
        with patch(
            "modules.ui.workflows.prompt_select", return_value=_continue("skip")
        ) as mock_select:
            WorkflowUI.configure_resume_mode(config)
        choices = mock_select.call_args.args[1]
        assert "retry_errors" in [value for value, _ in choices]

    @pytest.mark.unit
    def test_retry_errors_maps_to_skip_plus_flag(self) -> None:
        """Selecting retry_errors sets resume_mode=skip and retry_errors=True."""
        config = UserConfiguration()
        with patch(
            "modules.ui.workflows.prompt_select",
            return_value=_continue("retry_errors"),
        ):
            ok = WorkflowUI.configure_resume_mode(config)
        assert ok is True
        assert config.resume_mode == "skip"
        assert config.retry_errors is True

    @pytest.mark.unit
    def test_overwrite_clears_retry_errors(self) -> None:
        """Overwrite clears any previously-set retry_errors flag."""
        config = UserConfiguration(retry_errors=True)
        with patch(
            "modules.ui.workflows.prompt_select", return_value=_continue("overwrite")
        ):
            ok = WorkflowUI.configure_resume_mode(config)
        assert ok is True
        assert config.resume_mode == "overwrite"
        assert config.retry_errors is False

    @pytest.mark.unit
    def test_skip_clears_retry_errors(self) -> None:
        """Skip clears any previously-set retry_errors flag."""
        config = UserConfiguration(retry_errors=True)
        with patch(
            "modules.ui.workflows.prompt_select", return_value=_continue("skip")
        ):
            WorkflowUI.configure_resume_mode(config)
        assert config.resume_mode == "skip"
        assert config.retry_errors is False

    @pytest.mark.unit
    def test_back_returns_false(self) -> None:
        """A BACK action returns False without mutating resume_mode."""
        config = UserConfiguration()
        with patch("modules.ui.workflows.prompt_select", return_value=_back()):
            ok = WorkflowUI.configure_resume_mode(config)
        assert ok is False


class TestConfigureBatchProcessingMissingKey:
    """Missing API key steps back instead of raising mid-wizard (fix F)."""

    @pytest.mark.unit
    def test_missing_api_key_returns_false(self) -> None:
        """When no API key resolves, the wizard prints an error and returns False."""
        config = UserConfiguration(transcription_method="gpt")
        fake_service = MagicMock()
        fake_service.get_model_config.return_value = {
            "transcription_model": {"provider": "openai", "name": "gpt-4o"}
        }
        with (
            patch(
                "modules.config.service.get_config_service",
                return_value=fake_service,
            ),
            patch(
                "modules.llm.providers.factory.resolve_api_key_optional",
                return_value=None,
            ),
            patch("modules.ui.workflows.print_error") as mock_error,
        ):
            ok = WorkflowUI.configure_batch_processing(config)
        assert ok is False
        mock_error.assert_called_once()


class TestConfigureBatchProcessingUnsupportedProvider:
    """Item 8a: providers without a batch API are not offered batch processing;
    the wizard forces synchronous mode rather than silently falling back to
    full-price sync at submission time."""

    @staticmethod
    def _service(provider: str, name: str) -> MagicMock:
        svc = MagicMock()
        svc.get_model_config.return_value = {
            "transcription_model": {"provider": provider, "name": name}
        }
        return svc

    @pytest.mark.unit
    def test_unsupported_provider_skips_batch_prompt(self) -> None:
        config = UserConfiguration(transcription_method="gpt")
        with (
            patch(
                "modules.config.service.get_config_service",
                return_value=self._service("openrouter", "some-model"),
            ),
            patch(
                "modules.llm.providers.factory.resolve_api_key_optional",
                return_value="key",
            ),
            patch("modules.ui.workflows.prompt_select") as mock_select,
            patch.object(WorkflowUI, "configure_schema_selection", return_value=True),
            patch.object(WorkflowUI, "configure_additional_context", return_value=True),
            patch.object(
                WorkflowUI, "configure_additional_context_image", return_value=True
            ),
        ):
            ok = WorkflowUI.configure_batch_processing(config)
        assert ok is True
        assert config.use_batch_processing is False
        mock_select.assert_not_called()

    @pytest.mark.unit
    def test_supported_provider_still_prompts(self) -> None:
        config = UserConfiguration(transcription_method="gpt")
        with (
            patch(
                "modules.config.service.get_config_service",
                return_value=self._service("openai", "gpt-4o"),
            ),
            patch(
                "modules.llm.providers.factory.resolve_api_key_optional",
                return_value="key",
            ),
            patch(
                "modules.ui.workflows.prompt_select",
                return_value=_continue("yes"),
            ) as mock_select,
            patch.object(WorkflowUI, "configure_schema_selection", return_value=True),
            patch.object(WorkflowUI, "configure_additional_context", return_value=True),
            patch.object(
                WorkflowUI, "configure_additional_context_image", return_value=True
            ),
        ):
            ok = WorkflowUI.configure_batch_processing(config)
        assert ok is True
        assert config.use_batch_processing is True
        mock_select.assert_called_once()


class TestConfigureAutoModeSingleConfirm:
    """Item 3: auto mode must not double-confirm. _configure_auto_mode persists
    decisions and returns True without its own Proceed prompt; the generic
    summary step owns the single confirmation."""

    @pytest.mark.unit
    def test_no_auto_specific_confirm(self, tmp_path: Path) -> None:
        # resume_mode=overwrite bypasses the resume-skip filtering branch.
        config = UserConfiguration(processing_type="auto", resume_mode="overwrite")
        selector = MagicMock()
        decision = SimpleNamespace(file_path=tmp_path / "a.pdf")
        selector.create_decisions.return_value = [decision]
        config.auto_selector = selector

        with patch("modules.ui.workflows.prompt_yes_no") as mock_confirm:
            ok = WorkflowUI._configure_auto_mode(config, tmp_path, {})

        assert ok is True
        mock_confirm.assert_not_called()
        assert config.auto_decisions == [decision]
        assert config.selected_items == [tmp_path]


class TestSelectMobiFiles:
    """Tests for the MOBI selection helper (D1)."""

    @pytest.mark.unit
    def test_select_all_mobis(self, tmp_path: Path) -> None:
        """'all' selects every MOBI/AZW file, recursively."""
        (tmp_path / "a.mobi").write_bytes(b"")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.azw3").write_bytes(b"")
        config = UserConfiguration(processing_type="mobis")
        with patch("modules.ui.workflows.prompt_select", return_value=_continue("all")):
            ok = WorkflowUI._select_mobi_files(config, tmp_path)
        assert ok is True
        assert config.process_all is True
        assert {p.name for p in (config.selected_items or [])} == {"a.mobi", "b.azw3"}

    @pytest.mark.unit
    def test_no_mobis_found_returns_false(self, tmp_path: Path) -> None:
        """An empty directory yields a graceful False."""
        config = UserConfiguration(processing_type="mobis")
        with patch("modules.ui.workflows.prompt_select", return_value=_continue("all")):
            ok = WorkflowUI._select_mobi_files(config, tmp_path)
        assert ok is False


class TestDisplayCompletionSummary:
    """Tests for the comprehensive completion overview (E1/E4)."""

    @staticmethod
    def _service(paths_config: dict[str, Any]) -> MagicMock:
        svc = MagicMock()
        svc.get_paths_config.return_value = paths_config
        return svc

    @pytest.mark.unit
    def test_output_format_and_skipped_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Output extension follows config.output_format; skipped count shown."""
        config = UserConfiguration(processing_type="pdfs", output_format="md")
        paths = {
            "general": {"input_paths_is_output_path": False},
            "file_paths": {"PDFs": {"output": str(tmp_path / "pdfs_out")}},
        }
        with patch(
            "modules.config.service.get_config_service",
            return_value=self._service(paths),
        ):
            WorkflowUI.display_completion_summary(
                config, processed_count=2, failed_count=0, skipped=3
            )
        out = capsys.readouterr().out
        assert "Skipped (already complete): 3" in out
        assert ".md files" in out
        assert ".txt files" not in out

    @pytest.mark.unit
    def test_batch_reports_submitted_and_failed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Batch branch reports processed as submitted plus a failure line."""
        config = UserConfiguration(
            processing_type="pdfs",
            transcription_method="gpt",
            use_batch_processing=True,
        )
        paths: dict[str, Any] = {"general": {}, "file_paths": {}}
        with patch(
            "modules.config.service.get_config_service",
            return_value=self._service(paths),
        ):
            WorkflowUI.display_completion_summary(
                config, processed_count=3, failed_count=2
            )
        out = capsys.readouterr().out
        assert "Jobs submitted: 3" in out
        assert "Failed submissions: 2" in out

    @pytest.mark.unit
    def test_auto_output_location_resolved(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Auto mode reports the resolved Auto output directory."""
        config = UserConfiguration(processing_type="auto")
        auto_out = tmp_path / "auto_out"
        paths = {
            "general": {"input_paths_is_output_path": False},
            "file_paths": {"Auto": {"output": str(auto_out)}},
        }
        with patch(
            "modules.config.service.get_config_service",
            return_value=self._service(paths),
        ):
            WorkflowUI.display_completion_summary(config, processed_count=1)
        out = capsys.readouterr().out
        assert str(auto_out.resolve()) in out

    @pytest.mark.unit
    def test_token_usage_shown_when_enabled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Daily token usage line appears only when the daily limit is enabled."""
        config = UserConfiguration(processing_type="pdfs")
        paths: dict[str, Any] = {"general": {}, "file_paths": {}}
        tracker = MagicMock()
        tracker.get_stats.return_value = {
            "tokens_used_today": 1000,
            "daily_limit": 5000,
            "usage_percentage": 20.0,
        }
        with (
            patch(
                "modules.config.service.get_config_service",
                return_value=self._service(paths),
            ),
            patch(
                "modules.infra.token_budget.get_token_tracker",
                return_value=tracker,
            ),
        ):
            WorkflowUI.display_completion_summary(
                config,
                processed_count=1,
                concurrency_config={"daily_token_limit": {"enabled": True}},
            )
        out = capsys.readouterr().out
        assert "Daily token usage: 1,000/5,000 (20.0%)" in out

    @pytest.mark.unit
    def test_token_usage_hidden_when_disabled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No token line when the daily limit is disabled or config absent."""
        config = UserConfiguration(processing_type="pdfs")
        paths: dict[str, Any] = {"general": {}, "file_paths": {}}
        with patch(
            "modules.config.service.get_config_service",
            return_value=self._service(paths),
        ):
            WorkflowUI.display_completion_summary(config, processed_count=1)
        out = capsys.readouterr().out
        assert "Daily token usage" not in out


class TestAutoWizardStepOrder:
    """The auto-mode state machine visits resume_mode before item selection."""

    @pytest.mark.asyncio
    async def test_auto_flow_step_order(self, tmp_path: Path) -> None:
        from main import unified_transcriber as ut

        calls: list[str] = []

        def _proc_type(config: UserConfiguration) -> bool:
            calls.append("processing_type")
            config.processing_type = "auto"
            return True

        def _resume(config: UserConfiguration) -> bool:
            calls.append("resume_mode")
            return True

        def _select(
            config: UserConfiguration,
            base_dir: Path,
            paths_config: Any = None,
        ) -> bool:
            calls.append("item_selection")
            # No GPT decisions -> auto_schema_selection is a no-op step.
            config.auto_decisions = []
            return True

        def _schema(config: UserConfiguration) -> bool:
            calls.append("auto_schema_selection")
            return True

        def _page(config: UserConfiguration) -> bool:
            calls.append("page_range")
            return True

        def _summary(config: UserConfiguration) -> bool:
            calls.append("summary")
            return True

        with (
            patch.object(WorkflowUI, "display_welcome"),
            patch.object(
                WorkflowUI, "configure_processing_type", side_effect=_proc_type
            ),
            patch.object(WorkflowUI, "configure_resume_mode", side_effect=_resume),
            patch.object(
                WorkflowUI, "select_items_for_processing", side_effect=_select
            ),
            patch.object(WorkflowUI, "configure_auto_mode_schema", side_effect=_schema),
            patch.object(WorkflowUI, "configure_page_range", side_effect=_page),
            patch.object(
                WorkflowUI, "display_processing_summary", side_effect=_summary
            ),
        ):
            config = await ut.configure_user_workflow_interactive(
                tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, {}
            )

        assert config.processing_type == "auto"
        # resume_mode precedes item_selection in the reordered auto flow.
        assert calls.index("resume_mode") < calls.index("item_selection")
        assert calls == [
            "processing_type",
            "resume_mode",
            "item_selection",
            "auto_schema_selection",
            "page_range",
            "summary",
        ]
