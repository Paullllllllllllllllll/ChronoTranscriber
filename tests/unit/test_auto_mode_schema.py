"""Unit tests for auto mode schema selection functionality.

Tests the schema selection for both interactive and CLI modes in auto mode.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from argparse import Namespace

from modules.ui.core import UserConfiguration
from modules.ui.workflows import WorkflowUI
from modules.llm.schema_utils import list_schema_options


class TestConfigureAutoModeSchema:
    """Tests for WorkflowUI.configure_auto_mode_schema method."""
    
    @pytest.mark.unit
    def test_skips_when_no_gpt_decisions(self):
        """Test that schema selection is skipped when no GPT decisions exist."""
        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[],
        )
        
        result = WorkflowUI.configure_auto_mode_schema(config)
        
        assert result is True
        # Schema should remain unset
        assert config.selected_schema_name is None
    
    @pytest.mark.unit
    def test_skips_when_decisions_are_non_gpt(self):
        """Test that schema selection is skipped when all decisions use non-GPT methods."""
        # Create mock decisions with non-GPT methods
        mock_decision = MagicMock()
        mock_decision.method = "tesseract"
        
        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[mock_decision],
        )
        
        result = WorkflowUI.configure_auto_mode_schema(config)
        
        assert result is True
    
    @pytest.mark.unit
    def test_prompts_when_gpt_decisions_exist(self):
        """Test that schema selection is prompted when GPT decisions exist."""
        # Create mock decision with GPT method
        mock_decision = MagicMock()
        mock_decision.method = "gpt"
        
        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[mock_decision],
        )
        
        # Mock the schema selection and context configuration
        with patch.object(WorkflowUI, 'configure_schema_selection', return_value=True) as mock_schema:
            with patch.object(WorkflowUI, 'configure_additional_context', return_value=True) as mock_context:
                result = WorkflowUI.configure_auto_mode_schema(config)
        
        assert result is True
        mock_schema.assert_called_once_with(config)
        mock_context.assert_called_once_with(config)
    
    @pytest.mark.unit
    def test_returns_false_when_schema_selection_cancelled(self):
        """Test that False is returned when user cancels schema selection."""
        mock_decision = MagicMock()
        mock_decision.method = "gpt"
        
        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[mock_decision],
        )
        
        with patch.object(WorkflowUI, 'configure_schema_selection', return_value=False):
            result = WorkflowUI.configure_auto_mode_schema(config)
        
        assert result is False
    
    @pytest.mark.unit
    def test_returns_false_when_context_selection_cancelled(self):
        """Test that False is returned when user cancels context selection."""
        mock_decision = MagicMock()
        mock_decision.method = "gpt"
        
        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[mock_decision],
        )
        
        with patch.object(WorkflowUI, 'configure_schema_selection', return_value=True):
            with patch.object(WorkflowUI, 'configure_additional_context', return_value=False):
                result = WorkflowUI.configure_auto_mode_schema(config)
        
        assert result is False


class TestAutoModeCliSchemaSelection:
    """Tests for CLI mode auto with schema argument."""
    
    @pytest.mark.unit
    def test_auto_mode_with_schema_argument(self, temp_dir, mock_paths_config):
        """Test that --schema argument works with --auto mode."""
        from main.unified_transcriber import create_config_from_cli_args
        from modules.core.auto_selector import AutoSelector
        
        # Create test files in temp directory
        test_file = temp_dir / "test_image_folder"
        test_file.mkdir()
        (test_file / "page_001.png").write_bytes(b"")
        
        # Update paths config for test
        mock_paths_config['file_paths']['Auto'] = {
            'input': str(temp_dir),
            'output': str(temp_dir / "output"),
        }
        
        # Create mock args
        args = Namespace(
            auto=True,
            input=str(temp_dir),
            output=str(temp_dir / "output"),
            schema="markdown_transcription_schema",
            context=None,
            type=None,
            method=None,
            batch=False,
            files=None,
            recursive=False,
        )
        
        # Mock list_schema_options to return our test schema
        schemas_dir = Path(__file__).parent.parent.parent / "schemas"
        mock_options = [
            ("markdown_transcription_schema", schemas_dir / "markdown_transcription_schema.json"),
        ]
        
        with patch('main.unified_transcriber.list_schema_options', return_value=mock_options):
            config = create_config_from_cli_args(
                args,
                temp_dir,
                temp_dir / "output",
                mock_paths_config
            )
        
        assert config.processing_type == "auto"
        assert config.selected_schema_name == "markdown_transcription_schema"
        assert config.selected_schema_path is not None
    
    @pytest.mark.unit
    def test_auto_mode_invalid_schema_raises_error(self, temp_dir, mock_paths_config):
        """Test that invalid schema name raises ValueError."""
        from main.unified_transcriber import create_config_from_cli_args
        
        # Create test files
        test_folder = temp_dir / "test_folder"
        test_folder.mkdir()
        (test_folder / "page_001.png").write_bytes(b"")
        
        mock_paths_config['file_paths']['Auto'] = {
            'input': str(temp_dir),
            'output': str(temp_dir / "output"),
        }
        
        args = Namespace(
            auto=True,
            input=str(temp_dir),
            output=str(temp_dir / "output"),
            schema="nonexistent_schema",
            context=None,
            type=None,
            method=None,
            batch=False,
            files=None,
            recursive=False,
        )
        
        with patch('main.unified_transcriber.list_schema_options', return_value=[]):
            with pytest.raises(ValueError, match="Schema 'nonexistent_schema' not found"):
                create_config_from_cli_args(
                    args,
                    temp_dir,
                    temp_dir / "output",
                    mock_paths_config
                )
    
    @pytest.mark.unit
    def test_auto_mode_default_schema_when_not_specified(self, temp_dir, mock_paths_config):
        """Test that default schema is used when --schema is not specified."""
        from main.unified_transcriber import create_config_from_cli_args
        
        # Create test files
        test_folder = temp_dir / "test_folder"
        test_folder.mkdir()
        (test_folder / "page_001.png").write_bytes(b"")
        
        mock_paths_config['file_paths']['Auto'] = {
            'input': str(temp_dir),
            'output': str(temp_dir / "output"),
        }
        
        args = Namespace(
            auto=True,
            input=str(temp_dir),
            output=str(temp_dir / "output"),
            schema=None,  # Not specified
            context=None,
            type=None,
            method=None,
            batch=False,
            files=None,
            recursive=False,
        )
        
        config = create_config_from_cli_args(
            args,
            temp_dir,
            temp_dir / "output",
            mock_paths_config
        )
        
        assert config.selected_schema_name == "markdown_transcription_schema"
        assert config.selected_schema_path is not None


class TestAutoModeContextSelection:
    """Tests for context selection in auto mode."""
    
    @pytest.mark.unit
    def test_auto_mode_with_context_argument(self, temp_dir, mock_paths_config):
        """Test that --context argument works with --auto mode."""
        from main.unified_transcriber import create_config_from_cli_args
        
        # Create test files
        test_folder = temp_dir / "test_folder"
        test_folder.mkdir()
        (test_folder / "page_001.png").write_bytes(b"")
        
        # Create context file
        context_file = temp_dir / "context.txt"
        context_file.write_text("Test context")
        
        mock_paths_config['file_paths']['Auto'] = {
            'input': str(temp_dir),
            'output': str(temp_dir / "output"),
        }
        
        args = Namespace(
            auto=True,
            input=str(temp_dir),
            output=str(temp_dir / "output"),
            schema=None,
            context=str(context_file),
            type=None,
            method=None,
            batch=False,
            files=None,
            recursive=False,
        )
        
        config = create_config_from_cli_args(
            args,
            temp_dir,
            temp_dir / "output",
            mock_paths_config
        )
        
        assert config.additional_context_path is not None
        assert config.additional_context_path.exists()
