"""Unit tests for modules/core/mode_selector.py.

Tests dual CLI/interactive mode selection logic.
"""

from __future__ import annotations

import sys
import pytest
from argparse import ArgumentParser
from unittest.mock import patch, MagicMock

from modules.core.mode_selector import (
    run_with_mode_detection,
    run_sync_with_mode_detection,
)


def create_test_parser():
    """Create a simple test argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--test", default="value")
    return parser


class TestRunWithModeDetection:
    """Tests for run_with_mode_detection function."""
    
    @pytest.mark.unit
    def test_interactive_mode_returns_none_args(self):
        """Test that interactive mode returns None for args."""
        mock_config = {
            "general": {"interactive_mode": True}
        }
        
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            config_service, interactive, args, paths = run_with_mode_detection(
                interactive_handler=MagicMock(),
                cli_handler=MagicMock(),
                parser_factory=create_test_parser,
                script_name="test_script",
            )
        
        assert interactive is True
        assert args is None
    
    @pytest.mark.unit
    def test_cli_mode_parses_args(self):
        """Test that CLI mode parses arguments."""
        mock_config = {
            "general": {"interactive_mode": False}
        }
        
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            with patch.object(sys, 'argv', ['script', '--test', 'custom']):
                config_service, interactive, args, paths = run_with_mode_detection(
                    interactive_handler=MagicMock(),
                    cli_handler=MagicMock(),
                    parser_factory=create_test_parser,
                    script_name="test_script",
                )
        
        assert interactive is False
        assert args is not None
        assert args.test == "custom"
    
    @pytest.mark.unit
    def test_config_load_failure_exits(self):
        """Test that configuration load failure causes exit."""
        with patch('modules.core.mode_selector.get_config_service', side_effect=Exception("Config error")):
            with patch('modules.core.mode_selector.print_error'):
                with pytest.raises(SystemExit) as exc_info:
                    run_with_mode_detection(
                        interactive_handler=MagicMock(),
                        cli_handler=MagicMock(),
                        parser_factory=create_test_parser,
                        script_name="test_script",
                    )
                
                assert exc_info.value.code == 1
    
    @pytest.mark.unit
    def test_returns_config_service(self):
        """Test that config service is returned."""
        mock_config = {"general": {"interactive_mode": True}}
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            config_service, _, _, _ = run_with_mode_detection(
                interactive_handler=MagicMock(),
                cli_handler=MagicMock(),
                parser_factory=create_test_parser,
                script_name="test_script",
            )
        
        assert config_service is mock_service
    
    @pytest.mark.unit
    def test_returns_paths_config(self):
        """Test that paths config is returned."""
        mock_config = {"general": {"interactive_mode": True}, "file_paths": {}}
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            _, _, _, paths = run_with_mode_detection(
                interactive_handler=MagicMock(),
                cli_handler=MagicMock(),
                parser_factory=create_test_parser,
                script_name="test_script",
            )
        
        assert paths == mock_config


class TestRunSyncWithModeDetection:
    """Tests for run_sync_with_mode_detection function."""
    
    @pytest.mark.unit
    def test_same_behavior_as_async(self):
        """Test that sync version has same behavior."""
        mock_config = {"general": {"interactive_mode": True}}
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            config_service, interactive, args, paths = run_sync_with_mode_detection(
                interactive_handler=MagicMock(),
                cli_handler=MagicMock(),
                parser_factory=create_test_parser,
                script_name="test_script",
            )
        
        assert interactive is True
        assert args is None


class TestModeDetectionDefaults:
    """Tests for default behavior when config values are missing."""
    
    @pytest.mark.unit
    def test_defaults_to_interactive_mode(self):
        """Test that missing interactive_mode defaults to True."""
        mock_config = {"general": {}}  # No interactive_mode key
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            _, interactive, _, _ = run_with_mode_detection(
                interactive_handler=MagicMock(),
                cli_handler=MagicMock(),
                parser_factory=create_test_parser,
                script_name="test_script",
            )
        
        assert interactive is True
    
    @pytest.mark.unit
    def test_empty_config_defaults_to_interactive(self):
        """Test that empty config defaults to interactive mode."""
        mock_config = {}  # No general section
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = mock_config
        
        with patch('modules.core.mode_selector.get_config_service', return_value=mock_service):
            _, interactive, _, _ = run_with_mode_detection(
                interactive_handler=MagicMock(),
                cli_handler=MagicMock(),
                parser_factory=create_test_parser,
                script_name="test_script",
            )
        
        assert interactive is True
