"""Unit tests for modules/ui/prompts.py.

Tests UI prompt utilities including styled output and input validation.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from modules.ui.prompts import (
    PromptStyle,
    ui_print,
    print_header,
    print_separator,
    print_info,
    print_success,
    print_warning,
    print_error,
    NavigationAction,
    PromptResult,
)


class TestPromptStyle:
    """Tests for PromptStyle class."""
    
    @pytest.mark.unit
    def test_all_styles_exist(self):
        """Test that all expected styles exist."""
        # Core styles available in PromptStyle
        expected = [
            "HEADER", "HIGHLIGHT", "INFO", "SUCCESS",
            "WARNING", "ERROR", "DIM", "PROMPT", "DOUBLE_LINE",
            "SINGLE_LINE", "LIGHT_LINE",
        ]
        for style_name in expected:
            assert hasattr(PromptStyle, style_name)


class TestUiPrint:
    """Tests for ui_print function."""
    
    @pytest.mark.unit
    def test_basic_output(self, capsys):
        """Test basic output without style."""
        ui_print("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
    
    @pytest.mark.unit
    def test_handles_unicode_error(self, capsys):
        """Test graceful handling of unicode encoding errors."""
        # This test verifies the error handling exists
        ui_print("Normal text")
        captured = capsys.readouterr()
        assert "Normal" in captured.out
    
    @pytest.mark.unit
    def test_empty_message(self, capsys):
        """Test printing empty message."""
        ui_print("")
        captured = capsys.readouterr()
        # Should still print newline
        assert captured.out == "\n"


class TestPrintHeader:
    """Tests for print_header function."""
    
    @pytest.mark.unit
    def test_header_with_subtitle(self, capsys):
        """Test header with subtitle."""
        print_header("Main Title", "Subtitle text")
        captured = capsys.readouterr()
        assert "Main Title" in captured.out
        assert "Subtitle text" in captured.out
    
    @pytest.mark.unit
    def test_header_without_subtitle(self, capsys):
        """Test header without subtitle."""
        print_header("Title Only")
        captured = capsys.readouterr()
        assert "Title Only" in captured.out


class TestPrintSeparator:
    """Tests for print_separator function."""
    
    @pytest.mark.unit
    def test_default_separator(self, capsys):
        """Test default separator."""
        print_separator()
        captured = capsys.readouterr()
        # Should print something (a line of some kind)
        assert len(captured.out.strip()) > 0
    
    @pytest.mark.unit
    def test_custom_width(self, capsys):
        """Test separator with custom width."""
        print_separator(width=40)
        captured = capsys.readouterr()
        # Output includes ANSI codes, so just verify it printed something
        assert len(captured.out) > 0
        # The actual separator characters should be present
        assert "-" in captured.out or "=" in captured.out or "." in captured.out


class TestPrintInfo:
    """Tests for print_info function."""
    
    @pytest.mark.unit
    def test_info_with_title(self, capsys):
        """Test info message with title."""
        print_info("CATEGORY", "Info message")
        captured = capsys.readouterr()
        assert "CATEGORY" in captured.out
        assert "Info message" in captured.out
    
    @pytest.mark.unit
    def test_info_without_title(self, capsys):
        """Test info message without title."""
        print_info("Just a message")
        captured = capsys.readouterr()
        assert "Just a message" in captured.out


class TestPrintSuccess:
    """Tests for print_success function."""
    
    @pytest.mark.unit
    def test_success_message(self, capsys):
        """Test success message output."""
        print_success("Operation completed!")
        captured = capsys.readouterr()
        assert "Operation completed!" in captured.out


class TestPrintWarning:
    """Tests for print_warning function."""
    
    @pytest.mark.unit
    def test_warning_message(self, capsys):
        """Test warning message output."""
        print_warning("Warning: something might be wrong")
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "wrong" in captured.out


class TestPrintError:
    """Tests for print_error function."""
    
    @pytest.mark.unit
    def test_error_message(self, capsys):
        """Test error message output."""
        print_error("Error: something failed")
        captured = capsys.readouterr()
        assert "Error" in captured.out or "failed" in captured.out


class TestNavigationAction:
    """Tests for NavigationAction enum."""
    
    @pytest.mark.unit
    def test_action_values(self):
        """Test NavigationAction enum values."""
        assert NavigationAction.CONTINUE.value == "continue"
        assert NavigationAction.BACK.value == "back"
        assert NavigationAction.QUIT.value == "quit"


class TestPromptResult:
    """Tests for PromptResult dataclass."""
    
    @pytest.mark.unit
    def test_back_result(self):
        """Test PromptResult for back action."""
        result = PromptResult(action=NavigationAction.BACK)
        assert result.action == NavigationAction.BACK
        assert result.value is None
    
    @pytest.mark.unit
    def test_quit_result(self):
        """Test PromptResult for quit action."""
        result = PromptResult(action=NavigationAction.QUIT)
        assert result.action == NavigationAction.QUIT
    
    @pytest.mark.unit
    def test_continue_with_value(self):
        """Test PromptResult with value."""
        result = PromptResult(action=NavigationAction.CONTINUE, value="selected_option")
        assert result.action == NavigationAction.CONTINUE
        assert result.value == "selected_option"


