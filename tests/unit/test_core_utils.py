"""Unit tests for modules/core/utils.py."""

from __future__ import annotations

import pytest
import warnings
from unittest.mock import patch, MagicMock


class TestConsolePrint:
    """Tests for console_print function (deprecated)."""

    @pytest.mark.unit
    def test_prints_message(self, capsys):
        """Test that message is printed to console."""
        from modules.core.utils import console_print
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            console_print("Test message")
        
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    @pytest.mark.unit
    def test_emits_deprecation_warning(self):
        """Test that deprecation warning is emitted."""
        from modules.core.utils import console_print
        
        with pytest.warns(DeprecationWarning, match="console_print is deprecated"):
            console_print("Test")


class TestCheckExit:
    """Tests for check_exit function."""

    @pytest.mark.unit
    def test_exits_on_q(self):
        """Test exits when user inputs 'q'."""
        from modules.core.utils import check_exit
        
        with pytest.raises(SystemExit) as exc_info:
            check_exit("q")
        
        assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_exits_on_exit(self):
        """Test exits when user inputs 'exit'."""
        from modules.core.utils import check_exit
        
        with pytest.raises(SystemExit) as exc_info:
            check_exit("exit")
        
        assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_exits_case_insensitive(self):
        """Test exit is case insensitive."""
        from modules.core.utils import check_exit
        
        with pytest.raises(SystemExit):
            check_exit("Q")
        
    @pytest.mark.unit
    def test_does_not_exit_on_other_input(self):
        """Test does not exit for other inputs."""
        from modules.core.utils import check_exit
        
        # Should not raise
        check_exit("continue")
        check_exit("yes")
        check_exit("1")


class TestSafeInput:
    """Tests for safe_input function."""

    @pytest.mark.unit
    def test_returns_stripped_input(self):
        """Test that input is stripped of whitespace."""
        from modules.core.utils import safe_input
        
        with patch('builtins.input', return_value="  test input  "):
            result = safe_input("Enter: ")
        
        assert result == "test input"

    @pytest.mark.unit
    def test_shows_prompt(self):
        """Test that prompt is passed to input."""
        from modules.core.utils import safe_input
        
        mock_input = MagicMock(return_value="test")
        with patch('builtins.input', mock_input):
            safe_input("Enter value: ")
        
        mock_input.assert_called_once_with("Enter value: ")

    @pytest.mark.unit
    def test_exits_on_error(self):
        """Test that function exits on input error."""
        from modules.core.utils import safe_input
        
        with patch('builtins.input', side_effect=EOFError("No input")):
            with pytest.raises(SystemExit) as exc_info:
                safe_input("Enter: ")
        
        assert exc_info.value.code == 1
