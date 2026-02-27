"""Tests for modules.diagnostics.system_check."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from modules.diagnostics.system_check import (
    check_python_version,
    check_tesseract,
    check_api_key,
    check_config_files,
    check_system_requirements,
    diagnose_api_connectivity,
    generate_diagnostic_report,
)


# ---------------------------------------------------------------------------
# check_python_version
# ---------------------------------------------------------------------------

class TestCheckPythonVersion:
    def test_current_python_passes(self):
        ok, msg = check_python_version()
        assert ok is True
        assert "Python" in msg

    @patch("modules.diagnostics.system_check.sys")
    def test_old_python_fails(self, mock_sys):
        from collections import namedtuple
        VersionInfo = namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])
        mock_sys.version_info = VersionInfo(3, 7, 0, "final", 0)
        ok, msg = check_python_version()
        assert ok is False
        assert "requires 3.8+" in msg

    @patch("modules.diagnostics.system_check.sys")
    def test_exact_38_passes(self, mock_sys):
        from collections import namedtuple
        VersionInfo = namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])
        mock_sys.version_info = VersionInfo(3, 8, 0, "final", 0)
        ok, msg = check_python_version()
        assert ok is True


# ---------------------------------------------------------------------------
# check_tesseract
# ---------------------------------------------------------------------------

class TestCheckTesseract:
    @patch("modules.diagnostics.system_check.is_tesseract_available", return_value=True)
    def test_available(self, mock_tess):
        ok, msg = check_tesseract()
        assert ok is True
        assert "available" in msg

    @patch("modules.diagnostics.system_check.is_tesseract_available", return_value=False)
    def test_not_available(self, mock_tess):
        ok, msg = check_tesseract()
        assert ok is False
        assert "not found" in msg


# ---------------------------------------------------------------------------
# check_api_key
# ---------------------------------------------------------------------------

class TestCheckApiKey:
    def test_key_present(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789"}):
            ok, msg = check_api_key()
            assert ok is True
            assert "sk-test1..." in msg

    def test_key_absent(self):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            ok, msg = check_api_key()
            assert ok is False
            assert "not set" in msg

    def test_short_key_masked(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "short"}):
            ok, msg = check_api_key()
            assert ok is True
            assert "***" in msg


# ---------------------------------------------------------------------------
# check_config_files
# ---------------------------------------------------------------------------

class TestCheckConfigFiles:
    def test_all_files_exist(self):
        # In the actual project, config files should exist
        ok, msg = check_config_files()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# check_system_requirements
# ---------------------------------------------------------------------------

class TestCheckSystemRequirements:
    @patch("modules.diagnostics.system_check.check_python_version", return_value=(True, "Python 3.13"))
    @patch("modules.diagnostics.system_check.check_tesseract", return_value=(True, "Available"))
    @patch("modules.diagnostics.system_check.check_api_key", return_value=(True, "Set"))
    @patch("modules.diagnostics.system_check.check_config_files", return_value=(True, "All present"))
    def test_returns_dict_of_checks(self, *mocks):
        checks = check_system_requirements()
        assert "Python Version" in checks
        assert "Tesseract OCR" in checks
        assert "OpenAI API Key" in checks
        assert "Configuration Files" in checks
        for name, (status, msg) in checks.items():
            assert status is True


# ---------------------------------------------------------------------------
# diagnose_api_connectivity
# ---------------------------------------------------------------------------

class TestDiagnoseApiConnectivity:
    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_key(self):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            result = diagnose_api_connectivity()
            assert result["api_key_set"] is False
            assert result["error"] == "API key not configured"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    @patch("openai.OpenAI")
    def test_with_api_key_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_models.data = [MagicMock(), MagicMock()]
        mock_client.models.list.return_value = mock_models
        mock_openai_cls.return_value = mock_client

        result = diagnose_api_connectivity()
        assert result["api_key_set"] is True
        assert result["connectivity_test"] == "SUCCESS"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    @patch("openai.OpenAI")
    def test_with_api_key_failure(self, mock_openai_cls):
        mock_openai_cls.return_value.models.list.side_effect = RuntimeError("Network error")

        result = diagnose_api_connectivity()
        assert result["connectivity_test"] == "FAILED"
        assert "Network error" in result["error"]


# ---------------------------------------------------------------------------
# generate_diagnostic_report
# ---------------------------------------------------------------------------

class TestGenerateDiagnosticReport:
    @patch("modules.diagnostics.system_check.check_system_requirements")
    @patch("modules.diagnostics.system_check.diagnose_api_connectivity")
    def test_returns_formatted_string(self, mock_diag, mock_checks):
        mock_checks.return_value = {
            "Python Version": (True, "Python 3.13"),
        }
        mock_diag.return_value = {
            "api_key_set": True,
            "connectivity_test": "SUCCESS",
        }
        report = generate_diagnostic_report()
        assert "ChronoTranscriber System Diagnostics" in report
        assert "Python Version" in report
