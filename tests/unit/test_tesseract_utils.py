"""Tests for modules.images.tesseract_runtime."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from modules.images.tesseract_runtime import (
    configure_tesseract_executable,
    is_tesseract_available,
    ensure_tesseract_available,
    perform_ocr,
)


# ---------------------------------------------------------------------------
# configure_tesseract_executable
# ---------------------------------------------------------------------------

class TestConfigureTesseractExecutable:
    def test_empty_config(self) -> None:
        configure_tesseract_executable({})

    def test_empty_string_cmd(self) -> None:
        config = {
            "tesseract_image_processing": {
                "ocr": {"tesseract_cmd": ""}
            }
        }
        configure_tesseract_executable(config)

    def test_whitespace_only_cmd(self) -> None:
        config = {
            "tesseract_image_processing": {
                "ocr": {"tesseract_cmd": "   "}
            }
        }
        configure_tesseract_executable(config)

    @patch("modules.images.tesseract_runtime.pytesseract")
    def test_valid_path_sets_cmd(self, mock_pyt, tmp_path: Path) -> None:
        exe = tmp_path / "tesseract.exe"
        exe.write_text("fake", encoding="utf-8")
        config = {
            "tesseract_image_processing": {
                "ocr": {"tesseract_cmd": str(exe)}
            }
        }
        configure_tesseract_executable(config)
        assert mock_pyt.pytesseract.tesseract_cmd == str(exe)

    def test_nonexistent_path_warns(self, tmp_path: Path) -> None:
        config = {
            "tesseract_image_processing": {
                "ocr": {"tesseract_cmd": str(tmp_path / "missing.exe")}
            }
        }
        configure_tesseract_executable(config)  # Should not raise


# ---------------------------------------------------------------------------
# is_tesseract_available
# ---------------------------------------------------------------------------

class TestIsTesseractAvailable:
    @patch("modules.images.tesseract_runtime.pytesseract")
    def test_available(self, mock_pyt) -> None:
        mock_pyt.get_tesseract_version.return_value = "5.3.0"
        mock_pyt.TesseractNotFoundError = Exception
        assert is_tesseract_available() is True

    @patch("modules.images.tesseract_runtime.pytesseract")
    def test_not_found(self, mock_pyt) -> None:
        mock_pyt.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
        mock_pyt.get_tesseract_version.side_effect = mock_pyt.TesseractNotFoundError()
        assert is_tesseract_available() is False

    @patch("modules.images.tesseract_runtime.pytesseract")
    def test_unexpected_error(self, mock_pyt) -> None:
        mock_pyt.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
        mock_pyt.get_tesseract_version.side_effect = RuntimeError("unexpected")
        assert is_tesseract_available() is False


# ---------------------------------------------------------------------------
# ensure_tesseract_available
# ---------------------------------------------------------------------------

class TestEnsureTesseractAvailable:
    @patch("modules.images.tesseract_runtime.is_tesseract_available", return_value=True)
    def test_returns_true_when_available(self, mock_check) -> None:
        assert ensure_tesseract_available() is True

    @patch("modules.images.tesseract_runtime.is_tesseract_available", return_value=False)
    def test_returns_false_when_not_available(self, mock_check) -> None:
        assert ensure_tesseract_available() is False


# ---------------------------------------------------------------------------
# perform_ocr
# ---------------------------------------------------------------------------

class TestPerformOcr:
    @patch("modules.images.tesseract_runtime.pytesseract")
    @patch("modules.images.tesseract_runtime.Image")
    def test_successful_ocr(self, mock_image_cls, mock_pyt, tmp_path: Path) -> None:
        mock_pyt.image_to_string.return_value = "  Hello World  "
        mock_img = MagicMock()
        mock_image_cls.open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_image_cls.open.return_value.__exit__ = MagicMock(return_value=False)

        result = perform_ocr(tmp_path / "test.png", "--oem 3 --psm 6")
        assert result == "Hello World"

    @patch("modules.images.tesseract_runtime.pytesseract")
    @patch("modules.images.tesseract_runtime.Image")
    def test_empty_text_returns_placeholder(self, mock_image_cls, mock_pyt, tmp_path: Path) -> None:
        mock_pyt.image_to_string.return_value = "   "
        mock_img = MagicMock()
        mock_image_cls.open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_image_cls.open.return_value.__exit__ = MagicMock(return_value=False)

        result = perform_ocr(tmp_path / "test.png", "--oem 3 --psm 6")
        assert result == "[No transcribable text]"

    @patch("modules.images.tesseract_runtime.Image")
    def test_exception_returns_none(self, mock_image_cls, tmp_path: Path) -> None:
        mock_image_cls.open.side_effect = RuntimeError("file error")
        result = perform_ocr(tmp_path / "test.png", "--oem 3 --psm 6")
        assert result is None