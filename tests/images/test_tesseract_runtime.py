"""Tesseract executable selection, availability probing and OCR execution."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytesseract import TesseractNotFoundError

from chronotranscriber.images.tesseract_runtime import (
    ensure_tesseract_available,
    is_tesseract_available,
    perform_ocr,
    use_tesseract_command,
)


class TestUseTesseractCommand:
    @pytest.mark.parametrize("command", [None, "", "   "])
    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    def test_blank_command_leaves_the_executable_unchanged(
        self, mock_pyt, command: str | None
    ) -> None:
        mock_pyt.pytesseract.tesseract_cmd = "tesseract"
        use_tesseract_command(command)
        assert mock_pyt.pytesseract.tesseract_cmd == "tesseract"

    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    def test_existing_path_becomes_the_executable(
        self, mock_pyt, tmp_path: Path
    ) -> None:
        exe = tmp_path / "tesseract.exe"
        exe.write_text("fake", encoding="utf-8")
        use_tesseract_command(str(exe))
        assert mock_pyt.pytesseract.tesseract_cmd == str(exe)

    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    def test_missing_path_warns_and_is_ignored(
        self, mock_pyt, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_pyt.pytesseract.tesseract_cmd = "tesseract"
        with caplog.at_level(logging.WARNING):
            use_tesseract_command(str(tmp_path / "missing.exe"))
        assert mock_pyt.pytesseract.tesseract_cmd == "tesseract"
        assert "does not exist" in caplog.text


class TestIsTesseractAvailable:
    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    def test_available(self, mock_pyt) -> None:
        mock_pyt.get_tesseract_version.return_value = "5.3.0"
        assert is_tesseract_available() is True

    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    def test_missing_executable_is_unavailable(self, mock_pyt) -> None:
        mock_pyt.get_tesseract_version.side_effect = TesseractNotFoundError()
        assert is_tesseract_available() is False

    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    def test_unexpected_probe_error_is_unavailable(self, mock_pyt) -> None:
        mock_pyt.get_tesseract_version.side_effect = RuntimeError("unexpected")
        assert is_tesseract_available() is False


class TestEnsureTesseractAvailable:
    @pytest.mark.parametrize("available", [True, False])
    def test_returns_the_probe_result(self, available: bool) -> None:
        with patch(
            "chronotranscriber.images.tesseract_runtime.is_tesseract_available",
            return_value=available,
        ):
            assert ensure_tesseract_available() is available


class TestPerformOcr:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Hello World  ", "Hello World"),
            ("   ", "[No transcribable text]"),
        ],
    )
    @patch("chronotranscriber.images.tesseract_runtime.pytesseract")
    @patch("chronotranscriber.images.tesseract_runtime.Image")
    def test_text_is_stripped_and_blank_output_gets_a_placeholder(
        self, mock_image_cls, mock_pyt, tmp_path: Path, raw: str, expected: str
    ) -> None:
        mock_pyt.image_to_string.return_value = raw
        mock_img = MagicMock()
        mock_image_cls.open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_image_cls.open.return_value.__exit__ = MagicMock(return_value=False)

        result = perform_ocr(tmp_path / "test.png", "--oem 3 --psm 6")
        assert result == expected

    @patch("chronotranscriber.images.tesseract_runtime.Image")
    def test_a_failing_image_returns_none(self, mock_image_cls, tmp_path: Path) -> None:
        mock_image_cls.open.side_effect = RuntimeError("file error")
        result = perform_ocr(tmp_path / "test.png", "--oem 3 --psm 6")
        assert result is None
