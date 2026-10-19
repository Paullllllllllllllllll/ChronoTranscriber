"""Tesseract OCR utilities and setup.

Provides functions for Tesseract availability checking, configuration,
and OCR execution with proper error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import TesseractNotFoundError

logger = logging.getLogger(__name__)

TESSERACT_MISSING = (
    "Tesseract is not installed or not in PATH.\n"
    "- Install: https://github.com/tesseract-ocr/tesseract"
    " (Windows: official installer)\n"
    "- Or set 'tools.tesseract_cmd' in the settings file to the full"
    " path, e.g.:\n"
    "  C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe"
)
"""The message for a run whose Tesseract executable cannot be found."""


def use_tesseract_command(command: str | None) -> None:
    """Run Tesseract from *command* (the settings' ``tools.tesseract_cmd``).

    Without a command the executable stays as it is (found on PATH by
    default); a command that does not exist is reported and ignored.
    """
    tess_cmd = (command or "").strip()
    if tess_cmd:
        try:
            cmd_path = Path(tess_cmd)
            if cmd_path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(cmd_path)
                logger.info(f"Using Tesseract executable: {cmd_path}")
            else:
                logger.warning(f"Configured tesseract_cmd does not exist: {cmd_path}")
        except (OSError, ValueError) as e:
            logger.warning(f"Could not set tesseract_cmd '{tess_cmd}': {e}")


def is_tesseract_available() -> bool:
    """Check if Tesseract is available and accessible.

    Returns:
        True if Tesseract is available, False otherwise.
    """
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except TesseractNotFoundError:
        return False
    except Exception as e:  # noqa: BLE001 - any probe failure means unavailable
        logger.error(f"Unexpected error checking Tesseract availability: {e}")
        return False


def ensure_tesseract_available() -> bool:
    """Verify that Tesseract is available; log :data:`TESSERACT_MISSING` if not.

    Returns:
        True if available, False otherwise.
    """
    if is_tesseract_available():
        return True
    logger.error(TESSERACT_MISSING)
    return False


def perform_ocr(img_path: Path, tesseract_config: str) -> str | None:
    """Perform OCR on an image using Tesseract.

    Args:
        img_path: Path to image file.
        tesseract_config: Tesseract configuration string.

    Returns:
        Extracted text, or placeholder if no text found, or None on error.
    """
    try:
        with Image.open(img_path) as img:
            text = pytesseract.image_to_string(img, config=tesseract_config)
            return text.strip() if text.strip() else "[No transcribable text]"
    except Exception as e:  # noqa: BLE001 - a failed image is logged and skipped
        logger.error(f"Tesseract OCR error on {img_path.name}: {e}")
        return None
