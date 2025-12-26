"""Tesseract OCR utilities and setup.

Provides functions for Tesseract availability checking, configuration,
and OCR execution with proper error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image

from modules.infra.logger import setup_logger
from modules.ui import print_error

logger = setup_logger(__name__)


def configure_tesseract_executable(image_processing_config: dict) -> None:
    """Configure Tesseract executable path from configuration.
    
    Args:
        image_processing_config: Image processing configuration dictionary.
    """
    ocr_config = (
        image_processing_config
        .get('tesseract_image_processing', {})
        .get('ocr', {})
    )
    tess_cmd = (ocr_config.get('tesseract_cmd') or '').strip()
    
    if tess_cmd:
        try:
            cmd_path = Path(tess_cmd)
            if cmd_path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(cmd_path)
                logger.info(f"Using Tesseract executable: {cmd_path}")
            else:
                logger.warning(f"Configured tesseract_cmd does not exist: {cmd_path}")
        except Exception as e:
            logger.warning(f"Could not set tesseract_cmd '{tess_cmd}': {e}")


def is_tesseract_available() -> bool:
    """Check if Tesseract is available and accessible.
    
    Returns:
        True if Tesseract is available, False otherwise.
    """
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except getattr(pytesseract, 'TesseractNotFoundError', Exception):
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking Tesseract availability: {e}")
        return False


def ensure_tesseract_available() -> bool:
    """Verify that Tesseract is available, printing error message if not.
    
    Returns:
        True if available, False otherwise.
    """
    if is_tesseract_available():
        return True
    
    print_error(
        "Tesseract is not installed or not in PATH.\n"
        "- Install: https://github.com/tesseract-ocr/tesseract (Windows: official installer)\n"
        "- Or set 'tesseract_image_processing.ocr.tesseract_cmd' in config/image_processing_config.yaml to the full path, e.g.:\n"
        "  C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe"
    )
    return False


def perform_ocr(img_path: Path, tesseract_config: str) -> Optional[str]:
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
    except Exception as e:
        logger.error(f"Tesseract OCR error on {img_path.name}: {e}")
        return None
