"""Image encoding utilities for LLM API requests."""

from __future__ import annotations

import base64
from pathlib import Path

from modules.config.constants import SUPPORTED_IMAGE_FORMATS
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Encode an image file to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (base64_data, mime_type)

    Raises:
        ValueError: If the image format is not supported
    """
    ext = image_path.suffix.lower()
    mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, mime_type


def encode_image_to_data_url(image_path: Path) -> str:
    """Encode an image file as a data URL.

    Args:
        image_path: Path to the image file

    Returns:
        Data URL string (data:{mime};base64,{data})

    Raises:
        ValueError: If the image format is not supported
    """
    data, mime_type = encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{data}"
