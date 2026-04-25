"""Image preprocessing, encoding, and OCR runtime.

Absorbs the former modules.processing.{image_utils,tesseract_utils} and
modules.images.encoding into one package covering every image
transformation from loaded bytes to provider-ready payload, plus
Tesseract executable lookup and OCR invocation.
"""

from modules.images.encoding import (
    encode_image_to_base64,
    encode_image_to_data_url,
)
from modules.images.pipeline import ImageProcessor
from modules.images.tesseract_runtime import (
    configure_tesseract_executable,
    ensure_tesseract_available,
    is_tesseract_available,
    perform_ocr,
)

__all__ = [
    "ImageProcessor",
    "encode_image_to_base64",
    "encode_image_to_data_url",
    "configure_tesseract_executable",
    "ensure_tesseract_available",
    "is_tesseract_available",
    "perform_ocr",
]
