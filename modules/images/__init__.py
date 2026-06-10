"""Image preprocessing, encoding, and OCR runtime.

Absorbs the former modules.processing.{image_utils,tesseract_utils} and
modules.images.encoding into one package covering every image
transformation from loaded bytes to provider-ready payload, plus
Tesseract executable lookup and OCR invocation.
"""

from modules.images.encoding import (
    encode_bytes_to_base64,
    encode_image_to_base64,
    encode_image_to_data_url,
)
from modules.images.page_stream import (
    PagePayload,
    compute_folder_skip_names,
    compute_pdf_skip_indices,
    load_image_payload,
    render_single_pdf_page_payload,
    resolve_image_settings,
    stream_folder_payloads,
    stream_pdf_payloads,
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
    "PagePayload",
    "compute_folder_skip_names",
    "compute_pdf_skip_indices",
    "encode_bytes_to_base64",
    "encode_image_to_base64",
    "encode_image_to_data_url",
    "load_image_payload",
    "render_single_pdf_page_payload",
    "resolve_image_settings",
    "stream_folder_payloads",
    "stream_pdf_payloads",
    "configure_tesseract_executable",
    "ensure_tesseract_available",
    "is_tesseract_available",
    "perform_ocr",
]
