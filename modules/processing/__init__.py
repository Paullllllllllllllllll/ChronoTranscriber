"""Document processing package.

Provides PDF, EPUB, image processing, OCR, text extraction, and post-processing utilities.
"""

# Avoid circular imports - use direct imports instead of re-exporting
from .epub_utils import EPUBProcessor, EPUBTextExtraction
from .postprocess import postprocess_transcription, postprocess_file, postprocess_text
from .model_utils import detect_model_type, get_image_config_section_name

__all__ = [
    "detect_model_type",
    "get_image_config_section_name",
    "PDFProcessor",
    "native_extract_pdf_text",
    "ImageProcessor",
    "extract_transcribed_text",
    "format_page_line",
    "EPUBProcessor",
    "EPUBTextExtraction",
    "configure_tesseract_executable",
    "is_tesseract_available",
    "ensure_tesseract_available",
    "perform_ocr",
    # Post-processing
    "postprocess_transcription",
    "postprocess_file",
    "postprocess_text",
]
