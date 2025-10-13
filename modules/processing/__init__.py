"""Document processing package.

Provides PDF, EPUB, image processing, OCR, and text extraction utilities.
"""

# Avoid circular imports - use direct imports instead of re-exporting
from .epub_utils import EPUBProcessor, EPUBTextExtraction

__all__ = [
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
]
