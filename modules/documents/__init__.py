"""Document loaders and classification.

Absorbs the former modules.processing.{pdf_utils,epub_utils,mobi_utils},
modules.documents.auto_selector, and modules.documents.page_range into a single
deep module. Deals with opening PDFs, EPUBs, and MOBIs, classifying
files for auto mode, and page-range selection.
"""

from modules.documents.auto_selector import AutoSelector, FileDecision
from modules.documents.epub import EPUBProcessor, EPUBTextExtraction
from modules.documents.mobi import MOBIProcessor, MOBITextExtraction
from modules.documents.page_range import PageRange
from modules.documents.pdf import PDFProcessor, native_extract_pdf_text

__all__ = [
    "PDFProcessor",
    "native_extract_pdf_text",
    "EPUBProcessor",
    "EPUBTextExtraction",
    "MOBIProcessor",
    "MOBITextExtraction",
    "AutoSelector",
    "FileDecision",
    "PageRange",
]
