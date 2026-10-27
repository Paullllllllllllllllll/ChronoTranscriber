"""PDF files: page counts, text layers, LLM page images and Tesseract pages.

:class:`PdfPages` renders the pages of a PDF for an LLM request through
``common.images`` and names them ``page_NNNN_pre_processed.jpg``;
:class:`PDFProcessor` detects a text layer, extracts it and renders pages
for Tesseract.
"""

from __future__ import annotations

import asyncio
import logging
import math
import types
from pathlib import Path
from typing import Any, ClassVar

import fitz
from PIL import Image

from chronotranscriber.common.images import PdfPageSource
from chronotranscriber.images.pipeline import ImageProcessor, TesseractSettings
from chronotranscriber.sources.images import (
    PAYLOAD_SHA256_KEY,
    PageImage,
    page_image,
    with_resampling,
)

__all__ = [
    "PDF_PAGE_NAME",
    "PDFProcessor",
    "PdfPages",
    "native_extract_pdf_text",
    "render_pdf_page",
]

logger = logging.getLogger(__name__)

PDF_PAGE_NAME = "page_{number:04d}_pre_processed.jpg"
"""The name of a PDF page image; ``number`` is 1-based."""


class PdfPages(PdfPageSource):
    """The pages of one PDF, named ``page_NNNN_pre_processed.jpg``.

    Rendered pages are downscaled with the filter of
    :func:`~chronotranscriber.sources.images.with_resampling`.
    """

    name_template: ClassVar[str] = PDF_PAGE_NAME

    def __init__(
        self,
        pdf_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        section: str = "",
    ) -> None:
        super().__init__(
            pdf_path,
            with_resampling(img_cfg),
            model_type,
            section=section,
            payload_sha256_key=PAYLOAD_SHA256_KEY,
        )


def render_pdf_page(
    pdf_path: Path, page_index: int, img_cfg: dict[str, Any], model_type: str
) -> PageImage:
    """Render and encode the page at *page_index* (0-based) of a PDF.

    Raises:
        IndexError: The PDF has no such page.
    """
    with PdfPages(pdf_path, img_cfg, model_type) as pages:
        if not 0 <= page_index < len(pages):
            raise IndexError(
                f"Page index {page_index} out of range for {pdf_path.name} "
                f"({len(pages)} pages)"
            )
        return page_image(pages.build_payload(page_index), pages)


def _get_effective_dpi(page: fitz.Page, dpi: int, max_pixels: int) -> int:
    """Return *dpi*, reduced where the rendered page would exceed *max_pixels*."""
    if max_pixels <= 0:
        return dpi
    rect = page.rect
    pixels_at_dpi = (rect.width / 72 * dpi) * (rect.height / 72 * dpi)
    if pixels_at_dpi <= max_pixels:
        return dpi
    effective = max(1, int(dpi * math.sqrt(max_pixels / pixels_at_dpi)))
    logger.info(
        "Page: %.0f MP at %d DPI exceeds limit (%.0f MP); reducing to %d DPI",
        pixels_at_dpi / 1e6,
        dpi,
        max_pixels / 1e6,
        effective,
    )
    return effective


class PDFProcessor:
    """Text-layer detection, text extraction and Tesseract rendering of a PDF.

    *tesseract* holds the run's Tesseract settings; the defaults without.
    """

    def __init__(
        self, pdf_path: Path, tesseract: TesseractSettings | None = None
    ) -> None:
        self.pdf_path = pdf_path
        self.tesseract = tesseract if tesseract is not None else TesseractSettings()
        self.doc: fitz.Document | None = None

    def is_native_pdf(self) -> bool:
        """Check if the PDF is native (searchable) or scanned.

        Returns:
            bool: True if native, False otherwise.
        """
        try:
            with fitz.open(self.pdf_path) as doc:
                return any(page.get_text().strip() for page in doc)
        except Exception as e:  # noqa: BLE001 - any unreadable PDF counts as scanned
            logger.error(f"Error checking if PDF is native: {self.pdf_path}, {e}")
            return False

    def open_pdf(self) -> None:
        """Open the PDF document."""
        try:
            self.doc = fitz.open(self.pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {self.pdf_path}, {e}")
            raise

    def close_pdf(self) -> None:
        """Close the opened PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self) -> PDFProcessor:
        self.open_pdf()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close_pdf()

    async def process_images_for_tesseract(
        self,
        preprocessed_folder: Path,
        target_dpi: int,
        page_indices: list[int] | None = None,
    ) -> list[Path]:
        """Render PDF pages at *target_dpi* and preprocess them for Tesseract.

        Saves lossless PNG or TIFF files and embeds the render resolution if
        the settings ask for it.

        Args:
            preprocessed_folder: Path to save processed images.
            target_dpi: DPI for rendering PDF pages.
            page_indices: Optional list of 0-based page indices to process.
                If None, all pages.
        """
        preprocessed_folder.mkdir(parents=True, exist_ok=True)

        try:
            if self.doc is None:
                self.open_pdf()
            assert self.doc is not None

            settings = self.tesseract
            preproc_cfg = dict(settings.preprocessing)
            suffix = settings.suffix
            total_pages = int(self.doc.page_count)
            pages = (
                page_indices if page_indices is not None else list(range(total_pages))
            )
            out_paths: list[Path] = [
                preprocessed_folder / f"page_{i + 1:04d}_tess_preprocessed{suffix}"
                for i in pages
            ]

            sem = asyncio.Semaphore(max(1, settings.concurrency))

            async def bound_worker(page_index: int, out_path: Path) -> bool:
                async with sem:
                    return await asyncio.to_thread(
                        PDFProcessor._process_single_page_tesseract,
                        self.pdf_path,
                        page_index,
                        target_dpi,
                        out_path,
                        preproc_cfg,
                        settings.embed_dpi,
                        settings.max_pixels,
                    )

            tasks = [
                bound_worker(pg, op) for pg, op in zip(pages, out_paths, strict=False)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, res in enumerate(results):
                pg = pages[idx]
                if isinstance(res, Exception):
                    logger.error(
                        "Error Tesseract-preprocessing page %d from %s: %s",
                        pg + 1,
                        self.pdf_path,
                        res,
                    )
                elif not res:
                    logger.error(
                        "Tesseract-preprocessing page %d did not complete "
                        "successfully.",
                        pg + 1,
                    )
                else:
                    logger.info(
                        "[Tesseract] Processed page %d -> %s dpi=%d",
                        pg + 1,
                        out_paths[idx].name,
                        target_dpi,
                    )

            return [p for p, res in zip(out_paths, results, strict=True) if res is True]

        except Exception as e:
            logger.exception(
                f"Error in Tesseract PDF image processing {self.pdf_path.name}: {e}"
            )
            return []

    @staticmethod
    def _process_single_page_tesseract(
        pdf_path: Path,
        page_index: int,
        target_dpi: int,
        out_path: Path,
        tess_cfg: dict[str, Any],
        embed_dpi: bool,
        max_pixels: int = 0,
    ) -> bool:
        """Render one page, preprocess it for Tesseract and write the image.

        Runs in a worker thread with its own document. Returns True on success.
        """
        try:
            # A document per thread: PyMuPDF documents are not thread-safe.
            with fitz.open(pdf_path) as doc:
                page = doc[page_index]
                effective_dpi = _get_effective_dpi(page, target_dpi, max_pixels)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(effective_dpi / 72, effective_dpi / 72),
                    alpha=False,
                )
                # samples_mv avoids the bytes copy that pix.samples makes; PIL
                # copies the buffer for mode "RGB", so img outlives the pixmap.
                img = Image.frombuffer(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples_mv,
                    "raw",
                    "RGB",
                    0,
                    1,
                )

            processed_img, _diag = ImageProcessor.preprocess_for_tesseract(
                img, tess_cfg
            )
            if embed_dpi:
                # Tag the DPI the page was rendered at, not the requested
                # target: the max-pixels guard lowers it.
                processed_img.save(out_path, dpi=(effective_dpi, effective_dpi))
            else:
                processed_img.save(out_path)
            return True
        except Exception as e:  # noqa: BLE001 - a failed page must not stop the batch
            logger.error(
                f"Worker failed on page {page_index + 1} of {pdf_path.name}: {e}"
            )
            return False

    def page_count(self) -> int:
        """Return the number of pages; the document is closed afterwards."""
        with fitz.open(self.pdf_path) as doc:
            return int(doc.page_count)


def native_extract_pdf_pages(pdf_path: Path, page_indices: list[int]) -> dict[int, str]:
    """Return the PyMuPDF text of each existing page of *page_indices*.

    Pages read before an extraction error are returned; the error is logged.
    """
    pages: dict[int, str] = {}
    try:
        with PDFProcessor(pdf_path) as proc:
            if proc.doc:
                for idx in page_indices:
                    if 0 <= idx < proc.doc.page_count:
                        pages[idx] = proc.doc[idx].get_text()
    except Exception as e:
        logger.exception(f"Failed native PDF extraction on {pdf_path.name}: {e}")
    return pages


def native_extract_pdf_text(
    pdf_path: Path, page_indices: list[int] | None = None
) -> str:
    """Extract text from a native (searchable) PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        page_indices: Optional list of 0-based page indices. If None, all pages.

    Returns the extracted text.
    """
    parts: list[str] = []
    try:
        with PDFProcessor(pdf_path) as proc:
            if proc.doc:
                if page_indices is not None:
                    for idx in page_indices:
                        if 0 <= idx < proc.doc.page_count:
                            parts.append(proc.doc[idx].get_text())
                else:
                    for page in proc.doc:
                        parts.append(page.get_text())
    except Exception as e:
        logger.exception(f"Failed native PDF extraction on {pdf_path.name}: {e}")
    return "".join(parts)
