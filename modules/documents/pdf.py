# modules/pdf_utils.py

from __future__ import annotations

import asyncio
import math
import types
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from modules.config.service import get_config_service
from modules.images.pipeline import ImageProcessor
from modules.infra.logger import setup_logger
from modules.infra.paths import create_safe_directory_name, create_safe_filename

logger = setup_logger(__name__)


def _get_effective_dpi(page: fitz.Page, dpi: int, max_pixels: int) -> int:
    """Return DPI reduced so the rendered page stays within max_pixels,
    or dpi unchanged."""
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
    """
    A class for handling PDF processing tasks.
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self.doc: fitz.Document | None = None

    def is_native_pdf(self) -> bool:
        """
        Check if the PDF is native (searchable) or scanned.

        Returns:
            bool: True if native, False otherwise.
        """
        try:
            with fitz.open(self.pdf_path) as doc:
                # any() short-circuits: one page with text is enough to classify
                # the PDF as searchable; no need to read every page.
                return any(page.get_text().strip() for page in doc)
        except Exception as e:
            logger.error(f"Error checking if PDF is native: {self.pdf_path}, {e}")
            return False

    def open_pdf(self) -> None:
        """
        Open the PDF document.
        """
        try:
            self.doc = fitz.open(self.pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {self.pdf_path}, {e}")
            raise

    def close_pdf(self) -> None:
        """
        Close the opened PDF document.
        """
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
        """
        Render PDF pages at target_dpi and preprocess for Tesseract OCR.
        Saves lossless PNG/TIFF, preserves resolution, embeds DPI metadata if enabled.

        Args:
            preprocessed_folder: Path to save processed images.
            target_dpi: DPI for rendering PDF pages.
            page_indices: Optional list of 0-based page indices to process.
                If None, all pages.
        """
        preprocessed_folder.mkdir(parents=True, exist_ok=True)

        from modules.ui import print_error

        try:
            # Ensure PDF is open
            if self.doc is None:
                self.open_pdf()
            assert self.doc is not None

            # Load Tesseract preprocessing config
            config_service = get_config_service()
            img_cfg = config_service.get_image_processing_config()
            tip_cfg = img_cfg.get("tesseract_image_processing", {})
            preproc_cfg = tip_cfg.get("preprocessing", {})
            output_format = str(preproc_cfg.get("output_format", "png")).lower()
            embed_dpi = bool(preproc_cfg.get("embed_dpi_metadata", True))
            max_pixels = int(img_cfg.get("max_pixels_per_page", 0))
            # Concurrency settings
            conc_cfg = config_service.get_concurrency_config()
            img_conc = conc_cfg.get("concurrency", {}).get("image_processing", {})
            concurrency_limit = int(img_conc.get("concurrency_limit", 4))

            # Determine which pages to process
            suffix = ".png" if output_format == "png" else ".tif"
            total_pages = int(self.doc.page_count)
            pages = (
                page_indices if page_indices is not None else list(range(total_pages))
            )
            out_paths: list[Path] = [
                preprocessed_folder / f"page_{i + 1:04d}_tess_preprocessed{suffix}"
                for i in pages
            ]

            sem = asyncio.Semaphore(max(1, concurrency_limit))

            async def bound_worker(page_index: int, out_path: Path) -> bool:
                async with sem:
                    return await asyncio.to_thread(
                        PDFProcessor._process_single_page_tesseract,
                        self.pdf_path,
                        page_index,
                        target_dpi,
                        out_path,
                        preproc_cfg,
                        embed_dpi,
                        max_pixels,
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

            return [p for p in out_paths if p.exists()]

        except Exception as e:
            logger.exception(
                f"Error in Tesseract PDF image processing {self.pdf_path.name}: {e}"
            )
            print_error(f"Failed Tesseract preprocessing for {self.pdf_path.name}.")
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
        """
        Thread worker: open PDF, render a single page, preprocess for
        Tesseract, and write output. Returns True on success.
        """
        try:
            # Open a local document instance to avoid PyMuPDF thread-safety issues
            with fitz.open(pdf_path) as doc:
                page = doc[page_index]
                effective_dpi = _get_effective_dpi(page, target_dpi, max_pixels)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(effective_dpi / 72, effective_dpi / 72),
                    alpha=False,
                )
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            processed_img, _diag = ImageProcessor.preprocess_for_tesseract(
                img, tess_cfg
            )
            if embed_dpi:
                # Embed the DPI the page was actually rendered at, not the
                # requested target: a max-pixels downscale lowers the effective
                # DPI, and tagging the target would misreport it to Tesseract
                # (B15).
                processed_img.save(out_path, dpi=(effective_dpi, effective_dpi))
            else:
                processed_img.save(out_path)
            return True
        except Exception as e:
            logger.error(
                f"Worker failed on page {page_index + 1} of {pdf_path.name}: {e}"
            )
            return False

    def prepare_output_folder(
        self,
        pdf_output_dir: Path,
        relative_key: str | None = None,
    ) -> tuple[Path, Path, Path]:
        """
        Prepares the output directory for this PDF file.

        Returns:
            Tuple[Path, Path, Path]: A tuple containing:
              - parent_folder: Directory for this PDF's outputs.
              - output_txt_path: File path for the final transcription text.
              - temp_jsonl_path: File path for temporary JSONL records.
        """
        key = relative_key if relative_key is not None else self.pdf_path.stem
        safe_dir_name = create_safe_directory_name(key)

        # Create parent folder with safe directory name
        parent_folder = pdf_output_dir / safe_dir_name
        parent_folder.mkdir(parents=True, exist_ok=True)

        # Create safe filenames (truncated with hash if needed,
        # considering full path length).
        # No _transcription suffix to keep filenames shorter.
        output_txt_name = create_safe_filename(
            self.pdf_path.stem, ".txt", parent_folder
        )
        temp_jsonl_name = create_safe_filename(
            self.pdf_path.stem, ".jsonl", parent_folder
        )

        output_txt_path = parent_folder / output_txt_name
        temp_jsonl_path = parent_folder / temp_jsonl_name

        if not temp_jsonl_path.exists():
            temp_jsonl_path.touch()

        return parent_folder, output_txt_path, temp_jsonl_path


def native_extract_pdf_text(
    pdf_path: Path, page_indices: list[int] | None = None
) -> str:
    """
    Extract text from a native (searchable) PDF using PyMuPDF.

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
