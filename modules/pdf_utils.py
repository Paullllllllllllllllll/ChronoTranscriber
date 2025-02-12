# modules/pdf_utils.py

import logging
from pathlib import Path
from typing import Optional
import fitz
import asyncio

logger = logging.getLogger(__name__)


class PDFProcessor:
	"""
    A class for handling PDF processing tasks.
    """

	def __init__(self, pdf_path: Path) -> None:
		self.pdf_path = pdf_path
		self.doc: Optional[fitz.Document] = None

	def is_native_pdf(self) -> bool:
		"""
        Check if the PDF is native (searchable) or scanned.

        Returns:
            bool: True if native, False otherwise.
        """
		try:
			with fitz.open(self.pdf_path) as doc:
				text = ""
				for page in doc:
					text += page.get_text()
				return bool(text.strip())
		except Exception as e:
			logger.error(
				f"Error checking if PDF is native: {self.pdf_path}, {e}")
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

	async def extract_images(self, output_dir: Path, dpi: int = 300) -> None:
		"""
        Extract all pages of the PDF as images at the specified DPI.

        Parameters:
            output_dir (Path): Directory to save extracted images.
            dpi (int): DPI for rendering pages.
        """
		output_dir.mkdir(parents=True, exist_ok=True)
		try:
			await asyncio.to_thread(self._extract_images_sync, output_dir, dpi)
		except Exception as e:
			logger.error(
				f"Failed to extract images from PDF: {self.pdf_path}, {e}")

	def _extract_images_sync(self, output_dir: Path, dpi: int) -> None:
		"""
        Synchronously extract images from the PDF.
        """
		try:
			with fitz.open(self.pdf_path) as doc:
				for page_num, page in enumerate(doc, start=1):
					try:
						mat = fitz.Matrix(dpi / 72, dpi / 72)
						pix = page.get_pixmap(matrix=mat, alpha=False)
						image_path = output_dir / f"{self.pdf_path.stem}_page_{page_num}.png"
						pix.save(str(image_path))
						logger.info(
							f"Extracted page {page_num} as image: {image_path}")
					except Exception as e:
						logger.error(
							f"Error extracting page {page_num} from {self.pdf_path}: {e}")
		except Exception as e:
			logger.error(
				f"Failed to extract images from PDF: {self.pdf_path}, {e}")
