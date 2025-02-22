# modules/pdf_utils.py

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
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
            logger.error(f"Failed to extract images from PDF: {self.pdf_path}, {e}")
            raise

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
                        logger.info(f"Extracted page {page_num} as image: {image_path}")
                    except Exception as e:
                        logger.error(f"Error extracting page {page_num} from {self.pdf_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to extract images from PDF: {self.pdf_path}, {e}")

    async def process_images(self, raw_images_folder: Path, preprocessed_folder: Path, target_dpi: int) -> List[Path]:
        """
        Extracts and pre-processes images from the PDF.

        Returns:
            List[Path]: List of processed image paths.
        """
        raw_images_folder.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(self._extract_images_sync, raw_images_folder, target_dpi)
        except Exception as e:
            logger.exception(f"Error extracting images from PDF {self.pdf_path.name}: {e}")
            from modules.utils import console_print
            console_print(f"[ERROR] Failed to extract images from {self.pdf_path.name}.")
            return []
        # Import ImageProcessor and supported extensions from image_utils
        from modules.image_utils import ImageProcessor, SUPPORTED_IMAGE_EXTENSIONS
        image_files: List[Path] = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(list(raw_images_folder.glob(f"*{ext}")))
        if not image_files:
            from modules.utils import console_print
            console_print(f"[WARN] No images extracted from {self.pdf_path.name}.")
            return []
        processed_image_paths = [preprocessed_folder / f"{img.stem}_pre_processed{img.suffix}" for img in image_files]
        # Use the multiprocessing helper from ImageProcessor
        await asyncio.to_thread(ImageProcessor.process_images_multiprocessing, image_files, processed_image_paths)
        return [p for p in processed_image_paths if p.exists()]

    def prepare_output_folder(self, pdf_output_dir: Path) -> Tuple[Path, Path, Path]:
        """
        Prepares the output directory for this PDF file.

        Returns:
            Tuple[Path, Path, Path]: A tuple containing:
              - parent_folder: Directory for this PDF's outputs.
              - output_txt_path: File path for the final transcription text.
              - temp_jsonl_path: File path for temporary JSONL records.
        """
        parent_folder = pdf_output_dir / self.pdf_path.stem
        parent_folder.mkdir(parents=True, exist_ok=True)
        output_txt_path = parent_folder / f"{self.pdf_path.stem}_transcription.txt"
        temp_jsonl_path = parent_folder / f"{self.pdf_path.stem}_transcription.jsonl"
        if not temp_jsonl_path.exists():
            temp_jsonl_path.touch()
        return parent_folder, output_txt_path, temp_jsonl_path

    def choose_transcription_method(self) -> Tuple[Dict[str, str], List[str]]:
        """
        Determines the valid transcription methods based on whether the PDF is native.

        Returns:
            Tuple[Dict[str, str], List[str]]: A tuple containing:
              - valid_methods: A dictionary mapping user choices to method identifiers.
              - method_options: A list of descriptive options to display.
        """
        valid_methods = {}
        method_options = []
        if self.is_native_pdf():
            valid_methods["1"] = "native"
            method_options.append("1. Native text extraction")
            valid_methods["2"] = "tesseract"
            method_options.append("2. Tesseract")
            valid_methods["3"] = "gpt"
            method_options.append("3. GPT")
        else:
            valid_methods["1"] = "tesseract"
            method_options.append("1. Tesseract")
            valid_methods["2"] = "gpt"
            method_options.append("2. GPT")
        return valid_methods, method_options

def native_extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a native (searchable) PDF using PyMuPDF.
    Returns the extracted text.
    """
    pdf_processor = PDFProcessor(pdf_path)
    text = ""
    try:
        pdf_processor.open_pdf()
        if pdf_processor.doc:
            for page in pdf_processor.doc:
                text += page.get_text()
        pdf_processor.close_pdf()
    except Exception as e:
        logger.exception(f"Failed native PDF extraction on {pdf_path.name}: {e}")
    return text
