# modules/pdf_utils.py

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import fitz
import asyncio
from PIL import Image

from modules.config_loader import ConfigLoader
from modules.image_utils import ImageProcessor

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
        # Load JPEG quality from config (fallback to 95)
        try:
            cfg = ConfigLoader()
            cfg.load_configs()
            img_cfg = cfg.get_image_processing_config().get('image_processing', {})
            jpeg_quality = int(img_cfg.get('jpeg_quality', 95))
        except Exception:
            jpeg_quality = 95
        try:
            await asyncio.to_thread(self._extract_images_sync, output_dir, dpi, jpeg_quality)
        except Exception as e:
            logger.error(f"Failed to extract images from PDF: {self.pdf_path}, {e}")
            raise

    def _extract_images_sync(self, output_dir: Path, dpi: int, jpeg_quality: int) -> None:
        """
        Synchronously extract images from the PDF with JPEG compression.
        """
        try:
            with fitz.open(self.pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    try:
                        mat = fitz.Matrix(dpi / 72, dpi / 72)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        # Change format from PNG to JPEG with quality=85
                        image_path = output_dir / f"{self.pdf_path.stem}_page_{page_num}.jpg"
                        pix.save(str(image_path), output="jpeg", jpg_quality=jpeg_quality)
                        logger.info(
                            f"Extracted page {page_num} as image: {image_path} (quality={jpeg_quality})")
                    except Exception as e:
                        logger.error(
                            f"Error extracting page {page_num} from {self.pdf_path}: {e}")
        except Exception as e:
            logger.error(
                f"Failed to extract images from PDF: {self.pdf_path}, {e}")

    async def process_images(self, preprocessed_folder: Path, target_dpi: int) -> List[Path]:
        """
        Extracts images from PDF and directly pre-processes them without saving raw images.

        Returns:
            List[Path]: List of processed image paths.
        """
        preprocessed_folder.mkdir(parents=True, exist_ok=True)

        from modules.utils import console_print

        try:
            # Open PDF if not already open
            if self.doc is None:
                self.open_pdf()

            processed_image_paths = []

            # Process each page directly
            for page_num in range(self.doc.page_count):
                try:
                    # Get the page
                    page = self.doc[page_num]

                    # Create output path for processed image
                    processed_path = preprocessed_folder / f"page_{page_num + 1:04d}_pre_processed.jpg"

                    # Render page to pixmap at target DPI
                    pix = page.get_pixmap(matrix=fitz.Matrix(target_dpi / 72, target_dpi / 72))

                    # Convert pixmap to PIL Image
                    img_data = pix.samples
                    img = Image.frombytes("RGB", [pix.width, pix.height], img_data)

                    # Apply processing directly to the image
                    # Load processing config
                    config_loader = ConfigLoader()
                    config_loader.load_configs()
                    image_cfg = config_loader.get_image_processing_config().get('image_processing', {})

                    # Handle transparency (if needed)
                    if image_cfg.get('handle_transparency', True):
                        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[-1])
                            img = background

                    # Convert to grayscale (if enabled)
                    if image_cfg.get('grayscale_conversion', True):
                        from PIL import ImageOps
                        img = ImageOps.grayscale(img)

                    # Resize based on llm_detail and resize_profile
                    detail = (image_cfg.get('llm_detail', 'high') or 'high')
                    final_img = ImageProcessor.resize_for_detail(img, detail, image_cfg)

                    # Save processed image with configurable JPEG quality
                    jpeg_quality = int(image_cfg.get('jpeg_quality', 95))
                    final_img.save(
                        processed_path,
                        format='JPEG',
                        quality=jpeg_quality
                    )

                    processed_image_paths.append(processed_path)
                    logger.info(
                        f"Processed page {page_num + 1} as image: {processed_path} size={final_img.size} quality={jpeg_quality} detail={detail}")

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1} from {self.pdf_path}: {e}")

            return processed_image_paths

        except Exception as e:
            logger.exception(f"Error extracting and processing images from PDF {self.pdf_path.name}: {e}")
            console_print(f"[ERROR] Failed to process images from {self.pdf_path.name}.")
            return []

    def prepare_output_folder(self, pdf_output_dir: Path) -> Tuple[Path, Path, Path]:
        """
        Prepares the output directory for this PDF file.

        Returns:
            Tuple[Path, Path, Path]: A tuple containing:
              - parent_folder: Directory for this PDF's outputs.
              - output_txt_path: File path for the final transcription text.
              - temp_jsonl_path: File path for temporary JSONL records.
        """
        safe_stem = self.pdf_path.stem.strip()
        parent_folder = pdf_output_dir / safe_stem
        parent_folder.mkdir(parents=True, exist_ok=True)
        output_txt_path = parent_folder / f"{safe_stem}_transcription.txt"
        temp_jsonl_path = parent_folder / f"{safe_stem}_transcription.jsonl"
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

        # If the PDF is native, method "1" = "native"
        if self.is_native_pdf():
            valid_methods["1"] = "native"
            method_options.append("1. Native text extraction")

        # Method "2" = Tesseract
        valid_methods["2"] = "tesseract"
        method_options.append("2. Tesseract")

        # Method "3" = GPT
        valid_methods["3"] = "gpt"
        method_options.append("3. GPT")

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
