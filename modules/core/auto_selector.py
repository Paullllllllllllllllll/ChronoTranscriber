"""Auto mode file detection and method selection logic.

This module inspects a directory and automatically determines the best transcription
method for each discovered file based on configuration and file characteristics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from modules.infra.logger import setup_logger
from modules.processing.pdf_utils import PDFProcessor
from modules.processing.image_utils import SUPPORTED_IMAGE_EXTENSIONS

logger = setup_logger(__name__)


@dataclass
class FileDecision:
    """Represents a decision for how to process a single file."""
    file_path: Path
    file_type: str  # "pdf", "image", "epub"
    method: str  # "native", "tesseract", "gpt"
    reason: str  # Explanation for the decision


class AutoSelector:
    """Automatically selects transcription methods based on file inspection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize auto selector with configuration.
        
        Args:
            config: paths_config dictionary with auto_mode settings
        """
        self.config = config
        general = config.get("general", {})
        
        # Load auto mode settings with defaults
        self.pdf_use_ocr_for_scanned = general.get("auto_mode_pdf_use_ocr_for_scanned", True)
        self.pdf_use_ocr_for_searchable = general.get("auto_mode_pdf_use_ocr_for_searchable", False)
        self.pdf_ocr_method = general.get("auto_mode_pdf_ocr_method", "tesseract")
        self.image_ocr_method = general.get("auto_mode_image_ocr_method", "tesseract")
        
        # Check GPT availability
        self.gpt_available = bool(os.getenv("OPENAI_API_KEY"))
        
        logger.info(
            f"AutoSelector initialized: pdf_use_ocr_for_scanned={self.pdf_use_ocr_for_scanned}, "
            f"pdf_use_ocr_for_searchable={self.pdf_use_ocr_for_searchable}, "
            f"pdf_ocr_method={self.pdf_ocr_method}, image_ocr_method={self.image_ocr_method}, "
            f"gpt_available={self.gpt_available}"
        )
    
    def scan_directory(self, input_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
        """Scan directory and categorize files.
        
        Args:
            input_dir: Directory to scan
            
        Returns:
            Tuple of (pdf_files, image_files, epub_files)
        """
        pdfs = []
        images = []
        epubs = []
        
        if not input_dir.exists() or not input_dir.is_dir():
            logger.warning(f"Input directory does not exist or is not a directory: {input_dir}")
            return pdfs, images, epubs
        
        # Scan all files (non-recursive for simplicity)
        for item in input_dir.iterdir():
            if item.is_file():
                suffix = item.suffix.lower()
                if suffix == ".pdf":
                    pdfs.append(item)
                elif suffix == ".epub":
                    epubs.append(item)
                elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
                    images.append(item)
            elif item.is_dir():
                # Check if it's an image folder (contains images)
                folder_images = [
                    f for f in item.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ]
                if folder_images:
                    images.append(item)  # Treat folder as a single image collection
        
        logger.info(
            f"Scanned {input_dir}: found {len(pdfs)} PDFs, "
            f"{len(images)} image items, {len(epubs)} EPUBs"
        )
        
        return pdfs, images, epubs
    
    def decide_pdf_method(self, pdf_path: Path) -> Tuple[str, str]:
        """Decide transcription method for a PDF.
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (method, reason)
        """

        processor = PDFProcessor(pdf_path)

        # Check if PDF has native text
        is_native = processor.is_native_pdf()

        if is_native:
            if not self.pdf_use_ocr_for_searchable:
                return "native", "PDF contains searchable text"
            force_context = "Searchable PDF forced to OCR"
        else:
            if not self.pdf_use_ocr_for_scanned:
                return "native", "Non-searchable PDF detected, OCR forcing disabled"
            force_context = "Non-searchable PDF"

        method = self.pdf_ocr_method

        if method == "gpt":
            if self.gpt_available:
                return "gpt", f"{force_context} using GPT OCR"
            else:
                logger.warning(
                    f"GPT OCR requested but API key not available for {pdf_path.name}. "
                    "Falling back to Tesseract."
                )
                return "tesseract", f"{force_context}, GPT unavailable, using Tesseract fallback"
        else:
            return "tesseract", f"{force_context} using {method} OCR"
    
    def decide_image_method(self, image_path: Path) -> Tuple[str, str]:
        """Decide transcription method for an image or image folder.
        
        Args:
            image_path: Path to an image file or directory containing images
        Returns:
            Tuple of (method, reason)
        """
        # Check preferred method availability
        if self.image_ocr_method == "gpt":
            if self.gpt_available:
                return "gpt", "Image OCR using GPT"
            else:
                logger.warning(
                    f"GPT OCR requested but API key not available for {image_path.name}. "
                    "Falling back to Tesseract."
                )
                return "tesseract", "GPT unavailable, using Tesseract fallback"
        else:
            return "tesseract", f"Image OCR using {self.image_ocr_method}"
    
    def decide_epub_method(self, epub_path: Path) -> Tuple[str, str]:
        """Decide transcription method for an EPUB.
        
        Args:
            epub_path: Path to EPUB file
            
        Returns:
            Tuple of (method, reason)
        """
        # EPUBs currently only support native extraction
        return "native", "EPUB native text extraction"
    
    def create_decisions(self, input_dir: Path) -> List[FileDecision]:
        """Create processing decisions for all files in directory.
        
        Args:
            input_dir: Input directory to scan
            
        Returns:
            List of FileDecision objects
        """
        decisions = []
        
        pdfs, images, epubs = self.scan_directory(input_dir)
        
        # Process PDFs
        for pdf in pdfs:
            method, reason = self.decide_pdf_method(pdf)
            decisions.append(FileDecision(
                file_path=pdf,
                file_type="pdf",
                method=method,
                reason=reason
            ))
        
        # Process images
        for img in images:
            method, reason = self.decide_image_method(img)
            file_type = "image_folder" if img.is_dir() else "image"
            decisions.append(FileDecision(
                file_path=img,
                file_type=file_type,
                method=method,
                reason=reason
            ))
        
        # Process EPUBs
        for epub in epubs:
            method, reason = self.decide_epub_method(epub)
            decisions.append(FileDecision(
                file_path=epub,
                file_type="epub",
                method=method,
                reason=reason
            ))
        
        logger.info(f"Created {len(decisions)} processing decisions for auto mode")
        
        return decisions
    
    def print_decision_summary(self, decisions: List[FileDecision]) -> None:
        """Print a summary of all decisions to console.
        
        Args:
            decisions: List of FileDecision objects
        """
        from modules.ui.prompts import print_header, print_info, print_separator, ui_print, PromptStyle

        print_header("AUTO MODE DECISIONS", f"{len(decisions)} files discovered")
        
        if not decisions:
            print_info("No files found to process.")
            return
        
        # Group by method
        by_method: Dict[str, List[FileDecision]] = {}
        for decision in decisions:
            method = decision.method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(decision)
        
        # Display summary by method
        for method, items in by_method.items():
            ui_print(f"\n  {method.upper()} ({len(items)} files):", PromptStyle.HIGHLIGHT)
            print_separator(PromptStyle.LIGHT_LINE, 80)
            for i, item in enumerate(items[:5], 1):  # Show first 5
                ui_print(f"    {i}. {item.file_path.name} — {item.reason}", PromptStyle.INFO)
            if len(items) > 5:
                ui_print(f"    ... and {len(items) - 5} more", PromptStyle.DIM)
        
        print_separator(PromptStyle.SINGLE_LINE, 80)
        ui_print("")
