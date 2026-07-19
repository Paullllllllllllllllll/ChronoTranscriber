"""Auto mode file detection and method selection logic.

This module inspects a directory and automatically determines the best transcription
method for each discovered file based on configuration and file characteristics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.config.constants import SUPPORTED_MOBI_EXTENSIONS
from modules.documents.pdf import PDFProcessor
from modules.images.pipeline import SUPPORTED_IMAGE_EXTENSIONS
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class FileDecision:
    """Represents a decision for how to process a single file."""

    file_path: Path
    file_type: str  # "pdf", "image", "epub", "mobi"
    method: str  # "native", "tesseract", "gpt"
    reason: str  # Explanation for the decision


class AutoSelector:
    """Automatically selects transcription methods based on file inspection."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize auto selector with configuration.

        Args:
            config: paths_config dictionary with auto_mode settings
        """
        self.config = config
        general = config.get("general", {})

        # Load auto mode settings with defaults
        self.pdf_use_ocr_for_scanned = general.get(
            "auto_mode_pdf_use_ocr_for_scanned", True
        )
        self.pdf_use_ocr_for_searchable = general.get(
            "auto_mode_pdf_use_ocr_for_searchable", False
        )
        self.pdf_ocr_method = general.get("auto_mode_pdf_ocr_method", "tesseract")
        self.image_ocr_method = general.get("auto_mode_image_ocr_method", "tesseract")

        # Check LLM ("gpt" method) availability for the *configured* provider,
        # not OpenAI specifically. The auto pipeline's "gpt" OCR method targets
        # whichever provider model_config.yaml selects (OpenAI, Anthropic, Google,
        # OpenRouter, or a custom endpoint).
        self.gpt_available = self._resolve_gpt_available()

        logger.info(
            "AutoSelector initialized: "
            "pdf_use_ocr_for_scanned=%s, "
            "pdf_use_ocr_for_searchable=%s, "
            "pdf_ocr_method=%s, image_ocr_method=%s, "
            "gpt_available=%s",
            self.pdf_use_ocr_for_scanned,
            self.pdf_use_ocr_for_searchable,
            self.pdf_ocr_method,
            self.image_ocr_method,
            self.gpt_available,
        )

    def _resolve_gpt_available(self) -> bool:
        """Return True when the configured LLM provider has a usable API key.

        Resolves the provider from ``model_config.yaml`` (explicit ``provider``
        or auto-detected from the model name) and checks its key through the same
        factory helpers the provider pipeline uses, honoring the optional
        ``api_keys_config.yaml`` env-var remap. Falls back to the legacy
        ``OPENAI_API_KEY`` check if resolution fails for any reason.
        """
        try:
            from modules.config.service import get_config_service
            from modules.llm.providers.factory import (
                ProviderType,
                detect_provider_from_model,
                resolve_api_key_optional,
            )

            model_config = get_config_service().get_model_config() or {}
            tm = model_config.get("transcription_model", {}) or {}
            provider = tm.get("provider")
            model = tm.get("name")

            provider_type: ProviderType
            if provider:
                try:
                    provider_type = ProviderType(str(provider).lower())
                except ValueError:
                    provider_type = detect_provider_from_model(str(model or ""))
            elif model:
                provider_type = detect_provider_from_model(str(model))
            else:
                provider_type = ProviderType.OPENAI

            return resolve_api_key_optional(provider_type) is not None
        except Exception as e:  # defensive: never let detection break auto mode
            logger.debug(f"Could not resolve configured provider key: {e}")
            return bool(os.getenv("OPENAI_API_KEY"))

    def scan_directory(
        self, input_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
        """Scan directory and categorize files.

        Args:
            input_dir: Directory to scan

        Returns:
            Tuple of (pdf_files, image_files, epub_files, mobi_files)
        """
        pdfs: list[Path] = []
        images: list[Path] = []
        epubs: list[Path] = []
        mobis: list[Path] = []

        if not input_dir.exists() or not input_dir.is_dir():
            logger.warning(
                f"Input directory does not exist or is not a directory: {input_dir}"
            )
            return pdfs, images, epubs, mobis

        # Scan all files (non-recursive for simplicity).
        # os.scandir() is used instead of Path.iterdir() so that DirEntry
        # caches is_file()/is_dir() from the OS scan and avoids a separate
        # stat() syscall per entry (significant on large directories).
        loose_images = 0
        with os.scandir(input_dir) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    suffix = os.path.splitext(entry.name)[1].lower()
                    if suffix == ".pdf":
                        pdfs.append(Path(entry.path))
                    elif suffix == ".epub":
                        epubs.append(Path(entry.path))
                    elif suffix in SUPPORTED_MOBI_EXTENSIONS:
                        mobis.append(Path(entry.path))
                    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
                        # Loose image files have no pipeline in auto mode: the
                        # router cannot transcribe a bare image, so emitting a
                        # decision for one only produced a per-file "unknown
                        # type" warning while the item counted as processed.
                        # Skip them here and surface a single actionable notice.
                        loose_images += 1
                elif entry.is_dir(follow_symlinks=False):
                    # Check if it's an image folder (contains images)
                    has_images = False
                    with os.scandir(entry.path) as inner:
                        for fe in inner:
                            if (
                                fe.is_file(follow_symlinks=False)
                                and os.path.splitext(fe.name)[1].lower()
                                in SUPPORTED_IMAGE_EXTENSIONS
                            ):
                                has_images = True
                                break
                    if has_images:
                        images.append(
                            Path(entry.path)
                        )  # Treat folder as a single image collection

        if loose_images:
            from modules.ui import print_warning

            logger.warning(
                "Skipped %d loose image file(s) in %s: unsupported in auto mode.",
                loose_images,
                input_dir,
            )
            print_warning(
                "loose image files are not supported in auto mode; place them "
                "in a subfolder to process the folder."
            )

        logger.info(
            f"Scanned {input_dir}: found {len(pdfs)} PDFs, "
            f"{len(images)} image items, {len(epubs)} EPUBs, {len(mobis)} MOBIs"
        )

        return pdfs, images, epubs, mobis

    def decide_pdf_method(self, pdf_path: Path) -> tuple[str, str]:
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
                return (
                    "tesseract",
                    f"{force_context}, GPT unavailable, using Tesseract fallback",
                )
        else:
            return "tesseract", f"{force_context} using {method} OCR"

    def decide_image_method(self, image_path: Path) -> tuple[str, str]:
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
                    "GPT OCR requested but API key not available for %s. "
                    "Falling back to Tesseract.",
                    image_path.name,
                )
                return "tesseract", "GPT unavailable, using Tesseract fallback"
        else:
            return "tesseract", f"Image OCR using {self.image_ocr_method}"

    def decide_epub_method(self, epub_path: Path) -> tuple[str, str]:
        """Decide transcription method for an EPUB.

        Args:
            epub_path: Path to EPUB file

        Returns:
            Tuple of (method, reason)
        """
        # EPUBs currently only support native extraction
        return "native", "EPUB native text extraction"

    def decide_mobi_method(self, mobi_path: Path) -> tuple[str, str]:
        """Decide transcription method for a MOBI/Kindle file.

        Args:
            mobi_path: Path to MOBI file

        Returns:
            Tuple of (method, reason)
        """
        # MOBIs are unpacked and extracted natively
        return "native", "MOBI native text extraction (via unpack)"

    def create_decisions(self, input_dir: Path) -> list[FileDecision]:
        """Create processing decisions for all files in directory.

        Args:
            input_dir: Input directory to scan

        Returns:
            List of FileDecision objects
        """
        decisions = []

        pdfs, images, epubs, mobis = self.scan_directory(input_dir)

        # Process PDFs
        for pdf in pdfs:
            method, reason = self.decide_pdf_method(pdf)
            decisions.append(
                FileDecision(
                    file_path=pdf, file_type="pdf", method=method, reason=reason
                )
            )

        # Process images
        for img in images:
            method, reason = self.decide_image_method(img)
            file_type = "image_folder" if img.is_dir() else "image"
            decisions.append(
                FileDecision(
                    file_path=img, file_type=file_type, method=method, reason=reason
                )
            )

        # Process EPUBs
        for epub in epubs:
            method, reason = self.decide_epub_method(epub)
            decisions.append(
                FileDecision(
                    file_path=epub, file_type="epub", method=method, reason=reason
                )
            )

        # Process MOBIs
        for mobi in mobis:
            method, reason = self.decide_mobi_method(mobi)
            decisions.append(
                FileDecision(
                    file_path=mobi, file_type="mobi", method=method, reason=reason
                )
            )

        logger.info(f"Created {len(decisions)} processing decisions for auto mode")

        return decisions

    def print_decision_summary(self, decisions: list[FileDecision]) -> None:
        """Print a summary of all decisions to console.

        Args:
            decisions: List of FileDecision objects
        """
        from modules.ui.prompts import (
            PromptStyle,
            print_header,
            print_info,
            print_separator,
            ui_print,
        )

        print_header("AUTO MODE DECISIONS", f"{len(decisions)} files discovered")

        if not decisions:
            print_info("No files found to process.")
            return

        # Group by method
        by_method: dict[str, list[FileDecision]] = {}
        for decision in decisions:
            method = decision.method
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(decision)

        # Display summary by method
        for method, items in by_method.items():
            ui_print(
                f"\n  {method.upper()} ({len(items)} files):", PromptStyle.HIGHLIGHT
            )
            print_separator(PromptStyle.LIGHT_LINE, 80)
            for i, item in enumerate(items[:5], 1):  # Show first 5
                ui_print(
                    f"    {i}. {item.file_path.name} — {item.reason}", PromptStyle.INFO
                )
            if len(items) > 5:
                ui_print(f"    ... and {len(items) - 5} more", PromptStyle.DIM)

        print_separator(PromptStyle.SINGLE_LINE, 80)
        ui_print("")
