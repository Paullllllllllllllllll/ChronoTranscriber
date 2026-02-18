# modules/workflow.py
from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import aiofiles
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

from modules.infra.logger import setup_logger
from modules.ui.core import UserConfiguration
from modules.ui import print_info, print_warning, print_error, print_success
from modules.processing.pdf_utils import PDFProcessor, native_extract_pdf_text
from modules.processing.epub_utils import EPUBProcessor
from modules.processing.mobi_utils import MOBIProcessor
from modules.processing.image_utils import ImageProcessor
from modules.processing.tesseract_utils import (
    configure_tesseract_executable,
    ensure_tesseract_available,
)
from modules.processing.postprocess import postprocess_transcription
from modules.core.token_guard import check_and_wait_for_token_limit
from modules.core.resume import ResumeChecker, ProcessingState
from modules.core.safe_paths import create_safe_filename
from modules.core.path_config import PathConfig
from modules.core.batch_submission import submit_batch
from modules.core.transcription_pipeline import (
    run_transcription_pipeline,
    write_output_from_jsonl,
)

logger = setup_logger(__name__)


class TransientFileTracker:
    """Tracks transient files created during processing for cleanup on interruption.
    
    This class ensures that temporary files (.jsonl) and preprocessed image folders
    are cleaned up when processing is interrupted (e.g., by token limit exit or Ctrl+C).
    """
    
    def __init__(self) -> None:
        self._jsonl_files: List[Tuple[Path, str]] = []  # (path, method)
        self._preprocessed_folders: List[Tuple[Path, str]] = []  # (path, source_name)
        self._processing_settings: Dict[str, Any] = {}
        self._use_batch_processing: bool = False
    
    def configure(
        self,
        processing_settings: Dict[str, Any],
        use_batch_processing: bool = False
    ) -> None:
        """Configure cleanup behavior based on settings."""
        self._processing_settings = processing_settings
        self._use_batch_processing = use_batch_processing
    
    def register_jsonl(self, path: Path, method: str) -> None:
        """Register a JSONL file for potential cleanup."""
        self._jsonl_files.append((path, method))
    
    def register_preprocessed_folder(self, path: Path, source_name: str) -> None:
        """Register a preprocessed images folder for potential cleanup."""
        self._preprocessed_folders.append((path, source_name))
    
    def mark_jsonl_complete(self, path: Path) -> None:
        """Mark a JSONL file as successfully processed (remove from tracking)."""
        self._jsonl_files = [(p, m) for p, m in self._jsonl_files if p != path]
    
    def mark_preprocessed_complete(self, path: Path) -> None:
        """Mark a preprocessed folder as successfully processed (remove from tracking)."""
        self._preprocessed_folders = [(p, n) for p, n in self._preprocessed_folders if p != path]
    
    def cleanup_pending(self) -> None:
        """Clean up all pending transient files that weren't successfully processed.
        
        This is called when processing exits prematurely due to interruption.
        """
        # Clean up JSONL files
        for jsonl_path, method in self._jsonl_files:
            is_batch = method == "gpt" and self._use_batch_processing
            retain = self._processing_settings.get("retain_temporary_jsonl", True)
            if not retain and not is_batch:
                try:
                    if jsonl_path.exists():
                        jsonl_path.unlink()
                        logger.info(f"Cleaned up interrupted JSONL: {jsonl_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up JSONL {jsonl_path}: {e}")
        
        # Clean up preprocessed folders
        keep_preprocessed = self._processing_settings.get("keep_preprocessed_images", True)
        if not keep_preprocessed:
            for folder_path, source_name in self._preprocessed_folders:
                try:
                    if folder_path.exists():
                        shutil.rmtree(folder_path, ignore_errors=True)
                        logger.info(f"Cleaned up interrupted preprocessed folder for {source_name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up preprocessed folder {folder_path}: {e}")
        
        # Clear tracking lists
        self._jsonl_files.clear()
        self._preprocessed_folders.clear()
    
    def clear(self) -> None:
        """Clear all tracked files without cleanup (for successful completion)."""
        self._jsonl_files.clear()
        self._preprocessed_folders.clear()


class WorkflowManager:
    """
    Manages the processing workflow for PDFs and images based on user configuration.
    """

    def __init__(self,
                 user_config: UserConfiguration,
                 paths_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 concurrency_config: Dict[str, Any],
                 image_processing_config: Dict[str, Any]
                 ):
        self.user_config = user_config
        self.paths_config = paths_config
        self.model_config = model_config
        self.concurrency_config = concurrency_config
        self.image_processing_config = image_processing_config
        self.processing_settings = paths_config.get("general", {})

        # Configure Tesseract executable if provided
        configure_tesseract_executable(image_processing_config)
        self.ocr_config = (
            image_processing_config
            .get('tesseract_image_processing', {})
            .get('ocr', {})
        )
        
        # Load post-processing configuration from image_processing_config
        self.postprocessing_config = image_processing_config.get("postprocessing", {})

        # Resolve output directories via PathConfig
        pc = PathConfig.from_paths_config(paths_config)
        self.use_input_as_output = pc.use_input_as_output
        self.pdf_output_dir = pc.pdf_output_dir
        self.image_output_dir = pc.image_output_dir
        self.epub_output_dir = pc.epub_output_dir
        self.mobi_output_dir = pc.mobi_output_dir
        pc.ensure_output_dirs()

        # Resume checker
        self.resume_mode = user_config.resume_mode
        self.resume_checker = ResumeChecker(
            resume_mode=self.resume_mode,
            paths_config=paths_config,
            use_input_as_output=self.use_input_as_output,
            pdf_output_dir=self.pdf_output_dir,
            image_output_dir=self.image_output_dir,
            epub_output_dir=self.epub_output_dir,
            mobi_output_dir=self.mobi_output_dir,
        )

        # When resume mode is active, preserve JSONL files so page-level
        # resume works across runs.
        if self.resume_mode == "skip":
            self.processing_settings = dict(self.processing_settings)
            self.processing_settings["retain_temporary_jsonl"] = True

        # Initialize transient file tracker for cleanup on interruption
        self._transient_tracker = TransientFileTracker()
        self._transient_tracker.configure(
            self.processing_settings,
            use_batch_processing=user_config.use_batch_processing
        )

    async def _route_auto_item(self, item: Path, transcriber: Optional[Any]) -> None:
        """Route a single item to the correct processor based on its actual type.

        Used by auto mode and as a fallback when processing_type is unknown.
        """
        from modules.config.constants import SUPPORTED_MOBI_EXTENSIONS

        if item.is_dir():
            await self.process_single_image_folder(item, transcriber)
        elif item.suffix.lower() == ".pdf":
            await self.process_single_pdf(item, transcriber)
        elif item.suffix.lower() == ".epub":
            await self.process_single_epub(item)
        elif item.suffix.lower() in SUPPORTED_MOBI_EXTENSIONS:
            await self.process_single_mobi(item)
        else:
            logger.warning(f"Unknown file type for item: {item}")
            print_warning(f"Skipping unknown file type: {item.name}")

    def _ensure_tesseract_available(self) -> bool:
        """Verify that Tesseract is available.
        
        Returns:
            True if available, False otherwise.
        """
        return ensure_tesseract_available()

    async def _submit_batch_with_backend(
        self,
        image_files: List[Path],
        temp_jsonl_path: Path,
        parent_folder: Path,
        source_name: str,
    ) -> Optional[Any]:
        """Submit a batch using the provider-agnostic batch backend.
        
        Delegates to :func:`modules.core.batch_submission.submit_batch`.
        """
        return await submit_batch(
            image_files=image_files,
            temp_jsonl_path=temp_jsonl_path,
            parent_folder=parent_folder,
            source_name=source_name,
            model_config=self.model_config,
            user_config=self.user_config,
        )

    def _log_token_usage(self, phase: str, idx: int = 0, total: int = 0) -> None:
        """Log and print token usage statistics (consolidated helper).
        
        Args:
            phase: Label such as 'Initial', 'after item 3/10', or 'Final'.
            idx: Current item index (0 for non-item phases).
            total: Total item count (0 for non-item phases).
        """
        token_cfg = self.concurrency_config.get("daily_token_limit", {})
        if not token_cfg.get("enabled", False):
            return
        if self.user_config.transcription_method != "gpt" and phase != "Initial":
            return
        from modules.infra.token_tracker import get_token_tracker
        stats = get_token_tracker().get_stats()
        if idx and total:
            msg = (
                f"Token usage {phase} item {idx}/{total}: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
        else:
            msg = (
                f"{phase} token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
            if phase == "Initial":
                msg_extra = f" - {stats['tokens_remaining']:,} tokens remaining today"
                logger.info(msg + msg_extra)
                print_info(f"Daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)")
                return
        logger.info(msg)
        print_info(msg)

    async def process_selected_items(self,
                                     transcriber: Optional[Any] = None) -> None:
        """
        Process all selected items based on the user configuration.
        """
        selected = list(self.user_config.selected_items or [])

        # --- Resume filtering ---
        processing_type = self.user_config.processing_type or ""
        if self.resume_mode != "overwrite" and processing_type:
            selected, skipped = self.resume_checker.filter_items(selected, processing_type)
            if skipped:
                print_info(
                    f"Resume: skipping {len(skipped)} already-processed item(s)"
                )
                for sr in skipped:
                    logger.info("Skipped (already processed): %s — %s", sr.item.name, sr.reason)
        # --- End resume filtering ---

        total_items = len(selected)
        print_info(f"Beginning processing of {total_items} item(s)...")
        self._log_token_usage("Initial")

        processed_count = 0
        failed_count = 0
        interrupted = False
        try:
            for idx, item in enumerate(selected, 1):
                # Check token limit before starting each new item (only for GPT method)
                if self.user_config.transcription_method == "gpt":
                    if not await check_and_wait_for_token_limit(self.concurrency_config):
                        # User cancelled wait - stop processing
                        logger.info(f"Processing stopped by user. Processed {processed_count}/{total_items} items.")
                        print_info(f"Processing stopped. Completed {processed_count}/{total_items} items.")
                        interrupted = True
                        break
                
                print_info(f"Processing item {idx}/{total_items}: {item.name}")

                try:
                    if self.user_config.processing_type == "images":
                        await self.process_single_image_folder(item, transcriber)
                    elif self.user_config.processing_type == "pdfs":
                        await self.process_single_pdf(item, transcriber)
                    elif self.user_config.processing_type == "epubs":
                        await self.process_single_epub(item)
                    elif self.user_config.processing_type == "mobis":
                        await self.process_single_mobi(item)
                    elif self.user_config.processing_type == "auto":
                        await self._route_auto_item(item, transcriber)
                    else:
                        await self._route_auto_item(item, transcriber)
                except Exception as e:
                    failed_count += 1
                    logger.exception(f"Failed to process item {idx}/{total_items} ({item.name}): {e}")
                    print_error(f"Failed to process '{item.name}': {e}")

                processed_count += 1
                print_info(f"Completed item {idx}/{total_items}")
                self._log_token_usage("after", idx, total_items)
        except (KeyboardInterrupt, asyncio.CancelledError):
            interrupted = True
            raise
        finally:
            # Clean up any pending transient files on interruption or error
            if interrupted or failed_count > 0:
                self._transient_tracker.cleanup_pending()
            else:
                self._transient_tracker.clear()

        if failed_count > 0:
            print_warning(
                f"Processed {processed_count}/{total_items} item(s) with {failed_count} failure(s)."
            )
        else:
            print_info(f"All {processed_count}/{total_items} item(s) processed successfully.")
        
        self._log_token_usage("Final")

    async def process_single_epub(self, epub_path: Path) -> None:
        """Extract and save text from a single EPUB file."""
        await self._process_native_ebook(
            file_path=epub_path,
            processor_cls=EPUBProcessor,
            format_label="EPUB",
            default_output_dir=self.epub_output_dir,
        )

    async def process_single_mobi(self, mobi_path: Path) -> None:
        """Extract and save text from a single MOBI file."""
        await self._process_native_ebook(
            file_path=mobi_path,
            processor_cls=MOBIProcessor,
            format_label="MOBI",
            default_output_dir=self.mobi_output_dir,
        )

    async def _process_native_ebook(
        self,
        file_path: Path,
        processor_cls: type,
        format_label: str,
        default_output_dir: Path,
    ) -> None:
        """Shared logic for extracting text from EPUB/MOBI files.
        
        Args:
            file_path: Path to the ebook file.
            processor_cls: EPUBProcessor or MOBIProcessor class.
            format_label: Human-readable format name for log messages.
            default_output_dir: Default output directory for this format.
        """
        print_info(f"Processing {format_label}: {file_path.name}")

        # Resolve page/section range if configured
        section_indices = None
        if self.user_config.page_range is not None:
            # We need a preliminary extraction to count sections, but that's
            # expensive.  Instead, pass section_indices through and let the
            # processor clamp internally.  For the log message we do a quick
            # count by reading the spine / item list without full extraction.
            section_indices_raw = self.user_config.page_range.resolve(2**31)
            if section_indices_raw:
                section_indices = section_indices_raw
                print_info(
                    f"Page range: {self.user_config.page_range.describe()} "
                    f"(applied to {format_label} sections)"
                )

        processor = processor_cls(file_path)
        try:
            extraction = processor.extract_text(section_indices=section_indices)
        except Exception as exc:
            logger.exception("Failed to extract %s %s: %s", format_label, file_path.name, exc)
            print_error(f"Failed to extract text from {file_path.name}.")
            return

        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the ebook
            _parent_folder, _ = processor.prepare_output_folder(file_path.parent)
            # Final .txt goes directly next to the ebook file
            output_txt_path = file_path.parent / create_safe_filename(
                file_path.stem, ".txt", file_path.parent)
        else:
            _parent_folder, output_txt_path = processor.prepare_output_folder(default_output_dir)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        rendered_text = extraction.to_plain_text()
        # Apply post-processing if enabled
        processed_text = postprocess_transcription(rendered_text, self.postprocessing_config)
        try:
            output_txt_path.write_text(processed_text, encoding="utf-8")
        except Exception as exc:
            logger.exception("Failed to write %s transcription for %s: %s", format_label, file_path.name, exc)
            print_error(f"Failed to write output for {file_path.name}.")
            return

        # Include source_format in success message when available (e.g. MOBI)
        source_fmt = getattr(extraction, "source_format", None)
        suffix = f" (via {source_fmt})" if source_fmt else ""
        print_success(f"Extracted text from '{file_path.name}'{suffix} -> {output_txt_path.name}")

    def _cleanup_preprocessed(self, preprocessed_folder: Path, source_name: str) -> None:
        """Remove preprocessed images folder if the setting says to discard them."""
        if not self.processing_settings.get("keep_preprocessed_images", True):
            if preprocessed_folder.exists():
                try:
                    shutil.rmtree(preprocessed_folder, ignore_errors=True)
                except Exception as e:
                    logger.exception(
                        f"Error cleaning up preprocessed images for {source_name}: {e}")

    def _cleanup_temp_jsonl(self, temp_jsonl_path: Path, method: str) -> None:
        """Remove temporary JSONL unless retained or needed for batch tracking."""
        is_batch = method == "gpt" and self.user_config.use_batch_processing
        if not self.processing_settings.get("retain_temporary_jsonl", True) and not is_batch:
            try:
                temp_jsonl_path.unlink()
                print_info(f"Deleted temporary file: {temp_jsonl_path.name}")
            except Exception as e:
                logger.exception(
                    f"Error deleting temporary file {temp_jsonl_path}: {e}")
                print_error(f"Could not delete temporary file {temp_jsonl_path.name}: {e}")
        elif is_batch:
            print_info(f"Preserving {temp_jsonl_path.name} for batch tracking (required for retrieval)")

    async def process_single_pdf(self, pdf_path: Path,
                                 transcriber: Optional[Any]) -> None:
        """
        Processes a single PDF file for transcription based on the user configuration.
        """
        # Resolve per-file context and update transcriber before processing
        if transcriber is not None and not self.user_config.additional_context_path:
            from modules.llm.context_utils import resolve_context_for_file
            ctx_content, ctx_path = resolve_context_for_file(pdf_path)
            transcriber.update_context(ctx_content)

        pdf_processor = PDFProcessor(pdf_path)
        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the PDF
            parent_folder, _, temp_jsonl_path = pdf_processor.prepare_output_folder(pdf_path.parent)
            # Final .txt goes directly next to the PDF
            output_txt_path = pdf_path.parent / create_safe_filename(
                pdf_path.stem, ".txt", pdf_path.parent)
        else:
            parent_folder, output_txt_path, temp_jsonl_path = pdf_processor.prepare_output_folder(
                self.pdf_output_dir)
        method: str = self.user_config.transcription_method or "gpt"

        print_info(f"Processing PDF: {pdf_path.name}")
        print_info(f"Using method: {method}")

        # Register transient files for cleanup on interruption
        self._transient_tracker.register_jsonl(temp_jsonl_path, method)

        # Resolve page range if configured
        page_indices = None
        if self.user_config.page_range is not None:
            pdf_processor.open_pdf()
            total_pages = pdf_processor.doc.page_count
            page_indices = self.user_config.page_range.resolve(total_pages)
            if not page_indices:
                print_warning(
                    f"Page range '{self.user_config.page_range.describe()}' "
                    f"yielded no pages for '{pdf_path.name}' ({total_pages} pages). Skipping."
                )
                pdf_processor.close_pdf()
                return
            print_info(
                f"Page range: processing {len(page_indices)} of {total_pages} pages "
                f"({self.user_config.page_range.describe()})"
            )

        if method == "tesseract" and not self._ensure_tesseract_available():
            return

        # Check if method is valid for this PDF
        if method == "native" and not pdf_processor.is_native_pdf():
            print_warning(f"PDF '{pdf_path.name}' is not searchable. Switching to tesseract method.")
            method = "tesseract"  # Fall back to Tesseract if native extraction isn't possible

        # Native PDF extraction
        if method == "native":
            text = native_extract_pdf_text(pdf_path, page_indices=page_indices)
            try:
                async with aiofiles.open(temp_jsonl_path, 'a',
                                         encoding='utf-8') as jfile:
                    record = {
                        "file_name": pdf_path.name,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "method": "native",
                        "text_chunk": text,
                        "pre_processed_image": None
                    }
                    await jfile.write(json.dumps(record) + '\n')
                # Apply post-processing if enabled
                processed_text = postprocess_transcription(text, self.postprocessing_config)
                output_txt_path.write_text(processed_text, encoding='utf-8')
                print_success(f"Extracted text from '{pdf_path.name}' using native method -> {output_txt_path.name}")
            except Exception as e:
                logger.exception(
                    f"Error writing native extraction output for {pdf_path.name}: {e}")
                print_error(f"Failed to write output: {e}")

            self._cleanup_temp_jsonl(temp_jsonl_path, method)
            # Mark JSONL as complete (successfully processed)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        # Non-native PDF: Extract images
        if method == "tesseract":
            # Use separate folder and pipeline for Tesseract
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
            self._transient_tracker.register_preprocessed_folder(preprocessed_folder, pdf_path.name)
            target_dpi = (self.image_processing_config
                          .get('tesseract_image_processing', {})
                          .get('target_dpi', 300))
            print_info(f"Extracting and preprocessing images for Tesseract at {target_dpi} DPI...")
            processed_image_files = await pdf_processor.process_images_for_tesseract(
                preprocessed_folder, target_dpi, page_indices=page_indices)
        else:
            preprocessed_folder = parent_folder / "preprocessed_images"
            preprocessed_folder.mkdir(exist_ok=True)
            self._transient_tracker.register_preprocessed_folder(preprocessed_folder, pdf_path.name)
            target_dpi = (self.image_processing_config
                          .get('api_image_processing', {})
                          .get('target_dpi', 300))
            # Get provider and model name from model config for provider-specific preprocessing
            tm = self.model_config.get("transcription_model", {})
            provider = tm.get("provider", "openai")
            model_name = tm.get("name", "")
            print_info(f"Extracting and processing images from PDF with {target_dpi} DPI...")
            processed_image_files = await pdf_processor.process_images(
                preprocessed_folder, target_dpi, provider=provider, model_name=model_name,
                page_indices=page_indices)

        # Rely on extraction order; order_index will follow the list order
        print_info(f"Extracted {len(processed_image_files)} page images from PDF.")

        # Handle GPT batch mode (multi-provider via batch backends)
        if method == "gpt" and self.user_config.use_batch_processing:
            handle = await self._submit_batch_with_backend(
                processed_image_files,
                temp_jsonl_path,
                parent_folder,
                pdf_path.stem,
            )
            if handle is not None:
                # Batch submitted successfully - cleanup and return
                self._cleanup_preprocessed(preprocessed_folder, pdf_path.name)
                # Mark transient files as complete (batch will handle JSONL separately)
                self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)
                self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
                return
            # If handle is None, fall through to synchronous processing

        # Synchronous processing for GPT or Tesseract
        print_info(f"Starting {method} transcription for {len(processed_image_files)} images...")

        await self._process_images_with_method(
            processed_image_files,
            method,
            transcriber,
            temp_jsonl_path,
            output_txt_path,
            pdf_path.name
        )

        self._cleanup_preprocessed(preprocessed_folder, pdf_path.name)
        print_success(f"Saved transcription for PDF '{pdf_path.name}' -> {output_txt_path.name}")
        self._cleanup_temp_jsonl(temp_jsonl_path, method)
        # Mark transient files as complete (successfully processed)
        self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)
        self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)

    async def process_single_image_folder(self, folder: Path,
                                          transcriber: Optional[Any]) -> None:
        """
        Processes all images in a given folder based on the user configuration.
        """
        # Resolve per-folder context and update transcriber before processing
        if (transcriber is not None 
            and not self.user_config.additional_context_path
            and getattr(self.user_config, 'use_hierarchical_context', True)):
            from modules.llm.context_utils import resolve_context_for_folder
            ctx_content, ctx_path = resolve_context_for_folder(folder)
            transcriber.update_context(ctx_content)

        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the image folder
            parent_folder, preprocessed_folder, temp_jsonl_path, _ = ImageProcessor.prepare_image_folder(
                folder, folder.parent)
            # Final .txt goes directly next to the image folder (one level up)
            output_txt_path = folder.parent / create_safe_filename(
                folder.name, ".txt", folder.parent)
        else:
            parent_folder, preprocessed_folder, temp_jsonl_path, output_txt_path = ImageProcessor.prepare_image_folder(
                folder, self.image_output_dir)
        method: str = self.user_config.transcription_method or "gpt"

        print_info(f"Processing folder: {folder.name}")
        print_info(f"Using method: {method}")

        # Register transient files for cleanup on interruption
        self._transient_tracker.register_jsonl(temp_jsonl_path, method)
        self._transient_tracker.register_preprocessed_folder(preprocessed_folder, folder.name)

        # Resolve page range for image folders (indices into sorted file list)
        page_indices = None
        if self.user_config.page_range is not None:
            from modules.config.constants import SUPPORTED_IMAGE_EXTENSIONS
            total_images = len([
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ])
            page_indices = self.user_config.page_range.resolve(total_images)
            if not page_indices:
                print_warning(
                    f"Page range '{self.user_config.page_range.describe()}' "
                    f"yielded no images for '{folder.name}' ({total_images} images). Skipping."
                )
                return
            print_info(
                f"Page range: processing {len(page_indices)} of {total_images} images "
                f"({self.user_config.page_range.describe()})"
            )

        if method == "tesseract" and not self._ensure_tesseract_available():
            return

        # Process images directly from source folder to preprocessed folder
        if method == "tesseract":
            # Tesseract uses a different preprocessed folder
            self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)  # Clear initial registration
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
            self._transient_tracker.register_preprocessed_folder(preprocessed_folder, folder.name)
            print_info(f"Preprocessing images for Tesseract...")
            processed_files = ImageProcessor.process_and_save_images_for_tesseract(
                folder, preprocessed_folder, page_indices=page_indices)
        else:
            # Get provider and model name from model config for provider-specific preprocessing
            tm = self.model_config.get("transcription_model", {})
            provider = tm.get("provider", "openai")
            model_name = tm.get("name", "")
            print_info(f"Processing images from folder for {provider.upper()}...")
            processed_files = ImageProcessor.process_and_save_images(
                folder, preprocessed_folder, provider=provider, model_name=model_name,
                page_indices=page_indices)

        if not processed_files:
            print_warning(f"No images found or processed in {folder}.")
            return

        # Deterministic ordering for folders: sort by filename
        processed_files.sort(key=lambda x: x.name.lower())

        # Handle batch mode for GPT (multi-provider via batch backends)
        if method == "gpt" and self.user_config.use_batch_processing:
            handle = await self._submit_batch_with_backend(
                processed_files,
                temp_jsonl_path,
                parent_folder,
                folder.name,
            )
            if handle is not None:
                # Batch submitted successfully - cleanup and return
                self._cleanup_preprocessed(preprocessed_folder, f"folder '{folder.name}'")
                # Mark transient files as complete (batch will handle JSONL separately)
                self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)
                self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
                return
            # If handle is None, fall through to synchronous processing

        # Synchronous processing (non-batch GPT or Tesseract)
        print_info(f"Starting {method} transcription for {len(processed_files)} images...")

        await self._process_images_with_method(
            processed_files,
            method,
            transcriber,
            temp_jsonl_path,
            output_txt_path,
            folder.name,
            is_folder=True
        )

        self._cleanup_preprocessed(preprocessed_folder, f"folder '{folder.name}'")
        print_success(f"Transcription completed for folder '{folder.name}' -> {output_txt_path.name}")
        self._cleanup_temp_jsonl(temp_jsonl_path, method)
        # Mark transient files as complete (successfully processed)
        self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)
        self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)

    async def _process_images_with_method(
            self,
            image_files: List[Path],
            method: str,
            transcriber: Optional[Any],
            temp_jsonl_path: Path,
            output_txt_path: Path,
            source_name: str,
            is_folder: bool = False
    ) -> None:
        """Process images using the specified method.

        Delegates to :func:`modules.core.transcription_pipeline.run_transcription_pipeline`.
        """
        await run_transcription_pipeline(
            image_files=image_files,
            method=method,
            transcriber=transcriber,
            temp_jsonl_path=temp_jsonl_path,
            output_txt_path=output_txt_path,
            source_name=source_name,
            concurrency_config=self.concurrency_config,
            image_processing_config=self.image_processing_config,
            postprocessing_config=self.postprocessing_config,
            is_folder=is_folder,
        )

    def _write_output_from_jsonl(self, jsonl_path: Path, output_path: Path) -> bool:
        """Write combined output from JSONL records.

        Delegates to :func:`modules.core.transcription_pipeline.write_output_from_jsonl`.
        """
        return write_output_from_jsonl(jsonl_path, output_path, self.postprocessing_config)