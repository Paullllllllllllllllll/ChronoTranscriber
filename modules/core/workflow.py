# modules/workflow.py
from __future__ import annotations

import asyncio
import datetime
import json
import math
import shutil
import time
import aiofiles
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

from modules.infra.logger import setup_logger
from modules.ui.core import UserConfiguration
from modules.ui import print_info, print_warning, print_error, print_success
from modules.processing.pdf_utils import PDFProcessor, native_extract_pdf_text
from modules.processing.epub_utils import EPUBProcessor, EPUBTextExtraction
from modules.processing.mobi_utils import MOBIProcessor, MOBITextExtraction
from modules.processing.image_utils import ImageProcessor
from modules.processing.tesseract_utils import (
    configure_tesseract_executable,
    ensure_tesseract_available,
    perform_ocr,
)
from modules.llm.batch.batching import get_batch_chunk_size
from modules.llm.batch.backends import get_batch_backend, BatchRequest, BatchHandle
from modules.llm.batch.backends.factory import supports_batch
from modules.llm import transcribe_image_with_llm
from modules.infra.concurrency import run_concurrent_transcription_tasks
from modules.processing.text_processing import extract_transcribed_text, format_page_line
from modules.processing.postprocess import postprocess_transcription
from modules.core.token_guard import check_and_wait_for_token_limit
from modules.core.resume import ResumeChecker, ProcessingState
from modules.core.safe_paths import create_safe_filename
from modules.operations.jsonl_utils import (
    get_processed_image_names,
    read_jsonl_records,
    extract_transcription_records,
)

logger = setup_logger(__name__)


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

        # Check if output should be colocated with input files
        self.use_input_as_output = self.processing_settings.get("input_paths_is_output_path", False)

        # Set up default output directories (used when use_input_as_output is False)
        self.pdf_output_dir = Path(
            paths_config.get('file_paths', {}).get('PDFs', {}).get('output',
                                                                   'pdfs_out'))
        self.image_output_dir = Path(
            paths_config.get('file_paths', {}).get('Images', {}).get('output',
                                                                     'images_out'))
        self.epub_output_dir = Path(
            paths_config.get('file_paths', {}).get('EPUBs', {}).get('output',
                                                                    'epubs_out'))
        self.mobi_output_dir = Path(
            paths_config.get('file_paths', {}).get('MOBIs', {}).get('output',
                                                                    'mobis_out'))

        # Ensure default directories exist (only needed when not using input as output)
        if not self.use_input_as_output:
            self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            self.epub_output_dir.mkdir(parents=True, exist_ok=True)
            self.mobi_output_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> Optional[BatchHandle]:
        """Submit a batch using the provider-agnostic batch backend.
        
        Args:
            image_files: List of image paths to process
            temp_jsonl_path: Path to the temp JSONL file for tracking
            parent_folder: Parent folder for debug artifacts
            source_name: Name of the source (PDF or folder name)
        
        Returns:
            BatchHandle if submission successful, None otherwise
        """
        tm = self.model_config.get("transcription_model", {})
        provider = tm.get("provider", "openai")
        
        # Check if provider supports batch processing
        if not supports_batch(provider):
            print_warning(
                f"Provider '{provider}' does not support batch processing. "
                f"Falling back to synchronous mode."
            )
            return None
        
        total_images = len(image_files)
        chunk_size = get_batch_chunk_size()
        expected_batches = math.ceil(total_images / max(1, chunk_size))
        
        # Telemetry
        logger.info(
            "[Batch] Preparing submission: provider=%s, images=%d, chunk_size=%d",
            provider, total_images, chunk_size,
        )
        print_info(
            f"Batch telemetry -> provider={provider}, images={total_images}, chunk_size={chunk_size}"
        )
        print_info(f"Submitting batch job for {total_images} images...")
        
        # Early marker
        try:
            async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
                await f.write(json.dumps({"batch_session": {"status": "submitting", "provider": provider}}) + "\n")
        except Exception:
            logger.warning("Could not write early batch_session marker")
        
        # Record image metadata
        try:
            async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
                for idx, img_path in enumerate(image_files):
                    image_record = {
                        "image_metadata": {
                            "pre_processed_image": str(img_path),
                            "image_name": img_path.name,
                            "order_index": idx,
                            "custom_id": f"req-{idx + 1}"
                        }
                    }
                    await f.write(json.dumps(image_record) + "\n")
        except Exception as e:
            logger.warning("Failed writing image_metadata before batch submit: %s", e)
        
        # Build batch requests
        batch_requests = [
            BatchRequest(
                custom_id=f"req-{idx + 1}",
                image_path=img_path,
                order_index=idx,
                image_info={
                    "image_name": img_path.name,
                    "order_index": idx,
                    "page_number": idx + 1,
                },
            )
            for idx, img_path in enumerate(image_files)
        ]
        
        # Load system prompt
        from modules.config.config_loader import PROJECT_ROOT
        from modules.config.service import get_config_service
        
        pcfg = get_config_service().get_paths_config()
        general = pcfg.get("general", {})
        override_prompt = general.get("transcription_prompt_path")
        system_prompt_path = (
            Path(override_prompt)
            if override_prompt
            else (PROJECT_ROOT / "system_prompt" / "system_prompt.txt")
        )
        if not system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt missing: {system_prompt_path}")
        system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
        
        # Load additional context - use explicit path or hierarchical resolution
        additional_context = None
        if self.user_config.additional_context_path:
            ctx_path = Path(self.user_config.additional_context_path)
            if ctx_path.exists():
                additional_context = ctx_path.read_text(encoding="utf-8").strip()
        else:
            # Use hierarchical context resolution for folder/file-specific context
            from modules.llm.context_utils import resolve_context_for_folder
            context_content, context_path = resolve_context_for_folder(parent_folder)
            if context_content:
                additional_context = context_content
                logger.info(f"Using resolved context from: {context_path}")
        
        # Submit via backend
        try:
            backend = get_batch_backend(provider)
            handle = await asyncio.to_thread(
                backend.submit_batch,
                batch_requests,
                tm,
                system_prompt=system_prompt,
                schema_path=self.user_config.selected_schema_path,
                additional_context=additional_context,
            )
        except Exception as e:
            logger.exception(f"Batch submission failed for {source_name}: {e}")
            print_error(
                f"Batch submission failed for {source_name}. "
                f"Falling back to synchronous processing."
            )
            return None
        
        # Write tracking record with provider info
        try:
            async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
                tracking_record = {
                    "batch_tracking": {
                        "batch_id": handle.batch_id,
                        "provider": handle.provider,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "metadata": handle.metadata,
                    }
                }
                await f.write(json.dumps(tracking_record) + "\n")
        except Exception as e:
            logger.warning("Post-submission tracking write failed for %s: %s", source_name, e)
        
        # Write debug artifact
        try:
            debug_payload = {
                "source": source_name,
                "provider": provider,
                "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "expected_batches": int(expected_batches),
                "batch_ids": [handle.batch_id],
                "total_images": int(total_images),
                "chunk_size": int(chunk_size),
            }
            debug_path = parent_folder / f"{source_name}_batch_submission_debug.json"
            debug_path.write_text(json.dumps(debug_payload, indent=2), encoding='utf-8')
        except Exception as e:
            logger.warning("Failed to write batch submission debug artifact for %s: %s", source_name, e)
        
        # Mark submitted
        try:
            async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as f:
                await f.write(json.dumps({"batch_session": {"status": "submitted", "provider": provider}}) + "\n")
        except Exception:
            logger.warning("Could not write submitted batch_session marker")
        
        print_success(f"Batch submitted for '{source_name}'.")
        print_info("The batch will be processed asynchronously. Use 'check_batches.py' to monitor status.")
        
        return handle

    async def tesseract_ocr_image(self, img_path: Path,
                                  tesseract_config: str) -> Optional[str]:
        """Perform OCR on an image using Tesseract.
        
        Args:
            img_path: Path to image file.
            tesseract_config: Tesseract configuration string.
            
        Returns:
            Extracted text, or placeholder if no text found, or None on error.
        """
        return perform_ocr(img_path, tesseract_config)

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
        
        # Display initial token usage if enabled
        token_cfg = self.concurrency_config.get("daily_token_limit", {})
        if token_cfg.get("enabled", False):
            from modules.infra.token_tracker import get_token_tracker
            token_tracker = get_token_tracker()
            stats = token_tracker.get_stats()
            logger.info(
                f"Token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%) - "
                f"{stats['tokens_remaining']:,} tokens remaining today"
            )
            print_info(
                f"Daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )

        processed_count = 0
        failed_count = 0
        for idx, item in enumerate(selected, 1):
            # Check token limit before starting each new item (only for GPT method)
            if self.user_config.transcription_method == "gpt":
                if not await check_and_wait_for_token_limit(self.concurrency_config):
                    # User cancelled wait - stop processing
                    logger.info(f"Processing stopped by user. Processed {processed_count}/{total_items} items.")
                    print_info(f"Processing stopped. Completed {processed_count}/{total_items} items.")
                    break
            
            print_info(f"Processing item {idx}/{total_items}: {item.name}")

            try:
                if self.user_config.processing_type == "images":
                    # Process image folder
                    await self.process_single_image_folder(
                        item,
                        transcriber
                    )
                elif self.user_config.processing_type == "pdfs":
                    # Process PDF file
                    await self.process_single_pdf(
                        item,
                        transcriber
                    )
                elif self.user_config.processing_type == "epubs":
                    await self.process_single_epub(item)
                elif self.user_config.processing_type == "mobis":
                    await self.process_single_mobi(item)
                elif self.user_config.processing_type == "auto":
                    # Auto mode: route each item based on its actual type
                    await self._route_auto_item(item, transcriber)
                else:
                    # Fallback - try to detect type from extension
                    await self._route_auto_item(item, transcriber)
            except Exception as e:
                failed_count += 1
                logger.exception(f"Failed to process item {idx}/{total_items} ({item.name}): {e}")
                print_error(f"Failed to process '{item.name}': {e}")

            processed_count += 1
            print_info(f"Completed item {idx}/{total_items}")
            
            # Log and print token usage after each item if enabled
            if token_cfg.get("enabled", False) and self.user_config.transcription_method == "gpt":
                token_tracker = get_token_tracker()
                stats = token_tracker.get_stats()
                usage_msg = (
                    f"Token usage after item {idx}/{total_items}: "
                    f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                    f"({stats['usage_percentage']:.1f}%)"
                )
                logger.info(usage_msg)
                print_info(usage_msg)

        if failed_count > 0:
            print_warning(
                f"Processed {processed_count}/{total_items} item(s) with {failed_count} failure(s)."
            )
        else:
            print_info(f"All {processed_count}/{total_items} item(s) processed successfully.")
        
        # Final token usage statistics
        if token_cfg.get("enabled", False) and self.user_config.transcription_method == "gpt":
            token_tracker = get_token_tracker()
            stats = token_tracker.get_stats()
            logger.info(
                f"Final token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
            print_info(
                f"Final daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )

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
            return

        # Non-native PDF: Extract images
        if method == "tesseract":
            # Use separate folder and pipeline for Tesseract
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
            target_dpi = (self.image_processing_config
                          .get('tesseract_image_processing', {})
                          .get('target_dpi', 300))
            print_info(f"Extracting and preprocessing images for Tesseract at {target_dpi} DPI...")
            processed_image_files = await pdf_processor.process_images_for_tesseract(
                preprocessed_folder, target_dpi, page_indices=page_indices)
        else:
            preprocessed_folder = parent_folder / "preprocessed_images"
            preprocessed_folder.mkdir(exist_ok=True)
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

    async def process_single_image_folder(self, folder: Path,
                                          transcriber: Optional[Any]) -> None:
        """
        Processes all images in a given folder based on the user configuration.
        """
        # Resolve per-folder context and update transcriber before processing
        if transcriber is not None and not self.user_config.additional_context_path:
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
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
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
        """
        Helper method to process images using the specified method.
        Skips images that have already been successfully processed (exist in JSONL).
        """
        # Load already-processed image names from existing JSONL to avoid duplicates
        already_processed = get_processed_image_names(temp_jsonl_path)
        
        # Filter out already-processed images
        if already_processed:
            original_count = len(image_files)
            image_files = [img for img in image_files if img.name not in already_processed]
            skipped_count = original_count - len(image_files)
            if skipped_count > 0:
                print_info(f"Skipping {skipped_count} already-processed images (found in JSONL)")
                logger.info(f"Skipped {skipped_count} images already in {temp_jsonl_path.name}")
        
        if not image_files:
            print_info("All images already processed. Regenerating output file from JSONL...")
            self._write_output_from_jsonl(temp_jsonl_path, output_txt_path)
            return

        async def transcribe_single_image_task(
                img_path: Path,
                trans: Optional[Any],
                method: str,
                tesseract_config: str = "--oem 3 --psm 6",
                order_index: int = 0,
        ) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]], int]:
            """
            Transcribes a single image file using either GPT (via OpenAI) or Tesseract OCR.
            Returns a tuple containing the image path, image name, and the transcription result.
            """
            image_name = img_path.name
            final_text: Optional[str] = None
            try:
                if method == "gpt":
                    if not trans:
                        logger.error(
                            "No transcriber instance provided for GPT usage.")
                        return (str(img_path), image_name,
                                f"[transcription error: {image_name}]", None, order_index)
                    result = await transcribe_image_with_llm(img_path, trans)
                    logger.debug(
                        f"LLM response for {img_path.name}: {result}")
                    try:
                        final_text = extract_transcribed_text(result, image_name)
                    except Exception as e:
                        logger.error(
                            f"Error extracting transcription for {img_path.name}: {e}. Marking as transcription error.")
                        final_text = f"[transcription error: {image_name}]"
                    return (str(img_path), image_name, final_text, result, order_index)
                elif method == "tesseract":
                    final_text = await self.tesseract_ocr_image(img_path, tesseract_config)
                else:
                    logger.error(
                        f"Unknown transcription method '{method}' for image {img_path.name}")
                    final_text = None
                return (str(img_path), image_name, final_text, None, order_index)
            except Exception as e:
                logger.exception(
                    f"Error transcribing {img_path.name} with method '{method}': {e}")
                return (
                    str(img_path), image_name,
                    f"[transcription error: {image_name}]",
                    None,
                    order_index
                )

        # Build a mapping from image name to original order index for correct ordering
        # We need to preserve the original page order even when resuming
        all_image_names = {img.name: idx for idx, img in enumerate(image_files)}
        
        # Set up args for processing
        if method == "gpt":
            args_list = [
                (
                    img,
                    transcriber,
                    method,
                    (self.image_processing_config
                     .get('tesseract_image_processing', {})
                     .get('ocr', {})
                     .get('tesseract_config', "--oem 3 --psm 6")),
                    all_image_names[img.name],  # Use original index for correct ordering
                )
                for img in image_files
            ]
        else:
            args_list = [
                (
                    img,
                    None,
                    method,
                    (self.image_processing_config
                     .get('tesseract_image_processing', {})
                     .get('ocr', {})
                     .get('tesseract_config', "--oem 3 --psm 6")),
                    all_image_names[img.name],  # Use original index for correct ordering
                )
                for img in image_files
            ]

        transcription_conf = self.concurrency_config.get("concurrency", {}).get(
            "transcription", {})
        concurrency_limit = transcription_conf.get("concurrency_limit", 20)
        delay_between_tasks = transcription_conf.get("delay_between_tasks", 0)

        # Set up streaming writes to JSONL as results arrive
        write_lock = asyncio.Lock()
        async with aiofiles.open(temp_jsonl_path, 'a', encoding='utf-8') as jfile:
            async def on_result_write(result_tuple: Any) -> None:
                # Expecting: (img_path_str, image_name, text, raw_response, order_index)
                if not result_tuple or len(result_tuple) < 5:
                    return
                img_path_str, image_name, text_chunk, raw_response, order_index = result_tuple
                if text_chunk is None:
                    return
                # Build base record
                record: Dict[str, Any]
                if is_folder:
                    record = {
                        "folder_name": source_name,
                        "pre_processed_image": img_path_str,
                        "image_name": image_name,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "method": method,
                        "order_index": order_index,
                        "text_chunk": text_chunk,
                    }
                else:
                    record = {
                        "file_name": source_name,
                        "pre_processed_image": img_path_str,
                        "image_name": image_name,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "method": method,
                        "order_index": order_index,
                        "text_chunk": text_chunk,
                    }

                # Include raw_response and request_context only for GPT
                if method == "gpt" and raw_response is not None and transcriber is not None:
                    try:
                        # Best-effort request context from the transcriber
                        ctx = {}
                        extractor = getattr(transcriber, "extractor", None)
                        if extractor is not None:
                            ctx = {
                                "model": getattr(extractor, "model", None),
                                "service_tier": getattr(extractor, "service_tier", None),
                                "max_output_tokens": getattr(extractor, "max_output_tokens", None),
                                "temperature": getattr(extractor, "temperature", None),
                                "top_p": getattr(extractor, "top_p", None),
                                "presence_penalty": getattr(extractor, "presence_penalty", None),
                                "frequency_penalty": getattr(extractor, "frequency_penalty", None),
                                "stop": getattr(extractor, "stop", None),
                                "seed": getattr(extractor, "seed", None),
                                "reasoning": getattr(extractor, "reasoning", None),
                                "text": getattr(extractor, "text_params", None),
                                "detail": None,
                            }
                        record["request_context"] = ctx
                        record["raw_response"] = raw_response
                    except Exception as _:
                        # If for any reason we cannot attach context, continue with base record
                        pass

                # Serialize safely
                async with write_lock:
                    await jfile.write(json.dumps(record) + "\n")
                    await jfile.flush()

            try:
                print_info(f"Processing with concurrency limit of {concurrency_limit}...")
                results = await run_concurrent_transcription_tasks(
                    transcribe_single_image_task,
                    args_list,
                    concurrency_limit,
                    delay_between_tasks,
                    on_result=on_result_write,
                )
            except Exception as e:
                logger.exception(
                    f"Error running concurrent transcription tasks for {source_name}: {e}")
                print_error(f"Concurrency error for {source_name}.")
                return

        # Combine the transcription text in page order with unified page headers
        try:
            # results tuple: (img_path_str, image_name, text, raw_response, order_index)
            ordered = sorted(
                [r for r in results if r and r[2] is not None], key=lambda r: r[4]
            )
            lines: List[str] = []
            for (_p, image_name, text_chunk, _raw, order_index) in ordered:
                page_number = int(order_index) + 1 if isinstance(order_index, int) else None
                lines.append(format_page_line(text_chunk, page_number, image_name))
            combined_text = "\n".join(lines)
            # Apply post-processing if enabled
            processed_text = postprocess_transcription(combined_text, self.postprocessing_config)
            output_txt_path.write_text(processed_text, encoding='utf-8')
        except Exception as e:
            logger.exception(
                f"Error writing combined transcription output for {source_name}: {e}")
            print_error(f"Failed to write combined output for {source_name}.")
            return

    def _write_output_from_jsonl(self, jsonl_path: Path, output_path: Path) -> bool:
        """Write combined output text from JSONL transcription records.
        
        Args:
            jsonl_path: Path to JSONL file with transcription records.
            output_path: Path to write combined text output.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            records = read_jsonl_records(jsonl_path)
            transcriptions = extract_transcription_records(records, deduplicate=True)
            
            if not transcriptions:
                print_warning(f"No valid transcription records found in {jsonl_path.name}")
                return False
            
            # Sort by order_index for correct page order
            ordered = sorted(transcriptions, key=lambda r: r.get("order_index", 0))
            
            lines: List[str] = []
            for record in ordered:
                image_name = record.get("image_name", "")
                text_chunk = record.get("text_chunk", "")
                order_index = record.get("order_index", 0)
                page_number = int(order_index) + 1 if isinstance(order_index, int) else None
                lines.append(format_page_line(text_chunk, page_number, image_name))
            
            combined_text = "\n".join(lines)
            processed_text = postprocess_transcription(combined_text, self.postprocessing_config)
            output_path.write_text(processed_text, encoding='utf-8')
            print_success(f"Output written: {output_path.name}")
            return True
        except Exception as e:
            logger.exception(f"Error writing output from JSONL {jsonl_path.name}: {e}")
            print_error(f"Failed to write output from {jsonl_path.name}.")
            return False