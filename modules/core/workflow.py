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
        
        # Load additional context - use hierarchical resolution if no explicit path
        additional_context = None
        if self.user_config.additional_context_path:
            ctx_path = Path(self.user_config.additional_context_path)
            if ctx_path.exists():
                additional_context = ctx_path.read_text(encoding="utf-8").strip()
        else:
            # Use hierarchical context resolution for folder/file-specific context
            from modules.llm.context_utils import resolve_context_for_folder
            context_content, context_path = resolve_context_for_folder(
                parent_folder,
                global_context_path=PROJECT_ROOT / "additional_context" / "additional_context.txt"
            )
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
        total_items = len(self.user_config.selected_items)
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
        for idx, item in enumerate(self.user_config.selected_items, 1):
            # Check token limit before starting each new item (only for GPT method)
            if self.user_config.transcription_method == "gpt":
                if not await check_and_wait_for_token_limit(self.concurrency_config):
                    # User cancelled wait - stop processing
                    logger.info(f"Processing stopped by user. Processed {processed_count}/{total_items} items.")
                    print_info(f"Processing stopped. Completed {processed_count}/{total_items} items.")
                    break
            
            print_info(f"Processing item {idx}/{total_items}: {item.name}")

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
            else:
                # Fallback - try to detect type from extension
                suffix = item.suffix.lower()
                if suffix == ".epub":
                    await self.process_single_epub(item)
                elif suffix == ".mobi" or suffix in {".azw", ".azw3", ".kfx"}:
                    await self.process_single_mobi(item)
                else:
                    logger.warning(f"Unknown processing type for item: {item}")
                    print_warning(f"Skipping unknown file type: {item.name}")

            processed_count += 1
            print_info(f"Completed item {idx}/{total_items}")
            
            # Log token usage after each item if enabled
            if token_cfg.get("enabled", False) and self.user_config.transcription_method == "gpt":
                token_tracker = get_token_tracker()
                stats = token_tracker.get_stats()
                logger.info(
                    f"Token usage after item {idx}/{total_items}: "
                    f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                    f"({stats['usage_percentage']:.1f}%)"
                )

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
        print_info(f"Processing EPUB: {epub_path.name}")

        processor = EPUBProcessor(epub_path)
        try:
            extraction: EPUBTextExtraction = processor.extract_text()
        except Exception as exc:
            logger.exception("Failed to extract EPUB %s: %s", epub_path.name, exc)
            print_error(f"Failed to extract text from {epub_path.name}.")
            return

        # Determine output directory: use input file's parent when input_paths_is_output_path is True
        output_dir = epub_path.parent if self.use_input_as_output else self.epub_output_dir
        _parent_folder, output_txt_path = processor.prepare_output_folder(output_dir)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        rendered_text = extraction.to_plain_text()
        # Apply post-processing if enabled
        processed_text = postprocess_transcription(rendered_text, self.postprocessing_config)
        try:
            output_txt_path.write_text(processed_text, encoding="utf-8")
        except Exception as exc:
            logger.exception("Failed to write EPUB transcription for %s: %s", epub_path.name, exc)
            print_error(f"Failed to write output for {epub_path.name}.")
            return

        print_success(f"Extracted text from '{epub_path.name}' -> {output_txt_path.name}")

    async def process_single_mobi(self, mobi_path: Path) -> None:
        """Extract and save text from a single MOBI file."""
        print_info(f"Processing MOBI: {mobi_path.name}")

        processor = MOBIProcessor(mobi_path)
        try:
            extraction: MOBITextExtraction = processor.extract_text()
        except Exception as exc:
            logger.exception("Failed to extract MOBI %s: %s", mobi_path.name, exc)
            print_error(f"Failed to extract text from {mobi_path.name}.")
            return

        # Determine output directory: use input file's parent when input_paths_is_output_path is True
        output_dir = mobi_path.parent if self.use_input_as_output else self.mobi_output_dir
        _parent_folder, output_txt_path = processor.prepare_output_folder(output_dir)
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        rendered_text = extraction.to_plain_text()
        # Apply post-processing if enabled
        processed_text = postprocess_transcription(rendered_text, self.postprocessing_config)
        try:
            output_txt_path.write_text(processed_text, encoding="utf-8")
        except Exception as exc:
            logger.exception("Failed to write MOBI transcription for %s: %s", mobi_path.name, exc)
            print_error(f"Failed to write output for {mobi_path.name}.")
            return

        print_success(f"Extracted text from '{mobi_path.name}' (via {extraction.source_format}) -> {output_txt_path.name}")

    async def process_single_pdf(self, pdf_path: Path,
                                 transcriber: Optional[Any]) -> None:
        """
        Processes a single PDF file for transcription based on the user configuration.
        """
        pdf_processor = PDFProcessor(pdf_path)
        # Determine output directory: use input file's parent when input_paths_is_output_path is True
        output_dir = pdf_path.parent if self.use_input_as_output else self.pdf_output_dir
        parent_folder, output_txt_path, temp_jsonl_path = pdf_processor.prepare_output_folder(
            output_dir)
        method = self.user_config.transcription_method

        print_info(f"Processing PDF: {pdf_path.name}")
        print_info(f"Using method: {method}")

        if method == "tesseract" and not self._ensure_tesseract_available():
            return

        # Check if method is valid for this PDF
        if method == "native" and not pdf_processor.is_native_pdf():
            print_warning(f"PDF '{pdf_path.name}' is not searchable. Switching to tesseract method.")
            method = "tesseract"  # Fall back to Tesseract if native extraction isn't possible

        # Native PDF extraction
        if method == "native":
            text = native_extract_pdf_text(pdf_path)
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

            # Cleanup if not retaining
            if not self.processing_settings.get("retain_temporary_jsonl", True):
                try:
                    temp_jsonl_path.unlink()
                    print_info(f"Deleted temporary file: {temp_jsonl_path.name}")
                except Exception as e:
                    logger.exception(
                        f"Error deleting temporary file {temp_jsonl_path}: {e}")
                    print_error(f"Could not delete temporary file {temp_jsonl_path.name}: {e}")

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
                preprocessed_folder, target_dpi)
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
                preprocessed_folder, target_dpi, provider=provider, model_name=model_name)

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
                if not self.processing_settings.get("keep_preprocessed_images", True):
                    if preprocessed_folder.exists():
                        try:
                            shutil.rmtree(preprocessed_folder, ignore_errors=True)
                        except Exception as e:
                            logger.exception(
                                f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")
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

        # If keep_preprocessed_images is false, delete preprocessed folder
        if not self.processing_settings.get("keep_preprocessed_images", True):
            if preprocessed_folder.exists():
                try:
                    shutil.rmtree(preprocessed_folder, ignore_errors=True)
                except Exception as e:
                    logger.exception(
                        f"Error cleaning up preprocessed images for {pdf_path.name}: {e}")

        print_success(f"Saved transcription for PDF '{pdf_path.name}' -> {output_txt_path.name}")

        # Remove temporary JSONL if not retaining AND not using batch processing
        # For batch processing, we must keep the JSONL files as they contain batch tracking info
        if not self.processing_settings.get("retain_temporary_jsonl",
                                            True) and not (
                method == "gpt" and self.user_config.use_batch_processing):
            try:
                temp_jsonl_path.unlink()
                print_info(f"Deleted temporary file: {temp_jsonl_path.name}")
            except Exception as e:
                logger.exception(
                    f"Error deleting temporary file {temp_jsonl_path}: {e}")
                print_error(f"Could not delete temporary file {temp_jsonl_path.name}: {e}")
        elif method == "gpt" and self.user_config.use_batch_processing:
            print_info(f"Preserving {temp_jsonl_path.name} for batch tracking (required for retrieval)")

    async def process_single_image_folder(self, folder: Path,
                                          transcriber: Optional[Any]) -> None:
        """
        Processes all images in a given folder based on the user configuration.
        """
        # Determine output directory: use input folder itself when input_paths_is_output_path is True
        # This places output files inside the source folder
        output_dir = folder if self.use_input_as_output else self.image_output_dir
        parent_folder, preprocessed_folder, temp_jsonl_path, output_txt_path = ImageProcessor.prepare_image_folder(
            folder, output_dir)
        method = self.user_config.transcription_method

        print_info(f"Processing folder: {folder.name}")
        print_info(f"Using method: {method}")

        if method == "tesseract" and not self._ensure_tesseract_available():
            return

        # Process images directly from source folder to preprocessed folder
        if method == "tesseract":
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
            print_info(f"Preprocessing images for Tesseract...")
            processed_files = ImageProcessor.process_and_save_images_for_tesseract(folder, preprocessed_folder)
        else:
            # Get provider and model name from model config for provider-specific preprocessing
            tm = self.model_config.get("transcription_model", {})
            provider = tm.get("provider", "openai")
            model_name = tm.get("name", "")
            print_info(f"Processing images from folder for {provider.upper()}...")
            processed_files = ImageProcessor.process_and_save_images(
                folder, preprocessed_folder, provider=provider, model_name=model_name)

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
                if not self.processing_settings.get("keep_preprocessed_images", True):
                    if preprocessed_folder.exists():
                        try:
                            shutil.rmtree(preprocessed_folder, ignore_errors=True)
                        except Exception as e:
                            logger.exception(
                                f"Error cleaning up preprocessed images for folder '{folder.name}': {e}")
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

        # Delete preprocessed folder if not retaining
        if not self.processing_settings.get("keep_preprocessed_images", True):
            if preprocessed_folder.exists():
                try:
                    shutil.rmtree(preprocessed_folder, ignore_errors=True)
                except Exception as e:
                    logger.exception(
                        f"Error cleaning up preprocessed images for folder '{folder.name}': {e}")

        print_success(f"Transcription completed for folder '{folder.name}' -> {output_txt_path.name}")

        # Delete temporary JSONL if not retaining AND not using batch processing
        # For batch processing, we must keep the JSONL files as they contain batch tracking info
        if not self.processing_settings.get("retain_temporary_jsonl",
                                            True) and not (
                method == "gpt" and self.user_config.use_batch_processing):
            try:
                temp_jsonl_path.unlink()
                print_info(f"Deleted temporary file: {temp_jsonl_path.name}")
            except Exception as e:
                logger.exception(
                    f"Error deleting temporary file {temp_jsonl_path}: {e}")
                print_error(f"Could not delete temporary file {temp_jsonl_path.name}: {e}")
        elif method == "gpt" and self.user_config.use_batch_processing:
            print_info(f"Preserving {temp_jsonl_path.name} for batch tracking (required for retrieval)")

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
        """

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
                    idx,
                )
                for idx, img in enumerate(image_files)
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
                    idx,
                )
                for idx, img in enumerate(image_files)
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