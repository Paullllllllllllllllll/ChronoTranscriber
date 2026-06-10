# modules/workflow.py
from __future__ import annotations

import asyncio
import datetime
import json
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles

from modules.batch.submission import submit_batch
from modules.documents.epub import EPUBProcessor
from modules.documents.mobi import MOBIProcessor
from modules.documents.pdf import PDFProcessor, native_extract_pdf_text
from modules.images.page_stream import (
    PagePayload,
    compute_folder_skip_names,
    compute_pdf_skip_indices,
    folder_image_name,
    list_folder_images,
    resolve_image_settings,
    stream_folder_payloads,
    stream_pdf_payloads,
)
from modules.images.pipeline import ImageProcessor
from modules.images.tesseract_runtime import (
    configure_tesseract_executable,
    ensure_tesseract_available,
)
from modules.infra.logger import setup_logger
from modules.infra.paths import (
    PathConfig,
    create_safe_filename,
    mirror_output_path,
)
from modules.infra.token_budget import check_and_wait_for_token_limit
from modules.postprocess.writer import write_transcription_output
from modules.transcribe.pipeline import (
    build_file_provenance,
    run_streaming_transcription_pipeline,
    run_transcription_pipeline,
    write_output_from_jsonl,
)
from modules.transcribe.resume import ResumeChecker
from modules.transcribe.user_config import UserConfiguration
from modules.ui import print_error, print_info, print_success, print_warning

logger = setup_logger(__name__)


@dataclass
class ProcessingSummary:
    """Outcome counts for a `process_selected_items` run.

    `processed` counts only items that completed without raising; `failed`
    counts items that raised during processing. `total` is the number of
    items actually selected for processing (after resume filtering), so
    `processed + failed` may be less than `total` when the run is interrupted.
    """

    processed: int = 0
    failed: int = 0
    total: int = 0


def _relative_key(item: Path, input_root: Path | None) -> str | None:
    if input_root is None:
        return None
    try:
        return str(item.relative_to(input_root))
    except ValueError:
        return None


class TransientFileTracker:
    """Tracks transient files created during processing for cleanup on interruption.

    This class ensures that temporary files (.jsonl) and preprocessed image folders
    are cleaned up when processing is interrupted (e.g., by token limit exit or Ctrl+C).
    """

    def __init__(self) -> None:
        self._jsonl_files: list[tuple[Path, str]] = []  # (path, method)
        self._preprocessed_folders: list[tuple[Path, str]] = []  # (path, source_name)
        self._processing_settings: dict[str, Any] = {}
        self._use_batch_processing: bool = False

    def configure(
        self, processing_settings: dict[str, Any], use_batch_processing: bool = False
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
        """Mark a preprocessed folder as successfully processed (remove from
        tracking)."""
        self._preprocessed_folders = [
            (p, n) for p, n in self._preprocessed_folders if p != path
        ]

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
        keep_preprocessed = self._processing_settings.get(
            "keep_preprocessed_images", True
        )
        if not keep_preprocessed:
            for folder_path, source_name in self._preprocessed_folders:
                try:
                    if folder_path.exists():
                        shutil.rmtree(folder_path, ignore_errors=True)
                        logger.info(
                            "Cleaned up interrupted preprocessed folder for %s",
                            source_name,
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up preprocessed folder {folder_path}: {e}"
                    )

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

    def __init__(
        self,
        user_config: UserConfiguration,
        paths_config: dict[str, Any],
        model_config: dict[str, Any],
        concurrency_config: dict[str, Any],
        image_processing_config: dict[str, Any],
    ) -> None:
        self.user_config = user_config
        self.paths_config = paths_config
        self.model_config = model_config
        self.concurrency_config = concurrency_config
        self.image_processing_config = image_processing_config
        self.processing_settings = paths_config.get("general", {})

        # Configure Tesseract executable if provided
        configure_tesseract_executable(image_processing_config)
        self.ocr_config = image_processing_config.get(
            "tesseract_image_processing", {}
        ).get("ocr", {})

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

        # Output mode
        self.output_mode: str = getattr(user_config, "output_mode", "hash")
        self.input_root: Path | None = getattr(user_config, "input_root", None)

        # Resume checker
        self.resume_mode = user_config.resume_mode
        self.output_format = getattr(user_config, "output_format", "txt") or "txt"
        self.resume_checker = ResumeChecker(
            resume_mode=self.resume_mode,
            paths_config=paths_config,
            use_input_as_output=self.use_input_as_output,
            pdf_output_dir=self.pdf_output_dir,
            image_output_dir=self.image_output_dir,
            epub_output_dir=self.epub_output_dir,
            mobi_output_dir=self.mobi_output_dir,
            output_format=self.output_format,
            output_mode=self.output_mode,
            input_root=self.input_root,
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
            use_batch_processing=user_config.use_batch_processing,
        )

    async def _route_auto_item(self, item: Path, transcriber: Any | None) -> None:
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
        payloads: list[PagePayload],
        temp_jsonl_path: Path,
        parent_folder: Path,
        source_name: str,
        file_provenance: dict[str, Any] | None = None,
    ) -> Any | None:
        """Submit a batch using the provider-agnostic batch backend.

        Delegates to :func:`modules.batch.submission.submit_batch`.
        """
        return await submit_batch(
            payloads=payloads,
            temp_jsonl_path=temp_jsonl_path,
            parent_folder=parent_folder,
            source_name=source_name,
            model_config=self.model_config,
            user_config=self.user_config,
            file_provenance=file_provenance,
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
        from modules.infra.token_budget import get_token_tracker

        stats = get_token_tracker().get_stats()
        if idx and total:
            msg = (
                f"Token usage {phase} item {idx}/{total}: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
        else:
            used = stats["tokens_used_today"]
            limit = stats["daily_limit"]
            pct = stats["usage_percentage"]
            msg = f"{phase} token usage: {used:,}/{limit:,} ({pct:.1f}%)"
            if phase == "Initial":
                msg_extra = f" - {stats['tokens_remaining']:,} tokens remaining today"
                logger.info(msg + msg_extra)
                print_info(f"Daily token usage: {used:,}/{limit:,} ({pct:.1f}%)")
                return
        logger.info(msg)
        print_info(msg)

    async def process_selected_items(
        self, transcriber: Any | None = None
    ) -> ProcessingSummary:
        """
        Process all selected items based on the user configuration.

        Returns a `ProcessingSummary` with the real success/failure counts so
        callers can render an accurate completion summary.
        """
        selected = list(self.user_config.selected_items or [])

        # Resume filtering (item level): skip items with complete output files.
        # A second layer of page-level resume filtering occurs inside
        # run_transcription_pipeline() via JSONL scanning; both layers are
        # intentional — this one avoids re-entering the pipeline entirely,
        # while the inner one allows resuming partially-transcribed items.
        processing_type = self.user_config.processing_type or ""
        if self.resume_mode != "overwrite" and processing_type:
            selected, skipped = self.resume_checker.filter_items(
                selected, processing_type
            )
            if skipped:
                print_info(f"Resume: skipping {len(skipped)} already-processed item(s)")
                for sr in skipped:
                    logger.info(
                        "Skipped (already processed): %s — %s", sr.item.name, sr.reason
                    )
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
                if (
                    self.user_config.transcription_method == "gpt"
                    and not await check_and_wait_for_token_limit(
                        self.concurrency_config
                    )
                ):
                    # User cancelled wait - stop processing
                    logger.info(
                        "Processing stopped by user. Processed %d/%d items.",
                        processed_count,
                        total_items,
                    )
                    print_info(
                        f"Processing stopped."
                        f" Completed {processed_count}/{total_items} items."
                    )
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
                        logger.warning(
                            "Unexpected processing_type %r for item %s;"
                            " routing via auto",
                            self.user_config.processing_type,
                            item.name,
                        )
                        await self._route_auto_item(item, transcriber)
                except Exception as e:
                    failed_count += 1
                    logger.exception(
                        f"Failed to process item {idx}/{total_items} ({item.name}): {e}"
                    )
                    print_error(f"Failed to process '{item.name}': {e}")
                else:
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
                f"Processed {processed_count}/{total_items} item(s)"
                f" with {failed_count} failure(s)."
            )
        else:
            print_info(
                f"All {processed_count}/{total_items} item(s) processed successfully."
            )

        self._log_token_usage("Final")

        return ProcessingSummary(
            processed=processed_count, failed=failed_count, total=total_items
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
        processor_cls: type[Any],
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
            logger.exception(
                "Failed to extract %s %s: %s", format_label, file_path.name, exc
            )
            print_error(f"Failed to extract text from {file_path.name}.")
            return

        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the ebook
            _parent_folder, _ = processor.prepare_output_folder(file_path.parent)
            # Final .txt goes directly next to the ebook file
            output_txt_path = file_path.parent / create_safe_filename(
                file_path.stem, ".txt", file_path.parent
            )
        else:
            _parent_folder, output_txt_path = processor.prepare_output_folder(
                default_output_dir
            )
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)

        rendered_text = extraction.to_plain_text()
        # Write output using central writer
        output_format = getattr(self.user_config, "output_format", "txt") or "txt"
        pages = [{"text": rendered_text, "page_number": None, "image_name": None}]
        try:
            actual_path = write_transcription_output(
                pages,
                output_txt_path,
                output_format=output_format,
                postprocess=True,
                postprocessing_config=self.postprocessing_config,
            )
        except Exception as exc:
            logger.exception(
                "Failed to write %s transcription for %s: %s",
                format_label,
                file_path.name,
                exc,
            )
            print_error(f"Failed to write output for {file_path.name}.")
            return

        # Include source_format in success message when available (e.g. MOBI)
        source_fmt = getattr(extraction, "source_format", None)
        suffix = f" (via {source_fmt})" if source_fmt else ""
        print_success(
            f"Extracted text from '{file_path.name}'{suffix} -> {actual_path.name}"
        )

    def _cleanup_preprocessed(
        self, preprocessed_folder: Path, source_name: str
    ) -> None:
        """Remove preprocessed images folder if the setting says to discard them."""
        if (
            not self.processing_settings.get("keep_preprocessed_images", True)
            and preprocessed_folder.exists()
        ):
            try:
                shutil.rmtree(preprocessed_folder, ignore_errors=True)
            except Exception as e:
                logger.exception(
                    f"Error cleaning up preprocessed images for {source_name}: {e}"
                )

    async def _handle_batch_submission(
        self,
        payloads: list[PagePayload],
        temp_jsonl_path: Path,
        parent_folder: Path,
        source_stem: str,
        file_provenance: dict[str, Any] | None = None,
    ) -> bool:
        """Try to submit a batch job; return True if submitted, False to fall
        through to synchronous processing."""
        if not self.user_config.use_batch_processing:
            return False
        handle = await self._submit_batch_with_backend(
            payloads,
            temp_jsonl_path,
            parent_folder,
            source_stem,
            file_provenance=file_provenance,
        )
        if handle is None:
            return False
        self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
        return True

    def _resolve_target_dpi(self, method: str) -> int:
        """Return the configured DPI for the given transcription method."""
        section = (
            "tesseract_image_processing"
            if method == "tesseract"
            else "api_image_processing"
        )
        return int(self.image_processing_config.get(section, {}).get("target_dpi", 300))

    def _cleanup_temp_jsonl(self, temp_jsonl_path: Path, method: str) -> None:
        """Remove temporary JSONL unless retained or needed for batch tracking."""
        is_batch = method == "gpt" and self.user_config.use_batch_processing
        if (
            not self.processing_settings.get("retain_temporary_jsonl", True)
            and not is_batch
        ):
            try:
                temp_jsonl_path.unlink()
                print_info(f"Deleted temporary file: {temp_jsonl_path.name}")
            except Exception as e:
                logger.exception(
                    f"Error deleting temporary file {temp_jsonl_path}: {e}"
                )
                print_error(
                    f"Could not delete temporary file {temp_jsonl_path.name}: {e}"
                )
        elif is_batch:
            print_info(
                f"Preserving {temp_jsonl_path.name} for batch tracking"
                f" (required for retrieval)"
            )

    async def _process_gpt_streaming(
        self,
        *,
        source_path: Path,
        source_name: str,
        source_stem: str,
        is_folder: bool,
        total_units: int,
        page_indices: list[int] | None,
        parent_folder: Path,
        temp_jsonl_path: Path,
        output_txt_path: Path,
        transcriber: Any | None,
    ) -> None:
        """Shared GPT streaming flow for PDFs and image folders.

        Applies the page-level resume skip-set and page slice BEFORE any
        rendering, then feeds the in-memory payload producer into either
        batch submission or the synchronous streaming pipeline.
        """
        output_format = getattr(self.user_config, "output_format", "txt") or "txt"

        # Overwrite mode: clear the stale JSONL before computing the skip-set.
        if self.resume_mode == "overwrite" and temp_jsonl_path.exists():
            temp_jsonl_path.write_text("", encoding="utf-8")
            logger.info(f"Cleared stale JSONL cache: {temp_jsonl_path.name}")

        tm = self.model_config.get("transcription_model", {})
        provider = tm.get("provider", "openai")
        model_name = tm.get("name", "")
        img_cfg, model_type, target_dpi, max_pixels = resolve_image_settings(
            provider, model_name
        )

        all_indices = (
            page_indices if page_indices is not None else list(range(total_units))
        )

        # Page-level resume: subtract already-transcribed pages BEFORE
        # rendering anything.
        if is_folder:
            files = list_folder_images(source_path)
            skip_names = (
                compute_folder_skip_names(temp_jsonl_path)
                if self.resume_mode != "overwrite"
                else set()
            )
            needed = [
                i
                for i in all_indices
                if 0 <= i < len(files)
                and folder_image_name(files[i]) not in skip_names
                and files[i].name not in skip_names
            ]
        else:
            skip_indices = (
                compute_pdf_skip_indices(temp_jsonl_path)
                if self.resume_mode != "overwrite"
                else set()
            )
            needed = [
                i for i in all_indices if 0 <= i < total_units and i not in skip_indices
            ]

        skipped = len(all_indices) - len(needed)
        if skipped > 0:
            print_info(
                f"Skipping {skipped} already-processed page(s) (found in JSONL)"
            )
            logger.info(
                f"Skipped {skipped} pages already in {temp_jsonl_path.name}"
            )

        if not needed:
            print_info(
                "All pages already processed. Regenerating output file from JSONL..."
            )
            write_output_from_jsonl(
                temp_jsonl_path,
                output_txt_path,
                self.postprocessing_config,
                output_format=output_format,
            )
            self._cleanup_temp_jsonl(temp_jsonl_path, "gpt")
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        if is_folder:
            print_info(
                f"Streaming {len(needed)} image(s) with in-memory preprocessing..."
            )
            payload_source: AsyncIterator[PagePayload] = stream_folder_payloads(
                source_path,
                img_cfg=img_cfg,
                model_type=model_type,
                page_indices=needed,
            )
        else:
            print_info(
                f"Streaming {len(needed)} page(s) at {target_dpi} DPI"
                f" with in-memory preprocessing..."
            )
            payload_source = stream_pdf_payloads(
                source_path,
                target_dpi=target_dpi,
                img_cfg=img_cfg,
                model_type=model_type,
                max_pixels=max_pixels,
                page_indices=needed,
            )

        file_provenance = build_file_provenance(
            source_path, img_cfg, model_type, max_pixels
        )

        # Batch mode: materialize compact payloads (raw pages are freed per
        # page by the producer), then submit via the batch backend.
        if self.user_config.use_batch_processing:
            payloads = [p async for p in payload_source]
            if await self._handle_batch_submission(
                payloads,
                temp_jsonl_path,
                parent_folder,
                source_stem,
                file_provenance=file_provenance,
            ):
                return

            async def _replay() -> AsyncIterator[PagePayload]:
                for p in payloads:
                    yield p

            payload_source = _replay()
            print_info("Falling back to synchronous processing...")

        print_info(f"Starting gpt transcription for {len(needed)} images...")
        await run_streaming_transcription_pipeline(
            payload_source,
            transcriber,
            temp_jsonl_path,
            output_txt_path,
            source_name,
            self.concurrency_config,
            self.postprocessing_config,
            is_folder=is_folder,
            output_format=output_format,
            file_provenance=file_provenance,
        )
        print_success(
            f"Saved transcription for '{source_name}' -> {output_txt_path.name}"
        )
        self._cleanup_temp_jsonl(temp_jsonl_path, "gpt")
        self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)

    async def process_single_pdf(self, pdf_path: Path, transcriber: Any | None) -> None:
        """
        Processes a single PDF file for transcription based on the user configuration.
        """
        # Resolve per-file context and update transcriber before processing
        if transcriber is not None and not self.user_config.additional_context_path:
            from modules.config.context import resolve_context_for_file

            ctx_content, ctx_path = resolve_context_for_file(pdf_path)
            transcriber.update_context(ctx_content)

        # Resolve per-file context image
        if transcriber is not None:
            if self.user_config.additional_context_image_path:
                transcriber.update_context_image(
                    self.user_config.additional_context_image_path
                )
            else:
                from modules.config.context import resolve_context_image_for_file

                ctx_img = resolve_context_image_for_file(pdf_path)
                transcriber.update_context_image(ctx_img)

        pdf_processor = PDFProcessor(pdf_path)
        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the PDF
            parent_folder, _, temp_jsonl_path = pdf_processor.prepare_output_folder(
                pdf_path.parent
            )
            # Final .txt goes directly next to the PDF
            output_txt_path = pdf_path.parent / create_safe_filename(
                pdf_path.stem, ".txt", pdf_path.parent
            )
        elif self.output_mode == "mirror" and self.input_root is not None:
            mirror_dir = mirror_output_path(
                pdf_path.parent, self.input_root, self.pdf_output_dir
            )
            mirror_dir.mkdir(parents=True, exist_ok=True)
            ext = f".{self.output_format}"
            output_txt_path = mirror_dir / create_safe_filename(
                pdf_path.stem, ext, mirror_dir
            )
            temp_jsonl_name = create_safe_filename(pdf_path.stem, ".jsonl", mirror_dir)
            temp_jsonl_path = mirror_dir / temp_jsonl_name
            if not temp_jsonl_path.exists():
                temp_jsonl_path.touch()
            parent_folder = mirror_dir
        else:
            rel_key = _relative_key(pdf_path, self.input_root)
            parent_folder, output_txt_path, temp_jsonl_path = (
                pdf_processor.prepare_output_folder(
                    self.pdf_output_dir, relative_key=rel_key
                )
            )
        method: str = self.user_config.transcription_method or "gpt"

        print_info(f"Processing PDF: {pdf_path.name}")
        print_info(f"Using method: {method}")

        # Register transient files for cleanup on interruption
        self._transient_tracker.register_jsonl(temp_jsonl_path, method)

        # Resolve page range if configured
        page_indices = None
        if self.user_config.page_range is not None:
            pdf_processor.open_pdf()
            assert pdf_processor.doc is not None
            total_pages = pdf_processor.doc.page_count
            page_indices = self.user_config.page_range.resolve(total_pages)
            if not page_indices:
                print_warning(
                    f"Page range '{self.user_config.page_range.describe()}'"
                    f" yielded no pages for '{pdf_path.name}'"
                    f" ({total_pages} pages). Skipping."
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
            print_warning(
                f"PDF '{pdf_path.name}' is not searchable."
                f" Switching to tesseract method."
            )
            method = "tesseract"  # Fall back to Tesseract (native not possible)

        # Native PDF extraction
        if method == "native":
            text = native_extract_pdf_text(pdf_path, page_indices=page_indices)
            output_format = getattr(self.user_config, "output_format", "txt") or "txt"
            try:
                async with aiofiles.open(
                    temp_jsonl_path, "a", encoding="utf-8"
                ) as jfile:
                    record = {
                        "file_name": pdf_path.name,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "method": "native",
                        "text_chunk": text,
                        "pre_processed_image": None,
                    }
                    await jfile.write(json.dumps(record) + "\n")
                # Write output using central writer
                pages = [{"text": text, "page_number": None, "image_name": None}]
                actual_path = write_transcription_output(
                    pages,
                    output_txt_path,
                    output_format=output_format,
                    postprocess=True,
                    postprocessing_config=self.postprocessing_config,
                )
                print_success(
                    f"Extracted text from '{pdf_path.name}' using native method"
                    f" -> {actual_path.name}"
                )
            except Exception as e:
                logger.exception(
                    f"Error writing native extraction output for {pdf_path.name}: {e}"
                )
                print_error(f"Failed to write output: {e}")

            self._cleanup_temp_jsonl(temp_jsonl_path, method)
            # Mark JSONL as complete (successfully processed)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        # Non-native PDF: Tesseract keeps the file-based pipeline
        if method == "tesseract":
            # Use separate folder and pipeline for Tesseract
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
            self._transient_tracker.register_preprocessed_folder(
                preprocessed_folder, pdf_path.name
            )
            target_dpi = self._resolve_target_dpi(method)
            print_info(
                f"Extracting and preprocessing images for Tesseract"
                f" at {target_dpi} DPI..."
            )
            processed_image_files = await pdf_processor.process_images_for_tesseract(
                preprocessed_folder, target_dpi, page_indices=page_indices
            )
            print_info(f"Extracted {len(processed_image_files)} page images from PDF.")
            print_info(
                f"Starting {method} transcription for"
                f" {len(processed_image_files)} images..."
            )

            await self._process_images_with_method(
                processed_image_files,
                method,
                transcriber,
                temp_jsonl_path,
                output_txt_path,
                pdf_path.name,
            )

            self._cleanup_preprocessed(preprocessed_folder, pdf_path.name)
            print_success(
                f"Saved transcription for PDF '{pdf_path.name}'"
                f" -> {output_txt_path.name}"
            )
            self._cleanup_temp_jsonl(temp_jsonl_path, method)
            # Mark transient files as complete (successfully processed)
            self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        # GPT method: pages are rendered, preprocessed, and encoded fully in
        # memory (no preprocessed_images folder is written).
        if pdf_processor.doc is None:
            pdf_processor.open_pdf()
        assert pdf_processor.doc is not None
        total_pages = pdf_processor.doc.page_count
        pdf_processor.close_pdf()

        await self._process_gpt_streaming(
            source_path=pdf_path,
            source_name=pdf_path.name,
            source_stem=pdf_path.stem,
            is_folder=False,
            total_units=total_pages,
            page_indices=page_indices,
            parent_folder=parent_folder,
            temp_jsonl_path=temp_jsonl_path,
            output_txt_path=output_txt_path,
            transcriber=transcriber,
        )

    async def process_single_image_folder(
        self, folder: Path, transcriber: Any | None
    ) -> None:
        """
        Processes all images in a given folder based on the user configuration.
        """
        # Resolve per-folder context and update transcriber before processing
        if (
            transcriber is not None
            and not self.user_config.additional_context_path
            and getattr(self.user_config, "use_hierarchical_context", True)
        ):
            from modules.config.context import resolve_context_for_folder

            ctx_content, ctx_path = resolve_context_for_folder(folder)
            transcriber.update_context(ctx_content)

        # Resolve per-folder context image
        if transcriber is not None:
            if self.user_config.additional_context_image_path:
                transcriber.update_context_image(
                    self.user_config.additional_context_image_path
                )
            else:
                from modules.config.context import (
                    resolve_context_image_for_folder,
                )

                ctx_img = resolve_context_image_for_folder(folder)
                transcriber.update_context_image(ctx_img)

        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the image folder
            parent_folder, _preprocessed_folder, temp_jsonl_path, _ = (
                ImageProcessor.prepare_image_folder(folder, folder.parent)
            )
            # Final .txt goes directly next to the image folder (one level up)
            output_txt_path = folder.parent / create_safe_filename(
                folder.name, ".txt", folder.parent
            )
        elif self.output_mode == "mirror" and self.input_root is not None:
            mirror_dir = mirror_output_path(
                folder, self.input_root, self.image_output_dir
            )
            mirror_dir.mkdir(parents=True, exist_ok=True)
            ext = f".{self.output_format}"
            output_txt_path = mirror_dir / create_safe_filename(
                folder.name, ext, mirror_dir
            )
            temp_jsonl_name = create_safe_filename(folder.name, ".jsonl", mirror_dir)
            temp_jsonl_path = mirror_dir / temp_jsonl_name
            if not temp_jsonl_path.exists():
                temp_jsonl_path.touch()
            parent_folder = mirror_dir
        else:
            rel_key = _relative_key(folder, self.input_root)
            parent_folder, _preprocessed_folder, temp_jsonl_path, output_txt_path = (
                ImageProcessor.prepare_image_folder(
                    folder, self.image_output_dir, relative_key=rel_key
                )
            )
        method: str = self.user_config.transcription_method or "gpt"

        print_info(f"Processing folder: {folder.name}")
        print_info(f"Using method: {method}")

        # Register transient files for cleanup on interruption
        self._transient_tracker.register_jsonl(temp_jsonl_path, method)

        # Resolve page range for image folders (indices into sorted file list)
        page_indices = None
        if self.user_config.page_range is not None:
            from modules.config.constants import SUPPORTED_IMAGE_EXTENSIONS

            total_images = len(
                [
                    p
                    for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ]
            )
            page_indices = self.user_config.page_range.resolve(total_images)
            if not page_indices:
                print_warning(
                    f"Page range '{self.user_config.page_range.describe()}'"
                    f" yielded no images for '{folder.name}'"
                    f" ({total_images} images). Skipping."
                )
                return
            print_info(
                f"Page range: processing {len(page_indices)} of {total_images} images "
                f"({self.user_config.page_range.describe()})"
            )

        if method == "tesseract" and not self._ensure_tesseract_available():
            return

        # Tesseract keeps the file-based pipeline
        if method == "tesseract":
            preprocessed_folder = parent_folder / "preprocessed_images_tesseract"
            preprocessed_folder.mkdir(exist_ok=True)
            self._transient_tracker.register_preprocessed_folder(
                preprocessed_folder, folder.name
            )
            print_info("Preprocessing images for Tesseract...")
            processed_files = ImageProcessor.process_and_save_images_for_tesseract(
                folder, preprocessed_folder, page_indices=page_indices
            )

            if not processed_files:
                print_warning(f"No images found or processed in {folder}.")
                return

            # Deterministic ordering for folders: sort by filename
            processed_files.sort(key=lambda x: x.name.lower())

            print_info(
                f"Starting {method} transcription for"
                f" {len(processed_files)} images..."
            )
            await self._process_images_with_method(
                processed_files,
                method,
                transcriber,
                temp_jsonl_path,
                output_txt_path,
                folder.name,
                is_folder=True,
            )

            self._cleanup_preprocessed(preprocessed_folder, f"folder '{folder.name}'")
            print_success(
                f"Transcription completed for folder '{folder.name}'"
                f" -> {output_txt_path.name}"
            )
            self._cleanup_temp_jsonl(temp_jsonl_path, method)
            # Mark transient files as complete (successfully processed)
            self._transient_tracker.mark_preprocessed_complete(preprocessed_folder)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        # GPT method: source images are preprocessed and encoded fully in
        # memory (no preprocessed_images folder is written).
        total_images = len(list_folder_images(folder))
        if total_images == 0:
            print_warning(f"No images found or processed in {folder}.")
            return

        try:
            await self._process_gpt_streaming(
                source_path=folder,
                source_name=folder.name,
                source_stem=folder.name,
                is_folder=True,
                total_units=total_images,
                page_indices=page_indices,
                parent_folder=parent_folder,
                temp_jsonl_path=temp_jsonl_path,
                output_txt_path=output_txt_path,
                transcriber=transcriber,
            )
        except RuntimeError as e:
            print_error(f"Skipping folder '{folder.name}': {e}")
            return

    async def _process_images_with_method(
        self,
        image_files: list[Path],
        method: str,
        transcriber: Any | None,
        temp_jsonl_path: Path,
        output_txt_path: Path,
        source_name: str,
        is_folder: bool = False,
    ) -> None:
        """Process images using the specified method.

        Delegates to :func:`modules.transcribe.pipeline.run_transcription_pipeline`.
        """
        output_format = getattr(self.user_config, "output_format", "txt") or "txt"
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
            resume_mode=self.resume_mode,
            output_format=output_format,
        )
