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

from modules.audio.audio_stream import (
    compute_audio_skip_indices,
    stream_audio_chunks,
)
from modules.audio.chunker import (
    ChunkSpec,
    chunk_plan_signature,
    estimate_output_bytes_per_second,
    plan_chunks,
)
from modules.audio.constants import (
    DEFAULT_MIN_CHUNK_SECONDS,
    DEFAULT_TARGET_CHUNK_SECONDS,
    GEMINI_INLINE_LIMIT_BYTES,
    OPENAI_UPLOAD_LIMIT_BYTES,
)
from modules.audio.ffmpeg_runtime import (
    configure_ffmpeg_executables,
    ensure_ffmpeg_available,
    is_ffmpeg_available,
    probe_duration_seconds,
)
from modules.audio.paths import prepare_audio_output
from modules.audio.whisper_runtime import ensure_faster_whisper_available
from modules.batch.submission import submit_batch
from modules.config.service import get_config_service
from modules.documents.epub import EPUBProcessor
from modules.documents.mobi import MOBIProcessor
from modules.documents.pdf import PDFProcessor, native_extract_pdf_text
from modules.images.page_stream import (
    PagePayload,
    _raise_if_failure_rate_excessive,
    compute_folder_skip_names,
    compute_pdf_skip_indices,
    folder_image_name,
    legacy_folder_image_name,
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
    natural_sort_key,
)
from modules.infra.token_budget import (
    check_and_wait_for_token_limit,
    get_token_tracker,
)
from modules.postprocess.writer import resolve_output_path, write_transcription_output
from modules.transcribe.pipeline import (
    BudgetExhaustedError,
    OutputWriteError,
    PageTranscriptionError,
    PayloadHandler,
    PayloadResult,
    build_file_provenance,
    run_streaming_transcription_pipeline,
    run_transcription_pipeline,
    transcribe_audio_payload,
    transcribe_audio_payload_whisper,
    write_output_from_jsonl,
)
from modules.transcribe.resume import ResumeChecker
from modules.transcribe.user_config import UserConfiguration
from modules.ui import print_error, print_info, print_success, print_warning

logger = setup_logger(__name__)

# Methods served by a paid remote API: they share the token-budget gate and the
# usage logging. The local backends (tesseract, whisper, native) are exempt from
# both.
_API_METHODS: tuple[str, ...] = ("gpt", "audio-api")

# Methods served by the audio workflow.
_AUDIO_METHODS: tuple[str, ...] = ("audio-api", "whisper")

# Default emitted-chunk encoding settings when audio_config.chunking omits them.
_DEFAULT_CHUNK_FORMAT = "mp3"
_DEFAULT_SAMPLE_RATE = 16000


def _read_chunk_plan_signature(jsonl_path: Path) -> str | None:
    """Return the chunk-plan signature recorded in *jsonl_path*, if any.

    The signature lives inside the ``file_provenance`` metadata record (not as
    a new top-level JSONL key), so the record schema the image workflow writes
    is untouched. The LAST provenance record wins: a re-run under a changed
    plan appends a fresh one.
    """
    if not jsonl_path.exists():
        return None
    from modules.batch.jsonl import read_jsonl_records

    signature: str | None = None
    for record in read_jsonl_records(jsonl_path):
        provenance = record.get("file_provenance")
        if isinstance(provenance, dict):
            candidate = provenance.get("chunk_plan_signature")
            if isinstance(candidate, str) and candidate:
                signature = candidate
    return signature


@dataclass(frozen=True)
class _AudioChunkSettings:
    """Resolved ``audio_config.chunking`` knobs for one recording."""

    provider: str
    max_request_bytes: int
    target_seconds: int
    overlap_seconds: float
    min_chunk_seconds: int
    chunk_format: str
    mono: bool
    sample_rate: int
    apply_to_local: bool


@dataclass
class ProcessingSummary:
    """Outcome counts for a `process_selected_items` run.

    `processed` counts only items that completed without raising; `failed`
    counts items that raised during processing. `total` is the number of
    items actually selected for processing (after resume filtering), so
    `processed + failed` may be less than `total` when the run is interrupted.

    `skipped` counts items excluded by item-level resume filtering (already
    complete); these are not part of `total`.
    """

    processed: int = 0
    failed: int = 0
    total: int = 0
    skipped: int = 0


def _relative_key(item: Path, input_root: Path | None) -> str | None:
    if input_root is None:
        return None
    try:
        rel = str(item.relative_to(input_root))
    except ValueError:
        return None
    # A single-file --input makes item == input_root, so the relative key
    # degenerates to "." and would hash into a hidden ".-<hash>" output
    # directory (CT-1). Fall back to None so the output directory name is
    # derived from the item's own stem/name instead.
    if rel in (".", ""):
        return None
    return rel


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
        *,
        audio_config: dict[str, Any] | None = None,
    ) -> None:
        self.user_config = user_config
        self.paths_config = paths_config
        self.model_config = model_config
        self.concurrency_config = concurrency_config
        self.image_processing_config = image_processing_config
        self.processing_settings = paths_config.get("general", {})
        # Audio settings are optional and read from their own file; callers that
        # already resolved them (CLI overrides) pass them in.
        self.audio_config: dict[str, Any] = (
            audio_config
            if audio_config is not None
            else get_config_service().get_audio_config()
        )

        # Configure Tesseract and FFmpeg executables if provided. Both only set
        # module-level state, so they are cheap on every construction.
        configure_tesseract_executable(image_processing_config)
        configure_ffmpeg_executables(self.audio_config)
        self.ocr_config = image_processing_config.get(
            "tesseract_image_processing", {}
        ).get("ocr", {})

        # Load post-processing configuration from image_processing_config
        self.postprocessing_config = image_processing_config.get("postprocessing", {})
        # Speech transcripts get their own post-processing profile when the
        # audio config defines one; None means "fall back to the image profile".
        self.audio_postprocessing_config: dict[str, Any] | None = (
            self.audio_config.get("postprocessing") or None
        )

        # Resolve output directories via PathConfig
        pc = PathConfig.from_paths_config(paths_config)
        self.use_input_as_output = pc.use_input_as_output
        self.pdf_output_dir = pc.pdf_output_dir
        self.image_output_dir = pc.image_output_dir
        self.epub_output_dir = pc.epub_output_dir
        self.mobi_output_dir = pc.mobi_output_dir
        # Created lazily by the audio workflow: ensure_output_dirs() must not
        # materialize an audio_out/ directory for image or PDF runs.
        self.audio_output_dir = pc.audio_output_dir
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
            audio_output_dir=self.audio_output_dir,
            output_format=self.output_format,
            output_mode=self.output_mode,
            input_root=self.input_root,
            retry_errors=getattr(user_config, "retry_errors", False),
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
        if (
            self.user_config.transcription_method not in _API_METHODS
            and phase != "Initial"
        ):
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
        skipped_count = 0
        if self.resume_mode != "overwrite" and processing_type:
            selected, skipped = self.resume_checker.filter_items(
                selected, processing_type
            )
            skipped_count = len(skipped)
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
                # Check token limit before starting each new item (GPT method
                # only; batch mode is exempt from token limiting entirely).
                if (
                    self.user_config.transcription_method in _API_METHODS
                    and not self.user_config.use_batch_processing
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
                    elif self.user_config.processing_type == "audio":
                        await self.process_single_audio(item, transcriber)
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
            processed=processed_count,
            failed=failed_count,
            total=total_items,
            skipped=skipped_count,
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

        # Resolve page/section range if configured. The real section count is
        # only known after extraction, so we extract ALL sections and slice
        # against that count. Resolving up front against a 2**31 sentinel is
        # unsafe: `last:N` lands near 2**31 and the processor's `0 <= i < len`
        # filter drops every index -> silently empty output written with a
        # success message; an open span (`3-`) materializes ~2.1 billion indices
        # into a set -> hang/OOM (CT-10). Indices address non-empty extracted
        # sections in reading order.
        page_range = self.user_config.page_range
        if page_range is not None and page_range.is_empty_spec():
            page_range = None
        if page_range is not None:
            print_info(
                f"Page range: {page_range.describe()} "
                f"(applied to {format_label} sections)"
            )

        processor = processor_cls(file_path)
        try:
            extraction = processor.extract_text()
        except Exception as exc:
            logger.exception(
                "Failed to extract %s %s: %s", format_label, file_path.name, exc
            )
            print_error(f"Failed to extract text from {file_path.name}.")
            return

        if page_range is not None:
            total_sections = len(extraction.sections)
            keep = page_range.resolve(total_sections)
            if not keep:
                # A valid range that selects nothing (e.g. section 50 of a
                # 3-section book). Warn and skip rather than write empty output
                # with a success message, matching the PDF page-range path.
                print_warning(
                    f"Page range '{page_range.describe()}' selected no "
                    f"{format_label} sections for '{file_path.name}' "
                    f"({total_sections} section(s)). Skipping."
                )
                return
            extraction.sections = [extraction.sections[i] for i in keep]

        # Determine output directory and prepare working folder
        if self.use_input_as_output:
            # Working files go in a hash-suffixed subdirectory next to the ebook
            _parent_folder, _ = processor.prepare_output_folder(file_path.parent)
            # Final .txt goes directly next to the ebook file
            output_txt_path = file_path.parent / create_safe_filename(
                file_path.stem, ".txt", file_path.parent
            )
        else:
            # Key the output folder by the input-relative path (not the bare
            # stem) so two same-stem ebooks in different subdirectories under a
            # recursive --input do not overwrite each other's output (and later
            # both resume COMPLETE). Mirrors the PDF/image-folder path.
            rel_key = _relative_key(file_path, self.input_root)
            _parent_folder, output_txt_path = processor.prepare_output_folder(
                default_output_dir, relative_key=rel_key
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

    def _skip_redundant_batch_submission(
        self, temp_jsonl_path: Path, source_name: str
    ) -> bool:
        """Return True when a batch submission must be skipped for this item.

        Two protected states, both detected from the item's temp JSONL:

        - A batch is already pending (tracking records without a subsequent
          ``finalized`` marker): resubmitting would pay for every page a
          second time and append duplicate results.
        - The JSONL was already finalized by check_batches (and not
          resubmitted): batch page-level resume cannot subtract finalized
          pages, so a resubmission (e.g. via ``--retry-errors``) would
          resubmit ALL pages, not just failed ones.

        Overwrite resume mode clears the temp JSONL before this point, so the
        guard never blocks a deliberate redo.
        """
        if not temp_jsonl_path.exists():
            return False
        from modules.batch.status import _parse_temp_file_metadata

        try:
            meta = _parse_temp_file_metadata(temp_jsonl_path)
        except OSError as e:
            logger.warning(
                "Could not inspect %s for pending batches: %s",
                temp_jsonl_path.name,
                e,
            )
            return False
        finalized = "finalized" in meta["batch_session_statuses"]
        has_pending = bool(meta["post_finalize_batch_ids"]) or (
            not finalized and bool(meta["batch_ids"])
        )
        if has_pending:
            print_warning(
                f"A batch for '{source_name}' is already pending; skipping "
                f"resubmission. Run check_batches to retrieve it, or use "
                f"resume mode 'overwrite' to discard it and resubmit."
            )
            return True
        if finalized:
            print_warning(
                f"'{source_name}' was already finalized from a batch run; "
                f"skipping batch resubmission (it would resubmit every page). "
                f"Use repair for failed pages, or resume mode "
                f"'overwrite' to redo the whole item."
            )
            return True
        return False

    async def _handle_batch_submission(
        self,
        payloads: list[PagePayload],
        temp_jsonl_path: Path,
        parent_folder: Path,
        source_stem: str,
        file_provenance: dict[str, Any] | None = None,
    ) -> bool:
        """Try to submit a batch job; return True if submitted, False to fall
        through to synchronous processing.

        A genuine submission failure raises ``BatchSubmissionError`` unless the
        user opted into ``--sync-fallback``; propagating it makes the item count
        as failed (non-zero exit) rather than silently reprocessing the whole
        book at full synchronous price (decision 5).
        """
        if not self.user_config.use_batch_processing:
            return False
        from modules.batch.submission import BatchSubmissionError

        try:
            handle = await self._submit_batch_with_backend(
                payloads,
                temp_jsonl_path,
                parent_folder,
                source_stem,
                file_provenance=file_provenance,
            )
        except BatchSubmissionError:
            if getattr(self.user_config, "sync_fallback", False):
                print_warning(
                    f"Batch submission failed for '{source_stem}'; falling back to "
                    f"synchronous processing (--sync-fallback enabled)."
                )
                return False
            raise
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

    def _withhold_partial_output(
        self, output_txt_path: Path, output_format: str
    ) -> None:
        """Remove a finalized output written before a budget-partial give-up.

        The streaming pipeline finalizes the output from the JSONL on every
        pass, including the pass that exhausted the budget; that file omits the
        deferred pages yet looks complete. Removing it leaves only the temp
        JSONL, from which page-level resume rebuilds the full output on the next
        run. The JSONL and its resume marker are left untouched. Resolves the
        real extension so md/json outputs are withheld too.
        """
        actual_path = resolve_output_path(output_txt_path, output_format)
        try:
            if actual_path.exists():
                actual_path.unlink()
                print_info(
                    f"Withheld partial output {actual_path.name}; it will be "
                    f"rebuilt from the JSONL on resume."
                )
        except OSError as e:
            logger.warning("Could not remove partial output %s: %s", actual_path, e)

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

        # Guard a zero-page source (an empty/corrupt PDF, or a folder that lost
        # its images) up front. Without it the resume path prints "All pages
        # already processed. Regenerating..." and then raises OutputWriteError
        # from an empty JSONL — a misleading failure. Mirror the folder-path
        # "No images found" guard. (page_indices is always non-empty here; an
        # empty page range is skipped by the callers before this point.)
        resolved_total = len(page_indices) if page_indices is not None else total_units
        if resolved_total == 0:
            print_warning(f"No pages found in '{source_name}'.")
            return

        # Overwrite mode: clear the stale JSONL before computing the skip-set.
        if self.resume_mode == "overwrite" and temp_jsonl_path.exists():
            temp_jsonl_path.write_text("", encoding="utf-8")
            logger.info(f"Cleared stale JSONL cache: {temp_jsonl_path.name}")

        tm = self.model_config.get("transcription_model", {})
        provider = tm.get("provider", "openai")
        model_name = tm.get("name", "")
        img_cfg, model_type, target_dpi, max_pixels, render_strategy = (
            resolve_image_settings(provider, model_name)
        )

        all_indices = (
            page_indices if page_indices is not None else list(range(total_units))
        )

        def compute_needed(first_pass: bool) -> list[int]:
            """Page indices not yet recorded in the temp JSONL.

            Recomputed each pass so a budget re-pass sees pages completed by
            the previous pass. On the first pass in overwrite mode the JSONL was
            already cleared, so the skip-set is empty; later passes always read
            the live JSONL to skip what this run has written so far.
            """
            use_skip = (not first_pass) or self.resume_mode != "overwrite"
            retry_errors = getattr(self.user_config, "retry_errors", False)
            if is_folder:
                files = list_folder_images(source_path)
                skip_names = (
                    compute_folder_skip_names(
                        temp_jsonl_path, exclude_errors=retry_errors
                    )
                    if use_skip
                    else set()
                )
                return [
                    i
                    for i in all_indices
                    if 0 <= i < len(files)
                    and folder_image_name(files[i]) not in skip_names
                    and legacy_folder_image_name(files[i]) not in skip_names
                    and files[i].name not in skip_names
                ]
            skip_indices = (
                compute_pdf_skip_indices(temp_jsonl_path, exclude_errors=retry_errors)
                if use_skip
                else set()
            )
            return [
                i for i in all_indices if 0 <= i < total_units and i not in skip_indices
            ]

        def build_source(indices: list[int]) -> AsyncIterator[PagePayload]:
            """Build a fresh streaming page source over the given indices.

            Rebuilt each pass: the in-memory producer is a one-shot async
            iterator, so a budget re-pass needs a new source over the still-
            pending pages.
            """
            if is_folder:
                return stream_folder_payloads(
                    source_path,
                    img_cfg=img_cfg,
                    model_type=model_type,
                    page_indices=indices,
                )
            return stream_pdf_payloads(
                source_path,
                target_dpi=target_dpi,
                img_cfg=img_cfg,
                model_type=model_type,
                max_pixels=max_pixels,
                render_strategy=render_strategy,
                page_indices=indices,
            )

        # Refuse to resume an artifact written by an incompatible resume
        # format (decision 1); overwrite mode already cleared it above.
        if self.resume_mode != "overwrite":
            from modules.batch.jsonl import verify_resume_compatible

            verify_resume_compatible(temp_jsonl_path)

        # Page-level resume: subtract already-transcribed pages BEFORE
        # rendering anything.
        needed = compute_needed(first_pass=True)

        skipped = len(all_indices) - len(needed)
        if skipped > 0:
            print_info(f"Skipping {skipped} already-processed page(s) (found in JSONL)")
            logger.info(f"Skipped {skipped} pages already in {temp_jsonl_path.name}")

        if not needed:
            print_info(
                "All pages already processed. Regenerating output file from JSONL..."
            )
            if not write_output_from_jsonl(
                temp_jsonl_path,
                output_txt_path,
                self.postprocessing_config,
                output_format=output_format,
            ):
                # Raise BEFORE cleanup/mark-complete: a failed regeneration must
                # not delete the temp JSONL (the only copy of the transcriptions)
                # nor file the item as complete. Propagating counts it failed and
                # leaves the JSONL for the next resume run.
                raise OutputWriteError(source_name)
            self._cleanup_temp_jsonl(temp_jsonl_path, "gpt")
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        file_provenance = build_file_provenance(
            source_path, img_cfg, model_type, max_pixels
        )

        # Batch mode: materialize compact payloads (raw pages are freed per
        # page by the producer), then submit via the batch backend. Batch is
        # exempt from token limiting (it is pre-priced and submitted whole).
        if self.user_config.use_batch_processing:
            # Batch results are finalized by check_batches and never written
            # back as per-page transcription records, so page-level resume
            # cannot see them: without this guard a re-run before (or after)
            # finalization resubmits the ENTIRE document as a second paid
            # batch and doubles/clobbers the output.
            if self._skip_redundant_batch_submission(temp_jsonl_path, source_name):
                return
            print_info(
                f"Streaming {len(needed)} page(s) with in-memory preprocessing..."
            )
            payloads = [p async for p in build_source(needed)]
            if await self._handle_batch_submission(
                payloads,
                temp_jsonl_path,
                parent_folder,
                source_stem,
                file_provenance=file_provenance,
            ):
                return
            print_info("Falling back to synchronous processing...")

        # Synchronous streaming with a chunk/page-level token-budget loop. Each
        # pass admits pages until the daily budget is exhausted, then drains,
        # waits for the daily reset, and re-passes over the still-pending pages
        # (a fresh source rebuilt over the recomputed skip-set).
        print_info(f"Starting gpt transcription for {len(needed)} images...")
        tracker = get_token_tracker()
        # Per-key token-accounting stamp so the daily-reset wait applies the
        # right per-key pool cap and names the exhausted key.
        token_stamp = getattr(
            getattr(transcriber, "provider", None), "token_stamp", None
        )
        first_provenance: dict[str, Any] | None = file_provenance
        completed_fully = True
        stalled_resets = 0
        # A page-level failure on the SAME pass that exhausted the budget is
        # remembered here. The pipeline finalizes the (truncated) output then
        # raises PageTranscriptionError BEFORE we can inspect `exhausted`, so a
        # bare propagation would skip the withhold + JSONL-protection flow and
        # leave a truncated txt that resume files COMPLETE (CT-7). We catch it,
        # fold it into the budget re-pass/withhold flow, and re-raise an
        # appropriate failure once the loop settles.
        page_failure: PageTranscriptionError | None = None
        while True:
            exhausted = asyncio.Event()
            try:
                await run_streaming_transcription_pipeline(
                    build_source(needed),
                    transcriber,
                    temp_jsonl_path,
                    output_txt_path,
                    source_name,
                    self.concurrency_config,
                    self.postprocessing_config,
                    is_folder=is_folder,
                    output_format=output_format,
                    file_provenance=first_provenance,
                    tracker=tracker,
                    exhausted=exhausted,
                    total_pages=len(needed),
                )
            except PageTranscriptionError as pte:
                if not exhausted.is_set():
                    # No budget deferral: pages genuinely failed and the output
                    # (with error placeholders) is complete. Preserve the CT-2
                    # contract — propagate unchanged so the item counts failed
                    # and the JSONL/output survive for --errors-only repair.
                    raise
                # Budget exhausted on the same pass a page failed. Remember the
                # newest failure (its counts reflect the latest pass) and fall
                # through to the wait/re-pass/withhold flow below.
                page_failure = pte
            if not exhausted.is_set():
                break

            # Provenance is written once, on the first pass.
            first_provenance = None
            before = len(needed)
            needed = compute_needed(first_pass=False)
            made_progress = len(needed) < before
            if not needed:
                break

            print_warning(
                f"Daily token budget reached; {len(needed)} page(s) deferred. "
                f"Waiting for daily reset..."
            )
            # Reservation-aware: admission control defers pages on a per-page
            # reservation estimate while actual usage is still just under the
            # cap, so the plain is_limit_reached() check would return instantly
            # and spin this loop without progress. would_block_next_page() makes
            # the wait actually wait until the daily reset frees enough budget.
            if not await check_and_wait_for_token_limit(
                self.concurrency_config, reservation_aware=True, stamp=token_stamp
            ):
                completed_fully = False
                break

            # Safeguard: if even a full day's reset yields no progress twice
            # running, a single page exceeds the entire daily budget; stop.
            if not made_progress:
                stalled_resets += 1
                if stalled_resets >= 2:
                    print_warning(
                        "A single page appears to exceed the entire daily token "
                        "budget; stopping. Raise daily_tokens to process the "
                        "remaining pages."
                    )
                    completed_fully = False
                    break
            else:
                stalled_resets = 0

        if completed_fully and page_failure is None:
            print_success(
                f"Saved transcription for '{source_name}' -> {output_txt_path.name}"
            )
            self._cleanup_temp_jsonl(temp_jsonl_path, "gpt")
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
        elif completed_fully:
            # All deferred pages were eventually transcribed, but a page failed
            # on an earlier pass. The finalized output is COMPLETE with error
            # placeholders (the CT-2 case), so it must NOT be withheld — resume
            # would only rebuild the same placeholders. Leave the output and the
            # JSONL in place (matching the non-budget CT-2 path, which propagates
            # without cleanup) and re-raise so the item counts failed.
            assert page_failure is not None  # narrowed by the branch conditions
            print_warning(
                f"Transcription for '{source_name}' completed with "
                f"{page_failure.failed_pages} failed page(s); output retained "
                f"with error placeholders."
            )
            raise page_failure
        else:
            # Budget exhausted mid-document (or the wait was cancelled / the
            # per-page estimate exceeds the daily limit). The streaming pipeline
            # already finalized the output from the JSONL on the exhausting pass,
            # so a truncated {stem}.txt would look complete to
            # resume/orchestration. Withhold it and keep only the temp JSONL
            # (with its resume marker) so page-level resume rebuilds the full
            # output on the next run. Remove the JSONL from the transient cleanup
            # list so a False retain_temporary_jsonl setting cannot delete the
            # resume artifact when this item is counted as failed.
            deferred_pages = len(needed)
            completed_pages = len(all_indices) - deferred_pages
            self._withhold_partial_output(output_txt_path, output_format)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            if page_failure is not None:
                # Both a budget deferral AND a page failure occurred. The
                # withheld output is rebuilt from the JSONL on resume; the page
                # failure's error placeholder is already in the JSONL and
                # resurfaces (as PageTranscriptionError) on the resume run that
                # finalizes the output. Report both now; the budget deferral is
                # why we withhold and need resume, so raise BudgetExhaustedError.
                print_info(
                    f"Partial transcription for '{source_name}': {deferred_pages} "
                    f"page(s) deferred by the daily token budget and "
                    f"{page_failure.failed_pages} page(s) failed. Withheld the "
                    f"partial output; {completed_pages} completed/attempted "
                    f"page(s) retained in {temp_jsonl_path.name} for resume."
                )
            else:
                print_info(
                    f"Partial transcription for '{source_name}': {deferred_pages} "
                    f"page(s) deferred by the daily token budget. Withheld the "
                    f"partial output; {completed_pages} completed page(s) retained "
                    f"in {temp_jsonl_path.name} for resume on the next run."
                )
            raise BudgetExhaustedError(
                source_name,
                deferred_pages=deferred_pages,
                completed_pages=completed_pages,
            )

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
            # Close the handle opened only to resolve the page range. The native
            # path opens its own handle, the Tesseract preprocessor reopens as
            # needed, and the GPT branch reopens below, so nothing downstream
            # depends on this one staying open — leaving it open leaked the file
            # handle on the native/Tesseract paths.
            pdf_processor.close_pdf()

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
                    await jfile.write(json.dumps(record, ensure_ascii=False) + "\n")
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
                # Propagate so the item counts failed and cleanup/mark-complete
                # below is skipped, preserving the temp JSONL for resume rather
                # than filing a broken item as processed.
                raise

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

            # A page that fails to render/preprocess is silently dropped by the
            # PDF Tesseract preprocessor (it returns only the files that were
            # written, and [] on wholesale failure). Without this guard the item
            # would report success with pages permanently missing. Compare the
            # expected page count against what was produced: below the threshold
            # a warning is emitted, above it the item is raised as failed.
            if page_indices is not None:
                expected_pages = len(page_indices)
            elif pdf_processor.doc is not None:
                expected_pages = int(pdf_processor.doc.page_count)
            else:
                expected_pages = len(processed_image_files)
            _raise_if_failure_rate_excessive(
                pdf_path.name,
                expected_pages,
                expected_pages - len(processed_image_files),
            )

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

    # ------------------------------------------------------------------
    # Audio workflow
    # ------------------------------------------------------------------

    def _audio_chunk_settings(self) -> _AudioChunkSettings:
        """Read ``audio_config.chunking`` with its documented defaults."""
        chunking = self.audio_config.get("chunking", {}) or {}
        section = self.audio_config.get("audio_transcription", {}) or {}
        provider = str(section.get("provider") or "openai").strip().lower()
        provider_limit = (
            GEMINI_INLINE_LIMIT_BYTES
            if provider == "google"
            else OPENAI_UPLOAD_LIMIT_BYTES
        )
        configured_cap = int(chunking.get("max_request_bytes") or 0)
        return _AudioChunkSettings(
            provider=provider,
            max_request_bytes=configured_cap or provider_limit,
            target_seconds=int(
                chunking.get("target_seconds") or DEFAULT_TARGET_CHUNK_SECONDS
            ),
            overlap_seconds=float(chunking.get("overlap_seconds") or 0.0),
            min_chunk_seconds=DEFAULT_MIN_CHUNK_SECONDS,
            chunk_format=str(chunking.get("chunk_format") or _DEFAULT_CHUNK_FORMAT),
            mono=bool(chunking.get("mono", True)),
            sample_rate=int(chunking.get("sample_rate") or _DEFAULT_SAMPLE_RATE),
            apply_to_local=bool(chunking.get("apply_to_local", False)),
        )

    def _plan_audio_specs(
        self,
        audio_path: Path,
        method: str,
        settings: _AudioChunkSettings,
        *,
        page_range_active: bool,
    ) -> tuple[list[ChunkSpec] | None, bool]:
        """Plan the chunks one recording is cut into.

        Probing costs an ffprobe subprocess, so it is skipped entirely where it
        cannot change the answer: a local Whisper run without
        ``chunking.apply_to_local`` sends the file whole (faster-whisper handles
        long inputs natively) and no page range is in play.

        Returns:
            ``(specs, planned)``. ``specs`` is None when the item must be
            skipped — FFmpeg is required to split an oversized recording but is
            unavailable, mirroring the Tesseract "runtime missing" skip.
            ``planned`` is False when the plan is an unprobed whole-file
            fallback, in which case chunk-index page ranges are meaningless.
        """
        whole_file = [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]
        size_bytes = audio_path.stat().st_size

        if not (page_range_active or method == "audio-api" or settings.apply_to_local):
            return whole_file, False

        # Only the remote venues enforce a payload ceiling; a local run may
        # always fall back to the whole file.
        oversized = method == "audio-api" and size_bytes > settings.max_request_bytes
        available = ensure_ffmpeg_available() if oversized else is_ffmpeg_available()
        if not available:
            if oversized:
                print_error(
                    f"'{audio_path.name}' is {size_bytes / 1048576:.1f} MB and must "
                    f"be split for the {settings.provider} API, but FFmpeg is "
                    f"unavailable. Skipping."
                )
                return None, False
            logger.info(
                "FFmpeg unavailable; sending %s as a single request.", audio_path.name
            )
            return whole_file, False

        duration = probe_duration_seconds(audio_path)
        if duration is None or duration <= 0:
            if oversized:
                logger.error(
                    "Could not probe the duration of %s; it cannot be split.",
                    audio_path.name,
                )
                print_error(
                    f"Could not determine the duration of '{audio_path.name}';"
                    f" skipping (it exceeds the per-request size limit)."
                )
                return None, False
            print_warning(
                f"Could not determine the duration of '{audio_path.name}';"
                f" transcribing it as a single request."
            )
            return whole_file, False

        specs = plan_chunks(
            duration_seconds=duration,
            size_bytes=size_bytes,
            target_seconds=settings.target_seconds,
            max_request_bytes=settings.max_request_bytes,
            output_bytes_per_second=estimate_output_bytes_per_second(
                settings.chunk_format,
                settings.sample_rate,
                settings.mono,
                source_bytes_per_second=size_bytes / duration,
            ),
            overlap_seconds=settings.overlap_seconds,
            min_chunk_seconds=settings.min_chunk_seconds,
        )
        if len(specs) > 1:
            print_info(
                f"Planned {len(specs)} chunk(s) for {duration / 60:.1f} minute(s) "
                f"of audio."
            )
        return specs, True

    def _audio_handler(
        self, method: str, transcriber: Any | None
    ) -> tuple[PayloadHandler | None, Any, asyncio.Event | None]:
        """Resolve the per-chunk handler, token tracker, and budget event.

        Returns ``(None, None, None)`` when the local runtime is missing, so the
        caller skips the item the way the Tesseract path does instead of
        counting it as a failure. Only the remote method is budgeted; the local
        one costs no tokens.
        """
        if method == "audio-api":
            if transcriber is None:
                raise ValueError(
                    "No audio transcriber was opened for the 'audio-api' method."
                )
            return transcribe_audio_payload, get_token_tracker(), asyncio.Event()

        if not ensure_faster_whisper_available():
            return None, None, None
        whisper_cfg = self.audio_config.get("local_whisper", {}) or {}

        async def _whisper_handler(payload: Any, _transcriber: Any) -> PayloadResult:
            return await transcribe_audio_payload_whisper(payload, whisper_cfg)

        return _whisper_handler, None, None

    def _audio_concurrency_config(self) -> dict[str, Any]:
        """Concurrency config with the audio concurrency override applied.

        ``audio_transcription.concurrency_limit`` overrides
        ``concurrency.transcription.concurrency_limit`` for audio runs only, so
        speech requests can be paced independently of image transcription. The
        shared config object is copied, never mutated.
        """
        section = self.audio_config.get("audio_transcription", {}) or {}
        limit = section.get("concurrency_limit")
        if not limit:
            return self.concurrency_config
        concurrency = dict(self.concurrency_config.get("concurrency", {}) or {})
        transcription = dict(concurrency.get("transcription", {}) or {})
        transcription["concurrency_limit"] = int(limit)
        concurrency["transcription"] = transcription
        return {**self.concurrency_config, "concurrency": concurrency}

    def _audio_file_provenance(
        self,
        audio_path: Path,
        method: str,
        settings: _AudioChunkSettings,
        specs: list[ChunkSpec],
        signature: str,
    ) -> dict[str, Any]:
        """File-level reproducibility record for one audio run.

        Uses the same top-level ``file_provenance`` key the image workflow
        writes, so every JSONL reader still classifies it as metadata. The
        chunk-plan signature rides INSIDE that dict: resume compares it against
        the current plan before honoring any skip set, because chunk indices
        are positional and shift when the plan's parameters change.
        """
        section = self.audio_config.get("audio_transcription", {}) or {}
        if method == "audio-api":
            venue = settings.provider
            model = str((section.get(settings.provider) or {}).get("model", "") or "")
        else:
            venue = "faster-whisper"
            local = self.audio_config.get("local_whisper", {}) or {}
            model = str(local.get("model_path") or local.get("model_size") or "")
        return {
            "file_provenance": {
                "source_file": str(audio_path),
                "source_bytes": audio_path.stat().st_size,
                "method": method,
                "provider": venue,
                "model": model,
                "chunk_plan_signature": signature,
                "chunk_count": len(specs),
                "chunking": {
                    "target_seconds": settings.target_seconds,
                    "overlap_seconds": settings.overlap_seconds,
                    "max_request_bytes": settings.max_request_bytes,
                    "chunk_format": settings.chunk_format,
                    "mono": settings.mono,
                    "sample_rate": settings.sample_rate,
                },
                "timestamp": datetime.datetime.now().isoformat(),
            }
        }

    def _audio_resume_skip(
        self, temp_jsonl_path: Path, source_name: str, signature: str
    ) -> set[int]:
        """Chunk indices already transcribed, honored only for the same plan.

        Chunk indices are positional: a changed ``chunking`` setting (or a
        different page range) moves every boundary, so records written under the
        old plan describe different audio. On a signature mismatch the artifact
        is therefore reset rather than merged — a plan with fewer chunks would
        otherwise leave the old plan's trailing records in the JSONL, and the
        transcript (deduplicated per chunk name, not per plan) would silently
        splice a segment from the previous segmentation into the new one.
        """
        if self.resume_mode == "overwrite" or not temp_jsonl_path.exists():
            return set()
        skip = compute_audio_skip_indices(
            temp_jsonl_path,
            exclude_errors=getattr(self.user_config, "retry_errors", False),
        )
        if not skip:
            return set()
        stored = _read_chunk_plan_signature(temp_jsonl_path)
        if stored is not None and stored != signature:
            print_warning(
                f"The chunk plan for '{source_name}' changed since the last run "
                f"({stored} -> {signature}); discarding the stale chunk records "
                f"and re-transcribing the recording."
            )
            temp_jsonl_path.write_text("", encoding="utf-8")
            logger.info(
                "Reset %s: chunk plan changed from %s to %s.",
                temp_jsonl_path.name,
                stored,
                signature,
            )
            return set()
        print_info(
            f"Skipping {len(skip)} already-transcribed chunk(s) (found in JSONL)"
        )
        return skip

    async def process_single_audio(
        self, audio_path: Path, transcriber: Any | None
    ) -> None:
        """Transcribe a single audio recording.

        Structurally the PDF flow's sibling — resolve the output paths, plan the
        units of work, subtract what the temp JSONL already holds, stream the
        rest through the shared transcription pipeline, and rebuild the
        transcript from the JSONL — with time-sliced chunks in place of pages.
        Always synchronous: neither venue offers a batch speech-to-text API, so
        this path never reaches ``modules/batch``.
        """
        method = self.user_config.transcription_method or ""
        if method not in _AUDIO_METHODS:
            logger.error(
                "Unsupported transcription method %r for audio item %s",
                method,
                audio_path.name,
            )
            raise ValueError(
                f"Audio transcription requires method 'audio-api' or 'whisper'; "
                f"got {method!r}."
            )

        print_info(f"Processing audio: {audio_path.name}")
        print_info(f"Using method: {method}")

        if self.user_config.use_batch_processing:
            print_warning(
                "Audio transcription has no batch API; processing synchronously."
            )

        # Created lazily, so a non-audio run never materializes audio_out/.
        if not self.use_input_as_output:
            self.audio_output_dir.mkdir(parents=True, exist_ok=True)

        parent_folder, output_txt_path, temp_jsonl_path = prepare_audio_output(
            audio_path,
            output_dir=self.audio_output_dir,
            input_paths_is_output_path=self.use_input_as_output,
            output_mode=self.output_mode,
            input_root=self.input_root,
            output_format=self.output_format,
        )
        self._transient_tracker.register_jsonl(temp_jsonl_path, method)

        settings = self._audio_chunk_settings()
        page_range = self.user_config.page_range
        if page_range is not None and page_range.is_empty_spec():
            page_range = None

        specs, planned = self._plan_audio_specs(
            audio_path,
            method,
            settings,
            page_range_active=page_range is not None,
        )
        if specs is None:
            return

        if page_range is not None and not planned:
            print_warning(
                "Page ranges select audio chunks and require FFmpeg;"
                f" transcribing all of '{audio_path.name}' instead."
            )
            page_range = None
        if page_range is not None:
            keep = set(page_range.resolve(len(specs)))
            if not keep:
                print_warning(
                    f"Page range '{page_range.describe()}' selected no chunks for "
                    f"'{audio_path.name}' ({len(specs)} chunk(s)). Skipping."
                )
                return
            if len(keep) < len(specs):
                print_info(
                    f"Page range: processing {len(keep)} of {len(specs)} chunks "
                    f"({page_range.describe()})"
                )
                specs = [spec for spec in specs if spec.index in keep]

        # Overwrite mode clears the stale JSONL before the skip set is read;
        # otherwise refuse an artifact written by an incompatible format.
        if self.resume_mode == "overwrite" and temp_jsonl_path.exists():
            temp_jsonl_path.write_text("", encoding="utf-8")
            logger.info(f"Cleared stale JSONL cache: {temp_jsonl_path.name}")
        else:
            from modules.batch.jsonl import verify_resume_compatible

            verify_resume_compatible(temp_jsonl_path)

        signature = chunk_plan_signature(specs)
        skip = self._audio_resume_skip(temp_jsonl_path, audio_path.name, signature)

        postprocessing_config = (
            self.audio_postprocessing_config or self.postprocessing_config
        )

        needed = [spec for spec in specs if spec.index not in skip]
        if not needed:
            print_info(
                "All chunks already transcribed. Regenerating output file from JSONL..."
            )
            if not write_output_from_jsonl(
                temp_jsonl_path,
                output_txt_path,
                postprocessing_config,
                output_format=self.output_format,
            ):
                raise OutputWriteError(audio_path.name)
            self._cleanup_temp_jsonl(temp_jsonl_path, method)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            return

        handler, tracker, exhausted = self._audio_handler(method, transcriber)
        if handler is None:
            return

        # A whole-file plan never writes a chunk file, so the working directory
        # is only created (and registered for cleanup) when cutting is real.
        work_dir = parent_folder / "audio_chunks"
        multi_chunk = len(specs) > 1
        if multi_chunk:
            work_dir.mkdir(parents=True, exist_ok=True)
            self._transient_tracker.register_preprocessed_folder(
                work_dir, audio_path.name
            )

        print_info(f"Starting {method} transcription for {len(needed)} chunk(s)...")
        page_failure: PageTranscriptionError | None = None
        try:
            await run_streaming_transcription_pipeline(
                stream_audio_chunks(
                    audio_path,
                    specs=specs,
                    work_dir=work_dir,
                    chunk_format=settings.chunk_format,
                    mono=settings.mono,
                    sample_rate=settings.sample_rate,
                    skip_indices=skip,
                ),
                transcriber,
                temp_jsonl_path,
                output_txt_path,
                audio_path.name,
                self._audio_concurrency_config(),
                postprocessing_config,
                is_folder=False,
                output_format=self.output_format,
                file_provenance=self._audio_file_provenance(
                    audio_path, method, settings, specs, signature
                ),
                tracker=tracker,
                exhausted=exhausted,
                total_pages=len(needed),
                method=method,
                handler=handler,
            )
        except PageTranscriptionError as pte:
            # No budget deferral: the output is complete (with error
            # placeholders), so propagate unchanged and keep every artifact.
            if exhausted is None or not exhausted.is_set():
                raise
            page_failure = pte

        if exhausted is not None and exhausted.is_set():
            # The daily budget ran out mid-recording. The pipeline skips the
            # final write on an exhausting pass, but an earlier pass may have
            # left one behind: withhold it so a truncated transcript is never
            # mistaken for a complete one, and protect the JSONL from cleanup so
            # chunk-level resume rebuilds the transcript after the daily reset.
            transcribed = compute_audio_skip_indices(temp_jsonl_path)
            deferred = [spec for spec in specs if spec.index not in transcribed]
            completed = len(specs) - len(deferred)
            self._withhold_partial_output(output_txt_path, self.output_format)
            self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)
            failure_note = (
                f" and {page_failure.failed_pages} chunk(s) failed"
                if page_failure is not None
                else ""
            )
            print_info(
                f"Partial transcription for '{audio_path.name}': {len(deferred)} "
                f"chunk(s) deferred by the daily token budget{failure_note}. "
                f"Withheld the partial output; {completed} completed chunk(s) "
                f"retained in {temp_jsonl_path.name} for resume on the next run."
            )
            raise BudgetExhaustedError(
                audio_path.name,
                deferred_pages=len(deferred),
                completed_pages=completed,
            )

        if multi_chunk:
            self._cleanup_preprocessed(work_dir, audio_path.name)
            self._transient_tracker.mark_preprocessed_complete(work_dir)
        print_success(
            f"Saved transcription for '{audio_path.name}' -> {output_txt_path.name}"
        )
        self._cleanup_temp_jsonl(temp_jsonl_path, method)
        self._transient_tracker.mark_jsonl_complete(temp_jsonl_path)

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
            processed_files, order_map = (
                ImageProcessor.process_and_save_images_for_tesseract(
                    folder, preprocessed_folder, page_indices=page_indices
                )
            )

            # A source image that fails to preprocess is silently dropped (only
            # the files actually written are returned), so guard the drop the
            # same way the PDF path does: order_map has one entry per attempted
            # image, so its length is the expected count. Below the threshold a
            # warning fires; above it the item is raised as failed.
            _raise_if_failure_rate_excessive(
                folder.name,
                len(order_map),
                len(order_map) - len(processed_files),
            )

            if not processed_files:
                print_warning(f"No images found or processed in {folder}.")
                return

            # Deterministic natural ordering for folders (B5).
            processed_files.sort(key=lambda x: natural_sort_key(x.name))

            print_info(
                f"Starting {method} transcription for {len(processed_files)} images..."
            )
            await self._process_images_with_method(
                processed_files,
                method,
                transcriber,
                temp_jsonl_path,
                output_txt_path,
                folder.name,
                is_folder=True,
                order_override=order_map,
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

        # Propagate failures (ResumeFormatError, the render failure-rate guard,
        # PageTranscriptionError, BudgetExhaustedError) so the item counts failed
        # and the run exits non-zero, matching the PDF path (honest exit codes,
        # CT-8). A prior `except RuntimeError` here silently swallowed
        # ResumeFormatError and the render guard, filing broken folders as
        # processed with exit 0.
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

    async def _process_images_with_method(
        self,
        image_files: list[Path],
        method: str,
        transcriber: Any | None,
        temp_jsonl_path: Path,
        output_txt_path: Path,
        source_name: str,
        is_folder: bool = False,
        order_override: dict[str, int] | None = None,
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
            retry_errors=getattr(self.user_config, "retry_errors", False),
            order_override=order_override,
        )
