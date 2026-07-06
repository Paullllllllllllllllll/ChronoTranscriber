"""Resume-aware processing utilities for ChronoTranscriber.

This module provides the ResumeChecker class that determines whether a given
input file or folder has already been processed, enabling skip/resume behavior
when re-running the tool on partially or fully processed directories.

Output existence is detected by replicating the deterministic output-path logic
used by each processor's ``prepare_output_folder`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from modules.infra.logger import setup_logger
from modules.infra.paths import (
    create_safe_directory_name,
    create_safe_filename,
    mirror_output_path,
)

logger = setup_logger(__name__)


class ProcessingState(Enum):
    """Represents the processing state of an input item."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class ResumeResult:
    """Result of a resume check for a single item."""

    item: Path
    state: ProcessingState
    output_path: Path | None = None
    reason: str = ""


class ResumeChecker:
    """Determine whether input files/folders have already been processed.

    The checker replicates the deterministic output-path logic from each
    processor's ``prepare_output_folder`` to locate potential output files
    without actually invoking the full processing pipeline.

    Args:
        resume_mode: One of ``"skip"`` or ``"overwrite"``.
        paths_config: Paths configuration dictionary.
        use_input_as_output: Whether output is co-located with input files.
        pdf_output_dir: Configured PDF output directory.
        image_output_dir: Configured image output directory.
        epub_output_dir: Configured EPUB output directory.
        mobi_output_dir: Configured MOBI output directory.
    """

    def __init__(
        self,
        resume_mode: str,
        paths_config: dict[str, Any],
        *,
        use_input_as_output: bool = False,
        pdf_output_dir: Path | None = None,
        image_output_dir: Path | None = None,
        epub_output_dir: Path | None = None,
        mobi_output_dir: Path | None = None,
        output_format: str = "txt",
        output_mode: str = "hash",
        input_root: Path | None = None,
        retry_errors: bool = False,
    ) -> None:
        self.resume_mode = resume_mode
        self.paths_config = paths_config
        self.use_input_as_output = use_input_as_output
        # When True, an otherwise-complete output that still contains
        # "[transcription error]" placeholders is reported PARTIAL so the item
        # is re-entered and its error pages retried (decision 13).
        self.retry_errors = retry_errors
        self.output_format = output_format
        self._output_ext = f".{output_format}" if output_format != "txt" else ".txt"
        self.output_mode = output_mode
        self.input_root = input_root

        fp = paths_config.get("file_paths", {})
        self.pdf_output_dir = pdf_output_dir or Path(
            fp.get("PDFs", {}).get("output", "pdfs_out")
        )
        self.image_output_dir = image_output_dir or Path(
            fp.get("Images", {}).get("output", "images_out")
        )
        self.epub_output_dir = epub_output_dir or Path(
            fp.get("EPUBs", {}).get("output", "epubs_out")
        )
        self.mobi_output_dir = mobi_output_dir or Path(
            fp.get("MOBIs", {}).get("output", "mobis_out")
        )

    def _output_complete_state(self, item: Path, output_path: Path) -> ProcessingState:
        """COMPLETE, unless --retry-errors and the output has error placeholders.

        In that case the item is reported PARTIAL so it is re-entered and the
        page-level resume (with ``exclude_errors``) retries just the failed
        pages.
        """
        if not self.retry_errors:
            return ProcessingState.COMPLETE
        try:
            text = output_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ProcessingState.COMPLETE
        if "[transcription error" in text.lower():
            logger.info(
                "Retry-errors: %s contains error placeholders; marking PARTIAL.",
                output_path.name,
            )
            return ProcessingState.PARTIAL
        return ProcessingState.COMPLETE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_skip(self, item: Path, processing_type: str) -> ResumeResult:
        """Check if *item* should be skipped based on existing output.

        Args:
            item: Input file or folder path.
            processing_type: One of ``"pdfs"``, ``"images"``, ``"epubs"``,
                ``"mobis"``, or ``"auto"`` (auto-detect from item type).

        Returns:
            A :class:`ResumeResult` describing the item's state.
        """
        if self.resume_mode == "overwrite":
            return ResumeResult(
                item=item, state=ProcessingState.NONE, reason="overwrite mode"
            )

        if processing_type == "pdfs":
            return self._check_pdf(item)
        elif processing_type == "images":
            return self._check_image_folder(item)
        elif processing_type == "epubs":
            return self._check_epub(item)
        elif processing_type == "mobis":
            return self._check_mobi(item)
        elif processing_type == "auto":
            return self._check_auto(item)
        else:
            return ResumeResult(
                item=item, state=ProcessingState.NONE, reason="unknown type"
            )

    def _check_auto(self, item: Path) -> ResumeResult:
        """Auto-detect item type and delegate to the appropriate check."""
        if item.is_dir():
            return self._check_image_folder(item)
        suffix = item.suffix.lower()
        if suffix == ".pdf":
            return self._check_pdf(item)
        if suffix == ".epub":
            return self._check_epub(item)
        if suffix in {".mobi", ".azw", ".azw3", ".kfx"}:
            return self._check_mobi(item)
        return ResumeResult(
            item=item, state=ProcessingState.NONE, reason="unknown type"
        )

    def filter_items(
        self,
        items: list[Path],
        processing_type: str,
    ) -> tuple[list[Path], list[ResumeResult]]:
        """Partition *items* into those that need processing and those skipped.

        Args:
            items: Input file/folder paths.
            processing_type: Document type string.

        Returns:
            Tuple of ``(to_process, skipped)`` where *skipped* contains
            :class:`ResumeResult` entries for items that were filtered out.
        """
        if self.resume_mode == "overwrite":
            return list(items), []

        to_process: list[Path] = []
        skipped: list[ResumeResult] = []

        for item in items:
            result = self.should_skip(item, processing_type)
            if result.state == ProcessingState.COMPLETE:
                skipped.append(result)
            else:
                to_process.append(item)

        return to_process, skipped

    # ------------------------------------------------------------------
    # Per-type checks (mirror prepare_output_folder logic)
    # ------------------------------------------------------------------

    def _resolve_output_dir(self, item: Path, default_output_dir: Path) -> Path:
        """Resolve the output directory for an item."""
        if self.use_input_as_output:
            return item.parent if item.is_file() else item
        return default_output_dir

    def _check_output_exists(
        self,
        item: Path,
        item_stem: str,
        default_output_dir: Path,
        *,
        supports_partial_jsonl: bool = True,
        working_dir_base_override: Path | None = None,
        hash_key: str | None = None,
    ) -> ResumeResult:
        """Generic output-existence check shared by all document types.

        Args:
            item: Input file or folder path.
            item_stem: Name used for output file derivation (e.g.
                ``pdf_path.stem``, ``folder.name``).
            default_output_dir: Configured output directory for this type.
            supports_partial_jsonl: If ``True``, also check for partial JSONL.
            working_dir_base_override: If provided, use this as the base for
                the hash-suffixed working directory instead of deriving from
                ``use_input_as_output`` / *default_output_dir*.  Used by image
                folders whose working directory lives in the parent, not inside
                the folder.
            hash_key: If provided, used instead of *item_stem* for the
                directory hash (to match relative-path-aware hashing).
        """
        # -- Co-located output check (use_input_as_output) ----------------
        # For files the co-locate directory is item.parent; for folders it is
        # also item.parent (the caller passes the folder itself as *item*).
        co_locate_dir = item.parent

        if self.use_input_as_output:
            out_name = create_safe_filename(
                item_stem,
                self._output_ext,
                co_locate_dir,
            )
            out_path = co_locate_dir / out_name
            if out_path.exists() and out_path.stat().st_size > 0:
                return ResumeResult(
                    item=item,
                    state=self._output_complete_state(item, out_path),
                    output_path=out_path,
                    reason=f"output exists: {out_path.name}",
                )

        # -- Determine working-dir base -----------------------------------
        if working_dir_base_override is not None:
            working_dir_base = working_dir_base_override
        elif self.use_input_as_output:
            working_dir_base = item.parent
        else:
            working_dir_base = default_output_dir

        # -- Hash-suffixed working directory (legacy / JSONL location) -----
        safe_dir = create_safe_directory_name(hash_key or item_stem)
        parent_folder = working_dir_base / safe_dir

        if not parent_folder.exists():
            return ResumeResult(
                item=item,
                state=ProcessingState.NONE,
                reason="no output folder",
            )

        txt_name_in_dir = create_safe_filename(
            item_stem,
            self._output_ext,
            parent_folder,
        )
        txt_path_in_dir = parent_folder / txt_name_in_dir

        if txt_path_in_dir.exists() and txt_path_in_dir.stat().st_size > 0:
            return ResumeResult(
                item=item,
                state=self._output_complete_state(item, txt_path_in_dir),
                output_path=txt_path_in_dir,
                reason=f"output exists: {txt_path_in_dir.name}",
            )

        # -- Optional partial-JSONL check ----------------------------------
        if supports_partial_jsonl:
            jsonl_name = create_safe_filename(item_stem, ".jsonl", parent_folder)
            jsonl_path = parent_folder / jsonl_name
            if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                return ResumeResult(
                    item=item,
                    state=ProcessingState.PARTIAL,
                    output_path=jsonl_path,
                    reason=f"partial JSONL exists: {jsonl_path.name}",
                )

        return ResumeResult(item=item, state=ProcessingState.NONE, reason="no output")

    # -- Thin per-type wrappers -------------------------------------------

    def _check_pdf(self, pdf_path: Path) -> ResumeResult:
        if self.output_mode == "mirror" and self.input_root is not None:
            return self._check_mirror_output(
                pdf_path,
                pdf_path.parent,
                self.pdf_output_dir,
                pdf_path.stem,
            )
        rel_key = self._relative_key(pdf_path)
        return self._check_output_exists(
            pdf_path,
            pdf_path.stem,
            self.pdf_output_dir,
            hash_key=rel_key,
        )

    def _check_image_folder(self, folder: Path) -> ResumeResult:
        if self.output_mode == "mirror" and self.input_root is not None:
            return self._check_mirror_output(
                folder,
                folder,
                self.image_output_dir,
                folder.name,
            )
        rel_key = self._relative_key(folder)
        return self._check_output_exists(
            folder,
            folder.name,
            self.image_output_dir,
            working_dir_base_override=(
                folder.parent if self.use_input_as_output else None
            ),
            hash_key=rel_key,
        )

    def _relative_key(self, item: Path) -> str | None:
        if self.input_root is None:
            return None
        try:
            rel = str(item.relative_to(self.input_root))
        except ValueError:
            return None
        # A single-file --input makes item == input_root, so the relative key
        # degenerates to "." and would hash into a hidden ".-<hash>" directory
        # that never matches the real output dir — item-level resume then never
        # recognizes the completed item (CT-1). Fall back to None so the hash is
        # derived from the item's own stem/name, matching manager._relative_key.
        if rel in (".", ""):
            return None
        return rel

    def _check_mirror_output(
        self,
        item: Path,
        mirror_item: Path,
        output_dir: Path,
        stem: str,
    ) -> ResumeResult:
        assert self.input_root is not None
        m_dir = mirror_output_path(mirror_item, self.input_root, output_dir)
        txt_name = create_safe_filename(stem, self._output_ext, m_dir)
        txt_path = m_dir / txt_name
        if txt_path.exists() and txt_path.stat().st_size > 0:
            return ResumeResult(
                item=item,
                state=self._output_complete_state(item, txt_path),
                output_path=txt_path,
                reason=f"mirror output exists: {txt_path.name}",
            )
        return ResumeResult(
            item=item,
            state=ProcessingState.NONE,
            reason="no mirror output",
        )

    def _check_epub(self, epub_path: Path) -> ResumeResult:
        return self._check_output_exists(
            epub_path,
            epub_path.stem,
            self.epub_output_dir,
            supports_partial_jsonl=False,
        )

    def _check_mobi(self, mobi_path: Path) -> ResumeResult:
        return self._check_output_exists(
            mobi_path,
            mobi_path.stem,
            self.mobi_output_dir,
            supports_partial_jsonl=False,
        )
