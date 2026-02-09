"""Resume-aware processing utilities for ChronoTranscriber.

This module provides the ResumeChecker class that determines whether a given
input file or folder has already been processed, enabling skip/resume behavior
when re-running the tool on partially or fully processed directories.

Output existence is detected by replicating the deterministic output-path logic
used by each processor's ``prepare_output_folder`` method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.core.safe_paths import create_safe_directory_name, create_safe_filename
from modules.infra.logger import setup_logger

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
    output_path: Optional[Path] = None
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
        paths_config: Dict[str, Any],
        *,
        use_input_as_output: bool = False,
        pdf_output_dir: Optional[Path] = None,
        image_output_dir: Optional[Path] = None,
        epub_output_dir: Optional[Path] = None,
        mobi_output_dir: Optional[Path] = None,
    ) -> None:
        self.resume_mode = resume_mode
        self.paths_config = paths_config
        self.use_input_as_output = use_input_as_output

        fp = paths_config.get("file_paths", {})
        self.pdf_output_dir = pdf_output_dir or Path(fp.get("PDFs", {}).get("output", "pdfs_out"))
        self.image_output_dir = image_output_dir or Path(fp.get("Images", {}).get("output", "images_out"))
        self.epub_output_dir = epub_output_dir or Path(fp.get("EPUBs", {}).get("output", "epubs_out"))
        self.mobi_output_dir = mobi_output_dir or Path(fp.get("MOBIs", {}).get("output", "mobis_out"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_skip(self, item: Path, processing_type: str) -> ResumeResult:
        """Check if *item* should be skipped based on existing output.

        Args:
            item: Input file or folder path.
            processing_type: One of ``"pdfs"``, ``"images"``, ``"epubs"``, ``"mobis"``.

        Returns:
            A :class:`ResumeResult` describing the item's state.
        """
        if self.resume_mode == "overwrite":
            return ResumeResult(item=item, state=ProcessingState.NONE, reason="overwrite mode")

        if processing_type == "pdfs":
            return self._check_pdf(item)
        elif processing_type == "images":
            return self._check_image_folder(item)
        elif processing_type == "epubs":
            return self._check_epub(item)
        elif processing_type == "mobis":
            return self._check_mobi(item)
        else:
            return ResumeResult(item=item, state=ProcessingState.NONE, reason="unknown type")

    def filter_items(
        self,
        items: List[Path],
        processing_type: str,
    ) -> Tuple[List[Path], List[ResumeResult]]:
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

        to_process: List[Path] = []
        skipped: List[ResumeResult] = []

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

    def _check_pdf(self, pdf_path: Path) -> ResumeResult:
        output_dir = self._resolve_output_dir(pdf_path, self.pdf_output_dir)
        safe_dir = create_safe_directory_name(pdf_path.stem)
        parent_folder = output_dir / safe_dir

        if not parent_folder.exists():
            return ResumeResult(item=pdf_path, state=ProcessingState.NONE, reason="no output folder")

        txt_name = create_safe_filename(pdf_path.stem, ".txt", parent_folder)
        txt_path = parent_folder / txt_name

        jsonl_name = create_safe_filename(pdf_path.stem, ".jsonl", parent_folder)
        jsonl_path = parent_folder / jsonl_name

        if txt_path.exists() and txt_path.stat().st_size > 0:
            return ResumeResult(
                item=pdf_path, state=ProcessingState.COMPLETE,
                output_path=txt_path,
                reason=f"output exists: {txt_path.name}",
            )

        if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            return ResumeResult(
                item=pdf_path, state=ProcessingState.PARTIAL,
                output_path=jsonl_path,
                reason=f"partial JSONL exists: {jsonl_path.name}",
            )

        return ResumeResult(item=pdf_path, state=ProcessingState.NONE, reason="no output")

    def _check_image_folder(self, folder: Path) -> ResumeResult:
        output_dir = self.image_output_dir if not self.use_input_as_output else folder
        safe_dir = create_safe_directory_name(folder.name)
        parent_folder = output_dir / safe_dir

        if not parent_folder.exists():
            return ResumeResult(item=folder, state=ProcessingState.NONE, reason="no output folder")

        txt_name = create_safe_filename(folder.name, ".txt", parent_folder)
        txt_path = parent_folder / txt_name

        jsonl_name = create_safe_filename(folder.name, ".jsonl", parent_folder)
        jsonl_path = parent_folder / jsonl_name

        if txt_path.exists() and txt_path.stat().st_size > 0:
            return ResumeResult(
                item=folder, state=ProcessingState.COMPLETE,
                output_path=txt_path,
                reason=f"output exists: {txt_path.name}",
            )

        if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
            return ResumeResult(
                item=folder, state=ProcessingState.PARTIAL,
                output_path=jsonl_path,
                reason=f"partial JSONL exists: {jsonl_path.name}",
            )

        return ResumeResult(item=folder, state=ProcessingState.NONE, reason="no output")

    def _check_epub(self, epub_path: Path) -> ResumeResult:
        output_dir = self._resolve_output_dir(epub_path, self.epub_output_dir)
        safe_dir = create_safe_directory_name(epub_path.stem)
        parent_folder = output_dir / safe_dir

        if not parent_folder.exists():
            return ResumeResult(item=epub_path, state=ProcessingState.NONE, reason="no output folder")

        txt_name = create_safe_filename(epub_path.stem, ".txt", parent_folder)
        txt_path = parent_folder / txt_name

        if txt_path.exists() and txt_path.stat().st_size > 0:
            return ResumeResult(
                item=epub_path, state=ProcessingState.COMPLETE,
                output_path=txt_path,
                reason=f"output exists: {txt_path.name}",
            )

        return ResumeResult(item=epub_path, state=ProcessingState.NONE, reason="no output")

    def _check_mobi(self, mobi_path: Path) -> ResumeResult:
        output_dir = self._resolve_output_dir(mobi_path, self.mobi_output_dir)
        safe_dir = create_safe_directory_name(mobi_path.stem)
        parent_folder = output_dir / safe_dir

        if not parent_folder.exists():
            return ResumeResult(item=mobi_path, state=ProcessingState.NONE, reason="no output folder")

        txt_name = create_safe_filename(mobi_path.stem, ".txt", parent_folder)
        txt_path = parent_folder / txt_name

        if txt_path.exists() and txt_path.stat().st_size > 0:
            return ResumeResult(
                item=mobi_path, state=ProcessingState.COMPLETE,
                output_path=txt_path,
                reason=f"output exists: {txt_path.name}",
            )

        return ResumeResult(item=mobi_path, state=ProcessingState.NONE, reason="no output")
