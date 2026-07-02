# modules/ui/core.py
"""Core UI data structures.

This module contains the UserConfiguration dataclass which stores user's
processing preferences throughout the workflow.

Note: The legacy UserPrompt class has been removed. Use:
- modules.ui.workflows.WorkflowUI for interactive workflows
- modules.ui.batch_display for batch-related displays
- modules.ui.prompts for basic prompt utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modules.documents.auto_selector import AutoSelector
    from modules.documents.page_range import PageRange


@dataclass
class UserConfiguration:
    """Stores user's processing preferences to avoid re-prompting during workflow.

    Attributes:
        processing_type: Type of documents ("images", "pdfs", "epubs", or "auto")
        transcription_method: Method to use ("native", "tesseract", or "gpt")
        use_batch_processing: Whether to use batch processing for GPT
        selected_items: List of files or folders to process
        process_all: Flag indicating whether to process all items
        selected_schema_name: Name of the selected transcription schema (GPT only)
        selected_schema_path: Path to the selected schema file (GPT only)
        additional_context_path: Path to explicit global context file (GPT only)
        use_hierarchical_context: Whether to use file/folder-specific context resolution
            When True and additional_context_path is None, context is resolved per-file
            using the hierarchy: file-specific > folder-specific > general fallback
        auto_decisions: List of FileDecision objects for auto mode
        auto_selector: Cached AutoSelector instance for auto mode
        page_range: Optional page-range filter (first N, last N, or explicit spans)
    """

    processing_type: str | None = None
    transcription_method: str | None = None
    use_batch_processing: bool = False
    # When a batch submission fails, fall back to (full-price) synchronous
    # processing only if the user opted in; default off (decision 5).
    sync_fallback: bool = False
    selected_items: list[Path] | None = None
    process_all: bool = False
    selected_schema_name: str | None = None
    selected_schema_path: Path | None = None
    additional_context_path: Path | None = None
    additional_context_image_path: Path | None = None
    use_hierarchical_context: bool = (
        True  # Enable file/folder-specific context resolution
    )
    auto_decisions: list[Any] | None = None
    auto_selector: AutoSelector | None = None
    resume_mode: str = "skip"
    # When True, pages whose prior output is a "[transcription error]"
    # placeholder are re-processed on resume (decision 13). Default: off.
    retry_errors: bool = False
    page_range: PageRange | None = None
    output_format: str = "txt"
    output_mode: str = "hash"
    input_root: Path | None = None

    def __post_init__(self) -> None:
        if self.selected_items is None:
            self.selected_items = []

    def __str__(self) -> str:
        if self.processing_type == "auto":
            decision_count = len(self.auto_decisions) if self.auto_decisions else 0
            page_text = (
                f", Pages: {self.page_range.describe()}" if self.page_range else ""
            )
            return (
                f"Processing type: auto, Decisions: {decision_count} files{page_text}"
            )

        method_name = {
            "native": "Native PDF extraction",
            "tesseract": "Tesseract OCR",
            "gpt": "GPT-based transcription",
        }.get(self.transcription_method or "", self.transcription_method)
        batch_text = " with batch processing" if self.use_batch_processing else ""
        schema_text = (
            f", Schema: {self.selected_schema_name}"
            if self.transcription_method == "gpt" and self.selected_schema_name
            else ""
        )
        page_text = f", Pages: {self.page_range.describe()}" if self.page_range else ""
        return (
            f"Processing type: {self.processing_type}, "
            f"Method: {method_name}{batch_text}{schema_text}, "
            f"Process all: {self.process_all}, "
            f"Selected items: {len(self.selected_items or [])}{page_text}"
        )
