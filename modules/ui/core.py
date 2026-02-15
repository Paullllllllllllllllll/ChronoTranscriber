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
from typing import Any, List, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from modules.core.auto_selector import AutoSelector
    from modules.core.page_range import PageRange


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

    processing_type: Optional[str] = None
    transcription_method: Optional[str] = None
    use_batch_processing: bool = False
    selected_items: List[Path] | None = None
    process_all: bool = False
    selected_schema_name: Optional[str] = None
    selected_schema_path: Optional[Path] = None
    additional_context_path: Optional[Path] = None
    use_hierarchical_context: bool = True  # Enable file/folder-specific context resolution
    auto_decisions: Optional[List[Any]] = None
    auto_selector: Optional["AutoSelector"] = None
    resume_mode: str = "skip"
    page_range: Optional["PageRange"] = None

    def __post_init__(self) -> None:
        if self.selected_items is None:
            self.selected_items = []

    def __str__(self) -> str:
        if self.processing_type == "auto":
            decision_count = len(self.auto_decisions) if self.auto_decisions else 0
            page_text = f", Pages: {self.page_range.describe()}" if self.page_range else ""
            return f"Processing type: auto, Decisions: {decision_count} files{page_text}"
        
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
        page_text = (
            f", Pages: {self.page_range.describe()}"
            if self.page_range
            else ""
        )
        return (
            f"Processing type: {self.processing_type}, "
            f"Method: {method_name}{batch_text}{schema_text}, "
            f"Process all: {self.process_all}, "
            f"Selected items: {len(self.selected_items or [])}{page_text}"
        )
