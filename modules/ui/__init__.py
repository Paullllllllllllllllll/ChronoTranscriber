# modules/ui/__init__.py
"""User interface components for ChronoTranscriber.

Provides:
- Core data structures (UserConfiguration)
- Prompt utilities with navigation support
- Workflow UI components
- Batch display utilities
"""

from .core import UserConfiguration
from .prompts import (
    NavigationAction,
    PromptResult,
    PromptStyle,
    ui_print,
    ui_input,
    print_header,
    print_separator,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_navigation_help,
    prompt_select,
    prompt_yes_no,
    prompt_text,
    prompt_multiselect,
    confirm_action,
)
from .workflows import WorkflowUI
from .batch_display import (
    display_batch_summary,
    display_batch_processing_progress,
    display_batch_cancellation_results,
    print_transcription_item_error,
    print_transcription_not_possible,
    print_no_transcribable_text,
    display_page_error_summary,
    display_transcription_not_possible_summary,
)

__all__ = [
    # Core data structures
    "UserConfiguration",
    # Navigation
    "NavigationAction",
    "PromptResult",
    "PromptStyle",
    # Print utilities
    "ui_print",
    "ui_input",
    "print_header",
    "print_separator",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    "print_navigation_help",
    # Prompt functions
    "prompt_select",
    "prompt_yes_no",
    "prompt_text",
    "prompt_multiselect",
    "confirm_action",
    # Workflow UI
    "WorkflowUI",
    # Batch display
    "display_batch_summary",
    "display_batch_processing_progress",
    "display_batch_cancellation_results",
    "print_transcription_item_error",
    "print_transcription_not_possible",
    "print_no_transcribable_text",
    "display_page_error_summary",
    "display_transcription_not_possible_summary",
]
