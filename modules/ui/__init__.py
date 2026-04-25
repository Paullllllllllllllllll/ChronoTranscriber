# modules/ui/__init__.py
"""User-interface components for ChronoTranscriber.

Terminal prompts, ANSI styling, navigation helpers, the workflow wizard,
and batch-display formatters. Contains no domain state; the transcription
``UserConfiguration`` lives in ``modules.transcribe.user_config``.
"""

from .batch_display import (
    display_batch_cancellation_results,
    display_batch_processing_progress,
    display_batch_summary,
    display_page_error_summary,
    display_transcription_not_possible_summary,
    print_no_transcribable_text,
    print_transcription_item_error,
    print_transcription_not_possible,
)
from .prompts import (
    NavigationAction,
    PromptResult,
    PromptStyle,
    confirm_action,
    print_error,
    print_header,
    print_info,
    print_navigation_help,
    print_separator,
    print_success,
    print_warning,
    prompt_multiselect,
    prompt_select,
    prompt_text,
    prompt_yes_no,
    ui_input,
    ui_print,
)
from .workflows import WorkflowUI

__all__ = [
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
