"""Core utility functions for console interaction and user input.

DEPRECATED: For styled console output, use modules.ui.prompts functions:
- print_info(), print_success(), print_warning(), print_error()
- ui_print() for custom styling

The functions in this module are kept for backward compatibility only.
"""

from __future__ import annotations

import sys
import logging
import warnings


def console_print(message: str) -> None:
    """Print a message to the console.
    
    .. deprecated::
        Use modules.ui.prompts functions instead:
        - print_info() for informational messages
        - print_warning() for warnings  
        - print_error() for errors
        - print_success() for success messages
    """
    warnings.warn(
        "console_print is deprecated. Use modules.ui.prompts functions instead "
        "(print_info, print_warning, print_error, print_success).",
        DeprecationWarning,
        stacklevel=2
    )
    print(message)


def check_exit(user_input: str) -> None:
    """Exit the script if user input indicates quit ('q' or 'exit').
    
    .. deprecated::
        Navigation is now handled by modules.ui.prompts with NavigationAction.
    """
    if user_input.lower() in ["q", "exit"]:
        from modules.ui import print_info
        print_info("Exiting as requested.")
        sys.exit(0)


def safe_input(prompt: str) -> str:
    """Read input safely, exiting on error.

    Args:
        prompt: The prompt message to display to the user.

    Returns:
        The user input trimmed of whitespace.
        
    .. deprecated::
        Use modules.ui.prompts.prompt_text() instead for styled prompts.
    """
    try:
        return input(prompt).strip()
    except Exception as e:
        logging.error(f"Error reading input: {e}")
        from modules.ui import print_error
        print_error("Unable to read input. Exiting.")
        sys.exit(1)
