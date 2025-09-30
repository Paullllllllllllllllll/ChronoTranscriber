"""Core utility functions for console interaction and user input."""

from __future__ import annotations

import sys
import logging


def console_print(message: str) -> None:
    """Print a message to the console."""
    print(message)


def check_exit(user_input: str) -> None:
    """Exit the script if user input indicates quit ('q' or 'exit')."""
    if user_input.lower() in ["q", "exit"]:
        console_print("[INFO] Exiting as requested.")
        sys.exit(0)


def safe_input(prompt: str) -> str:
    """
    Read input safely, exiting on error.

    Args:
        prompt: The prompt message to display to the user.

    Returns:
        The user input trimmed of whitespace.
    """
    try:
        return input(prompt).strip()
    except Exception as e:
        logging.error(f"Error reading input: {e}")
        console_print("[ERROR] Unable to read input. Exiting.")
        sys.exit(1)
