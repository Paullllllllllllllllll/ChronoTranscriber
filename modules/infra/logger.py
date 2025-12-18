"""Logging infrastructure for the application.

Provides centralized logger configuration with file and console handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from modules.config.config_loader import PROJECT_ROOT
from modules.config.service import get_config_service


def setup_logger(name: str) -> logging.Logger:
    """
    Set up and return a logger with file and console handlers.

    Logs are written to the configured logs directory (from paths_config.yaml)
    or PROJECT_ROOT/logs as a fallback. Console handler only shows warnings
    and errors, while the file handler captures all INFO level and above.

    Args:
        name: Name of the logger (typically __name__ from the calling module).

    Returns:
        Configured logger instance.
    """
    # Be resilient: try reading paths_config for logs_dir; fall back to
    # PROJECT_ROOT/logs if anything fails.
    try:
        # Get paths_config for logs_dir
        paths_config = get_config_service().get_paths_config()
        logs_dir_value = paths_config.get("general", {}).get("logs_dir")
        if not logs_dir_value:
            logs_dir = PROJECT_ROOT / "logs"
            log_file = logs_dir / "application.log"
        else:
            logs_path = Path(logs_dir_value)
            if not logs_path.is_absolute():
                logs_path = (PROJECT_ROOT / logs_path).resolve()
            # Check if path points to a file (has .log extension) or directory
            if logs_path.suffix == ".log":
                log_file = logs_path
                logs_dir = logs_path.parent
            else:
                logs_dir = logs_path
                log_file = logs_dir / "application.log"
    except Exception:
        logs_dir = PROJECT_ROOT / "logs"
        log_file = logs_dir / "application.log"

    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Add a console handler that only outputs warnings and errors.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    return logger