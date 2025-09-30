"""Logging infrastructure for the application.

Provides centralized logger configuration with file and console handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from modules.config.config_loader import ConfigLoader, PROJECT_ROOT


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
        config_loader = ConfigLoader()
        # Do not call load_configs() here; we only need paths_config for logs_dir
        paths_config = config_loader.get_paths_config()
        logs_dir_value = paths_config.get("general", {}).get("logs_dir")
        if not logs_dir_value:
            logs_dir = PROJECT_ROOT / "logs"
        else:
            logs_dir = Path(logs_dir_value)
            if not logs_dir.is_absolute():
                logs_dir = (PROJECT_ROOT / logs_dir).resolve()
    except Exception:
        logs_dir = PROJECT_ROOT / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "application.log"
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