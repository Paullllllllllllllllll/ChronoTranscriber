"""Logging infrastructure for the application.

Provides centralized logger configuration with file and console handlers.

All module loggers share a single ``RotatingFileHandler`` attached to a
common ancestor logger (``"modules"``).  This avoids opening the same log
file from dozens of independent handlers — which on Windows causes
``PermissionError`` when any single handler triggers rotation while the
others still hold the file open.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from modules.config.config_loader import PROJECT_ROOT
from modules.config.service import get_config_service

_APP_LOGGER_NAME = "chrono"
_initialized = False


def _ensure_root_handlers() -> None:
    """Attach file and console handlers to the shared app logger once."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return
    _initialized = True

    try:
        paths_config = get_config_service().get_paths_config()
        logs_dir_value = paths_config.get("general", {}).get("logs_dir")
        if not logs_dir_value:
            logs_dir = PROJECT_ROOT / "logs"
            log_file = logs_dir / "application.log"
        else:
            logs_path = Path(logs_dir_value)
            if not logs_path.is_absolute():
                logs_path = (PROJECT_ROOT / logs_path).resolve()
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

    root = logging.getLogger(_APP_LOGGER_NAME)
    root.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)


def setup_logger(name: str) -> logging.Logger:
    """Return a logger under the ``chrono`` hierarchy.

    Every caller's *name* (e.g. ``"modules.llm.transcriber"`` or
    ``"main.unified_transcriber"``) is mapped to
    ``"chrono.<name>"``, making it a child of the single
    ``"chrono"`` logger that owns the shared
    ``RotatingFileHandler``.  This guarantees exactly one open
    file descriptor for the log file across the entire process.

    Args:
        name: Name of the logger (typically ``__name__``).

    Returns:
        Configured logger instance.
    """
    _ensure_root_handlers()
    logger = logging.getLogger(f"{_APP_LOGGER_NAME}.{name}")
    logger.setLevel(logging.INFO)
    return logger
