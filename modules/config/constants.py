"""Centralized constants used across the application.

Defines supported image formats, batch statuses, and other shared constants.
"""

from __future__ import annotations

# Supported image extensions and their MIME types for data URLs
SUPPORTED_IMAGE_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# OpenAI Batch API terminal statuses (states that indicate batch completion)
TERMINAL_BATCH_STATUSES = {"completed", "expired", "cancelled", "failed"}
