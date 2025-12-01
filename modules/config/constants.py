"""Centralized constants used across the application.

Defines supported image formats, batch statuses, and other shared constants.
"""

from __future__ import annotations

# Supported image extensions and their MIME types for data URLs
# This is the single source of truth for image format support
SUPPORTED_IMAGE_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Convenience set of supported extensions (derived from SUPPORTED_IMAGE_FORMATS)
SUPPORTED_IMAGE_EXTENSIONS = set(SUPPORTED_IMAGE_FORMATS.keys())

# OpenAI Batch API terminal statuses (states that indicate batch completion)
TERMINAL_BATCH_STATUSES = {"completed", "expired", "cancelled", "failed"}
