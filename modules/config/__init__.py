"""Configuration management package.

Provides configuration loading, constants, and path resolution utilities.
"""

# Avoid circular imports - use direct imports instead of re-exporting
__all__ = [
    "ConfigLoader",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "SUPPORTED_IMAGE_FORMATS",
    "TERMINAL_BATCH_STATUSES",
]
