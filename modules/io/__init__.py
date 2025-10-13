"""Input/Output utilities package.

Provides path validation, directory management, and file I/O operations.
"""

# Avoid circular imports - use direct imports instead of re-exporting
__all__ = [
    "validate_paths",
    "ensure_directory",
    "ensure_directories",
    "ensure_parent_directory",
    "get_output_directories_from_config",
    "get_input_directories_from_config",
    "get_logs_directory",
    "collect_scan_directories",
]
