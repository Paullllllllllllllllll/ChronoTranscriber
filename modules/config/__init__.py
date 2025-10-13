"""Configuration management package.

Provides configuration loading, constants, path resolution utilities, and
centralized configuration service.
"""

# Avoid circular imports - use direct imports instead of re-exporting
__all__ = [
    "ConfigLoader",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "SUPPORTED_IMAGE_FORMATS",
    "TERMINAL_BATCH_STATUSES",
    "ConfigService",
    "get_config_service",
    "get_model_config",
    "get_paths_config",
    "get_concurrency_config",
    "get_image_processing_config",
]
