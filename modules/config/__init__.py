"""Configuration management package.

Provides configuration loading, constants, path resolution utilities, and
centralized configuration service.

Submodules:
- config_loader: YAML configuration loading (ConfigLoader, PROJECT_ROOT, CONFIG_DIR)
- constants: Application constants (SUPPORTED_IMAGE_FORMATS, SUPPORTED_IMAGE_EXTENSIONS, TERMINAL_BATCH_STATUSES)
- service: Configuration service singleton (ConfigService, get_config_service, etc.)

Note: Use direct imports from submodules:
    from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
    from modules.config.constants import SUPPORTED_IMAGE_FORMATS
    from modules.config.service import get_config_service
"""
