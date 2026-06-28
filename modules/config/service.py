"""Centralized configuration service with caching and singleton pattern.

Provides a single point of access to all configuration dictionaries, eliminating
redundant ConfigLoader instantiations and ensuring consistent configuration state
across the application.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

from modules.config.config_loader import ConfigLoader


class ConfigService:
    """Thread-safe singleton configuration service with lazy loading and caching."""

    _instance: ConfigService | None = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> ConfigService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        # Prevent re-initialization
        if self._initialized:
            return

        self._loader: ConfigLoader | None = None
        self._model_config: dict[str, Any] | None = None
        self._paths_config: dict[str, Any] | None = None
        self._concurrency_config: dict[str, Any] | None = None
        self._image_processing_config: dict[str, Any] | None = None
        self._api_keys_config: dict[str, Any] | None = None
        self._initialized = True

    def load(self, config_path: Path | None = None) -> None:
        """Load all configurations.

        Args:
            config_path: Optional path to model_config.yaml. If None, uses default.
        """
        with self._lock:
            self._loader = ConfigLoader(config_path)
            self._loader.load_configs()
            # Clear cached configs to force reload
            self._model_config = None
            self._paths_config = None
            self._concurrency_config = None
            self._image_processing_config = None
            self._api_keys_config = None

    def _ensure_loaded(self) -> None:
        """Ensure configuration is loaded, loading with defaults if necessary."""
        if self._loader is None:
            self.load()

    def get_model_config(self) -> dict[str, Any]:
        """Get model configuration (cached).

        Returns:
            Model configuration dictionary.
        """
        self._ensure_loaded()
        if self._model_config is None:
            with self._lock:
                if self._model_config is None and self._loader is not None:
                    self._model_config = self._loader.get_model_config()
        return self._model_config.copy() if self._model_config else {}

    def get_paths_config(self) -> dict[str, Any]:
        """Get paths configuration (cached).

        Returns:
            Paths configuration dictionary with normalized paths.
        """
        self._ensure_loaded()
        if self._paths_config is None:
            with self._lock:
                if self._paths_config is None and self._loader is not None:
                    self._paths_config = self._loader.get_paths_config()
        return self._paths_config.copy() if self._paths_config else {}

    def get_concurrency_config(self) -> dict[str, Any]:
        """Get concurrency configuration (cached).

        Returns:
            Concurrency configuration dictionary.
        """
        self._ensure_loaded()
        if self._concurrency_config is None:
            with self._lock:
                if self._concurrency_config is None and self._loader is not None:
                    self._concurrency_config = self._loader.get_concurrency_config()
        return self._concurrency_config.copy() if self._concurrency_config else {}

    def get_image_processing_config(self) -> dict[str, Any]:
        """Get image processing configuration (cached).

        Returns:
            Image processing configuration dictionary.
        """
        self._ensure_loaded()
        if self._image_processing_config is None:
            with self._lock:
                if self._image_processing_config is None and self._loader is not None:
                    self._image_processing_config = (
                        self._loader.get_image_processing_config()
                    )
        return (
            self._image_processing_config.copy()
            if self._image_processing_config
            else {}
        )

    def get_api_keys_config(self) -> dict[str, Any]:
        """Get the optional API-keys env-var mapping (cached).

        Returns:
            Mapping of provider name to the environment variable holding its
            API key. Empty dict when the optional file is absent.
        """
        self._ensure_loaded()
        if self._api_keys_config is None:
            with self._lock:
                if self._api_keys_config is None and self._loader is not None:
                    self._api_keys_config = self._loader.get_api_keys_config()
        return self._api_keys_config.copy() if self._api_keys_config else {}

    def get_prompt_caching_config(self) -> dict[str, Any]:
        """Get prompt caching configuration from concurrency config.

        Returns:
            Prompt caching configuration dictionary.  Falls back to
            ``{"enabled": False}`` when the section is absent.
        """
        conc_cfg = self.get_concurrency_config() or {}
        return cast(dict[str, Any], conc_cfg.get("prompt_caching", {"enabled": False}))

    def reload(self, config_path: Path | None = None) -> None:
        """Force reload of all configurations.

        Args:
            config_path: Optional path to model_config.yaml. If None, uses default.
        """
        self.load(config_path)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        with cls._lock:
            cls._instance = None


# Convenience functions for direct access
def get_config_service() -> ConfigService:
    """Get the singleton ConfigService instance.

    Returns:
        ConfigService singleton.
    """
    return ConfigService()


def get_model_config() -> dict[str, Any]:
    """Get model configuration.

    Returns:
        Model configuration dictionary.
    """
    return get_config_service().get_model_config()


def get_paths_config() -> dict[str, Any]:
    """Get paths configuration.

    Returns:
        Paths configuration dictionary.
    """
    return get_config_service().get_paths_config()


def get_concurrency_config() -> dict[str, Any]:
    """Get concurrency configuration.

    Returns:
        Concurrency configuration dictionary.
    """
    return get_config_service().get_concurrency_config()


def get_image_processing_config() -> dict[str, Any]:
    """Get image processing configuration.

    Returns:
        Image processing configuration dictionary.
    """
    return get_config_service().get_image_processing_config()


def get_api_keys_config() -> dict[str, Any]:
    """Get the optional API-keys env-var mapping.

    Returns:
        Mapping of provider name to the environment variable holding its API
        key. Empty dict when the optional file is absent.
    """
    return get_config_service().get_api_keys_config()


def get_prompt_caching_config() -> dict[str, Any]:
    """Get prompt caching configuration.

    Returns:
        Prompt caching configuration dictionary.
    """
    return get_config_service().get_prompt_caching_config()
