# modules/config_loader.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from modules.model_capabilities import ensure_image_support


DEFAULT_CONFIG_PATH = Path("config") / "model_config.yaml"
DEFAULT_PATHS_CONFIG_PATH = Path("config") / "paths_config.yaml"
DEFAULT_CONCURRENCY_CONFIG_PATH = Path("config") / "concurrency_config.yaml"
DEFAULT_IMAGE_PROCESSING_CONFIG_PATH = Path("config") / "image_processing_config.yaml"


@dataclass(slots=True)
class _TranscriptionModel:
    name: str
    expects_image_inputs: bool = True
    # Optional advanced fields (read if present; safe defaults otherwise)
    max_output_tokens: Optional[int] = None
    service_tier: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    seed: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None


class ConfigLoader:
    """
    Loads configuration files and exposes normalized dictionaries to callers.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._raw: Dict[str, Any] = {}
        self._paths: Optional[Dict[str, Any]] = None
        self._concurrency: Optional[Dict[str, Any]] = None
        self._image_processing: Optional[Dict[str, Any]] = None

    @staticmethod
    def _load_yaml_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_configs(self) -> None:
        """
        Load YAML configuration into memory.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

        # Validate environment API key presence if relevant
        if "OPENAI_API_KEY" not in os.environ:
            # Do not crash here; some batch flows may inject keys via the OpenAI SDK internally.
            pass

        # Normalize and validate transcription model's image capability
        tm_raw = self._raw.get("transcription_model") or self._raw.get("extraction_model") or {}
        tm = _TranscriptionModel(
            name=str(tm_raw.get("name", "gpt-4o-2024-08-06")),
            expects_image_inputs=bool(tm_raw.get("expects_image_inputs", True)),
            max_output_tokens=tm_raw.get("max_output_tokens"),
            service_tier=tm_raw.get("service_tier"),
            temperature=tm_raw.get("temperature"),
            top_p=tm_raw.get("top_p"),
            frequency_penalty=tm_raw.get("frequency_penalty"),
            presence_penalty=tm_raw.get("presence_penalty"),
            stop=tm_raw.get("stop"),
            seed=tm_raw.get("seed"),
            reasoning=tm_raw.get("reasoning"),
            text=tm_raw.get("text"),
        )

        # Fail fast if images are required but the model cannot accept them
        ensure_image_support(tm.name, tm.expects_image_inputs)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Return the raw configuration dictionary. This preserves legacy callers while
        ensuring that image capability was validated in `load_configs()`.
        """
        return self._raw.copy()

    def get_paths_config(self) -> Dict[str, Any]:
        """Load and return the paths configuration (cached)."""
        if self._paths is None:
            self._paths = self._load_yaml_file(DEFAULT_PATHS_CONFIG_PATH)
        return self._paths.copy()

    def get_concurrency_config(self) -> Dict[str, Any]:
        """Load and return the concurrency configuration (cached)."""
        if self._concurrency is None:
            self._concurrency = self._load_yaml_file(DEFAULT_CONCURRENCY_CONFIG_PATH)
        return self._concurrency.copy()

    def get_image_processing_config(self) -> Dict[str, Any]:
        """Load and return the image processing configuration (cached)."""
        if self._image_processing is None:
            self._image_processing = self._load_yaml_file(DEFAULT_IMAGE_PROCESSING_CONFIG_PATH)
        return self._image_processing.copy()
