# modules/config_loader.py
import yaml
from pathlib import Path
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    A class for loading and validating configuration files.
    """
    REQUIRED_PATHS = ['general', 'file_paths']
    REQUIRED_MODEL_CONFIG = ['transcription_model']
    REQUIRED_IMAGE_PROCESSING = ['image_processing']
    REQUIRED_CONCURRENCY = ['concurrency']

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if config_dir is None:
            self.config_dir = Path(__file__).resolve().parent.parent / 'config'
        else:
            self.config_dir = config_dir
        self.paths_config: Optional[Dict[str, Any]] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.image_processing_config: Optional[Dict[str, Any]] = None
        self.concurrency_config: Optional[Dict[str, Any]] = None

    def load_configs(self) -> None:
        """
        Load all configuration files and validate them.
        """
        self.paths_config = self._load_yaml('paths_config.yaml')
        self.model_config = self._load_yaml('model_config.yaml')
        self.image_processing_config = self._load_yaml('image_processing_config.yaml')
        self.concurrency_config = self._load_yaml('concurrency_config.yaml')

        self._validate_paths_config(self.paths_config)
        self._validate_model_config(self.model_config)
        self._validate_image_processing_config(self.image_processing_config)
        self._validate_concurrency_config(self.concurrency_config)
        logger.info("All configurations loaded and validated successfully.")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        config_path = self.config_dir / filename
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Missing configuration file: {config_path}")
        with config_path.open('r', encoding='utf-8') as f:
            content = f.read()
            content = content.replace("\\", "/")
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {filename}: {e}")
                raise

    def _validate_paths_config(self, config: Dict[str, Any]) -> None:
        for key in self.REQUIRED_PATHS:
            if key not in config:
                logger.error(f"Missing '{key}' in paths_config.yaml")
                raise KeyError(f"'{key}' is required in paths_config.yaml")

    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        for key in self.REQUIRED_MODEL_CONFIG:
            if key not in config:
                logger.error(f"Missing '{key}' in model_config.yaml")
                raise KeyError(f"'{key}' is required in model_config.yaml")

    def _validate_image_processing_config(self, config: Dict[str, Any]) -> None:
        for key in self.REQUIRED_IMAGE_PROCESSING:
            if key not in config:
                logger.error(f"Missing '{key}' in image_processing_config.yaml")
                raise KeyError(f"'{key}' is required in image_processing_config.yaml")

    def _validate_concurrency_config(self, config: Dict[str, Any]) -> None:
        for key in self.REQUIRED_CONCURRENCY:
            if key not in config:
                logger.error(f"Missing '{key}' in concurrency_config.yaml")
                raise KeyError(f"'{key}' is required in concurrency_config.yaml")

    def get_paths_config(self) -> Dict[str, Any]:
        assert self.paths_config is not None, "Paths config has not been loaded."
        return self.paths_config

    def get_model_config(self) -> Dict[str, Any]:
        assert self.model_config is not None, "Model config has not been loaded."
        return self.model_config

    def get_image_processing_config(self) -> Dict[str, Any]:
        assert self.image_processing_config is not None, "Image processing config has not been loaded."
        return self.image_processing_config

    def get_concurrency_config(self) -> Dict[str, Any]:
        assert self.concurrency_config is not None, "Concurrency config has not been loaded."
        return self.concurrency_config
