# modules/config_loader.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any; from typing import Dict; from typing import Optional

import yaml

from modules.model_capabilities import ensure_image_support


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _expand_path_str(p: str) -> Path:
    """
    Expand ~ and environment variables in a path string and return a Path.
    """
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _compute_config_dir() -> Path:
    """
    Resolve CHRONO_CONFIG_DIR robustly:
    - If absolute: use it.
    - If relative: resolve against PROJECT_ROOT.
    - If unset: default to PROJECT_ROOT/config.
    """
    raw = os.environ.get("CHRONO_CONFIG_DIR")
    if raw:
        expanded = _expand_path_str(raw)
        return (expanded if expanded.is_absolute()
                else (PROJECT_ROOT / expanded)).resolve()
    return (PROJECT_ROOT / "config").resolve()


CONFIG_DIR = _compute_config_dir()
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
DEFAULT_PATHS_CONFIG_PATH = CONFIG_DIR / "paths_config.yaml"
DEFAULT_CONCURRENCY_CONFIG_PATH = CONFIG_DIR / "concurrency_config.yaml"
DEFAULT_IMAGE_PROCESSING_CONFIG_PATH = CONFIG_DIR / "image_processing_config.yaml"


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
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.scanner.ScannerError as e:
            raise ValueError(
                f"YAML parsing error in {path}.\n"
                f"Tip: Windows paths in double quotes require escaped backslashes "
                f'(e.g., "C:\\\\Users\\\\name"), or use single quotes '
                f"(e.g., 'C:\\Users\\name'), or forward slashes "
                f"(e.g., C:/Users/name).\nOriginal error: {e}"
            ) from e

    def load_configs(self) -> None:
        """
        Load YAML configuration into memory.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Missing configuration file: {self.config_path}"
            )
        with self.config_path.open("r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}

        # Validate environment API key presence if relevant
        if "OPENAI_API_KEY" not in os.environ:
            # Do not crash here; some batch flows may inject keys via the SDK.
            pass

        # Normalize and validate transcription model's image capability
        tm_raw = (
            self._raw.get("transcription_model")
            or self._raw.get("extraction_model")
            or {}
        )
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
        """
        Load, normalize, and return the paths configuration (cached).
        - Expands ~ and envvars.
        - If allow_relative_paths is true, resolve against general.base_directory
          (itself resolved against PROJECT_ROOT when relative).
        - If allow_relative_paths is false, enforce absolute paths, but still
          expand envvars/home.
        """
        if self._paths is None:
            raw = self._load_yaml_file(DEFAULT_PATHS_CONFIG_PATH)
            self._paths = self._normalize_paths_config(raw)
        return self._paths.copy()

    def get_concurrency_config(self) -> Dict[str, Any]:
        """Load and return the concurrency configuration (cached)."""
        if self._concurrency is None:
            self._concurrency = self._load_yaml_file(
                DEFAULT_CONCURRENCY_CONFIG_PATH
            )
        return self._concurrency.copy()

    def get_image_processing_config(self) -> Dict[str, Any]:
        """Load and return the image processing configuration (cached)."""
        if self._image_processing is None:
            self._image_processing = self._load_yaml_file(
                DEFAULT_IMAGE_PROCESSING_CONFIG_PATH
            )
        return self._image_processing.copy()

    # -------- Internal helpers for path normalization --------

    @staticmethod
    def _to_abs(p: Optional[str], base: Path) -> Optional[str]:
        if not p:
            return p
        cand = _expand_path_str(str(p))
        if cand.is_absolute():
            return str(cand.resolve())
        return str((base / cand).resolve())

    def _normalize_paths_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        # Copy we can mutate safely
        out = dict(cfg or {})
        general = dict(out.get("general", {}))
        allow_rel = bool(general.get("allow_relative_paths", False))

        # Resolve base directory (used when allow_relative_paths is true)
        base_dir_raw = general.get("base_directory")
        if not base_dir_raw:
            base_path = PROJECT_ROOT
        else:
            base_candidate = _expand_path_str(str(base_dir_raw))
            base_path = (
                base_candidate
                if base_candidate.is_absolute()
                else (PROJECT_ROOT / base_candidate)
            )
        base_path = base_path.resolve()

        # logs_dir
        logs_dir_raw = general.get("logs_dir")
        if logs_dir_raw:
            if allow_rel:
                general["logs_dir"] = self._to_abs(logs_dir_raw, base_path)
            else:
                abs_p = _expand_path_str(str(logs_dir_raw))
                general["logs_dir"] = str(
                    abs_p.resolve()
                    if abs_p.is_absolute()
                    else (PROJECT_ROOT / abs_p).resolve()
                )

        # reflect resolved base_directory (as absolute)
        general["base_directory"] = str(base_path)
        out["general"] = general

        # file_paths
        file_paths = dict(out.get("file_paths", {}))
        for section in ("PDFs", "Images"):
            if section in file_paths and isinstance(file_paths[section], dict):
                sec = dict(file_paths[section])
                for k in ("input", "output"):
                    rawp = sec.get(k)
                    if not rawp:
                        continue
                    if allow_rel:
                        sec[k] = self._to_abs(rawp, base_path)
                    else:
                        pth = _expand_path_str(str(rawp))
                        sec[k] = str(
                            pth.resolve()
                            if pth.is_absolute()
                            else (PROJECT_ROOT / pth).resolve()
                        )
                file_paths[section] = sec
        out["file_paths"] = file_paths

        return out