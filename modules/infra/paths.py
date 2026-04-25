"""Filesystem path utilities: Windows MAX_PATH safety, typed path config,
directory ensure/create/scan helpers.

Consolidates the former modules.infra.paths, modules.infra.paths,
and modules.infra.paths into a single module.

Exports:
    create_safe_directory_name, create_safe_filename, create_safe_log_filename,
        ensure_path_safe
    PathConfig
    ensure_directory, ensure_directories, ensure_parent_directory,
        get_output_directories_from_config, get_input_directories_from_config,
        get_logs_directory, collect_scan_directories
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.config.constants import DOCUMENT_CATEGORIES
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Windows MAX_PATH safety
# ---------------------------------------------------------------------------

WINDOWS_MAX_PATH = 260
MAX_SAFE_NAME_LENGTH = 80
HASH_LENGTH = 8
MIN_READABLE_LENGTH = 20


def _compute_name_hash(name: str) -> str:
    """Return an 8-character SHA-256 hex digest of *name*."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _truncate_name(name: str, max_length: int) -> str:
    """Truncate *name* to *max_length*, stripping trailing punctuation."""
    if len(name) <= max_length:
        return name
    return name[:max_length].rstrip(".-_ ")


def create_safe_directory_name(original_name: str, suffix: str = "") -> str:
    """Create a safe directory name that won't exceed Windows path limits."""
    name_hash = _compute_name_hash(original_name)

    reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for the dash
    available_length = MAX_SAFE_NAME_LENGTH - reserved_length

    truncated = _truncate_name(original_name, available_length)
    safe_name = f"{truncated}-{name_hash}{suffix}"

    return safe_name


def create_safe_filename(
    original_name: str,
    extension: str,
    parent_path: Optional[Path] = None,
) -> str:
    """Create a safe filename that won't exceed Windows path limits."""
    name_hash = _compute_name_hash(original_name)

    if parent_path is not None:
        parent_len = len(str(parent_path)) + 1  # +1 for path separator
        max_filename_len = WINDOWS_MAX_PATH - parent_len - 10  # -10 safety margin
        max_filename_len = max(
            MIN_READABLE_LENGTH + HASH_LENGTH + len(extension) + 1,
            min(max_filename_len, MAX_SAFE_NAME_LENGTH),
        )
    else:
        max_filename_len = MAX_SAFE_NAME_LENGTH

    reserved_length = len(extension) + HASH_LENGTH + 1  # +1 for the dash
    available_length = max_filename_len - reserved_length

    if len(original_name) > available_length:
        truncated = _truncate_name(
            original_name, max(MIN_READABLE_LENGTH, available_length)
        )
        safe_name = f"{truncated}-{name_hash}{extension}"
    else:
        safe_name = f"{original_name}{extension}"

    return safe_name


def create_safe_log_filename(base_name: str, log_type: str) -> str:
    """Create a safe log filename that won't exceed path limits."""
    suffix = f"_{log_type}_log.json"

    name_hash = _compute_name_hash(base_name)

    reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for dash
    available_length = MAX_SAFE_NAME_LENGTH - reserved_length

    truncated = _truncate_name(base_name, available_length)
    safe_filename = f"{truncated}-{name_hash}{suffix}"

    return safe_filename


def ensure_path_safe(path: Path) -> Path:
    r"""Ensure a path won't exceed Windows MAX_PATH limits.

    On Windows, resolves the path (which triggers the \\?\ extended-length
    prefix automatically in most cases under Python 3.6+).
    """
    if platform.system() == "Windows":
        try:
            return path.resolve()
        except OSError:
            return path

    return path


# ---------------------------------------------------------------------------
# Typed path configuration
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict[str, str]] = {
    "PDFs":   {"input": "pdfs_in",   "output": "pdfs_out"},
    "Images": {"input": "images_in", "output": "images_out"},
    "EPUBs":  {"input": "epubs_in",  "output": "epubs_out"},
    "MOBIs":  {"input": "mobis_in",  "output": "mobis_out"},
    "Auto":   {"input": "auto_in",   "output": "auto_out"},
}


def _get_path(file_paths: Dict[str, Any], section: str, key: str) -> Path:
    """Resolve a single path from the ``file_paths`` config section."""
    default = _DEFAULTS.get(section, {}).get(key, "")
    return Path(file_paths.get(section, {}).get(key, default))


@dataclass
class PathConfig:
    """Typed, pre-resolved view of all input/output directories."""

    pdf_input_dir: Path = field(default_factory=lambda: Path("pdfs_in"))
    pdf_output_dir: Path = field(default_factory=lambda: Path("pdfs_out"))
    image_input_dir: Path = field(default_factory=lambda: Path("images_in"))
    image_output_dir: Path = field(default_factory=lambda: Path("images_out"))
    epub_input_dir: Path = field(default_factory=lambda: Path("epubs_in"))
    epub_output_dir: Path = field(default_factory=lambda: Path("epubs_out"))
    mobi_input_dir: Path = field(default_factory=lambda: Path("mobis_in"))
    mobi_output_dir: Path = field(default_factory=lambda: Path("mobis_out"))
    auto_input_dir: Path = field(default_factory=lambda: Path("auto_in"))
    auto_output_dir: Path = field(default_factory=lambda: Path("auto_out"))

    use_input_as_output: bool = False

    @classmethod
    def from_paths_config(cls, paths_config: Dict[str, Any]) -> "PathConfig":
        """Build a :class:`PathConfig` from a raw ``paths_config`` dict."""
        fp = paths_config.get("file_paths", {})
        general = paths_config.get("general", {})
        return cls(
            pdf_input_dir=_get_path(fp, "PDFs", "input"),
            pdf_output_dir=_get_path(fp, "PDFs", "output"),
            image_input_dir=_get_path(fp, "Images", "input"),
            image_output_dir=_get_path(fp, "Images", "output"),
            epub_input_dir=_get_path(fp, "EPUBs", "input"),
            epub_output_dir=_get_path(fp, "EPUBs", "output"),
            mobi_input_dir=_get_path(fp, "MOBIs", "input"),
            mobi_output_dir=_get_path(fp, "MOBIs", "output"),
            auto_input_dir=_get_path(fp, "Auto", "input"),
            auto_output_dir=_get_path(fp, "Auto", "output"),
            use_input_as_output=general.get(
                "input_paths_is_output_path", False
            ),
        )

    def base_dirs_for_type(self, processing_type: str) -> tuple[Path, Path]:
        """Return ``(input_dir, output_dir)`` for a processing type string."""
        mapping: Dict[str, tuple[Path, Path]] = {
            "pdfs":   (self.pdf_input_dir,   self.pdf_output_dir),
            "images": (self.image_input_dir, self.image_output_dir),
            "epubs":  (self.epub_input_dir,  self.epub_output_dir),
            "mobis":  (self.mobi_input_dir,  self.mobi_output_dir),
            "auto":   (self.auto_input_dir,  self.auto_output_dir),
        }
        if processing_type not in mapping:
            raise ValueError(f"Unknown processing type: {processing_type!r}")
        return mapping[processing_type]

    def ensure_output_dirs(self) -> None:
        """Create all output directories (when *not* using input-as-output)."""
        if self.use_input_as_output:
            return
        for d in (self.pdf_output_dir, self.image_output_dir,
                  self.epub_output_dir, self.mobi_output_dir):
            d.mkdir(parents=True, exist_ok=True)

    def ensure_input_dirs(self) -> None:
        """Create all input directories (interactive-mode convenience)."""
        for d in (self.pdf_input_dir, self.image_input_dir,
                  self.epub_input_dir, self.auto_input_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.auto_output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Directory ensure / collect helpers
# ---------------------------------------------------------------------------

def ensure_directory(path: Path, create: bool = True) -> Path:
    """Ensure a directory exists, optionally creating it.

    Returns the resolved absolute path. Raises ValueError when the path
    exists but is not a directory; FileNotFoundError when create is False
    and the directory does not exist.
    """
    resolved = path.resolve()

    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"Path exists but is not a directory: {resolved}")
        return resolved

    if create:
        resolved.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {resolved}")
        return resolved

    raise FileNotFoundError(f"Directory does not exist: {resolved}")


def ensure_directories(*paths: Path, create: bool = True) -> List[Path]:
    """Ensure multiple directories exist, optionally creating them."""
    return [ensure_directory(p, create=create) for p in paths]


def ensure_parent_directory(file_path: Path, create: bool = True) -> Path:
    """Ensure the parent directory of a file exists."""
    parent = file_path.resolve().parent
    return ensure_directory(parent, create=create)


def get_output_directories_from_config(
    paths_config: dict[str, Any],
) -> dict[str, Path]:
    """Extract and create output directories from paths configuration."""
    file_paths = paths_config.get("file_paths", {})
    output_dirs: dict[str, Path] = {}

    for category in DOCUMENT_CATEGORIES:
        key = category.lower()
        if category in file_paths:
            output_path = file_paths[category].get("output")
            if output_path:
                output_dirs[key] = ensure_directory(Path(output_path))

    return output_dirs


def get_input_directories_from_config(
    paths_config: dict[str, Any],
) -> dict[str, Path]:
    """Extract input directories from paths configuration."""
    file_paths = paths_config.get("file_paths", {})
    input_dirs: dict[str, Path] = {}

    for category in DOCUMENT_CATEGORIES:
        key = category.lower()
        if category in file_paths:
            input_path = file_paths[category].get("input")
            if input_path:
                input_dirs[key] = Path(input_path)

    return input_dirs


def get_logs_directory(
    paths_config: dict[str, Any],
    create: bool = True,
) -> Optional[Path]:
    """Get logs directory from configuration."""
    logs_dir = paths_config.get("general", {}).get("logs_dir")
    if logs_dir:
        return ensure_directory(Path(logs_dir), create=create)
    return None


def collect_scan_directories(paths_config: dict[str, Any]) -> List[Path]:
    """Collect all unique input and output directories for scanning.

    Useful for batch checking and repair operations that need to scan
    multiple directories for artifacts.
    """
    file_paths = paths_config.get("file_paths", {})
    scan_dirs: set[Path] = set()

    for category in DOCUMENT_CATEGORIES:
        if category in file_paths:
            for key in ["input", "output"]:
                path_str = file_paths[category].get(key)
                if path_str:
                    dir_path = Path(path_str)
                    if dir_path.exists() or key == "output":
                        scan_dirs.add(
                            ensure_directory(
                                dir_path, create=(key == "output")
                            )
                        )

    return sorted(scan_dirs)


__all__ = [
    # safe-paths
    "WINDOWS_MAX_PATH",
    "MAX_SAFE_NAME_LENGTH",
    "HASH_LENGTH",
    "MIN_READABLE_LENGTH",
    "create_safe_directory_name",
    "create_safe_filename",
    "create_safe_log_filename",
    "ensure_path_safe",
    # path-config
    "PathConfig",
    # directory-utils
    "ensure_directory",
    "ensure_directories",
    "ensure_parent_directory",
    "get_output_directories_from_config",
    "get_input_directories_from_config",
    "get_logs_directory",
    "collect_scan_directories",
]
