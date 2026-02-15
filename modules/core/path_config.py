"""Centralized path resolution from paths_config dictionaries.

Eliminates repeated ``paths_config.get('file_paths', {}).get(...)`` chains
that were previously scattered across workflow.py and unified_transcriber.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


# Default directory names used when a config key is missing.
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
    """Typed, pre-resolved view of all input/output directories.

    Construct via :meth:`from_paths_config` to parse a raw ``paths_config``
    dictionary once and access every directory through plain attributes.
    """

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

    # general settings surfaced for convenience
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
            use_input_as_output=general.get("input_paths_is_output_path", False),
        )

    def base_dirs_for_type(self, processing_type: str) -> tuple[Path, Path]:
        """Return ``(input_dir, output_dir)`` for a given processing type string."""
        mapping: Dict[str, tuple[Path, Path]] = {
            "pdfs":   (self.pdf_input_dir,   self.pdf_output_dir),
            "images": (self.image_input_dir,  self.image_output_dir),
            "epubs":  (self.epub_input_dir,   self.epub_output_dir),
            "mobis":  (self.mobi_input_dir,   self.mobi_output_dir),
            "auto":   (self.auto_input_dir,   self.auto_output_dir),
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
        """Create all input directories (interactive mode convenience)."""
        for d in (self.pdf_input_dir, self.image_input_dir,
                  self.epub_input_dir, self.auto_input_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.auto_output_dir.mkdir(parents=True, exist_ok=True)
