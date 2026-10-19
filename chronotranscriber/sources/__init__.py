"""Input sources: discovery and type detection.

:func:`discover_items` finds the items a run of one source type processes;
:mod:`chronotranscriber.sources.auto` scans a folder for auto mode. The
submodules read the sources: PDFs, image folders, EPUB, MOBI and audio.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from chronotranscriber.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronotranscriber.sources.audio.constants import SUPPORTED_AUDIO_EXTENSIONS

__all__ = [
    "EXTENSIONS",
    "SOURCE_TYPES",
    "SUPPORTED_MOBI_EXTENSIONS",
    "DiscoveryError",
    "SourceItem",
    "discover_items",
    "has_images",
    "source_type",
]

SOURCE_TYPES = ("pdf", "images", "epub", "mobi", "audio")

SUPPORTED_MOBI_EXTENSIONS = (".mobi", ".azw", ".azw3", ".kfx")
"""Extensions of MOBI and Kindle ebooks."""

EXTENSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "pdf": (".pdf",),
        "epub": (".epub",),
        "mobi": SUPPORTED_MOBI_EXTENSIONS,
        "audio": tuple(sorted(SUPPORTED_AUDIO_EXTENSIONS)),
    }
)
"""File extensions of the file sources; image folders are directories."""


class DiscoveryError(ValueError):
    """The input cannot serve the requested source type."""


@dataclass(frozen=True)
class SourceItem:
    """One input item: a file, or for images a folder of page images."""

    path: Path
    type: str

    @property
    def name(self) -> str:
        """The item's file or folder name."""
        return self.path.name


def source_type(path: Path) -> str | None:
    """Return the source type of *path* by its suffix, or None.

    A directory counts as an image folder.
    """
    if path.is_dir():
        return "images"
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".epub":
        return "epub"
    if suffix in SUPPORTED_MOBI_EXTENSIONS:
        return "mobi"
    if suffix in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    return None


def has_images(folder: Path) -> bool:
    """Whether *folder* directly holds a supported page image."""
    return any(
        entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for entry in folder.iterdir()
    )


def discover_items(
    input_path: Path,
    kind: str,
    *,
    files: Sequence[str] = (),
    recursive: bool = False,
) -> list[SourceItem]:
    """Return the items of type *kind* at *input_path*.

    Image folders: the named *files* (folders) inside the input, its direct
    subfolders with *recursive*, else the input folder itself. Other types:
    the input file itself, the named *files* inside the input folder, or
    the files with the type's extensions (all subfolders with *recursive*).

    Raises:
        DiscoveryError: An image run whose input is not a folder, or named
            *files* that are not present in the input folder (for images: that
            are not folders).
    """
    if kind == "images":
        if not input_path.is_dir():
            raise DiscoveryError(
                f"For image processing, input must be a directory: {input_path}"
            )
        if files:
            paths = _named(
                input_path, files, Path.is_dir, f"are not folders in {input_path}"
            )
        elif recursive:
            paths = [entry for entry in input_path.iterdir() if entry.is_dir()]
        else:
            paths = [input_path]
        return [SourceItem(path, kind) for path in paths]

    extensions = EXTENSIONS.get(kind, ())
    if input_path.is_file():
        paths = [input_path]
    elif files:
        paths = _named(input_path, files, Path.is_file, f"do not exist in {input_path}")
    elif recursive:
        paths = [
            path
            for ext in extensions
            for path in input_path.rglob(f"*{ext}", case_sensitive=False)
        ]
    else:
        paths = [
            path
            for ext in extensions
            for path in input_path.glob(f"*{ext}", case_sensitive=False)
        ]
    return [SourceItem(path, kind) for path in paths]


def _named(
    input_path: Path,
    files: Sequence[str],
    present: Callable[[Path], bool],
    what: str,
) -> list[Path]:
    """Return the paths of *files* in *input_path*, each one *present*.

    Raises:
        DiscoveryError: Naming the files that are not *present*.
    """
    paths = [input_path / name for name in files]
    missing = [
        name for name, path in zip(files, paths, strict=True) if not present(path)
    ]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise DiscoveryError(f"--files names that {what}: {names}")
    return paths
