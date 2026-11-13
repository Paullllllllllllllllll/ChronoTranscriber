"""Input sources: discovery and type detection.

:func:`find_items` finds the items a run of one source type processes;
:mod:`chronotranscriber.sources.auto` scans a folder for auto mode. The
submodules read the sources: PDFs, image folders, EPUB, MOBI and audio.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from chronotranscriber.common.discovery import (
    SKIPPED_FOLDERS,
    DiscoveryError,
    image_folders,
    ordered,
    resolve_named,
    scan,
)
from chronotranscriber.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronotranscriber.sources.audio.constants import SUPPORTED_AUDIO_EXTENSIONS

__all__ = [
    "EXTENSIONS",
    "SKIPPED_FOLDERS",
    "SOURCE_TYPES",
    "SUPPORTED_MOBI_EXTENSIONS",
    "DiscoveryError",
    "Found",
    "SourceItem",
    "below",
    "discover_items",
    "find_items",
    "SIDECAR_SUFFIXES",
    "WORKING_FOLDERS",
    "has_images",
    "is_sidecar",
    "named",
    "scan_input",
    "skipped_folders",
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

WORKING_FOLDERS: Final = frozenset({"preprocessed_images_tesseract", "audio_chunks"})
"""The subfolders of an item's working folder kept by ``--keep-working-files``."""

SIDECAR_SUFFIXES: Final = ("_transcr_context", "_transcr_context_image")
"""Stem endings of the context sidecars, which are never pages or items."""


def is_sidecar(path: Path) -> bool:
    """Whether *path* is a context sidecar (text or image) of an item."""
    return path.stem.endswith(SIDECAR_SUFFIXES)


def skipped_folders(
    input_path: Path, output_dir: Path | None, working: list[str] | None = None
) -> Callable[[str], bool]:
    """Return the *skip* test of a scan below *input_path*.

    It leaves out the working folders of :data:`WORKING_FOLDERS` and the
    output folder *output_dir* when it lies below the input folder; the
    shared scan leaves out :data:`SKIPPED_FOLDERS` itself. *working*
    receives the path of each working folder left out, since a folder of
    the user's own can bear such a name.
    """
    inside = output_dir is not None and input_path in output_dir.parents

    def skip(folder: str) -> bool:
        if folder.rpartition("/")[2] in WORKING_FOLDERS:
            if working is not None:
                working.append(folder)
            return True
        return inside and input_path / folder == output_dir

    return skip


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
    """Whether *folder* directly holds a supported page image, not a sidecar."""
    return any(
        entry.is_file()
        and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not is_sidecar(entry)
        for entry in folder.iterdir()
    )


@dataclass(frozen=True)
class Found:
    """The items discovery found and what it left out.

    *loose_images* counts the page images directly in an auto-mode folder,
    which no auto-mode pipeline takes. *nested* lists each image folder left
    out because it lies below another image folder, with that folder.
    *working* lists the folders left out for bearing the name of a kept
    working folder (:data:`WORKING_FOLDERS`), as paths below the input.
    """

    items: tuple[SourceItem, ...]
    loose_images: int = 0
    nested: tuple[tuple[Path, Path], ...] = ()
    working: tuple[str, ...] = ()


def below(files: Sequence[Path], folder: Path) -> list[Path]:
    """Return the *files* that lie in a subfolder of *folder*, not in it."""
    return [path for path in files if os.path.dirname(path) != str(folder)]


def scan_input(
    input_path: Path,
    *,
    depth: int | None,
    output_dir: Path | None = None,
    working: list[str] | None = None,
) -> list[Path]:
    """Return the files below *input_path* that can be pages or items.

    The shared scan to *depth* levels, leaving out the folders of
    :func:`skipped_folders`, which fills *working*, and the context sidecars
    (:func:`is_sidecar`).
    """
    skip = skipped_folders(input_path, output_dir, working)
    return [
        path
        for path in scan(input_path, depth=depth, skip=skip)
        if not is_sidecar(path)
    ]


def find_items(
    input_path: Path,
    kind: str,
    *,
    files: Sequence[str] = (),
    recursive: bool = False,
    output_dir: Path | None = None,
) -> Found:
    """Return the items of type *kind* at *input_path*, in natural path order.

    Image folders: the folders the *files* name (:func:`named`), with
    *recursive* every folder below the input that directly holds a page
    image, else the input folder itself. Other types: the input file itself,
    the files the *files* name, or the files with the type's extensions in
    the folder (at every level with *recursive*). The folder search is
    :func:`scan_input`, *output_dir* the run's output folder. A path named
    twice is taken once. The order is that of
    :func:`~chronotranscriber.common.discovery.ordered`.

    Raises:
        DiscoveryError: An image run whose input is not a folder, or named
            *files* that are not present in the input folder (for images: that
            are not folders).
    """
    nested: tuple[tuple[Path, Path], ...] = ()
    working: list[str] = []
    if kind == "images":
        if not input_path.is_dir():
            raise DiscoveryError(
                f"For image processing, input must be a directory: {input_path}"
            )
        if files:
            paths = named(input_path, files, Path.is_dir, "are not folders")
        elif recursive:
            folders = image_folders(
                below(
                    scan_input(
                        input_path, depth=None, output_dir=output_dir, working=working
                    ),
                    input_path,
                ),
                SUPPORTED_IMAGE_EXTENSIONS,
            )
            paths, nested = list(folders.folders), folders.nested
        else:
            paths = [input_path]
    elif input_path.is_file():
        paths = [input_path]
    elif files:
        paths = named(input_path, files, Path.is_file, "do not exist")
    else:
        extensions = EXTENSIONS.get(kind, ())
        found = scan_input(
            input_path,
            depth=None if recursive else 0,
            output_dir=output_dir,
            working=working,
        )
        return Found(
            tuple(
                SourceItem(path, kind)
                for path in found
                if path.name.lower().endswith(extensions)
            ),
            working=tuple(working),
        )
    items = ordered(dict.fromkeys(paths), input_path, lambda path: path)
    return Found(
        tuple(SourceItem(path, kind) for path in items),
        nested=nested,
        working=tuple(working),
    )


def discover_items(
    input_path: Path,
    kind: str,
    *,
    files: Sequence[str] = (),
    recursive: bool = False,
) -> list[SourceItem]:
    """Return the items of :func:`find_items`.

    Raises:
        DiscoveryError: As :func:`find_items`.
    """
    return list(find_items(input_path, kind, files=files, recursive=recursive).items)


def named(
    input_path: Path,
    files: Sequence[str],
    present: Callable[[Path], bool],
    missing: str,
) -> list[Path]:
    """Return the paths the ``--files`` names give below *input_path*.

    A name is a path relative to the input folder, with forward or back
    slashes; each must be *present*, and *missing* words the error.

    Raises:
        DiscoveryError: Naming the files that are not *present*.
    """
    names = [name.replace("\\", "/") for name in files]
    return resolve_named(input_path, names, present=present, missing=missing)
