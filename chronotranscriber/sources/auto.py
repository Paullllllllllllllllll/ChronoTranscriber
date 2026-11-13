"""Auto mode: which items a folder holds and which method serves each.

:func:`scan` finds the PDFs, image folders, EPUBs and MOBIs of a folder,
directly in it or at every level; :func:`decide` picks the method of one
item: EPUB and MOBI take their text, a PDF with a text layer takes it when
searchable PDFs are read as text, and every other PDF and image folder takes
the run's method.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from chronotranscriber.common import discovery
from chronotranscriber.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronotranscriber.sources import (
    Found,
    SourceItem,
    has_images,
    named,
    scan_input,
    source_type,
)
from chronotranscriber.sources.pdf import PDFProcessor

__all__ = ["decide", "file_item", "find", "scan"]

logger = logging.getLogger(__name__)

_ORDER = ("pdf", "images", "epub", "mobi")


def scan(
    folder: Path, *, recursive: bool = False, output_dir: Path | None = None
) -> Found:
    """Return the items of *folder* in auto mode: PDFs, image folders, EPUB, MOBI.

    Without *recursive* the files directly in *folder* and its subfolders
    that directly hold a page image; with it the files and image folders at
    every level, an image folder below another one left out and listed in
    ``nested``. The search is :func:`~chronotranscriber.sources.scan_input`,
    *output_dir* the run's output folder. Loose images directly in *folder*
    have no auto-mode pipeline; they are counted, not returned. The items
    come in the natural order of their paths.
    """
    found: list[SourceItem] = []
    loose_images = 0
    images: list[Path] = []
    working: list[str] = []
    depth = None if recursive else 1
    for path in scan_input(folder, depth=depth, output_dir=output_dir, working=working):
        top = os.path.dirname(path) == str(folder)
        is_image = path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        if not top and is_image:
            images.append(path)
        elif top or recursive:
            kind = source_type(Path(path.name))
            if kind in _ORDER:
                found.append(SourceItem(path, kind))
            elif top and is_image:
                loose_images += 1
    folders = discovery.image_folders(images, SUPPORTED_IMAGE_EXTENSIONS)
    found += [SourceItem(path, "images") for path in folders.folders]
    items = tuple(discovery.ordered(found, folder, lambda item: item.path))
    logger.info(
        "Scanned %s: %s",
        folder,
        ", ".join(
            f"{sum(item.type == kind for item in items)} {kind}" for kind in _ORDER
        ),
    )
    return Found(
        items,
        loose_images=loose_images,
        nested=folders.nested,
        working=tuple(working),
    )


def find(
    input_path: Path,
    *,
    files: Sequence[str] = (),
    recursive: bool = False,
    output_dir: Path | None = None,
) -> Found:
    """Return the auto-mode items of an input file or folder.

    A folder gives the items *files* name (:func:`~chronotranscriber.sources.named`):
    PDF, EPUB and MOBI files and folders that directly hold a page image;
    without *files* those of :func:`scan`. A path named twice is taken once.

    Raises:
        DiscoveryError: Naming the *files* that are no auto-mode item.
    """
    if input_path.is_file():
        single = file_item(input_path)
        return Found((single,) if single is not None else ())
    if not files:
        return scan(input_path, recursive=recursive, output_dir=output_dir)
    paths = named(
        input_path,
        files,
        _named_item,
        "are not PDF, EPUB or MOBI files or image folders",
    )
    items = [
        SourceItem(path, "images") if path.is_dir() else file_item(path)
        for path in dict.fromkeys(paths)
    ]
    return Found(
        tuple(
            discovery.ordered(
                [item for item in items if item is not None],
                input_path,
                lambda item: item.path,
            )
        )
    )


def _named_item(path: Path) -> bool:
    """Whether *path* is an auto-mode item: a PDF, EPUB or MOBI, or an image folder."""
    if path.is_dir():
        return has_images(path)
    return path.is_file() and file_item(path) is not None


def file_item(path: Path) -> SourceItem | None:
    """Return a single input file as an auto-mode item: PDF, EPUB or MOBI."""
    kind = source_type(path)
    if kind in ("pdf", "epub", "mobi"):
        return SourceItem(path, kind)
    return None


def decide(item: SourceItem, method: str, searchable_pdfs: str) -> str:
    """Return the method of *item* in auto mode.

    *method* serves scanned PDFs and image folders, and PDFs with a text
    layer unless *searchable_pdfs* is ``text``.
    """
    if item.type in ("epub", "mobi"):
        return "native"
    if (
        item.type == "pdf"
        and searchable_pdfs == "text"
        and PDFProcessor(item.path).is_native_pdf()
    ):
        return "native"
    return method
