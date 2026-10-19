"""Auto mode: which items a folder holds and which method serves each.

:func:`scan` finds the PDFs, image folders, EPUBs and MOBIs directly in a
folder; :func:`decide` picks the method of one item: EPUB and MOBI take
their text, a PDF with a text layer takes it when searchable PDFs are read
as text, and every other PDF and image folder takes the run's method.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from chronotranscriber.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronotranscriber.sources import SourceItem, source_type
from chronotranscriber.sources.pdf import PDFProcessor

__all__ = ["AutoScan", "decide", "file_item", "scan"]

logger = logging.getLogger(__name__)

_ORDER = ("pdf", "images", "epub", "mobi")


@dataclass(frozen=True)
class AutoScan:
    """The items of a folder in auto mode and the loose images it skipped."""

    items: tuple[SourceItem, ...]
    loose_images: int = 0


def _is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def _folder_has_images(path: str) -> bool:
    with os.scandir(path) as entries:
        return any(
            entry.is_file(follow_symlinks=False) and _is_image_name(entry.name)
            for entry in entries
        )


def scan(folder: Path) -> AutoScan:
    """Return the items directly in *folder*: PDFs, image folders, EPUB, MOBI.

    Subfolders count when they directly hold a page image. Loose images
    have no auto-mode pipeline; they are counted, not returned.
    """
    found: dict[str, list[SourceItem]] = {kind: [] for kind in _ORDER}
    loose_images = 0
    with os.scandir(folder) as entries:
        for entry in entries:
            path = Path(entry.path)
            if entry.is_file(follow_symlinks=False):
                kind = source_type(Path(entry.name))
                if kind in found:
                    found[kind].append(SourceItem(path, kind))
                elif _is_image_name(entry.name):
                    loose_images += 1
            elif entry.is_dir(follow_symlinks=False) and _folder_has_images(entry.path):
                found["images"].append(SourceItem(path, "images"))
    items = tuple(item for kind in _ORDER for item in found[kind])
    logger.info(
        "Scanned %s: %s",
        folder,
        ", ".join(f"{len(found[kind])} {kind}" for kind in _ORDER),
    )
    return AutoScan(items=items, loose_images=loose_images)


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
