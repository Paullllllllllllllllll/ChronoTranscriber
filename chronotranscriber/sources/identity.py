"""Input identity: the ``input_sha256`` a store document records for its input.

A file's identity is its SHA-256. An image folder's identity is the SHA-256
of a JSON list of ``[file name, file SHA-256]`` pairs, one per supported
image in page order (:func:`~chronotranscriber.sources.images.list_folder_images`),
so adding, removing, editing or renaming an image, or anything that changes
which file is page *n*, changes it. ``transcribe``, ``repair`` and the
classify commands all use :func:`input_identity`.

:func:`input_snapshot` is the cheap check between planning and the lease:
each file's name, size and modification time in nanoseconds, in the same
order; a run hashes again only when the snapshot differs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from chronotranscriber.common.store import file_sha256
from chronotranscriber.output.paths import natural_sort_key
from chronotranscriber.sources.images import list_folder_images

__all__ = ["Snapshot", "input_identity", "input_snapshot", "reordered"]

type Snapshot = tuple[tuple[str, int, int], ...]
"""Each file's name, size and ``st_mtime_ns``, in page order."""


def _files(path: Path) -> list[Path]:
    return list_folder_images(path) if path.is_dir() else [path]


def _sha256(path: Path) -> str:
    digest = file_sha256(path)
    if digest is None:
        raise FileNotFoundError(f"No such file: {path}")
    return digest


def input_identity(path: Path) -> str:
    """Return the identity of the file or image folder at *path*.

    Raises:
        OSError: The input or one of its images cannot be read.
    """
    if not path.is_dir():
        return _sha256(path)
    listing = [[image.name, _sha256(image)] for image in list_folder_images(path)]
    text = json.dumps(listing, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_snapshot(path: Path) -> Snapshot:
    """Return the name, size and modification time of the input's files.

    Raises:
        OSError: The input or one of its images cannot be read.
    """
    entries = []
    for file in _files(path):
        stat = file.stat()
        entries.append((file.name, stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


_LLM_LABEL = "_pre_processed.jpg"
_TESSERACT_LABEL = "_tess_preprocessed"


def _known(label: str) -> bool:
    """Whether *label* has the LLM or the Tesseract form of a folder page."""
    return label.endswith(_LLM_LABEL) or f"{_TESSERACT_LABEL}." in label


def _names_image(label: str, image: Path) -> bool:
    """Whether a page unit's *label* names *image* (LLM or Tesseract naming)."""
    if label == f"{image.name}{_LLM_LABEL}":
        return True
    return any(
        label.startswith(f"{base}{_TESSERACT_LABEL}.")
        for base in (image.stem, image.name)
    )


def reordered(name: str, folder: Path, labels: Mapping[int, str | None]) -> str | None:
    """Return why a stored folder document's pages no longer fit its images.

    *labels* maps the index of every stored page unit to its label. The
    stored pages must be exactly the folder's images, one unit per image,
    so an added or removed image refuses the document, and each label must
    name the image the page order now puts at its index. Before v1.2.1 a
    folder image could land at another index: images whose names tie in
    natural order (``page_1``, ``page_01``) took the file system's listing
    order, and Tesseract filed an image named like a PDF page under the
    number in its name. A label of another form, or none, cannot be
    checked; it is trusted only when the image names do not tie.

    Raises:
        OSError: The folder cannot be listed.
    """
    images = list_folder_images(folder)
    if set(labels) != set(range(len(images))):
        return (
            f"The images of '{name}' changed: it stores {len(labels)} page(s), the "
            f"folder holds {len(images)} image(s). Skipping it; use --force to "
            "transcribe it from scratch."
        )
    tied = any(
        natural_sort_key(first.name) == natural_sort_key(second.name)
        for first, second in zip(images, images[1:], strict=False)
    )
    for idx, label in sorted(labels.items()):
        if label is not None and _known(label):
            fits = _names_image(label, images[idx])
        else:
            fits = not tied
        if not fits:
            return (
                f"The page order of '{name}' changed: its stored page {idx + 1} "
                f"does not name {images[idx].name}, the image the page order "
                "(natural order, ties broken by file name) puts there. Skipping "
                "it; use --force to transcribe it from scratch."
            )
    return None
