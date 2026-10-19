"""Output paths: item layouts, natural sorting and Windows-safe names.

:func:`item_paths` derives where an input item writes: beside the input, in
a hash-suffixed folder below the output folder, or in the mirrored input
tree. Names stay within MAX_PATH.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def natural_sort_key(name: str) -> list[object]:
    """Return a case-insensitive natural-sort key for a file name.

    Splits digit runs so that ``page_2`` sorts before ``page_10``. The LLM
    and Tesseract methods share it, so a page index names the same file
    under either method.
    """
    parts = _NATURAL_SORT_RE.split(name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


WINDOWS_MAX_PATH = 260
MAX_SAFE_NAME_LENGTH = 80
HASH_LENGTH = 8
MIN_READABLE_LENGTH = 20


class OutputMode(StrEnum):
    """Output directory layout mode."""

    HASH = "hash"
    MIRROR = "mirror"


def mirror_output_path(
    item: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Return the mirrored output directory for *item*.

    Replicates the relative path of *item* under *output_root*.
    """
    try:
        rel = item.relative_to(input_root)
    except ValueError:
        rel = Path(item.name)
    return output_root / rel


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
    parent_path: Path | None = None,
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


def relative_key(item: Path, input_root: Path | None) -> str | None:
    """Return *item* relative to *input_root* as a hash key, if one applies.

    A single-file input makes *item* equal to the root; the key is then None,
    so the folder name derives from the item's own name.
    """
    if input_root is None:
        return None
    try:
        rel = str(item.relative_to(input_root))
    except ValueError:
        return None
    if rel in (".", ""):
        return None
    return rel


@dataclass(frozen=True)
class ItemPaths:
    """Where one input item writes.

    *output* is the rendered file of the first text format. *folder* is the
    item's folder: the folder of the rendered files in the hash and mirror
    layouts, beside the input the folder kept only with
    ``--keep-working-files``.
    """

    folder: Path
    output: Path


def item_paths(
    item: Path,
    *,
    output_dir: Path | None,
    layout: str,
    input_root: Path | None,
    output_format: str = "txt",
    mirror: bool = True,
) -> ItemPaths:
    """Return the paths of *item*; nothing is created.

    *output_dir* None writes the output beside the input; its *folder* is a
    hash-suffixed folder next to it. The mirror layout
    (when *mirror* allows it) writes into the input tree replicated below
    *output_dir* and names the output with its real extension; the hash
    layout uses one hash-suffixed folder per item, keyed by the
    input-relative path. Outside the mirror layout the output is named with
    ``.txt`` and the writer gives it the real extension.
    """
    is_folder = item.is_dir()
    stem = item.name if is_folder else item.stem
    if output_dir is None:
        folder = item.parent / create_safe_directory_name(stem)
        output = item.parent / create_safe_filename(stem, ".txt", item.parent)
    elif mirror and layout == OutputMode.MIRROR and input_root is not None:
        folder = mirror_output_path(
            item if is_folder else item.parent, input_root, output_dir
        )
        output = folder / create_safe_filename(stem, f".{output_format}", folder)
    else:
        key = relative_key(item, input_root)
        folder = output_dir / create_safe_directory_name(key or stem)
        output = folder / create_safe_filename(stem, ".txt", folder)
    return ItemPaths(folder=folder, output=output)


def prepare_item_paths(
    item: Path,
    *,
    output_dir: Path | None,
    layout: str,
    input_root: Path | None,
    output_format: str = "txt",
    mirror: bool = True,
) -> ItemPaths:
    """Return :func:`item_paths` after creating the folder of the output.

    No working folder and no file is created.
    """
    paths = item_paths(
        item,
        output_dir=output_dir,
        layout=layout,
        input_root=input_root,
        output_format=output_format,
        mirror=mirror,
    )
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = [
    "WINDOWS_MAX_PATH",
    "MAX_SAFE_NAME_LENGTH",
    "HASH_LENGTH",
    "MIN_READABLE_LENGTH",
    "ItemPaths",
    "OutputMode",
    "create_safe_directory_name",
    "create_safe_filename",
    "item_paths",
    "mirror_output_path",
    "natural_sort_key",
    "prepare_item_paths",
    "relative_key",
]
