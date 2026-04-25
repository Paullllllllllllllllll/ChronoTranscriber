"""Context resolution utilities for ChronoTranscriber.

This module provides hierarchical context resolution for transcription tasks,
using filename-suffix-based matching across three resolution levels.

Context Resolution Hierarchy (most specific wins):
1. File-specific:   {input_stem}_transcr_context.txt   next to the input file
2. Folder-specific: {parent_folder}_transcr_context.txt next to the input's parent folder
3. General fallback: context/transcr_context.txt        in the project root

Suffix: transcr_context
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from modules.infra.logger import setup_logger
from modules.config.config_loader import PROJECT_ROOT

logger = setup_logger(__name__)

_CONTEXT_DIR = PROJECT_ROOT / "context"

# Default threshold for context size warning (in characters)
DEFAULT_CONTEXT_SIZE_THRESHOLD = 4000

_SUFFIX = "transcr_context"


def _resolve_context(
    suffix: str,
    input_path: Optional[Path] = None,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Generic hierarchical context resolution.

    Searches for context in this order:
    1. File-specific:   {input_stem}_{suffix}.txt   in the same directory as *input_path*
    2. Folder-specific: {parent_folder}_{suffix}.txt in the grandparent directory
    3. General fallback: context/{suffix}.txt        in the project context directory

    Parameters
    ----------
    suffix : str
        Context-file suffix without leading underscore (e.g. ``"transcr_context"``).
    input_path : Optional[Path]
        Path to the input file or folder (enables file- and folder-specific lookup).
    context_dir : Optional[Path]
        Override for the project-level context directory (defaults to
        ``PROJECT_ROOT/context``).
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    effective_context_dir = context_dir or _CONTEXT_DIR
    filename_suffix = f"_{suffix}.txt"

    if input_path is not None:
        input_path = Path(input_path).resolve()

        if input_path.is_file():
            # 1. File-specific context
            file_specific = input_path.with_name(f"{input_path.stem}{filename_suffix}")
            if file_specific.exists():
                content = _read_and_validate_context(file_specific, size_threshold)
                if content:
                    logger.info(f"Using file-specific context: {file_specific}")
                    return content, file_specific

            # 2. Folder-specific context (parent folder name)
            parent_folder = input_path.parent
            if parent_folder.parent.exists():
                folder_specific = parent_folder.parent / f"{parent_folder.name}{filename_suffix}"
                if folder_specific.exists():
                    content = _read_and_validate_context(folder_specific, size_threshold)
                    if content:
                        logger.info(f"Using folder-specific context: {folder_specific}")
                        return content, folder_specific

        elif input_path.is_dir():
            # For folders: folder-specific context lives next to the folder
            if input_path.parent.exists():
                folder_specific = input_path.parent / f"{input_path.name}{filename_suffix}"
                if folder_specific.exists():
                    content = _read_and_validate_context(folder_specific, size_threshold)
                    if content:
                        logger.info(f"Using folder-specific context: {folder_specific}")
                        return content, folder_specific

    # 3. General fallback
    general_fallback = effective_context_dir / f"{suffix}.txt"
    if general_fallback.exists():
        content = _read_and_validate_context(general_fallback, size_threshold)
        if content:
            logger.info(f"Using general context: {general_fallback}")
            return content, general_fallback

    logger.debug(f"No {suffix} context found")
    return None, None


def resolve_context_for_file(
    file_path: Path,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve transcription context for a specific file.

    Parameters
    ----------
    file_path : Path
        Path to the input file (PDF, EPUB, image, etc.)
    context_dir : Optional[Path]
        Override for the project-level context directory.
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    return _resolve_context(_SUFFIX, file_path, context_dir, size_threshold)


def resolve_context_for_folder(
    folder_path: Path,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve transcription context for an image folder.

    Parameters
    ----------
    folder_path : Path
        Path to the image folder.
    context_dir : Optional[Path]
        Override for the project-level context directory.
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    return _resolve_context(_SUFFIX, folder_path, context_dir, size_threshold)


def resolve_context_for_image(
    image_path: Path,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve transcription context for a specific image file.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    context_dir : Optional[Path]
        Override for the project-level context directory.
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    return _resolve_context(_SUFFIX, image_path, context_dir, size_threshold)


def load_context_from_path(
    context_path: Optional[Path],
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Optional[str]:
    """Load context from a specific path with validation.

    Parameters
    ----------
    context_path : Optional[Path]
        Path to the context file
    size_threshold : int
        Character count threshold for size warning

    Returns
    -------
    Optional[str]
        The context content, or None if path is None or file doesn't exist
    """
    if context_path is None:
        return None

    context_path = Path(context_path)
    if not context_path.exists():
        return None

    return _read_and_validate_context(context_path, size_threshold)


def _read_and_validate_context(
    context_path: Path,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Optional[str]:
    """Read and validate a context file.

    Parameters
    ----------
    context_path : Path
        Path to the context file
    size_threshold : int
        Character count threshold for size warning

    Returns
    -------
    Optional[str]
        The context content, or None if file is empty or unreadable
    """
    try:
        content = context_path.read_text(encoding="utf-8").strip()

        if not content:
            return None

        if len(content) > size_threshold:
            logger.warning(
                f"Context file '{context_path.name}' is large ({len(content):,} chars). "
                f"Consider reducing to under {size_threshold:,} chars for optimal performance."
            )

        return content

    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(f"Failed to read context file {context_path}: {exc}")
        return None
