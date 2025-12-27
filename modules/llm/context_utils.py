"""Context resolution utilities for ChronoTranscriber.

This module provides hierarchical context resolution for transcription tasks,
supporting file-specific, folder-specific, and global fallback context files.

Context Resolution Hierarchy:
1. File-specific: A .txt file with the same name as the input file
   - For PDFs: document.pdf → document.txt (same directory)
   - For EPUBs: book.epub → book.txt (same directory)
   - For images: image.png → image.txt (same directory)
2. Folder-specific: A .txt file with the same name as the parent folder
   - For image folders: my_images/ → my_images.txt (in parent directory)
   - For PDFs in subfolder: pdfs/archive/ → archive.txt (in pdfs/)
3. Global fallback: additional_context/additional_context.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from modules.infra.logger import setup_logger
from modules.config.config_loader import PROJECT_ROOT

logger = setup_logger(__name__)

# Default threshold for context size warning (in characters)
DEFAULT_CONTEXT_SIZE_THRESHOLD = 4000


def resolve_context_for_file(
    file_path: Path,
    global_context_path: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve context for a specific file using the hierarchy.
    
    Searches for context in this order:
    1. File-specific: <filename_without_ext>.txt in the same directory
    2. Folder-specific: <parent_folder_name>.txt in the grandparent directory
    3. Global fallback: The provided global_context_path or default location
    
    Parameters
    ----------
    file_path : Path
        Path to the input file (PDF, EPUB, or image)
    global_context_path : Optional[Path]
        Path to global context file (fallback). If None, uses default location.
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    file_path = Path(file_path).resolve()
    
    # 1. File-specific context: same name with .txt extension
    file_specific = file_path.with_suffix(".txt")
    if file_specific.exists() and file_specific != file_path:
        content = _read_and_validate_context(file_specific, size_threshold)
        if content:
            logger.debug(f"Using file-specific context: {file_specific}")
            return content, file_specific
    
    # 2. Folder-specific context: folder name + .txt in parent directory
    parent_folder = file_path.parent
    if parent_folder.parent.exists():
        folder_specific = parent_folder.parent / f"{parent_folder.name}.txt"
        if folder_specific.exists():
            content = _read_and_validate_context(folder_specific, size_threshold)
            if content:
                logger.debug(f"Using folder-specific context: {folder_specific}")
                return content, folder_specific
    
    # 3. Global fallback
    if global_context_path is None:
        global_context_path = PROJECT_ROOT / "additional_context" / "additional_context.txt"
    
    if global_context_path.exists():
        content = _read_and_validate_context(global_context_path, size_threshold)
        if content:
            logger.debug(f"Using global context: {global_context_path}")
            return content, global_context_path
    
    return None, None


def resolve_context_for_folder(
    folder_path: Path,
    global_context_path: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve context for an image folder using the hierarchy.
    
    Searches for context in this order:
    1. Folder-specific: <folder_name>.txt in the parent directory
    2. In-folder context: context.txt inside the folder itself
    3. Global fallback: The provided global_context_path or default location
    
    Parameters
    ----------
    folder_path : Path
        Path to the image folder
    global_context_path : Optional[Path]
        Path to global context file (fallback). If None, uses default location.
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    folder_path = Path(folder_path).resolve()
    
    # 1. Folder-specific context: folder name + .txt in parent directory
    if folder_path.parent.exists():
        folder_specific = folder_path.parent / f"{folder_path.name}.txt"
        if folder_specific.exists():
            content = _read_and_validate_context(folder_specific, size_threshold)
            if content:
                logger.debug(f"Using folder-specific context: {folder_specific}")
                return content, folder_specific
    
    # 2. In-folder context.txt
    in_folder_context = folder_path / "context.txt"
    if in_folder_context.exists():
        content = _read_and_validate_context(in_folder_context, size_threshold)
        if content:
            logger.debug(f"Using in-folder context: {in_folder_context}")
            return content, in_folder_context
    
    # 3. Global fallback
    if global_context_path is None:
        global_context_path = PROJECT_ROOT / "additional_context" / "additional_context.txt"
    
    if global_context_path.exists():
        content = _read_and_validate_context(global_context_path, size_threshold)
        if content:
            logger.debug(f"Using global context: {global_context_path}")
            return content, global_context_path
    
    return None, None


def resolve_context_for_image(
    image_path: Path,
    global_context_path: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve context for a specific image file.
    
    Searches for context in this order:
    1. Image-specific: <image_name>.txt in the same directory
    2. Folder-specific: <parent_folder_name>.txt in grandparent directory
    3. In-folder context: context.txt in the image's directory
    4. Global fallback: The provided global_context_path or default location
    
    Parameters
    ----------
    image_path : Path
        Path to the image file
    global_context_path : Optional[Path]
        Path to global context file (fallback). If None, uses default location.
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    image_path = Path(image_path).resolve()
    
    # 1. Image-specific context
    image_specific = image_path.with_suffix(".txt")
    if image_specific.exists():
        content = _read_and_validate_context(image_specific, size_threshold)
        if content:
            logger.debug(f"Using image-specific context: {image_specific}")
            return content, image_specific
    
    # 2. Folder-specific context (folder name + .txt in grandparent)
    parent_folder = image_path.parent
    if parent_folder.parent.exists():
        folder_specific = parent_folder.parent / f"{parent_folder.name}.txt"
        if folder_specific.exists():
            content = _read_and_validate_context(folder_specific, size_threshold)
            if content:
                logger.debug(f"Using folder-specific context: {folder_specific}")
                return content, folder_specific
    
    # 3. In-folder context.txt
    in_folder_context = parent_folder / "context.txt"
    if in_folder_context.exists():
        content = _read_and_validate_context(in_folder_context, size_threshold)
        if content:
            logger.debug(f"Using in-folder context: {in_folder_context}")
            return content, in_folder_context
    
    # 4. Global fallback
    if global_context_path is None:
        global_context_path = PROJECT_ROOT / "additional_context" / "additional_context.txt"
    
    if global_context_path.exists():
        content = _read_and_validate_context(global_context_path, size_threshold)
        if content:
            logger.debug(f"Using global context: {global_context_path}")
            return content, global_context_path
    
    return None, None


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
        
        # Warn if context is large
        if len(content) > size_threshold:
            logger.warning(
                f"Context file '{context_path.name}' is large ({len(content):,} chars). "
                f"Consider reducing to under {size_threshold:,} chars for optimal performance."
            )
        
        return content
        
    except Exception as e:
        logger.warning(f"Failed to read context file {context_path}: {e}")
        return None


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
