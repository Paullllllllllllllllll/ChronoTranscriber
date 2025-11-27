"""Directory management utilities.

Provides centralized functions for directory creation, validation, and path
resolution to eliminate redundant logic across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


def ensure_directory(path: Path, create: bool = True) -> Path:
    """Ensure a directory exists, optionally creating it.
    
    Args:
        path: Path to directory.
        create: If True, create directory if it doesn't exist.
        
    Returns:
        Resolved absolute path to directory.
        
    Raises:
        ValueError: If path exists but is not a directory.
        FileNotFoundError: If path doesn't exist and create is False.
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
    """Ensure multiple directories exist, optionally creating them.
    
    Args:
        *paths: Variable number of directory paths.
        create: If True, create directories if they don't exist.
        
    Returns:
        List of resolved absolute paths.
        
    Raises:
        ValueError: If any path exists but is not a directory.
        FileNotFoundError: If any path doesn't exist and create is False.
    """
    return [ensure_directory(p, create=create) for p in paths]


def ensure_parent_directory(file_path: Path, create: bool = True) -> Path:
    """Ensure the parent directory of a file exists.
    
    Args:
        file_path: Path to file (not directory).
        create: If True, create parent directory if it doesn't exist.
        
    Returns:
        Resolved absolute path to parent directory.
        
    Raises:
        ValueError: If parent path exists but is not a directory.
        FileNotFoundError: If parent doesn't exist and create is False.
    """
    parent = file_path.resolve().parent
    return ensure_directory(parent, create=create)


def get_output_directories_from_config(paths_config: dict) -> dict[str, Path]:
    """Extract and create output directories from paths configuration.
    
    Args:
        paths_config: Paths configuration dictionary.
        
    Returns:
        Dictionary mapping category names to output directory paths.
        Keys: 'pdfs', 'images', 'epubs', 'auto'
    """
    file_paths = paths_config.get("file_paths", {})
    output_dirs = {}
    
    for category, key in [("PDFs", "pdfs"), ("Images", "images"), 
                          ("EPUBs", "epubs"), ("Auto", "auto")]:
        if category in file_paths:
            output_path = file_paths[category].get("output")
            if output_path:
                output_dirs[key] = ensure_directory(Path(output_path))
    
    return output_dirs


def get_input_directories_from_config(paths_config: dict) -> dict[str, Path]:
    """Extract input directories from paths configuration.
    
    Args:
        paths_config: Paths configuration dictionary.
        
    Returns:
        Dictionary mapping category names to input directory paths.
        Keys: 'pdfs', 'images', 'epubs', 'auto'
    """
    file_paths = paths_config.get("file_paths", {})
    input_dirs = {}
    
    for category, key in [("PDFs", "pdfs"), ("Images", "images"), 
                          ("EPUBs", "epubs"), ("Auto", "auto")]:
        if category in file_paths:
            input_path = file_paths[category].get("input")
            if input_path:
                input_dirs[key] = Path(input_path)
    
    return input_dirs


def get_logs_directory(paths_config: dict, create: bool = True) -> Optional[Path]:
    """Get logs directory from configuration.
    
    Args:
        paths_config: Paths configuration dictionary.
        create: If True, create directory if it doesn't exist.
        
    Returns:
        Path to logs directory, or None if not configured.
    """
    logs_dir = paths_config.get("general", {}).get("logs_dir")
    if logs_dir:
        return ensure_directory(Path(logs_dir), create=create)
    return None


def collect_scan_directories(paths_config: dict) -> List[Path]:
    """Collect all unique input and output directories for scanning.
    
    Useful for batch checking and repair operations that need to scan
    multiple directories for artifacts.
    
    Args:
        paths_config: Paths configuration dictionary.
        
    Returns:
        List of unique directory paths.
    """
    file_paths = paths_config.get("file_paths", {})
    scan_dirs = set()
    
    for category in ["PDFs", "Images", "EPUBs", "Auto"]:
        if category in file_paths:
            for key in ["input", "output"]:
                path_str = file_paths[category].get(key)
                if path_str:
                    dir_path = Path(path_str)
                    if dir_path.exists() or key == "output":
                        # Create output dirs, but only add input dirs if they exist
                        scan_dirs.add(ensure_directory(dir_path, create=(key == "output")))
    
    return sorted(scan_dirs)
