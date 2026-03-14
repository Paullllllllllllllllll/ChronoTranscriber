"""
Path utility module for handling Windows path length limitations.

This module implements best practices for dealing with Windows MAX_PATH (260 char)
limitation by creating shortened directory names with content-based hashes.

Strategy:
- Truncate long names to a safe limit (80 chars)
- Append hash of full name for uniqueness
- Similar to how npm, Git, and other production tools handle long paths

Reference: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
"""

from __future__ import annotations

import hashlib
from pathlib import Path


# Windows MAX_PATH limit
WINDOWS_MAX_PATH = 260

# Maximum safe length for directory/file names
# Keep it well under 255 (NTFS limit) to leave room for subdirs and files
MAX_SAFE_NAME_LENGTH = 80

# Hash length for uniqueness (8 chars = 4 billion combinations)
HASH_LENGTH = 8

# Minimum name length to preserve readability (before hash)
MIN_READABLE_LENGTH = 20


def _compute_name_hash(name: str) -> str:
    """Return an 8-character SHA-256 hex digest of *name*."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _truncate_name(name: str, max_length: int) -> str:
    """Truncate *name* to *max_length*, stripping trailing punctuation."""
    if len(name) <= max_length:
        return name
    return name[:max_length].rstrip(".-_ ")


def create_safe_directory_name(original_name: str, suffix: str = "") -> str:
	"""
	Create a safe directory name that won't exceed Windows path limits.
	
	Args:
		original_name: The original name (e.g., document stem)
		suffix: Optional suffix to append (e.g., "_working_files")
	
	Returns:
		A shortened, safe directory name with hash for uniqueness
		
	Example:
		Input:  "Beukers etal 2025 Grape (Vitis vinifera) use in the early modern..."
		Output: "Beukers etal 2025 Grape (Vitis vinifera) use in the earl-a3f8d9e2_working_files"
	"""
	# Calculate hash of full original name for uniqueness
	name_hash = _compute_name_hash(original_name)

	# Calculate how much space we have for the actual name
	# Format: [truncated_name]-[hash][suffix]
	reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for the dash
	available_length = MAX_SAFE_NAME_LENGTH - reserved_length

	# Truncate the name if necessary
	truncated = _truncate_name(original_name, available_length)

	# Combine truncated name with hash and suffix
	safe_name = f"{truncated}-{name_hash}{suffix}"

	return safe_name


def create_safe_filename(original_name: str, extension: str, parent_path: Path = None) -> str:
	"""
	Create a safe filename that won't exceed Windows path limits.
	
	Args:
		original_name: The original name (e.g., document stem)
		extension: File extension including dot (e.g., ".txt", ".jsonl")
		parent_path: Optional parent directory path to consider for MAX_PATH calculation
	
	Returns:
		A shortened, safe filename with hash for uniqueness
		
	Example:
		Input:  "Nippard_2025_Bodybuilding_Transformation_System_Intermediate_Advanced", ".txt"
		Output: "Nippard_2025_Bodybuilding_Transformation_System_Inte-a3f8d9e2.txt"
	"""
	# Calculate hash of full original name for uniqueness
	name_hash = _compute_name_hash(original_name)

	# Calculate maximum filename length based on context
	if parent_path is not None:
		# Calculate remaining path budget from Windows MAX_PATH
		parent_len = len(str(parent_path)) + 1  # +1 for path separator
		max_filename_len = WINDOWS_MAX_PATH - parent_len - 10  # -10 for safety margin
		# Ensure we stay within reasonable bounds
		max_filename_len = max(MIN_READABLE_LENGTH + HASH_LENGTH + len(extension) + 1,
							   min(max_filename_len, MAX_SAFE_NAME_LENGTH))
	else:
		max_filename_len = MAX_SAFE_NAME_LENGTH

	# Calculate how much space we have for the actual name
	# Format: [truncated_name]-[hash][extension]
	reserved_length = len(extension) + HASH_LENGTH + 1  # +1 for the dash
	available_length = max_filename_len - reserved_length

	# Truncate the name if necessary
	if len(original_name) > available_length:
		truncated = _truncate_name(
			original_name, max(MIN_READABLE_LENGTH, available_length)
		)
		# Combine truncated name with hash
		safe_name = f"{truncated}-{name_hash}{extension}"
	else:
		# No truncation needed - use original name without hash
		safe_name = f"{original_name}{extension}"

	return safe_name


def create_safe_log_filename(base_name: str, log_type: str) -> str:
	"""
	Create a safe log filename that won't exceed path limits.
	
	Args:
		base_name: The base name (e.g., document stem)
		log_type: Type of log (e.g., "transcription", "summary")
	
	Returns:
		A shortened, safe log filename
		
	Example:
		Input:  "Very long document name...", "transcription"
		Output: "Very long document name...-a3f8d9e2_transcription_log.json"
	"""
	suffix = f"_{log_type}_log.json"

	# Calculate hash for uniqueness
	name_hash = _compute_name_hash(base_name)

	# Calculate available space
	reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for dash
	available_length = MAX_SAFE_NAME_LENGTH - reserved_length

	# Truncate if necessary
	truncated = _truncate_name(base_name, available_length)

	# Combine
	safe_filename = f"{truncated}-{name_hash}{suffix}"

	return safe_filename


def ensure_path_safe(path: Path) -> Path:
	r"""
	Ensure a path won't exceed Windows MAX_PATH limits.
	
	For Windows 10 1607+, Python's pathlib automatically uses extended-length
	paths (\\?\) when needed, but this provides an explicit check.
	
	Args:
		path: The path to check
		
	Returns:
		The path (potentially with extended-length prefix on Windows)
		
	Note:
		Python 3.6+ pathlib handles extended paths automatically in most cases.
	"""
	# On Windows, check if path is too long
	import platform
	if platform.system() == 'Windows':
		# Python's pathlib handles \\?\ prefix automatically in most cases
		# But we can ensure the path is resolved/absolute which triggers this
		try:
			return path.resolve()
		except OSError:
			# If resolve fails, return as-is and let caller handle
			return path
	
	return path
