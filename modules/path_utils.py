# modules/path_utils.py
import sys
from pathlib import Path
from typing import Dict, Any

from modules.utils import console_print


def validate_paths(paths_config: Dict[str, Any]) -> None:
	"""
	Validate path configurations based on the allow_relative_paths setting.

	If allow_relative_paths is enabled, paths should already be resolved.
	Otherwise, verify that all paths are absolute.

	Parameters:
		paths_config (Dict[str, Any]): The loaded paths configuration.
	"""
	general = paths_config.get("general", {})
	allow_relative_paths = general.get("allow_relative_paths", False)

	# Skip validation if using relative paths - they should have been resolved by ConfigLoader
	if allow_relative_paths:
		return

	error_found = False

	# Validate general logs_dir
	logs_dir = general.get("logs_dir")
	if logs_dir and not Path(logs_dir).is_absolute():
		console_print(
			f"[ERROR] The 'logs_dir' path '{logs_dir}' is not absolute. Please use an absolute path or enable allow_relative_paths in paths_config.yaml.")
		error_found = True

	# Validate transcription_prompt_path if present
	prompt_path = general.get("transcription_prompt_path")
	if prompt_path and not Path(prompt_path).is_absolute():
		console_print(
			f"[ERROR] The 'transcription_prompt_path' path '{prompt_path}' is not absolute. Please use an absolute path or enable allow_relative_paths in paths_config.yaml.")
		error_found = True

	# Validate file paths for PDFs and Images
	file_paths = paths_config.get("file_paths", {})
	for category in ["PDFs", "Images"]:
		if category in file_paths:
			for path_key in ["input", "output"]:
				path_value = file_paths[category].get(path_key)
				if path_value and not Path(path_value).is_absolute():
					console_print(
						f"[ERROR] The {path_key} path for {category} ('{path_value}') is not absolute. Please use absolute paths or enable allow_relative_paths in paths_config.yaml.")
					error_found = True

	if error_found:
		sys.exit(1)