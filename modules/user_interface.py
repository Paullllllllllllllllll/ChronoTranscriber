# modules/user_interface.py
from pathlib import Path
from typing import List
from modules.logger import setup_logger
from modules.utils import console_print, check_exit, safe_input

logger = setup_logger(__name__)


def select_folders(directory: Path) -> List[Path]:
	"""
    List subdirectories in the given directory and allow the user to select folders.

    Parameters:
        directory (Path): The directory to scan.

    Returns:
        List[Path]: The list of selected folder paths.
    """
	folders = [d for d in directory.iterdir() if d.is_dir()]
	if not folders:
		print(f"No folders found in {directory}.")
		logger.info(f"No folders found in {directory}.")
		return []

	print(f"Folders found in {directory}:")
	for idx, folder in enumerate(folders, 1):
		print(f"{idx}. {folder.name}")

	selected = input(
		"Enter the numbers of the folders to select, separated by commas (or type 'q' to exit): ").strip()
	if selected.lower() in ["q", "exit"]:
		print("Exiting.")
		exit(0)
	try:
		indices = [int(i.strip()) - 1 for i in selected.split(',') if
		           i.strip().isdigit()]
		selected_folders = [folders[i] for i in indices if
		                    0 <= i < len(folders)]
		if not selected_folders:
			print("No valid folders selected.")
			logger.info("No valid folders selected by the user.")
		return selected_folders
	except ValueError:
		print("Invalid input. Please enter numbers separated by commas.")
		logger.error("User entered invalid folder selection input.")
		return []


def select_files(directory: Path, extension: str) -> List[Path]:
	"""
    List files with a given extension and allow the user to select files.

    Parameters:
        directory (Path): The directory to scan.
        extension (str): The file extension to filter by.

    Returns:
        List[Path]: The list of selected file paths.
    """
	files = [f for f in directory.iterdir() if
	         f.is_file() and f.suffix.lower() == extension.lower()]
	if not files:
		print(f"No files with extension '{extension}' found in {directory}.")
		logger.info(
			f"No files with extension '{extension}' found in {directory}.")
		return []

	print(f"Files with extension '{extension}' found in {directory}:")
	for idx, file in enumerate(files, 1):
		print(f"{idx}. {file.name}")

	selected = input(
		"Enter the numbers of the files to select, separated by commas (or type 'q' to exit): ").strip()
	if selected.lower() in ["q", "exit"]:
		print("Exiting.")
		exit(0)
	try:
		indices = [int(i.strip()) - 1 for i in selected.split(',') if
		           i.strip().isdigit()]
		selected_files = [files[i] for i in indices if 0 <= i < len(files)]
		if not selected_files:
			print("No valid files selected.")
			logger.info("No valid files selected by the user.")
		return selected_files
	except ValueError:
		print("Invalid input. Please enter numbers separated by commas.")
		logger.error("User entered invalid file selection input.")
		return []

def select_option(prompt: str, options: List[str]) -> str:
    """
    Display a prompt and a numbered list of options, then return the user's choice as a string.
    """
    console_print(prompt)
    for idx, option in enumerate(options, 1):
        console_print(f"{idx}. {option}")
    choice = safe_input("Enter the number of your choice (or type 'q' to exit): ")
    check_exit(choice)
    return choice.strip()
