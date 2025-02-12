# modules/utils.py

import re


def extract_page_number_from_filename(filename: str) -> int:
	"""
    Extract the page number from a filename using a standard pattern.

    Parameters:
        filename (str): The filename from which to extract the page number.

    Returns:
        int: The extracted page number if found; otherwise, a placeholder value.
    """
	match = re.search(r'_page_(\d+)', filename)
	if match:
		return int(match.group(1))
	else:
		return 9999999  # Placeholder for unexpected filenames
