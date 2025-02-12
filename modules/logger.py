# modules/logger.py
import logging
from pathlib import Path
from modules.config_loader import ConfigLoader
from typing import Any


def setup_logger(name: str) -> logging.Logger:
	"""
    Set up and return a logger with file and console handlers.

    Parameters:
        name (str): Name of the logger.

    Returns:
        logging.Logger: The configured logger.
    """
	config_loader = ConfigLoader()
	config_loader.load_configs()
	paths_config = config_loader.get_paths_config()
	logs_dir = Path(paths_config.get('general', {}).get('logs_dir', 'logs'))
	logs_dir.mkdir(parents=True, exist_ok=True)
	log_file = logs_dir / "application.log"
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		file_handler = logging.FileHandler(log_file, encoding='utf-8')
		formatter = logging.Formatter(
			'%(asctime)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
		# Add a console handler that only outputs warnings and errors.
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.WARNING)
		console_formatter = logging.Formatter('%(levelname)s: %(message)s')
		console_handler.setFormatter(console_formatter)
		logger.addHandler(console_handler)
	return logger
