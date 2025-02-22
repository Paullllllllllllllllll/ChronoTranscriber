# modules/image_utils.py
import shutil
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
from typing import List, Optional, Tuple

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.multiprocessing_utils import run_multiprocessing_tasks

logger = setup_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp',
                              '.gif', '.webp'}


class ImageProcessor:
	def __init__(self, image_path: Path) -> None:
		if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
			logger.error(f"Unsupported image format: {image_path.suffix}")
			raise ValueError(f"Unsupported image format: {image_path.suffix}")
		self.image_path = image_path

		config_loader = ConfigLoader()
		config_loader.load_configs()
		self.image_config = config_loader.get_image_processing_config()

	def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
		"""
		Convert the image to grayscale if enabled.
		Returns:
			Image.Image: The grayscale image.
		"""
		if self.image_config.get('grayscale_conversion', True):
			return ImageOps.grayscale(image)
		return image

	def remove_borders(self, image: Image.Image) -> Image.Image:
		"""
		Remove borders from the image based on configuration.
		Returns:
			Image.Image: The cropped image.
		"""
		border_removal = self.image_config.get('border_removal', {})
		if not border_removal.get('enabled', True):
			return image

		border_color = border_removal.get('border_color', 255)
		tolerance = border_removal.get('tolerance', 15)
		min_border = border_removal.get('min_border', 20)
		padding = 20

		img_np = np.array(image)
		mask = img_np < (border_color - tolerance)

		rows = np.any(mask, axis=1)
		cols = np.any(mask, axis=0)
		if not rows.any() or not cols.any():
			return image

		top = max(np.argmax(rows) - padding, min_border)
		bottom = min(len(rows) - np.argmax(rows[::-1]) + padding,
		             len(rows) - min_border)
		left = max(np.argmax(cols) - padding, min_border)
		right = min(len(cols) - np.argmax(cols[::-1]) + padding,
		            len(cols) - min_border)

		if top >= bottom or left >= right:
			return image

		return image.crop((left, top, right, bottom))

	def handle_transparency(self, image: Image.Image) -> Image.Image:
		"""
		Handle transparency by pasting the image onto a white background.
		Returns:
			Image.Image: The processed image.
		"""
		if self.image_config.get('handle_transparency', True):
			if image.mode in ('RGBA', 'LA') or (
					image.mode == 'P' and 'transparency' in image.info):
				background = Image.new("RGB", image.size, (255, 255, 255))
				background.paste(image, mask=image.split()[-1])
				return background
		return image

	def adjust_dpi(self, image: Image.Image) -> Image.Image:
		"""
		Adjust the DPI of the image based on configuration.
		Returns:
			Image.Image: The resized image with updated DPI.
		"""
		adjust_dpi = self.image_config.get('adjust_dpi', {})
		target_dpi = self.image_config.get('image_processing', {}).get(
			'target_dpi', 300)
		min_pixels = adjust_dpi.get('min_pixels', 1500)
		max_pixels = adjust_dpi.get('max_pixels', 3000)

		current_dpi = image.info.get('dpi', (300, 300))[0]
		width_in = image.width / current_dpi
		height_in = image.height / current_dpi
		new_width = int(width_in * target_dpi)
		new_height = int(height_in * target_dpi)

		scale_factor = 1.0
		if new_width < min_pixels or new_height < min_pixels:
			scale_factor = max(min_pixels / new_width, min_pixels / new_height)
		elif new_width > max_pixels or new_height > max_pixels:
			scale_factor = min(max_pixels / new_width, max_pixels / new_height)

		if scale_factor != 1.0:
			new_width = int(new_width * scale_factor)
			new_height = int(new_height * scale_factor)
			image = image.resize((new_width, new_height),
			                     Image.Resampling.LANCZOS)
			logger.info(
				f"Resized image to {new_width}x{new_height} pixels for optimal DPI.")

		image.info['dpi'] = (target_dpi, target_dpi)
		return image

	def process_image(self, output_path: Path) -> str:
		"""
		Process the image and save it to the given output path.
		Returns:
			str: A message indicating the outcome.
		"""
		try:
			with Image.open(self.image_path) as img:
				img = self.handle_transparency(img)
				img = self.convert_to_grayscale(img)
				img = self.remove_borders(img)
				img = self.adjust_dpi(img)
				img.save(
					output_path,
					dpi=(
						self.image_config.get('image_processing', {}).get(
							'target_dpi', 300),
						self.image_config.get('image_processing', {}).get(
							'target_dpi', 300)
					)
				)
			return f"Processed and saved: {output_path.name}"
		except Exception as e:
			logger.error(f"Error processing image {self.image_path.name}: {e}")
			return f"Failed to process {self.image_path.name}: {e}"

	# --- Static Methods for Folder-Level Processing ---

	@staticmethod
	def prepare_image_folder(folder: Path, image_output_dir: Path) -> Tuple[
		Path, Path, Path, Path, Path]:
		"""
		Prepares the output directories for processing an image folder.

		Returns a tuple of:
		  - parent_folder: The directory for outputs related to this folder.
		  - raw_images_folder: Where raw images will be stored.
		  - preprocessed_folder: Where preprocessed images will be stored.
		  - temp_jsonl_path: File for recording transcription logs.
		  - output_txt_path: Final transcription text file.
		"""
		parent_folder = image_output_dir / folder.name
		parent_folder.mkdir(parents=True, exist_ok=True)
		raw_images_folder = parent_folder / "raw_images"
		raw_images_folder.mkdir(exist_ok=True)
		preprocessed_folder = parent_folder / "preprocessed_images"
		preprocessed_folder.mkdir(exist_ok=True)
		temp_jsonl_path = parent_folder / f"{folder.name}_transcription.jsonl"
		if not temp_jsonl_path.exists():
			temp_jsonl_path.touch()
		output_txt_path = parent_folder / f"{folder.name}_transcription.txt"
		return parent_folder, raw_images_folder, preprocessed_folder, temp_jsonl_path, output_txt_path

	@staticmethod
	def copy_images_to_raw(source_folder: Path, raw_images_folder: Path) -> \
	List[Path]:
		"""
		Copies image files from the source folder to the designated raw images folder.

		Returns a list of copied image paths.
		"""
		image_files: List[Path] = []
		for ext in SUPPORTED_IMAGE_EXTENSIONS:
			image_files.extend(list(source_folder.glob(f'*{ext}')))
		for file in image_files:
			try:
				shutil.copy(file, raw_images_folder / file.name)
			except Exception as e:
				logger.exception(
					f"Error copying file {file} to raw_images_folder: {e}")
		return list(raw_images_folder.glob("*"))

	@staticmethod
	def process_images_multiprocessing(image_paths: List[Path],
	                                   output_paths: List[Path]) -> List[
		Optional[str]]:
		"""
		Process images using multiprocessing.

		Returns:
			List[Optional[str]]: Results from the image processing tasks.
		"""
		args_list = list(zip(image_paths, output_paths))
		results = run_multiprocessing_tasks(ImageProcessor._process_image_task,
		                                    args_list, processes=12)
		return results

	@staticmethod
	def _process_image_task(img_path: Path, out_path: Path) -> str:
		"""
		Process a single image: creates an ImageProcessor instance for the image
		and processes it, saving the output to out_path.
		"""
		processor = ImageProcessor(img_path)
		return processor.process_image(out_path)

