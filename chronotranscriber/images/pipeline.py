"""Tesseract preprocessing: grayscale, polarity, deskew, denoise, binarization.

:class:`TesseractSettings` holds what a run's Tesseract preprocessing needs;
:func:`tesseract_settings` builds it from the ``--ocr-preprocess`` preset and
the image-processing concurrency of the settings file.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from chronotranscriber.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronotranscriber.output.paths import natural_sort_key

__all__ = [
    "OCR_PRESETS",
    "TESSERACT_PREPROCESSING",
    "ImageProcessor",
    "TesseractSettings",
    "tesseract_settings",
]

logger = logging.getLogger(__name__)

TESSERACT_PREPROCESSING: Mapping[str, Any] = MappingProxyType(
    {
        "flatten_alpha": True,
        "grayscale": True,
        "invert_to_dark_on_light": "auto",
        "deskew": True,
        "denoise": "median",
        "binarization": "otsu",
        "sauvola_window": 25,
        "sauvola_k": 0.2,
        "morphology": "none",
        "morph_kernel": 3,
        "border_px": 10,
        "output_format": "png",
        "preserve_resolution": True,
        "embed_dpi_metadata": True,
    }
)
"""The default Tesseract preprocessing steps."""

OCR_PRESETS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "default": {},
        "sauvola": {"binarization": "sauvola"},
        "none": {
            "invert_to_dark_on_light": "never",
            "deskew": False,
            "denoise": "none",
        },
    }
)
"""The steps each ``--ocr-preprocess`` preset changes."""


@dataclass(frozen=True)
class TesseractSettings:
    """Rendering and preprocessing of the pages Tesseract reads.

    *max_pixels* bounds a rendered PDF page (0 disables the guard);
    *concurrency* is the number of pages preprocessed at once.
    """

    preprocessing: Mapping[str, Any] = field(
        default_factory=lambda: dict(TESSERACT_PREPROCESSING)
    )
    target_dpi: int = 300
    max_pixels: int = 24_000_000
    concurrency: int = 4

    @property
    def output_format(self) -> str:
        """``png`` or ``tiff``, the lossless format of the preprocessed pages."""
        return str(self.preprocessing.get("output_format", "png")).lower()

    @property
    def suffix(self) -> str:
        """The file suffix of the preprocessed pages."""
        return ".png" if self.output_format == "png" else ".tif"

    @property
    def embed_dpi(self) -> bool:
        """Whether the preprocessed pages record their resolution."""
        return bool(self.preprocessing.get("embed_dpi_metadata", True))


def tesseract_settings(
    preset: str = "default", concurrency: int = 4
) -> TesseractSettings:
    """Return the Tesseract settings of the ``--ocr-preprocess`` *preset*."""
    steps = {**TESSERACT_PREPROCESSING, **OCR_PRESETS[preset]}
    return TesseractSettings(preprocessing=steps, concurrency=max(1, concurrency))


class ImageProcessor:
    """Static Tesseract preprocessing toolbox."""

    @staticmethod
    def _pil_to_np(image: Image.Image) -> np.ndarray:
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            # Converting to RGBA first makes the mask the alpha channel; for
            # mode "P", split()[-1] would be the palette index.
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            image = background
        if image.mode == "RGB":
            arr = np.array(image)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif image.mode in ("L", "I;16"):
            return np.array(image)
        else:
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _auto_invert_if_needed(gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        border = 10
        border = min(border, h // 4, w // 4) if h > 0 and w > 0 else 0
        if border > 0:
            top = gray[:border, :]
            bottom = gray[-border:, :]
            left = gray[:, :border]
            right = gray[:, -border:]
            bg_mean = np.mean(
                np.concatenate(
                    [top.flatten(), bottom.flatten(), left.flatten(), right.flatten()]
                )
            )
        else:
            bg_mean = np.mean(gray)
        if bg_mean < 127:
            return cv2.bitwise_not(gray)
        return gray

    # Long-edge cap (px) for skew estimation, whose cost grows with pixel
    # count; the rotation itself runs at full resolution.
    _DESKEW_ESTIMATE_MAX_SIDE = 1500

    @staticmethod
    def _estimate_skew_angle(gray: np.ndarray) -> float | None:
        """Estimate the page skew angle, downsampling large inputs first.

        The angle is estimated on a copy whose longest side is capped at
        ``_DESKEW_ESTIMATE_MAX_SIDE``; images already within the cap are used
        as-is. Returns ``None`` when no skew could be determined.
        """
        # Imported here: deskew (with scikit-image) takes a quarter second to
        # import, and only Tesseract runs need it.
        from deskew import determine_skew

        h, w = gray.shape[:2]
        longest = max(h, w)
        if longest > ImageProcessor._DESKEW_ESTIMATE_MAX_SIDE:
            scale = ImageProcessor._DESKEW_ESTIMATE_MAX_SIDE / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small = gray
        angle = determine_skew(small)
        return None if angle is None else float(angle)

    @staticmethod
    def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
        try:
            angle = ImageProcessor._estimate_skew_angle(gray)
            if angle is None:
                return gray, 0.0
            angle_deg = float(angle)
            if abs(angle_deg) < 0.1:
                return gray, angle_deg
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rotated = cv2.warpAffine(
                gray,
                m,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            return rotated, angle_deg
        except Exception:  # noqa: BLE001 - deskewing is optional; keep the page as is
            return gray, 0.0

    @staticmethod
    def _denoise(gray: np.ndarray, method: str) -> np.ndarray:
        m = (method or "none").lower()
        if m == "median":
            return cv2.medianBlur(gray, 3)
        if m == "bilateral":
            return cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    @staticmethod
    def _binarize(gray: np.ndarray, method: str, window: int, k: float) -> np.ndarray:
        m = (method or "otsu").lower()
        if m == "sauvola":
            from skimage.filters import threshold_sauvola

            w = window if isinstance(window, int) and window % 2 == 1 else 25
            thresh = threshold_sauvola(  # type: ignore[no-untyped-call]
                gray, window_size=w, k=float(k) if k is not None else 0.2
            )
            binary = (gray > thresh).astype(np.uint8) * 255
            return np.asarray(binary, dtype=np.uint8)
        if m in ("adaptive", "adaptive_otsu", "adaptive-gaussian"):
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
            )
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _morph(binary: np.ndarray, op: str, ksize: int) -> np.ndarray:
        opn = (op or "none").lower()
        if opn == "none":
            return binary
        k = max(1, int(ksize))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        if opn == "open":
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if opn == "close":
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if opn == "erode":
            return cv2.erode(binary, kernel, iterations=1)
        if opn == "dilate":
            return cv2.dilate(binary, kernel, iterations=1)
        return binary

    @staticmethod
    def preprocess_for_tesseract(
        image: Image.Image, cfg: Mapping[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Preprocess a PIL image for Tesseract according to *cfg*.

        The steps run in order: alpha flattening, grayscale, polarity, deskew,
        denoise, binarization, morphology and a white border. Returns the
        ``L``-mode image of 0/255 values and a diagnostics dict.
        """
        diag: dict[str, Any] = {}
        np_img = ImageProcessor._pil_to_np(image)
        gray = ImageProcessor._ensure_grayscale(np_img)
        invert_mode = str(cfg.get("invert_to_dark_on_light", "auto")).lower()
        if invert_mode == "always":
            gray = cv2.bitwise_not(gray)
            diag["inverted"] = True
        elif invert_mode == "auto":
            before_mean = float(np.mean(gray)) if gray.size else 0.0
            gray = ImageProcessor._auto_invert_if_needed(gray)
            after_mean = float(np.mean(gray)) if gray.size else 0.0
            diag["inverted"] = after_mean > before_mean
        else:
            diag["inverted"] = False
        if bool(cfg.get("deskew", True)):
            gray, angle = ImageProcessor._deskew(gray)
            diag["skew_angle"] = float(angle)
        else:
            diag["skew_angle"] = 0.0
        gray = ImageProcessor._denoise(gray, cfg.get("denoise", "median"))
        binary = ImageProcessor._binarize(
            gray,
            cfg.get("binarization", "otsu"),
            int(cfg.get("sauvola_window", 25)),
            float(cfg.get("sauvola_k", 0.2)),
        )
        diag["binarization"] = cfg.get("binarization", "otsu")
        binary = ImageProcessor._morph(
            binary, cfg.get("morphology", "none"), int(cfg.get("morph_kernel", 3))
        )
        diag["morphology"] = cfg.get("morphology", "none")
        b = int(cfg.get("border_px", 10))
        if b and b > 0:
            binary = cv2.copyMakeBorder(
                binary, b, b, b, b, cv2.BORDER_CONSTANT, value=255
            )
        pil_bin = Image.fromarray(binary)
        if pil_bin.mode != "L":
            pil_bin = pil_bin.convert("L")
        return pil_bin, diag

    @staticmethod
    def source_images(source_folder: Path) -> list[Path]:
        """Return the supported images of *source_folder* in natural order."""
        image_files: list[Path] = [
            p
            for p in source_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        image_files.sort(key=lambda p: natural_sort_key(p.name))
        return image_files

    @staticmethod
    def tesseract_image_names(
        source_folder: Path, settings: TesseractSettings | None = None
    ) -> list[str]:
        """Return the preprocessed file name of every image, in page order.

        Source files sharing a stem across extensions (``scan_001.png`` and
        ``scan_001.tif``) would map to one name, so only the colliding stems
        keep their extension in the name. Collisions are detected over the
        full listing, so names stay stable under page ranges.
        """
        tesseract = settings if settings is not None else TesseractSettings()
        image_files = ImageProcessor.source_images(source_folder)
        stem_counts = Counter(p.stem for p in image_files)
        dup_stems = {stem for stem, n in stem_counts.items() if n > 1}
        return [
            f"{img.name if img.stem in dup_stems else img.stem}"
            f"_tess_preprocessed{tesseract.suffix}"
            for img in image_files
        ]

    @staticmethod
    def process_and_save_images_for_tesseract(
        source_folder: Path,
        preprocessed_folder: Path,
        page_indices: list[int] | None = None,
        *,
        settings: TesseractSettings | None = None,
    ) -> tuple[list[Path], dict[str, int]]:
        """Preprocess the images of *source_folder* for Tesseract, losslessly.

        Args:
            source_folder: Path to the source folder containing original images.
            preprocessed_folder: Path to save processed images.
            page_indices: Optional 0-based indices into the sorted image list.
                If None, all images are processed.
            settings: The run's Tesseract settings; the defaults without.

        Returns:
            A tuple of (successfully written output paths, order map). The order
            map is ``{output_name: absolute_index}`` where the absolute index is
            each file's position in the full sorted source listing (before the
            page-range filter), the unit index of the image.
        """
        tesseract = settings if settings is not None else TesseractSettings()
        image_files = ImageProcessor.source_images(source_folder)
        if not image_files:
            return [], {}
        names = ImageProcessor.tesseract_image_names(source_folder, tesseract)

        if page_indices is not None:
            selected = [i for i in page_indices if 0 <= i < len(image_files)]
        else:
            selected = list(range(len(image_files)))

        preprocessed_folder.mkdir(parents=True, exist_ok=True)

        out_paths: list[Path] = []
        order_map: dict[str, int] = {}
        for abs_idx in selected:
            out_path = preprocessed_folder / names[abs_idx]
            out_paths.append(out_path)
            order_map[out_path.name] = abs_idx

        with ThreadPoolExecutor(max_workers=tesseract.concurrency) as pool:
            results = list(
                pool.map(
                    ImageProcessor._tesseract_preprocess_image_task,
                    [image_files[abs_idx] for abs_idx in selected],
                    out_paths,
                    [tesseract] * len(out_paths),
                )
            )

        written = [
            p for p, res in zip(out_paths, results, strict=True) if res is not None
        ]
        return written, order_map

    @staticmethod
    def _tesseract_preprocess_image_task(
        img_path: Path, out_path: Path, settings: TesseractSettings
    ) -> str | None:
        """Worker: open, preprocess with Tesseract pipeline, and save losslessly."""
        try:
            with Image.open(img_path) as im:
                # Honor the EXIF orientation; exif_transpose copies the whole
                # image, so it runs only when there is an orientation to apply.
                if im.getexif().get(0x0112, 1) != 1:
                    oriented = ImageOps.exif_transpose(im) or im
                else:
                    oriented = im
                processed_img, diag = ImageProcessor.preprocess_for_tesseract(
                    oriented, settings.preprocessing
                )
                if settings.embed_dpi:
                    dpi = settings.target_dpi
                    processed_img.save(out_path, dpi=(dpi, dpi))
                else:
                    processed_img.save(out_path)
                logger.debug(
                    "Tesseract-preprocessed %s -> %s diag=%s",
                    img_path.name,
                    out_path.name,
                    diag,
                )
                return str(out_path)
        except Exception as e:  # noqa: BLE001 - a failed image is logged and skipped
            logger.error(f"Error Tesseract-preprocessing image {img_path.name}: {e}")
            return None
