# modules/image_utils.py

from __future__ import annotations

import io
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from deskew import determine_skew
from PIL import Image, ImageOps
from skimage.filters import threshold_sauvola

from modules.config.constants import SUPPORTED_IMAGE_EXTENSIONS
from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.infra.multiprocessing_utils import run_multiprocessing_tasks
from modules.infra.paths import (
    create_safe_directory_name,
    create_safe_filename,
    natural_sort_key,
)

logger = setup_logger(__name__)

IMAGE_FAILURE_RATE_THRESHOLD = 0.5


class ImageProcessor:
    """Static image-preprocessing toolbox.

    Provider-bound preprocessing happens fully in memory via
    :meth:`process_pil` (used by the streaming pipeline in
    ``modules.images.page_stream``); the remaining methods cover folder
    preparation, the JP2 FFmpeg fallback, and the Tesseract pipeline.
    """

    @staticmethod
    def _cap_longest_side(image: Image.Image, max_side: int) -> Image.Image:
        """Downscale *image* so its longest side does not exceed *max_side*.

        Returns the image unchanged when it already fits; otherwise resizes
        with LANCZOS resampling, preserving aspect ratio.
        """
        w, h = image.size
        longest = max(w, h)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def resize_for_detail(
        image: Image.Image,
        detail: str,
        img_cfg: dict[str, Any],
        model_type: str = "openai",
    ) -> Image.Image:
        """
        Resize strategy based on desired LLM detail and model type.
        - low: downscale longest side to low_max_side_px.
        - high: fit/pad into high_target_box [width, height] (OpenAI/Google) or
                cap longest side to high_max_side_px (Anthropic).
        - auto: default to 'high' strategy.

        Args:
            image: PIL Image to resize
            detail: Detail level ('low', 'high', 'auto')
            img_cfg: Config dict for the provider
            model_type: 'openai', 'google', or 'anthropic' for
                provider-specific resizing
        """
        # Normalize flags and defaults
        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return image
        detail_norm = (detail or "high").lower()
        if detail_norm not in (
            "low",
            "high",
            "auto",
            "medium",
            "ultra_high",
            "original",
        ):
            detail_norm = "high"

        # Low detail: cap longest side (same for all providers)
        if detail_norm == "low":
            max_side = int(img_cfg.get("low_max_side_px", 512))
            return ImageProcessor._cap_longest_side(image, max_side)

        # Original detail (GPT-5.4+): cap to max side / max pixels, no padding
        if detail_norm == "original":
            max_side = int(img_cfg.get("original_max_side_px", 6000))
            max_pixels = int(img_cfg.get("original_max_pixels", 10240000))
            w, h = image.size
            # Cap longest side
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                w = max(1, int(w * scale))
                h = max(1, int(h * scale))
                image = image.resize((w, h), Image.Resampling.LANCZOS)
            # Cap total pixel budget
            if w * h > max_pixels:
                scale = (max_pixels / float(w * h)) ** 0.5
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            return image

        # High/auto/medium/ultra_high: provider-specific strategy
        if model_type == "anthropic":
            # Anthropic: cap longest side to high_max_side_px (no padding)
            max_side = int(img_cfg.get("high_max_side_px", 1568))
            return ImageProcessor._cap_longest_side(image, max_side)
        else:
            # OpenAI/Google: fit into box and pad with white
            box = img_cfg.get("high_target_box", [768, 1536])
            try:
                target_width = int(box[0])
                target_height = int(box[1])
            except Exception:
                target_width, target_height = 768, 1536
            orig_width, orig_height = image.size
            scale_w = target_width / orig_width
            scale_h = target_height / orig_height
            scale = min(scale_w, scale_h)
            new_width = max(1, int(orig_width * scale))
            new_height = max(1, int(orig_height * scale))
            resized_img = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            if resized_img.mode not in ("RGB", "L"):
                resized_img = resized_img.convert("RGB")
            if resized_img.mode == "L":
                final_img = Image.new("L", (target_width, target_height), 255)
            else:
                final_img = Image.new(
                    "RGB", (target_width, target_height), (255, 255, 255)
                )
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_img.paste(resized_img, (paste_x, paste_y))
            return final_img

    @staticmethod
    def derive_render_zoom(
        page_width_pt: float,
        page_height_pt: float,
        target_dpi: int,
        max_pixels: int,
        img_cfg: dict[str, Any],
        model_type: str,
    ) -> float:
        """Render zoom (px/pt) for the "direct" LLM-payload render strategy.

        Given a PDF page's dimensions in points, compute the float zoom at which
        to rasterize the page so that the rendered pixels already match the
        dimensions the active :meth:`resize_for_detail` profile would produce
        from a ``target_dpi`` render. This avoids rendering pixels that the
        provider (or the local resize) would only discard again.

        The zoom never exceeds ``target_dpi / 72`` (quality ceiling; no
        render-upscaling, matching the existing no-upscale semantics), and the
        ``max_pixels`` per-page guard is applied last. When the active profile
        would not shrink the ``target_dpi`` render at all, the returned zoom is
        exactly ``target_dpi / 72`` so the payload stays byte-identical to the
        legacy ("supersample") render.
        """
        target_zoom = target_dpi / 72.0
        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return ImageProcessor._apply_max_pixels_zoom(
                page_width_pt, page_height_pt, target_zoom, max_pixels
            )

        w0 = page_width_pt * target_zoom
        h0 = page_height_pt * target_zoom
        if w0 <= 0 or h0 <= 0:
            return target_zoom
        longest = max(w0, h0)

        detail = ImageProcessor.resolve_detail(img_cfg, model_type).lower()
        if detail not in ("low", "high", "auto", "medium", "ultra_high", "original"):
            detail = "high"

        if detail == "low":
            max_side = int(img_cfg.get("low_max_side_px", 512))
            scale = min(1.0, max_side / longest)
        elif detail == "original":
            max_side = int(img_cfg.get("original_max_side_px", 6000))
            max_pix = int(img_cfg.get("original_max_pixels", 10240000))
            s1 = min(1.0, max_side / longest)
            w1 = w0 * s1
            h1 = h0 * s1
            s2 = math.sqrt(max_pix / (w1 * h1)) if w1 * h1 > max_pix else 1.0
            scale = s1 * s2
        elif model_type == "anthropic":
            max_side = int(img_cfg.get("high_max_side_px", 1568))
            scale = min(1.0, max_side / longest)
        else:
            # OpenAI/Google: fit into high_target_box. A box larger than the
            # target render would upscale; the ceiling forbids render-upscaling,
            # so cap the scale at 1.0 and let the resize_for_detail safety net
            # perform any box upscaling (byte-identical to supersample there).
            box = img_cfg.get("high_target_box", [768, 1536])
            try:
                box_w = int(box[0])
                box_h = int(box[1])
            except (TypeError, ValueError, IndexError):
                box_w, box_h = 768, 1536
            scale = min(1.0, box_w / w0, box_h / h0)

        render_zoom = target_zoom * scale
        return ImageProcessor._apply_max_pixels_zoom(
            page_width_pt, page_height_pt, render_zoom, max_pixels
        )

    @staticmethod
    def _apply_max_pixels_zoom(
        page_width_pt: float,
        page_height_pt: float,
        zoom: float,
        max_pixels: int,
    ) -> float:
        """Reduce *zoom* so the rendered page stays within *max_pixels*.

        Float analogue of ``modules.documents.pdf._get_effective_dpi`` operating
        on a zoom factor rather than an integer DPI. Returns *zoom* unchanged
        when the budget is disabled or already satisfied.
        """
        if max_pixels <= 0:
            return zoom
        pixels = (page_width_pt * zoom) * (page_height_pt * zoom)
        if pixels <= max_pixels:
            return zoom
        return zoom * math.sqrt(max_pixels / pixels)

    @staticmethod
    def resolve_detail(img_cfg: dict[str, Any], model_type: str) -> str:
        """Resolve the effective detail/resolution setting for a provider.

        Google uses ``media_resolution``, Anthropic keys off ``resize_profile``,
        and OpenAI-compatible providers use ``llm_detail``.
        """
        if model_type == "google":
            return str(img_cfg.get("media_resolution", "high") or "high")
        if model_type == "anthropic":
            return str(img_cfg.get("resize_profile", "auto") or "auto")
        return str(img_cfg.get("llm_detail", "high") or "high")

    @staticmethod
    def process_pil(
        img: Image.Image,
        img_cfg: dict[str, Any],
        model_type: str = "openai",
    ) -> tuple[bytes, int, int]:
        """Preprocess a PIL image fully in memory and return JPEG bytes.

        Applies the same transform chain as the path-based pipeline:
        transparency flattening, optional grayscale conversion,
        provider-specific resizing, mode coercion, and JPEG encoding at the
        configured quality.

        Args:
            img: Source PIL image.
            img_cfg: Provider-specific image-processing config section.
            model_type: 'openai', 'google', 'anthropic', or 'custom'.

        Returns:
            Tuple of (jpeg_bytes, width, height) of the encoded image.
        """
        detail = ImageProcessor.resolve_detail(img_cfg, model_type)

        if img_cfg.get("handle_transparency", True) and (
            img.mode in ("RGBA", "LA")
            or (img.mode == "P" and "transparency" in img.info)
        ):
            # Convert to RGBA first so the mask is the real alpha channel. For
            # mode "P", img.split()[-1] is the palette-index channel, not alpha,
            # which would flatten with the wrong mask (B16).
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            img = background

        if img_cfg.get("grayscale_conversion", True) and img.mode != "L":
            img = ImageOps.grayscale(img)

        img = ImageProcessor.resize_for_detail(img, detail, img_cfg, model_type)

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        jpeg_quality = int(img_cfg.get("jpeg_quality", 95))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality)
        width, height = img.size
        return buffer.getvalue(), width, height

    # --- Static Methods for Folder-Level Processing ---

    @staticmethod
    def prepare_image_folder(
        folder: Path,
        image_output_dir: Path,
        relative_key: str | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        """
        Prepares the output directories for processing an image folder.

        Returns a tuple of:
          - parent_folder: The directory for outputs related to this folder.
          - preprocessed_folder: Where preprocessed images would be stored
            (path only; created lazily by the Tesseract pipeline — the GPT
            pipeline preprocesses fully in memory).
          - temp_jsonl_path: File for recording transcription logs.
          - output_txt_path: Final transcription text file.
        """
        key = relative_key if relative_key is not None else folder.name
        safe_dir_name = create_safe_directory_name(key)

        # Create parent folder with safe directory name
        parent_folder = image_output_dir / safe_dir_name
        parent_folder.mkdir(parents=True, exist_ok=True)

        # Path of the (Tesseract-only) preprocessed images subfolder;
        # not created here.
        preprocessed_folder = parent_folder / "preprocessed_images"

        # Create safe filenames (truncated with hash if needed,
        # considering full path length).
        # No _transcription suffix to keep filenames shorter.
        temp_jsonl_name = create_safe_filename(folder.name, ".jsonl", parent_folder)
        output_txt_name = create_safe_filename(folder.name, ".txt", parent_folder)

        temp_jsonl_path = parent_folder / temp_jsonl_name
        if not temp_jsonl_path.exists():
            temp_jsonl_path.touch()
        output_txt_path = parent_folder / output_txt_name

        return parent_folder, preprocessed_folder, temp_jsonl_path, output_txt_path

    # ================== FFmpeg fallback decoding ==================

    @staticmethod
    def _decode_with_ffmpeg(image_path: Path) -> Image.Image:
        """Decode an image file via FFmpeg and return a PIL Image.

        Used as a fallback for JP2/J2K files that openjpeg cannot handle
        (e.g. sub-8-bit / bilevel codestreams).

        Args:
            image_path: Source image path.

        Returns:
            PIL Image loaded into memory.

        Raises:
            OSError: If FFmpeg returns a non-zero exit code.
            RuntimeError: If FFmpeg is not available on PATH.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(image_path),
                    "-frames:v",
                    "1",
                    str(tmp_path),
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                stderr_tail = result.stderr.decode(errors="replace")[-300:]
                raise OSError(
                    f"FFmpeg exited with code {result.returncode}: {stderr_tail}"
                )
            with Image.open(tmp_path) as img:
                img.load()
                return img.copy()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # ================== Tesseract-specific preprocessing ==================

    @staticmethod
    def _pil_to_np(image: Image.Image) -> np.ndarray:
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            # Convert to RGBA first so the flatten mask is the true alpha channel
            # (for mode "P", split()[-1] would be the palette index, not alpha) (B16).
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
        # Estimate background by sampling 10-px border
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
        # If background is dark (mean < 127), invert to make it light
        if bg_mean < 127:
            return cv2.bitwise_not(gray)
        return gray

    # Long-edge cap (px) for skew-angle estimation. determine_skew's
    # Canny+Hough cost scales with pixel count and dominates the Tesseract
    # preprocessing path (~900 ms/page at full resolution). The estimated
    # angle on a page downsampled to this cap matches the full-resolution
    # estimate to within a fraction of a degree, so we estimate the angle on a
    # downsampled copy and apply the rotation at full resolution.
    _DESKEW_ESTIMATE_MAX_SIDE = 1500

    @staticmethod
    def _estimate_skew_angle(gray: np.ndarray) -> float | None:
        """Estimate the page skew angle, downsampling large inputs first.

        The angle is estimated on a copy whose longest side is capped at
        ``_DESKEW_ESTIMATE_MAX_SIDE``; images already within the cap are used
        as-is. Returns ``None`` when no skew could be determined.
        """
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
        except Exception:
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
            w = window if isinstance(window, int) and window % 2 == 1 else 25
            thresh = threshold_sauvola(
                gray, window_size=w, k=float(k) if k is not None else 0.2
            )
            binary = (gray > thresh).astype(np.uint8) * 255
            return np.asarray(binary, dtype=np.uint8)
        if m in ("adaptive", "adaptive_otsu", "adaptive-gaussian"):
            # Use Gaussian adaptive threshold
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
            )
        # Otsu as default fallback
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
        image: Image.Image, cfg: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Preprocess a PIL image for Tesseract OCR according to cfg.
        Returns (PIL.Image in 'L' mode with 0/255, diagnostics dict)
        """
        diag: dict[str, Any] = {}
        # 1) Flatten alpha and to np
        np_img = ImageProcessor._pil_to_np(image)
        # 2) Grayscale
        gray = ImageProcessor._ensure_grayscale(np_img)
        # 3) Polarity
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
        # 4) Deskew
        if bool(cfg.get("deskew", True)):
            gray, angle = ImageProcessor._deskew(gray)
            diag["skew_angle"] = float(angle)
        else:
            diag["skew_angle"] = 0.0
        # 5) Denoise
        gray = ImageProcessor._denoise(gray, cfg.get("denoise", "median"))
        # 6) Binarization
        binary = ImageProcessor._binarize(
            gray,
            cfg.get("binarization", "otsu"),
            int(cfg.get("sauvola_window", 25)),
            float(cfg.get("sauvola_k", 0.2)),
        )
        diag["binarization"] = cfg.get("binarization", "otsu")
        # 7) Morphology
        binary = ImageProcessor._morph(
            binary, cfg.get("morphology", "none"), int(cfg.get("morph_kernel", 3))
        )
        diag["morphology"] = cfg.get("morphology", "none")
        # 8) Border
        b = int(cfg.get("border_px", 10))
        if b and b > 0:
            binary = cv2.copyMakeBorder(
                binary, b, b, b, b, cv2.BORDER_CONSTANT, value=255
            )
        # Convert back to PIL 'L'
        pil_bin = Image.fromarray(binary)
        if pil_bin.mode != "L":
            pil_bin = pil_bin.convert("L")
        return pil_bin, diag

    @staticmethod
    def process_and_save_images_for_tesseract(
        source_folder: Path,
        preprocessed_folder: Path,
        page_indices: list[int] | None = None,
    ) -> tuple[list[Path], dict[str, int]]:
        """
        Process images for Tesseract (lossless, full resolution) and save as PNG/TIFF.

        Args:
            source_folder: Path to the source folder containing original images.
            preprocessed_folder: Path to save processed images.
            page_indices: Optional 0-based indices into the sorted image list.
                If None, all images are processed.

        Returns:
            A tuple of (successfully written output paths, order map). The order
            map is ``{output_name: absolute_index}`` where the absolute index is
            each file's position in the FULL sorted source listing (before the
            page-range filter), so a resumed run over a different page subset
            keeps every page at its true position in the merged output.
        """
        config_service = get_config_service()
        tip_cfg = config_service.get_image_processing_config().get(
            "tesseract_image_processing", {}
        )
        preproc_cfg = tip_cfg.get("preprocessing", {})
        output_format = str(preproc_cfg.get("output_format", "png")).lower()
        target_dpi = int(tip_cfg.get("target_dpi", 300))
        embed_dpi = bool(preproc_cfg.get("embed_dpi_metadata", True))
        # Concurrency settings
        conc = config_service.get_concurrency_config()
        img_conc = conc.get("concurrency", {}).get("image_processing", {})
        processes = int(img_conc.get("concurrency_limit", 4))

        image_files: list[Path] = [
            p
            for p in source_folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not image_files:
            return [], {}

        # Deterministic natural ordering by filename (B5): page_2 before page_10.
        image_files.sort(key=lambda p: natural_sort_key(p.name))

        # Source files sharing a stem across extensions (scan_001.png +
        # scan_001.tif) would map to ONE output name, silently clobbering each
        # other on disk and dropping a page from the output. Mirror the CT-9
        # GPT-path fix with extension-inclusive names, but only for the
        # colliding stems, so existing resume artifacts for the common
        # (collision-free) case keep matching. Duplicates are detected over the
        # FULL sorted folder listing so naming stays stable under page ranges.
        stem_counts = Counter(p.stem for p in image_files)
        dup_stems = {stem for stem, n in stem_counts.items() if n > 1}

        # Apply page-range filter, keeping each file's absolute index (its
        # position in the FULL sorted listing) so the merged output can be
        # ordered correctly even under a resumed page subset.
        if page_indices is not None:
            selected = [
                (i, image_files[i]) for i in page_indices if 0 <= i < len(image_files)
            ]
        else:
            selected = list(enumerate(image_files))

        preprocessed_folder.mkdir(parents=True, exist_ok=True)
        suffix = ".png" if output_format == "png" else ".tif"

        out_paths: list[Path] = []
        order_map: dict[str, int] = {}
        for abs_idx, img in selected:
            out_path = preprocessed_folder / (
                f"{img.name if img.stem in dup_stems else img.stem}"
                f"_tess_preprocessed{suffix}"
            )
            out_paths.append(out_path)
            order_map[out_path.name] = abs_idx

        # Build args for multiprocessing
        args_list = [
            (img_path, out_path, preproc_cfg, output_format, target_dpi, embed_dpi)
            for (_abs_idx, img_path), out_path in zip(selected, out_paths, strict=False)
        ]

        # Run in process pool honoring concurrency config
        run_multiprocessing_tasks(
            ImageProcessor._tesseract_preprocess_image_task,
            args_list,
            processes=processes,
        )

        # Return files that were successfully written, plus the order map.
        return [p for p in out_paths if p.exists()], order_map

    @staticmethod
    def _tesseract_preprocess_image_task(
        img_path: Path,
        out_path: Path,
        cfg: dict[str, Any],
        output_format: str,
        target_dpi: int,
        embed_dpi: bool,
    ) -> str | None:
        """Worker: open, preprocess with Tesseract pipeline, and save losslessly."""
        try:
            with Image.open(img_path) as im:
                # Honor EXIF orientation before preprocessing (B14).
                # exif_transpose always makes a full-image copy; skip it when
                # there is no orientation to apply (tag absent or == 1) and use
                # the image directly (byte-identical output).
                if im.getexif().get(0x0112, 1) != 1:
                    oriented = ImageOps.exif_transpose(im) or im
                else:
                    oriented = im
                processed_img, diag = ImageProcessor.preprocess_for_tesseract(
                    oriented, cfg
                )
                if embed_dpi:
                    processed_img.save(out_path, dpi=(target_dpi, target_dpi))
                else:
                    processed_img.save(out_path)
                logger.debug(
                    "Tesseract-preprocessed %s -> %s diag=%s",
                    img_path.name,
                    out_path.name,
                    diag,
                )
                return str(out_path)
        except Exception as e:
            logger.error(f"Error Tesseract-preprocessing image {img_path.name}: {e}")
            return None
