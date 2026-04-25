"""Unit tests for modules.images.pipeline and modules.images.encoding.

Covers ``encode_image_to_data_url``, the in-place image transforms on
``ImageProcessor`` (grayscale, transparency flattening, resize-for-detail),
the Tesseract-path numpy helpers (``_pil_to_np``, ``_deskew``,
``_ensure_grayscale``), and the folder preparation helper.

All PIL images are constructed in-memory; numpy arrays are small; no real
OCR or FFmpeg process is invoked.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Priming import to avoid a circular-import chain when this module is the
# first to touch modules.images.
import modules.transcribe.dual_mode  # noqa: F401

from modules.images.encoding import encode_image_to_data_url
from modules.images.pipeline import ImageProcessor


# ---------------------------------------------------------------------------
# encode_image_to_data_url
# ---------------------------------------------------------------------------

class TestEncodeImageToDataURL:

    @pytest.mark.unit
    def test_png_round_trip(self, tmp_path: Path) -> None:
        """Round-trip a 1x1 PNG through encode_image_to_data_url."""
        img_path = tmp_path / "tiny.png"
        Image.new("RGB", (1, 1), (255, 0, 0)).save(img_path, format="PNG")

        url = encode_image_to_data_url(img_path)
        assert url.startswith("data:image/png;base64,")

        payload_b64 = url.split(",", 1)[1]
        payload = base64.b64decode(payload_b64)
        assert len(payload) > 0
        # Valid PNG magic header
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.unit
    def test_jpeg_mime(self, tmp_path: Path) -> None:
        """A .jpg file returns image/jpeg in the data URL."""
        img_path = tmp_path / "tiny.jpg"
        Image.new("RGB", (2, 2), (0, 255, 0)).save(img_path, format="JPEG")

        url = encode_image_to_data_url(img_path)
        assert url.startswith("data:image/jpeg;base64,")

    @pytest.mark.unit
    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "file.xyz"
        bogus.write_bytes(b"nothing")
        with pytest.raises(ValueError, match="Unsupported image format"):
            encode_image_to_data_url(bogus)


# ---------------------------------------------------------------------------
# ImageProcessor instance methods: convert_to_grayscale / handle_transparency
# ---------------------------------------------------------------------------

@pytest.fixture
def image_processor(tmp_path: Path) -> ImageProcessor:
    """Build an ImageProcessor instance backed by a tiny PNG file."""
    img_path = tmp_path / "source.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(img_path, format="PNG")
    return ImageProcessor(img_path, provider="openai")


class TestConvertToGrayscale:

    @pytest.mark.unit
    def test_rgb_becomes_L(self, image_processor: ImageProcessor) -> None:
        rgb = Image.new("RGB", (4, 4), (128, 128, 128))
        # Force grayscale_conversion on (defaults to True but keep explicit)
        image_processor.img_cfg["grayscale_conversion"] = True
        out = image_processor.convert_to_grayscale(rgb)
        assert out.mode == "L"

    @pytest.mark.unit
    def test_already_L_returns_same_object(
        self, image_processor: ImageProcessor
    ) -> None:
        gray = Image.new("L", (4, 4), 128)
        image_processor.img_cfg["grayscale_conversion"] = True
        out = image_processor.convert_to_grayscale(gray)
        assert out is gray  # short-circuit returns the input

    @pytest.mark.unit
    def test_disabled_returns_input_unchanged(
        self, image_processor: ImageProcessor
    ) -> None:
        rgb = Image.new("RGB", (4, 4), (10, 20, 30))
        image_processor.img_cfg["grayscale_conversion"] = False
        out = image_processor.convert_to_grayscale(rgb)
        assert out is rgb
        assert out.mode == "RGB"


class TestHandleTransparency:

    @pytest.mark.unit
    def test_rgba_flattened_to_rgb(self, image_processor: ImageProcessor) -> None:
        rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        image_processor.img_cfg["handle_transparency"] = True
        out = image_processor.handle_transparency(rgba)
        assert out.mode == "RGB"

    @pytest.mark.unit
    def test_rgb_input_unchanged(self, image_processor: ImageProcessor) -> None:
        rgb = Image.new("RGB", (4, 4), (10, 20, 30))
        image_processor.img_cfg["handle_transparency"] = True
        out = image_processor.handle_transparency(rgb)
        assert out is rgb


# ---------------------------------------------------------------------------
# ImageProcessor.resize_for_detail
# ---------------------------------------------------------------------------

class TestResizeForDetail:

    @pytest.mark.unit
    def test_low_detail_shrinks_long_side(self) -> None:
        img = Image.new("RGB", (2000, 500), (0, 0, 0))
        cfg = {"low_max_side_px": 256, "resize_profile": "auto"}
        out = ImageProcessor.resize_for_detail(img, "low", cfg, model_type="openai")
        assert max(out.size) == 256

    @pytest.mark.unit
    def test_low_detail_no_upscale_when_below_cap(self) -> None:
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        cfg = {"low_max_side_px": 512, "resize_profile": "auto"}
        out = ImageProcessor.resize_for_detail(img, "low", cfg, model_type="openai")
        assert out.size == (100, 100)

    @pytest.mark.unit
    def test_anthropic_high_caps_long_side(self) -> None:
        img = Image.new("RGB", (3000, 1000), (0, 0, 0))
        cfg = {"high_max_side_px": 1568, "resize_profile": "auto"}
        out = ImageProcessor.resize_for_detail(
            img, "high", cfg, model_type="anthropic"
        )
        # int() truncation may drop the longest side by one pixel; allow a
        # small tolerance so the test is robust to floating-point rounding.
        assert max(out.size) <= 1568
        assert max(out.size) >= 1567
        # Aspect-ratio preserved (scale of ~0.52)
        assert out.size[0] > out.size[1]

    @pytest.mark.unit
    def test_openai_high_fits_into_target_box_with_padding(self) -> None:
        img = Image.new("RGB", (3000, 1500), (0, 0, 0))
        cfg = {"high_target_box": [768, 1536], "resize_profile": "auto"}
        out = ImageProcessor.resize_for_detail(
            img, "high", cfg, model_type="openai"
        )
        # OpenAI branch pads to exactly the target box dimensions
        assert out.size == (768, 1536)

    @pytest.mark.unit
    def test_resize_profile_none_disables_resize(self) -> None:
        img = Image.new("RGB", (3000, 1500), (0, 0, 0))
        cfg = {"resize_profile": "none"}
        out = ImageProcessor.resize_for_detail(
            img, "high", cfg, model_type="openai"
        )
        assert out.size == (3000, 1500)


# ---------------------------------------------------------------------------
# Tesseract helpers: _pil_to_np / _ensure_grayscale / _deskew
# ---------------------------------------------------------------------------

class TestTesseractHelpers:

    @pytest.mark.unit
    def test_pil_to_np_rgb_returns_bgr_array(self) -> None:
        img = Image.new("RGB", (4, 3), (10, 20, 30))
        arr = ImageProcessor._pil_to_np(img)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4, 3)
        assert arr.dtype == np.uint8
        # PIL (R,G,B) = (10, 20, 30) -> cv2 BGR = (30, 20, 10)
        assert tuple(arr[0, 0].tolist()) == (30, 20, 10)

    @pytest.mark.unit
    def test_pil_to_np_gray_returns_2d_array(self) -> None:
        img = Image.new("L", (4, 3), 128)
        arr = ImageProcessor._pil_to_np(img)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)
        assert arr.dtype == np.uint8

    @pytest.mark.unit
    def test_pil_to_np_rgba_is_flattened(self) -> None:
        img = Image.new("RGBA", (4, 3), (255, 0, 0, 128))
        arr = ImageProcessor._pil_to_np(img)
        # After flattening onto white, result should be a 3-channel BGR array
        assert arr.ndim == 3
        assert arr.shape[-1] == 3

    @pytest.mark.unit
    def test_ensure_grayscale_3d_to_2d(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        out = ImageProcessor._ensure_grayscale(rgb)
        assert out.ndim == 2
        assert out.shape == (4, 4)

    @pytest.mark.unit
    def test_ensure_grayscale_2d_returns_same(self) -> None:
        gray = np.zeros((4, 4), dtype=np.uint8)
        out = ImageProcessor._ensure_grayscale(gray)
        assert out.shape == (4, 4)

    @pytest.mark.unit
    def test_deskew_returns_shape_preserved_and_angle_float(self) -> None:
        gray = np.full((20, 20), 255, dtype=np.uint8)
        # Draw a horizontal line so determine_skew has content
        gray[10, :] = 0
        rotated, angle = ImageProcessor._deskew(gray)
        assert isinstance(rotated, np.ndarray)
        assert rotated.shape == gray.shape
        assert isinstance(angle, float)


# ---------------------------------------------------------------------------
# ImageProcessor.prepare_image_folder
# ---------------------------------------------------------------------------

class TestPrepareImageFolder:

    @pytest.mark.unit
    def test_creates_expected_subfolders_and_files(self, tmp_path: Path) -> None:
        source = tmp_path / "source_images"
        source.mkdir()
        out_dir = tmp_path / "image_out"
        out_dir.mkdir()

        parent, preprocessed, temp_jsonl, out_txt = (
            ImageProcessor.prepare_image_folder(source, out_dir)
        )

        assert parent.exists() and parent.is_dir()
        assert parent.parent == out_dir
        assert preprocessed.exists() and preprocessed.is_dir()
        assert preprocessed.parent == parent
        assert preprocessed.name == "preprocessed_images"
        assert temp_jsonl.exists()  # touched into existence
        assert temp_jsonl.suffix == ".jsonl"
        assert out_txt.suffix == ".txt"
        assert out_txt.parent == parent
