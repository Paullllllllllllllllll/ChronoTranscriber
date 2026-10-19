"""Tesseract image preprocessing in chronotranscriber.images.pipeline.

Covers the array conversion, grayscale and deskew helpers, binarization, the
``--ocr-preprocess`` presets, and the folder preparation that writes
preprocessed pages. Images are built in memory; no OCR process is invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from chronotranscriber.images.pipeline import (
    TESSERACT_PREPROCESSING,
    ImageProcessor,
    TesseractSettings,
    tesseract_settings,
)


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


def _make_skewed_page(width: int, height: int, angle_deg: float) -> np.ndarray:
    """Build a synthetic printed page (dark text rows on white) skewed by
    ``angle_deg`` degrees, returned as an 8-bit grayscale array."""
    import cv2

    page = np.full((height, width), 255, dtype=np.uint8)
    # Evenly spaced horizontal "text" rows.
    for y in range(80, height - 80, 60):
        page[y : y + 8, 60 : width - 60] = 0
    center = (width // 2, height // 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        page,
        m,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


class TestDeskewDownsampling:
    """The skew angle is estimated on a downsampled copy (long edge capped)
    and applied to the full-resolution image; the estimate must stay close to
    the full-resolution estimate."""

    @pytest.mark.unit
    @pytest.mark.parametrize("angle", [-5.0, -2.0, 3.0, 7.0])
    def test_downsampled_angle_matches_full_res(self, angle: float) -> None:
        from deskew import determine_skew

        # Large page so the long edge exceeds the estimation cap.
        page = _make_skewed_page(2000, 2600, angle)
        full = determine_skew(page)
        est = ImageProcessor._estimate_skew_angle(page)
        assert full is not None and est is not None
        # Downsampled estimate tracks the full-resolution estimate closely.
        assert abs(float(est) - float(full)) <= 0.2

    @pytest.mark.unit
    def test_small_image_not_downsampled(self) -> None:
        # Below the cap: estimate equals the direct determine_skew result.
        from deskew import determine_skew

        gray = np.full((400, 400), 255, dtype=np.uint8)
        gray[200, :] = 0
        direct = determine_skew(gray)
        est = ImageProcessor._estimate_skew_angle(gray)
        assert (direct is None) == (est is None)
        if direct is not None and est is not None:
            assert abs(float(direct) - float(est)) < 1e-9


class TestBinarization:
    """Binarization yields a two-level image; Otsu is the default method."""

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["otsu", "sauvola", "adaptive"])
    def test_binarize_returns_a_two_level_image(self, method: str) -> None:
        gray = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
        out = ImageProcessor._binarize(gray, method, 25, 0.2)
        assert out.dtype == np.uint8
        assert set(np.unique(out).tolist()) <= {0, 255}

    @pytest.mark.unit
    def test_preprocess_uses_otsu_when_no_method_is_configured(self) -> None:
        img = Image.new("L", (64, 64), 200)
        _out, diag = ImageProcessor.preprocess_for_tesseract(img, {})
        assert diag["binarization"] == "otsu"


class TestPresets:
    @pytest.mark.unit
    def test_default_preset_keeps_every_step(self) -> None:
        settings = tesseract_settings("default", 3)
        assert dict(settings.preprocessing) == dict(TESSERACT_PREPROCESSING)
        assert settings.concurrency == 3
        assert (settings.target_dpi, settings.suffix) == (300, ".png")

    @pytest.mark.unit
    def test_sauvola_preset_changes_only_binarization(self) -> None:
        steps = tesseract_settings("sauvola").preprocessing
        assert steps["binarization"] == "sauvola"
        assert steps["deskew"] is True

    @pytest.mark.unit
    def test_none_preset_skips_polarity_deskew_and_denoise(self) -> None:
        steps = tesseract_settings("none").preprocessing
        assert steps["invert_to_dark_on_light"] == "never"
        assert steps["deskew"] is False
        assert steps["denoise"] == "none"


def _fake_task(img_path: Path, out_path: Path, settings: Any) -> str:
    out_path.write_bytes(b"out")
    return str(out_path)


class TestTesseractStemCollision:
    """Source files sharing a stem across extensions must not overwrite each
    other's preprocessed output.

    Only colliding stems get extension-inclusive names; collision-free files
    keep the ``{stem}_tess_preprocessed`` name so earlier resume records
    still match.
    """

    def _run(self, tmp_path: Path, filenames: list[str]) -> list[Path]:
        source = tmp_path / "src"
        source.mkdir()
        for name in filenames:
            (source / name).write_bytes(b"fake image data")
        out_dir = tmp_path / "pre"
        with patch.object(
            ImageProcessor,
            "_tesseract_preprocess_image_task",
            staticmethod(_fake_task),
        ):
            files, _order_map = ImageProcessor.process_and_save_images_for_tesseract(
                source, out_dir
            )
            return files

    @pytest.mark.unit
    def test_colliding_stems_get_distinct_outputs(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, ["scan_001.png", "scan_001.tif", "scan_002.png"])
        names = sorted(p.name for p in result)
        assert len(names) == len(set(names)) == 3
        assert "scan_001.png_tess_preprocessed.png" in names
        assert "scan_001.tif_tess_preprocessed.png" in names
        # Non-colliding stem keeps the stem-only name.
        assert "scan_002_tess_preprocessed.png" in names

    @pytest.mark.unit
    def test_collision_free_folder_keeps_stem_names(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, ["page_1.png", "page_2.png"])
        names = sorted(p.name for p in result)
        assert names == [
            "page_1_tess_preprocessed.png",
            "page_2_tess_preprocessed.png",
        ]


class TestTesseractFolderPreprocessing:
    @pytest.mark.unit
    def test_writes_lossless_pages_at_the_target_resolution(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "src"
        source.mkdir()
        Image.new("RGB", (80, 60), (240, 240, 240)).save(source / "page_1.png")
        (source / "page_2.png").write_bytes(b"not an image")
        # A stale output of an earlier run must not stand in for the failed page.
        (tmp_path / "pre").mkdir()
        (tmp_path / "pre" / "page_2_tess_preprocessed.png").write_bytes(b"stale")
        settings = TesseractSettings(target_dpi=200, concurrency=2)

        files, order = ImageProcessor.process_and_save_images_for_tesseract(
            source, tmp_path / "pre", settings=settings
        )

        assert [p.name for p in files] == ["page_1_tess_preprocessed.png"]
        assert order == {
            "page_1_tess_preprocessed.png": 0,
            "page_2_tess_preprocessed.png": 1,
        }
        with Image.open(files[0]) as image:
            assert image.mode == "L"
            assert image.info["dpi"] == pytest.approx((200, 200), abs=0.1)
