"""Unit tests for modules.images.pipeline and modules.images.encoding.

Covers ``encode_image_to_data_url``, ``encode_bytes_to_base64``, the
in-memory preprocessing chain (``process_pil``/``resolve_detail``/
``resize_for_detail``), the Tesseract-path numpy helpers (``_pil_to_np``,
``_deskew``, ``_ensure_grayscale``), and the folder preparation helper.

All PIL images are constructed in-memory; numpy arrays are small; no real
OCR or FFmpeg process is invoked.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Priming import to avoid a circular-import chain when this module is the
# first to touch modules.images.
import modules.transcribe.dual_mode  # noqa: F401
from modules.images.encoding import encode_bytes_to_base64, encode_image_to_data_url
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
# encode_bytes_to_base64
# ---------------------------------------------------------------------------


class TestEncodeBytesToBase64:
    @pytest.mark.unit
    def test_round_trip(self) -> None:
        data = b"\xff\xd8\xff\xe0binary jpeg-ish bytes"
        encoded = encode_bytes_to_base64(data)
        assert base64.b64decode(encoded) == data


# ---------------------------------------------------------------------------
# ImageProcessor.resolve_detail / process_pil (in-memory chain)
# ---------------------------------------------------------------------------


class TestResolveDetail:
    @pytest.mark.unit
    def test_google_uses_media_resolution(self) -> None:
        cfg = {"media_resolution": "medium", "llm_detail": "high"}
        assert ImageProcessor.resolve_detail(cfg, "google") == "medium"

    @pytest.mark.unit
    def test_anthropic_uses_resize_profile(self) -> None:
        cfg = {"resize_profile": "high", "llm_detail": "low"}
        assert ImageProcessor.resolve_detail(cfg, "anthropic") == "high"

    @pytest.mark.unit
    def test_openai_uses_llm_detail(self) -> None:
        cfg = {"llm_detail": "original"}
        assert ImageProcessor.resolve_detail(cfg, "openai") == "original"

    @pytest.mark.unit
    def test_defaults(self) -> None:
        assert ImageProcessor.resolve_detail({}, "openai") == "high"
        assert ImageProcessor.resolve_detail({}, "anthropic") == "auto"
        assert ImageProcessor.resolve_detail({}, "google") == "high"


class TestProcessPil:
    @pytest.mark.unit
    def test_returns_jpeg_bytes_and_size(self) -> None:
        img = Image.new("RGB", (40, 20), (128, 128, 128))
        cfg = {"grayscale_conversion": True, "resize_profile": "none"}
        data, width, height = ImageProcessor.process_pil(img, cfg, "openai")
        assert data[:3] == b"\xff\xd8\xff"  # JPEG magic
        assert (width, height) == (40, 20)
        with Image.open(io.BytesIO(data)) as out:
            assert out.mode == "L"  # grayscale applied
            assert out.size == (40, 20)

    @pytest.mark.unit
    def test_transparency_flattened(self) -> None:
        rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        cfg = {
            "handle_transparency": True,
            "grayscale_conversion": False,
            "resize_profile": "none",
        }
        data, _w, _h = ImageProcessor.process_pil(rgba, cfg, "openai")
        with Image.open(io.BytesIO(data)) as out:
            assert out.mode == "RGB"

    @pytest.mark.unit
    def test_grayscale_disabled_keeps_rgb(self) -> None:
        img = Image.new("RGB", (8, 8), (10, 20, 30))
        cfg = {"grayscale_conversion": False, "resize_profile": "none"}
        data, _w, _h = ImageProcessor.process_pil(img, cfg, "openai")
        with Image.open(io.BytesIO(data)) as out:
            assert out.mode == "RGB"

    @pytest.mark.unit
    def test_resize_applied_via_detail(self) -> None:
        img = Image.new("RGB", (3000, 1500), (0, 0, 0))
        cfg = {
            "grayscale_conversion": False,
            "llm_detail": "high",
            "high_target_box": [768, 1536],
            "resize_profile": "auto",
        }
        _data, width, height = ImageProcessor.process_pil(img, cfg, "openai")
        assert (width, height) == (768, 1536)

    @pytest.mark.unit
    def test_original_detail_caps_pixels(self) -> None:
        img = Image.new("RGB", (4000, 4000), (0, 0, 0))
        cfg = {
            "grayscale_conversion": True,
            "llm_detail": "original",
            "original_max_side_px": 6000,
            "original_max_pixels": 1_000_000,
            "resize_profile": "auto",
        }
        _data, width, height = ImageProcessor.process_pil(img, cfg, "openai")
        assert width * height <= 1_000_000


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
        out = ImageProcessor.resize_for_detail(img, "high", cfg, model_type="anthropic")
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
        out = ImageProcessor.resize_for_detail(img, "high", cfg, model_type="openai")
        # OpenAI branch pads to exactly the target box dimensions
        assert out.size == (768, 1536)

    @pytest.mark.unit
    def test_resize_profile_none_disables_resize(self) -> None:
        img = Image.new("RGB", (3000, 1500), (0, 0, 0))
        cfg = {"resize_profile": "none"}
        out = ImageProcessor.resize_for_detail(img, "high", cfg, model_type="openai")
        assert out.size == (3000, 1500)


# ---------------------------------------------------------------------------
# ImageProcessor.derive_render_zoom ("direct" render strategy)
# ---------------------------------------------------------------------------

# A4 in PostScript points (72 pt/inch): 210 x 297 mm.
_A4_W_PT = 595.0
_A4_H_PT = 842.0


class TestDeriveRenderZoom:
    @pytest.mark.unit
    def test_original_under_caps_renders_at_target_dpi(self) -> None:
        """A page whose target_dpi render already fits the original caps must
        derive a zoom of exactly target_dpi/72 (byte-identical to supersample)."""
        cfg = {
            "llm_detail": "original",
            "resize_profile": "high",
            "original_max_side_px": 6000,
            "original_max_pixels": 10240000,
        }
        # A4 at 300 DPI: long side ~3508 px, ~8.7 MP -> within both caps.
        zoom = ImageProcessor.derive_render_zoom(
            _A4_W_PT, _A4_H_PT, 300, 0, cfg, "openai"
        )
        assert zoom == pytest.approx(300 / 72.0)

    @pytest.mark.unit
    def test_original_over_pixel_cap_reduces_zoom(self) -> None:
        cfg = {
            "llm_detail": "original",
            "resize_profile": "high",
            "original_max_side_px": 6000,
            "original_max_pixels": 10240000,
        }
        # 800 x 1200 pt at 300 DPI -> ~3333 x 5000 px = ~16.7 MP > 10.24 MP.
        zoom = ImageProcessor.derive_render_zoom(800.0, 1200.0, 300, 0, cfg, "openai")
        rendered_px = (800.0 * zoom) * (1200.0 * zoom)
        assert rendered_px == pytest.approx(10240000, rel=1e-6)
        assert zoom < 300 / 72.0

    @pytest.mark.unit
    def test_anthropic_high_derives_long_edge_dpi(self) -> None:
        cfg = {"resize_profile": "auto", "high_max_side_px": 2576}
        zoom = ImageProcessor.derive_render_zoom(
            _A4_W_PT, _A4_H_PT, 300, 0, cfg, "anthropic"
        )
        long_px_at_target = _A4_H_PT * 300 / 72.0
        expected_dpi = 300 * 2576 / long_px_at_target
        assert zoom * 72.0 == pytest.approx(expected_dpi, rel=1e-6)
        # Long edge lands at the cap.
        assert _A4_H_PT * zoom == pytest.approx(2576, abs=1.0)

    @pytest.mark.unit
    def test_anthropic_small_page_not_upscaled(self) -> None:
        """A page smaller than the cap keeps target_dpi (no render-upscaling)."""
        cfg = {"resize_profile": "auto", "high_max_side_px": 2576}
        # 200 x 100 pt at 72 DPI -> long side 200 px, far below 2576.
        zoom = ImageProcessor.derive_render_zoom(200.0, 100.0, 72, 0, cfg, "anthropic")
        assert zoom == pytest.approx(72 / 72.0)

    @pytest.mark.unit
    def test_openai_box_fit_downscales(self) -> None:
        cfg = {
            "resize_profile": "high",
            "llm_detail": "high",
            "high_target_box": [768, 1536],
        }
        zoom = ImageProcessor.derive_render_zoom(
            _A4_W_PT, _A4_H_PT, 300, 0, cfg, "openai"
        )
        # Fit A4 (aspect ~0.707) into 768x1536 (aspect 0.5): width-limited, so
        # the width edge lands on 768.
        assert _A4_W_PT * zoom == pytest.approx(768, abs=1.0)
        assert zoom < 300 / 72.0

    @pytest.mark.unit
    def test_openai_box_larger_than_target_not_upscaled(self) -> None:
        cfg = {
            "resize_profile": "high",
            "llm_detail": "high",
            "high_target_box": [768, 1536],
        }
        # 100 x 50 pt at 72 DPI -> 100x50 px, smaller than the box; no upscale.
        zoom = ImageProcessor.derive_render_zoom(100.0, 50.0, 72, 0, cfg, "openai")
        assert zoom == pytest.approx(72 / 72.0)

    @pytest.mark.unit
    def test_resize_profile_none_returns_target_zoom(self) -> None:
        cfg = {"resize_profile": "none"}
        zoom = ImageProcessor.derive_render_zoom(
            _A4_W_PT, _A4_H_PT, 300, 0, cfg, "openai"
        )
        assert zoom == pytest.approx(300 / 72.0)

    @pytest.mark.unit
    def test_max_pixels_guard_applies(self) -> None:
        """The max_pixels guard reduces even a profile-derived zoom."""
        cfg = {"resize_profile": "none"}
        # None profile would render A4 at 300 DPI (~8.7 MP); cap at 2 MP.
        zoom = ImageProcessor.derive_render_zoom(
            _A4_W_PT, _A4_H_PT, 300, 2_000_000, cfg, "openai"
        )
        rendered_px = (_A4_W_PT * zoom) * (_A4_H_PT * zoom)
        assert rendered_px == pytest.approx(2_000_000, rel=1e-6)


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
    """Tesseract-path binarization: default is Otsu (fast); Sauvola and
    adaptive remain available via the ``binarization`` config option."""

    @pytest.mark.unit
    def test_default_method_is_otsu(self) -> None:
        # A gradient image so Otsu produces a clean two-level output.
        gray = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
        default_out = ImageProcessor._binarize(gray, None, 25, 0.2)  # type: ignore[arg-type]
        otsu_out = ImageProcessor._binarize(gray, "otsu", 25, 0.2)
        assert np.array_equal(default_out, otsu_out)
        # Binary output: only 0 and 255.
        assert set(np.unique(default_out).tolist()) <= {0, 255}

    @pytest.mark.unit
    def test_sauvola_still_available(self) -> None:
        gray = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
        out = ImageProcessor._binarize(gray, "sauvola", 25, 0.2)
        assert out.dtype == np.uint8
        assert set(np.unique(out).tolist()) <= {0, 255}

    @pytest.mark.unit
    def test_adaptive_still_available(self) -> None:
        gray = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
        out = ImageProcessor._binarize(gray, "adaptive", 25, 0.2)
        assert set(np.unique(out).tolist()) <= {0, 255}

    @pytest.mark.unit
    def test_preprocess_default_binarization_diag_is_otsu(self) -> None:
        img = Image.new("L", (64, 64), 200)
        _out, diag = ImageProcessor.preprocess_for_tesseract(img, {})
        assert diag["binarization"] == "otsu"


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

        parent, preprocessed, temp_jsonl, out_txt = ImageProcessor.prepare_image_folder(
            source, out_dir
        )

        assert parent.exists() and parent.is_dir()
        assert parent.parent == out_dir
        # The preprocessed folder is Tesseract-only and created lazily;
        # prepare_image_folder returns the path without creating it.
        assert not preprocessed.exists()
        assert preprocessed.parent == parent
        assert preprocessed.name == "preprocessed_images"
        assert temp_jsonl.exists()  # touched into existence
        assert temp_jsonl.suffix == ".jsonl"
        assert out_txt.suffix == ".txt"
        assert out_txt.parent == parent


# ---------------------------------------------------------------------------
# ImageProcessor.process_and_save_images_for_tesseract — stem collisions
# ---------------------------------------------------------------------------


class TestTesseractStemCollision:
    """Regression: source files sharing a stem across extensions must not
    clobber each other's preprocessed output (mirrors the CT-9 GPT-path fix).

    Only colliding stems get extension-inclusive names; collision-free files
    keep the legacy ``{stem}_tess_preprocessed`` name so existing resume
    artifacts still match.
    """

    def _run(self, tmp_path: Path, filenames: list[str]) -> list[Path]:
        from unittest.mock import patch

        source = tmp_path / "src"
        source.mkdir()
        for name in filenames:
            (source / name).write_bytes(b"fake image data")
        out_dir = tmp_path / "pre"

        def fake_pool(func, args_list, processes=None):  # noqa: ANN001, ANN202
            # Sequential fake: touch each out_path so the naming logic is
            # exercised without real image preprocessing.
            results = []
            for args in args_list:
                args[1].write_bytes(b"out")
                results.append(str(args[1]))
            return results

        with (
            patch(
                "modules.images.pipeline.run_multiprocessing_tasks",
                side_effect=fake_pool,
            ),
            patch("modules.images.pipeline.get_config_service") as mock_cs,
        ):
            mock_cs.return_value.get_image_processing_config.return_value = {}
            mock_cs.return_value.get_concurrency_config.return_value = {}
            return ImageProcessor.process_and_save_images_for_tesseract(source, out_dir)

    @pytest.mark.unit
    def test_colliding_stems_get_distinct_outputs(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, ["scan_001.png", "scan_001.tif", "scan_002.png"])
        names = sorted(p.name for p in result)
        assert len(names) == len(set(names)) == 3
        assert "scan_001.png_tess_preprocessed.png" in names
        assert "scan_001.tif_tess_preprocessed.png" in names
        # Non-colliding stem keeps the legacy name for resume compatibility.
        assert "scan_002_tess_preprocessed.png" in names

    @pytest.mark.unit
    def test_collision_free_folder_keeps_legacy_names(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, ["page_1.png", "page_2.png"])
        names = sorted(p.name for p in result)
        assert names == [
            "page_1_tess_preprocessed.png",
            "page_2_tess_preprocessed.png",
        ]
