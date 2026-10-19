"""Native-resolution rendering, model caps and repair re-renders of page images."""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import fitz
import numpy as np
import pytest
from PIL import Image

from chronotranscriber.batch.bodies import build_plan
from chronotranscriber.batch.repair import (
    ImageSettings,
    RepairDocument,
    page_image,
    repair_image_settings,
)
from chronotranscriber.common.batch import BatchRequest
from chronotranscriber.common.native import encode_payload, guarded_payload
from chronotranscriber.common.store import UnitRow
from chronotranscriber.output.store import image_settings_record
from chronotranscriber.settings import Settings
from chronotranscriber.sources.images import load_image, resolve_image_settings
from chronotranscriber.sources.pdf import render_pdf_page
from chronotranscriber.spec import ImageSpec, ModelSpec, resolve_spec

RESOLVE = "chronotranscriber.batch.repair.resolve_image_settings"


def model_spec(model: str = "gpt-6-astra") -> ModelSpec:
    """The run's model spec for *model*."""
    return resolve_spec({"input": Path("."), "model": model}).spec.model


def settings(
    model: str = "gpt-6-astra", provider: str = "openai", **values: Any
) -> dict[str, Any]:
    """The resolved settings of *model*, with *values* set afterwards."""
    cfg = resolve_image_settings(provider, model, ImageSpec(dpi="native"))[0]
    return {**cfg, **values}


def scan_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    image.paste((40, 80, 120), (width // 4, height // 4, width // 2, height // 2))
    return encode_payload(image, "png", 95)[0]


def scan_page(doc: Any, dpi: int = 300, width: int = 144, height: int = 216) -> Any:
    page = doc.new_page(width=width, height=height)
    page.insert_image(
        page.rect,
        stream=scan_bytes(round(width * dpi / 72), round(height * dpi / 72)),
    )
    return page


def scan_pdf(path: Path, dpi: int, width: int = 144, height: int = 216) -> Path:
    with fitz.open() as doc:
        scan_page(doc, dpi, width, height)
        doc.save(path)
    return path


@pytest.mark.parametrize("dpi", [150, 360, 600])
@pytest.mark.parametrize("strategy", ["direct", "supersample"])
def test_native_full_page(tmp_path: Path, dpi: int, strategy: str) -> None:
    pdf = scan_pdf(tmp_path / "scan.pdf", dpi)
    cfg = settings(max_pixels_per_page=100, render_strategy=strategy)

    page = render_pdf_page(pdf, 0, cfg, "openai")

    provenance = page.provenance()
    assert abs(provenance["width"] - dpi * 2) <= 1
    assert abs(provenance["height"] - dpi * 3) <= 1
    assert provenance["render_dpi"] == pytest.approx(dpi)
    assert provenance["downscale_reason"] == "none"


@pytest.mark.parametrize("strategy", ["direct", "supersample"])
def test_numeric_model_caps_and_memory_guard(tmp_path: Path, strategy: str) -> None:
    pdf = scan_pdf(tmp_path / "large.pdf", 300, 720, 960)
    for model in ("gpt-6-astra", "gpt-5.4"):
        cfg = settings(model, target_dpi=300, render_strategy=strategy)
        provenance = render_pdf_page(pdf, 0, cfg, "openai").provenance()
        if model == "gpt-6-astra":
            assert (provenance["width"], provenance["height"]) == (3000, 4000)
        else:
            patches = math.ceil(provenance["width"] / 32) * math.ceil(
                provenance["height"] / 32
            )
            assert patches <= 10000
            assert provenance["downscale_reason"] == "model_cap"
    cfg = settings(
        target_dpi=300, max_pixels_per_page=1_000_000, render_strategy=strategy
    )
    provenance = render_pdf_page(pdf, 0, cfg, "openai").provenance()
    assert provenance["width"] * provenance["height"] <= 1_003_000
    assert provenance["downscale_reason"] == "memory_guard"


@pytest.mark.parametrize(
    "model,side,patches",
    [("claude-opus-5", 2576, 4784), ("claude-haiku-4-5", 1568, 1568)],
)
def test_native_anthropic_render_cap(
    tmp_path: Path, model: str, side: int, patches: int
) -> None:
    pdf = scan_pdf(tmp_path / "scan.pdf", 600, 288, 432)
    cfg = settings(model, "anthropic", max_pixels_per_page=100)

    provenance = render_pdf_page(pdf, 0, cfg, "anthropic").provenance()

    width, height = provenance["width"], provenance["height"]
    assert max(width, height) <= side
    assert math.ceil(width / 28) * math.ceil(height / 28) <= patches
    assert provenance["downscale_reason"] == "model_cap"
    assert provenance["source_dpi_x"] == pytest.approx(600)


def test_large_native_ignores_the_numeric_memory_guard(tmp_path: Path) -> None:
    pdf = scan_pdf(tmp_path / "large.pdf", 300, 1440, 1200)
    native = render_pdf_page(pdf, 0, settings(), "openai").provenance()
    assert (native["width"], native["height"]) == (6000, 5000)
    assert native["downscale_reason"] == "none"
    numeric = render_pdf_page(pdf, 0, settings(target_dpi=300), "openai").provenance()
    assert numeric["width"] * numeric["height"] == pytest.approx(24_000_000, rel=0.001)
    assert numeric["downscale_reason"] == "memory_guard"


def test_png_falls_back_to_jpeg_over_the_byte_limit(tmp_path: Path) -> None:
    image = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    )
    png, _ = encode_payload(image, "png", 95)
    jpg, _, _ = guarded_payload(image, "jpeg", 70, 0)
    limit = 4 * ((len(jpg) + 2) // 3)
    path = tmp_path / "source.png"
    path.write_bytes(png)
    cfg = settings(
        payload_format="png",
        grayscale_conversion=False,
        jpeg_quality=70,
        max_image_bytes=limit,
    )

    page = load_image(path, 0, cfg, "openai")

    assert page.provenance()["format_fallback"]
    assert page.mime_type == "image/jpeg"
    assert page.image_name == "source.png_pre_processed.jpg"


def test_a_downscale_logs_one_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "source.png"
    Image.new("L", (800, 1200), "white").save(path, dpi=(150, 150))
    with caplog.at_level("INFO"):
        page = load_image(path, 0, settings(original_max_side_px=600), "openai")
    lines = [r.message for r in caplog.records if "reason model_cap" in r.message]
    assert len(lines) == 1
    assert "800x1200" in lines[0] and "400x600" in lines[0]
    assert page.provenance()["file_dpi_metadata"] == pytest.approx((150, 150), abs=0.1)


def unit(idx: int, label: str, source_index: int | str | None = None) -> UnitRow:
    """A failed page unit as the store holds it."""
    return UnitRow(
        doc_id=1,
        kind="page",
        idx=idx,
        status="failed",
        outcome="failed",
        source_index=source_index,
        label=label,
        content=f"[transcription error: {label}]",
        tool="chronotranscriber",
        model="gpt-6-astra",
        provider="openai",
        run_id="run-1",
        attempts=1,
        usage=None,
        error="boom",
        error_type=None,
        provenance=None,
        response_id=None,
        stale=0,
        updated_at=None,
    )


def document(
    tmp_path: Path, source: Path | None, record: Mapping[str, Any] | None
) -> RepairDocument:
    """The stored document of *source* with the image-settings *record*."""
    return RepairDocument(
        root=tmp_path,
        doc_id=1,
        name="document",
        input_path=source,
        image_settings=record,
        output=tmp_path / "document.txt",
        formats=("txt",),
    )


def recorded(cfg: Mapping[str, Any], model_type: str = "openai") -> dict[str, Any]:
    """The image-settings record a transcription run stores for *cfg*."""
    return image_settings_record(cfg, model_type)


def test_repair_rerenders_a_pdf_page_from_its_source(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    with fitz.open() as doc:
        for i in range(3):
            doc.new_page(width=200, height=100).insert_text((20, 50), f"P{i}")
        doc.save(pdf)
    cfg = {**settings(), "target_dpi": 72, "llm_detail": "high"}
    cfg["resolved_detail"] = "high"
    stored = document(tmp_path, pdf, recorded(cfg))
    with patch(RESOLVE, side_effect=AssertionError("Must use recorded settings")):
        images, warning = repair_image_settings(stored, model_spec(), None)

    page = page_image(stored, unit(1, "page_0002_pre_processed.jpg", 1), images)

    assert warning is None and images.recorded
    assert page.render_error is None
    assert (page.index, page.page_index) == (1, 1)
    assert page.image_name == "page_0002_pre_processed.jpg"
    assert page.mime_type == "image/jpeg"
    assert Image.open(io.BytesIO(base64.b64decode(page.base64))).format == "JPEG"
    assert page.provenance()["source_dpi_x"] == pytest.approx(72)
    assert images.request_options() == {"image_detail": "high"}


def test_repair_marks_a_page_without_source_as_unrendered(tmp_path: Path) -> None:
    stored = document(tmp_path, tmp_path / "gone.pdf", recorded(settings()))
    images = ImageSettings(settings(), "openai", True)

    page = page_image(stored, unit(0, "page_0001_pre_processed.jpg", 0), images)

    assert page.render_error is not None and "missing" in page.render_error
    assert page.base64 == ""
    assert page.image_name == "page_0001_pre_processed.jpg"


@pytest.mark.parametrize("raw_on_disk", [False, True])
def test_repair_uses_the_recorded_settings(tmp_path: Path, raw_on_disk: bool) -> None:
    source_dir = tmp_path / "document"
    source_dir.mkdir()
    cfg = settings(payload_format="png", jpeg_quality=60)
    if raw_on_disk:
        (source_dir / "source.png").write_bytes(scan_bytes(300, 450))
        source = source_dir
        target = unit(0, "source.png_pre_processed.jpg", "source.png_pre_processed.jpg")
    else:
        source = scan_pdf(tmp_path / "source.pdf", 150)
        target = unit(0, "page_0001_pre_processed.jpg", 0)
    stored = document(tmp_path, source, recorded(cfg))
    with patch(RESOLVE, side_effect=AssertionError("Must use recorded settings")):
        images, warning = repair_image_settings(stored, model_spec(), None)
        page = page_image(stored, target, images)
    assert warning is None
    assert page.render_error is None
    assert page.mime_type == "image/png"
    assert images.request_options() == {"image_detail": "original"}
    image = Image.open(io.BytesIO(base64.b64decode(page.base64)))
    assert image.size == (300, 450)
    if not raw_on_disk:
        assert page.provenance()["source_dpi_x"] == pytest.approx(150)
    # A repair under another model re-derives caps and detail.
    current = settings("gpt-5.4")
    with patch(
        RESOLVE, return_value=(current, "openai", "api_image_processing")
    ) as resolve:
        images, warning = repair_image_settings(stored, model_spec("gpt-5.4"), None)
    assert resolve.called
    assert not images.recorded and warning is not None and "gpt-5.4" in warning
    assert images.request_options() == {}


def test_repair_finds_a_folder_image_by_its_position(tmp_path: Path) -> None:
    folder = tmp_path / "document"
    folder.mkdir()
    (folder / "a.png").write_bytes(scan_bytes(100, 150))
    (folder / "b.png").write_bytes(scan_bytes(300, 450))
    cfg = settings(payload_format="png", original_max_side_px=225)
    stored = document(tmp_path, folder, recorded(cfg))
    with patch(RESOLVE, side_effect=AssertionError("Must use recorded settings")):
        images, _warning = repair_image_settings(stored, model_spec(), None)

    page = page_image(stored, unit(1, "b_tess_preprocessed.png", "other"), images)

    assert images.request_options() == {"image_detail": "original"}
    assert page.mime_type == "image/png" and page.index == 1
    assert page.image_name == "b_tess_preprocessed.png"
    image = Image.open(io.BytesIO(base64.b64decode(page.base64)))
    assert image.size == (150, 225)


def test_repair_batch_uses_the_recorded_detail() -> None:
    plan = build_plan(model_spec(), Settings(), schema_path=None, detail="high")
    body = plan.body(
        BatchRequest(
            "page",
            image_base64="fixture",
            mime_type="image/png",
            image_detail="original",
        )
    )
    image = body["input"][1]["content"][-1]
    assert image["detail"] == "original"
    assert image["image_url"] == "data:image/png;base64,fixture"


def test_repair_without_recorded_settings_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "legacy.png").write_bytes(scan_bytes(100, 150))
    stored = document(tmp_path, folder, None)
    with patch(RESOLVE, return_value=(settings(), "openai", "api_image_processing")):
        images, warning = repair_image_settings(stored, model_spec(), None)
    page = page_image(stored, unit(0, "legacy.png_pre_processed.jpg"), images)
    assert page.render_error is None
    assert warning is not None and "No recorded image settings" in warning
    assert "No recorded image settings" in caplog.text
