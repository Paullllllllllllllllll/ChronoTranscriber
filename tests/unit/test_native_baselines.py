"""Numeric payload baselines captured before native image support."""

import io
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

from modules.images.page_stream import (
    load_image_payload,
    render_single_pdf_page_payload,
)


def numeric_config(provider: str) -> dict[str, Any]:
    cfg = {
        "target_dpi": 300,
        "grayscale_conversion": provider != "custom",
        "handle_transparency": True,
        "jpeg_quality": 90 if provider == "custom" else 100,
        "llm_detail": "original" if provider == "openai" else "high",
        "media_resolution": "high",
        "resize_profile": "auto" if provider == "anthropic" else "high",
        "low_max_side_px": 2500 if provider == "custom" else 512,
        "high_target_box": [2500, 2500] if provider == "custom" else [768, 1536],
        "original_max_side_px": 6000,
        "original_max_pixels": 10240000,
        "high_max_side_px": 2576,
    }
    return cfg


def fixture_files(folder: Path) -> tuple[Path, Path]:
    image = Image.new("RGB", (240, 360), "white")
    image.putdata(
        [
            ((x * 7) % 256, (y * 3) % 256, (x + y) % 256)
            for y in range(360)
            for x in range(240)
        ]
    )
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    image_path = folder / "fixture.png"
    image_path.write_bytes(raw.getvalue())
    pdf_path = folder / "fixture.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=144, height=216)
        page.insert_image(page.rect, stream=raw.getvalue())
        page.insert_text((10, 20), "Synthetic fixture")
        doc.save(pdf_path)
    return pdf_path, image_path


_COMMON = (
    "7ece63e4894f90a9168b0f622a6dd9e77a3f544bdc9d84d85f06b5a94837fa9c",
    "e9982b2e71d89dd1aea443bbafa3256cb45b2e3230ca556f0f5643b15b1cea29",
)
_GOOGLE = (
    "8cc0e4ebe0a60aa535795c393937f44e30db9a7338bfee0cc6ad83ceaf389a3a",
    "2204dbc42baef9f11afe5631e4b524a449a86d592034fc485edf1d2f48aee871",
)
_CUSTOM = (
    "6db83ea1b1e863585251c0a9e33fbef477c7c3cacc28712a0d8117680ede1a4a",
    "231a6e56a53a2ba11bc2aa15c9f408fac2c4264e2879c1aae4a67867f47a9243",
)
BASELINES = {
    (provider, strategy): hashes
    for provider, hashes in (
        ("openai", _COMMON),
        ("anthropic", _COMMON),
        ("google", _GOOGLE),
        ("custom", _CUSTOM),
    )
    for strategy in ("direct", "supersample")
}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "custom"])
@pytest.mark.parametrize("strategy", ["direct", "supersample"])
def test_numeric_payload_baseline(tmp_path: Path, provider: str, strategy: str) -> None:
    pdf, image = fixture_files(tmp_path)
    from modules.images.settings import resolved_settings

    cfg = numeric_config(provider)
    model = "claude-opus-5" if provider == "anthropic" else "gpt-5.6-luna"
    cfg = resolved_settings(
        cfg,
        provider,
        model,
        provider,
        f"{provider}_image_processing",
        24000000,
        strategy,
    )
    pdf_payload = render_single_pdf_page_payload(
        pdf,
        0,
        target_dpi=300,
        img_cfg=cfg,
        model_type=provider,
        max_pixels=24000000,
        render_strategy=strategy,
    )
    image_payload = load_image_payload(image, 0, img_cfg=cfg, model_type=provider)
    assert (pdf_payload.sha256, image_payload.sha256) == BASELINES[provider, strategy]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        pdf, image = fixture_files(Path(temporary))
        for provider in ("openai", "anthropic", "google", "custom"):
            for strategy in ("direct", "supersample"):
                cfg = numeric_config(provider)
                payload = render_single_pdf_page_payload(
                    pdf,
                    0,
                    target_dpi=300,
                    img_cfg=cfg,
                    model_type=provider,
                    max_pixels=24000000,
                    render_strategy=strategy,
                )
                source = load_image_payload(image, 0, img_cfg=cfg, model_type=provider)
                print(repr((provider, strategy)), repr((payload.sha256, source.sha256)))
