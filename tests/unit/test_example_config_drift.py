"""Keep the public provider sections in the shared configuration layout."""

from pathlib import Path

import yaml

from modules.images.settings import resolved_settings


def test_example_config_drift() -> None:
    path = Path(__file__).parents[2] / "config/image_processing_config.example.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    providers = ["api", "anthropic", "google", "custom"]
    assert list(config)[:6] == [
        "render_strategy",
        "max_pixels_per_page",
        *(f"{name}_image_processing" for name in providers),
    ]
    common = [
        "target_dpi",
        "native_fallback_dpi",
        "payload_format",
        "max_image_bytes",
        "grayscale_conversion",
        "handle_transparency",
        "jpeg_quality",
    ]
    for name in providers:
        section = f"{name}_image_processing"
        detail = (
            ["media_resolution"]
            if name == "google"
            else ([] if name == "anthropic" else ["llm_detail"])
        )
        assert list(config[section]) == common + detail + [
            "resize_profile",
            "low_max_side_px",
            "high_target_box",
        ]
        provider = "openai" if name == "api" else name
        model = "claude-opus-5" if name == "anthropic" else "gpt-6-astra"
        resolved_settings(
            config[section],
            provider,
            model,
            provider,
            section,
            config["max_pixels_per_page"],
            config["render_strategy"],
        )
    assert [config[f"{name}_image_processing"]["target_dpi"] for name in providers] == [
        "native",
        "native",
        300,
        300,
    ]
    assert [
        config[f"{name}_image_processing"]["jpeg_quality"] for name in providers
    ] == [
        95,
        95,
        95,
        90,
    ]
