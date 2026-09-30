"""Connect portable image policy to the application capability registry."""

from typing import Any

from modules.config.capabilities import detect_capabilities
from modules.images.native import model_image_cap, validate_image_settings
from modules.images.pipeline import ImageProcessor


def resolved_settings(
    cfg: dict[str, Any],
    provider: str,
    model: str,
    model_type: str,
    section: str,
    max_pixels: int,
    strategy: str,
) -> dict[str, Any]:
    provider = provider.lower()
    validate_image_settings(cfg, section)
    result = {
        "target_dpi": 300,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 0,
        "jpeg_quality": 95,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "resize_profile": "auto",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
        **cfg,
    }
    caps = detect_capabilities(model)
    detail = (
        (
            str(result.get("llm_detail", "high"))
            if provider == "openrouter"
            else ImageProcessor.resolve_detail(result, model_type)
        )
        .lower()
        .strip()
    )
    if provider == "openrouter" and not caps.supports_image_detail:
        # No detail parameter reaches the model, so size by the section's own
        # profile (Anthropic tier, Google resolution) rather than llm_detail.
        detail = ImageProcessor.resolve_detail(result, model_type).lower()
    elif provider in ("openai", "openrouter"):
        allowed = {"low", "high"}
        if provider == "openai" and caps.supports_image_detail_original:
            allowed.add("original")
        if detail not in allowed:
            detail = caps.default_image_detail
    image_caps = (
        detect_capabilities(model.split("/")[-1])
        if provider == "openrouter" and model_type == "anthropic"
        else caps
    )
    result.update(
        model_name=model,
        provider=provider,
        resolved_detail=detail,
        image_original_patch_cap_30k=(
            caps.image_original_patch_cap_30k and provider == "openai"
        ),
        image_high_res_tier=image_caps.image_high_res_tier,
        max_pixels_per_page=max_pixels,
        render_strategy=strategy,
    )
    cap = model_image_cap(model_type, model, detail, result)
    result["cap_policy"] = cap.policy if cap else "profile-v1"
    if (
        result["target_dpi"] == "native"
        and not cap
        and (result["resize_profile"] == "none")
    ):
        raise ValueError(f"Invalid resize_profile in {section}: native needs a bound")
    return result
