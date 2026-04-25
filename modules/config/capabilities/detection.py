"""Capability and provider detection: all lookup logic that traverses the
registry (including OpenRouter passthrough) plus provider/model-type helpers
used throughout the codebase.
"""

from __future__ import annotations

from typing import Optional

from modules.config.capabilities.registry import (
    Capabilities,
    ProviderType,
    _MODEL_REGISTRY,
    _OPENAI_REASONING_BASE,
    _OPENROUTER_BASE,
    _build_caps,
)


def _norm(name: str) -> str:
    return name.strip().lower()


def detect_provider(model_name: str) -> ProviderType:
    """Detect LLM provider from model name.

    Canonical provider detection. All other modules should use this function
    or delegate to it.
    """
    m = _norm(model_name)

    if m.startswith("openrouter/"):
        return "openrouter"
    if "/" in m:
        prefix = m.split("/")[0]
        if prefix in ("openai", "anthropic", "google", "meta", "mistral"):
            return "openrouter"
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if (m.startswith("gemini") or m.startswith("gemma")
            or m.startswith("models/") or "google" in m):
        return "google"
    if any(m.startswith(p) for p in ("gpt", "o1", "o3", "o4", "chatgpt", "text-")):
        return "openai"
    if any(x in m for x in ("llama", "mistral", "mixtral", "qwen", "deepseek")):
        return "openrouter"
    if "/" in m:
        return "openrouter"
    return "unknown"


def detect_capabilities(model_name: str) -> Capabilities:
    """Map a model name to its known capabilities.

    Covers OpenAI, Anthropic, Google, and OpenRouter models via a static
    registry with provider base templates and per-model overrides.
    """
    m = _norm(model_name)

    # --- Static registry lookup (covers OpenAI, Anthropic, Google) ----------
    for prefixes, family, base, overrides in _MODEL_REGISTRY:
        if any(m.startswith(p) for p in prefixes):
            return _build_caps(model_name, family, base, overrides)

    # --- o3 (not o3-mini, not o3-pro) — requires negative-prefix logic ------
    if m == "o3" or (
        m.startswith("o3-")
        and not m.startswith("o3-mini")
        and not m.startswith("o3-pro")
    ):
        return _build_caps(
            model_name, "o3", _OPENAI_REASONING_BASE,
            dict(supports_structured_outputs=False),
        )

    # --- o1 (not o1-mini, not o1-pro) — requires negative-prefix logic ------
    if (
        m == "o1"
        or m.startswith("o1-20")
        or (
            m.startswith("o1")
            and not m.startswith("o1-mini")
            and not m.startswith("o1-pro")
        )
    ):
        return _build_caps(
            model_name, "o1", _OPENAI_REASONING_BASE,
            dict(
                api_preference="either",
                supports_reasoning_effort=False,
                supports_structured_outputs=False,
            ),
        )

    # --- OpenRouter models (dynamic matching on underlying model) -----------
    if m.startswith("openrouter/") or "/" in m:
        underlying = m.split("/")[-1] if "/" in m else m

        if "deepseek" in m:
            is_r1 = "deepseek-r1" in m
            is_terminus = "terminus" in m
            return _build_caps(
                model_name, "openrouter-deepseek", _OPENROUTER_BASE,
                dict(
                    is_reasoning_model=is_r1 or is_terminus,
                    supports_reasoning_effort=True,
                    supports_sampler_controls=not is_r1,
                    supports_top_p=not is_r1,
                ),
            )

        if "gpt-oss" in m:
            return _build_caps(
                model_name, "openrouter-gpt-oss", _OPENROUTER_BASE,
                dict(
                    supports_image_detail=True, default_image_detail="high",
                    is_reasoning_model=True,
                    supports_reasoning_effort=True,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                ),
            )

        if "gpt-5" in m:
            or_overrides = dict(
                supports_image_detail=True, default_image_detail="high",
                is_reasoning_model=True, supports_reasoning_effort=True,
                supports_sampler_controls=False, supports_top_p=False,
            )
            if "gpt-5.4" in m:
                or_overrides["supports_image_detail_original"] = True
            if "gpt-5.3" in m:
                or_overrides["is_reasoning_model"] = False
                or_overrides["supports_reasoning_effort"] = False
                or_overrides["supports_sampler_controls"] = True
                or_overrides["supports_top_p"] = True
            return _build_caps(
                model_name, "openrouter-gpt5", _OPENROUTER_BASE, or_overrides
            )

        if any(x in m for x in (
            "/o1", "/o3", "/o4", "openai/o1", "openai/o3", "openai/o4"
        )):
            return _build_caps(
                model_name, "openrouter-o-series", _OPENROUTER_BASE,
                dict(
                    supports_image_detail=True, default_image_detail="high",
                    is_reasoning_model=True,
                    supports_reasoning_effort=True,
                    supports_image_input="mini" not in m,
                    supports_sampler_controls=False, supports_top_p=False,
                ),
            )

        if "claude" in underlying or "anthropic/" in m:
            return _build_caps(
                model_name, "openrouter-claude", _OPENROUTER_BASE,
                dict(
                    is_reasoning_model=True,
                    supports_reasoning_effort=True,
                    max_context_tokens=200000,
                ),
            )

        if "gemini" in underlying or "google/" in m:
            is_thinking = any(
                x in m for x in (
                    "gemini-2.5", "gemini-3", "gemini-2-5", "gemini-3-"
                )
            )
            return _build_caps(
                model_name, "openrouter-gemini", _OPENROUTER_BASE,
                dict(
                    is_reasoning_model=is_thinking,
                    supports_reasoning_effort=True,
                    supports_media_resolution=True,
                    default_media_resolution="high",
                    max_context_tokens=1000000, max_output_tokens=8192,
                ),
            )

        if "llama" in underlying or "meta/" in m:
            return _build_caps(
                model_name, "openrouter-llama", _OPENROUTER_BASE,
                dict(
                    supports_image_input="vision" in m or "llama-3.2" in m,
                    supports_frequency_penalty=True,
                    supports_presence_penalty=True,
                ),
            )

        if "mistral" in underlying or "mixtral" in m:
            return _build_caps(
                model_name, "openrouter-mistral", _OPENROUTER_BASE,
                dict(supports_image_input="pixtral" in m),
            )

        if "qwen" in underlying or "qwen" in m:
            return _build_caps(
                model_name, "openrouter-qwen", _OPENROUTER_BASE,
                dict(
                    is_reasoning_model=True,
                    supports_reasoning_effort=True,
                    supports_image_input=True,
                    supports_structured_outputs=True,
                    max_context_tokens=131072,
                ),
            )

        return _build_caps(
            model_name, "openrouter", _OPENROUTER_BASE, {}
        )

    # --- Fallback: conservative text-only -----------------------------------
    return Capabilities(
        model=model_name,
        family="unknown",
        provider=detect_provider(model_name),
        supports_responses_api=True,
        supports_chat_completions=True,
        api_preference="responses",
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_developer_messages=True,
        supports_image_input=False,
        supports_image_detail=False,
        default_image_detail="high",
        supports_structured_outputs=True,
        supports_json_mode=True,
        supports_function_calling=True,
        supports_sampler_controls=True,
        supports_top_p=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
    )


# ---------------------------------------------------------------------------
# Model-type helpers (absorbed from the former modules.config.capabilities.detection)
# ---------------------------------------------------------------------------

def detect_model_type(provider: str, model_name: Optional[str] = None) -> str:
    """Detect the underlying model type from provider and model name.

    Allows correct preprocessing even when using models via OpenRouter.
    For example, 'google/gemini-2.5-flash' via OpenRouter should use Google
    config.

    Returns one of: 'google', 'anthropic', 'openai', or 'custom'.
    """
    provider = provider.lower()
    model_name = model_name.lower() if model_name else ""

    # Direct providers take precedence
    if provider == "custom":
        return "custom"
    if provider == "google":
        return "google"
    if provider == "anthropic":
        return "anthropic"
    if provider == "openai":
        return "openai"

    # For OpenRouter or unknown providers, detect from model name
    if model_name:
        if "gemini" in model_name or "google/" in model_name:
            return "google"
        if "claude" in model_name or "anthropic/" in model_name:
            return "anthropic"
        if any(
            x in model_name
            for x in ["gpt-", "o1", "o3", "o4", "openai/"]
        ):
            return "openai"

    return "openai"


def get_image_config_section_name(model_type: str) -> str:
    """Get the image processing config section name for a model type."""
    if model_type == "custom":
        return "custom_image_processing"
    elif model_type == "google":
        return "google_image_processing"
    elif model_type == "anthropic":
        return "anthropic_image_processing"
    else:
        return "api_image_processing"


__all__ = [
    "detect_provider",
    "detect_capabilities",
    "detect_model_type",
    "get_image_config_section_name",
]
