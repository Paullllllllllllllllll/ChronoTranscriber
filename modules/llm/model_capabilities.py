# modules/llm/model_capabilities.py
"""Model capability detection and feature gating.

Centralised, registry-based capability detection for all LLM providers.
Ported from ChronoMiner's elegant registry pattern and extended with
ChronoTranscriber-specific fields (max_output_tokens, media_resolution,
fine-grained sampler flags for the Anthropic top_p nuance).

Used by:
- LangChain provider classes (disabled_params, parameter gating)
- Batch API processing (OpenAI-specific, bypasses LangChain)
- ensure_image_support() fail-fast check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ImageDetail = Literal["auto", "high", "low"]
ApiPref = Literal["responses", "chat_completions", "either", "langchain"]
ProviderType = Literal["openai", "anthropic", "google", "openrouter", "unknown"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """
    Canonical registry record describing a model's surfaced abilities.

    This is the **single source of truth** for capability gating across the
    entire repository — batch API, LangChain providers, and fail-fast checks
    all read from the same ``Capabilities`` instance.

    Key capability flags
    --------------------
    - ``supports_sampler_controls``: master flag; False disables all sampler
      params via LangChain's ``disabled_params``
    - ``supports_top_p``: granular override for Claude 4.5+ which accepts
      temperature but rejects top_p
    - ``supports_structured_outputs``: if False, ``response_format`` is disabled
    - ``supports_reasoning_effort``: if True, reasoning controls are enabled
    - ``is_reasoning_model``: indicates o1, o3, gpt-5, or Claude thinking models
    """

    # Core identity
    model: str
    family: str
    provider: ProviderType = "openai"

    # API surface
    supports_responses_api: bool = True
    supports_chat_completions: bool = True
    api_preference: ApiPref = "responses"

    # Reasoning / control
    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    supports_developer_messages: bool = True

    # Vision / images
    supports_image_input: bool = False
    supports_image_detail: bool = False
    default_image_detail: str = "high"
    supports_media_resolution: bool = False   # Google-style media_resolution
    default_media_resolution: str = "high"

    # Structured outputs
    supports_structured_outputs: bool = True
    supports_json_mode: bool = True
    supports_function_calling: bool = True

    # Sampler controls (fine-grained for the Claude 4.5 top_p nuance)
    supports_sampler_controls: bool = True    # master flag
    supports_top_p: bool = True               # separate for Claude 4.5+
    supports_frequency_penalty: bool = True   # False for Anthropic/Google
    supports_presence_penalty: bool = True    # False for Anthropic/Google

    # Context / output limits
    max_context_tokens: int = 128000
    max_output_tokens: int = 4096


def _norm(name: str) -> str:
    return name.strip().lower()


def detect_provider(model_name: str) -> ProviderType:
    """
    Detect LLM provider from model name.

    This is the canonical provider detection function.  All other modules
    should use this function or delegate to it.
    """
    m = _norm(model_name)

    if m.startswith("openrouter/") or "/" in m:
        return "openrouter"
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if m.startswith("gemini") or "google" in m:
        return "google"
    if any(m.startswith(p) for p in ("gpt", "o1", "o3", "o4", "text-")):
        return "openai"
    return "unknown"


# ---------------------------------------------------------------------------
# Provider-level capability defaults.  Each model entry in the registry below
# only needs to declare the fields that *differ* from its provider default.
# ---------------------------------------------------------------------------

_OPENAI_REASONING_BASE: dict = dict(
    provider="openai",
    supports_responses_api=True,
    supports_chat_completions=True,
    api_preference="responses",
    is_reasoning_model=True,
    supports_reasoning_effort=True,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=True,
    default_image_detail="high",
    supports_structured_outputs=True,
    supports_json_mode=True,
    supports_function_calling=True,
    supports_sampler_controls=False,
    supports_top_p=False,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=200000,
    max_output_tokens=100000,
)

_OPENAI_STANDARD_BASE: dict = dict(
    provider="openai",
    supports_responses_api=True,
    supports_chat_completions=True,
    api_preference="responses",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=True,
    default_image_detail="high",
    supports_structured_outputs=True,
    supports_json_mode=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    supports_top_p=True,
    supports_frequency_penalty=True,
    supports_presence_penalty=True,
    max_context_tokens=128000,
    max_output_tokens=16384,
)

_ANTHROPIC_BASE: dict = dict(
    provider="anthropic",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    default_image_detail="auto",
    supports_structured_outputs=True,
    supports_json_mode=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    supports_top_p=True,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=200000,
    max_output_tokens=8192,
)

_GOOGLE_BASE: dict = dict(
    provider="google",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    default_image_detail="auto",
    supports_media_resolution=True,
    default_media_resolution="high",
    supports_structured_outputs=True,
    supports_json_mode=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    supports_top_p=True,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=1000000,
    max_output_tokens=8192,
)

_OPENROUTER_BASE: dict = dict(
    provider="openrouter",
    supports_responses_api=False,
    supports_chat_completions=True,
    api_preference="langchain",
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_developer_messages=True,
    supports_image_input=True,
    supports_image_detail=False,
    default_image_detail="auto",
    supports_structured_outputs=True,
    supports_json_mode=True,
    supports_function_calling=True,
    supports_sampler_controls=True,
    supports_top_p=True,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=128000,
    max_output_tokens=4096,
)


# ---------------------------------------------------------------------------
# Static model registry.  Each entry is (prefixes, family, base, overrides).
# Order matters: more specific prefixes MUST come before less specific ones.
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: list[tuple[tuple[str, ...], str, dict, dict]] = [
    # --- OpenAI GPT-5.x family (reasoning, Responses-native) ---
    (("gpt-5.2",), "gpt-5.2", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000, max_output_tokens=128000,
    )),
    (("gpt-5.1",), "gpt-5.1", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000, max_output_tokens=128000,
    )),
    (("gpt-5",), "gpt-5", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000, max_output_tokens=128000,
    )),
    # --- OpenAI o-series reasoning models ---
    (("o4-mini", "o4"), "o4-mini", _OPENAI_REASONING_BASE, {}),
    (("o3-pro",), "o3-pro", _OPENAI_REASONING_BASE, {}),
    (("o3-mini",), "o3-mini", _OPENAI_REASONING_BASE, dict(
        supports_image_input=False, supports_image_detail=False,
    )),
    # o3 (not o3-mini, not o3-pro) — requires negative-prefix logic below
    (("o1-pro",), "o1-pro", _OPENAI_REASONING_BASE, dict(
        supports_structured_outputs=False,
    )),
    (("o1-mini",), "o1-mini", _OPENAI_REASONING_BASE, dict(
        supports_responses_api=False, api_preference="chat_completions",
        supports_reasoning_effort=False, supports_developer_messages=False,
        supports_image_input=False, supports_image_detail=False,
        supports_structured_outputs=False, supports_json_mode=False,
        supports_function_calling=False,
        max_context_tokens=128000, max_output_tokens=65536,
    )),
    # o1 (not o1-mini, not o1-pro) — requires negative-prefix logic below
    # --- OpenAI GPT-4o / GPT-4.1 (standard, non-reasoning) ---
    (("gpt-4o",), "gpt-4o", _OPENAI_STANDARD_BASE, {}),
    (("gpt-4.1",), "gpt-4.1", _OPENAI_STANDARD_BASE, dict(
        max_context_tokens=1000000, max_output_tokens=32768,
    )),
    # --- Anthropic Claude models (most-specific first) ---
    # Claude 4.6 (thinking, no top_p)
    (("claude-opus-4-6", "claude-opus-4.6"), "claude-opus-4.6", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=128000,
    )),
    (("claude-sonnet-4-6", "claude-sonnet-4.6"), "claude-sonnet-4.6", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=65536,
    )),
    # Claude 4.5 (thinking, no top_p)
    (("claude-opus-4-5", "claude-opus-4.5"), "claude-opus-4.5", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=32768,
    )),
    (("claude-sonnet-4-5", "claude-sonnet-4.5"), "claude-sonnet-4.5", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=16384,
    )),
    (("claude-haiku-4-5", "claude-haiku-4.5"), "claude-haiku-4.5", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False,
    )),
    # Claude 4.1
    (("claude-opus-4-1", "claude-opus-4.1"), "claude-opus-4.1", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=16384,
    )),
    # Claude 4 (sonnet = non-reasoning; opus = reasoning)
    (("claude-sonnet-4",), "claude-sonnet-4", _ANTHROPIC_BASE, {}),
    (("claude-opus-4",), "claude-opus-4", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=16384,
    )),
    # Claude 3.7
    (("claude-3-7-sonnet", "claude-3.7-sonnet"), "claude-3.7-sonnet", _ANTHROPIC_BASE, {}),
    # Claude 3.5
    (("claude-3-5-sonnet", "claude-3.5-sonnet"), "claude-3.5-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-5-haiku", "claude-3.5-haiku"), "claude-3.5-haiku", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False,
    )),
    # Claude 3
    (("claude-3-opus",), "claude-3-opus", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False, max_output_tokens=4096,
    )),
    (("claude-3-sonnet",), "claude-3-sonnet", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False, max_output_tokens=4096,
    )),
    (("claude-3-haiku",), "claude-3-haiku", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False, max_output_tokens=4096,
    )),
    # Claude generic fallback
    (("claude",), "claude", _ANTHROPIC_BASE, {}),
    # --- Google Gemini models (most-specific first) ---
    (("gemini-3-flash", "gemini-3.0-flash"), "gemini-3-flash", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1048576, max_output_tokens=65536,
    )),
    (("gemini-3-pro", "gemini-3.0-pro"), "gemini-3-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=2000000, max_output_tokens=65536,
    )),
    (("gemini-3-preview", "gemini-3.0-preview"), "gemini-3-preview", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1048576, max_output_tokens=65536,
    )),
    (("gemini-3", "gemini-3.0"), "gemini-3", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1000000, max_output_tokens=65536,
    )),
    (("gemini-2.5-pro", "gemini-2-5-pro"), "gemini-2.5-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=2000000, max_output_tokens=65536,
    )),
    (("gemini-2.5-flash-lite", "gemini-2-5-flash-lite"), "gemini-2.5-flash-lite", _GOOGLE_BASE, dict(
        max_context_tokens=1048576, max_output_tokens=32768,
    )),
    (("gemini-2.5-flash", "gemini-2-5-flash"), "gemini-2.5-flash", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1000000, max_output_tokens=32768,
    )),
    (("gemini-2.0-flash", "gemini-2-flash", "gemini-2.0"), "gemini-2.0-flash", _GOOGLE_BASE, {}),
    (("gemini-1.5-pro", "gemini-1-5-pro"), "gemini-1.5-pro", _GOOGLE_BASE, dict(
        max_context_tokens=2000000,
    )),
    (("gemini-1.5-flash", "gemini-1-5-flash"), "gemini-1.5-flash", _GOOGLE_BASE, {}),
    (("gemini",), "gemini", _GOOGLE_BASE, {}),
]


def _build_caps(model_name: str, family: str, base: dict, overrides: dict) -> Capabilities:
    """Merge *base* defaults with *overrides* and return a Capabilities instance."""
    merged = {**base, **overrides}
    merged["model"] = model_name
    merged["family"] = family
    return Capabilities(**merged)


def detect_capabilities(model_name: str) -> Capabilities:
    """
    Map a model name to its known capabilities.

    Covers OpenAI, Anthropic, Google, and OpenRouter models via a static
    registry with provider base templates and per-model overrides.
    """
    m = _norm(model_name)

    # --- Static registry lookup (covers OpenAI, Anthropic, Google) ----------
    for prefixes, family, base, overrides in _MODEL_REGISTRY:
        if any(m.startswith(p) for p in prefixes):
            return _build_caps(model_name, family, base, overrides)

    # --- o3 (not o3-mini, not o3-pro) — requires negative-prefix logic ------
    if m == "o3" or (m.startswith("o3-") and not m.startswith("o3-mini") and not m.startswith("o3-pro")):
        return _build_caps(model_name, "o3", _OPENAI_REASONING_BASE, dict(
            supports_structured_outputs=False,
        ))

    # --- o1 (not o1-mini, not o1-pro) — requires negative-prefix logic ------
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini") and not m.startswith("o1-pro")):
        return _build_caps(model_name, "o1", _OPENAI_REASONING_BASE, dict(
            api_preference="either",
            supports_reasoning_effort=False,
            supports_structured_outputs=False,
        ))

    # --- OpenRouter models (dynamic matching on underlying model) -----------
    if m.startswith("openrouter/") or "/" in m:
        underlying = m.split("/")[-1] if "/" in m else m

        # DeepSeek via OpenRouter
        if "deepseek" in m:
            is_r1 = "deepseek-r1" in m
            is_terminus = "terminus" in m
            return _build_caps(model_name, "openrouter-deepseek", _OPENROUTER_BASE, dict(
                is_reasoning_model=is_r1 or is_terminus,
                supports_reasoning_effort=True,
                supports_sampler_controls=not is_r1,
                supports_top_p=not is_r1,
            ))

        # GPT-OSS via OpenRouter
        if "gpt-oss" in m:
            return _build_caps(model_name, "openrouter-gpt-oss", _OPENROUTER_BASE, dict(
                supports_image_detail=True, default_image_detail="high",
                is_reasoning_model=True, supports_reasoning_effort=True,
                supports_frequency_penalty=True, supports_presence_penalty=True,
            ))

        # GPT-5 via OpenRouter
        if "gpt-5" in m:
            return _build_caps(model_name, "openrouter-gpt5", _OPENROUTER_BASE, dict(
                supports_image_detail=True, default_image_detail="high",
                is_reasoning_model=True, supports_reasoning_effort=True,
                supports_sampler_controls=False, supports_top_p=False,
            ))

        # o-series via OpenRouter
        if any(x in m for x in ("/o1", "/o3", "/o4", "openai/o1", "openai/o3", "openai/o4")):
            return _build_caps(model_name, "openrouter-o-series", _OPENROUTER_BASE, dict(
                supports_image_detail=True, default_image_detail="high",
                is_reasoning_model=True, supports_reasoning_effort=True,
                supports_image_input="mini" not in m,
                supports_sampler_controls=False, supports_top_p=False,
            ))

        # Claude via OpenRouter
        if "claude" in underlying or "anthropic/" in m:
            return _build_caps(model_name, "openrouter-claude", _OPENROUTER_BASE, dict(
                is_reasoning_model=True, supports_reasoning_effort=True,
                max_context_tokens=200000,
            ))

        # Gemini via OpenRouter
        if "gemini" in underlying or "google/" in m:
            is_thinking = any(x in m for x in ("gemini-2.5", "gemini-3", "gemini-2-5", "gemini-3-"))
            return _build_caps(model_name, "openrouter-gemini", _OPENROUTER_BASE, dict(
                is_reasoning_model=is_thinking, supports_reasoning_effort=True,
                supports_media_resolution=True, default_media_resolution="high",
                max_context_tokens=1000000, max_output_tokens=8192,
            ))

        # Llama via OpenRouter
        if "llama" in underlying or "meta/" in m:
            return _build_caps(model_name, "openrouter-llama", _OPENROUTER_BASE, dict(
                supports_image_input="vision" in m or "llama-3.2" in m,
                supports_frequency_penalty=True, supports_presence_penalty=True,
            ))

        # Mistral via OpenRouter
        if "mistral" in underlying or "mixtral" in m:
            return _build_caps(model_name, "openrouter-mistral", _OPENROUTER_BASE, dict(
                supports_image_input="pixtral" in m,
            ))

        # Generic OpenRouter fallback
        return _build_caps(model_name, "openrouter", _OPENROUTER_BASE, {})

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


# -------- Safety gate (fail-fast) --------

class CapabilityError(ValueError):
    """Raised when a selected model is incompatible with the configured pipeline."""


def ensure_image_support(model_name: str, images_required: bool) -> None:
    """
    Fail fast if the pipeline intends to send images but the model can't accept them.

    Parameters
    ----------
    model_name : str
        Selected model id/alias from configuration.
    images_required : bool
        True if the current pipeline path will submit image inputs (our OCR path).
    """
    caps = detect_capabilities(model_name)
    if images_required and not caps.supports_image_input:
        raise CapabilityError(
            "The current pipeline sends image inputs, but the selected model "
            f"'{model_name}' does not support image inputs. Choose an image-capable model "
            "(e.g., gpt-5, o1, o3, gpt-4o, gpt-4.1) or set 'expects_image_inputs: false' "
            "in model_config.yaml to run a text-only flow."
        )
