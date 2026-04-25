"""Capability registry: the Capabilities dataclass, provider base dicts, and
the static _MODEL_REGISTRY tuple table. This module holds data; all logic
that traverses the registry lives in `detection.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ImageDetail = Literal["auto", "high", "low", "original"]
ApiPref = Literal["responses", "chat_completions", "either", "langchain"]
ProviderType = Literal[
    "openai", "anthropic", "google", "openrouter", "custom", "unknown"
]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Canonical registry record describing a model's surfaced abilities.

    Single source of truth for capability gating across the repository
    (batch API, LangChain providers, and fail-fast checks all read from the
    same Capabilities instance).

    Key capability flags
    --------------------
    - supports_sampler_controls: master flag; False disables all sampler
      params via LangChain's disabled_params.
    - supports_top_p: granular override for Claude 4.5+ which accepts
      temperature but rejects top_p.
    - supports_structured_outputs: if False, response_format is disabled.
    - supports_reasoning_effort: if True, reasoning controls are enabled.
    - is_reasoning_model: indicates o1, o3, gpt-5, or Claude thinking models.
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
    supports_image_detail_original: bool = False  # Only gpt-5.4+
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


# ---------------------------------------------------------------------------
# Provider-level capability defaults. Each model entry in the registry below
# only needs to declare the fields that *differ* from its provider default.
# ---------------------------------------------------------------------------

_OPENAI_REASONING_BASE: dict[str, Any] = dict(
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

_OPENAI_STANDARD_BASE: dict[str, Any] = dict(
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
    # The Responses API (api_preference="responses") does not accept
    # frequency_penalty / presence_penalty kwargs; passing them raises
    # TypeError from AsyncResponses.parse(). Keep these disabled at the
    # base so LangChain's disabled_params filter drops them before the
    # call. Individual models can re-enable them if they route through
    # Chat Completions instead.
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=128000,
    max_output_tokens=16384,
)

_ANTHROPIC_BASE: dict[str, Any] = dict(
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

_GOOGLE_BASE: dict[str, Any] = dict(
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

_OPENROUTER_BASE: dict[str, Any] = dict(
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
# Static model registry. Each entry is (prefixes, family, base, overrides).
# Order matters: more specific prefixes MUST come before less specific ones.
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: list[tuple[tuple[str, ...], str, dict[str, Any], dict[str, Any]]] = [
    # --- OpenAI GPT-5.x family (reasoning, Responses-native) ---
    (("gpt-5.4-pro",), "gpt-5.4-pro", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=1050000,
        max_output_tokens=128000, supports_image_detail_original=True,
    )),
    (("gpt-5.4-mini",), "gpt-5.4-mini", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
        max_output_tokens=128000,
    )),
    (("gpt-5.4-nano",), "gpt-5.4-nano", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
        max_output_tokens=128000,
    )),
    (("gpt-5.4",), "gpt-5.4", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=1050000,
        max_output_tokens=128000, supports_image_detail_original=True,
    )),
    (("gpt-5.3-codex",), "gpt-5.3-codex", _OPENAI_STANDARD_BASE, dict(
        max_context_tokens=400000, max_output_tokens=128000,
        supports_image_input=False, supports_image_detail=False,
    )),
    (("gpt-5.3",), "gpt-5.3", _OPENAI_STANDARD_BASE, dict(
        max_context_tokens=400000, max_output_tokens=128000,
    )),
    (("gpt-5.2-codex",), "gpt-5.2-codex", _OPENAI_STANDARD_BASE, dict(
        max_context_tokens=400000, max_output_tokens=128000,
        supports_image_input=False, supports_image_detail=False,
    )),
    (("gpt-5.2",), "gpt-5.2", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
        max_output_tokens=128000,
    )),
    (("gpt-5.1",), "gpt-5.1", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
        max_output_tokens=128000,
    )),
    (("gpt-5",), "gpt-5", _OPENAI_REASONING_BASE, dict(
        supports_chat_completions=False, max_context_tokens=400000,
        max_output_tokens=128000,
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
    # Claude 4.7
    (("claude-opus-4-7", "claude-opus-4.7"), "claude-opus-4.7",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_context_tokens=1000000,
        max_output_tokens=128000,
    )),
    # Claude 4.6
    (("claude-opus-4-6", "claude-opus-4.6"), "claude-opus-4.6",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_context_tokens=1000000,
        max_output_tokens=128000,
    )),
    (("claude-sonnet-4-6", "claude-sonnet-4.6"), "claude-sonnet-4.6",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_context_tokens=1000000,
        max_output_tokens=65536,
    )),
    # Claude 4.5
    (("claude-opus-4-5", "claude-opus-4.5"), "claude-opus-4.5",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=65536,
    )),
    (("claude-sonnet-4-5", "claude-sonnet-4.5"), "claude-sonnet-4.5",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=65536,
    )),
    (("claude-haiku-4-5", "claude-haiku-4.5"), "claude-haiku-4.5",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False,
    )),
    # Claude 4.1
    (("claude-opus-4-1", "claude-opus-4.1"), "claude-opus-4.1",
     _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=32768,
    )),
    # Claude 4
    (("claude-sonnet-4",), "claude-sonnet-4", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=16384,
    )),
    (("claude-opus-4",), "claude-opus-4", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=16384,
    )),
    # Claude 3.7 / 3.5 / 3 / fallback
    (("claude-3-7-sonnet", "claude-3.7-sonnet"), "claude-3.7-sonnet",
     _ANTHROPIC_BASE, {}),
    (("claude-3-5-sonnet", "claude-3.5-sonnet"), "claude-3.5-sonnet",
     _ANTHROPIC_BASE, {}),
    (("claude-3-5-haiku", "claude-3.5-haiku"), "claude-3.5-haiku",
     _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False,
    )),
    (("claude-3-opus",), "claude-3-opus", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False,
        max_output_tokens=4096,
    )),
    (("claude-3-sonnet",), "claude-3-sonnet", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False,
        max_output_tokens=4096,
    )),
    (("claude-3-haiku",), "claude-3-haiku", _ANTHROPIC_BASE, dict(
        supports_structured_outputs=False, supports_json_mode=False,
        max_output_tokens=4096,
    )),
    (("claude",), "claude", _ANTHROPIC_BASE, {}),
    # --- Google Gemma (via Gemini API) ---
    (("gemma-4-31b-it",), "gemma-4-31b", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=262144,
    )),
    (("gemma-4-26b-a4b-it",), "gemma-4-26b-moe", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=262144,
    )),
    (("gemma",), "gemma", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=262144,
    )),
    # --- Google Gemini ---
    (("gemini-3.1-pro-preview", "gemini-3-1-pro-preview"), "gemini-3.1-pro",
     _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1000000, max_output_tokens=65536,
    )),
    (("gemini-3.1-flash-lite-preview", "gemini-3-1-flash-lite-preview"),
     "gemini-3.1-flash-lite", _GOOGLE_BASE, dict(
        max_context_tokens=1000000, max_output_tokens=65536,
    )),
    (("gemini-3.1-flash-image-preview", "gemini-3-1-flash-image-preview"),
     "gemini-3.1-flash-image", _GOOGLE_BASE, dict(
        max_context_tokens=128000, max_output_tokens=32768,
    )),
    (("gemini-3-pro-image-preview",), "gemini-3-pro-image",
     _GOOGLE_BASE, dict(
        max_context_tokens=65536, max_output_tokens=32768,
    )),
    (("gemini-3-flash", "gemini-3.0-flash"), "gemini-3-flash",
     _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1048576, max_output_tokens=65536,
    )),
    (("gemini-3-pro", "gemini-3.0-pro"), "gemini-3-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=2000000, max_output_tokens=65536,
    )),
    (("gemini-3-preview", "gemini-3.0-preview"), "gemini-3-preview",
     _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1048576, max_output_tokens=65536,
    )),
    (("gemini-3", "gemini-3.0"), "gemini-3", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1000000, max_output_tokens=65536,
    )),
    (("gemini-2.5-pro", "gemini-2-5-pro"), "gemini-2.5-pro",
     _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=2000000, max_output_tokens=65536,
    )),
    (("gemini-2.5-flash-lite", "gemini-2-5-flash-lite"),
     "gemini-2.5-flash-lite", _GOOGLE_BASE, dict(
        max_context_tokens=1048576, max_output_tokens=32768,
    )),
    # gemini-2.5-flash is a thinking model but does NOT accept an explicit
    # thinking_level / thinking_budget via the SDK — the API rejects those
    # requests with "Thinking level is not supported for this model." Keep
    # reasoning_effort support disabled so the sampler-filter drops it.
    (("gemini-2.5-flash", "gemini-2-5-flash"), "gemini-2.5-flash",
     _GOOGLE_BASE, dict(
        is_reasoning_model=False, supports_reasoning_effort=False,
        max_context_tokens=1000000, max_output_tokens=32768,
    )),
    (("gemini-2.0-flash", "gemini-2-flash", "gemini-2.0"),
     "gemini-2.0-flash", _GOOGLE_BASE, {}),
    (("gemini-1.5-pro", "gemini-1-5-pro"), "gemini-1.5-pro",
     _GOOGLE_BASE, dict(max_context_tokens=2000000)),
    (("gemini-1.5-flash", "gemini-1-5-flash"), "gemini-1.5-flash",
     _GOOGLE_BASE, {}),
    (("gemini",), "gemini", _GOOGLE_BASE, {}),
]


def _build_caps(
    model_name: str, family: str, base: dict[str, Any], overrides: dict[str, Any]
) -> Capabilities:
    """Merge *base* defaults with *overrides* and return a Capabilities."""
    merged = {**base, **overrides}
    merged["model"] = model_name
    merged["family"] = family
    return Capabilities(**merged)


__all__ = [
    "ImageDetail",
    "ApiPref",
    "ProviderType",
    "Capabilities",
]
