# modules/model_capabilities.py
"""Model capability detection and feature gating.

This module provides capability detection for OpenAI models, used primarily for:
- Batch API processing (OpenAI-specific)
- Legacy synchronous processing

Note:
    For LangChain-based synchronous transcription, capability guarding is now
    handled by LangChain's `disabled_params` feature in the provider classes.
    See `modules/llm/providers/openai_provider.py` for the LangChain approach.
    
    LangChain handles:
    - Parameter filtering via `disabled_params` (temperature, top_p, etc.)
    - Automatic retry with exponential backoff
    - Structured output parsing
    
    This module is still used for:
    - Batch API processing (which bypasses LangChain)
    - ensure_image_support() fail-fast check
    - Legacy code compatibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ImageDetail = Literal["auto", "high", "low"]
ApiPref = Literal["responses", "chat_completions", "either"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """
    Canonical registry record describing a model's surfaced abilities in the repo.

    Notes
    -----
    - This centralises feature gating for the Responses API payload builders (sync + batch).
    - "Reasoning" here means *model family*, not whether we send a GPT-5 `reasoning` parameter.
      The public `reasoning`/`text.verbosity` controls should only be sent for GPT-5.
    """

    # Core identity
    model: str
    family: str  # e.g., "gpt-5", "o1", "o1-mini", "o3", "o3-mini", "gpt-4o", "gpt-4.1"

    # API surface
    supports_responses_api: bool
    supports_chat_completions: bool
    api_preference: ApiPref = "responses"

    # Reasoning / control
    is_reasoning_model: bool = False
    # Whether to attach the public GPT-5 `reasoning` / `text.verbosity` controls
    supports_reasoning_effort: bool = False
    # Developer messages/system aliasing nuances (currently informative)
    supports_developer_messages: bool = False

    # Vision / images
    supports_image_input: bool = False
    supports_image_detail: bool = True
    default_ocr_detail: ImageDetail = "high"

    # Structure / tools
    supports_structured_outputs: bool = False
    supports_function_calling: bool = False

    # Sampler knobs (temperature, top_p, penalties)
    supports_sampler_controls: bool = True  # False for reasoning families & GPT-5


def _norm(name: str) -> str:
    return name.strip().lower()


def detect_capabilities(model_name: str) -> Capabilities:
    """
    Map a model name to its known capabilities (conservative defaults).
    Families covered:
      - GPT-5 (reasoning)
      - o3 (reasoning, full)
      - o3-mini (small reasoning; NO vision)
      - o1 (reasoning, full; vision)
      - o1-mini (small reasoning; NO vision; Chat Completions oriented)
      - gpt-4o*, gpt-4.1* (non-reasoning; vision; structured outputs; sampler controls)
    """
    m = _norm(model_name)

    # ---- GPT-5 (Responses-native, vision, structured outputs, NO sampler controls) ----
    if m.startswith("gpt-5"):
        return Capabilities(
            model=model_name,
            family="gpt-5",
            supports_responses_api=True,
            supports_chat_completions=False,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=True,      # Attach GPT-5 reasoning/text controls
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # ---- o3 (full reasoning; vision; avoid public reasoning param; no sampler controls) ----
    if m == "o3" or m.startswith("o3-20") or (m.startswith("o3") and not m.startswith("o3-mini")):
        return Capabilities(
            model=model_name,
            family="o3",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=False,      # Do NOT send GPT-5 'reasoning' block
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=False,    # Keep off by default for o-series
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # ---- o3-mini (small reasoning; NO vision; Responses OK for text) ----
    if m.startswith("o3-mini"):
        return Capabilities(
            model=model_name,
            family="o3-mini",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=False,           # no vision
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=True,     # text-side dev features ok
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # ---- o1 (full reasoning; vision; avoid sampler controls) ----
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini")):
        return Capabilities(
            model=model_name,
            family="o1",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="either",
            is_reasoning_model=True,
            supports_reasoning_effort=False,      # Do NOT send GPT-5 'reasoning' block
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=False,    # keep conservative
            supports_function_calling=True,
            supports_sampler_controls=False,
        )

    # ---- o1-mini (small reasoning; NO vision; prefer Chat Completions for text-only) ----
    if m.startswith("o1-mini"):
        return Capabilities(
            model=model_name,
            family="o1-mini",
            supports_responses_api=False,         # treat as not Responses-first
            supports_chat_completions=True,
            api_preference="chat_completions",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
            supports_developer_messages=False,
            supports_image_input=False,           # no vision
            supports_image_detail=False,
            default_ocr_detail="high",
            supports_structured_outputs=False,
            supports_function_calling=False,
            supports_sampler_controls=False,
        )

    # ---- GPT-4o family (multimodal; structured outputs; sampler controls) ----
    if m.startswith("gpt-4o"):
        return Capabilities(
            model=model_name,
            family="gpt-4o",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
        )

    # ---- GPT-4.1 family (multimodal; structured outputs; sampler controls) ----
    if m.startswith("gpt-4.1"):
        return Capabilities(
            model=model_name,
            family="gpt-4.1",
            supports_responses_api=True,
            supports_chat_completions=True,
            api_preference="responses",
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_developer_messages=True,
            supports_image_input=True,
            supports_image_detail=True,
            default_ocr_detail="high",
            supports_structured_outputs=True,
            supports_function_calling=True,
            supports_sampler_controls=True,
        )

    # ---- Fallback (conservative text-only) ----
    return Capabilities(
        model=model_name,
        family="unknown",
        supports_responses_api=True,
        supports_chat_completions=True,
        api_preference="responses",
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_developer_messages=True,
        supports_image_input=False,
        supports_image_detail=False,
        default_ocr_detail="high",
        supports_structured_outputs=True,
        supports_function_calling=True,
        supports_sampler_controls=True,
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
        # Explicitly highlight known non-vision small reasoning models.
        raise CapabilityError(
            "The current pipeline sends image inputs, but the selected model "
            f"'{model_name}' does not support image inputs. Choose an image-capable model "
            "(e.g., gpt-5, o1, o3, gpt-4o, gpt-4.1) or set 'expects_image_inputs: false' "
            "in model_config.yaml to run a text-only flow."
        )
