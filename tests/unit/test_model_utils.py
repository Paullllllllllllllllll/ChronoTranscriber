from __future__ import annotations

import pytest

from modules.processing.model_utils import detect_model_type, get_image_config_section_name


@pytest.mark.unit
def test_detect_model_type_direct_provider_takes_precedence() -> None:
    assert detect_model_type("google", "gpt-4o") == "google"
    assert detect_model_type("anthropic", "gpt-4o") == "anthropic"
    assert detect_model_type("openai", "claude-3") == "openai"


@pytest.mark.unit
def test_detect_model_type_openrouter_infers_from_model_name() -> None:
    assert detect_model_type("openrouter", "google/gemini-2.5-flash") == "google"
    assert detect_model_type("openrouter", "anthropic/claude-3-5-sonnet") == "anthropic"
    assert detect_model_type("openrouter", "openai/gpt-4o") == "openai"


@pytest.mark.unit
def test_detect_model_type_defaults_to_openai() -> None:
    assert detect_model_type("openrouter", None) == "openai"
    assert detect_model_type("unknown", "") == "openai"


@pytest.mark.unit
def test_get_image_config_section_name() -> None:
    assert get_image_config_section_name("google") == "google_image_processing"
    assert get_image_config_section_name("anthropic") == "anthropic_image_processing"
    assert get_image_config_section_name("openai") == "api_image_processing"
    assert get_image_config_section_name("anything") == "api_image_processing"
