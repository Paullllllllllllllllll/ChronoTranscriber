"""Unit tests for modules/llm/model_capabilities.py.

Tests model capability detection and feature support.
"""

from __future__ import annotations

import pytest

from modules.config.capabilities import (
    Capabilities,
    CapabilityError,
    detect_capabilities,
    ensure_image_support,
)


class TestCapabilities:
    """Tests for Capabilities dataclass."""

    @pytest.mark.unit
    def test_gpt4o_capabilities(self) -> None:
        """Test GPT-4o capability values."""
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_image_input is True
        assert caps.supports_structured_outputs is True
        assert caps.supports_sampler_controls is True
        assert caps.is_reasoning_model is False

    @pytest.mark.unit
    def test_reasoning_model_capabilities(self) -> None:
        """Test reasoning model capabilities."""
        caps = detect_capabilities("o1")
        assert caps.is_reasoning_model is True
        assert caps.supports_sampler_controls is False


class TestDetectCapabilities:
    """Tests for detect_capabilities function."""

    @pytest.mark.unit
    def test_gpt4o_capabilities(self) -> None:
        """Test capabilities for GPT-4o model."""
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_image_input is True
        assert caps.supports_structured_outputs is True
        assert caps.supports_sampler_controls is True

    @pytest.mark.unit
    def test_o1_reasoning_model(self) -> None:
        """Test capabilities for o1 reasoning model."""
        caps = detect_capabilities("o1")
        assert caps.is_reasoning_model is True
        assert caps.supports_sampler_controls is False

    @pytest.mark.unit
    def test_o3_reasoning_model(self) -> None:
        """Test capabilities for o3 reasoning model."""
        caps = detect_capabilities("o3")
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_unknown_model_defaults(self) -> None:
        """Test that unknown model gets default capabilities."""
        caps = detect_capabilities("unknown-model-xyz")
        # Should return sensible defaults
        assert isinstance(caps, Capabilities)
        assert caps.family == "unknown"

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Test that model name matching is case-insensitive."""
        caps1 = detect_capabilities("gpt-4o")
        caps2 = detect_capabilities("GPT-4O")
        assert caps1.supports_image_input == caps2.supports_image_input


class TestEnsureImageSupport:
    """Tests for ensure_image_support function."""

    @pytest.mark.unit
    def test_vision_model_passes(self) -> None:
        """Test that vision-capable model doesn't raise."""
        ensure_image_support("gpt-4o", images_required=True)  # Should not raise

    @pytest.mark.unit
    def test_non_vision_model_raises(self) -> None:
        """Test that non-vision model raises CapabilityError."""
        with pytest.raises(CapabilityError):
            ensure_image_support("o1-mini", images_required=True)

    @pytest.mark.unit
    def test_non_vision_model_ok_when_not_required(self) -> None:
        """Test that non-vision model is OK when images not required."""
        ensure_image_support("o1-mini", images_required=False)  # Should not raise


class TestReasoningModelDetection:
    """Tests for reasoning model detection."""

    @pytest.mark.unit
    def test_o1_is_reasoning(self) -> None:
        """Test that o1 models are recognized as reasoning."""
        assert detect_capabilities("o1").is_reasoning_model is True
        assert detect_capabilities("o1-mini").is_reasoning_model is True

    @pytest.mark.unit
    def test_o3_is_reasoning(self) -> None:
        """Test that o3 models are recognized as reasoning."""
        assert detect_capabilities("o3").is_reasoning_model is True
        assert detect_capabilities("o3-mini").is_reasoning_model is True

    @pytest.mark.unit
    def test_gpt4_not_reasoning(self) -> None:
        """Test that GPT-4 models are not reasoning models."""
        assert detect_capabilities("gpt-4o").is_reasoning_model is False

    @pytest.mark.unit
    def test_gpt5_is_reasoning(self) -> None:
        """Test that GPT-5 models are reasoning models."""
        assert detect_capabilities("gpt-5").is_reasoning_model is True


class TestGPT54Capabilities:
    """Tests for GPT-5.4 and GPT-5.3 model capabilities."""

    @pytest.mark.unit
    def test_gpt54_capabilities(self) -> None:
        """Test GPT-5.4 capability values."""
        caps = detect_capabilities("gpt-5.4")
        assert caps.max_context_tokens == 1050000
        assert caps.max_output_tokens == 128000
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_image_detail_original is True
        assert caps.supports_image_input is True
        assert caps.supports_sampler_controls is False

    @pytest.mark.unit
    def test_gpt54_pro_capabilities(self) -> None:
        """Test GPT-5.4 Pro capability values."""
        caps = detect_capabilities("gpt-5.4-pro")
        assert caps.max_context_tokens == 1050000
        assert caps.max_output_tokens == 128000
        assert caps.is_reasoning_model is True
        assert caps.supports_image_detail_original is True

    @pytest.mark.unit
    def test_gpt53_instant_capabilities(self) -> None:
        """Test GPT-5.3 Instant is a non-reasoning standard model."""
        caps = detect_capabilities("gpt-5.3")
        assert caps.max_context_tokens == 400000
        assert caps.max_output_tokens == 128000
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False
        assert caps.supports_sampler_controls is True
        assert caps.supports_image_input is True

    @pytest.mark.unit
    def test_gpt54_original_detail_accepted(self) -> None:
        """Test that original detail is gated to gpt-5.4+."""
        caps_54 = detect_capabilities("gpt-5.4")
        assert caps_54.supports_image_detail_original is True

        caps_52 = detect_capabilities("gpt-5.2")
        assert caps_52.supports_image_detail_original is False

    @pytest.mark.unit
    def test_gpt52_no_original_detail(self) -> None:
        """Test that GPT-5.2 does not support original image detail."""
        caps = detect_capabilities("gpt-5.2")
        assert caps.supports_image_detail_original is False


class TestGPT54VariantCapabilities:
    """Tests for GPT-5.4 mini/nano and Codex models."""

    @pytest.mark.unit
    def test_gpt54_mini_has_distinct_family(self) -> None:
        caps = detect_capabilities("gpt-5.4-mini")
        assert caps.family == "gpt-5.4-mini"
        assert caps.max_context_tokens == 400000
        assert caps.max_output_tokens == 128000
        assert caps.is_reasoning_model is True
        assert caps.supports_image_detail_original is False

    @pytest.mark.unit
    def test_gpt54_nano_has_distinct_family(self) -> None:
        caps = detect_capabilities("gpt-5.4-nano")
        assert caps.family == "gpt-5.4-nano"
        assert caps.max_context_tokens == 400000
        assert caps.max_output_tokens == 128000
        assert caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_gpt53_codex_is_non_reasoning_no_image(self) -> None:
        caps = detect_capabilities("gpt-5.3-codex")
        assert caps.family == "gpt-5.3-codex"
        assert caps.is_reasoning_model is False
        assert caps.supports_image_input is False
        assert caps.max_context_tokens == 400000

    @pytest.mark.unit
    def test_gpt52_codex_is_non_reasoning_no_image(self) -> None:
        caps = detect_capabilities("gpt-5.2-codex")
        assert caps.family == "gpt-5.2-codex"
        assert caps.is_reasoning_model is False
        assert caps.supports_image_input is False


class TestClaudeUpdatedCapabilities:
    """Tests for updated Anthropic Claude model capabilities."""

    @pytest.mark.unit
    def test_claude_opus_47_capabilities(self) -> None:
        caps = detect_capabilities("claude-opus-4-7")
        assert caps.family == "claude-opus-4.7"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_top_p is False
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 128000

    @pytest.mark.unit
    def test_claude_opus_46_context_window(self) -> None:
        caps = detect_capabilities("claude-opus-4-6")
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 128000

    @pytest.mark.unit
    def test_claude_sonnet_46_context_window(self) -> None:
        caps = detect_capabilities("claude-sonnet-4-6")
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 65536

    @pytest.mark.unit
    def test_claude_sonnet_45_output_tokens(self) -> None:
        caps = detect_capabilities("claude-sonnet-4-5-20250929")
        assert caps.max_output_tokens == 65536

    @pytest.mark.unit
    def test_claude_opus_45_output_tokens(self) -> None:
        caps = detect_capabilities("claude-opus-4-5-20251101")
        assert caps.max_output_tokens == 65536

    @pytest.mark.unit
    def test_claude_opus_41_output_tokens(self) -> None:
        caps = detect_capabilities("claude-opus-4-1-20250805")
        assert caps.max_output_tokens == 32768

    @pytest.mark.unit
    def test_claude_sonnet_4_is_reasoning(self) -> None:
        caps = detect_capabilities("claude-sonnet-4-20250514")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True


class TestGemini31Capabilities:
    """Tests for Gemini 3.1 model capabilities."""

    @pytest.mark.unit
    def test_gemini_31_pro_preview(self) -> None:
        caps = detect_capabilities("gemini-3.1-pro-preview")
        assert caps.family == "gemini-3.1-pro"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 65536

    @pytest.mark.unit
    def test_gemini_31_flash_lite_preview(self) -> None:
        caps = detect_capabilities("gemini-3.1-flash-lite-preview")
        assert caps.family == "gemini-3.1-flash-lite"
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 65536

    @pytest.mark.unit
    def test_gemini_31_flash_image_preview(self) -> None:
        caps = detect_capabilities("gemini-3.1-flash-image-preview")
        assert caps.family == "gemini-3.1-flash-image"
        assert caps.max_context_tokens == 128000
        assert caps.max_output_tokens == 32768

    @pytest.mark.unit
    def test_gemini_3_pro_image_preview(self) -> None:
        caps = detect_capabilities("gemini-3-pro-image-preview")
        assert caps.family == "gemini-3-pro-image"
        assert caps.max_context_tokens == 65536
        assert caps.max_output_tokens == 32768

    @pytest.mark.unit
    def test_gemini_31_dash_alias(self) -> None:
        caps = detect_capabilities("gemini-3-1-pro-preview")
        assert caps.family == "gemini-3.1-pro"


class TestAudioCapabilities:
    """Tests for the audio-input capability flags and their gate functions."""

    @pytest.mark.unit
    def test_gpt_transcribe_is_an_audio_family(self) -> None:
        caps = detect_capabilities("gpt-transcribe")
        assert caps.family == "openai-audio"
        assert caps.supports_audio_input is True
        assert caps.audio_transcription_only is True

    @pytest.mark.unit
    def test_gpt_4o_transcribe_is_not_claimed_by_the_gpt_4o_prefix(self) -> None:
        """The transcribe ids must win over the bare ``gpt-4o`` vision profile."""
        caps = detect_capabilities("gpt-4o-transcribe")
        assert caps.family == "openai-audio"
        assert caps.supports_image_input is False

    @pytest.mark.unit
    def test_gpt_4o_mini_transcribe_is_an_audio_family(self) -> None:
        assert detect_capabilities("gpt-4o-mini-transcribe").family == "openai-audio"

    @pytest.mark.unit
    def test_whisper_1_is_an_audio_family(self) -> None:
        caps = detect_capabilities("whisper-1")
        assert caps.family == "openai-audio"
        assert caps.supports_audio_input is True

    @pytest.mark.unit
    def test_gpt_4o_does_not_accept_audio(self) -> None:
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_audio_input is False
        assert caps.audio_transcription_only is False

    @pytest.mark.unit
    def test_gpt_5_does_not_accept_audio(self) -> None:
        assert detect_capabilities("gpt-5").supports_audio_input is False

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model",
        ["gemini-3-flash", "gemini-3-pro-preview", "gemini-3.1-pro-preview"],
    )
    def test_gemini_models_accept_audio(self, model: str) -> None:
        assert detect_capabilities(model).supports_audio_input is True

    @pytest.mark.unit
    @pytest.mark.parametrize("model", ["gemma-4-31b-it", "gemma"])
    def test_gemma_models_do_not_accept_audio(self, model: str) -> None:
        assert detect_capabilities(model).supports_audio_input is False

    @pytest.mark.unit
    def test_ensure_audio_support_passes_for_an_audio_model(self) -> None:
        from modules.config.capabilities import ensure_audio_support

        assert ensure_audio_support("gpt-transcribe") is None

    @pytest.mark.unit
    def test_ensure_audio_support_raises_for_a_chat_only_model(self) -> None:
        from modules.config.capabilities import ensure_audio_support

        with pytest.raises(CapabilityError, match="does not support them"):
            ensure_audio_support("gpt-5")

    @pytest.mark.unit
    def test_ensure_image_support_warns_but_does_not_raise_for_audio_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An audio-only model must not break config load for an audio run."""
        with caplog.at_level("WARNING"):
            assert ensure_image_support("gpt-transcribe", True) is None
        assert "accepts audio but not image inputs" in caplog.text

    @pytest.mark.unit
    def test_ensure_image_support_still_raises_for_a_text_only_model(self) -> None:
        # o1-mini accepts neither images nor audio: the audio-only downgrade
        # must not have widened into a blanket pass.
        with pytest.raises(CapabilityError):
            ensure_image_support("o1-mini", True)
