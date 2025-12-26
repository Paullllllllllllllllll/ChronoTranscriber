"""Unit tests for modules/llm/model_capabilities.py.

Tests model capability detection and feature support.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from modules.llm.model_capabilities import (
    Capabilities,
    detect_capabilities,
    ensure_image_support,
    CapabilityError,
)


class TestCapabilities:
    """Tests for Capabilities dataclass."""
    
    @pytest.mark.unit
    def test_gpt4o_capabilities(self):
        """Test GPT-4o capability values."""
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_image_input is True
        assert caps.supports_structured_outputs is True
        assert caps.supports_sampler_controls is True
        assert caps.is_reasoning_model is False
    
    @pytest.mark.unit
    def test_reasoning_model_capabilities(self):
        """Test reasoning model capabilities."""
        caps = detect_capabilities("o1")
        assert caps.is_reasoning_model is True
        assert caps.supports_sampler_controls is False


class TestDetectCapabilities:
    """Tests for detect_capabilities function."""
    
    @pytest.mark.unit
    def test_gpt4o_capabilities(self):
        """Test capabilities for GPT-4o model."""
        caps = detect_capabilities("gpt-4o")
        assert caps.supports_image_input is True
        assert caps.supports_structured_outputs is True
        assert caps.supports_sampler_controls is True
    
    @pytest.mark.unit
    def test_o1_reasoning_model(self):
        """Test capabilities for o1 reasoning model."""
        caps = detect_capabilities("o1")
        assert caps.is_reasoning_model is True
        assert caps.supports_sampler_controls is False
    
    @pytest.mark.unit
    def test_o3_reasoning_model(self):
        """Test capabilities for o3 reasoning model."""
        caps = detect_capabilities("o3")
        assert caps.is_reasoning_model is True
    
    @pytest.mark.unit
    def test_unknown_model_defaults(self):
        """Test that unknown model gets default capabilities."""
        caps = detect_capabilities("unknown-model-xyz")
        # Should return sensible defaults
        assert isinstance(caps, Capabilities)
        assert caps.family == "unknown"
    
    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test that model name matching is case-insensitive."""
        caps1 = detect_capabilities("gpt-4o")
        caps2 = detect_capabilities("GPT-4O")
        assert caps1.supports_image_input == caps2.supports_image_input


class TestEnsureImageSupport:
    """Tests for ensure_image_support function."""
    
    @pytest.mark.unit
    def test_vision_model_passes(self):
        """Test that vision-capable model doesn't raise."""
        ensure_image_support("gpt-4o", images_required=True)  # Should not raise
    
    @pytest.mark.unit
    def test_non_vision_model_raises(self):
        """Test that non-vision model raises CapabilityError."""
        with pytest.raises(CapabilityError):
            ensure_image_support("o1-mini", images_required=True)
    
    @pytest.mark.unit
    def test_non_vision_model_ok_when_not_required(self):
        """Test that non-vision model is OK when images not required."""
        ensure_image_support("o1-mini", images_required=False)  # Should not raise


class TestReasoningModelDetection:
    """Tests for reasoning model detection."""
    
    @pytest.mark.unit
    def test_o1_is_reasoning(self):
        """Test that o1 models are recognized as reasoning."""
        assert detect_capabilities("o1").is_reasoning_model is True
        assert detect_capabilities("o1-mini").is_reasoning_model is True
    
    @pytest.mark.unit
    def test_o3_is_reasoning(self):
        """Test that o3 models are recognized as reasoning."""
        assert detect_capabilities("o3").is_reasoning_model is True
        assert detect_capabilities("o3-mini").is_reasoning_model is True
    
    @pytest.mark.unit
    def test_gpt4_not_reasoning(self):
        """Test that GPT-4 models are not reasoning models."""
        assert detect_capabilities("gpt-4o").is_reasoning_model is False
    
    @pytest.mark.unit
    def test_gpt5_is_reasoning(self):
        """Test that GPT-5 models are reasoning models."""
        assert detect_capabilities("gpt-5").is_reasoning_model is True
