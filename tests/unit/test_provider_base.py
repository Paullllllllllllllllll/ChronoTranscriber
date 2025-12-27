"""Unit tests for modules/llm/providers/base.py.

Tests base provider abstraction and common utilities.
"""

from __future__ import annotations

import base64
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.llm.providers.base import (
    ProviderCapabilities,
    TranscriptionResult,
    BaseProvider,
)


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""
    
    @pytest.mark.unit
    def test_default_values(self):
        """Test default capability values."""
        caps = ProviderCapabilities(
            provider_name="test",
            model_name="test-model",
        )
        
        assert caps.supports_vision is False
        assert caps.supports_structured_output is False
        assert caps.is_reasoning_model is False
        assert caps.supports_temperature is True
        assert caps.supports_streaming is True
    
    @pytest.mark.unit
    def test_vision_capabilities(self):
        """Test vision-related capabilities."""
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="gpt-4o",
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
        )
        
        assert caps.supports_vision is True
        assert caps.supports_image_detail is True
        assert caps.default_image_detail == "high"
    
    @pytest.mark.unit
    def test_reasoning_model_capabilities(self):
        """Test reasoning model capabilities."""
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="o1",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
        )
        
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_temperature is False
    
    @pytest.mark.unit
    def test_frozen_dataclass(self):
        """Test that capabilities are immutable."""
        caps = ProviderCapabilities(
            provider_name="test",
            model_name="test-model",
        )
        
        with pytest.raises(AttributeError):
            caps.provider_name = "modified"
    
    @pytest.mark.unit
    def test_token_limits(self):
        """Test token limit attributes."""
        caps = ProviderCapabilities(
            provider_name="test",
            model_name="test-model",
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
        
        assert caps.max_context_tokens == 200000
        assert caps.max_output_tokens == 8192


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""
    
    @pytest.mark.unit
    def test_basic_result(self):
        """Test basic transcription result."""
        result = TranscriptionResult(content="Transcribed text here")
        
        assert result.content == "Transcribed text here"
        assert result.error is None
        assert result.no_transcribable_text is False
        assert result.transcription_not_possible is False
    
    @pytest.mark.unit
    def test_json_content_parsing(self):
        """Test automatic parsing of JSON content."""
        json_content = json.dumps({
            "transcription": "Parsed text",
            "no_transcribable_text": True,
            "transcription_not_possible": False,
        })
        
        result = TranscriptionResult(content=json_content)
        
        assert result.parsed_output is not None
        assert result.no_transcribable_text is True
        assert result.transcription_not_possible is False
    
    @pytest.mark.unit
    def test_non_json_content(self):
        """Test handling of non-JSON content."""
        result = TranscriptionResult(content="Plain text transcription")
        
        assert result.parsed_output is None
        assert result.no_transcribable_text is False
    
    @pytest.mark.unit
    def test_token_usage(self):
        """Test token usage tracking."""
        result = TranscriptionResult(
            content="Text",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
    
    @pytest.mark.unit
    def test_error_result(self):
        """Test error in result."""
        result = TranscriptionResult(
            content="",
            error="API rate limit exceeded",
        )
        
        assert result.error == "API rate limit exceeded"
    
    @pytest.mark.unit
    def test_raw_response_stored(self):
        """Test raw response storage."""
        raw = {"id": "resp_123", "model": "gpt-4o"}
        result = TranscriptionResult(content="Text", raw_response=raw)
        
        assert result.raw_response == raw
    
    @pytest.mark.unit
    def test_transcription_not_possible_flag(self):
        """Test transcription_not_possible flag parsing."""
        json_content = json.dumps({
            "transcription": "",
            "no_transcribable_text": False,
            "transcription_not_possible": True,
        })
        
        result = TranscriptionResult(content=json_content)
        
        assert result.transcription_not_possible is True
    
    @pytest.mark.unit
    def test_malformed_json_handled(self):
        """Test that malformed JSON doesn't crash."""
        result = TranscriptionResult(content="{invalid json")
        
        assert result.parsed_output is None
        assert result.no_transcribable_text is False
    
    @pytest.mark.unit
    def test_json_with_extra_whitespace(self):
        """Test JSON parsing with leading/trailing whitespace."""
        json_content = "  \n" + json.dumps({
            "transcription": "Text",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }) + "  \n"
        
        result = TranscriptionResult(content=json_content)
        
        assert result.parsed_output is not None


class TestBaseProviderStatics:
    """Tests for BaseProvider static methods."""
    
    @pytest.mark.unit
    def test_encode_image_to_base64(self, temp_dir):
        """Test image encoding to base64."""
        # Create a simple test image file
        image_path = temp_dir / "test.png"
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        image_path.write_bytes(image_data)
        
        b64_data, mime_type = BaseProvider.encode_image_to_base64(image_path)
        
        assert mime_type == "image/png"
        assert b64_data == base64.b64encode(image_data).decode("utf-8")
    
    @pytest.mark.unit
    def test_encode_jpeg_image(self, temp_dir):
        """Test JPEG image encoding."""
        image_path = temp_dir / "test.jpg"
        image_data = b"\xff\xd8\xff" + b"\x00" * 100
        image_path.write_bytes(image_data)
        
        b64_data, mime_type = BaseProvider.encode_image_to_base64(image_path)
        
        assert mime_type == "image/jpeg"
    
    @pytest.mark.unit
    def test_unsupported_format_raises(self, temp_dir):
        """Test that unsupported format raises ValueError."""
        image_path = temp_dir / "test.xyz"
        image_path.write_bytes(b"data")
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            BaseProvider.encode_image_to_base64(image_path)
    
    @pytest.mark.unit
    def test_create_data_url(self):
        """Test data URL creation."""
        result = BaseProvider.create_data_url("abc123", "image/png")
        
        assert result == "data:image/png;base64,abc123"
    
    @pytest.mark.unit
    def test_create_data_url_jpeg(self):
        """Test data URL for JPEG."""
        result = BaseProvider.create_data_url("xyz789", "image/jpeg")
        
        assert result == "data:image/jpeg;base64,xyz789"


class TestBaseProviderInit:
    """Tests for BaseProvider initialization."""
    
    @pytest.mark.unit
    def test_cannot_instantiate_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseProvider(api_key="key", model="model")
    
    @pytest.mark.unit
    def test_concrete_implementation_attributes(self):
        """Test that concrete implementation has correct attributes."""
        # Create a minimal concrete implementation for testing
        class ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"
            
            def get_capabilities(self):
                return ProviderCapabilities(
                    provider_name="test",
                    model_name=self.model,
                )
            
            async def transcribe_image(self, *args, **kwargs):
                pass
            
            async def transcribe_image_from_base64(self, *args, **kwargs):
                pass
            
            async def close(self):
                pass
        
        provider = ConcreteProvider(
            api_key="test-key",
            model="test-model",
            temperature=0.5,
            max_tokens=2048,
            timeout=30.0,
        )
        
        assert provider.api_key == "test-key"
        assert provider.model == "test-model"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048
        assert provider.timeout == 30.0


class TestProviderCapabilitiesEdgeCases:
    """Edge case tests for ProviderCapabilities."""
    
    @pytest.mark.unit
    def test_google_media_resolution(self):
        """Test Google-specific media resolution settings."""
        caps = ProviderCapabilities(
            provider_name="google",
            model_name="gemini-pro",
            supports_media_resolution=True,
            default_media_resolution="high",
        )
        
        assert caps.supports_media_resolution is True
        assert caps.default_media_resolution == "high"
    
    @pytest.mark.unit
    def test_structured_output_capability(self):
        """Test structured output capability."""
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="gpt-4o",
            supports_structured_output=True,
            supports_json_mode=True,
        )
        
        assert caps.supports_structured_output is True
        assert caps.supports_json_mode is True
