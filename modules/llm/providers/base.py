"""Base provider abstraction for LLM integrations.

Defines the common interface that all LLM providers must implement.
"""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modules.config.constants import SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes capabilities of an LLM provider/model combination."""
    
    provider_name: str
    model_name: str
    
    # Vision/multimodal
    supports_vision: bool = False
    supports_image_detail: bool = True  # OpenAI-style "detail" parameter
    default_image_detail: str = "high"  # "low", "high", "auto"
    supports_media_resolution: bool = False  # Google-style media_resolution parameter
    default_media_resolution: str = "high"  # "low", "medium", "high", "ultra_high", "auto"
    
    # Structured outputs
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    
    # Reasoning models
    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    
    # Sampler controls
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_frequency_penalty: bool = True
    supports_presence_penalty: bool = True
    
    # Streaming
    supports_streaming: bool = True
    
    # Context window
    max_context_tokens: int = 128000
    max_output_tokens: int = 4096


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    
    # Core result
    content: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    # Parsed structured output (if schema was provided)
    parsed_output: Optional[Dict[str, Any]] = None
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Transcription status flags (from schema response)
    no_transcribable_text: bool = False
    transcription_not_possible: bool = False
    
    # Error information
    error: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Parse transcription status flags from content if available."""
        if self.content and not self.parsed_output:
            try:
                stripped = self.content.strip()
                if stripped.startswith("{"):
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        self.parsed_output = parsed
                        self.no_transcribable_text = parsed.get("no_transcribable_text", False)
                        self.transcription_not_possible = parsed.get("transcription_not_possible", False)
            except json.JSONDecodeError:
                pass


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.
    
    All providers must implement these methods to work with the transcription pipeline.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize the provider.
        
        Args:
            api_key: API key for the provider
            model: Model name/identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_config = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider/model combination."""
        pass
    
    @abstractmethod
    async def transcribe_image(
        self,
        image_path: Path,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe text from an image.
        
        Args:
            image_path: Path to the image file
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            media_resolution: Media resolution for Google ("low", "medium", "high", "ultra_high", "auto")
        
        Returns:
            TranscriptionResult with the transcription and metadata
        """
        pass
    
    @abstractmethod
    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image data
            mime_type: MIME type of the image (e.g., "image/jpeg")
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            media_resolution: Media resolution for Google ("low", "medium", "high", "ultra_high", "auto")
        
        Returns:
            TranscriptionResult with the transcription and metadata
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g., HTTP sessions)."""
        pass
    
    async def __aenter__(self) -> "BaseProvider":
        """Async context manager entry."""
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Async context manager exit."""
        await self.close()
        return False
    
    @staticmethod
    def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Tuple of (base64_data, mime_type)
        
        Raises:
            ValueError: If the image format is not supported
        """
        ext = image_path.suffix.lower()
        mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
        if not mime_type:
            raise ValueError(f"Unsupported image format: {ext}")
        
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return data, mime_type
    
    @staticmethod
    def create_data_url(base64_data: str, mime_type: str) -> str:
        """Create a data URL from base64 data.
        
        Args:
            base64_data: Base64-encoded image data
            mime_type: MIME type of the image
        
        Returns:
            Data URL string
        """
        return f"data:{mime_type};base64,{base64_data}"
