"""LLM Provider abstraction layer using LangChain.

This module provides a unified interface for multiple LLM providers:
- OpenAI (GPT-4o, GPT-5, o1, o3, etc.)
- Anthropic (Claude 3 family)
- Google (Gemini family)
- OpenRouter (200+ models)

All providers support:
- Structured outputs via JSON schema
- Vision/multimodal inputs
- Streaming responses
- Retry logic with exponential backoff
"""

from modules.llm.providers.base import (
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
)
from modules.llm.providers.factory import (
    get_provider,
    get_available_providers,
    ProviderType,
)

__all__ = [
    "BaseProvider",
    "ProviderCapabilities",
    "TranscriptionResult",
    "get_provider",
    "get_available_providers",
    "ProviderType",
]
