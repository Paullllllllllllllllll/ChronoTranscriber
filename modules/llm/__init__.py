"""Language model integration package.

Provides multi-provider LLM integration using LangChain, with support for:
- OpenAI (GPT-4o, GPT-5, o1, o3, etc.)
- Anthropic (Claude 3 family)
- Google (Gemini family)
- OpenRouter (200+ models)

Also includes schema utilities, prompt rendering, LLM response parsing,
and content-quality validators for transcription output.

Capability detection and context-file resolution moved to
``modules.config.capabilities`` and ``modules.config.context`` respectively.
"""

from modules.llm.providers import (
    BaseProvider,
    ProviderCapabilities,
    ProviderType,
    TranscriptionResult,
    get_available_providers,
    get_provider,
)
from modules.llm.schema_utils import list_schema_options
from modules.llm.transcriber import (
    LangChainTranscriber,
    open_transcriber,
    transcribe_image_with_llm,
)

__all__ = [
    # LangChain API
    "LangChainTranscriber",
    "open_transcriber",
    "transcribe_image_with_llm",
    # Provider abstraction
    "BaseProvider",
    "ProviderCapabilities",
    "TranscriptionResult",
    "get_provider",
    "get_available_providers",
    "ProviderType",
    # Schema utilities
    "list_schema_options",
]
