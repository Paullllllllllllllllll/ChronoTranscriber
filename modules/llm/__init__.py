"""Language model integration package.

Provides multi-provider LLM integration using LangChain, with support for:
- OpenAI (GPT-4o, GPT-5, o1, o3, etc.)
- Anthropic (Claude 3 family)
- Google (Gemini family)
- OpenRouter (200+ models)

Also includes batch processing, schema utilities, and model capabilities.

Note: Imports are lazy to avoid circular import issues with config_loader.
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name in ("LangChainTranscriber", "open_transcriber", "transcribe_image_with_llm"):
        from modules.llm.transcriber import (
            LangChainTranscriber,
            open_transcriber,
            transcribe_image_with_llm,
        )
        return {
            "LangChainTranscriber": LangChainTranscriber,
            "open_transcriber": open_transcriber,
            "transcribe_image_with_llm": transcribe_image_with_llm,
        }[name]
    
    if name in ("BaseProvider", "ProviderCapabilities", "TranscriptionResult",
                "get_provider", "get_available_providers", "ProviderType"):
        from modules.llm.providers import (
            BaseProvider,
            ProviderCapabilities,
            TranscriptionResult,
            get_provider,
            get_available_providers,
            ProviderType,
        )
        return {
            "BaseProvider": BaseProvider,
            "ProviderCapabilities": ProviderCapabilities,
            "TranscriptionResult": TranscriptionResult,
            "get_provider": get_provider,
            "get_available_providers": get_available_providers,
            "ProviderType": ProviderType,
        }[name]
    
    if name == "list_schema_options":
        from modules.llm.schema_utils import list_schema_options
        return list_schema_options
    
    if name in ("detect_capabilities", "ensure_image_support", "Capabilities"):
        from modules.llm.model_capabilities import (
            detect_capabilities,
            ensure_image_support,
            Capabilities,
        )
        return {
            "detect_capabilities": detect_capabilities,
            "ensure_image_support": ensure_image_support,
            "Capabilities": Capabilities,
        }[name]
    
    raise AttributeError(f"module 'modules.llm' has no attribute '{name}'")

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
    # Model capabilities
    "detect_capabilities",
    "ensure_image_support",
    "Capabilities",
]
