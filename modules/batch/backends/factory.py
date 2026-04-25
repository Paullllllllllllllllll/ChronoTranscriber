"""Factory function for batch processing backends.

Provides a unified way to get the appropriate batch backend for a given provider.
"""

from __future__ import annotations

from typing import Optional

from modules.infra.logger import setup_logger
from modules.batch.backends.base import BatchBackend

logger = setup_logger(__name__)

# Lazy-loaded backend instances (singletons)
_backends: dict[str, BatchBackend] = {}


def get_batch_backend(provider: Optional[str] = None) -> BatchBackend:
    """Get the batch backend for a given provider.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google').
                  If None, attempts to detect from model_config.yaml.

    Returns:
        BatchBackend instance for the specified provider.

    Raises:
        ValueError: If provider is not supported or cannot be determined.
    """
    # Normalize provider name
    if provider is None:
        # Try to detect from config
        try:
            from modules.config.service import get_config_service
            mc = get_config_service().get_model_config()
            tm = mc.get("transcription_model", {})
            provider = tm.get("provider")
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.debug("Could not auto-detect provider from config: %s", e)

    if provider is None:
        raise ValueError(
            "Provider not specified and could not be detected from config. "
            "Set 'provider' in transcription_model config or pass it explicitly."
        )

    provider = provider.lower().strip()

    # Check for cached instance
    if provider in _backends:
        return _backends[provider]

    # Create new backend
    backend: BatchBackend
    if provider == "openai":
        from modules.batch.backends.openai_backend import OpenAIBatchBackend
        backend = OpenAIBatchBackend()

    elif provider == "anthropic":
        from modules.batch.backends.anthropic_backend import AnthropicBatchBackend
        backend = AnthropicBatchBackend()

    elif provider == "google":
        from modules.batch.backends.google_backend import GoogleBatchBackend
        backend = GoogleBatchBackend()

    elif provider == "openrouter":
        # OpenRouter doesn't have a batch API - fall back to error
        raise ValueError(
            "OpenRouter does not support batch processing. "
            "Use synchronous mode or switch to a provider with batch support "
            "(openai, anthropic, google)."
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported batch providers: openai, anthropic, google"
        )

    # Cache and return
    _backends[provider] = backend
    logger.info("Created batch backend for provider: %s", provider)
    return backend


def supports_batch(provider: str) -> bool:
    """Check if a provider supports batch processing.

    Args:
        provider: Provider name to check.

    Returns:
        True if the provider supports batch processing.
    """
    return provider.lower().strip() in ("openai", "anthropic", "google")


def clear_backend_cache() -> None:
    """Clear the cached backend instances.

    Useful for testing or when configuration changes.
    """
    global _backends
    _backends = {}
