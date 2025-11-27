"""Provider factory for dynamic LLM provider selection.

Creates provider instances based on configuration or explicit parameters.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional, Type

from modules.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


# Lazy import mapping to avoid circular imports and unnecessary dependencies
_PROVIDER_CLASSES: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "modules.llm.providers.openai_provider.OpenAIProvider",
    ProviderType.ANTHROPIC: "modules.llm.providers.anthropic_provider.AnthropicProvider",
    ProviderType.GOOGLE: "modules.llm.providers.google_provider.GoogleProvider",
    ProviderType.OPENROUTER: "modules.llm.providers.openrouter_provider.OpenRouterProvider",
}

# Environment variable names for API keys
_API_KEY_ENV_VARS: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.GOOGLE: "GOOGLE_API_KEY",
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
}


def _import_provider_class(provider_type: ProviderType) -> Type[BaseProvider]:
    """Dynamically import a provider class."""
    module_path = _PROVIDER_CLASSES[provider_type]
    module_name, class_name = module_path.rsplit(".", 1)
    
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_available_providers() -> list[ProviderType]:
    """Return list of provider types that have API keys configured."""
    available = []
    for provider_type, env_var in _API_KEY_ENV_VARS.items():
        if os.environ.get(env_var):
            available.append(provider_type)
    return available


def detect_provider_from_model(model_name: str) -> ProviderType:
    """Attempt to detect the provider from the model name.
    
    Args:
        model_name: The model name/identifier
    
    Returns:
        Best-guess ProviderType based on model name patterns
    """
    m = model_name.lower().strip()
    
    # OpenRouter format: provider/model
    if "/" in m:
        prefix = m.split("/")[0]
        if prefix in ("openai", "anthropic", "google", "meta", "mistral"):
            return ProviderType.OPENROUTER
    
    # OpenAI models
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return ProviderType.OPENAI
    
    # Anthropic models
    if "claude" in m:
        return ProviderType.ANTHROPIC
    
    # Google models
    if "gemini" in m or m.startswith("models/"):
        return ProviderType.GOOGLE
    
    # Llama, Mistral, etc. via OpenRouter
    if any(x in m for x in ["llama", "mistral", "mixtral", "qwen", "deepseek"]):
        return ProviderType.OPENROUTER
    
    # Default to OpenAI
    return ProviderType.OPENAI


def get_api_key_for_provider(
    provider_type: ProviderType,
    api_key: Optional[str] = None,
) -> str:
    """Get the API key for a provider.
    
    Args:
        provider_type: The provider type
        api_key: Optional explicit API key
    
    Returns:
        The API key to use
    
    Raises:
        ValueError: If no API key is available
    """
    if api_key:
        return api_key
    
    env_var = _API_KEY_ENV_VARS.get(provider_type)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    
    # Fallback: try OPENAI_API_KEY for OpenRouter if OPENROUTER_API_KEY not set
    if provider_type == ProviderType.OPENROUTER:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            logger.warning(
                "OPENROUTER_API_KEY not set, falling back to OPENAI_API_KEY. "
                "This may not work with all OpenRouter features."
            )
            return key
    
    raise ValueError(
        f"No API key found for provider {provider_type.value}. "
        f"Set {env_var} environment variable or pass api_key parameter."
    )


def get_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: Optional[float] = None,
    **kwargs,
) -> BaseProvider:
    """Create a provider instance.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google", "openrouter")
                  If None, attempts to detect from model name or config
        model: Model name/identifier
        api_key: Optional explicit API key
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        timeout: Request timeout in seconds
        **kwargs: Provider-specific configuration
    
    Returns:
        Configured provider instance
    
    Raises:
        ValueError: If provider cannot be determined or API key is missing
    """
    # Load defaults from config if not provided
    if model is None or provider is None:
        try:
            from modules.config.service import get_config_service
            config = get_config_service().get_model_config()
            tm = config.get("transcription_model", {})
            
            if model is None:
                model = tm.get("name", "gpt-4o")
            if provider is None:
                provider = tm.get("provider")  # May still be None
            
            # Load other defaults from config
            if "temperature" not in kwargs and tm.get("temperature") is not None:
                temperature = float(tm.get("temperature", 0.0))
            if tm.get("max_output_tokens") is not None:
                max_tokens = int(tm.get("max_output_tokens", max_tokens))
            elif tm.get("max_tokens") is not None:
                max_tokens = int(tm.get("max_tokens", max_tokens))
            
            # Load optional parameters
            for key in ["top_p", "frequency_penalty", "presence_penalty", "top_k"]:
                if key not in kwargs and tm.get(key) is not None:
                    kwargs[key] = tm.get(key)
                    
        except Exception as e:
            logger.warning(f"Could not load config defaults: {e}")
            if model is None:
                model = "gpt-4o"
    
    # Determine provider type
    if provider is None:
        provider_type = detect_provider_from_model(model)
        logger.info(f"Auto-detected provider '{provider_type.value}' for model '{model}'")
    else:
        try:
            provider_type = ProviderType(provider.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {', '.join(p.value for p in ProviderType)}"
            )
    
    # Get API key
    resolved_api_key = get_api_key_for_provider(provider_type, api_key)
    
    # Import and instantiate provider
    provider_class = _import_provider_class(provider_type)
    
    return provider_class(
        api_key=resolved_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )


def get_provider_for_transcription(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    schema_path: Optional[str] = None,
    system_prompt_path: Optional[str] = None,
    additional_context_path: Optional[str] = None,
) -> BaseProvider:
    """Create a provider configured for transcription.
    
    Loads configuration from model_config.yaml and concurrency_config.yaml.
    
    Args:
        api_key: Optional explicit API key
        model: Optional model override
        provider: Optional provider override
        schema_path: Path to JSON schema (for reference, not used directly)
        system_prompt_path: Path to system prompt (for reference, not used directly)
        additional_context_path: Path to additional context
    
    Returns:
        Configured provider instance ready for transcription
    """
    from modules.config.service import get_config_service
    
    # Load model config
    config = get_config_service().get_model_config()
    tm = config.get("transcription_model", {})
    
    # Load concurrency config for service tier
    try:
        cc = get_config_service().get_concurrency_config()
        service_tier = (
            (cc.get("concurrency", {}) or {})
            .get("transcription", {})
            .get("service_tier")
        )
    except Exception:
        service_tier = None
    
    # Build kwargs
    kwargs: Dict[str, Any] = {}
    
    if service_tier:
        kwargs["service_tier"] = service_tier
    
    # Load optional parameters from config
    for key in ["top_p", "frequency_penalty", "presence_penalty", "top_k"]:
        if tm.get(key) is not None:
            kwargs[key] = tm.get(key)
    
    return get_provider(
        provider=provider or tm.get("provider"),
        model=model or tm.get("name"),
        api_key=api_key,
        temperature=float(tm.get("temperature", 0.0)),
        max_tokens=int(
            tm.get("max_output_tokens")
            or tm.get("max_tokens", 4096)
        ),
        **kwargs,
    )
