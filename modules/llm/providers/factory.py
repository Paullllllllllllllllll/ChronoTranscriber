"""Provider factory for dynamic LLM provider selection.

Creates provider instances based on configuration or explicit parameters.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Type, cast

from modules.infra.logger import setup_logger
from modules.llm.providers.base import BaseProvider

logger = setup_logger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


# Lazy import mapping to avoid circular imports and unnecessary dependencies
_PROVIDER_CLASSES: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "modules.llm.providers.openai_provider.OpenAIProvider",
    ProviderType.ANTHROPIC: "modules.llm.providers.anthropic_provider.AnthropicProvider",
    ProviderType.GOOGLE: "modules.llm.providers.google_provider.GoogleProvider",
    ProviderType.OPENROUTER: "modules.llm.providers.openrouter_provider.OpenRouterProvider",
    ProviderType.CUSTOM: "modules.llm.providers.custom_provider.CustomProvider",
}

# Environment variable names for API keys
_API_KEY_ENV_VARS: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.GOOGLE: "GOOGLE_API_KEY",
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
    # Note: ProviderType.CUSTOM is intentionally omitted -- its API key env var
    # is user-configured via custom_endpoint.api_key_env_var in model_config.yaml.
}


def _import_provider_class(provider_type: ProviderType) -> Type[BaseProvider]:
    """Dynamically import a provider class."""
    module_path = _PROVIDER_CLASSES[provider_type]
    module_name, class_name = module_path.rsplit(".", 1)
    
    import importlib
    module = importlib.import_module(module_name)
    return cast(Type[BaseProvider], getattr(module, class_name))


def get_available_providers() -> list[ProviderType]:
    """Return list of provider types that have API keys configured."""
    available = []
    for provider_type, env_var in _API_KEY_ENV_VARS.items():
        if os.environ.get(env_var):
            available.append(provider_type)
    return available


def detect_provider_from_model(model_name: str) -> ProviderType:
    """Detect provider from model name, returning a ProviderType enum.

    Delegates to the canonical detect_provider() in model_capabilities
    and wraps the result in the ProviderType enum.
    """
    from modules.config.capabilities import detect_provider

    result = detect_provider(model_name)
    try:
        return ProviderType(result)
    except ValueError:
        return ProviderType.OPENAI  # Default fallback


def get_api_key_for_provider(
    provider_type: ProviderType,
    api_key: str | None = None,
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
    
    # Custom provider: read env var name from model config
    if provider_type == ProviderType.CUSTOM:
        try:
            from modules.config.service import get_config_service
            config = get_config_service().get_model_config()
            tm = config.get("transcription_model", {})
            custom_cfg = tm.get("custom_endpoint", {})
            env_var_name = custom_cfg.get("api_key_env_var")
            if env_var_name:
                key = os.environ.get(env_var_name)
                if key:
                    return key
            raise ValueError(
                f"Custom endpoint API key not found. "
                f"Set the environment variable specified in "
                f"custom_endpoint.api_key_env_var (currently: {env_var_name!r})."
            )
        except (KeyError, AttributeError, TypeError) as e:
            raise ValueError(
                f"Could not load custom endpoint config: {e}. "
                f"Ensure custom_endpoint.api_key_env_var is set in "
                f"model_config.yaml."
            ) from e

    raise ValueError(
        f"No API key found for provider {provider_type.value}. "
        f"Set {env_var} environment variable or pass api_key parameter."
    )


def get_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float | None = None,
    **kwargs: Any,
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
            
            # Load reasoning config (cross-provider)
            if "reasoning_config" not in kwargs and tm.get("reasoning") is not None:
                kwargs["reasoning_config"] = tm.get("reasoning")

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Could not load config defaults: {e}")
            if model is None:
                model = "gpt-4o"

    # Load custom endpoint config (must run even when provider/model are explicit)
    if (provider or "").lower() == "custom" and "base_url" not in kwargs:
        try:
            from modules.config.service import get_config_service
            config = get_config_service().get_model_config()
            tm = config.get("transcription_model", {})
            custom_cfg = tm.get("custom_endpoint", {})
            base_url = custom_cfg.get("base_url")
            if not base_url:
                raise ValueError(
                    "provider: custom requires "
                    "custom_endpoint.base_url in model_config.yaml"
                )
            kwargs["base_url"] = base_url

            # Forward user-configurable capabilities and prompt mode
            custom_capabilities = dict(custom_cfg.get("capabilities", {}))
            use_plain_text_prompt = bool(
                custom_cfg.get("use_plain_text_prompt", False)
            )

            if use_plain_text_prompt and custom_capabilities.get(
                "supports_structured_output", False
            ):
                logger.warning(
                    "use_plain_text_prompt=true is incompatible with "
                    "supports_structured_output=true; disabling structured output."
                )
                custom_capabilities["supports_structured_output"] = False

            kwargs["custom_capabilities"] = custom_capabilities
            kwargs["use_plain_text_prompt"] = use_plain_text_prompt
        except (KeyError, AttributeError, TypeError) as e:
            raise ValueError(
                f"Could not load custom endpoint config: {e}. "
                f"Ensure custom_endpoint.base_url is set in "
                f"model_config.yaml."
            ) from e

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
