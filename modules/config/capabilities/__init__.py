"""Model capability registry, detection, and parameter gating.

Three submodules (single source of truth for capabilities across providers):
- registry: Capabilities dataclass + provider base dicts + _MODEL_REGISTRY data.
- detection: detect_provider, detect_capabilities (with OpenRouter passthrough),
  plus model-type/image-config helpers.
- params: CapabilityError + ensure_image_support fail-fast safety gate.

Public API is re-exported here; callers should prefer
`from modules.config.capabilities import detect_capabilities, ...`.
"""

from modules.config.capabilities.detection import (
    detect_capabilities,
    detect_model_type,
    detect_provider,
    get_image_config_section_name,
)
from modules.config.capabilities.params import (
    CapabilityError,
    ensure_image_support,
)
from modules.config.capabilities.registry import (
    ApiPref,
    Capabilities,
    ImageDetail,
    ProviderType,
)

__all__ = [
    # registry
    "Capabilities",
    "ImageDetail",
    "ApiPref",
    "ProviderType",
    # detection
    "detect_capabilities",
    "detect_provider",
    "detect_model_type",
    "get_image_config_section_name",
    # params
    "CapabilityError",
    "ensure_image_support",
]
