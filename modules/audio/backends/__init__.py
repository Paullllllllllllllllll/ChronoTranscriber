"""Provider-specific speech-to-text backends.

One module per remote venue, each adapting a chunk payload to that provider's
request shape and returning the pipeline's legacy response dict. The contract
lives in :mod:`modules.audio.backends.base`; :func:`get_audio_backend` picks an
implementation without importing the one it does not need.
"""

from modules.audio.backends.base import (
    AudioBackend,
    build_legacy_response,
    error_response,
)
from modules.audio.backends.factory import get_audio_backend

__all__ = [
    "AudioBackend",
    "build_legacy_response",
    "error_response",
    "get_audio_backend",
]
