"""Batch API integration subpackage.

Provides batch job creation, submission, and management utilities
for multiple LLM providers (OpenAI, Anthropic, Google).

Usage:
    # Get a batch backend for a specific provider
    from modules.llm.batch.backends import get_batch_backend
    
    backend = get_batch_backend("openai")  # or "anthropic", "google"
    handle = backend.submit_batch(requests, model_config, ...)
    
    # Legacy OpenAI-specific functions (deprecated, use backends instead)
    from modules.llm.batch.batching import process_batch_transcription
"""

# Re-export backend abstractions
from modules.llm.batch.backends import (
    BatchBackend,
    BatchHandle,
    BatchStatus,
    BatchResultItem,
    BatchRequest,
    get_batch_backend,
)

# Legacy exports for backward compatibility
from modules.llm.batch.batching import get_batch_chunk_size

__all__ = [
    # New backend abstraction
    "BatchBackend",
    "BatchHandle",
    "BatchStatus",
    "BatchResultItem",
    "BatchRequest",
    "get_batch_backend",
    # Legacy
    "get_batch_chunk_size",
]
