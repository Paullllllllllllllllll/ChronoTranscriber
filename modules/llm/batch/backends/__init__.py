"""Batch processing backends for different LLM providers.

This package provides a unified interface for batch processing across providers:
- OpenAI: Uses the Batch API with /v1/responses endpoint
- Anthropic: Uses the Message Batches API
- Google: Uses the Gemini Batch API

Usage:
    from modules.llm.batch.backends import get_batch_backend
    
    backend = get_batch_backend("openai")
    handle = backend.submit_batch(requests, model_config, ...)
    status = backend.get_status(handle)
    results = backend.download_results(handle)
"""

from modules.llm.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchStatus,
    BatchResultItem,
    BatchRequest,
)
from modules.llm.batch.backends.factory import get_batch_backend

__all__ = [
    "BatchBackend",
    "BatchHandle",
    "BatchStatus",
    "BatchResultItem",
    "BatchRequest",
    "get_batch_backend",
]
