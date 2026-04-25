"""Provider-agnostic batch cancellation.

Thin façade over `BatchBackend.cancel()`. Callers that have only a bare
``batch_id`` (e.g. `main/cancel_batches.py`) use
``cancel_batch_by_id(provider, batch_id)`` so they never need to import
provider SDKs directly.
"""

from __future__ import annotations

from typing import Any

from modules.batch.backends import BatchHandle, get_batch_backend


def cancel_batch_by_id(
    provider: str,
    batch_id: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Cancel a batch by provider + bare batch id.

    Constructs a :class:`BatchHandle` and dispatches to the provider's
    ``BatchBackend.cancel()`` implementation. All provider-specific SDK
    details stay inside the backend.

    Returns ``True`` on successful cancellation, ``False`` on error.
    """
    handle = BatchHandle(
        provider=provider,
        batch_id=batch_id,
        metadata=metadata or {},
    )
    backend = get_batch_backend(provider)
    return backend.cancel(handle)


__all__ = ["cancel_batch_by_id"]
