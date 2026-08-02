"""Retry wrapper for plain SDK calls made by the audio backends.

The chat providers route every request through
:meth:`modules.llm.providers.base.BaseProvider._ainvoke_with_retry`, but the
OpenAI speech-to-text endpoint is called through the raw SDK rather than
LangChain, so it needs the same policy applied to a bare coroutine factory.

This is that policy, minus everything chat-specific (structured-output
validation, content-quality checks, input-token floors): shared rate limiter,
status-code-first classification, ``Retry-After`` as a backoff floor, and the
attempt budget from the same ``concurrency.transcription.retry`` block the chat
path reads.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from modules.infra.logger import setup_logger
from modules.llm.providers.base import (
    classify_http_status,
    is_connection_error,
    load_max_retries,
    parse_retry_after,
)

logger = setup_logger(__name__)

# Matches the chat path's tenacity.wait_exponential_jitter(initial=2, max=120).
_BACKOFF_INITIAL_S = 2
_BACKOFF_MAX_S = 120


async def _call_once[T](fn: Callable[[], Awaitable[T]], provider_name: str) -> T:
    """Run *fn* once through the provider's shared client-side rate limiter.

    Mirrors :meth:`BaseProvider._ainvoke_once`: acquire capacity off the event
    loop immediately before the call, then feed the outcome back so the
    adaptive multiplier reacts to 429s and 5xx the same way it does for chat.
    """
    from modules.infra.rate_limit import await_capacity, get_shared_rate_limiter

    limiter = get_shared_rate_limiter(provider_name)
    await await_capacity(limiter)
    try:
        result = await fn()
    except BaseException as exc:
        is_rate_limit, is_server_error = classify_http_status(exc)
        limiter.report_error(is_rate_limit=is_rate_limit or is_server_error)
        raise
    limiter.report_success()
    return result


async def acall_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    provider_name: str,
) -> T:
    """Await *fn* with the shared transient-failure retry policy.

    Retryable: HTTP 429 (rate limit) and 5xx, plus connection and timeout
    failures detected through the exception's ``__cause__`` chain. Everything
    else fails immediately.

    Backoff is exponential with jitter (floor 2 s, cap 120 s), raised to at
    least a server-sent ``Retry-After``. The attempt budget comes from
    ``concurrency.transcription.retry.attempts``.

    Args:
        fn: Zero-argument coroutine factory performing one API call. It must be
            safe to call repeatedly (build request kwargs outside, not inside).
        provider_name: Rate-limiter bucket, e.g. ``"openai"``.

    Returns:
        Whatever *fn* resolves to.

    Raises:
        BaseException: The last attempt's exception, re-raised unchanged.
    """
    import tenacity

    max_attempts = load_max_retries()

    def _should_retry(exc: BaseException) -> bool:
        if is_connection_error(exc):
            return True
        is_rate_limit, is_server_error = classify_http_status(exc)
        if is_rate_limit or is_server_error:
            logger.warning(
                "Transient API %s error, retrying: %s",
                "rate-limit (429)" if is_rate_limit else "server (5xx)",
                str(exc)[:200],
            )
            return True
        return False

    _base_wait = tenacity.wait_exponential_jitter(
        initial=_BACKOFF_INITIAL_S, max=_BACKOFF_MAX_S
    )

    def _wait(retry_state: tenacity.RetryCallState) -> float:
        """Exponential-jitter backoff, raised to honor Retry-After (cap 120s)."""
        computed = float(_base_wait(retry_state))
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = parse_retry_after(exc)
        if retry_after is not None:
            return min(float(_BACKOFF_MAX_S), max(computed, float(retry_after)))
        return computed

    async for attempt in tenacity.AsyncRetrying(
        retry=tenacity.retry_if_exception(_should_retry),
        wait=_wait,
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return await _call_once(fn, provider_name)
    # Unreachable: the retry loop always returns or raises.
    raise RuntimeError("retry loop exited without a result")


__all__ = ["acall_with_retry"]
