"""Per-phase HTTP timeouts for the OpenAI-family providers.

A scalar float handed to an httpx-backed client replaces ALL FOUR httpx
timeout phases (connect, read, write, pool), so a 900 s read budget silently
becomes a 900 s *connect* budget and a dead peer goes undetected for fifteen
minutes. :func:`build_httpx_timeout` keeps the configured value on the read
phase only and gives connect/write/pool their own, much tighter budgets.

Only the ``ChatOpenAI``-based providers (openai, openrouter, custom) and the
OpenAI audio backend get an :class:`httpx.Timeout`. The Anthropic and Google
LangChain wrappers require a plain float -- ``ChatAnthropic`` compares its
timeout with ``> 0`` and ``ChatGoogleGenerativeAI`` computes
``int(timeout * 1000)`` -- and are deliberately left on the scalar.

Note: ``httpx.Timeout`` is unhashable, which bypasses langchain-openai's
``lru_cache``-based httpx client sharing, so each chat model builds its own
client. That is accepted here: providers are constructed once per run.
"""

from __future__ import annotations

from typing import Any

import httpx

from modules.llm.providers.base import _load_transcription_config

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 30.0


def _positive_float(value: Any, fallback: float) -> float:
    """Coerce a config value to a positive float, else return ``fallback``.

    Booleans are rejected explicitly (``isinstance(True, int)`` is ``True``),
    as are non-numeric values and anything at or below zero.
    """
    if isinstance(value, bool):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def build_httpx_timeout(read_timeout: float | None) -> httpx.Timeout | None:
    """Build an :class:`httpx.Timeout` whose read phase is ``read_timeout``.

    Args:
        read_timeout: The configured request timeout in seconds, or ``None``
            to keep the SDK's own defaults.

    Returns:
        ``None`` when ``read_timeout`` is ``None``; otherwise a timeout whose
        positional default (and hence the read phase) is ``read_timeout``,
        with connect/write/pool taken from the ``concurrency.transcription``
        keys ``connect_timeout`` / ``write_timeout`` / ``pool_timeout``.
    """
    if read_timeout is None:
        return None

    try:
        trans = _load_transcription_config()
    except (KeyError, AttributeError, TypeError, ValueError):
        trans = {}

    connect = _positive_float(trans.get("connect_timeout"), DEFAULT_CONNECT_TIMEOUT)
    write = _positive_float(trans.get("write_timeout"), DEFAULT_WRITE_TIMEOUT)
    pool = _positive_float(trans.get("pool_timeout"), DEFAULT_POOL_TIMEOUT)

    # The positional argument is httpx's `default`; `read` is intentionally not
    # overridden so it inherits it -- that is the whole point of this helper.
    return httpx.Timeout(float(read_timeout), connect=connect, write=write, pool=pool)
