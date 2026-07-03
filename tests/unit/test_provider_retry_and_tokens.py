"""Tests for the single-authority retry loop and token accounting in base.py.

Covers B1B work items: status-code-first classification, Retry-After honoring,
failed-attempt token recovery, and cache-token full-weight commit.
"""

from __future__ import annotations

import time
from datetime import UTC
from types import SimpleNamespace
from typing import Any

import pytest
import tenacity

from modules.llm.providers.base import (
    ANTHROPIC_TOKEN_MAPPING,
    OPENAI_TOKEN_MAPPING,
    BaseProvider,
    _classify_status,
    _commit_tokens_from_exception,
    parse_retry_after,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Headers:
    def __init__(self, retry_after: str | None) -> None:
        self._ra = retry_after

    def get(self, key: str) -> str | None:
        if key.lower() == "retry-after":
            return self._ra
        return None


class _Resp:
    def __init__(self, retry_after: str | None) -> None:
        self.headers = _Headers(retry_after)


def _make_api_error(
    status: int,
    *,
    retry_after: str | None = None,
    body: dict[str, Any] | None = None,
) -> Exception:
    exc = Exception(f"HTTP {status}")
    exc.status_code = status  # type: ignore[attr-defined]
    if retry_after is not None:
        exc.response = _Resp(retry_after)  # type: ignore[attr-defined]
    if body is not None:
        exc.body = body  # type: ignore[attr-defined]
    return exc


class _FakeLLM:
    """LLM whose ainvoke raises a queue of errors, then returns *final*."""

    def __init__(self, errors: list[Exception], final: Any) -> None:
        self.errors = list(errors)
        self.final = final
        self.calls = 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        i = self.calls
        self.calls += 1
        if i < len(self.errors):
            raise self.errors[i]
        return self.final


class _MiniProvider(BaseProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    def get_capabilities(self) -> Any:
        return None

    async def transcribe_image_from_base64(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        pass


@pytest.fixture
def provider() -> _MiniProvider:
    return _MiniProvider(api_key="k", model="gpt-4o")


@pytest.fixture
def zero_base_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the exponential-jitter base wait to 0 so Retry-After alone drives it."""
    monkeypatch.setattr(
        tenacity, "wait_exponential_jitter", lambda **kw: tenacity.wait_fixed(0)
    )


_OK = SimpleNamespace(content="ok")


# --------------------------------------------------------------------------- #
# _classify_status
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_classify_status_429() -> None:
    assert _classify_status(_make_api_error(429)) == (True, False)


@pytest.mark.unit
def test_classify_status_5xx() -> None:
    assert _classify_status(_make_api_error(503)) == (False, True)


@pytest.mark.unit
def test_classify_status_4xx_not_retryable() -> None:
    assert _classify_status(_make_api_error(400)) == (False, False)


@pytest.mark.unit
def test_classify_status_via_status_attr() -> None:
    exc = Exception("boom")
    exc.status = 429  # type: ignore[attr-defined]
    assert _classify_status(exc) == (True, False)


@pytest.mark.unit
def test_classify_status_none() -> None:
    assert _classify_status(Exception("no status")) == (False, False)


# --------------------------------------------------------------------------- #
# parse_retry_after
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_parse_retry_after_seconds() -> None:
    assert parse_retry_after(_make_api_error(429, retry_after="7")) == pytest.approx(
        7.0
    )


@pytest.mark.unit
def test_parse_retry_after_http_date_future() -> None:
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    future = datetime.now(UTC) + timedelta(seconds=30)
    exc = _make_api_error(429, retry_after=format_datetime(future))
    value = parse_retry_after(exc)
    assert value is not None
    assert 20 <= value <= 31


@pytest.mark.unit
def test_parse_retry_after_none_when_absent() -> None:
    assert parse_retry_after(Exception("no headers")) is None
    assert parse_retry_after(None) is None


@pytest.mark.unit
def test_parse_retry_after_top_level_headers() -> None:
    exc = Exception("boom")
    exc.headers = _Headers("3")  # type: ignore[attr-defined]
    assert parse_retry_after(exc) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Retry classification in the loop
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.asyncio
async def test_429_is_retried_to_success(
    provider: _MiniProvider, zero_base_wait: None
) -> None:
    llm = _FakeLLM([_make_api_error(429)], _OK)
    result = await provider._ainvoke_with_retry(llm, [])
    assert result is _OK
    assert llm.calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_5xx_is_retried_to_success(
    provider: _MiniProvider, zero_base_wait: None
) -> None:
    llm = _FakeLLM([_make_api_error(500)], _OK)
    result = await provider._ainvoke_with_retry(llm, [])
    assert result is _OK
    assert llm.calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_400_is_not_retried(
    provider: _MiniProvider, zero_base_wait: None
) -> None:
    err = _make_api_error(400)
    llm = _FakeLLM([err], _OK)
    with pytest.raises(Exception) as exc_info:
        await provider._ainvoke_with_retry(llm, [])
    assert exc_info.value is err
    assert llm.calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_after_honored_in_wait(
    provider: _MiniProvider, zero_base_wait: None
) -> None:
    # Base wait is 0; only the Retry-After of 0.25 s should drive the delay.
    llm = _FakeLLM([_make_api_error(429, retry_after="0.25")], _OK)
    start = time.monotonic()
    result = await provider._ainvoke_with_retry(llm, [])
    elapsed = time.monotonic() - start
    assert result is _OK
    assert elapsed >= 0.25


# --------------------------------------------------------------------------- #
# Failed-attempt token recovery
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_commit_tokens_from_exception_total(monkeypatch: pytest.MonkeyPatch) -> None:
    committed: list[int] = []
    monkeypatch.setattr(
        "modules.infra.token_budget.get_token_tracker",
        lambda: SimpleNamespace(add_tokens=committed.append),
    )
    _commit_tokens_from_exception(
        _make_api_error(429, body={"usage": {"total_tokens": 321}})
    )
    assert committed == [321]


@pytest.mark.unit
def test_commit_tokens_from_exception_prompt_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committed: list[int] = []
    monkeypatch.setattr(
        "modules.infra.token_budget.get_token_tracker",
        lambda: SimpleNamespace(add_tokens=committed.append),
    )
    _commit_tokens_from_exception(
        _make_api_error(
            429, body={"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        )
    )
    assert committed == [150]


@pytest.mark.unit
def test_commit_tokens_from_exception_never_raises() -> None:
    # No usage anywhere: silently does nothing.
    _commit_tokens_from_exception(Exception("bare"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retried_attempt_recovers_tokens(
    provider: _MiniProvider, zero_base_wait: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed: list[int] = []
    monkeypatch.setattr(
        "modules.infra.token_budget.get_token_tracker",
        lambda: SimpleNamespace(
            add_tokens=lambda n: committed.append(n),
            get_tokens_used_today=lambda: 0,
        ),
    )
    err = _make_api_error(429, body={"usage": {"total_tokens": 42}})
    llm = _FakeLLM([err], _OK)
    result = await provider._ainvoke_with_retry(llm, [])
    assert result is _OK
    # The failed 429 attempt's usage was recovered before the retry.
    assert 42 in committed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_failure_recovers_tokens(
    provider: _MiniProvider, zero_base_wait: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed: list[int] = []
    monkeypatch.setattr(
        "modules.infra.token_budget.get_token_tracker",
        lambda: SimpleNamespace(add_tokens=committed.append),
    )
    # A 400 is not retried; its usage must still be recovered in the terminal path.
    err = _make_api_error(400, body={"usage": {"input_tokens": 10, "output_tokens": 5}})
    llm = _FakeLLM([err], _OK)
    with pytest.raises(Exception):  # noqa: B017 - asserting the 400 propagates
        await provider._ainvoke_with_retry(llm, [])
    assert 15 in committed


# --------------------------------------------------------------------------- #
# Cache-token full-weight commit (no double counting)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.asyncio
async def test_anthropic_cache_committed_at_full_weight(
    provider: _MiniProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed: list[int] = []
    monkeypatch.setattr(
        "modules.infra.token_budget.get_token_tracker",
        lambda: SimpleNamespace(
            add_tokens=lambda n: committed.append(n),
            get_tokens_used_today=lambda: 0,
        ),
    )
    # Raw Anthropic shape: cache tokens are reported SEPARATELY from input_tokens.
    msg = SimpleNamespace(
        content="ok",
        response_metadata={
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 100,
            }
        },
        usage_metadata=None,
    )
    await provider._process_llm_response(msg, ANTHROPIC_TOKEN_MAPPING)
    # (1000 + 200) + 800 (read) + 100 (creation) = 2100.
    assert committed == [2100]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_cache_not_double_counted(
    provider: _MiniProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed: list[int] = []
    monkeypatch.setattr(
        "modules.infra.token_budget.get_token_tracker",
        lambda: SimpleNamespace(
            add_tokens=lambda n: committed.append(n),
            get_tokens_used_today=lambda: 0,
        ),
    )
    # OpenAI shape: cached_tokens is a SUBSET of prompt_tokens (already in total).
    msg = SimpleNamespace(
        content="ok",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "total_tokens": 2500,
                "prompt_tokens_details": {"cached_tokens": 1500},
            }
        },
        usage_metadata=None,
    )
    await provider._process_llm_response(msg, OPENAI_TOKEN_MAPPING)
    # Total already includes the cached subset; commit stays at 2500.
    assert committed == [2500]
