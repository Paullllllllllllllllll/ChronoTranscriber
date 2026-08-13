"""Unit tests for timeout classification, retry budget, and the page watchdog.

Covers the defects seen in a stall incident: a read-timed-out request was
retried on the full transient budget, and nothing bounded the wall-clock time
one page could spend across all of its attempts.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import tenacity

from modules.config.capabilities import Capabilities
from modules.llm.providers.base import (
    BaseProvider,
    PageTimeoutError,
    is_timeout_error,
    load_page_timeout,
    load_timeout_attempts,
)


def _make_provider() -> Any:
    """Create a minimal concrete provider without running __init__."""

    class _ConcreteProvider(BaseProvider):
        @property
        def provider_name(self):
            return "test"

        def get_capabilities(self):
            return Capabilities(model="m", family="test")

        async def transcribe_image_from_base64(self, *a, **kw):
            pass

        async def close(self):
            pass

    return _ConcreteProvider.__new__(_ConcreteProvider)


def _instant_capacity() -> Any:
    """Patch out the rate limiter's threaded wait.

    ``await_capacity`` hops through ``asyncio.to_thread``; on a loaded machine
    that hop alone can cost more than the sub-second ceilings used below, which
    would make the watchdog fire before the first attempt even starts.
    """
    return patch(
        "modules.infra.rate_limit.await_capacity",
        new=AsyncMock(return_value=0.0),
    )


def _cfg(**transcription: Any) -> dict[str, Any]:
    """Build a concurrency config dict with the given transcription keys."""
    return {"concurrency": {"transcription": transcription}}


class TestIsTimeoutError:
    """Tests for the timeout classifier."""

    @pytest.mark.unit
    def test_read_timeout(self) -> None:
        """A bare httpx.ReadTimeout is a timeout."""
        assert is_timeout_error(httpx.ReadTimeout("read timed out")) is True

    @pytest.mark.unit
    def test_connect_timeout(self) -> None:
        """httpx.ConnectTimeout is a timeout (it subclasses TimeoutException)."""
        assert is_timeout_error(httpx.ConnectTimeout("connect timed out")) is True

    @pytest.mark.unit
    def test_sdk_wrapped_timeout(self) -> None:
        """A SDK error raised ``from`` an httpx timeout is found via __cause__."""
        try:
            try:
                raise httpx.ReadTimeout("read timed out")
            except httpx.ReadTimeout as inner:
                raise RuntimeError("Request timed out.") from inner
        except RuntimeError as exc:
            assert is_timeout_error(exc) is True

    @pytest.mark.unit
    def test_class_name_fallback_without_cause(self) -> None:
        """A bare SDK timeout type with no httpx cause matches on class name."""

        class APITimeoutError(Exception):
            pass

        assert is_timeout_error(APITimeoutError("timed out")) is True

    @pytest.mark.unit
    def test_builtin_timeout_error(self) -> None:
        """The builtin TimeoutError is a timeout."""
        assert is_timeout_error(TimeoutError("deadline")) is True

    @pytest.mark.unit
    def test_connect_error_is_not_timeout(self) -> None:
        """httpx.ConnectError is a connection failure, not a timeout."""
        assert is_timeout_error(httpx.ConnectError("refused")) is False

    @pytest.mark.unit
    def test_unrelated_error_is_not_timeout(self) -> None:
        """An unrelated exception is not a timeout."""
        assert is_timeout_error(ValueError("nope")) is False

    @pytest.mark.unit
    def test_cause_cycle_is_safe(self) -> None:
        """A cyclic __cause__ chain terminates instead of looping forever."""
        first = ValueError("a")
        second = ValueError("b")
        first.__cause__ = second
        second.__cause__ = first

        assert is_timeout_error(first) is False


class TestTimeoutRetryBudget:
    """Timeouts get their own, smaller retry budget."""

    async def _run(self, mock_llm: Any, cfg: dict[str, Any]) -> Any:
        provider = _make_provider()
        with (
            patch("modules.llm.providers.base.get_config_service") as mock_cs,
            patch(
                "tenacity.wait_exponential_jitter",
                return_value=tenacity.wait_none(),
            ),
        ):
            mock_cs.return_value.get_concurrency_config.return_value = cfg
            return await provider._ainvoke_with_retry(mock_llm, ["msg"])

    @pytest.mark.unit
    async def test_timeouts_stop_at_timeout_attempts(self) -> None:
        """A perpetually timing-out call stops after timeout_attempts tries."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        cfg = _cfg(
            page_timeout="off",
            retry={"attempts": 8, "timeout_attempts": 3},
        )

        with pytest.raises(httpx.ReadTimeout):
            await self._run(mock_llm, cfg)

        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.unit
    async def test_connection_errors_keep_full_budget(self) -> None:
        """A cheap ConnectError still gets the full transient attempt budget."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=httpx.ConnectError("refused"))
        cfg = _cfg(
            page_timeout="off",
            retry={"attempts": 8, "timeout_attempts": 3},
        )

        with pytest.raises(httpx.ConnectError):
            await self._run(mock_llm, cfg)

        assert mock_llm.ainvoke.call_count == 8

    @pytest.mark.unit
    async def test_timeout_attempts_clamped_to_attempts(self) -> None:
        """timeout_attempts above attempts is clamped to attempts."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        cfg = _cfg(
            page_timeout="off",
            retry={"attempts": 2, "timeout_attempts": 9},
        )

        with pytest.raises(httpx.ReadTimeout):
            await self._run(mock_llm, cfg)

        assert mock_llm.ainvoke.call_count == 2


class TestPageWatchdog:
    """The per-page wall-clock ceiling around the whole retry loop."""

    @pytest.mark.unit
    async def test_slow_call_raises_page_timeout_error(self) -> None:
        """A call that outlives the ceiling becomes a PageTimeoutError."""
        provider = _make_provider()
        mock_llm = MagicMock()

        async def _slow(*a: Any, **kw: Any) -> str:
            await asyncio.sleep(10)
            return "never"

        mock_llm.ainvoke = AsyncMock(side_effect=_slow)

        from modules.llm.providers.base import call_label

        with (
            patch("modules.llm.providers.base.load_page_timeout", return_value=0.2),
            _instant_capacity(),
            patch("modules.llm.providers.base.load_max_retries", return_value=3),
            pytest.raises(PageTimeoutError) as exc_info,
            call_label("page_0007.png"),
        ):
            await provider._ainvoke_with_retry(mock_llm, ["msg"])

        assert exc_info.value.label == "page_0007.png"
        assert exc_info.value.seconds == 0.2

    @pytest.mark.unit
    @pytest.mark.parametrize("disabled", [False, 0, "off"])
    async def test_disabled_values_skip_the_watchdog(self, disabled: Any) -> None:
        """False / 0 / "off" disable the watchdog and the call completes."""
        provider = _make_provider()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="ok")
        cfg = _cfg(page_timeout=disabled, retry={"attempts": 2})

        with patch("modules.llm.providers.base.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = cfg
            assert load_page_timeout() is None
            result = await provider._ainvoke_with_retry(mock_llm, ["msg"])

        assert result == "ok"

    @pytest.mark.unit
    async def test_cancelled_error_wrapping_timeout_is_not_retried(self) -> None:
        """Regression: a CancelledError carrying a ReadTimeout must not retry.

        tenacity swallows BaseException into the retry predicate, and a
        CancelledError raised while an httpx.ReadTimeout is propagating carries
        that timeout in ``__context__`` — without the guard the classifier
        would match it and retry past the watchdog's deadline.
        """
        provider = _make_provider()
        mock_llm = MagicMock()

        async def _hang_then_cancel(*a: Any, **kw: Any) -> str:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                try:
                    raise httpx.ReadTimeout("read timed out")
                except httpx.ReadTimeout:
                    # __context__ now holds the in-flight read timeout.
                    raise asyncio.CancelledError from None
            return "never"

        mock_llm.ainvoke = AsyncMock(side_effect=_hang_then_cancel)

        with (
            patch("modules.llm.providers.base.load_page_timeout", return_value=0.2),
            _instant_capacity(),
            patch("modules.llm.providers.base.load_max_retries", return_value=8),
            patch(
                "tenacity.wait_exponential_jitter",
                return_value=tenacity.wait_none(),
            ),
            pytest.raises(PageTimeoutError),
        ):
            await provider._ainvoke_with_retry(mock_llm, ["msg"])

        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.unit
    async def test_watchdog_fires_during_backoff_sleep(self) -> None:
        """The ceiling also bounds time spent asleep between attempts."""
        provider = _make_provider()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with (
            patch("modules.llm.providers.base.load_page_timeout", return_value=0.3),
            _instant_capacity(),
            patch("modules.llm.providers.base.load_max_retries", return_value=8),
            patch(
                "tenacity.wait_exponential_jitter",
                return_value=tenacity.wait_fixed(5),
            ),
            pytest.raises(PageTimeoutError),
        ):
            await provider._ainvoke_with_retry(mock_llm, ["msg"])

        # The first attempt failed fast; the watchdog fired inside the backoff.
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.unit
    async def test_unlabelled_page_uses_placeholder(self) -> None:
        """Without a bound label the error names ``<unknown page>``."""
        provider = _make_provider()
        mock_llm = MagicMock()

        async def _slow(*a: Any, **kw: Any) -> str:
            await asyncio.sleep(10)
            return "never"

        mock_llm.ainvoke = AsyncMock(side_effect=_slow)

        with (
            patch("modules.llm.providers.base.load_page_timeout", return_value=0.2),
            _instant_capacity(),
            patch("modules.llm.providers.base.load_max_retries", return_value=3),
            pytest.raises(PageTimeoutError) as exc_info,
        ):
            await provider._ainvoke_with_retry(mock_llm, ["msg"])

        assert exc_info.value.label == "<unknown page>"


class TestUserCancellation:
    """External cancellation must stay cancellation."""

    @pytest.mark.unit
    async def test_external_cancel_propagates(self) -> None:
        """task.cancel() propagates CancelledError, not PageTimeoutError."""
        provider = _make_provider()
        mock_llm = MagicMock()
        started = asyncio.Event()

        async def _hang(*a: Any, **kw: Any) -> str:
            started.set()
            await asyncio.sleep(30)
            return "never"

        mock_llm.ainvoke = AsyncMock(side_effect=_hang)

        with (
            patch("modules.llm.providers.base.load_page_timeout", return_value=30.0),
            _instant_capacity(),
            patch("modules.llm.providers.base.load_max_retries", return_value=3),
        ):
            task = asyncio.create_task(provider._ainvoke_with_retry(mock_llm, ["msg"]))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert task.cancelled()


class TestLoadPageTimeout:
    """Tests for the page_timeout config loader."""

    def _load(self, cfg: dict[str, Any]) -> float | None:
        with patch("modules.llm.providers.base.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = cfg
            return load_page_timeout()

    @pytest.mark.unit
    def test_auto_math_when_absent(self) -> None:
        """Absent key means auto: request_timeout * timeout_attempts + 300."""
        cfg = _cfg(request_timeout=900, retry={"timeout_attempts": 3, "attempts": 8})
        assert self._load(cfg) == 3000.0

    @pytest.mark.unit
    def test_explicit_auto_string(self) -> None:
        """The literal string "auto" produces the same value."""
        cfg = _cfg(
            request_timeout=900,
            page_timeout="auto",
            retry={"timeout_attempts": 3, "attempts": 8},
        )
        assert self._load(cfg) == 3000.0

    @pytest.mark.unit
    def test_auto_falls_back_when_request_timeout_garbage(self) -> None:
        """A non-numeric request_timeout falls back to 900 s."""
        cfg = _cfg(
            request_timeout="not-a-number",
            page_timeout="auto",
            retry={"timeout_attempts": 3, "attempts": 8},
        )
        assert self._load(cfg) == 3000.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value", [False, None, 0, "off", "none", "disabled", "OFF", " Disabled "]
    )
    def test_disabled_values(self, value: Any) -> None:
        """Falsy and explicit off-switches disable the watchdog."""
        cfg = _cfg(page_timeout=value, retry={"attempts": 8})
        assert self._load(cfg) is None

    @pytest.mark.unit
    def test_explicit_float(self) -> None:
        """An explicit number is used verbatim."""
        cfg = _cfg(page_timeout=1234.5, retry={"attempts": 8})
        assert self._load(cfg) == 1234.5

    @pytest.mark.unit
    def test_numeric_string(self) -> None:
        """A numeric string is coerced to float."""
        cfg = _cfg(page_timeout="600", retry={"attempts": 8})
        assert self._load(cfg) == 600.0

    @pytest.mark.unit
    def test_negative_disables(self) -> None:
        """A non-positive explicit value disables the watchdog."""
        cfg = _cfg(page_timeout=-5, retry={"attempts": 8})
        assert self._load(cfg) is None

    @pytest.mark.unit
    def test_garbage_string_falls_back_to_auto(self) -> None:
        """An unrecognised string behaves like "auto"."""
        cfg = _cfg(
            request_timeout=900,
            page_timeout="soonish",
            retry={"timeout_attempts": 3, "attempts": 8},
        )
        assert self._load(cfg) == 3000.0

    @pytest.mark.unit
    def test_garbage_type_disables(self) -> None:
        """An unusable type disables the watchdog rather than guessing."""
        cfg = _cfg(page_timeout=["nope"], retry={"attempts": 8})
        assert self._load(cfg) is None

    @pytest.mark.unit
    def test_returns_none_on_config_exception(self) -> None:
        """A broken config service disables the watchdog instead of raising."""
        with patch(
            "modules.llm.providers.base.get_config_service",
            side_effect=AttributeError("config unavailable"),
        ):
            assert load_page_timeout() is None


class TestLoadTimeoutAttempts:
    """Tests for the timeout_attempts config loader."""

    def _load(self, cfg: dict[str, Any]) -> int:
        with patch("modules.llm.providers.base.get_config_service") as mock_cs:
            mock_cs.return_value.get_concurrency_config.return_value = cfg
            return load_timeout_attempts()

    @pytest.mark.unit
    def test_defaults_to_3(self) -> None:
        """Returns 3 when the key is absent."""
        assert self._load(_cfg(retry={"attempts": 8})) == 3

    @pytest.mark.unit
    def test_reads_configured_value(self) -> None:
        """Reads timeout_attempts from concurrency.transcription.retry."""
        assert self._load(_cfg(retry={"attempts": 8, "timeout_attempts": 5})) == 5

    @pytest.mark.unit
    def test_clamped_to_attempts(self) -> None:
        """Never exceeds the general attempts budget."""
        assert self._load(_cfg(retry={"attempts": 2, "timeout_attempts": 9})) == 2

    @pytest.mark.unit
    def test_minimum_is_1(self) -> None:
        """Returns at least 1 even when configured as 0."""
        assert self._load(_cfg(retry={"attempts": 8, "timeout_attempts": 0})) == 1

    @pytest.mark.unit
    def test_returns_3_on_exception(self) -> None:
        """Returns 3 when the config service raises."""
        with patch(
            "modules.llm.providers.base.get_config_service",
            side_effect=AttributeError("config unavailable"),
        ):
            assert load_timeout_attempts() == 3
