"""Unit tests for modules/llm/providers/http_timeouts.py.

Covers the per-phase httpx timeout builder: the read phase carries the
configured request timeout while connect/write/pool keep their own tight
budgets, config overrides are validated, and the documented unhashability of
``httpx.Timeout`` is pinned.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from modules.llm.providers.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    build_httpx_timeout,
)


def _with_transcription_config(cfg: dict[str, Any]) -> Any:
    """Patch the config service so ``concurrency.transcription`` is ``cfg``."""
    mock_cs = MagicMock()
    mock_cs.return_value.get_concurrency_config.return_value = {
        "concurrency": {"transcription": cfg}
    }
    return patch("modules.llm.providers.base.get_config_service", mock_cs)


@pytest.mark.unit
class TestBuildHttpxTimeout:
    def test_none_is_passed_through(self) -> None:
        """No configured timeout means the SDK defaults stay untouched."""
        with _with_transcription_config({}):
            assert build_httpx_timeout(None) is None

    def test_defaults_apply_to_the_other_phases(self) -> None:
        """The scalar lands on read only; connect/write/pool use defaults."""
        with _with_transcription_config({}):
            timeout = build_httpx_timeout(900)

        assert timeout is not None
        assert timeout.read == 900
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT == 10.0
        assert timeout.write == DEFAULT_WRITE_TIMEOUT == 30.0
        assert timeout.pool == DEFAULT_POOL_TIMEOUT == 30.0

    def test_config_overrides_are_honored(self) -> None:
        with _with_transcription_config(
            {"connect_timeout": 3, "write_timeout": 7.5, "pool_timeout": 12}
        ):
            timeout = build_httpx_timeout(120)

        assert timeout is not None
        assert timeout.read == 120
        assert timeout.connect == 3.0
        assert timeout.write == 7.5
        assert timeout.pool == 12.0

    @pytest.mark.parametrize("bad", [0, -1, "nonsense", None, True, False, [5]])
    def test_invalid_overrides_fall_back_to_defaults(self, bad: Any) -> None:
        """Zero, negative, non-numeric and bool overrides are ignored."""
        with _with_transcription_config(
            {"connect_timeout": bad, "write_timeout": bad, "pool_timeout": bad}
        ):
            timeout = build_httpx_timeout(900)

        assert timeout is not None
        assert timeout.read == 900
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
        assert timeout.write == DEFAULT_WRITE_TIMEOUT
        assert timeout.pool == DEFAULT_POOL_TIMEOUT

    def test_broken_config_service_falls_back_to_defaults(self) -> None:
        mock_cs = MagicMock()
        mock_cs.return_value.get_concurrency_config.side_effect = AttributeError("x")
        with patch("modules.llm.providers.base.get_config_service", mock_cs):
            timeout = build_httpx_timeout(60)

        assert timeout is not None
        assert timeout.read == 60
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT

    def test_timeout_is_unhashable(self) -> None:
        """httpx.Timeout is unhashable, bypassing langchain's client cache.

        Each ChatOpenAI therefore builds its own httpx client (accepted:
        providers are per-run). If this ever starts passing, httpx made
        Timeout hashable and the client-sharing semantics changed.
        """
        with _with_transcription_config({}):
            timeout = build_httpx_timeout(900)

        assert isinstance(timeout, httpx.Timeout)
        with pytest.raises(TypeError):
            hash(timeout)
