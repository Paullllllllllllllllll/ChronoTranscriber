"""Unit tests for modules.batch.cancel.cancel_batch_by_id.

This façade is a thin provider-agnostic wrapper around ``BatchBackend.cancel``.
We mock ``get_batch_backend`` so that no provider SDKs are touched and assert
that:

- the call is routed to the right backend;
- a :class:`BatchHandle` is constructed with the supplied provider, batch id,
  and optional metadata;
- the boolean return value of ``backend.cancel()`` is propagated unchanged;
- the abstraction behaves identically for ``openai``, ``anthropic``, and
  ``google`` providers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import modules.batch.cancel as cancel_mod
from modules.batch.backends import BatchHandle
from modules.batch.cancel import cancel_batch_by_id

# ---------------------------------------------------------------------------
# Basic success / failure propagation
# ---------------------------------------------------------------------------


class TestCancelBatchByIdReturnPropagation:
    @pytest.mark.unit
    def test_returns_true_when_backend_cancels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=True)
        monkeypatch.setattr(
            cancel_mod, "get_batch_backend", MagicMock(return_value=backend)
        )

        result = cancel_batch_by_id("openai", "batch_xxx")
        assert result is True
        backend.cancel.assert_called_once()

    @pytest.mark.unit
    def test_returns_false_when_backend_reports_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=False)
        monkeypatch.setattr(
            cancel_mod, "get_batch_backend", MagicMock(return_value=backend)
        )

        result = cancel_batch_by_id("openai", "batch_xxx")
        assert result is False


# ---------------------------------------------------------------------------
# BatchHandle construction
# ---------------------------------------------------------------------------


class TestCancelBatchByIdHandleConstruction:
    @pytest.mark.unit
    def test_default_metadata_is_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=True)
        monkeypatch.setattr(
            cancel_mod, "get_batch_backend", MagicMock(return_value=backend)
        )

        cancel_batch_by_id("openai", "batch_abc")

        (passed_handle,), _kwargs = backend.cancel.call_args
        assert isinstance(passed_handle, BatchHandle)
        assert passed_handle.provider == "openai"
        assert passed_handle.batch_id == "batch_abc"
        assert passed_handle.metadata == {}

    @pytest.mark.unit
    def test_metadata_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=True)
        monkeypatch.setattr(
            cancel_mod, "get_batch_backend", MagicMock(return_value=backend)
        )

        meta: dict[str, Any] = {"submitted_at": "2026-01-01", "job": "xyz"}
        cancel_batch_by_id("anthropic", "msgbatch_123", metadata=meta)

        (passed_handle,), _kwargs = backend.cancel.call_args
        assert passed_handle.metadata == meta
        assert passed_handle.provider == "anthropic"
        assert passed_handle.batch_id == "msgbatch_123"

    @pytest.mark.unit
    def test_get_batch_backend_called_with_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=True)
        get_mock = MagicMock(return_value=backend)
        monkeypatch.setattr(cancel_mod, "get_batch_backend", get_mock)

        cancel_batch_by_id("google", "bxg_789")

        get_mock.assert_called_once_with("google")


# ---------------------------------------------------------------------------
# Provider-agnostic behaviour
# ---------------------------------------------------------------------------


class TestCancelBatchByIdProviderAgnostic:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "provider,batch_id",
        [
            ("openai", "batch_oai_1"),
            ("anthropic", "msgbatch_1"),
            ("google", "bxg_1"),
        ],
    )
    def test_success_for_all_providers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        batch_id: str,
    ) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=True)
        monkeypatch.setattr(
            cancel_mod, "get_batch_backend", MagicMock(return_value=backend)
        )

        assert cancel_batch_by_id(provider, batch_id) is True

        (passed_handle,), _ = backend.cancel.call_args
        assert passed_handle.provider == provider
        assert passed_handle.batch_id == batch_id

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "provider,batch_id",
        [
            ("openai", "batch_oai_f"),
            ("anthropic", "msgbatch_f"),
            ("google", "bxg_f"),
        ],
    )
    def test_failure_for_all_providers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        batch_id: str,
    ) -> None:
        backend = MagicMock()
        backend.cancel = MagicMock(return_value=False)
        monkeypatch.setattr(
            cancel_mod, "get_batch_backend", MagicMock(return_value=backend)
        )

        assert cancel_batch_by_id(provider, batch_id) is False


# ---------------------------------------------------------------------------
# Item 6: cancel_batches infers the provider from the batch id's shape so a
# non-OpenAI id passed via --batch-ids is routed to the correct facade.
# ---------------------------------------------------------------------------


class TestInferProviderFromBatchId:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_id,expected",
        [
            ("msgbatch_abc123", "anthropic"),
            ("batches/projects/x/locations/us/batchPredictionJobs/9", "google"),
            ("batch_openai_1", "openai"),
            ("some_unknown_shape", "openai"),
        ],
    )
    def test_infers_provider(self, batch_id: str, expected: str) -> None:
        from main.cancel_batches import _infer_provider_from_batch_id

        assert _infer_provider_from_batch_id(batch_id) == expected


class TestCancelBatchesRoutesByIdShape:
    @pytest.mark.unit
    def test_explicit_ids_routed_to_correct_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from argparse import Namespace

        import main.cancel_batches as cbm

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            cbm,
            "cancel_batch_by_id",
            lambda provider, bid: calls.append((provider, bid)) or True,
        )

        args = Namespace(
            batch_ids=["msgbatch_1", "batches/g1", "batch_o1"],
            force=True,
            json_summary=False,
        )
        cbm.CancelBatchesScript().run_cli(args)

        assert ("anthropic", "msgbatch_1") in calls
        assert ("google", "batches/g1") in calls
        assert ("openai", "batch_o1") in calls
