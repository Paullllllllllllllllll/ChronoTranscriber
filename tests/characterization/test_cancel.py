"""Characterization of batch cancellation through the fake batch services.

Each case seeds one pending batch per provider and one completed OpenAI
batch, then runs ``cancel_batches --force``: with ``--batch-ids`` for the
pending batch of one provider, or without ids, which cancels every
non-terminal batch the listings of the three providers return. The snapshot adds
``batches.json`` (the state of every seeded batch after the run) and, in
``requests.json``, the client constructions and API calls. Each case runs
twice in fresh roots and is compared with ``golden/cancel/<case>/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import WriteGuard
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_batch import FakeBatchService, Provider
from tests.characterization.golden import (
    check_files,
    compare,
    normalizer_for,
    snapshot,
)

pytestmark = pytest.mark.integration

PREFIXES: dict[Provider, str] = {
    "openai": "batch_fake",
    "anthropic": "msgbatch_fake",
    "google": "batches/fake",
}
"""Id prefix of each provider's batches, shaped like the real ids."""

NEVER_COMPLETES = 1_000
"""Status calls before a fake batch completes; more than any case makes."""


@pytest.fixture
def batch_service(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> FakeBatchService:
    """Install fake batch services whose batches stay pending."""
    service = FakeBatchService(fake_llm, polls_until_complete=NEVER_COMPLETES)
    return service.install(monkeypatch)


def seed(service: FakeBatchService) -> dict[str, str]:
    """Seed the batches of a case; return their ids by role."""
    service.reset()
    ids = {
        provider: service.submit(provider, prefix, []).batch_id
        for provider, prefix in PREFIXES.items()
    }
    completed = service.submit("openai", PREFIXES["openai"], [])
    completed.completed = True
    ids["openai_completed"] = completed.batch_id
    return ids


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cancel(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    batch_ids: Sequence[str] | None,
) -> tuple[RunResult, dict[str, str]]:
    """Run ``cancel_batches --force``; return the result and its snapshot."""
    result = run_tool(
        "cancel_batches", llm=llm, root=root, batch_ids=batch_ids, force=True
    )
    replacements = service.id_replacements()
    files = snapshot(result, replacements=replacements)
    norm = normalizer_for(result, replacements)
    records = service.request_records()
    requests = json.loads(files["requests.json"])
    requests["batch"] = norm.data(
        {"clients": records["clients"], "calls": records["calls"]}
    )
    files["requests.json"] = _dumps(requests)
    files["batches.json"] = _dumps(norm.data(service.batch_states()))
    return result, files


def run_case(
    case: str,
    llm: FakeLLM,
    service: FakeBatchService,
    case_root: Callable[[str], Path],
    guard: WriteGuard,
    role: str | None,
) -> tuple[RunResult, list[dict[str, Any]]]:
    """Seed and cancel in two fresh roots; check the second run's golden.

    *role* names the seeded batch to cancel by id; None cancels without ids.
    """
    snapshots: list[dict[str, str]] = []
    for name in ("run1", "run2"):
        ids = seed(service)
        batch_ids = [ids[role]] if role is not None else None
        result, files = cancel(llm, service, case_root(name), batch_ids)
        snapshots.append(files)
    problems = compare(snapshots[0], snapshots[1])
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)
    check_files(f"cancel/{case}", snapshots[1], on_write=guard.consume)
    return result, service.batch_states()


def _states(batches: list[dict[str, Any]]) -> dict[str, str]:
    return {batch["batch_id"]: batch["state"] for batch in batches}


@pytest.mark.parametrize("provider", list(PREFIXES))
def test_cancel_by_id(
    provider: Provider,
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    result, batches = run_case(
        f"by_id_{provider}",
        fake_llm,
        batch_service,
        case_root,
        write_guard,
        provider,
    )

    assert result.exit_code == 0, result.stdout
    assert result.summary is not None
    assert (result.summary["cancelled"], result.summary["failed"]) == (1, 0)
    cancelled = [batch for batch in batches if batch["state"] == "cancelled"]
    assert [batch["provider"] for batch in cancelled] == [provider]
    assert cancelled[0]["batch_id"].startswith(PREFIXES[provider])


def test_cancel_all_non_terminal(
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    result, batches = run_case(
        "all_non_terminal",
        fake_llm,
        batch_service,
        case_root,
        write_guard,
        None,
    )

    assert result.exit_code == 0, result.stdout
    assert result.summary is not None
    assert (result.summary["cancelled"], result.summary["failed"]) == (3, 0)
    assert list(_states(batches).values()) == [
        "cancelled",
        "cancelled",
        "cancelled",
        "completed",
    ]
