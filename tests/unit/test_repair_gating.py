"""Tests for the daily-token-budget gate on synchronous repair (B1).

Verifies that when the budget exhausts mid-repair, the deferred pages leave no
result, ``check_and_wait_for_token_limit`` is invoked, and a second pass repairs
exactly the deferred pages.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from modules.batch import repair as repair_mod
from modules.batch.repair import Job, RepairTarget


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_sync_mode_defers_and_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = [
        "Page 1: [transcription error]",
        "Page 2: [transcription error]",
        "Page 3: [transcription error]",
    ]
    final = tmp_path / "doc.txt"
    final.write_text("\n".join(lines), encoding="utf-8")
    repair_jsonl = tmp_path / "repair.jsonl"
    repair_jsonl.touch()

    job = Job(
        parent_folder=tmp_path,
        identifier="doc",
        final_txt_path=final,
        temp_jsonl_path=None,
        kind="PDF",
    )

    targets = [
        RepairTarget(
            order_index=i,
            image_name=f"p{i}.jpg",
            image_path=None,
            custom_id=None,
            line_index=i,
            page_number=i + 1,
            image_base64="Zm9v",
            mime_type="image/jpeg",
        )
        for i in range(3)
    ]
    monkeypatch.setattr(repair_mod, "_resolve_repair_targets", lambda *a, **k: targets)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class _ConfigService:
        def get_concurrency_config(self) -> dict[str, Any]:
            return {
                "concurrency": {
                    "transcription": {
                        "concurrency_limit": 2,
                        "delay_between_tasks": 0,
                    }
                }
            }

    monkeypatch.setattr(repair_mod, "get_config_service", lambda: _ConfigService())

    @contextlib.asynccontextmanager
    async def _fake_open(**kwargs: Any):
        yield MagicMock()

    monkeypatch.setattr(repair_mod, "open_transcriber", _fake_open)
    monkeypatch.setattr(repair_mod, "get_token_tracker", lambda: MagicMock())

    waited: list[Any] = []

    async def _fake_wait(cfg: Any) -> bool:
        waited.append(cfg)
        return True

    monkeypatch.setattr(repair_mod, "check_and_wait_for_token_limit", _fake_wait)

    passes: list[list[int]] = []

    async def _fake_run(
        worker: Any,
        args_list: list[Any],
        *,
        concurrency_limit: int,
        delay: float,
        on_result: Any,
        tracker: Any,
        exhausted: Any,
    ) -> list[Any]:
        passes.append([a[0].line_index for a in args_list])
        if len(passes) == 1:
            # First pass: process the first page, exhaust the budget, defer rest.
            exhausted.set()
            first = args_list[0][0]
            res = (first.line_index, "fixed", {})
            await on_result(res)
            return [res, *[None] * (len(args_list) - 1)]
        # Subsequent pass: process every remaining (deferred) page.
        results = []
        for target, _trans in args_list:
            res = (target.line_index, "fixed", {})
            await on_result(res)
            results.append(res)
        return results

    monkeypatch.setattr(repair_mod, "run_concurrent_transcription_tasks", _fake_run)

    await repair_mod._repair_sync_mode(
        job=job,
        model_config={"transcription_model": {"name": "gpt-4o"}},
        image_entries=[],
        failure_indices=[0, 1, 2],
        final_lines=list(lines),
        repair_jsonl_path=repair_jsonl,
    )

    # The wait gate fired once after exhaustion.
    assert len(waited) == 1
    # Exactly two passes: the second covered only the two deferred pages.
    assert len(passes) == 2
    assert passes[0] == [0, 1, 2]
    assert passes[1] == [1, 2]

    # The final file was fully repaired (no placeholders remain).
    out = final.read_text(encoding="utf-8").splitlines()
    assert all("[transcription error]" not in line for line in out)
