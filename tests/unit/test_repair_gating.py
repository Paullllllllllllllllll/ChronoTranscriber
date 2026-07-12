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

    waited: list[bool] = []

    async def _fake_wait(
        cfg: Any, reservation_aware: bool = False, stamp: Any = None
    ) -> bool:
        # Record the reservation_aware flag: the sync repair loop must pass True
        # so the wait actually waits while pages are reservation-blocked near the
        # cap instead of spinning and aborting (CT-8, mirrors the manager fix).
        waited.append(reservation_aware)
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
        stamp: Any = None,
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

    # The wait gate fired once after exhaustion, and it was reservation-aware.
    assert waited == [True]
    # Exactly two passes: the second covered only the two deferred pages.
    assert len(passes) == 2
    assert passes[0] == [0, 1, 2]
    assert passes[1] == [1, 2]

    # The final file was fully repaired (no placeholders remain).
    out = final.read_text(encoding="utf-8").splitlines()
    assert all("[transcription error]" not in line for line in out)


# ---------------------------------------------------------------------------
# Single-line marker file: truthful outcome + no false success
# (the live incident: a one-page images-mode txt whose only line is a
# placeholder, written without a trailing newline).
# ---------------------------------------------------------------------------


def _install_sync_stubs(monkeypatch: pytest.MonkeyPatch, worker_text: str) -> None:
    """Stub the LLM boundary so ``_repair_sync_mode`` runs offline.

    ``worker_text`` is the text every resolved target "transcribes" to; pass a
    real string to simulate a successful repair or ``"[transcription error]"``
    to simulate the re-transcription failing again.
    """
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

    async def _fake_run(
        worker: Any,
        args_list: list[Any],
        *,
        concurrency_limit: int,
        delay: float,
        on_result: Any,
        tracker: Any,
        exhausted: Any,
        stamp: Any = None,
    ) -> list[Any]:
        results = []
        for target, _trans in args_list:
            res = (target.line_index, worker_text, {})
            await on_result(res)
            results.append(res)
        return results

    monkeypatch.setattr(repair_mod, "run_concurrent_transcription_tasks", _fake_run)


_PREFIX_MARKER = "7126_recueil_p47.png_pre_processed.jpg: [transcription error]"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_line_prefix_marker_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-line file (no trailing newline) whose only line is a
    filename-prefix placeholder is spliced correctly and reports success."""
    final = tmp_path / "7126_recueil.txt"
    final.write_text(_PREFIX_MARKER, encoding="utf-8")  # no trailing newline
    assert "\n" not in final.read_text(encoding="utf-8")
    repair_jsonl = tmp_path / "repair.jsonl"
    repair_jsonl.touch()

    job = Job(
        parent_folder=tmp_path,
        identifier="7126_recueil",
        final_txt_path=final,
        temp_jsonl_path=None,
        kind="Images",
    )
    target = RepairTarget(
        order_index=0,
        image_name="7126_recueil_p47.png_pre_processed.jpg",
        image_path=None,
        custom_id=None,
        line_index=0,
        page_number=1,
        image_base64="Zm9v",
        mime_type="image/jpeg",
    )
    monkeypatch.setattr(repair_mod, "_resolve_repair_targets", lambda *a, **k: [target])
    _install_sync_stubs(monkeypatch, worker_text="Recueil de deux traites. Real text.")

    repaired, failed = await repair_mod._repair_sync_mode(
        job=job,
        model_config={"transcription_model": {"provider": "openai", "name": "gpt-4o"}},
        image_entries=[],
        failure_indices=[0],
        final_lines=[_PREFIX_MARKER],
        repair_jsonl_path=repair_jsonl,
    )

    assert (repaired, failed) == (1, 0)
    out = final.read_text(encoding="utf-8")
    assert "[transcription error]" not in out
    assert out == "Recueil de deux traites. Real text."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_line_prefix_marker_unresolved_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live incident: the single failure line cannot be resolved to an
    image, so nothing is written back. The tool must report the line as failed
    (not a silent ``(0, 0)`` success) and leave the marker intact."""
    final = tmp_path / "7126_recueil.txt"
    final.write_text(_PREFIX_MARKER, encoding="utf-8")  # no trailing newline
    repair_jsonl = tmp_path / "repair.jsonl"
    repair_jsonl.touch()

    job = Job(
        parent_folder=tmp_path,
        identifier="7126_recueil",
        final_txt_path=final,
        temp_jsonl_path=None,
        kind="Images",
    )
    # Simulate resolution failure (missing preprocessed image AND missing
    # source file), which drops every target.
    monkeypatch.setattr(repair_mod, "_resolve_repair_targets", lambda *a, **k: [])

    repaired, failed = await repair_mod._repair_sync_mode(
        job=job,
        model_config={"transcription_model": {"provider": "openai", "name": "gpt-4o"}},
        image_entries=[],
        failure_indices=[0],
        final_lines=[_PREFIX_MARKER],
        repair_jsonl_path=repair_jsonl,
    )

    assert (repaired, failed) == (0, 1)
    # File untouched; the placeholder is still present.
    assert final.read_text(encoding="utf-8") == _PREFIX_MARKER


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_line_marker_refailed_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When re-transcription fails again the placeholder is re-emitted; the
    outcome must be reported as failed, not success."""
    final = tmp_path / "7126_recueil.txt"
    final.write_text(_PREFIX_MARKER, encoding="utf-8")
    repair_jsonl = tmp_path / "repair.jsonl"
    repair_jsonl.touch()

    job = Job(
        parent_folder=tmp_path,
        identifier="7126_recueil",
        final_txt_path=final,
        temp_jsonl_path=None,
        kind="Images",
    )
    target = RepairTarget(
        order_index=0,
        image_name="7126_recueil_p47.png_pre_processed.jpg",
        image_path=None,
        custom_id=None,
        line_index=0,
        page_number=1,
        image_base64="Zm9v",
        mime_type="image/jpeg",
    )
    monkeypatch.setattr(repair_mod, "_resolve_repair_targets", lambda *a, **k: [target])
    _install_sync_stubs(monkeypatch, worker_text="[transcription error]")

    repaired, failed = await repair_mod._repair_sync_mode(
        job=job,
        model_config={"transcription_model": {"provider": "openai", "name": "gpt-4o"}},
        image_entries=[],
        failure_indices=[0],
        final_lines=[_PREFIX_MARKER],
        repair_jsonl_path=repair_jsonl,
    )

    assert (repaired, failed) == (0, 1)
    assert "[transcription error]" in final.read_text(encoding="utf-8")
