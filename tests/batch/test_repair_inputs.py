"""Repair inputs: a file, a database, or a folder with databases below it."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber.batch import repair as repair_module
from chronotranscriber.batch.repair import RepairPlanError, plan_repairs
from chronotranscriber.output.store import database_path
from chronotranscriber.settings import Settings
from tests.batch.test_repair_store import (
    FakeTranscriber,
    page_name,
    spec_for,
    store_document,
    unit_rows,
)


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> type[FakeTranscriber]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "PageTranscriber", FakeTranscriber)
    FakeTranscriber.texts = {}
    FakeTranscriber.calls = []
    FakeTranscriber.created = []
    return FakeTranscriber


async def test_a_folder_repairs_every_database_below_it(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    transcriber.texts = {page_name(idx): f"text {idx}" for idx in range(2)}
    first = await store_document(tmp_path / "a", ["ok", "failed"])
    second = await store_document(tmp_path / "b", ["failed", "failed"])
    await store_document(tmp_path / "c", ["ok"])

    repairs = plan_repairs(spec_for(tmp_path), Settings())

    assert [plan.failures for plan in repairs.plans] == [[1], [0, 1]]
    result = await repair_module.run_repairs(repairs)
    assert (result.selected, result.repaired, result.failed) == (3, 3, 0)
    assert result.documents == 2
    assert set(result.databases) == {first.db, second.db}
    assert [row[1] for row in unit_rows(second.db)] == ["done", "done"]


@pytest.mark.usefixtures("transcriber")
async def test_a_database_repairs_its_documents(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, ["failed", "ok"])

    repairs = plan_repairs(spec_for(database_path(stored.root)), Settings())

    assert [plan.failures for plan in repairs.plans] == [[0]]


@pytest.mark.usefixtures("transcriber")
async def test_a_file_names_its_one_document(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, ["failed", "ok"])

    repairs = plan_repairs(spec_for(stored.output), Settings())

    (plan,) = repairs.plans
    assert plan.document is not None and plan.document.doc_id == stored.doc_id


@pytest.mark.usefixtures("transcriber")
async def test_indices_need_one_document(tmp_path: Path) -> None:
    await store_document(tmp_path / "a", ["failed"])
    await store_document(tmp_path / "b", ["failed"])

    with pytest.raises(RepairPlanError, match="one document"):
        plan_repairs(spec_for(tmp_path, indices="0"), Settings())


@pytest.mark.usefixtures("transcriber")
async def test_clean_documents_leave_nothing_to_repair(tmp_path: Path) -> None:
    await store_document(tmp_path / "a", ["ok"])
    await store_document(tmp_path / "b", ["ok"])

    repairs = plan_repairs(spec_for(tmp_path), Settings())

    assert not repairs.to_repair
    assert repairs.message == "No failed pages detected matching your criteria."


def test_a_folder_without_a_database_has_nothing_to_repair(tmp_path: Path) -> None:
    repairs = plan_repairs(spec_for(tmp_path), Settings())

    assert not repairs.to_repair
    assert repairs.message is not None and "No stored document" in repairs.message


async def test_a_dry_run_plans_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    await store_document(tmp_path, ["failed"])

    repairs = plan_repairs(spec_for(tmp_path, "dry_run"), Settings())

    assert repairs.to_repair
    assert any("OPENAI_API_KEY" in warning for warning in repairs.warnings)
    with pytest.raises(RepairPlanError, match="OPENAI_API_KEY"):
        plan_repairs(spec_for(tmp_path), Settings())


async def test_the_cli_repairs_the_current_folder_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    await store_document(tmp_path / "a", ["failed"])
    await store_document(tmp_path / "b", ["failed"])
    monkeypatch.chdir(tmp_path)

    code = entry.main(["repair", "--model", "gpt-5.6-luna", "--dry-run"])

    assert code == 0
    assert "2 page(s) of 2 documents would be repaired" in capsys.readouterr().err
