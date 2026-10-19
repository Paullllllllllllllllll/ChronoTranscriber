"""Repair selection on the store: causes, indices, leases and open batches.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_repair.py -q

The documents are built by ``tests.batch.test_repair_store``; the plan reads
the store through read-only connections and writes nothing.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.batch import repair as repair_module
from chronotranscriber.batch.repair import RepairPlanError, plan_repair
from chronotranscriber.common.store import UnitKey
from chronotranscriber.output.store import open_store
from chronotranscriber.settings import Settings, Tools
from chronotranscriber.sources.audio.ffmpeg import (
    configure_ffmpeg_executables,
    get_ffmpeg_command,
)
from tests.batch.test_repair_store import (
    FakeTranscriber,
    rows,
    spec_for,
    store_document,
    tree,
    unit_rows,
)

OUTCOMES = ["failed", "not_possible", "no_text", "ok", "failed"]


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> type[FakeTranscriber]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "PageTranscriber", FakeTranscriber)
    FakeTranscriber.texts = {}
    FakeTranscriber.calls = []
    FakeTranscriber.created = []
    return FakeTranscriber


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize(
    ("flags", "selected"),
    [
        ((), [0, 4]),
        (("errors_only",), [0, 4]),
        (("not_possible",), [1]),
        (("no_text",), [2]),
        (("not_possible", "no_text"), [1, 2]),
        (("all_failures",), [0, 1, 2, 4]),
    ],
)
async def test_each_cause_selects_its_outcome(
    tmp_path: Path, flags: tuple[str, ...], selected: list[int]
) -> None:
    stored = await store_document(tmp_path, OUTCOMES)

    plan = plan_repair(spec_for(stored.output, *flags), Settings())

    assert plan.to_repair and plan.failures == selected


@pytest.mark.usefixtures("key")
async def test_indices_narrow_the_units(tmp_path: Path) -> None:
    stored = await store_document(tmp_path, OUTCOMES)

    plan = plan_repair(spec_for(stored.output, indices="3-4"), Settings())
    assert plan.failures == [4]

    plan = plan_repair(spec_for(stored.output, indices="3"), Settings())
    assert not plan.to_repair
    assert plan.message == "None of the specified indices matched detected failures."


async def test_a_clean_document_plans_nothing_and_needs_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stored = await store_document(tmp_path, ["ok", "ok"])

    plan = plan_repair(spec_for(stored.output), Settings())

    assert not plan.to_repair
    assert plan.message == "No failed pages detected matching your criteria."
    assert plan.outcome().selected == 0


def _hold_lease(db: Path, doc_id: int, holder: str = "other-run") -> None:
    connection = sqlite3.connect(db)
    try:
        with connection:
            connection.execute(
                "UPDATE documents SET lease_run_id = ?, lease_until = ? "
                "WHERE doc_id = ?",
                (holder, time.time() + 600, doc_id),
            )
    finally:
        connection.close()


async def test_a_document_leased_by_another_run_is_refused_with_a_note(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed"])
    _hold_lease(stored.db, stored.doc_id)
    before = unit_rows(stored.db)
    text = stored.output.read_bytes()

    plan = plan_repair(spec_for(stored.output), Settings())
    assert any("leased by run other-run" in warning for warning in plan.warnings)
    result = await repair_module.run_repair(plan)

    assert (result.repaired, result.failed, result.selected) == (0, 1, 1)
    assert result.message is not None and "another run holds" in result.message
    (note,) = rows(stored.db, "SELECT doc_id, message FROM notes")
    assert note[0] == stored.doc_id and "another run holds" in note[1]
    assert unit_rows(stored.db) == before
    assert stored.output.read_bytes() == text
    assert transcriber.calls == []


async def test_pages_are_rendered_with_the_ffmpeg_of_the_settings(
    tmp_path: Path,
    transcriber: type[FakeTranscriber],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed"])
    exe = tmp_path / "ffmpeg.exe"
    exe.write_text("x", encoding="utf-8")
    real = repair_module.page_image
    seen: list[str] = []

    def page_image(*args: Any) -> Any:
        seen.append(get_ffmpeg_command())
        return real(*args)

    monkeypatch.setattr(repair_module, "page_image", page_image)
    settings = Settings(tools=Tools(ffmpeg_cmd=str(exe)))
    try:
        await repair_module.repair_transcription(spec_for(stored.output), settings)
    finally:
        configure_ffmpeg_executables()

    assert seen == [str(exe)]


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize("state", ["submitting", "submitted", "unresolved"])
async def test_units_of_an_open_or_unresolved_submission_are_refused(
    tmp_path: Path, state: str
) -> None:
    stored = await store_document(tmp_path, ["failed", "ok", "failed"])
    async with open_store(stored.root) as store:
        await store.begin_run("transcribe")
        lease = await store.acquire(stored.doc_id)
        assert lease is not None
        submission = await store.open_submission(
            provider="openai",
            model="gpt-5.6-luna",
            units=[UnitKey(stored.doc_id, "page", 0)],
        )
        if state == "submitted":
            await store.record_batch(submission.submission_id, "batch-1")
        elif state == "unresolved":
            await store.mark_unresolved(submission.submission_id, "lost")

    plan = plan_repair(spec_for(stored.output), Settings())
    assert plan.failures == [2]
    assert any("batch check" in warning for warning in plan.warnings)

    plan = plan_repair(spec_for(stored.output, indices="0"), Settings())
    assert not plan.to_repair
    assert plan.message is not None and "batch check" in plan.message


def _snapshot(root: Path) -> tuple[list[str], dict[str, bytes]]:
    files = {
        name: (root / name).read_bytes()
        for name in tree(root)
        if (root / name).is_file()
    }
    return tree(root), files


@pytest.mark.usefixtures("key")
@pytest.mark.parametrize("outcomes", [["ok", "failed"], ["ok", "ok"]])
async def test_a_dry_run_writes_nothing(
    tmp_path: Path, transcriber: type[FakeTranscriber], outcomes: list[str]
) -> None:
    stored = await store_document(tmp_path, outcomes)
    before = _snapshot(tmp_path)

    result = await repair_module.repair_transcription(
        spec_for(stored.output, "dry_run"), Settings()
    )

    assert result.selected == outcomes.count("failed")
    assert result.dry_run is bool(result.selected)
    assert _snapshot(tmp_path) == before
    assert transcriber.calls == []


def _versionless(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE pages (name TEXT, idx INTEGER)")
            connection.execute("INSERT INTO documents VALUES ('doc')")
    finally:
        connection.close()


def test_a_versionless_database_is_refused_as_not_a_store(tmp_path: Path) -> None:
    output = tmp_path / "doc.txt"
    output.write_text("p0.jpg: [transcription error]\n", encoding="utf-8")
    db = tmp_path / "chronotranscriber.sqlite"
    _versionless(db)
    before = db.read_bytes()

    with pytest.raises(RepairPlanError, match="not a chronotranscriber") as caught:
        plan_repair(spec_for(output), Settings())

    assert "migrate" not in str(caught.value)
    assert caught.value.exit_code == 2
    assert db.read_bytes() == before


def test_a_file_no_store_records_is_a_plan_error(tmp_path: Path) -> None:
    output = tmp_path / "doc.txt"
    output.write_text("p0.jpg: [transcription error]\n", encoding="utf-8")

    with pytest.raises(RepairPlanError, match="not recorded"):
        plan_repair(spec_for(output), Settings())


def test_a_missing_file_is_a_plan_error(tmp_path: Path) -> None:
    with pytest.raises(RepairPlanError, match="does not exist"):
        plan_repair(spec_for(tmp_path / "missing.txt"), Settings())
