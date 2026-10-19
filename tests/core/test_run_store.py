"""Resume matrix of the run core on the store, through ``plan()`` and ``run()``.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/core/test_run_store.py -q

A scripted chat model answers every page (``tests.llm.helpers``), so no
request leaves the machine; ``script.requests`` counts the model calls. The
oracles read the store through ``mode=ro`` connections.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.sqlite import read, read_rows
from chronotranscriber.common.store import UnitKey
from chronotranscriber.output.store import DATABASE_NAME, open_store
from chronotranscriber.spec import to_argv
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import Script, StatusError
from tests.recording_events import RecordingEvents

PAGE: dict[str, Any] = {
    "image_analysis": "One line of print.",
    "transcription": "Page text",
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}
OTHER_MODEL = "gpt-5.6-terra"


def _pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Page {index + 1}")
    document.save(path)
    document.close()
    return path


class Scenario:
    """One PDF, one output folder and the scripted model of a test."""

    def __init__(self, tmp_path: Path, script: Script, pages: int = 3) -> None:
        self.pdf = _pdf(tmp_path / "in" / "doc.pdf", pages)
        self.out = tmp_path / "out"
        self.script = script
        self.events = RecordingEvents()

    @property
    def db(self) -> Path:
        return self.out / DATABASE_NAME

    def plan(self, **flags: Any) -> core.RunPlan:
        flags.setdefault("concurrency", 1)
        flags.setdefault("model", "gpt-5.6-luna")
        spec = make_spec(self.pdf, type="pdf", output=self.out, **flags)
        return core.plan(spec, make_settings(rate_limits=((100_000, 1),)))

    def run(self, **flags: Any) -> core.RunResult:
        self.events = RecordingEvents()
        return asyncio.run(core.run(self.plan(**flags), self.events))

    @property
    def calls(self) -> int:
        return len(self.script.requests)

    def units(self) -> list[dict[str, Any]]:
        return read_rows(self.db, "units", order_by=["idx"])

    def statuses(self) -> list[str]:
        return [row["status"] for row in self.units()]

    def document_status(self) -> str:
        rows = read(
            self.db,
            lambda connection: connection.execute(
                "SELECT status FROM document_status"
            ).fetchall(),
        )
        ((status,),) = rows
        return str(status)

    def notes(self) -> list[str]:
        return [row["message"] for row in read_rows(self.db, "notes")]

    def output(self) -> Path:
        (path,) = self.out.rglob("doc.txt")
        return path

    def text(self) -> str:
        return self.output().read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> Script:
    """Answer every page with the same transcription."""
    return helpers.install(monkeypatch, [PAGE])


@pytest.fixture
def scenario(tmp_path: Path, script: Script) -> Scenario:
    return Scenario(tmp_path, script)


@pytest.fixture
def six(tmp_path: Path, script: Script) -> Scenario:
    return Scenario(tmp_path, script, pages=6)


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_a_fresh_run_writes_one_unit_per_page_and_renders(
    scenario: Scenario,
) -> None:
    result = scenario.run()

    assert (result.complete, result.failed) == (1, 0)
    assert scenario.calls == 3
    rows = scenario.units()
    assert [(row["kind"], row["idx"], row["status"]) for row in rows] == [
        ("page", 0, "done"),
        ("page", 1, "done"),
        ("page", 2, "done"),
    ]
    assert [row["label"] for row in rows] == [
        f"page_000{n}_pre_processed.jpg" for n in (1, 2, 3)
    ]
    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert len(read_rows(scenario.db, "responses")) == 3
    assert scenario.text() == "Page text\nPage text\nPage text\n"
    assert result.items[0].outputs == (str(scenario.output()),)
    assert result.databases == (str(scenario.db.resolve()),)
    (run_row,) = read_rows(scenario.db, "runs")
    assert (run_row["command"], run_row["exit_code"]) == ("transcribe", 0)
    kinds = {row["kind"] for row in read_rows(scenario.db, "artifacts")}
    assert {"prompt", "schema"} <= kinds
    assert scenario.document_status() == "complete"


def test_a_second_run_skips_the_item_without_model_calls(
    scenario: Scenario,
) -> None:
    scenario.run()
    before = _bytes(scenario.output())

    run_plan = scenario.plan()
    result = scenario.run()

    assert [item.name for item in run_plan.skipped] == ["doc.pdf"]
    assert run_plan.to_process == ()
    assert not run_plan.render_only
    assert scenario.calls == 3
    assert result.items == ()
    assert _bytes(scenario.output()) == before


def test_a_pages_run_leaves_the_document_partial(six: Scenario) -> None:
    six.run(pages="3-4")

    assert six.calls == 2
    assert six.statuses() == ["pending", "pending", "done", "done"] + ["pending"] * 2
    assert six.document_status() == "partial"
    assert six.text() == "Page text\nPage text\n"

    run_plan = six.plan()
    assert [(item.name, item.state) for item in run_plan.to_process] == [
        ("doc.pdf", "partial")
    ]
    six.run()

    assert six.calls == 6
    assert six.statuses() == ["done"] * 6
    assert six.document_status() == "complete"
    assert six.text() == "Page text\n" * 6


def test_failed_units_are_kept_only_with_no_retry_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = helpers.install(monkeypatch, [PAGE, StatusError(400), PAGE])
    scenario = Scenario(tmp_path, script)

    first = scenario.run()

    assert (first.complete, first.failed) == (0, 1)
    assert scenario.statuses() == ["done", "failed", "done"]
    (failure,) = first.items[0].failures
    assert (failure.kind, failure.idx, failure.error_type) == ("page", 1, "StatusError")
    lines = scenario.text().splitlines()
    assert lines[1].startswith("page_0002_pre_processed.jpg: [transcription error")
    calls = scenario.calls

    plain = scenario.plan(retry_errors=False)
    assert [item.name for item in plain.skipped] == ["doc.pdf"]
    scenario.run(retry_errors=False)
    assert scenario.calls == calls

    retry = scenario.plan()
    assert [(item.name, item.state) for item in retry.to_process] == [
        ("doc.pdf", "partial")
    ]
    assert retry.to_process[0].resume.indices == (1,)
    second = scenario.run()

    assert (second.complete, second.failed) == (1, 0)
    assert scenario.calls == calls + 1
    assert scenario.statuses() == ["done"] * 3
    assert scenario.text() == "Page text\n" * 3


def test_an_output_without_a_store_row_is_skipped_as_unstored(
    scenario: Scenario,
) -> None:
    output = scenario.plan().to_process[0].output
    output.parent.mkdir(parents=True)
    output.write_text("an earlier transcription\n", encoding="utf-8")

    run_plan = scenario.plan()
    result = scenario.run()

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.unstored] == ["doc.pdf"]
    assert run_plan.unstored[0].state == "unstored"
    assert "--force" in run_plan.unstored[0].reason
    assert result.items == ()
    assert scenario.calls == 0
    assert output.read_text(encoding="utf-8") == "an earlier transcription\n"


def test_a_dry_run_lists_the_items_left_out_with_their_reasons(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    output = scenario.plan().to_process[0].output
    output.parent.mkdir(parents=True)
    output.write_text("an earlier transcription\n", encoding="utf-8")
    spec = make_spec(scenario.pdf, type="pdf", output=scenario.out, dry_run=True)

    code = entry.main(["transcribe", *to_argv(spec), "--json"])

    (line,) = capsys.readouterr().out.splitlines()
    summary = json.loads(line)
    assert code == 0
    assert summary["to_process"] == []
    assert "legacy" not in summary
    assert [entry["name"] for entry in summary["unstored"]] == ["doc.pdf"]
    assert "--force" in summary["unstored"][0]["reason"]
    assert (summary["blocked"], summary["refused"]) == ([], [])
    assert scenario.calls == 0


def test_force_transcribes_an_unstored_item_again(scenario: Scenario) -> None:
    output = scenario.plan().to_process[0].output
    output.parent.mkdir(parents=True)
    output.write_text("an earlier transcription\n", encoding="utf-8")

    result = scenario.run(force=True)

    assert result.complete == 1
    assert scenario.calls == 3
    assert output.read_text(encoding="utf-8") == "Page text\n" * 3


def test_a_deleted_file_of_a_complete_document_is_rendered_again(
    scenario: Scenario,
) -> None:
    scenario.run()
    output = scenario.output()
    before = _bytes(output)
    output.unlink()

    run_plan = scenario.plan()
    result = scenario.run()

    assert [item.name for item in run_plan.render_only] == ["doc.pdf"]
    assert run_plan.to_process == ()
    assert result.items == ()
    assert scenario.calls == 3
    assert _bytes(output) == before


def test_files_rendered_again_are_listed_in_the_outputs(
    scenario: Scenario,
) -> None:
    scenario.run()
    output = scenario.output()
    output.unlink()

    result = scenario.run()

    assert result.items == ()
    assert result.outputs == [str(output), str(scenario.db.resolve())]


_VERSIONLESS_SCHEMA = """
CREATE TABLE documents (
    name TEXT PRIMARY KEY, path TEXT, kind TEXT, hash TEXT, tool_version TEXT,
    model TEXT, image_settings_fingerprint TEXT, spec TEXT, pages_total INTEGER,
    pages_ok INTEGER, pages_failed INTEGER, pages_no_text INTEGER,
    pages_not_possible INTEGER, complete INTEGER NOT NULL, outputs TEXT,
    updated_at TEXT
);
CREATE TABLE pages (
    document TEXT NOT NULL, page_index INTEGER NOT NULL, page_number INTEGER,
    image TEXT, order_index INTEGER, status TEXT NOT NULL, error TEXT,
    model TEXT, usage TEXT, text TEXT, PRIMARY KEY (document, page_index)
);
INSERT INTO documents (name, kind, pages_total, pages_ok, complete)
VALUES ('doc-139d544b/doc', 'pdf', 3, 3, 1);
INSERT INTO pages (document, page_index, status, text)
VALUES ('doc-139d544b/doc', 0, 'ok', 'Page text');
"""


def _versionless_database(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_VERSIONLESS_SCHEMA)
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def test_a_database_without_schema_version_is_refused_with_exit_2(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _versionless_database(scenario.db)

    with pytest.raises(core.StoreRefusedError) as raised:
        scenario.plan()

    assert isinstance(raised.value, core.ConfigurationError)
    assert raised.value.exit_code == ExitCode.USAGE
    assert "not a chronotranscriber store" in str(raised.value)
    assert "migrate" not in str(raised.value)
    spec = make_spec(scenario.pdf, type="pdf", output=scenario.out)
    assert entry.main(["transcribe", *to_argv(spec), "--json"]) == 2
    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["items_total"] == 0
    assert "not a chronotranscriber store" in " ".join(captured.err.split())
    assert scenario.db.read_bytes() == before
    assert scenario.calls == 0


def test_a_newer_schema_is_refused_with_exit_2(scenario: Scenario) -> None:
    scenario.run()
    connection = sqlite3.connect(scenario.db)
    try:
        connection.execute("UPDATE meta SET version = 99 WHERE component = 'core'")
        connection.commit()
    finally:
        connection.close()
    before = scenario.db.read_bytes()

    with pytest.raises(core.StoreRefusedError, match="newer") as raised:
        scenario.plan()

    assert raised.value.exit_code == ExitCode.USAGE
    assert scenario.db.read_bytes() == before


def test_a_model_switch_resumes_with_a_warning_and_a_note(six: Scenario) -> None:
    six.run(pages="1-2")

    run_plan = six.plan(model=OTHER_MODEL)
    six.run(model=OTHER_MODEL)

    assert [item.state for item in run_plan.to_process] == ["partial"]
    warnings = [w for w in six.events.of("warning") if OTHER_MODEL in w]
    assert warnings and "gpt-5.6-luna" in warnings[0]
    assert any(OTHER_MODEL in note for note in six.notes())
    assert [row["model"] for row in six.units()] == ["gpt-5.6-luna"] * 2 + [
        OTHER_MODEL
    ] * 4
    assert six.statuses() == ["done"] * 6


def test_a_changed_user_image_setting_refuses_the_item(six: Scenario) -> None:
    six.run(pages="1-2")

    run_plan = six.plan(jpeg_quality=70)
    result = six.run(jpeg_quality=70)

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.refused] == ["doc.pdf"]
    assert "jpeg_quality" in run_plan.refused[0].reason
    assert [(item.name, item.success) for item in result.items] == [("doc.pdf", False)]
    assert six.calls == 2
    assert six.statuses() == ["done"] * 2 + ["pending"] * 4


def _lease_elsewhere(db: Path) -> None:
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'other-run', lease_until = ?, "
            "lease_generation = lease_generation + 1",
            (time.time() + 600,),
        )
        connection.commit()
    finally:
        connection.close()


def test_a_document_leased_by_a_live_holder_is_skipped_with_a_note(
    six: Scenario,
) -> None:
    six.run(pages="1-2")
    _lease_elsewhere(six.db)

    run_plan = six.plan()
    result = six.run()

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.blocked] == ["doc.pdf"]
    assert "leased by run other-run until " in run_plan.blocked[0].reason
    assert result.items == () and result.blocked == ("doc.pdf",)
    assert result.blocked_reasons == (run_plan.blocked[0].reason,)
    assert result.leased == ("doc.pdf",) and result.exit_code == 1
    assert six.calls == 2
    assert any("Skipping 'doc.pdf'" in note for note in six.notes())


def test_a_lease_taken_after_planning_skips_the_item_at_run_time(
    six: Scenario,
) -> None:
    six.run(pages="1-2")
    run_plan = six.plan()
    _lease_elsewhere(six.db)

    events = RecordingEvents()
    result = asyncio.run(core.run(run_plan, events))

    assert [item.name for item in run_plan.to_process] == ["doc.pdf"]
    assert result.items == () and result.blocked == ("doc.pdf",)
    (reason,) = result.blocked_reasons
    assert "leased by run other-run until " in reason
    assert result.leased == ("doc.pdf",) and result.exit_code == 1
    assert six.calls == 2
    assert any("another run holds" in note for note in six.notes())
    assert events.of("item_started") == []


def test_a_complete_item_leased_before_its_render_counts_as_blocked(
    scenario: Scenario,
) -> None:
    scenario.run()
    output = scenario.output()
    output.unlink()
    run_plan = scenario.plan()
    _lease_elsewhere(scenario.db)

    result = asyncio.run(core.run(run_plan))

    assert [item.name for item in run_plan.render_only] == ["doc.pdf"]
    assert result.skipped == () and result.blocked == ("doc.pdf",)
    assert result.leased == ("doc.pdf",) and result.exit_code == 1
    assert not output.exists()


def _submitting_elsewhere(root: Path, name: str, idxs: tuple[int, ...]) -> None:
    """Leave a submission ``submitting`` without a batch id, as a killed run does."""

    async def leave() -> None:
        store = open_store(root)
        await store.open()
        try:
            await store.begin_run("transcribe")
            doc_id = await store.ensure_document(name)
            lease = await store.acquire(doc_id)
            assert lease is not None
            await store.open_submission(
                provider="openai",
                model="gpt-5.6-luna",
                units=[UnitKey(doc_id, "page", idx) for idx in idxs],
            )
            await store.release(lease)
            await store.end_run(1)
        finally:
            await store.close()

    asyncio.run(leave())


def test_a_submission_left_submitting_is_marked_unresolved_at_start(
    six: Scenario,
) -> None:
    six.run(pages="1-2")
    _submitting_elsewhere(six.out, six.plan().to_process[0].document, (2, 3))

    run_plan = six.plan()
    result = six.run()

    assert [item.name for item in run_plan.blocked] == ["doc.pdf"]
    assert result.items == () and result.blocked == ("doc.pdf",)
    # Pages waiting for a batch are submitted work, not a failure.
    assert result.leased == () and result.exit_code == 0
    (submission,) = read_rows(six.db, "submissions")
    assert (submission["status"], submission["batch_id"]) == ("unresolved", None)
    assert any("unresolved" in warning for warning in six.events.of("warning"))
    assert six.calls == 2
    assert six.statuses() == ["done"] * 2 + ["pending"] * 4

    six.run()
    assert six.calls == 2
    assert six.statuses() == ["done"] * 2 + ["pending"] * 4


def _files(folder: Path) -> Iterator[Path]:
    return (path for path in folder.rglob("*") if path.is_file())


@pytest.mark.parametrize("layout", ["beside", "hash", "mirror"])
def test_an_llm_run_leaves_only_rendered_files_and_the_database(
    tmp_path: Path, script: Script, layout: str
) -> None:
    source = _pdf(tmp_path / "in" / "doc.pdf", 2)
    flags: dict[str, Any] = {"type": "pdf", "concurrency": 1, "formats": ("txt", "md")}
    if layout != "beside":
        flags.update(output=tmp_path / "out", layout=layout)
    spec = make_spec(tmp_path / "in", **flags)

    result = asyncio.run(
        core.run(core.plan(spec, make_settings(rate_limits=((100_000, 1),))))
    )

    assert result.complete == 1
    root = source.parent if layout == "beside" else tmp_path / "out"
    names = sorted(path.relative_to(root).as_posix() for path in _files(root))
    rendered = [name for name in names if name.endswith((".txt", ".md"))]
    assert {Path(name).name for name in rendered} == {"doc.txt", "doc.md"}
    others = sorted(set(names) - set(rendered))
    expected = ["chronotranscriber.sqlite"]
    if layout == "beside":
        expected.append("doc.pdf")
    assert others == expected
    for folder in (tmp_path / "in", root):
        assert not list(folder.rglob("*.jsonl"))
        assert not list(folder.rglob("*_batch_submission_debug.json"))
    assert sorted(path.name for path in source.parent.iterdir()) == (
        ["chronotranscriber.sqlite", "doc.md", "doc.pdf", "doc.txt"]
        if layout == "beside"
        else ["doc.pdf"]
    )
