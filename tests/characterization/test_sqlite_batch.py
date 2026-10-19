"""The store of batch-submitted items through submit, check and repair.

Each case submits a two-page scanned PDF with ``transcribe --batch`` against
the fake batch service, finalizes it with ``batch check`` and reads
``chronotranscriber.sqlite`` back after every command through short
``mode=ro`` connections. The submission plans the document's units as
pending and records the submission with its scoped custom ids before the
upload; check stores every result line and finalizes the units; repair
retries the failed unit; a later transcribe run leaves the store as it is.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.output.store import DATABASE_NAME
from tests.characterization.fakes import FakeLLM, Reply, permanent_error
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.inputs import make_scanned_pdf
from tests.characterization.runner import RunResult, run_cli

pytestmark = pytest.mark.integration

LAYOUTS = ("hash", "mirror", "beside")


@pytest.fixture
def batch_service(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> FakeBatchService:
    """Install the fake batch service over ``fake_llm``'s page scripts."""
    return FakeBatchService(fake_llm).install(monkeypatch)


class Case:
    """One PDF input, transcribed in one layout."""

    def __init__(self, llm: FakeLLM, root: Path, layout: str) -> None:
        self.llm = llm
        self.root = root
        self.layout = layout
        self.docs = root / "docs"
        self.pdf = make_scanned_pdf(self.docs / "doc.pdf", 2)
        beside = layout == "beside"
        self.folder = (self.docs if beside else root / "out").resolve()
        self.database = self.folder / DATABASE_NAME

    def transcribe(self, **options: Any) -> RunResult:
        beside = self.layout == "beside"
        result = run_cli(
            "transcribe",
            llm=self.llm,
            root=self.root,
            input=self.pdf,
            output=None if beside else self.root / "out",
            beside_input=beside,
            layout="mirror" if self.layout == "mirror" else None,
            type="pdf",
            method="llm",
            batch=True,
            **options,
        )
        assert result.exit_code == 0, result.stdout + result.stderr
        return result

    def check(self, *, failed: bool = False, **options: Any) -> RunResult:
        """Run a check; *failed*: it finalizes a failed page, so it exits 1."""
        result = run_cli(
            "batch check",
            llm=self.llm,
            root=self.root,
            input=self.folder,
            **options,
        )
        assert result.exit_code == int(failed), result.stdout + result.stderr
        return result

    def repair(self, transcription: Path) -> RunResult:
        result = run_cli(
            "repair",
            llm=self.llm,
            root=self.root,
            input=transcription,
            errors_only=True,
        )
        assert result.exit_code == 0, result.stdout + result.stderr
        return result

    def name(self, submitted: RunResult) -> str:
        """Return the item's document name: its output, relative, unsuffixed."""
        output = submitted.output(self.pdf).resolve().with_suffix("")
        return output.relative_to(self.folder).as_posix()

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Run *sql* on a read-only connection to the store and close it."""
        uri = f"{self.database.as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            cursor = connection.execute(sql)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        finally:
            connection.close()

    def document(self) -> dict[str, Any]:
        (row,) = self.query("SELECT * FROM documents")
        return row

    def units(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM units ORDER BY doc_id, kind, idx")

    def pages(self) -> list[tuple[Any, ...]]:
        rows = self.query("SELECT * FROM page_text")
        return [(row["order_index"], row["status"], row["outcome"]) for row in rows]

    def submissions(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM submissions ORDER BY rowid")

    def responses(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM responses ORDER BY response_id")

    def state(self) -> tuple[Any, ...]:
        """Return the store's document, units, submissions and responses."""
        document = {
            key: value
            for key, value in self.document().items()
            if key not in {"lease_run_id", "lease_until", "lease_generation"}
        }
        return document, self.units(), self.submissions(), self.responses()


def _spec(database_rows: list[dict[str, Any]]) -> dict[str, Any]:
    (run,) = database_rows
    return dict(json.loads(run["spec"]))


def _finalized_output(checked: RunResult, suffix: str = ".txt") -> Path:
    (output,) = [path for path in checked.created if path.suffix == suffix]
    return output


def _assert_submitted(case: Case, submitted: RunResult) -> None:
    """The store after a batch submit: one pending document, one submission."""
    assert submitted.calls == []
    assert [path.name for path in submitted.created] == [DATABASE_NAME]
    document = case.document()
    assert (document["name"], document["units_total"]) == (case.name(submitted), 2)
    assert document["units_pending"] == 2
    assert case.pages() == [(0, "pending", None), (1, "pending", None)]
    (submission,) = case.submissions()
    assert submission["status"] == "submitted"
    assert submission["batch_id"] and submission["upload_ref"]
    assert submission["request_count"] == 2
    requests = case.query("SELECT * FROM submission_requests ORDER BY idx")
    prefix = str(submission["submission_id"])[:8]
    assert [(row["idx"], row["custom_id"][:8]) for row in requests] == [
        (0, prefix),
        (1, prefix),
    ]
    spec = _spec(case.query("SELECT spec FROM runs"))
    assert spec["run.batch"] == {"value": True, "source": "flag"}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_submit_check_repair_and_a_later_run_keep_one_document(
    layout: str,
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
) -> None:
    case = Case(fake_llm, case_root("case"), layout)
    fake_llm.script(1, always=Reply(exception=permanent_error()))

    submitted = case.transcribe()

    _assert_submitted(case, submitted)

    checked = case.check(failed=True)

    assert checked.summary is not None
    assert checked.summary["finalized"] == 1
    output = _finalized_output(checked)
    assert case.pages() == [(0, "done", "ok"), (1, "failed", "failed")]
    (submission,) = case.submissions()
    assert submission["status"] == "finalized"
    responses = case.responses()
    assert [(row["idx"], row["raw_kind"]) for row in responses] == [
        (0, "line"),
        (1, "line"),
    ]
    assert {row["submission_id"] for row in responses} == {submission["submission_id"]}
    outputs = case.query("SELECT path, format, stale FROM outputs")
    assert outputs == [
        {
            "path": output.resolve().relative_to(case.folder).as_posix(),
            "format": "txt",
            "stale": 0,
        }
    ]

    state = case.state()
    again = case.transcribe()

    assert again.calls == []
    assert again.created == []
    assert again.summary is not None
    assert again.summary["items_skipped"] == 1
    assert case.state() == state

    fake_llm.script(1, always=None)
    repaired = case.repair(output)

    assert repaired.summary is not None
    assert repaired.summary["repaired"] == 1
    assert case.pages() == [(0, "done", "ok"), (1, "done", "ok")]
    (page,) = case.query("SELECT text FROM page_text WHERE order_index = 1")
    assert "Market Records, Part 02" in page["text"]
    assert case.document()["status"] == "complete"
    assert len(case.responses()) == 3
    assert "Market Records, Part 02" in output.read_text(encoding="utf-8")

    state = case.state()
    last = case.transcribe()

    assert last.calls == []
    assert last.created == []
    assert case.state() == state


def test_every_format_registers_its_output(
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
) -> None:
    case = Case(fake_llm, case_root("case"), "hash")

    submitted = case.transcribe(formats="md")

    assert case.document()["name"] == case.name(submitted)
    assert case.query("SELECT * FROM outputs") == []

    checked = case.check(formats="md")

    assert checked.summary is not None
    assert checked.summary["finalized"] == 1
    output = _finalized_output(checked, ".md")
    assert case.document()["status"] == "complete"
    (registered,) = case.query("SELECT path, format FROM outputs")
    assert registered == {
        "path": output.resolve().relative_to(case.folder).as_posix(),
        "format": "md",
    }
    assert _spec(case.query("SELECT spec FROM runs ORDER BY rowid LIMIT 1"))[
        "run.batch"
    ]["value"]


def test_an_open_submission_blocks_its_document_until_check(
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
) -> None:
    case = Case(fake_llm, case_root("case"), "hash")
    case.transcribe()
    state = case.state()

    # Pages waiting for a batch are submitted work: named, counted, exit 0.
    again = case.transcribe()

    assert again.calls == []
    assert again.created == []
    assert again.summary is not None
    assert again.summary["items_blocked"] == 1
    assert len(case.submissions()) == 1
    assert case.state()[1:] == state[1:]

    checked = case.check()

    assert checked.summary is not None
    assert checked.summary["finalized"] == 1
    assert case.pages() == [(0, "done", "ok"), (1, "done", "ok")]
