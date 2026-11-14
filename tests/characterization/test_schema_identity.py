"""The schema hash a run records, and the refusal of a run under another schema.

Each case reuses the scripted first run of ``test_repair``: a four-page
scanned PDF transcribed with the bundled markdown schema, page 2 failed and
pages 3 and 4 done with the no-text and not-possible flags. The other schema
is an invented copy with another description, so its text and hash differ.
The oracle is the store: ``runs.spec``, ``artifacts`` and the units.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import run as core
from chronotranscriber.common.store import Lease, Store
from chronotranscriber.llm.prompts import SCHEMAS_DIR
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.inputs import make_scanned_pdf
from tests.characterization.runner import RunResult, run_cli
from tests.characterization.test_repair import (  # noqa: F401
    PAGES,
    _empty_project_context,
    first_run,
)

pytestmark = pytest.mark.integration

BUNDLED = SCHEMAS_DIR / "markdown_transcription_schema.json"
CHANGED = "The schema changed for 'doc-"
OTHER_HASH = "f" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _other_schema(root: Path) -> Path:
    body = json.loads(BUNDLED.read_text(encoding="utf-8"))
    body["name"] = "ledger_pages"
    body["schema"]["description"] = "Pages of a ledger."
    path = root / "schemas" / "ledger_pages.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _connect(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(root / "out" / "chronotranscriber.sqlite")


def _query(root: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = _connect(root)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _execute(root: Path, *sql: str) -> None:
    connection = _connect(root)
    try:
        for statement in sql:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def _pending(root: Path, idx: int) -> None:
    _execute(
        root,
        "UPDATE documents SET status = 'partial'",
        f"UPDATE units SET status = 'pending', outcome = NULL, content = NULL, "
        f"response_id = NULL WHERE idx = {idx}",
    )


def _transcribe(llm: FakeLLM, root: Path, **options: Any) -> RunResult:
    return run_cli(
        "transcribe",
        llm=llm,
        root=root,
        input=root / "docs" / "doc.pdf",
        output=root / "out",
        type="pdf",
        method="llm",
        **options,
    )


def _repair(llm: FakeLLM, root: Path, **options: Any) -> RunResult:
    return run_cli("repair", llm=llm, root=root, input=root / "out", **options)


def _page_hashes(root: Path) -> dict[int, str | None]:
    rows = _query(
        root,
        "SELECT u.idx, json_extract(r.spec, '$.schema_sha256.value') FROM units u "
        "JOIN runs r ON r.run_id = u.run_id ORDER BY u.idx",
    )
    return {int(idx): value for idx, value in rows}


def _record_other_hash(root: Path) -> None:
    """Make the first run's record name another schema, as a parallel run would."""
    _execute(
        root,
        f"UPDATE runs SET spec = json_set(spec, '$.schema_sha256.value', "
        f"'{OTHER_HASH}') WHERE run_id = (SELECT run_id FROM units WHERE idx = 0)",
    )


def test_a_run_records_the_hash_of_the_schema_text_it_stores(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)

    ((entry,),) = _query(
        tmp_path, "SELECT json_extract(spec, '$.schema_sha256') FROM runs"
    )
    stored = _query(tmp_path, "SELECT sha256 FROM artifacts WHERE kind = 'schema'")

    assert json.loads(entry) == {"value": _sha256(BUNDLED), "source": "computed"}
    assert stored == [(_sha256(BUNDLED),)]
    dry = _transcribe(fake_llm, tmp_path, dry_run=True)
    assert dry.exit_code == 0, dry.stderr
    assert dry.summary is not None
    assert "schema_sha256" not in dry.summary["spec"]


def test_a_text_format_run_records_no_hash(tmp_path: Path, fake_llm: FakeLLM) -> None:
    first_run(fake_llm, tmp_path)
    _pending(tmp_path, 3)

    plain = _transcribe(
        fake_llm, tmp_path, response_format="text", schema=str(_other_schema(tmp_path))
    )

    assert plain.exit_code == 0, plain.stderr
    assert _page_hashes(tmp_path)[3] is None


def test_a_resume_under_another_schema_is_refused(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)
    _pending(tmp_path, 3)

    refused = _transcribe(fake_llm, tmp_path, schema=str(_other_schema(tmp_path)))

    assert refused.exit_code == 1
    assert CHANGED in refused.stderr and "--force" in refused.stderr
    assert refused.calls == []
    assert _query(tmp_path, "SELECT status FROM units WHERE idx = 3") == [("pending",)]
    resumed = _transcribe(fake_llm, tmp_path)
    assert resumed.exit_code == 0, resumed.stderr
    assert [call["page"] for call in resumed.calls] == [3]


def test_force_replaces_a_document_of_another_schema(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)
    _pending(tmp_path, 3)
    other = _other_schema(tmp_path)

    forced = _transcribe(fake_llm, tmp_path, schema=str(other), force=True)

    assert forced.exit_code == 0, forced.stderr
    assert set(_page_hashes(tmp_path).values()) == {_sha256(other)}


def test_retry_errors_in_a_complete_document_is_refused_under_another_schema(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)

    refused = _transcribe(
        fake_llm, tmp_path, retry_errors=True, schema=str(_other_schema(tmp_path))
    )
    kept = _transcribe(fake_llm, tmp_path, schema=str(_other_schema(tmp_path)))

    assert refused.exit_code == 1
    assert CHANGED in refused.stderr
    assert refused.calls == []
    assert kept.exit_code == 0, kept.stderr
    assert CHANGED not in kept.stderr


def test_a_run_without_a_recorded_hash_never_refuses(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)
    _execute(tmp_path, "UPDATE runs SET spec = json_remove(spec, '$.schema_sha256')")
    _pending(tmp_path, 3)
    other = _other_schema(tmp_path)

    resumed = _transcribe(fake_llm, tmp_path, schema=str(other))

    assert resumed.exit_code == 0, resumed.stderr
    assert [call["page"] for call in resumed.calls] == [3]
    assert _page_hashes(tmp_path)[3] == _sha256(other)


def test_pages_the_pre_pass_skipped_have_no_schema(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)
    _execute(
        tmp_path, "UPDATE units SET outcome = 'skipped_no_text' WHERE idx IN (0, 2)"
    )
    _pending(tmp_path, 3)

    resumed = _transcribe(fake_llm, tmp_path, schema=str(_other_schema(tmp_path)))

    assert resumed.exit_code == 0, resumed.stderr
    assert [call["page"] for call in resumed.calls] == [3]


def test_a_repair_in_a_complete_document_is_refused_under_another_schema(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)

    refused = _repair(fake_llm, tmp_path, schema=str(_other_schema(tmp_path)))

    assert refused.exit_code == 1
    assert CHANGED in refused.stderr
    assert "repair it with the earlier schema" in refused.stderr
    assert refused.calls == []
    assert _query(tmp_path, "SELECT status FROM units WHERE idx = 1") == [("failed",)]


def test_a_repair_replacing_every_page_of_the_other_schema_runs(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first_run(fake_llm, tmp_path)
    _execute(tmp_path, "UPDATE units SET outcome = 'skipped_no_text' WHERE idx = 0")
    other = _other_schema(tmp_path)

    repaired = _repair(fake_llm, tmp_path, schema=str(other), all_failures=True)

    assert repaired.exit_code == 0, repaired.stderr
    assert [call["page"] for call in repaired.calls] == [1, 2, 3]
    record = _query(
        tmp_path,
        "SELECT json_extract(spec, '$.schema_sha256.value') FROM runs "
        "WHERE command = 'repair'",
    )
    assert record == [(_sha256(other),)]
    assert (_sha256(other),) in _query(
        tmp_path, "SELECT sha256 FROM artifacts WHERE kind = 'schema'"
    )
    assert {_page_hashes(tmp_path)[idx] for idx in (1, 2, 3)} == {_sha256(other)}


def test_a_batch_page_takes_the_schema_of_the_submitting_run(
    tmp_path: Path, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeBatchService(fake_llm).install(monkeypatch)
    make_scanned_pdf(tmp_path / "docs" / "doc.pdf", PAGES)
    submitted = _transcribe(fake_llm, tmp_path, batch=True)
    assert submitted.exit_code == 0, submitted.stderr
    checked = run_cli(
        "batch check", llm=fake_llm, root=tmp_path, input=tmp_path / "out"
    )
    assert checked.exit_code == 0, checked.stderr
    assert set(_page_hashes(tmp_path).values()) == {None}
    _pending(tmp_path, 3)

    refused = _transcribe(fake_llm, tmp_path, schema=str(_other_schema(tmp_path)))

    assert refused.exit_code == 1
    assert CHANGED in refused.stderr
    assert _sha256(BUNDLED)[:12] in refused.stderr


def test_a_schema_recorded_after_planning_refuses_the_transcription_lease(
    tmp_path: Path, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_run(fake_llm, tmp_path)
    _pending(tmp_path, 3)
    settle = core._ItemRun._settle_identity

    async def other_run_first(self: core._ItemRun, *args: Any) -> str | None:
        _record_other_hash(tmp_path)
        return await settle(self, *args)

    monkeypatch.setattr(core._ItemRun, "_settle_identity", other_run_first)
    refused = _transcribe(fake_llm, tmp_path)

    assert refused.exit_code == 1
    assert CHANGED in refused.stderr
    assert refused.calls == []
    assert _query(tmp_path, "SELECT status FROM units WHERE idx = 3") == [("pending",)]


def test_a_schema_recorded_after_planning_refuses_the_repair_lease(
    tmp_path: Path, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_run(fake_llm, tmp_path)
    acquire = Store.acquire

    async def other_run_first(self: Store, doc_id: int) -> Lease | None:
        lease = await acquire(self, doc_id)
        _record_other_hash(tmp_path)
        return lease

    monkeypatch.setattr(Store, "acquire", other_run_first)
    refused = _repair(fake_llm, tmp_path)

    assert refused.exit_code == 1
    assert CHANGED in refused.stderr
    assert refused.calls == []
    assert _query(tmp_path, "SELECT status FROM units WHERE idx = 1") == [("failed",)]
