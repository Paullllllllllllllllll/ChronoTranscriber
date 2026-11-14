"""The schema of a done page: the recorded hash of the run that sent its request.

A batch page is answered under the run that finalized it, so its schema is
the one recorded by the run of its response's submission. The store rows are
written directly; the oracle is :func:`page_schemas` and :func:`schema_refusal`.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from chronotranscriber.output.store import (
    SCHEMA_ENTRY,
    database_path,
    open_store,
    page_schemas,
    schema_entry,
    schema_refusal,
)

SUBMITTED = "a" * 64
SYNCHRONOUS = "b" * 64


async def _runs(root: Path) -> tuple[int, dict[str, str]]:
    """Create a document and four runs; return its id and the runs by role."""
    async with open_store(root) as store:
        runs = {
            "submit": await store.begin_run(
                "transcribe", spec={SCHEMA_ENTRY: schema_entry(SUBMITTED)}
            ),
            "check": await store.begin_run("batch check", spec={}),
            "sync": await store.begin_run(
                "repair", spec={SCHEMA_ENTRY: schema_entry(SYNCHRONOUS)}
            ),
            "legacy": await store.begin_run("transcribe", spec={"model.model": {}}),
        }
        doc_id = await store.ensure_document("doc", kind="pdf")
    return doc_id, runs


def _seed(root: Path) -> int:
    doc_id, runs = asyncio.run(_runs(root))
    connection = sqlite3.connect(database_path(root))
    try:
        connection.execute(
            "INSERT INTO submissions (submission_id, tool, provider, model, run_id, "
            "intent_at, batch_id, status) VALUES ('sub-1', 'chronotranscriber', "
            "'openai', 'm', ?, 'now', 'batch-1', 'finalized')",
            (runs["submit"],),
        )
        response = connection.execute(
            "INSERT INTO responses (doc_id, kind, idx, run_id, tool, provider, model, "
            "submission_id, custom_id, created_at) VALUES (?, 'page', 0, ?, "
            "'chronotranscriber', 'openai', 'm', 'sub-1', 'page-0', 'now')",
            (doc_id, runs["check"]),
        ).lastrowid
        rows = [
            (0, "done", "ok", runs["check"], response),
            (1, "done", "ok", runs["sync"], None),
            (2, "done", "skipped_no_text", runs["sync"], None),
            (3, "done", "ok", runs["legacy"], None),
            (4, "pending", None, runs["sync"], None),
            (5, "failed", "failed", runs["submit"], None),
        ]
        connection.executemany(
            "INSERT INTO units (doc_id, kind, idx, status, outcome, tool, model, "
            "provider, run_id, response_id) VALUES (?, 'page', ?, ?, ?, "
            "'chronotranscriber', 'm', 'openai', ?, ?)",
            [(doc_id, *row) for row in rows],
        )
        connection.commit()
    finally:
        connection.close()
    return doc_id


def _connect(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(database_path(root))


def test_each_done_page_takes_the_hash_of_the_run_that_sent_its_request(
    tmp_path: Path,
) -> None:
    doc_id = _seed(tmp_path)
    connection = _connect(tmp_path)
    try:
        assert page_schemas(connection, doc_id) == {0: SUBMITTED, 1: SYNCHRONOUS}
    finally:
        connection.close()


def test_a_page_awaiting_a_batch_takes_the_hash_of_the_submitting_run(
    tmp_path: Path,
) -> None:
    doc_id = _seed(tmp_path)
    connection = _connect(tmp_path)
    try:
        (run_id,) = connection.execute(
            "SELECT run_id FROM runs WHERE command = 'repair'"
        ).fetchone()
        connection.execute(
            "INSERT INTO submissions (submission_id, tool, provider, model, run_id, "
            "intent_at, batch_id, status) VALUES ('sub-2', 'chronotranscriber', "
            "'openai', 'm', ?, 'now', 'batch-2', 'submitted')",
            (run_id,),
        )
        connection.execute(
            "INSERT INTO submission_requests (submission_id, custom_id, doc_id, "
            "kind, idx) VALUES ('sub-2', 'page-4', ?, 'page', 4)",
            (doc_id,),
        )
        connection.commit()
        schemas = page_schemas(connection, doc_id)
        refusal = schema_refusal(connection, doc_id, "doc", SUBMITTED, replaced=[1])
    finally:
        connection.close()

    assert schemas == {0: SUBMITTED, 1: SYNCHRONOUS, 4: SYNCHRONOUS}
    assert refusal is not None and SYNCHRONOUS[:12] in refusal


def test_a_page_left_in_place_with_another_hash_refuses_the_run(
    tmp_path: Path,
) -> None:
    doc_id = _seed(tmp_path)
    connection = _connect(tmp_path)
    try:
        refusal = schema_refusal(connection, doc_id, "doc", SUBMITTED)
        replaced = schema_refusal(connection, doc_id, "doc", SUBMITTED, replaced=[1])
        unknown = schema_refusal(connection, doc_id, "doc", None)
    finally:
        connection.close()

    assert refusal is not None
    assert "The schema changed for 'doc'" in refusal
    assert SYNCHRONOUS[:12] in refusal and SUBMITTED[:12] in refusal
    assert "--force" in refusal
    assert replaced is None
    assert unknown is None
