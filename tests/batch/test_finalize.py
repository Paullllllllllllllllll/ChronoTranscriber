"""Finalization: placeholders, outcomes and idempotence on the store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.batch.check import check_submissions
from chronotranscriber.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    MISSING_TEXT,
    PENDING,
    placeholder,
)
from chronotranscriber.batch.submit import submit_batch
from chronotranscriber.common.batch import BatchResultItem, BatchStatus
from chronotranscriber.common.capture import redact_payloads
from chronotranscriber.common.responses import ResponseOutcome
from chronotranscriber.common.sqlite import read
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.settings import Settings
from tests.batch.conftest import (
    World,
    leased_document,
    page_images,
    run_spec,
)


def item(custom_id: str, **values: Any) -> BatchResultItem:
    return BatchResultItem(custom_id=custom_id, **values)


@pytest.mark.parametrize(
    ("provider", "result", "text"),
    [
        (
            "openai",
            item(
                "a",
                error="bad image",
                error_code="invalid_request",
                origin="error_file",
                response={"status_code": 400},
            ),
            "[transcription error; code invalid_request; bad image]",
        ),
        (
            "openai",
            item(
                "a",
                error="bad image",
                error_code="invalid_request",
                response={"status_code": 400},
            ),
            "[transcription error; status 400; code invalid_request; bad image]",
        ),
        (
            "openai",
            item("a", error="HTTP 500", response={"status_code": 500}),
            "[transcription error; status 500]",
        ),
        (
            "anthropic",
            item("a", error="overloaded"),
            "[transcription error: overloaded]",
        ),
        ("google", item("a", error="no result", origin="missing"), MISSING_TEXT),
    ],
)
def test_placeholders_follow_each_provider(
    provider: str, result: BatchResultItem, text: str
) -> None:
    assert placeholder(provider, result) == text


async def submit_store_pages(root: Path, pages: int = 3) -> str:
    """Submit *pages* pages of document ``doc`` in *root*; return the submission."""
    async with open_store(root) as store:
        await store.begin_run("transcribe")
        handle = await leased_document(store)
        outcome = await submit_batch(
            page_images(pages),
            document=handle,
            spec=run_spec(root),
            settings=Settings(),
            schema_path=DEFAULT_SCHEMA_PATH,
        )
    assert outcome.status == "submitted"
    (submission_id,) = outcome.submission_ids
    return submission_id


def submission_row(root: Path, submission_id: str) -> dict[str, Any]:
    def query(connection: sqlite3.Connection) -> dict[str, Any]:
        cursor = connection.execute(
            "SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)
        )
        names = [column[0] for column in cursor.description]
        return dict(zip(names, cursor.fetchone(), strict=True))

    return read(database_path(root), query)


def unit_texts(root: Path) -> list[tuple[str, str]]:
    return read(
        database_path(root),
        lambda connection: connection.execute(
            "SELECT status, content FROM units ORDER BY idx"
        ).fetchall(),
    )


async def test_a_completed_batch_is_finalized_once(
    world: World, tmp_path: Path
) -> None:
    await submit_store_pages(tmp_path, pages=2)
    (batch,) = world.batches.values()
    first_id, second_id = batch.bodies
    world.dropped = {second_id}

    first = await check_submissions(tmp_path, Settings())

    (check,) = first.submissions
    assert (check.outcome, check.errors) == (FINALIZED_OUTCOME, 1)
    assert check.outputs == (tmp_path / "doc.txt",)
    assert unit_texts(tmp_path) == [
        ("done", f"text of {first_id}"),
        ("failed", MISSING_TEXT),
    ]

    world.calls.clear()
    again = await check_submissions(tmp_path, Settings())
    assert again.submissions == []
    assert world.calls == []


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (BatchStatus.IN_PROGRESS, PENDING),
        (BatchStatus.EXPIRED, FAILED),
        (BatchStatus.CANCELLED, FAILED),
        (BatchStatus.UNKNOWN, PENDING),
    ],
)
async def test_an_unfinished_batch_writes_no_unit(
    world: World, tmp_path: Path, status: BatchStatus, outcome: str
) -> None:
    world.status = status
    await submit_store_pages(tmp_path, pages=2)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == outcome
    assert [state for state, _content in unit_texts(tmp_path)] == ["pending"] * 2
    assert not (tmp_path / "doc.txt").exists()


async def test_a_missing_key_fails_the_submission(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=1)
    monkeypatch.delenv("OPENAI_API_KEY")

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FAILED
    assert "OPENAI_API_KEY" in (check.detail or "")
    assert submission_row(tmp_path, submission_id)["status"] == "submitted"


async def test_each_response_row_holds_the_body_its_line_sent(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id = await submit_store_pages(tmp_path, pages=2)
    batch = openai._retrieve(submission_row(tmp_path, submission_id)["batch_id"])
    lines = openai.data[batch.input_file_id].decode("utf-8").splitlines()
    sent = {line["custom_id"]: line["body"] for line in map(json.loads, lines)}
    openai.complete(batch.id)

    await check_submissions(tmp_path, Settings())

    stored = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT custom_id, request, usage FROM responses WHERE submission_id = ?",
            (submission_id,),
        ).fetchall(),
    )
    assert {custom_id for custom_id, _request, _usage in stored} == set(sent)
    for custom_id, request, usage in stored:
        assert json.loads(request) == redact_payloads(sent[custom_id])
        assert list(json.loads(usage)) == list(Usage().as_dict())


async def test_each_result_record_is_stored_as_downloaded(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id = await submit_store_pages(tmp_path)
    row = submission_row(tmp_path, submission_id)
    output, errors = openai.complete(row["batch_id"], failed=frozenset({1}))
    order: list[tuple[str, str, bool]] = []

    def at_delete(file_id: str) -> None:
        status = submission_row(tmp_path, submission_id)["status"]
        order.append((file_id, status, (tmp_path / "doc.txt").is_file()))

    openai.on_delete = at_delete

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert (check.outcome, check.errors) == (FINALIZED_OUTCOME, 1)
    assert check.outputs == (tmp_path / "doc.txt",)
    raw = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT custom_id, raw, raw_kind, idx FROM responses "
            "WHERE submission_id = ? ORDER BY response_id",
            (submission_id,),
        ).fetchall(),
    )
    lines = [
        *output.decode("utf-8").splitlines(),
        *errors.decode("utf-8").splitlines(),
    ]
    assert [stored for _cid, stored, _kind, _idx in raw] == lines
    for _cid, stored, kind, _idx in raw:
        assert kind == "line"
        assert stored.encode("utf-8") in output + errors
    assert [idx for _cid, _raw, _kind, idx in raw] == [0, 2, 1]
    units = read(
        database_path(tmp_path),
        lambda connection: connection.execute(
            "SELECT idx, status, content, response_id IS NOT NULL, provenance "
            "FROM units ORDER BY idx"
        ).fetchall(),
    )
    assert units == [
        (0, "done", "Zürich «0»", 1, '{"dpi": 300}'),
        (
            1,
            "failed",
            "[transcription error; code invalid_image; Bild unlesbar]",
            1,
            '{"dpi": 300}',
        ),
        (2, "done", "Zürich «2»", 1, '{"dpi": 300}'),
    ]
    assert (tmp_path / "doc.txt").read_text(encoding="utf-8") == (
        "Zürich «0»\n"
        "page_0002_pre_processed.jpg: [transcription error; code invalid_image; "
        "Bild unlesbar]\n"
        "Zürich «2»\n"
    )
    assert submission_row(tmp_path, submission_id)["status"] == "finalized"
    assert sorted(file_id for file_id, _status, _file in order) == [
        "file-1",
        "file-2",
        "file-3",
    ]
    assert all(status == "finalized" and rendered for _id, status, rendered in order)


@pytest.mark.parametrize("response_format", ["text", "schema"])
async def test_a_truncated_answer_fails_in_every_format(
    world: World, tmp_path: Path, response_format: str
) -> None:
    async with open_store(tmp_path) as store:
        chosen = {"value": response_format, "source": "flag"}
        await store.begin_run("transcribe", spec={"model.response_format": chosen})
        await submit_batch(
            page_images(1),
            document=await leased_document(store),
            spec=run_spec(tmp_path, response_format=response_format),
            settings=Settings(),
            schema_path=DEFAULT_SCHEMA_PATH,
        )
    (batch,) = world.batches.values()
    (custom_id,) = batch.bodies
    cut = item(custom_id, text="cut sho")
    cut.settle(ResponseOutcome("truncated", "finish_reason=MAX_TOKENS"), "empty")
    world.results[custom_id] = cut

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert unit_texts(tmp_path) == [
        (
            "failed",
            "[transcription error; code truncated; Response truncated at the "
            "output-token limit: finish_reason=MAX_TOKENS]",
        )
    ]


async def test_a_running_batch_records_its_status_and_stays_pending(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=1)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == PENDING
    row = submission_row(tmp_path, submission_id)
    assert (row["status"], row["provider_status"]) == ("submitted", "in_progress")
    assert row["last_checked_at"] is not None
    assert not (tmp_path / "doc.txt").exists()
    assert sdk["openai"].deleted == []


async def test_a_batch_ended_without_results_frees_its_units(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_store_pages(tmp_path, pages=2)
    batch = sdk["openai"]._retrieve(submission_row(tmp_path, submission_id)["batch_id"])
    batch.status = "expired"

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FAILED
    assert submission_row(tmp_path, submission_id)["status"] == "failed"
    async with open_store(tmp_path) as store:
        await store.begin_run("resume")
        handle = await leased_document(store)
        assert handle.todo() == [0, 1]
