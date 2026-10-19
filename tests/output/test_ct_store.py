"""ChronoTranscriber's store component: views, naming and image settings.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_ct_store.py -q

The view oracle queries ``page_text`` through a ``mode=ro`` connection over
a store seeded through the store API and compares the documented columns
and rows, failed and pending units included.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.native import image_settings_fingerprint
from chronotranscriber.common.store import (
    Lease,
    PlannedUnit,
    ResponseRow,
    Store,
    UnitKey,
    UnitResult,
    VersionlessDatabaseError,
)
from chronotranscriber.output.store import (
    COMPONENT,
    DATABASE_NAME,
    KIND_CHUNK,
    KIND_PAGE,
    KIND_TEXT,
    TOOL,
    VIEWS,
    database_path,
    document_name,
    image_settings_record,
    open_store,
    read_only_connection,
    read_store,
)

MODEL = "gpt-5.6-luna"
PROVIDER = "openai"
NAME = "doc-139d544b/doc"
USAGE = {"input_tokens": 1500, "output_tokens": 400, "total_tokens": 1900}

PAGE_TEXT_COLUMNS = (
    "document",
    "kind",
    "page_index",
    "image_name",
    "order_index",
    "text",
    "status",
    "model",
    "usage",
    "outcome",
    "error",
    "provider",
    "run_id",
)


@dataclass(frozen=True)
class Unit:
    """One unit to seed: its state and content."""

    idx: int
    status: str = "done"
    content: str | None = None
    label: str | None = None
    source_index: int | str | None = None
    model: str = MODEL
    kind: str = KIND_PAGE
    outcome: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None


def page(idx: int, status: str = "done", **changes: Any) -> Unit:
    """Return a PDF page unit with the content its status implies."""
    content = {"done": f"Page {idx + 1} text", "failed": "[transcription error]"}
    values: dict[str, Any] = {
        "idx": idx,
        "status": status,
        "content": content.get(status),
        "label": f"page_{idx + 1:04d}_pre_processed.jpg",
        "source_index": idx,
        "outcome": {"done": "ok", "failed": "failed"}.get(status),
        "error": "Error code: 400" if status == "failed" else None,
        "usage": USAGE if status == "done" else None,
    }
    values.update(changes)
    return Unit(**values)


def _result(unit: Unit) -> UnitResult:
    return UnitResult(
        kind=unit.kind,
        idx=unit.idx,
        status=unit.status,
        model=unit.model,
        provider=PROVIDER,
        content=unit.content,
        outcome=unit.outcome,
        label=unit.label,
        source_index=unit.source_index,
        usage=unit.usage,
        error=unit.error,
        error_type="api_error" if unit.status == "failed" else None,
    )


async def seed_document(
    store: Store,
    units: Sequence[Unit],
    *,
    name: str = NAME,
    kind: str = "pdf",
    image_settings: dict[str, Any] | None = None,
    order: Sequence[int] | None = None,
) -> Lease:
    """Create document *name*, plan *units* as pending and write the others.

    *order* gives the positions in *units* to write, in that order.
    """
    doc_id = await store.ensure_document(name, input_path=f"{name}.pdf", kind=kind)
    lease = await store.acquire(doc_id)
    assert lease is not None
    for unit_kind in dict.fromkeys(unit.kind for unit in units):
        planned = [
            PlannedUnit(unit.idx, label=unit.label, source_index=unit.source_index)
            for unit in units
            if unit.kind == unit_kind
        ]
        await store.plan_units(
            lease, unit_kind, planned, model=MODEL, provider=PROVIDER
        )
    if image_settings is not None:
        await store.set_image_settings(lease, image_settings)
    positions = order if order is not None else range(len(units))
    for position in positions:
        unit = units[position]
        if unit.status != "pending":
            await store.write_unit(lease, _result(unit))
    return lease


def build_store(
    root: Path,
    units: Sequence[Unit],
    *,
    name: str = NAME,
    image_settings: dict[str, Any] | None = None,
) -> Path:
    """Seed one document in *root*'s store; return the database path."""

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            await seed_document(store, units, name=name, image_settings=image_settings)
            await store.end_run(0)

    asyncio.run(main())
    return database_path(root)


def _query(database: Path, sql: str) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql)
        columns = tuple(description[0] for description in cursor.description)
        return columns, cursor.fetchall()
    finally:
        connection.close()


def test_the_component_adds_the_views_at_version_1(tmp_path: Path) -> None:
    database = build_store(tmp_path, [page(0)])

    _, meta = _query(database, "SELECT component, version FROM meta ORDER BY 1")
    _, views = _query(
        database, "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"
    )

    assert database.name == DATABASE_NAME
    assert COMPONENT.name == TOOL
    assert meta == [("chronotranscriber", 1), ("core", 2)]
    assert sorted(name for (name,) in views) == sorted(VIEWS)


def test_page_text_returns_every_unit_in_page_order(tmp_path: Path) -> None:
    pdf = [page(0), page(1, "failed"), page(2), page(3, "pending")]
    images = [
        Unit(
            0,
            content="Ledger one",
            label="a.png_pre_processed.jpg",
            source_index="a.png",
        ),
        Unit(
            1,
            content="Ledger two",
            label="b.png_pre_processed.jpg",
            source_index="b.png",
        ),
    ]
    chunks = [
        Unit(0, content="Spoken words", label="talk_chunk_0001.mp3", kind=KIND_CHUNK)
    ]

    async def main() -> str:
        async with open_store(tmp_path) as store:
            run_id = await store.begin_run("transcribe")
            await seed_document(store, pdf, order=[2, 1, 0, 3])
            await seed_document(store, images, name="ledger-fe14010b/ledger")
            await seed_document(store, chunks, name="talk-ce7611e7/talk")
            return run_id

    run_id = asyncio.run(main())
    columns, rows = _query(database_path(tmp_path), "SELECT * FROM page_text")

    assert columns == PAGE_TEXT_COLUMNS
    usage = json.dumps(USAGE)
    assert rows == [
        (NAME, "page", 0, "page_0001_pre_processed.jpg", 0, "Page 1 text", "done",
         MODEL, usage, "ok", None, PROVIDER, run_id),
        (NAME, "page", 1, "page_0002_pre_processed.jpg", 1, "[transcription error]",
         "failed", MODEL, None, "failed", "Error code: 400", PROVIDER, run_id),
        (NAME, "page", 2, "page_0003_pre_processed.jpg", 2, "Page 3 text", "done",
         MODEL, usage, "ok", None, PROVIDER, run_id),
        (NAME, "page", 3, "page_0004_pre_processed.jpg", 3, None, "pending",
         MODEL, None, None, None, PROVIDER, run_id),
        ("ledger-fe14010b/ledger", "page", None, "a.png_pre_processed.jpg", 0,
         "Ledger one", "done", MODEL, None, None, None, PROVIDER, run_id),
        ("ledger-fe14010b/ledger", "page", None, "b.png_pre_processed.jpg", 1,
         "Ledger two", "done", MODEL, None, None, None, PROVIDER, run_id),
        ("talk-ce7611e7/talk", "chunk", None, "talk_chunk_0001.mp3", 0,
         "Spoken words", "done", MODEL, None, None, None, PROVIDER, run_id),
    ]  # fmt: skip


def test_document_status_and_page_attempts(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("transcribe")
            lease = await seed_document(store, [page(0), page(1, "failed")])
            await store.append_response(
                ResponseRow(
                    provider=PROVIDER,
                    model=MODEL,
                    unit=UnitKey(lease.doc_id, KIND_PAGE, 1),
                    attempt=1,
                    raw='{"error": "invalid request"}',
                    raw_kind="http",
                    error="Error code: 400",
                    http_status=400,
                )
            )

    asyncio.run(main())
    database = database_path(tmp_path)
    status_columns, status = _query(database, "SELECT * FROM document_status")
    attempt_columns, attempts = _query(
        database,
        "SELECT document, kind, order_index, raw, raw_kind, http_status "
        "FROM page_attempts",
    )

    assert status_columns[:4] == ("document", "kind", "input_path", "status")
    assert [row[:8] for row in status] == [
        (NAME, "pdf", f"{NAME}.pdf", "failed", 2, 1, 1, 0)
    ]
    assert "raw" in _query(database, "SELECT * FROM page_attempts")[0]
    assert attempt_columns[0] == "document"
    assert attempts == [(NAME, "page", 1, '{"error": "invalid request"}', "http", 400)]


def test_read_helpers_open_short_read_only_connections(tmp_path: Path) -> None:
    assert read_store(tmp_path) is None
    with pytest.raises(FileNotFoundError), read_only_connection(tmp_path):
        pass

    build_store(tmp_path, [page(0), page(1)])
    store = read_store(tmp_path)
    assert store is not None
    document = store.document(NAME)
    assert document is not None and document.status == "complete"
    with read_only_connection(tmp_path) as connection:
        count = connection.execute("SELECT count(*) FROM page_text").fetchone()[0]
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM notes")
    assert count == 2


def test_read_helpers_refuse_a_versionless_database(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
    connection.execute("CREATE TABLE pages (document TEXT, page_index INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(VersionlessDatabaseError):
        read_store(tmp_path)
    with pytest.raises(VersionlessDatabaseError), read_only_connection(tmp_path):
        pass


@pytest.mark.parametrize(
    ("output", "root", "expected"),
    [
        ("out/doc-139d544b/doc.txt", "out", "doc-139d544b/doc"),
        ("out/sub/deep/c.txt", "out", "sub/deep/c"),
        ("docs/doc.txt", "docs", "doc"),
        ("elsewhere/my.file.txt", "out", "my.file"),
    ],
)
def test_document_name_is_the_output_relative_to_the_root(
    tmp_path: Path, output: str, root: str, expected: str
) -> None:
    assert document_name(tmp_path / output, tmp_path / root) == expected


def test_image_settings_record_matches_the_provenance_fingerprint() -> None:
    settings = {
        "jpeg_quality": 95,
        "high_target_box": (768, 1536),
        "model_name": MODEL,
        "provider": PROVIDER,
    }

    record = image_settings_record(settings, "openai", 24_000_000)

    assert record["high_target_box"] == [768, 1536]
    assert record["model_type"] == "openai"
    assert image_settings_fingerprint(record) == image_settings_fingerprint(
        {**settings, "max_pixels_per_page": 24_000_000, "model_type": "openai"}
    )


def test_text_units_have_no_page_index(tmp_path: Path) -> None:
    build_store(tmp_path, [Unit(0, content="Whole book", kind=KIND_TEXT)])

    _, rows = _query(
        database_path(tmp_path), "SELECT kind, page_index, text FROM page_text"
    )

    assert rows == [("text", None, "Whole book")]
