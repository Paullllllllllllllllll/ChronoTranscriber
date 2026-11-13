"""The txt and md renderers and render_item.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_render.py -q

The locked-file test holds the target open with ``open()``, which on Windows
shares it for reading and writing but not for deletion, so the replacement
fails.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.store import Lease, UnitResult
from chronotranscriber.output import render
from chronotranscriber.output.render import (
    FAILED_PLACEHOLDER,
    document_targets,
    preview_database,
    render_database,
    render_item,
    render_targets,
    text_formats,
)
from chronotranscriber.output.store import (
    KIND_PAGE,
    KIND_TEXT,
    database_path,
    open_store,
)
from tests.output.test_ct_store import NAME, Unit, page, seed_document


def _seed_and_render(
    root: Path, units: list[Unit], formats: tuple[str, ...] = ("txt", "md")
) -> tuple[bytes, bytes]:
    output = root / "doc-139d544b" / "doc.txt"

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("render")
            lease = await seed_document(store, units)
            await render_item(store, NAME, formats, output, lease)

    asyncio.run(main())
    return output.read_bytes(), output.with_suffix(".md").read_bytes()


def test_pending_units_are_left_out_and_failed_units_keep_their_header(
    tmp_path: Path,
) -> None:
    units = [
        page(0, "pending"),
        page(1),
        page(2, "failed", content=None),
        page(3, "failed", content="[transcription error: timeout]"),
    ]

    txt, md = _seed_and_render(tmp_path, units)

    lines = [
        "Page 2 text",
        f"page_0003_pre_processed.jpg: {FAILED_PLACEHOLDER}",
        "page_0004_pre_processed.jpg: [transcription error: timeout]",
    ]
    assert txt == ("\n".join(lines) + "\n").encode("utf-8")
    assert md == ("\n\n".join(lines) + "\n").encode("utf-8")


def test_placeholders_without_an_image_name_name_the_page(tmp_path: Path) -> None:
    units = [
        page(0, label=None, content="Text ending in blank lines\n\n\n"),
        page(1, "done", label=None, content="[No transcribable text]"),
    ]

    txt, _ = _seed_and_render(tmp_path, units)

    assert txt == b"Text ending in blank lines\nPage 2: [No transcribable text]\n"


def test_a_text_unit_is_written_as_stored(tmp_path: Path) -> None:
    units = [Unit(0, content="Book text.\n\nSecond paragraph.\n", kind=KIND_TEXT)]

    txt, md = _seed_and_render(tmp_path, units)

    assert txt == md == b"Book text.\n\nSecond paragraph.\n"


def test_a_document_without_rendered_units_gets_no_file(tmp_path: Path) -> None:
    output = tmp_path / "doc.txt"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("render")
            lease = await seed_document(store, [page(0, "pending")])
            report = await render_item(store, lease.doc_id, ("txt",), output, lease)
            assert report.written == report.stale == []

    asyncio.run(main())
    assert not output.exists()


def test_render_item_needs_the_lease_of_its_document(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("render")
            lease = await seed_document(store, [page(0)])
            other = Lease(lease.doc_id + 1, lease.holder, 1, lease.until)
            with pytest.raises(ValueError, match="lease"):
                await render_item(store, NAME, ("txt",), tmp_path / "d.txt", other)
            with pytest.raises(KeyError):
                await render_item(store, "missing", ("txt",), tmp_path / "d.txt", lease)

    asyncio.run(main())


def test_targets_follow_the_text_formats(tmp_path: Path) -> None:
    output = tmp_path / "my.file.txt"

    targets = render_targets(output, ("sqlite", "md", "txt"))

    assert text_formats(("sqlite",)) == ("txt",)
    assert [(target.path.name, target.format) for target in targets] == [
        ("my.file.md", "md"),
        ("my.file.txt", "txt"),
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing")
def test_a_locked_output_is_marked_stale_and_replaced_later(tmp_path: Path) -> None:
    output = tmp_path / "doc-139d544b" / "doc.txt"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("render")
            lease = await seed_document(store, [page(0)])
            first = await render_item(store, NAME, ("txt",), output, lease)
            assert first.written == [output]
            await store.write_unit(
                lease,
                UnitResult(
                    kind=KIND_PAGE,
                    idx=0,
                    status="done",
                    model="gpt-5.6-luna",
                    provider="openai",
                    content="Corrected text",
                ),
            )
            with open(output, "rb"):
                locked = await render_item(store, NAME, ("txt",), output, lease)
            assert locked.stale == [output]
            assert locked.written == []
            assert "marked stale" in locked.warnings[0]
            assert output.read_bytes() == b"Page 1 text\n"
            assert store.units(lease.doc_id)[0].content == "Corrected text"
            assert [row.stale for row in store.outputs(lease.doc_id)] == [1]

            later = await render_item(store, NAME, ("txt",), output, lease)
            assert later.written == [output]
            assert [row.stale for row in store.outputs(lease.doc_id)] == [0]

    asyncio.run(main())
    assert output.read_bytes() == b"Corrected text\n"


def _seeded(root: Path) -> Path:
    """Seed document NAME, render its txt file and return that file."""
    output = root / "doc-139d544b" / "doc.txt"

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            lease = await seed_document(store, [page(0), page(1)])
            await render_item(store, NAME, ("txt",), output, lease)

    asyncio.run(main())
    return output


def _change_page(root: Path, content: str, *, stale: bool = False) -> None:
    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            row = store.document(NAME)
            assert row is not None
            lease = await store.acquire(row.doc_id)
            assert lease is not None
            await store.write_unit(
                lease,
                UnitResult(
                    kind=KIND_PAGE,
                    idx=0,
                    status="done",
                    model="gpt-5.6-luna",
                    provider="openai",
                    content=content,
                ),
            )

    asyncio.run(main())
    if stale:
        connection = sqlite3.connect(database_path(root))
        try:
            connection.execute("UPDATE outputs SET stale = 1")
            connection.commit()
        finally:
            connection.close()


def test_render_database_rerenders_a_stale_output(tmp_path: Path) -> None:
    output = _seeded(tmp_path)
    _change_page(tmp_path, "Page 1 corrected", stale=True)

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.written == [output]
    assert not result.failed
    assert output.read_bytes() == b"Page 1 corrected\nPage 2 text\n"
    reader = open_store(tmp_path)
    (doc,) = reader.documents()
    assert [row.stale for row in reader.outputs(doc.doc_id)] == [0]


def test_render_database_leaves_a_hand_edited_file_without_force(
    tmp_path: Path,
) -> None:
    output = _seeded(tmp_path)
    output.write_bytes(b"edited by hand\n")
    _change_page(tmp_path, "Page 1 corrected")

    (kept,) = asyncio.run(render_database(database_path(tmp_path)))
    assert kept.kept == [output]
    assert output.read_bytes() == b"edited by hand\n"

    (forced,) = asyncio.run(render_database(database_path(tmp_path), force=True))
    assert forced.written == [output]
    assert output.read_bytes() == b"Page 1 corrected\nPage 2 text\n"


def test_render_database_adds_formats_beside_the_registered_file(
    tmp_path: Path,
) -> None:
    output = _seeded(tmp_path)

    (result,) = asyncio.run(
        render_database(database_path(tmp_path), formats=("txt", "md"))
    )

    assert result.unchanged == [output]
    assert result.written == [output.with_suffix(".md")]
    assert output.with_suffix(".md").read_bytes() == b"Page 1 text\n\nPage 2 text\n"


def _rows(root: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(database_path(root))
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


_OUTPUTS = (
    "SELECT doc_id, path, format, sha256, rendered_at, run_id, stale FROM outputs"
)
_LEASES = "SELECT lease_run_id, lease_until, lease_generation FROM documents"


def test_render_database_writes_nothing_for_a_matching_document(
    tmp_path: Path,
) -> None:
    output = _seeded(tmp_path)
    outputs, leases = _rows(tmp_path, _OUTPUTS), _rows(tmp_path, _LEASES)
    (runs,) = _rows(tmp_path, "SELECT count(*) FROM runs")

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.unchanged == [output]
    assert (result.written, result.kept, result.stale) == ([], [], [])
    assert not result.failed
    assert result.database == database_path(tmp_path)
    assert _rows(tmp_path, _OUTPUTS) == outputs
    assert _rows(tmp_path, _LEASES) == leases
    assert _rows(tmp_path, "SELECT count(*) FROM runs") == [(runs[0] + 1,)]


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE outputs SET stale = 1",
        "UPDATE outputs SET sha256 = 'recorded elsewhere'",
        "DELETE FROM outputs",
    ],
)
def test_render_database_records_a_matching_file_whose_row_differs(
    tmp_path: Path, sql: str
) -> None:
    output = _seeded(tmp_path)
    connection = sqlite3.connect(database_path(tmp_path))
    try:
        connection.execute(sql)
        connection.commit()
    finally:
        connection.close()
    ((_, _, generation),) = _rows(tmp_path, _LEASES)

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.unchanged == [output]
    reader = open_store(tmp_path)
    (doc,) = reader.documents()
    (row,) = reader.outputs(doc.doc_id)
    assert (row.path, row.stale) == ("doc-139d544b/doc.txt", 0)
    assert row.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert doc.lease_generation == generation + 1


def test_render_database_skips_a_document_another_run_holds(tmp_path: Path) -> None:
    _seeded(tmp_path)

    async def hold() -> None:
        store = open_store(tmp_path)
        await store.open()
        await store.begin_run("transcribe")
        row = store.document(NAME)
        assert row is not None
        assert await store.acquire(row.doc_id) is not None
        await store._require_writer().close()

    asyncio.run(hold())

    (result,) = asyncio.run(render_database(database_path(tmp_path)))

    assert result.skipped is not None and result.failed
    assert result.written == []


def test_render_database_reads_each_document_when_its_turn_comes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seeded(tmp_path)
    second = "doc-zzzzzzzz/other"

    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("transcribe")
            lease = await seed_document(store, [page(0)], name=second)
            await render_item(
                store, second, ("txt",), tmp_path / f"{second}.txt", lease
            )

    asyncio.run(main())
    real = render.read_only
    opened = 0

    @contextlib.contextmanager
    def read_only(path: Path) -> Iterator[sqlite3.Connection]:
        # Another run takes the second document after the first was read.
        nonlocal opened
        opened += 1
        if opened == 2:
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "UPDATE documents SET lease_run_id = 'other-run', "
                    "lease_until = ? WHERE name = ?",
                    (time.time() + 600, second),
                )
                connection.commit()
            finally:
                connection.close()
        with real(path) as connection:
            yield connection

    monkeypatch.setattr(render, "read_only", read_only)

    results = asyncio.run(render_database(database_path(tmp_path)))

    by_name = {result.name: result for result in results}
    assert by_name[NAME].unchanged and by_name[NAME].skipped is None
    assert by_name[second].skipped == "another run holds its lease"


def test_preview_database_writes_nothing(tmp_path: Path) -> None:
    output = _seeded(tmp_path)
    _change_page(tmp_path, "Page 1 corrected")
    database = database_path(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    (result,) = preview_database(database, formats=("txt", "md"))

    assert result.written == [output, output.with_suffix(".md")]
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_document_targets_follow_the_registered_outputs(tmp_path: Path) -> None:
    targets = document_targets(tmp_path, "a/b", [])
    assert [(t.path, t.format) for t in targets] == [(tmp_path / "a" / "b.txt", "txt")]
