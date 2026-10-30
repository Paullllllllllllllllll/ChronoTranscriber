"""The store-backed unit pipelines: one committed row per unit, then progress.

Run on Windows with:

    uv --directory <repo> run pytest tests/methods/test_units.py -q

Each test runs a pipeline on a fake handler against a real store in
``tmp_path`` and reads the rows back through ``mode=ro`` connections.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.sqlite import DatabaseFullError, read
from chronotranscriber.common.store import (
    LeaseLostError,
    Store,
    UnitKey,
    WriterOptions,
    current_unit,
)
from chronotranscriber.events import PageDone
from chronotranscriber.methods import Item, PageCounter
from chronotranscriber.methods.units import (
    DocumentHandle,
    UnitsFailedError,
    absolute_order,
    run_unit_files,
    run_unit_stream,
)
from tests.conftest import make_job, make_spec
from tests.methods.conftest import (
    MODEL,
    PROVIDER,
    TOOL,
    FakePayload,
    open_handle,
    payloads,
    stream,
    units,
)
from tests.recording_events import RecordingEvents

USAGE = {"input_tokens": 1500, "output_tokens": 40, "total_tokens": 1540}


def _answer(payload: FakePayload, text: str, **raw: Any) -> Any:
    response = {"output_text": text, "usage": dict(USAGE), **raw}
    return (payload, payload.image_name, text, response, payload.index)


class _CheckingEvents(RecordingEvents):
    """Read the reported unit's row through a read-only connection first."""

    def __init__(self, db: Path) -> None:
        super().__init__()
        self.db = db
        self.seen: list[tuple[int, str, int]] = []

    def page_done(self, event: PageDone) -> None:
        def query(connection: sqlite3.Connection) -> tuple[str, int]:
            (status,) = connection.execute(
                "SELECT status FROM units WHERE idx = ?", (event.page,)
            ).fetchone()
            (settled,) = connection.execute(
                "SELECT count(*) FROM units WHERE status != 'pending'"
            ).fetchone()
            return str(status), int(settled)

        status, settled = read(self.db, query)
        self.seen.append((event.page, status, settled))
        super().page_done(event)


def test_five_units_are_five_commits_each_visible_before_its_progress(
    tmp_path: Path,
) -> None:
    db = tmp_path / "chronotranscriber.sqlite"
    events = _CheckingEvents(db)
    contexts: list[UnitKey | None] = []
    writes: list[int] = []

    async def handler(payload: FakePayload, transcriber: Any) -> Any:
        assert transcriber == "client"
        contexts.append(current_unit())
        if payload.index == 2:
            placeholder = f"[transcription error: {payload.image_name}]"
            return _answer(payload, placeholder, error="model refused the page")
        return _answer(payload, f"Text of page {payload.index + 1}")

    async def main() -> DocumentHandle:
        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=5)
            write_unit = store.write_unit

            async def counting(lease: Any, result: Any) -> None:
                writes.append(result.idx)
                await write_unit(lease, result)

            store.write_unit = counting  # type: ignore[method-assign]
            counter = PageCounter(events, handle.name, total=5)
            with pytest.raises(UnitsFailedError) as raised:
                await run_unit_stream(
                    handle,
                    stream(payloads(5)),
                    "client",
                    handler,
                    concurrency=1,
                    on_unit=counter,
                )
            assert events.seen and len(events.of("page_done")) == 5
            error = raised.value
            assert str(error) == "1 of 5 page(s) failed to transcribe for 'doc.pdf'"
            assert (error.failed_pages, error.total_pages) == (1, 5)
            assert error.write_errors == []
            return handle

    handle = asyncio.run(main())

    assert writes == [0, 1, 2, 3, 4]
    assert events.seen == [
        (0, "done", 1),
        (1, "done", 2),
        (2, "failed", 3),
        (3, "done", 4),
        (4, "done", 5),
    ]
    assert [e.status for e in events.of("page_done")] == ["ok"] * 2 + ["failed"] + [
        "ok"
    ] * 2
    assert contexts == [UnitKey(handle.doc_id, "page", i) for i in range(5)]
    rows = units(db)
    assert [row["status"] for row in rows] == ["done"] * 2 + ["failed"] + ["done"] * 2
    failed = rows[2]
    assert failed["content"] == "[transcription error: page_0003.png]"
    assert failed["error"] == "model refused the page"
    assert failed["outcome"] == "failed"
    done = rows[0]
    assert done["content"] == "Text of page 1"
    assert (done["label"], done["source_index"], done["outcome"]) == (
        "page_0001.png",
        0,
        "ok",
    )
    assert (done["tool"], done["model"], done["provider"]) == (TOOL, MODEL, PROVIDER)
    assert json.loads(done["usage"]) == USAGE
    assert json.loads(done["provenance"]) == {"rendered": "page_0001.png"}
    assert (done["attempts"], done["response_id"], done["error"]) == (1, None, None)
    assert {row["run_id"] for row in rows} == {handle.run_id}


def test_a_full_write_queue_pauses_provider_calls_until_the_commit_lands(
    tmp_path: Path,
) -> None:
    """A reader holding a transaction blocks the commit; the run waits and warns."""
    db = tmp_path / "chronotranscriber.sqlite"
    log: list[str] = []
    warnings: list[float] = []
    readers: list[sqlite3.Connection] = []
    options = WriterOptions(
        busy_timeout=0.05,
        warn_interval=0.0,
        queue_size=1,
        on_busy=lambda _path, waited: warnings.append(waited),
    )

    async def main() -> None:
        async with Store(db, TOOL, writer=options) as store:
            handle = await open_handle(store, pages=2)
            first, second = payloads(2)
            gate = asyncio.Event()

            async def producer() -> AsyncIterator[FakePayload]:
                yield first
                await gate.wait()
                yield second

            async def handler(payload: FakePayload, _transcriber: Any) -> Any:
                log.append(f"call {payload.index}")
                if payload.index == 0:
                    reader = sqlite3.connect(
                        f"{db.resolve().as_uri()}?mode=ro",
                        uri=True,
                        isolation_level=None,
                    )
                    reader.execute("BEGIN")
                    reader.execute("SELECT count(*) FROM units").fetchone()
                    readers.append(reader)
                    gate.set()
                return _answer(payload, f"Text {payload.index}")

            async def release() -> None:
                while not warnings:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.3)
                log.append("release")
                readers[0].execute("COMMIT")
                readers[0].close()

            async with asyncio.timeout(30):
                processed, _ = await asyncio.gather(
                    run_unit_stream(handle, producer(), None, handler, concurrency=2),
                    release(),
                )
            assert processed == 2

    asyncio.run(main())

    assert log == ["call 0", "release", "call 1"]
    assert warnings
    assert [row["status"] for row in units(db)] == ["done", "done"]


def test_a_full_database_fails_the_unit_loudly_and_keeps_earlier_units(
    tmp_path: Path,
) -> None:
    db = tmp_path / "chronotranscriber.sqlite"
    events = RecordingEvents()

    async def main() -> UnitsFailedError:
        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=3)

            async def handler(payload: FakePayload, _transcriber: Any) -> Any:
                if payload.index < 2:
                    return _answer(payload, f"Text {payload.index}")
                await store._write(
                    lambda c: c.execute(
                        "PRAGMA max_page_count = "
                        f"{c.execute('PRAGMA page_count').fetchone()[0] + 1}"
                    )
                )
                return _answer(payload, "x" * 200_000)

            with pytest.raises(UnitsFailedError) as raised:
                await run_unit_stream(
                    handle,
                    stream(payloads(3)),
                    None,
                    handler,
                    concurrency=1,
                    on_unit=PageCounter(events, handle.name, total=3),
                )
            return raised.value

    error = asyncio.run(main())

    assert isinstance(error.__cause__, DatabaseFullError)
    assert [type(e) for e in error.write_errors] == [DatabaseFullError]
    assert "could not be stored: DatabaseFullError" in str(error)
    assert [event.page for event in events.of("page_done")] == [0, 1]
    rows = units(db)
    assert [row["status"] for row in rows] == ["done", "done", "pending"]
    assert [row["content"] for row in rows[:2]] == ["Text 0", "Text 1"]
    assert read(db, lambda c: c.execute("PRAGMA integrity_check").fetchone()[0]) == (
        "ok"
    )


def test_a_lost_lease_stops_the_document_and_is_raised(tmp_path: Path) -> None:
    db = tmp_path / "chronotranscriber.sqlite"
    now = [1_000.0]
    calls: list[int] = []

    async def main() -> None:
        async with Store(db, TOOL, clock=lambda: now[0]) as store:
            handle = await open_handle(store, pages=3)

            async def handler(payload: FakePayload, _transcriber: Any) -> Any:
                calls.append(payload.index)
                if payload.index == 1:
                    now[0] += store.lease_ttl + 1
                return _answer(payload, f"Text {payload.index}")

            with pytest.raises(LeaseLostError):
                await run_unit_stream(
                    handle, stream(payloads(3)), None, handler, concurrency=1
                )

    asyncio.run(main())

    assert calls == [0, 1]
    assert [row["status"] for row in units(db)] == ["done", "pending", "pending"]


def test_file_units_take_their_absolute_positions(tmp_path: Path) -> None:
    db = tmp_path / "chronotranscriber.sqlite"
    files = [
        tmp_path / "page_0003_tess_preprocessed.png",
        tmp_path / "page_0005_tess_preprocessed.png",
    ]
    called: list[tuple[str, int]] = []

    async def handler(path: Path, idx: int) -> Any:
        called.append((path.name, idx))
        return (str(path), path.name, f"OCR {idx}", None, idx)

    async def main() -> int:
        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=5, model="tesseract")
            return await run_unit_files(handle, files, handler, concurrency=2)

    assert asyncio.run(main()) == 2
    assert sorted(called) == [
        ("page_0003_tess_preprocessed.png", 2),
        ("page_0005_tess_preprocessed.png", 4),
    ]
    rows = {row["idx"]: row for row in units(db)}
    assert rows[2]["content"] == "OCR 2"
    assert (rows[4]["label"], rows[4]["source_index"]) == (
        "page_0005_tess_preprocessed.png",
        4,
    )
    assert rows[4]["model"] == "tesseract"
    assert [rows[i]["status"] for i in (0, 1, 3)] == ["pending"] * 3


def test_absolute_order_uses_page_names_then_the_folder_listing() -> None:
    files = [Path("page_0007_tess_preprocessed.jpg"), Path("b.png"), Path("c.png")]

    assert absolute_order(files, {"b.png": 4}) == {
        "page_0007_tess_preprocessed.jpg": 6,
        "b.png": 4,
        "c.png": 2,
    }


def test_absolute_order_puts_folder_images_named_like_pages_at_their_listing() -> None:
    files = [
        Path("page_1_tess_preprocessed.png"),
        Path("page_01_tess_preprocessed.png"),
    ]
    order = {files[0].name: 1, files[1].name: 0}

    assert absolute_order(files, order) == order


def test_todo_and_reset_follow_the_store(tmp_path: Path) -> None:
    db = tmp_path / "chronotranscriber.sqlite"

    async def main() -> list[list[int]]:
        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=4)
            await handle.write(0, status="done", content="a")
            await handle.write(1, status="failed", content="[transcription error]")
            seen = [handle.todo(), handle.todo(retry_errors=True)]
            assert await handle.reset([0]) == 1
            seen.append(handle.todo())
            return seen

    assert asyncio.run(main()) == [[2, 3], [1, 2, 3], [0, 2, 3]]


def test_items_and_jobs_default_to_no_store(tmp_path: Path) -> None:
    job = make_job(
        make_spec(tmp_path / "doc.pdf"), output_dir=None, input_root=tmp_path
    )
    item = Item(tmp_path / "doc.pdf", "pdf", "llm")

    assert job.store is None
    assert item.document is None


def test_the_working_dir_is_transient_unless_files_are_kept(tmp_path: Path) -> None:
    source = tmp_path / "in" / "doc.pdf"
    out = tmp_path / "out"
    transient = make_job(make_spec(source), output_dir=out, input_root=source.parent)
    kept = make_job(
        make_spec(source, keep_working_files=True),
        output_dir=out,
        input_root=source.parent,
    )

    with transient.working_dir(source) as folder:
        (folder / "chunk.wav").write_bytes(b"RIFF")
        assert folder.is_dir()
    assert not folder.exists()
    assert not out.exists()

    with kept.working_dir(source) as folder:
        (folder / "chunk.wav").write_bytes(b"RIFF")
    assert folder == kept.locate(source).folder
    assert (folder / "chunk.wav").is_file()
