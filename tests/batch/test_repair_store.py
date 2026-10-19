"""Repair on the store: synchronous and batch repairs of failed page units.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_repair_store.py -q

Each test builds an output folder whose ``chronotranscriber.sqlite`` holds one
PDF document of five page units and a rendered ``doc.txt``, as a
transcription run leaves them, then repairs the failed units. The fake
transcriber records one attempt per call through the store's recorder, as
the model caller does; the batch tests run the fake batch backend of
``tests.batch.conftest``. No request leaves the machine.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fitz
import pytest

from chronotranscriber.batch import repair as repair_module
from chronotranscriber.common.batch import BatchHandle, BatchResultItem
from chronotranscriber.common.capture import AttemptRecord, record_attempt
from chronotranscriber.common.sqlite import read
from chronotranscriber.common.store import PlannedUnit, ResponseRow, UnitKey, UnitResult
from chronotranscriber.output.render import render_item
from chronotranscriber.output.store import (
    database_path,
    image_settings_record,
    open_store,
)
from chronotranscriber.settings import Settings
from chronotranscriber.sources.images import resolve_image_settings
from chronotranscriber.sources.pdf import PDF_PAGE_NAME
from chronotranscriber.spec import RepairSpec, resolve_repair
from tests.batch.conftest import DEFAULT_MODELS, FakeBackend, World, answer

PROVIDER = "openai"
MODEL = DEFAULT_MODELS[PROVIDER]

CONTENTS: Mapping[str, str] = {
    "failed": "[transcription error: {name}]",
    "not_possible": "[Transcription not possible]",
    "no_text": "[No transcribable text]",
}


@dataclass(frozen=True)
class Stored:
    """A transcribed document: its output root, source, rendered file and id."""

    root: Path
    pdf: Path
    output: Path
    doc_id: int

    @property
    def db(self) -> Path:
        return database_path(self.root)


def page_name(idx: int) -> str:
    return PDF_PAGE_NAME.format(number=idx + 1)


def make_pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for number in range(pages):
            doc.new_page(width=200, height=100).insert_text((20, 50), f"P{number}")
        doc.save(path)
    return path


def image_record(provider: str, model: str) -> dict[str, Any]:
    config, model_type, _section = resolve_image_settings(provider, model, None)
    return image_settings_record(config, model_type)


async def store_document(
    tmp_path: Path,
    outcomes: Sequence[str],
    *,
    provider: str = PROVIDER,
    model: str | None = None,
    record: Mapping[str, Any] | None | bool = True,
    formats: Sequence[str] = ("txt",),
    folder: str = "out",
) -> Stored:
    """Store document ``doc`` with one page unit per entry of *outcomes*.

    An outcome is ``ok``, ``failed``, ``not_possible`` or ``no_text``. Each
    unit gets one earlier response row; the document's files are rendered.
    *record* True stores the image settings of *model*, a mapping stores
    that record and None stores none.
    """
    model = model or DEFAULT_MODELS[provider]
    pdf = make_pdf(tmp_path / "in" / "doc.pdf", len(outcomes))
    root = tmp_path / folder
    root.mkdir(parents=True, exist_ok=True)
    output = root / "doc.txt"
    async with open_store(root) as store:
        await store.begin_run("transcribe")
        doc_id = await store.ensure_document("doc", input_path=pdf, kind="pdf")
        lease = await store.acquire(doc_id)
        assert lease is not None
        if record is True:
            await store.set_image_settings(lease, image_record(provider, model))
        elif isinstance(record, Mapping):
            await store.set_image_settings(lease, record)
        await store.plan_units(
            lease,
            "page",
            [PlannedUnit(idx, page_name(idx), idx) for idx in range(len(outcomes))],
            model=model,
            provider=provider,
        )
        for idx, outcome in enumerate(outcomes):
            content = (
                f"Text of page {idx}"
                if outcome == "ok"
                else CONTENTS[outcome].format(name=page_name(idx))
            )
            response_id = await store.append_response(
                ResponseRow(
                    provider=provider,
                    model=model,
                    unit=UnitKey(doc_id, "page", idx),
                    role="transcription",
                    attempt=1,
                    raw=json.dumps({"first": idx}),
                    raw_kind="http",
                )
            )
            await store.write_unit(
                lease,
                UnitResult(
                    kind="page",
                    idx=idx,
                    status="failed" if outcome == "failed" else "done",
                    model=model,
                    provider=provider,
                    content=content,
                    outcome=outcome,
                    label=page_name(idx),
                    source_index=idx,
                    error=content if outcome == "failed" else None,
                    provenance={"dpi": 72},
                    response_id=response_id,
                    attempts=1,
                ),
            )
        await render_item(store, doc_id, formats, output, lease)
    return Stored(root, pdf, output, doc_id)


def spec_for(path: Path, *flags: str, **values: Any) -> RepairSpec:
    given: dict[str, Any] = {"input": path, **values}
    given.setdefault("provider", PROVIDER)
    if "model" not in given:
        given["model"] = DEFAULT_MODELS[given["provider"]]
    for flag in flags:
        given[flag] = True
    return resolve_repair(given).spec


def rows(db: Path, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
    def query(connection: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in connection.execute(sql, params).fetchall()]

    return read(db, query)


def unit_rows(db: Path) -> list[tuple[Any, ...]]:
    return rows(
        db,
        "SELECT idx, status, outcome, content, response_id FROM units ORDER BY idx",
    )


def tree(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def assert_no_side_files(root: Path) -> None:
    names = tree(root)
    assert not [name for name in names if "repairs" in name.split("/")]
    assert not [name for name in names if ".bak" in name]
    assert not [name for name in names if name.endswith(".jsonl")]


class FakeTranscriber:
    """Answers each page by its label from *texts*; None fails the page.

    Every call records one attempt through the installed recorder, as the
    model caller does, and is kept in *calls* with its request options.
    """

    texts: dict[str, str | None] = {}
    calls: list[tuple[str, dict[str, Any]]] = []
    created: list[dict[str, Any]] = []

    def __init__(self, model: Any, env: Any, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs
        FakeTranscriber.created.append(kwargs)

    def update_context_image(self, path: Path | None) -> None:
        self.options["context_image"] = path

    async def transcribe_image_from_base64(
        self, data: str, mime: str, *, label: str | None = None, **options: Any
    ) -> dict[str, Any]:
        name = label or ""
        FakeTranscriber.calls.append((name, options))
        text = self.texts.get(name)
        await record_attempt(
            AttemptRecord(
                provider=self.model.provider,
                model=self.model.model,
                role="repair",
                attempt=1,
                label=name,
                raw=json.dumps({"repair": name, "text": text}),
                raw_kind="http",
                error=None if text is not None else "boom",
            )
        )
        if text is None:
            return {"output_text": "", "error": "boom"}
        return {"output_text": answer(text)}

    async def close(self) -> None:
        return None


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> type[FakeTranscriber]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "PageTranscriber", FakeTranscriber)
    FakeTranscriber.texts = {}
    FakeTranscriber.calls = []
    FakeTranscriber.created = []
    return FakeTranscriber


async def test_a_sync_repair_rewrites_the_failed_units_and_their_file(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "ok", "failed", "ok"])
    before = rows(stored.db, "SELECT response_id, raw FROM responses")
    transcriber.texts = {page_name(1): "Fixed one", page_name(3): "Fixed three"}

    result = await repair_module.repair_transcription(
        spec_for(stored.output), Settings()
    )

    assert (result.repaired, result.failed, result.selected) == (2, 0, 2)
    assert result.outputs == (stored.output,)
    responses = rows(
        stored.db, "SELECT response_id, raw, idx FROM responses ORDER BY response_id"
    )
    assert len(responses) == 7
    assert [row[:2] for row in responses[:5]] == before
    assert [row[2] for row in responses[5:]] == [1, 3]
    units = unit_rows(stored.db)
    assert [unit[:4] for unit in units] == [
        (0, "done", "ok", "Text of page 0"),
        (1, "done", "ok", "Fixed one"),
        (2, "done", "ok", "Text of page 2"),
        (3, "done", "ok", "Fixed three"),
        (4, "done", "ok", "Text of page 4"),
    ]
    assert {units[1][4], units[3][4]} == {row[0] for row in responses[5:]}
    text = stored.output.read_text(encoding="utf-8")
    assert text == (
        "Text of page 0\nFixed one\nText of page 2\nFixed three\nText of page 4\n"
    )
    assert text.endswith("\n") and not text.endswith("\n\n")
    (run,) = rows(stored.db, "SELECT exit_code FROM runs WHERE command = 'repair'")
    assert run == (0,)
    assert_no_side_files(tmp_path)


async def test_a_unit_that_fails_again_stays_failed(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "failed"])
    transcriber.texts = {page_name(1): "Fixed one", page_name(2): None}

    result = await repair_module.repair_transcription(
        spec_for(stored.output), Settings()
    )

    assert (result.repaired, result.failed, result.selected) == (1, 1, 2)
    status = [unit[1] for unit in unit_rows(stored.db)]
    assert status == ["done", "done", "failed"]
    lines = stored.output.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["Text of page 0", "Fixed one"]
    assert lines[2].startswith(f"{page_name(2)}: [transcription error")
    (run,) = rows(stored.db, "SELECT exit_code FROM runs WHERE command = 'repair'")
    assert run == (1,)


async def test_the_recorded_detail_and_prompt_inputs_reach_the_transcriber(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    record = {**image_record(PROVIDER, MODEL), "resolved_detail": "low"}
    stored = await store_document(tmp_path, ["failed"], record=record)
    transcriber.texts = {page_name(0): "Fixed"}

    await repair_module.repair_transcription(spec_for(stored.output), Settings())

    assert transcriber.calls == [(page_name(0), {"image_detail": "low"})]
    (created,) = transcriber.created
    assert created["role"] == "repair"
    assert created["schema_path"] is not None


async def test_the_sidecar_context_image_reaches_the_transcriber(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    stored = await store_document(tmp_path, ["failed"])
    image = stored.pdf.with_name("doc_transcr_context_image.png")
    image.write_bytes(b"context image")
    transcriber.texts = {page_name(0): "Fixed"}

    await repair_module.repair_transcription(spec_for(stored.output), Settings())

    (created,) = transcriber.created
    assert created["context_image"] == image.resolve()


async def test_the_input_path_finds_the_document_too(
    tmp_path: Path, transcriber: type[FakeTranscriber]
) -> None:
    stored = await store_document(tmp_path, ["failed", "ok"], folder="in")
    plan = repair_module.plan_repair(spec_for(stored.pdf), Settings())
    assert plan.document is not None
    assert plan.document.doc_id == stored.doc_id
    assert plan.document.output == stored.output
    assert plan.failures == [0]


def store_raw_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every fake batch result a downloaded record, as the backends do."""
    real = FakeBackend._fetch_results

    def fetch(
        self: FakeBackend, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        items, error = real(self, handle)
        return [
            replace(
                item,
                raw=json.dumps({"custom_id": item.custom_id, "text": item.text}),
                raw_kind="line",
            )
            for item in items
        ], error

    monkeypatch.setattr(FakeBackend, "_fetch_results", fetch)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_a_batch_repair_submits_finalizes_and_renders_the_units(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    world.file_uploads = True
    store_raw_records(monkeypatch)
    stored = await store_document(
        tmp_path, ["ok", "failed", "ok", "not_possible", "ok"], provider=provider
    )

    result = await repair_module.repair_transcription(
        spec_for(stored.output, "all_failures", "batch", provider=provider),
        Settings(),
    )

    assert (result.repaired, result.failed, result.selected) == (2, 0, 2)
    (submission,) = rows(
        stored.db, "SELECT submission_id, status, batch_id, provider FROM submissions"
    )
    submission_id, status, batch_id, stored_provider = submission
    assert (status, stored_provider) == ("finalized", provider)
    requests = rows(
        stored.db,
        "SELECT custom_id, idx FROM submission_requests WHERE submission_id = ? "
        "ORDER BY idx",
        (submission_id,),
    )
    assert [idx for _custom_id, idx in requests] == [1, 3]
    responses = rows(
        stored.db,
        "SELECT custom_id, raw_kind, idx FROM responses WHERE submission_id = ? "
        "ORDER BY idx",
        (submission_id,),
    )
    assert [(kind, idx) for _cid, kind, idx in responses] == [("line", 1), ("line", 3)]
    assert len(rows(stored.db, "SELECT 1 FROM responses")) == 7
    units = unit_rows(stored.db)
    custom_ids = dict((idx, custom_id) for custom_id, idx in requests)
    assert units[1][1:4] == ("done", "ok", f"text of {custom_ids[1]}")
    assert units[3][1:4] == ("done", "ok", f"text of {custom_ids[3]}")
    provenance = rows(stored.db, "SELECT provenance FROM units WHERE idx = 1")
    assert "render_dpi" in json.loads(provenance[0][0])
    assert stored.output.read_text(encoding="utf-8") == (
        f"Text of page 0\ntext of {custom_ids[1]}\nText of page 2\n"
        f"text of {custom_ids[3]}\nText of page 4\n"
    )
    assert world.cleaned == [batch_id]
    assert_no_side_files(tmp_path)


async def test_a_batch_repair_sends_the_sidecar_context_image(
    world: World, tmp_path: Path
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed"])
    stored.pdf.with_name("doc_transcr_context_image.png").write_bytes(b"context")

    await repair_module.repair_transcription(
        spec_for(stored.output, "batch"), Settings()
    )

    (body,) = world.bodies()
    assert "data:image/png;base64,Y29udGV4dA==" in json.dumps(body)


class ApiError(Exception):
    """A provider error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - refused")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "status"),
    [(ApiError(400), "failed"), (RuntimeError("connection lost"), "unresolved")],
)
async def test_a_batch_not_accepted_leaves_the_units_as_they_were(
    world: World, tmp_path: Path, error: Exception, status: str
) -> None:
    world.errors[("openai", "submit")] = [error]
    stored = await store_document(tmp_path, ["ok", "failed"])
    before = unit_rows(stored.db)
    text = stored.output.read_bytes()

    result = await repair_module.repair_transcription(
        spec_for(stored.output, "batch"), Settings()
    )

    assert (result.repaired, result.failed) == (0, 1)
    assert unit_rows(stored.db) == before
    assert stored.output.read_bytes() == text
    assert rows(stored.db, "SELECT status FROM submissions") == [(status,)]
