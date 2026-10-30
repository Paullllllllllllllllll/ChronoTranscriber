"""The classify subcommands end to end, against a fake decision endpoint.

Every decision request goes to a mocked ``httpx`` transport that answers
each question of the OpenRouter body (the default model's dialect). The
commands run in-process through the console entry point; the oracle reads
the store's ``classifications`` view and the JSON line.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from chronotranscriber import __main__ as entry
from chronotranscriber import classify
from chronotranscriber.common.classifiers import load_classifier
from chronotranscriber.common.store import PlannedUnit, file_sha256
from chronotranscriber.output.store import KIND_PAGE, database_path, open_store
from chronotranscriber.run import item_location
from chronotranscriber.sources import SourceItem
from tests.characterization.inputs import make_image_folder, make_scanned_pdf
from tests.cli.conftest import json_lines

SHIPPED = {
    "page_kind": "page",
    "script_and_hand": "page",
    "language": "page",
    "layout": "page",
    "document_type": "document",
}


class FakeDecisions:
    """Answer every question of an OpenRouter decision body; keep the bodies."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def answer(self, question: dict[str, Any]) -> dict[str, Any]:
        if question["type"] == "choice":
            first = next(iter(question["criteria"]))
            return {"type": "choice", "choice": first, "probabilities": {first: 0.9}}
        if question["type"] == "score":
            return {"type": "score", "score": 1, "probabilities": {}}
        return {"type": "noul", "probability": 0.8}

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        answers = {
            name: self.answer(question) for name, question in body["questions"].items()
        }
        payload = {
            "model": body["model"],
            "answers": answers,
            "usage": {"input_tokens": 50, "output_tokens": 5},
        }
        return httpx.Response(200, json=payload)

    def images(self, body: dict[str, Any]) -> int:
        return sum(part["type"] == "image_url" for part in body["state"])


@pytest.fixture
def decisions(monkeypatch: pytest.MonkeyPatch) -> FakeDecisions:
    fake = FakeDecisions()
    mock = httpx.MockTransport(fake.handle)

    async def handle(
        transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        return await mock.handle_async_request(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return fake


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any]:
    code = entry.main(["classify", *argv, "--json"])
    (payload,) = json_lines(capsys.readouterr().out)
    return code, payload


def _view(root: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database_path(root))
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row) for row in connection.execute("SELECT * FROM classifications")
        ]
    finally:
        connection.close()


def _count(root: Path, table: str) -> int:
    connection = sqlite3.connect(database_path(root))
    try:
        return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def _document(pdf: Path, out: Path) -> str:
    return item_location(
        SourceItem(pdf, "pdf"), output_dir=out, layout="hash", input_root=pdf
    )[1]


def _classifier(tmp_path: Path, instructions: str) -> Path:
    path = tmp_path / "front.json"
    body = {
        "name": "front",
        "version": 1,
        "unit": "page",
        "per_request": 2,
        "questions": [
            {"name": "title", "type": "predicate", "instructions": instructions}
        ],
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", list(SHIPPED))
def test_every_shipped_classifier_loads(name: str) -> None:
    classifier = load_classifier(classify.CLASSIFIERS_DIR / f"{name}.json")
    assert (classifier.name, classifier.unit) == (name, SHIPPED[name])


def test_the_shipped_classifiers_are_the_listed_ones() -> None:
    assert classify.classifier_names() == sorted(SHIPPED)


def test_classify_pages_records_one_answer_per_page_and_question(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=3)
    out = tmp_path / "out"

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--classifier",
        "script_and_hand",
    )

    assert code == 0, payload
    assert payload["command"] == "classify"
    assert payload["unit"] == "page"
    assert payload["classifier"] == "script_and_hand"
    assert (
        payload["items_total"],
        payload["units_asked"],
        payload["units_reused"],
    ) == (
        1,
        3,
        0,
    )
    assert payload["usage"]["classify"]["input_tokens"] == 50
    assert len(decisions.bodies) == 1
    assert list(decisions.bodies[0]["questions"]) == [
        "script__01",
        "script__02",
        "script__03",
    ]
    rows = _view(out)
    assert [row["locator"] for row in rows] == [
        f"page_{n:04d}_pre_processed.jpg" for n in (1, 2, 3)
    ]
    assert [row["ordinal"] for row in rows] == [0, 1, 2]
    assert {row["document"] for row in rows} == {_document(pdf, out)}
    assert {row["input_sha256"] for row in rows} == {file_sha256(pdf)}
    assert {row["value"] for row in rows} == {"printed_roman"}
    assert all(row["doc_id"] is None and row["reused"] == 0 for row in rows)
    assert _count(out, "documents") == 0
    assert _count(out, "units") == 0


def test_classify_pages_groups_by_per_request_and_takes_pages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    folder = make_image_folder(tmp_path / "scans", pages=5)
    out = tmp_path / "out"
    classifier = _classifier(tmp_path, "The page is a title page.")

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(folder),
        "--type",
        "images",
        "--output",
        str(out),
        "--classifier",
        str(classifier),
        "--pages",
        "2-4",
    )

    assert code == 0, payload
    assert payload["units_asked"] == 3
    assert [len(body["questions"]) for body in decisions.bodies] == [2, 1]
    rows = _view(out)
    assert [row["ordinal"] for row in rows] == [1, 2, 3]
    assert {row["kind"] for row in rows} == {"predicate"}
    assert {row["probability"] for row in rows} == {0.8}


def test_classify_documents_sends_the_first_pages_in_one_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    folder = tmp_path / "in"
    make_scanned_pdf(folder / "long.pdf", pages=12)
    make_scanned_pdf(folder / "short.pdf", pages=2)
    out = tmp_path / "out"

    code, payload = _run(
        capsys,
        "documents",
        "--input",
        str(folder),
        "--all",
        "--output",
        str(out),
        "--classifier",
        "document_type",
    )

    assert code == 0, payload
    assert (payload["unit"], payload["units_asked"]) == ("document", 2)
    assert sorted(decisions.images(body) for body in decisions.bodies) == [2, 10]
    assert all(list(body["questions"]) == ["type"] for body in decisions.bodies)
    rows = _view(out)
    assert [(row["unit"], row["locator"], row["ordinal"]) for row in rows] == [
        ("document", "document", 0),
        ("document", "document", 0),
    ]
    assert {row["value"] for row in rows} == {"printed_book"}


def test_a_classifier_of_another_unit_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "book.pdf", pages=1)

    code, payload = _run(
        capsys, "documents", "--input", str(pdf), "--classifier", "page_kind"
    )

    assert code == 2
    assert payload["unit"] == "document"
    assert decisions.bodies == []


def test_a_missing_classifier_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "book.pdf", pages=1)

    assert _run(capsys, "pages", "--input", str(pdf))[0] == 2
    code, payload = _run(
        capsys, "pages", "--input", str(pdf), "--classifier", "no_such_classifier"
    )
    assert (code, payload["classifier"], payload["items_total"]) == (2, "", 0)
    assert decisions.bodies == []


def test_the_dry_run_counts_units_and_requests_and_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    decisions: FakeDecisions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=3)
    out = tmp_path / "out"

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--classifier",
        "page_kind",
        "--dry-run",
    )

    assert code == 0, payload
    assert payload["dry_run"] is True
    assert payload["to_classify"] == [
        {
            "name": "book.pdf",
            "type": "pdf",
            "document": _document(pdf, out),
            "units": 3,
            "requests": 1,
        }
    ]
    assert payload["spec"]["classify.classifier"]["value"] == "page_kind"
    assert decisions.bodies == []
    assert not out.exists()


def _pages(
    capsys: pytest.CaptureFixture[str], pdf: Path, out: Path, *extra: str
) -> Any:
    code, payload = _run(
        capsys, "pages", "--input", str(pdf), "--output", str(out), *extra
    )
    assert code == 0, payload
    return payload


def test_answers_are_reused_and_republished_across_classifier_revisions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=3)
    out = tmp_path / "out"
    path = _classifier(tmp_path, "The page is a title page.")
    revision_a = load_classifier(path).sha256

    first = _pages(capsys, pdf, out, "--classifier", str(path))
    _classifier(tmp_path, "The page is the book's title page.")
    second = _pages(capsys, pdf, out, "--classifier", str(path))
    assert {row["classifier_sha256"] for row in _view(out)} != {revision_a}
    _classifier(tmp_path, "The page is a title page.")
    calls = len(decisions.bodies)
    third = _pages(capsys, pdf, out, "--classifier", str(path))

    assert [p["units_asked"] for p in (first, second, third)] == [3, 3, 0]
    assert third["units_reused"] == 3
    assert len(decisions.bodies) == calls
    assert "classify" not in third["usage"]
    rows = _view(out)
    assert len(rows) == 3
    assert {row["classifier_sha256"] for row in rows} == {revision_a}
    assert {row["reused"] for row in rows} == {1}
    assert _count(out, "classification_rows") == 9


def test_force_asks_again_instead_of_reusing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=2)
    out = tmp_path / "out"

    _pages(capsys, pdf, out, "--classifier", "layout")
    forced = _pages(capsys, pdf, out, "--classifier", "layout", "--force")

    assert (forced["units_asked"], forced["units_reused"]) == (2, 0)
    assert len(decisions.bodies) == 2
    assert {row["reused"] for row in _view(out)} == {0}


async def _seed(out: Path, name: str, sha256: str) -> None:
    async with open_store(out) as store:
        await store.begin_run("transcribe")
        await store.ensure_document(name, input_sha256=sha256, kind="pdf")


def test_an_item_whose_stored_input_changed_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=2)
    out = tmp_path / "out"
    asyncio.run(_seed(out, _document(pdf, out), "0" * 64))

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--classifier",
        "layout",
    )

    assert code == 1
    assert (payload["items_refused"], payload["items_total"]) == (1, 0)
    assert decisions.bodies == []
    assert _count(out, "classification_rows") == 0


async def _seed_pages(out: Path, name: str, labels: list[str]) -> None:
    async with open_store(out) as store:
        await store.begin_run("transcribe")
        doc_id = await store.ensure_document(name, kind="images")
        lease = await store.acquire(doc_id)
        assert lease is not None
        units = [PlannedUnit(idx, label=label) for idx, label in enumerate(labels)]
        await store.plan_units(lease, KIND_PAGE, units, model="m", provider="p")
        await store.release(lease)


@pytest.mark.parametrize(
    ("first", "refused"), [("page_1.png", True), ("page_01.png", False)]
)
def test_a_legacy_folder_whose_stored_pages_moved_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    decisions: FakeDecisions,
    first: str,
    refused: bool,
) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    for image, color in (("page_1.png", "white"), ("page_01.png", "black")):
        Image.new("RGB", (40, 60), color).save(folder / image)
    out = tmp_path / "out"
    name = item_location(
        SourceItem(folder, "images"), output_dir=out, layout="hash", input_root=folder
    )[1]
    second = "page_01.png" if first == "page_1.png" else "page_1.png"
    labels = [f"{image}_pre_processed.jpg" for image in (first, second)]
    asyncio.run(_seed_pages(out, name, labels))

    _code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(folder),
        "--type",
        "images",
        "--output",
        str(out),
        "--classifier",
        "layout",
    )

    assert payload["items_refused"] == (1 if refused else 0)
    assert _count(out, "classification_rows") == (0 if refused else 2)


def test_a_change_found_at_the_write_refuses_the_item(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    decisions: FakeDecisions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=2)
    out = tmp_path / "out"
    asyncio.run(_seed(out, _document(pdf, out), "0" * 64))
    monkeypatch.setattr(classify, "_refusal", lambda item, stores: "")

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--classifier",
        "layout",
    )

    assert code == 1
    assert (payload["items_refused"], payload["units_asked"]) == (1, 0)
    assert len(decisions.bodies) == 1
    assert _count(out, "classification_rows") == 0


def test_a_replaced_pdf_leaves_no_answer_of_its_old_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=3)
    out = tmp_path / "out"
    _pages(capsys, pdf, out, "--classifier", "layout")
    make_scanned_pdf(pdf, pages=2, scan_size=(400, 600))

    _pages(capsys, pdf, out, "--classifier", "layout")

    rows = _view(out)
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert {row["input_sha256"] for row in rows} == {file_sha256(pdf)}


def test_an_image_folder_has_an_input_identity_that_follows_its_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    folder = make_image_folder(tmp_path / "scans", pages=3)
    out = tmp_path / "out"
    args = ("--type", "images", "--classifier", "layout")
    _pages(capsys, folder, out, *args)
    first = {row["input_sha256"] for row in _view(out)}
    sorted(folder.iterdir())[-1].unlink()

    _pages(capsys, folder, out, *args)

    rows = _view(out)
    assert len(first) == 1 and None not in first
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert {row["input_sha256"] for row in rows} != first
    assert len({row["input_sha256"] for row in rows}) == 1


def test_select_numbers_the_items_as_transcribe_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    from tests.characterization.inputs_ebooks import make_epub

    folder = tmp_path / "in"
    make_scanned_pdf(folder / "a.pdf", pages=1)
    make_epub(folder / "b.epub")
    make_scanned_pdf(folder / "c.pdf", pages=1)
    out = tmp_path / "out"

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(folder),
        "--select",
        "3",
        "--output",
        str(out),
        "--classifier",
        "layout",
    )
    assert code == 0, payload
    assert {row["document"].split("-")[0] for row in _view(out)} == {"c.pdf"}

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(folder),
        "--select",
        "2",
        "--output",
        str(out),
        "--classifier",
        "layout",
    )
    assert code == 2


def test_a_stored_document_of_the_same_input_gives_the_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=1)
    out = tmp_path / "out"

    async def seed() -> None:
        async with open_store(out) as store:
            await store.begin_run("transcribe")
            await store.ensure_document(
                "elsewhere/book", input_path=pdf, input_sha256=file_sha256(pdf)
            )

    asyncio.run(seed())
    _pages(capsys, pdf, out, "--classifier", "layout")

    assert {row["document"] for row in _view(out)} == {"elsewhere/book"}


def test_a_document_with_a_page_that_fails_to_render_is_not_asked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    decisions: FakeDecisions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=3)
    out = tmp_path / "out"
    real = classify.decision_jpeg
    calls: list[int] = []

    def flaky(data: bytes, max_px: int) -> bytes:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("cannot decode")
        return real(data, max_px)

    monkeypatch.setattr(classify, "decision_jpeg", flaky)

    code, payload = _run(
        capsys,
        "documents",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--classifier",
        "document_type",
    )

    assert code == 1
    assert (payload["units_failed"], payload["items_failed"]) == (1, 1)
    assert decisions.bodies == []


def test_a_page_range_that_selects_nothing_fails_the_item(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], decisions: FakeDecisions
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=2)

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(pdf),
        "--output",
        str(tmp_path / "out"),
        "--classifier",
        "layout",
        "--pages",
        "7-9",
    )

    assert code == 1
    assert payload["items_failed"] == 1
    assert decisions.bodies == []


def test_a_failed_request_fails_the_item(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    decisions: FakeDecisions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        decisions.bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    mock = httpx.MockTransport(refuse)

    async def handle(
        transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        return await mock.handle_async_request(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=2)
    out = tmp_path / "out"

    code, payload = _run(
        capsys,
        "pages",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--classifier",
        "layout",
    )

    assert code == 1
    assert (payload["items_failed"], payload["units_failed"]) == (1, 2)
    assert _count(out, "classification_rows") == 0
