"""Input identity: a changed input refuses resume, repair follows, folders count.

PDFs run through the native method and image folders through Tesseract
(``fakes_ocr.FakeTesseract``), so no model is called. The oracle is the
JSON line, the ``documents`` row and the ``notes`` of the store.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from chronotranscriber import __main__ as entry
from chronotranscriber import run as core
from chronotranscriber.common.store import PlannedUnit, UnitResult, file_sha256
from chronotranscriber.output.store import open_store
from chronotranscriber.sources.identity import input_identity
from chronotranscriber.sources.images import list_folder_images
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_ocr import (
    FakeTesseract,
    block_real_tesseract,
    tesseract_cmd_settings,
)
from tests.characterization.inputs import make_born_digital_pdf, make_image_folder
from tests.characterization.runner import RunResult, run_cli
from tests.cli.test_classify_command import FakeDecisions, decisions  # noqa: F401

CHANGED = ["Changed page one.", "Changed page two.", "Changed page three."]


@pytest.fixture
def transcribe(
    tmp_path: Path, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., RunResult]:
    """Return ``transcribe(input, **options)`` writing to ``tmp_path / out``."""
    FakeTesseract().install(monkeypatch)
    settings = tesseract_cmd_settings(block_real_tesseract(monkeypatch, tmp_path))

    def run(source: Path, **options: Any) -> RunResult:
        return run_cli(
            "transcribe",
            llm=fake_llm,
            root=tmp_path,
            settings=settings,
            input=source,
            output=tmp_path / "out",
            **options,
        )

    return run


def _document(tmp_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(tmp_path / "out" / "chronotranscriber.sqlite")
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute("SELECT * FROM documents")]
        notes = [row[0] for row in connection.execute("SELECT message FROM notes")]
    finally:
        connection.close()
    (row,) = rows
    row["notes"] = notes
    return row


def _native(transcribe: Callable[..., RunResult], pdf: Path, **options: Any) -> Any:
    return transcribe(pdf, type="pdf", method="native", **options)


@pytest.mark.parametrize("pages", ["1", None], ids=["partial", "complete"])
def test_a_changed_pdf_is_refused_until_force(
    tmp_path: Path, transcribe: Callable[..., RunResult], pages: str | None
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    first = _native(transcribe, pdf, pages=pages)
    assert first.exit_code == 0, first.stderr
    stored = _document(tmp_path)["input_sha256"]
    make_born_digital_pdf(pdf, 3, texts=CHANGED)

    refused = _native(transcribe, pdf)
    assert refused.exit_code == 1
    assert refused.summary is not None and refused.summary["items_failed"] == 1
    assert "Input changed for 'book-" in refused.stderr
    assert "since it was transcribed" in refused.stderr
    assert _document(tmp_path)["input_sha256"] == stored

    forced = _native(transcribe, pdf, force=True)
    assert forced.exit_code == 0, forced.stderr
    document = _document(tmp_path)
    assert document["input_sha256"] == file_sha256(pdf) != stored
    assert document["status"] == "complete"
    text = forced.output(pdf).read_text(encoding="utf-8")
    assert all(line in text for line in CHANGED)


def test_an_unchanged_rerun_resumes(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    assert _native(transcribe, pdf, pages="1").exit_code == 0

    rerun = _native(transcribe, pdf)

    assert rerun.exit_code == 0, rerun.stderr
    assert _document(tmp_path)["status"] == "complete"


def _remove(folder: Path) -> None:
    list_folder_images(folder)[-1].unlink()


def _add(folder: Path) -> None:
    Image.new("RGB", (40, 60), "white").save(folder / "page_009.png")


def _edit(folder: Path) -> None:
    Image.new("RGB", (40, 60), "black").save(list_folder_images(folder)[0])


@pytest.mark.parametrize("change", [_add, _remove, _edit])
def test_a_changed_image_folder_is_refused(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    change: Callable[[Path], None],
) -> None:
    folder = make_image_folder(tmp_path / "scans", 3)
    first = transcribe(folder, type="images", method="tesseract")
    assert first.exit_code == 0, first.stderr
    assert _document(tmp_path)["input_sha256"] == input_identity(folder)
    change(folder)

    rerun = transcribe(folder, type="images", method="tesseract")

    assert rerun.exit_code == 1
    assert "Input changed for 'scans-" in rerun.stderr


def test_a_legacy_document_records_its_identity_on_first_use(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    assert _native(transcribe, pdf, pages="1").exit_code == 0
    database = tmp_path / "out" / "chronotranscriber.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE documents SET input_sha256 = NULL")
    connection.close()

    rerun = _native(transcribe, pdf)

    assert rerun.exit_code == 0, rerun.stderr
    document = _document(tmp_path)
    assert document["input_sha256"] == file_sha256(pdf)
    assert any("for the first time" in note for note in document["notes"])


def test_an_input_that_cannot_be_hashed_is_refused(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)

    def unreadable(path: Path) -> str:
        raise PermissionError(f"locked: {path.name}")

    monkeypatch.setattr(core, "input_identity", unreadable)
    result = _native(transcribe, pdf, force=True)

    assert result.exit_code == 1
    assert "Cannot compute the input identity of 'book-" in result.stderr
    assert not (tmp_path / "out" / "chronotranscriber.sqlite").exists()


def test_an_input_changed_during_the_run_is_refused(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    lease = core._ItemRun.lease

    async def change_then_lease(self: core._ItemRun) -> Any:
        make_born_digital_pdf(pdf, 2, texts=CHANGED[:2])
        return await lease(self)

    monkeypatch.setattr(core._ItemRun, "lease", change_then_lease)
    result = _native(transcribe, pdf)

    assert result.exit_code == 1
    assert "during the run" in result.stderr
    connection = sqlite3.connect(tmp_path / "out" / "chronotranscriber.sqlite")
    try:
        assert connection.execute("SELECT count(*) FROM documents").fetchone() == (0,)
    finally:
        connection.close()


def _listing_reversed(monkeypatch: pytest.MonkeyPatch) -> None:
    real = Path.iterdir

    def iterdir(self: Path) -> Iterator[Path]:
        return iter(sorted(real(self), reverse=True))

    monkeypatch.setattr(Path, "iterdir", iterdir)


def test_page_order_and_identity_do_not_depend_on_the_listing_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = make_image_folder(tmp_path / "scans", 4)
    for name in ("page_1.png", "page_01.png"):
        Image.new("RGB", (30, 40), "gray").save(folder / name)
    order = [path.name for path in list_folder_images(folder)]
    identity = input_identity(folder)

    _listing_reversed(monkeypatch)

    assert [path.name for path in list_folder_images(folder)] == order
    assert input_identity(folder) == identity
    assert order.index("page_01.png") + 1 == order.index("page_1.png")


def test_repair_refuses_a_changed_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 2)
    out = tmp_path / "out"

    async def seed() -> None:
        async with open_store(out) as store:
            await store.begin_run("transcribe")
            doc_id = await store.ensure_document(
                "book", input_path=pdf, input_sha256="0" * 64, kind="pdf"
            )
            lease = await store.acquire(doc_id)
            assert lease is not None
            await store.plan_units(
                lease,
                "page",
                [PlannedUnit(0, "page_0001_pre_processed.jpg", 0)],
                model="gpt-6-luna",
                provider="openai",
            )
            await store.write_unit(
                lease,
                UnitResult(
                    kind="page",
                    idx=0,
                    status="failed",
                    model="gpt-6-luna",
                    provider="openai",
                    content="[transcription error]",
                    error="Error code: 500",
                    error_type="api_error",
                ),
            )

    asyncio.run(seed())
    code = entry.main(["repair", "--input", str(out), "--json"])
    captured = capsys.readouterr()

    assert code == 1
    assert '"failed": 1' in captured.out
    assert "Input changed for 'book'" in captured.err


def test_classify_and_transcription_record_the_same_folder_identity(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    decisions: FakeDecisions,  # noqa: F811
) -> None:
    folder = make_image_folder(tmp_path / "scans", 3)
    assert transcribe(folder, type="images", method="tesseract").exit_code == 0
    out = tmp_path / "out"

    code = entry.main(
        [
            "classify",
            "pages",
            "--input",
            str(folder),
            "--type",
            "images",
            "--output",
            str(out),
            "--classifier",
            "layout",
        ]
    )

    assert code == 0
    connection = sqlite3.connect(out / "chronotranscriber.sqlite")
    try:
        stored = connection.execute("SELECT input_sha256 FROM documents").fetchall()
        view = connection.execute(
            "SELECT DISTINCT input_sha256 FROM classifications"
        ).fetchall()
    finally:
        connection.close()
    assert stored == view == [(input_identity(folder),)]
