"""Forced replacements, legacy folders and repair under the input identity.

Native PDFs and Tesseract folders run without a model; the repair cases
reuse the scripted first run of ``test_repair`` (fake LLM). The oracle is
the store: the ``documents`` identity, the units, the ``page_types`` rows and
the notes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from chronotranscriber import run as core
from chronotranscriber.batch import repair as repair_module
from chronotranscriber.common.store import Store, file_sha256
from chronotranscriber.output import store as store_module
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_ocr import (
    FakeTesseract,
    block_real_tesseract,
    tesseract_cmd_settings,
)
from tests.characterization.inputs import (
    make_born_digital_pdf,
    make_scanned_pdf,
    page_text,
)
from tests.characterization.runner import RunResult, run_cli
from tests.characterization.test_input_identity import CHANGED
from tests.characterization.test_repair import (  # noqa: F401
    PAGES,
    _empty_project_context,
    first_run,
    repair,
)


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


def _query(root: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(root / "out" / "chronotranscriber.sqlite")
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _execute(root: Path, *sql: str) -> None:
    connection = sqlite3.connect(root / "out" / "chronotranscriber.sqlite")
    try:
        for statement in sql:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def _native(transcribe: Callable[..., RunResult], pdf: Path, **options: Any) -> Any:
    return transcribe(pdf, type="pdf", method="native", **options)


def test_force_selecting_no_page_keeps_the_identity_and_units(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    assert _native(transcribe, pdf).exit_code == 0
    before = _query(tmp_path, "SELECT input_sha256 FROM documents")
    units = _query(tmp_path, "SELECT idx, status, content FROM units ORDER BY idx")
    make_born_digital_pdf(pdf, 3, texts=CHANGED)

    _native(transcribe, pdf, force=True, pages="7-9")

    assert _query(tmp_path, "SELECT input_sha256 FROM documents") == before
    assert _query(tmp_path, "SELECT idx, status, content FROM units ORDER BY idx") == (
        units
    )
    assert _native(transcribe, pdf).exit_code == 1


def test_a_forced_replacement_drops_the_old_page_types_and_extra_units(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    assert _native(transcribe, pdf).exit_code == 0
    _execute(
        tmp_path,
        "INSERT INTO page_types (doc_id, idx, page_type, probabilities, taxonomy, "
        "profile, model, fingerprint, created_at) SELECT doc_id, 0, 'blank', '{}', "
        "'1', 'general', 'm', 'fp', 'now' FROM documents",
    )
    make_born_digital_pdf(pdf, 2, texts=CHANGED[:2])

    forced = _native(transcribe, pdf, force=True)

    assert forced.exit_code == 0, forced.stderr
    assert _query(tmp_path, "SELECT count(*) FROM page_types") == [(0,)]
    assert _query(tmp_path, "SELECT idx, status FROM units ORDER BY idx") == [
        (0, "done"),
        (1, "done"),
    ]
    assert _query(tmp_path, "SELECT input_sha256, status FROM documents") == [
        (file_sha256(pdf), "complete")
    ]


@pytest.mark.parametrize(("pages", "changed"), [(3, 2), (2, 3)])
def test_an_interrupted_forced_replacement_keeps_the_old_state(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    monkeypatch: pytest.MonkeyPatch,
    pages: int,
    changed: int,
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", pages)
    assert _native(transcribe, pdf).exit_code == 0
    _execute(
        tmp_path,
        "INSERT INTO page_types (doc_id, idx, page_type, probabilities, taxonomy, "
        "profile, model, fingerprint, created_at) SELECT doc_id, 0, 'blank', '{}', "
        "'1', 'general', 'm', 'fp', 'now' FROM documents",
    )
    documents = "SELECT input_sha256, status, units_total FROM documents"
    units = "SELECT idx, status, content, run_id FROM units ORDER BY idx"
    outputs = "SELECT path, stale FROM outputs"
    before = [_query(tmp_path, sql) for sql in (documents, units, outputs)]
    make_born_digital_pdf(pdf, changed, texts=CHANGED[:changed])

    def interrupted(connection: sqlite3.Connection, doc_id: int | None = None) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(store_module, "refresh_documents", interrupted)
    forced = _native(transcribe, pdf, force=True)

    assert forced.exit_code == 1
    assert [_query(tmp_path, sql) for sql in (documents, units, outputs)] == before
    assert _query(tmp_path, "SELECT count(*) FROM page_types") == [(1,)]


def test_a_forced_replacement_refuses_a_submission_holding_a_kept_unit(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    assert _native(transcribe, pdf).exit_code == 0
    units = "SELECT idx, status, content, run_id FROM units ORDER BY idx"
    before = _query(tmp_path, units)
    settle = core._ItemRun._settle_identity

    async def submit_then_settle(self: core._ItemRun, *args: Any) -> str | None:
        _execute(
            tmp_path,
            "INSERT INTO submissions (submission_id, tool, provider, model, run_id, "
            "intent_at, status) SELECT 'sub-1', 'chronotranscriber', 'openai', 'm', "
            "run_id, 'now', 'submitting' FROM runs LIMIT 1",
            "INSERT INTO submission_requests (submission_id, custom_id, doc_id, "
            "kind, idx) SELECT 'sub-1', 'c-0', doc_id, 'text', 0 FROM documents",
        )
        return await settle(self, *args)

    monkeypatch.setattr(core._ItemRun, "_settle_identity", submit_then_settle)
    forced = _native(transcribe, pdf, force=True)

    assert forced.exit_code == 1
    assert "Submission sub-1 still holds units" in forced.stderr
    assert _query(tmp_path, units) == before


def test_a_forced_replacement_marks_the_unrendered_formats_stale(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    both = _native(transcribe, pdf, formats=("txt", "md"))
    assert both.exit_code == 0, both.stderr
    markdown = both.output(pdf).with_suffix(".md")
    old = markdown.read_text(encoding="utf-8")
    make_born_digital_pdf(pdf, 3, texts=CHANGED)

    assert _native(transcribe, pdf, force=True, formats="txt").exit_code == 0
    assert markdown.read_text(encoding="utf-8") == old
    rendered = _native(transcribe, pdf, formats="md")

    assert rendered.exit_code == 0, rendered.stderr
    assert CHANGED[0] in markdown.read_text(encoding="utf-8")
    assert _query(tmp_path, "SELECT stale FROM outputs") == [(0,), (0,)]


def test_a_format_rendered_before_later_pages_is_rendered_again(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    first = _native(transcribe, pdf, pages="1", formats="txt")
    assert first.exit_code == 0, first.stderr
    text = first.output(pdf)
    assert page_text(2) not in text.read_text(encoding="utf-8")
    assert _native(transcribe, pdf, formats="md").exit_code == 0

    again = _native(transcribe, pdf, formats="txt")

    assert again.exit_code == 0, again.stderr
    assert page_text(2) in text.read_text(encoding="utf-8")
    assert _query(tmp_path, "SELECT stale FROM outputs") == [(0,), (0,)]


def _as_partial(root: Path) -> None:
    _execute(
        root,
        "UPDATE documents SET status = 'partial'",
        "UPDATE units SET status = 'pending', outcome = NULL, content = NULL "
        "WHERE idx = 1",
    )


def test_an_identity_changed_before_the_lease_is_kept_and_refuses(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    assert _native(transcribe, pdf).exit_code == 0
    _as_partial(tmp_path)
    other = "0" * 64
    ensure = Store.ensure_document

    async def other_run_first(self: Store, name: str, **options: Any) -> int:
        _execute(tmp_path, f"UPDATE documents SET input_sha256 = '{other}'")
        return await ensure(self, name, **options)

    monkeypatch.setattr(Store, "ensure_document", other_run_first)
    result = _native(transcribe, pdf)

    assert result.exit_code == 1
    assert "Input changed for 'book-" in result.stderr
    assert _query(tmp_path, "SELECT input_sha256 FROM documents") == [(other,)]
    assert _query(tmp_path, "SELECT status FROM units WHERE idx = 1") == [("pending",)]


def test_a_render_only_item_changed_under_its_lease_fails(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "docs" / "book.pdf", 3)
    first = _native(transcribe, pdf)
    assert first.exit_code == 0
    first.output(pdf).unlink()
    settle = core._ItemRun._settle_identity

    async def change_then_settle(self: core._ItemRun, *args: Any) -> str | None:
        make_born_digital_pdf(pdf, 2, texts=CHANGED[:2])
        return await settle(self, *args)

    monkeypatch.setattr(core._ItemRun, "_settle_identity", change_then_settle)
    result = _native(transcribe, pdf)

    assert result.exit_code == 1
    assert result.summary is not None and result.summary["items_failed"] == 1
    assert "during the run" in result.stderr
    assert not first.output(pdf).exists()


def _tied_folder(root: Path) -> Path:
    folder = root / "scans"
    folder.mkdir()
    Image.new("RGB", (40, 60), "white").save(folder / "page_1.png")
    Image.new("RGB", (40, 60), "black").save(folder / "page_01.png")
    return folder


def _as_legacy_listing(root: Path) -> None:
    """Record no identity and the page order the file system once gave."""
    _execute(
        root,
        "UPDATE documents SET input_sha256 = NULL",
        "UPDATE units SET label = CASE idx WHEN 0 THEN 'page_1_tess_preprocessed.png' "
        "ELSE 'page_01_tess_preprocessed.png' END",
    )


def test_a_legacy_folder_whose_page_order_changed_is_refused(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    folder = _tied_folder(tmp_path)
    assert transcribe(folder, type="images", method="tesseract").exit_code == 0
    _as_legacy_listing(tmp_path)

    rerun = transcribe(folder, type="images", method="tesseract")

    assert rerun.exit_code == 1
    assert "The page order of 'scans-" in rerun.stderr
    assert _query(tmp_path, "SELECT input_sha256 FROM documents") == [(None,)]


def test_a_legacy_folder_in_page_order_is_trusted(
    tmp_path: Path, transcribe: Callable[..., RunResult]
) -> None:
    folder = _tied_folder(tmp_path)
    assert transcribe(folder, type="images", method="tesseract").exit_code == 0
    _execute(tmp_path, "UPDATE documents SET input_sha256 = NULL, status = 'partial'")
    _execute(tmp_path, "UPDATE units SET status = 'pending' WHERE idx = 1")

    rerun = transcribe(folder, type="images", method="tesseract")

    assert rerun.exit_code == 0, rerun.stderr
    assert _query(tmp_path, "SELECT input_sha256 FROM documents") != [(None,)]


@pytest.mark.parametrize("change", ["added", "removed"])
def test_a_legacy_folder_with_other_images_is_refused(
    tmp_path: Path, transcribe: Callable[..., RunResult], change: str
) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    for number, color in ((1, "white"), (2, "black")):
        Image.new("RGB", (40, 60), color).save(folder / f"page_{number}.png")
    assert transcribe(folder, type="images", method="tesseract").exit_code == 0
    _execute(tmp_path, "UPDATE documents SET input_sha256 = NULL")
    if change == "added":
        Image.new("RGB", (40, 60), "gray").save(folder / "page_3.png")
    else:
        (folder / "page_2.png").unlink()

    rerun = transcribe(folder, type="images", method="tesseract")

    assert rerun.exit_code == 1
    assert "The images of 'scans-" in rerun.stderr
    assert _query(tmp_path, "SELECT input_sha256 FROM documents") == [(None,)]


@pytest.mark.parametrize(
    ("label", "legacy", "refused"),
    [
        ("page_1_tess_preprocessed.png", False, True),
        ("page_1_tess_preprocessed.png", True, True),
        ("scan 1", False, False),
        ("scan 1", True, False),
    ],
)
def test_a_folder_page_stored_under_another_image_is_refused(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    label: str,
    legacy: bool,
    refused: bool,
) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    Image.new("RGB", (40, 60), "white").save(folder / "a.png")
    Image.new("RGB", (40, 60), "black").save(folder / "page_1.png")
    assert transcribe(folder, type="images", method="tesseract").exit_code == 0
    _as_partial(tmp_path)
    _execute(tmp_path, f"UPDATE units SET label = '{label}' WHERE idx = 0")
    if legacy:
        _execute(tmp_path, "UPDATE documents SET input_sha256 = NULL")

    rerun = transcribe(folder, type="images", method="tesseract")

    assert rerun.exit_code == (1 if refused else 0), rerun.stderr
    assert ("The page order of 'scans-" in rerun.stderr) is refused
    if refused:
        assert "stored page 1 does not name a.png" in rerun.stderr
    else:
        again = transcribe(folder, type="images", method="tesseract")
        assert again.exit_code == 0, again.stderr


@pytest.mark.parametrize("legacy", [True, False])
def test_repair_refuses_a_folder_whose_page_order_changed(
    tmp_path: Path,
    transcribe: Callable[..., RunResult],
    fake_llm: FakeLLM,
    legacy: bool,
) -> None:
    folder = _tied_folder(tmp_path)
    assert transcribe(folder, type="images", method="tesseract").exit_code == 0
    ((identity,),) = _query(tmp_path, "SELECT input_sha256 FROM documents")
    _as_legacy_listing(tmp_path)
    if not legacy:
        _execute(tmp_path, f"UPDATE documents SET input_sha256 = '{identity}'")
    _execute(
        tmp_path,
        "UPDATE units SET status = 'failed', outcome = 'failed', "
        "error_type = 'api_error' WHERE idx = 0",
    )

    result = run_cli("repair", llm=fake_llm, root=tmp_path, input=tmp_path / "out")

    assert result.exit_code == 1
    assert "The page order of 'scans-" in result.stderr


def test_repair_records_a_legacy_identity_on_first_use(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    first = first_run(fake_llm, tmp_path)
    _execute(tmp_path, "UPDATE documents SET input_sha256 = NULL")

    repaired = repair(fake_llm, tmp_path, first)

    assert repaired.result.exit_code == 0, repaired.result.stderr
    pdf = tmp_path / "docs" / "doc.pdf"
    assert _query(tmp_path, "SELECT input_sha256 FROM documents") == [
        (file_sha256(pdf),)
    ]
    notes = [row[0] for row in _query(tmp_path, "SELECT message FROM notes")]
    assert any("for the first time" in note for note in notes)


def test_repair_refuses_an_unreadable_legacy_input(
    tmp_path: Path, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = first_run(fake_llm, tmp_path)
    _execute(tmp_path, "UPDATE documents SET input_sha256 = NULL")

    def unreadable(path: Path) -> str:
        raise PermissionError(f"locked: {path.name}")

    monkeypatch.setattr(repair_module, "input_identity", unreadable)
    repaired = repair(fake_llm, tmp_path, first)

    assert repaired.result.exit_code == 1
    assert "Cannot compute the input identity" in repaired.result.stderr
    assert _query(tmp_path, "SELECT input_sha256 FROM documents") == [(None,)]


def test_repair_refuses_an_input_changed_after_planning(
    tmp_path: Path, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = first_run(fake_llm, tmp_path)
    failed = _query(tmp_path, "SELECT idx FROM units WHERE status = 'failed'")
    checked = repair_module._leased_refusal

    def change_then_check(store: Any, plan: Any) -> str | None:
        make_scanned_pdf(tmp_path / "docs" / "doc.pdf", PAGES, scan_size=(400, 600))
        return checked(store, plan)

    monkeypatch.setattr(repair_module, "_leased_refusal", change_then_check)
    repaired = repair(fake_llm, tmp_path, first)

    assert repaired.result.exit_code == 1
    assert "during the repair" in repaired.result.stderr
    assert repaired.result.calls == []
    assert _query(tmp_path, "SELECT idx FROM units WHERE status = 'failed'") == failed
