"""Characterization of every document source with its local methods.

PDFs (native and Tesseract), image folders (Tesseract), EPUB and MOBI
(native), and auto mode on a mixed folder. Tesseract runs through
``fakes_ocr.FakeTesseract``; OpenCV preprocessing runs for real. Each case
runs twice in fresh roots, both snapshots must match, and the run is then
compared with ``golden/sources/<case>/``. Cases with OCR calls add
``ocr.json`` (one record per call: file name, image size, config string).
The MOBI input is a real Mobipocket file built by ``inputs_ebooks``, which
the ``mobi`` library unpacks to HTML.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import WriteGuard
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_ocr import (
    FakeTesseract,
    block_real_tesseract,
    ocr_text,
    tesseract_cmd_settings,
)
from tests.characterization.golden import check_files, compare, snapshot
from tests.characterization.inputs import (
    make_born_digital_pdf,
    make_image_folder,
    make_scanned_pdf,
    page_text,
)
from tests.characterization.inputs_ebooks import (
    BOOK_AUTHOR,
    CHAPTERS,
    MOBI_PARAGRAPHS,
    make_epub,
    make_mobi,
)
from tests.characterization.runner import RunResult, run_cli

pytestmark = pytest.mark.integration

PAGES = 3
CASE_PREFIX = "sources"

Build = Callable[[Path], RunResult]


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture(autouse=True)
def missing_tesseract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point pytesseract at a missing executable for every case."""
    return block_real_tesseract(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> FakeTesseract:
    """Install the fake Tesseract engine."""
    return FakeTesseract().install(monkeypatch)


@pytest.fixture
def source_case(
    fake_ocr: FakeTesseract,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> Callable[[str, Build], RunResult]:
    """Return ``run(case, build)``: run twice, compare, check the golden files."""

    def files_of(result: RunResult) -> dict[str, str]:
        files = snapshot(result)
        calls = fake_ocr.sorted_calls()
        if calls or fake_ocr.version_checks:
            files["ocr.json"] = _dumps(
                {"version_checks": fake_ocr.version_checks, "calls": calls}
            )
        return files

    def run(case: str, build: Build) -> RunResult:
        fake_ocr.reset()
        first = files_of(build(case_root("run1")))
        fake_ocr.reset()
        result = build(case_root("run2"))
        actual = files_of(result)
        problems = compare(first, actual)
        assert not problems, "two identical runs differ:\n" + "\n".join(problems)
        check_files(f"{CASE_PREFIX}/{case}", actual, on_write=write_guard.consume)
        return result

    return run


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _transcribe(
    llm: FakeLLM,
    root: Path,
    missing: Path,
    *,
    input: Path,
    type: str,
    **options: Any,
) -> RunResult:
    return run_cli(
        "transcribe",
        llm=llm,
        root=root,
        settings=tesseract_cmd_settings(missing),
        input=input,
        output=root / "out",
        type=type,
        **options,
    )


def _text(result: RunResult, item: Path) -> str:
    return result.output(item).read_text(encoding="utf-8")


# ============================================================================
# PDF
# ============================================================================
@pytest.mark.parametrize(
    ("case", "output_format"),
    [
        ("pdf_native_pages", None),
        ("pdf_native_pages_md", "md"),
    ],
)
def test_pdf_native_born_digital_pages(
    case: str,
    output_format: str | None,
    fake_llm: FakeLLM,
    fake_ocr: FakeTesseract,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        pdf = make_born_digital_pdf(root / "docs" / "born.pdf", PAGES)
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=pdf,
            type="pdf",
            method="native",
            pages="1,3",
            formats=output_format,
        )

    result = source_case(case, build)

    assert result.exit_code == 0
    assert result.calls == []
    assert fake_ocr.calls == []
    output = result.output(result.root / "docs" / "born.pdf")
    assert output in result.created
    assert output.suffix == f".{output_format or 'txt'}"
    text = output.read_text(encoding="utf-8")
    assert page_text(0) in text
    assert page_text(1) not in text
    assert page_text(2) in text


def test_pdf_native_scanned(
    fake_llm: FakeLLM,
    fake_ocr: FakeTesseract,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        pdf = make_scanned_pdf(root / "docs" / "scan.pdf", PAGES)
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=pdf,
            type="pdf",
            method="native",
        )

    result = source_case("pdf_native_scanned", build)

    assert result.exit_code == 0
    assert "not searchable" in result.stderr
    assert [call["page"] for call in fake_ocr.sorted_calls()] == [1, 2, 3]
    text = _text(result, result.root / "docs" / "scan.pdf")
    assert all(ocr_text(page) in text for page in (1, 2, 3))


def test_pdf_tesseract(
    fake_llm: FakeLLM,
    fake_ocr: FakeTesseract,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        pdf = make_scanned_pdf(root / "docs" / "scan.pdf", PAGES)
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=pdf,
            type="pdf",
            method="tesseract",
        )

    result = source_case("pdf_tesseract", build)

    assert result.exit_code == 0
    assert result.calls == []
    calls = fake_ocr.sorted_calls()
    assert [call["page"] for call in calls] == [1, 2, 3]
    assert {call["config"] for call in calls} == {"--oem 3 --psm 6"}
    text = _text(result, result.root / "docs" / "scan.pdf")
    assert all(ocr_text(page) in text for page in (1, 2, 3))
    assert not any("preprocessed" in path.name for path in result.created)


# ============================================================================
# Image folder
# ============================================================================
@pytest.mark.parametrize(
    ("case", "keep"),
    [
        ("images_tesseract", False),
        ("images_tesseract_keep_preprocessed", True),
    ],
)
def test_image_folder_tesseract(
    case: str,
    keep: bool,
    fake_llm: FakeLLM,
    fake_ocr: FakeTesseract,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        folder = make_image_folder(root / "scans", PAGES)
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=folder,
            type="images",
            method="tesseract",
            keep_working_files=keep,
        )

    result = source_case(case, build)

    assert result.exit_code == 0
    assert result.calls == []
    calls = fake_ocr.sorted_calls()
    assert [call["page"] for call in calls] == [1, 2, 3]
    text = _text(result, result.root / "scans")
    assert all(ocr_text(page) in text for page in (1, 2, 3))
    kept = [path for path in result.created if "_tess_preprocessed" in path.name]
    assert len(kept) == (PAGES if keep else 0)


# ============================================================================
# EPUB and MOBI
# ============================================================================
@pytest.mark.parametrize(
    ("case", "pages", "chapters"),
    [
        ("epub_native", None, (0, 1, 2)),
        ("epub_native_pages", "2-3", (1, 2)),
    ],
)
def test_epub_native(
    case: str,
    pages: str | None,
    chapters: tuple[int, ...],
    fake_llm: FakeLLM,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        book = make_epub(root / "books" / "annals.epub")
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=book,
            type="epub",
            method="native",
            pages=pages,
        )

    result = source_case(case, build)

    assert result.exit_code == 0
    text = _text(result, result.root / "books" / "annals.epub")
    assert BOOK_AUTHOR in text
    for index, (_heading, body) in enumerate(CHAPTERS):
        assert (body in text) == (index in chapters)


def test_mobi_native(
    fake_llm: FakeLLM,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        book = make_mobi(root / "books" / "notes.mobi")
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=book,
            type="mobi",
            method="native",
        )

    result = source_case("mobi_native", build)

    assert result.exit_code == 0
    text = " ".join(_text(result, result.root / "books" / "notes.mobi").split())
    assert all(paragraph in text for paragraph in MOBI_PARAGRAPHS)


# ============================================================================
# Auto mode
# ============================================================================
def _mixed_folder(root: Path) -> Path:
    folder = root / "mixed"
    make_born_digital_pdf(folder / "born.pdf", PAGES)
    make_scanned_pdf(folder / "scan.pdf", PAGES)
    make_image_folder(folder / "scans", PAGES)
    make_epub(folder / "annals.epub")
    make_mobi(folder / "notes.mobi")
    return folder


LOCAL_AUTO = {"method": "tesseract", "searchable_pdfs": "text"}
"""Tesseract for scanned PDFs and image folders, the text layer of the others."""


@pytest.mark.parametrize(
    ("case", "options", "llm_pages", "ocr_pages"),
    [
        ("auto_mixed_defaults", {}, 3 * PAGES, 0),
        ("auto_mixed_local_methods", LOCAL_AUTO, 0, 2 * PAGES),
    ],
)
def test_auto_mixed_folder(
    case: str,
    options: Mapping[str, Any],
    llm_pages: int,
    ocr_pages: int,
    fake_llm: FakeLLM,
    fake_ocr: FakeTesseract,
    missing_tesseract: Path,
    source_case: Callable[[str, Build], RunResult],
) -> None:
    def build(root: Path) -> RunResult:
        return _transcribe(
            fake_llm,
            root,
            missing_tesseract,
            input=_mixed_folder(root),
            type="auto",
            **options,
        )

    result = source_case(case, build)

    assert result.exit_code == 0
    assert result.summary is not None
    assert result.summary["items_complete"] == 5
    assert len(result.calls) == llm_pages
    assert len(fake_ocr.calls) == ocr_pages
    mixed = result.root / "mixed"
    for name in ("born.pdf", "scan.pdf", "scans", "annals.epub", "notes.mobi"):
        assert result.output(mixed / name) in result.created
