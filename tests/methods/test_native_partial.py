"""A native ``--pages`` run leaves its document partial; a later run completes it.

Run on Windows with:

    uv --directory <repo> run pytest tests/methods/test_native_partial.py -q

Each test drives the ``transcribe`` command in process (``--json``) over a
born-digital PDF or an EPUB with a title and an author. Oracle: after
``--pages`` the ``document_status`` view shows the document ``partial``
with only the selected units done and the ``.txt`` equals the selection's
text; a second run without ``--pages`` processes the item, extracts only
the units still pending, leaves it ``complete`` and writes a ``.txt``
byte-identical to one full run into a fresh output folder; ``--force``
with ``--pages`` resets every unit and leaves the document partial. A bare
number N in ``--pages`` selects the first N pages.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest
from ebooklib import epub

import chronotranscriber.methods.native as native_method
from chronotranscriber import __main__ as entry
from chronotranscriber.sources.pdf import native_extract_pdf_text

SECTIONS = ["Erstes Kapitel: Suppe.", "Zweites: Brot.", "Drittes: Kuchen."]


def _pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(3):
        page = document.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Seite {index + 1}")
    document.save(path)
    document.close()
    return path


def _epub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.EpubBook()
    book.set_identifier("ct-native-partial")
    book.set_title("Kochbuch")
    book.add_author("Anna Muster")
    chapters = []
    for number, text in enumerate(SECTIONS, 1):
        chapter = epub.EpubHtml(
            title=f"Kapitel {number}", file_name=f"chap_{number}.xhtml", lang="de"
        )
        chapter.content = f"<html><body><p>{text}</p></body></html>"
        book.add_item(chapter)
        chapters.append(chapter)
    book.toc = chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = chapters
    epub.write_epub(str(path), book)
    return path


def _run(
    capsys: pytest.CaptureFixture[str], item: Path, kind: str, output: Path, *flags: str
) -> dict[str, Any]:
    argv = [
        "transcribe",
        "--input",
        str(item),
        "--type",
        kind,
        "--method",
        "native",
        "--output",
        str(output),
        "--json",
        *flags,
    ]
    assert entry.main(argv) == 0
    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    payload: dict[str, Any] = json.loads(lines[-1])
    return payload


def _txt(payload: dict[str, Any]) -> bytes:
    (path,) = [p for p in payload["outputs"] if p.endswith(".txt")]
    return Path(path).read_bytes()


def _status(output: Path) -> tuple[str, int, int, int]:
    with sqlite3.connect(output / "chronotranscriber.sqlite") as connection:
        (row,) = connection.execute(
            "SELECT status, units_total, units_done, units_pending FROM document_status"
        ).fetchall()
    return str(row[0]), int(row[1]), int(row[2]), int(row[3])


def _done(output: Path) -> list[int]:
    with sqlite3.connect(output / "chronotranscriber.sqlite") as connection:
        rows = connection.execute(
            "SELECT order_index FROM page_text WHERE status = 'done' "
            "ORDER BY order_index"
        ).fetchall()
    return [int(row[0]) for row in rows]


@pytest.fixture
def extracted(monkeypatch: pytest.MonkeyPatch) -> list[list[int]]:
    """Record the page lists the native method extracts."""
    calls: list[list[int]] = []
    original = native_method.native_extract_pdf_pages

    def record(path: Path, indices: list[int]) -> dict[int, str]:
        calls.append(list(indices))
        return original(path, indices)

    monkeypatch.setattr(native_method, "native_extract_pdf_pages", record)
    return calls


def test_a_pdf_page_run_is_partial_and_a_full_run_completes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extracted: list[list[int]]
) -> None:
    pdf = _pdf(tmp_path / "inbox" / "book.pdf")
    out = tmp_path / "out"

    first = _run(capsys, pdf, "pdf", out, "--pages", "1")

    assert _status(out) == ("partial", 3, 1, 2)
    assert _txt(first) == (native_extract_pdf_text(pdf, [0]).strip() + "\n").encode()

    second = _run(capsys, pdf, "pdf", out)

    assert (second["items_complete"], second["items_skipped"]) == (1, 0)
    assert extracted == [[0], [1, 2]]
    assert _status(out) == ("complete", 3, 3, 0)
    fresh = _run(capsys, pdf, "pdf", tmp_path / "fresh")
    assert _txt(second) == _txt(fresh)
    assert b"Seite 3" in _txt(second)


def test_an_epub_section_run_is_partial_and_a_full_run_completes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book = _epub(tmp_path / "inbox" / "kochbuch.epub")
    out = tmp_path / "out"
    extraction = native_method.EPUBProcessor(book).extract_text()
    total = len(extraction.sections) + 1
    header = "# Title: Kochbuch\n# Author(s): Anna Muster"

    first = _run(capsys, book, "epub", out, "--pages", "2-3")

    assert _status(out) == ("partial", total, 3, total - 3)
    assert _done(out) == [0, 2, 3]
    partial_text = _txt(first).decode("utf-8")
    assert partial_text.startswith(header)
    assert "Suppe" not in partial_text
    assert "Brot" in partial_text

    second = _run(capsys, book, "epub", out)

    assert (second["items_complete"], second["items_skipped"]) == (1, 0)
    assert _status(out) == ("complete", total, total, 0)
    fresh = _run(capsys, book, "epub", tmp_path / "fresh")
    assert _txt(second) == _txt(fresh)
    assert _txt(second).decode("utf-8").startswith(header)
    assert _txt(second) == (extraction.to_plain_text().strip() + "\n").encode()


def test_a_range_beyond_the_last_page_skips_the_item_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extracted: list[list[int]]
) -> None:
    pdf = _pdf(tmp_path / "inbox" / "book.pdf")
    out = tmp_path / "out"
    argv = ["transcribe", "--input", str(pdf), "--type", "pdf", "--method"]
    argv += ["native", "--output", str(out), "--json", "--pages", "9-"]
    _run(capsys, pdf, "pdf", out, "--pages", "1")

    for state in ("partial", "complete"):
        assert entry.main(argv) == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out.splitlines()[-1])

        counts = ("items_total", "items_complete", "items_skipped")
        assert [payload[key] for key in counts] == [0, 0, 1], state
        assert not [p for p in payload["outputs"] if p.endswith(".txt")], state
        assert "Page range 'pages 9-'" in captured.err, state
        assert _status(out)[0] == state
        _run(capsys, pdf, "pdf", out)

    assert extracted == [[0], [1, 2]]


def test_force_with_pages_resets_every_unit_and_leaves_the_document_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = _pdf(tmp_path / "inbox" / "book.pdf")
    out = tmp_path / "out"
    _run(capsys, pdf, "pdf", out)
    assert _status(out) == ("complete", 3, 3, 0)

    forced = _run(capsys, pdf, "pdf", out, "--force", "--pages", "2")

    assert _status(out) == ("partial", 3, 2, 1)
    assert _done(out) == [0, 1]
    expected = native_extract_pdf_text(pdf, [0, 1]).strip() + "\n"
    assert _txt(forced) == expected.encode()
