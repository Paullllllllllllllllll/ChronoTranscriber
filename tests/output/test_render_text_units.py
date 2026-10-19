"""The native ``text`` units of a document render as one text.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_render_text_units.py -q

Byte oracle: for a PDF page selection the stored page units render exactly
as one unit holding ``native_extract_pdf_text(pdf, selection)``; for an
EPUB or MOBI selection the header and section units render exactly as one
unit holding ``to_plain_text()`` of the extraction filtered to the selected
sections. One unit's rendering is ``format_page_line(text, None, None)``
followed by one newline, in txt and md alike. A document with one unit
(``idx`` 0, ``source_index`` NULL) renders as that unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from chronotranscriber.common.store import PlannedUnit, UnitResult
from chronotranscriber.llm.response_parsing import format_page_line
from chronotranscriber.output.render import render_md, render_txt
from chronotranscriber.output.store import (
    KIND_TEXT,
    open_store,
    read_only_connection,
)
from chronotranscriber.sources._text import metadata_header, normalize_text
from chronotranscriber.sources.epub import EPUBTextExtraction
from chronotranscriber.sources.mobi import MOBITextExtraction
from chronotranscriber.sources.pdf import (
    native_extract_pdf_pages,
    native_extract_pdf_text,
)

PAGES = ["Erste Seite\nmit Suppe", "", "Dritte Seite", "Vierte Seite: Brot"]
SELECTIONS = {
    "full": None,
    "contiguous": [1, 2],
    "non_contiguous": [0, 2],
    "single": [3],
    "empty_only": [1],
}
SECTIONS = [
    "  Erstes Kapitel.\n\n\n  Suppe kochen.  ",
    "",
    "\n\n",
    "Zweites Kapitel:\nBrot backen.\n",
    "Schluss.",
]
SECTION_SELECTIONS = {
    "full": None,
    "contiguous": [2, 3],
    "non_contiguous": [0, 3, 4],
    "single": [4],
    "empty_only": [1, 2],
}
METADATA = {
    "title_and_authors": ("Kochbuch", ["Anna Muster", " ", "Ben Beispiel "]),
    "title_only": ("  Kochbuch  ", []),
    "authors_only": (None, ["Anna Muster"]),
    "blank_authors": (None, ["  "]),
    "none": (None, []),
}


def _old(text: str) -> str:
    """The rendering of one ``text`` unit holding *text*."""
    return format_page_line(text, None, None).rstrip("\n") + "\n"


def _text_pdf(path: Path) -> Path:
    document = fitz.open()
    for text in PAGES:
        page = document.new_page(width=300, height=200)
        if text:
            page.insert_text((20, 50), text)
    document.save(path)
    document.close()
    return path


async def _render(
    root: Path,
    kind: str | None,
    units: Sequence[tuple[int, int | None, str]],
    total: int,
) -> tuple[str, str]:
    """Store *units* (idx, source_index, content) of *total*; render txt and md."""
    async with open_store(root) as store:
        await store.begin_run("transcribe")
        doc_id = await store.ensure_document("doc", kind=kind)
        lease = await store.acquire(doc_id)
        assert lease is not None
        await store.plan_units(
            lease,
            KIND_TEXT,
            [PlannedUnit(idx) for idx in range(total)],
            model="native",
            provider="local",
        )
        for idx, source_index, content in units:
            await store.write_unit(
                lease,
                UnitResult(
                    kind=KIND_TEXT,
                    idx=idx,
                    status="done",
                    model="native",
                    provider="local",
                    content=content,
                    source_index=source_index,
                ),
            )
        await store.release(lease)
    with read_only_connection(root) as connection:
        return render_txt(connection, doc_id), render_md(connection, doc_id)


@pytest.mark.parametrize("kind", ["pdf", None])
@pytest.mark.parametrize("selection", list(SELECTIONS), ids=list(SELECTIONS))
async def test_pdf_page_units_render_as_the_selection_text(
    tmp_path: Path, selection: str, kind: str | None
) -> None:
    pdf = _text_pdf(tmp_path / "book.pdf")
    chosen = SELECTIONS[selection]
    indices = list(range(len(PAGES))) if chosen is None else chosen
    pages = native_extract_pdf_pages(pdf, indices)
    units = [(idx, idx, pages[idx]) for idx in indices]
    root = tmp_path / "out"
    root.mkdir()

    txt, md = await _render(root, kind, units, len(PAGES))

    expected = _old(native_extract_pdf_text(pdf, chosen))
    assert (txt, md) == (expected, expected)
    if selection == "full":
        assert "Dritte Seite" in txt


def _extraction(
    kind: str, title: str | None, authors: list[str], sections: list[str]
) -> EPUBTextExtraction | MOBITextExtraction:
    if kind == "epub":
        return EPUBTextExtraction(title, authors, sections)
    return MOBITextExtraction(title, authors, sections, "epub")


@pytest.mark.parametrize("kind", ["epub", "mobi"])
@pytest.mark.parametrize(
    ("metadata", "selection"),
    [
        *[("title_and_authors", name) for name in SECTION_SELECTIONS],
        *[(name, "non_contiguous") for name in METADATA if name != "title_and_authors"],
    ],
)
async def test_ebook_units_render_as_the_filtered_plain_text(
    tmp_path: Path, selection: str, metadata: str, kind: str
) -> None:
    title, authors = METADATA[metadata]
    chosen = SECTION_SELECTIONS[selection]
    keep = list(range(len(SECTIONS))) if chosen is None else chosen
    units: list[tuple[int, int | None, str]] = [
        (0, None, metadata_header(title, authors))
    ]
    units += [(i + 1, i, normalize_text(SECTIONS[i])) for i in keep]
    root = tmp_path / "out"
    root.mkdir()

    txt, md = await _render(root, kind, units, len(SECTIONS) + 1)

    filtered = _extraction(kind, title, authors, [SECTIONS[i] for i in keep])
    expected = _old(filtered.to_plain_text())
    assert (txt, md) == (expected, expected)


@pytest.mark.parametrize("kind", ["pdf", "epub", "mobi", None])
async def test_a_single_text_unit_renders_as_its_content(
    tmp_path: Path, kind: str | None
) -> None:
    text = "# Title: Kochbuch\n\nErstes Kapitel.\n\nZweites Kapitel.\n"

    txt, md = await _render(tmp_path, kind, [(0, None, text)], 1)

    assert (txt, md) == (_old(text), _old(text))
