"""The Tesseract and native methods write their units into the store.

Run on Windows with:

    uv --directory <repo> run pytest tests/methods/test_local_methods_store.py -q

The transcribers are called directly with an item that holds a leased store
document, as the run core hands it over. Tesseract preprocessing runs for
real on small generated pages; ``perform_ocr`` is replaced by a fake that
records each image and whether it existed. Oracle: one unit per page under
its absolute index, a resume that OCRs only the pending units, preprocessed
images that exist only while the item runs (unless kept), native text units
that render to the bytes the old writer produced, and no JSONL or folder
beside the input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest
from ebooklib import epub
from PIL import Image

import chronotranscriber.methods.native as native_method
import chronotranscriber.methods.tesseract as tesseract_method
from chronotranscriber.common.selection import parse_page_range
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.common.store import Store
from chronotranscriber.llm.response_parsing import format_page_line
from chronotranscriber.methods import Item, Job, NothingSelectedError
from chronotranscriber.methods.native import NativeTranscriber
from chronotranscriber.methods.tesseract import (
    PREPROCESSED_FOLDER,
    TesseractTranscriber,
)
from chronotranscriber.methods.units import DocumentHandle, UnitsFailedError
from chronotranscriber.output.render import render_item
from chronotranscriber.output.store import database_path, open_store
from chronotranscriber.sources.pdf import native_extract_pdf_text
from tests.conftest import make_job, make_spec
from tests.recording_events import RecordingEvents


class FakeOcr:
    """Stand-in for ``perform_ocr``: records every image and its existence."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.seen: list[Path] = []
        self.existed: list[bool] = []

    def __call__(self, path: Path, _config: str) -> str | None:
        self.seen.append(path)
        self.existed.append(path.is_file())
        if path.name in self.fail:
            return None
        return f"text of {path.name}"

    @property
    def names(self) -> list[str]:
        return sorted(path.name for path in self.seen)


@pytest.fixture
def ocr(monkeypatch: pytest.MonkeyPatch) -> FakeOcr:
    fake = FakeOcr()
    monkeypatch.setattr(tesseract_method, "perform_ocr", fake)
    monkeypatch.setattr(tesseract_method, "ensure_tesseract_available", lambda: True)
    return fake


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The output root that holds the store, apart from the inputs."""
    folder = tmp_path / "store"
    folder.mkdir()
    return folder


def _blank_pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=72, height=96)
    document.save(path)
    document.close()
    return path


def _text_pdf(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for line in lines:
        page = document.new_page(width=300, height=200)
        page.insert_text((20, 50), line)
    document.save(path)
    document.close()
    return path


def _image_folder(path: Path, names: list[str]) -> Path:
    path.mkdir(parents=True)
    for name in names:
        Image.new("RGB", (40, 60), "white").save(path / name)
    return path


def _epub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.EpubBook()
    book.set_identifier("ct-store-test")
    book.set_title("Kochbuch")
    book.add_author("Anna Muster")
    chapters = []
    for number, text in enumerate(["Erstes Kapitel: Suppe.", "Zweites: Brot."], 1):
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


class _FakeMobiExtraction:
    title = "Mobi"
    authors = ()
    sections = ("Erster Abschnitt", "Zweiter Abschnitt")

    def to_plain_text(self) -> str:
        return "# Title: Mobi\n\n" + "\n\n".join(self.sections) + "\n"


class _FakeMobiProcessor:
    def __init__(self, path: Path) -> None:
        self.path = path

    def extract_text(self) -> _FakeMobiExtraction:
        return _FakeMobiExtraction()


def _job(
    item: Path, source: str, method: str, *, output_dir: Path | None, **flags: Any
) -> Job:
    spec = make_spec(item, type=source, method=method, **flags)
    job: Job = make_job(
        spec,
        output_dir=output_dir,
        input_root=item.parent,
        events=RecordingEvents(),
    )
    return job


async def _document(
    store: Store, item: Path, kind: str | None = None
) -> DocumentHandle:
    if store.run_id is None:
        await store.begin_run("transcribe")
    doc_id = await store.ensure_document(item.stem, input_path=item, kind=kind)
    lease = await store.acquire(doc_id)
    assert lease is not None
    return DocumentHandle(store, doc_id, item.stem, lease, "run-model", "run-provider")


async def _transcribe(
    root: Path,
    transcriber: Any,
    item: Path,
    source: str,
    method: str,
    pages: str | None = None,
) -> list[Path]:
    """Run one item as the run core would, with a fresh run of the store."""
    async with open_store(root) as store:
        handle = await _document(store, item, source)
        try:
            return list(
                await transcriber.transcribe(
                    Item(
                        item,
                        source,
                        method,
                        pages=parse_page_range(pages) if pages else None,
                        document=handle,
                    )
                )
            )
        finally:
            await store.release(handle.lease)


async def _render(root: Path, name: str, output: Path) -> dict[str, bytes]:
    """Render document *name* into txt and md beside *output*; return the bytes."""
    async with open_store(root) as store:
        await store.begin_run("render")
        row = store.document(name)
        assert row is not None
        lease = await store.acquire(row.doc_id)
        assert lease is not None
        await render_item(store, row.doc_id, ("txt", "md"), output, lease)
        await store.release(lease)
    return {
        suffix: output.with_suffix(f".{suffix}").read_bytes()
        for suffix in ("txt", "md")
    }


def _units(root: Path) -> list[dict[str, Any]]:
    return read_rows(database_path(root), "units", order_by=["kind", "idx"])


def _status(root: Path) -> str:
    (row,) = read_rows(database_path(root), "documents")
    return str(row["status"])


def _old_bytes(text: str) -> dict[str, bytes]:
    """The txt and md bytes the old writer produced for one native page."""
    line = format_page_line(text, None, None)
    body = line.rstrip("\n") + "\n"
    return {"txt": body.encode("utf-8"), "md": body.encode("utf-8")}


def _no_side_files(inbox: Path, inputs: set[str], tmp_path: Path) -> None:
    assert {path.name for path in inbox.iterdir()} == inputs
    assert list(tmp_path.rglob("*.jsonl")) == []


class TestTesseractUnits:
    async def test_a_pdf_writes_one_unit_per_page_and_resumes_only_pending(
        self, tmp_path: Path, root: Path, ocr: FakeOcr
    ) -> None:
        inbox = tmp_path / "inbox"
        scan = _blank_pdf(inbox / "scan.pdf", 3)
        job = _job(scan, "pdf", "tesseract", output_dir=None)

        first = await _transcribe(
            root, TesseractTranscriber(job), scan, "pdf", "tesseract", pages="2-3"
        )

        assert first == []
        assert ocr.names == [
            "page_0002_tess_preprocessed.png",
            "page_0003_tess_preprocessed.png",
        ]
        rows = _units(root)
        assert [(row["kind"], row["idx"], row["status"]) for row in rows] == [
            ("page", 0, "pending"),
            ("page", 1, "done"),
            ("page", 2, "done"),
        ]
        assert [row["source_index"] for row in rows] == [0, 1, 2]
        assert rows[1]["content"] == "text of page_0002_tess_preprocessed.png"
        assert (rows[1]["provider"], rows[1]["model"]) == ("local", "tesseract")
        assert _status(root) == "partial"

        ocr.seen.clear()
        await _transcribe(root, TesseractTranscriber(job), scan, "pdf", "tesseract")

        assert ocr.names == ["page_0001_tess_preprocessed.png"]
        assert [row["status"] for row in _units(root)] == ["done"] * 3
        assert _status(root) == "complete"
        assert read_rows(database_path(root), "responses") == []
        _no_side_files(inbox, {"scan.pdf"}, tmp_path)

    async def test_a_folder_skips_done_images_and_retries_failed_ones(
        self, tmp_path: Path, root: Path, ocr: FakeOcr
    ) -> None:
        inbox = tmp_path / "inbox"
        folder = _image_folder(inbox / "scans", ["p1.png", "p2.png", "p10.png"])
        ocr.fail = {"p2_tess_preprocessed.png"}
        job = _job(folder, "images", "tesseract", output_dir=None)

        with pytest.raises(UnitsFailedError, match="1 of 3"):
            await _transcribe(
                root, TesseractTranscriber(job), folder, "images", "tesseract"
            )
        rows = _units(root)
        assert [(row["idx"], row["label"], row["status"]) for row in rows] == [
            (0, "p1_tess_preprocessed.png", "done"),
            (1, "p2_tess_preprocessed.png", "failed"),
            (2, "p10_tess_preprocessed.png", "done"),
        ]
        assert rows[1]["content"] == "[transcription error: p2_tess_preprocessed.png]"

        ocr.seen.clear()
        kept = _job(folder, "images", "tesseract", output_dir=None, retry_errors=False)
        await _transcribe(
            root, TesseractTranscriber(kept), folder, "images", "tesseract"
        )
        assert ocr.seen == []

        ocr.fail = set()
        retry = _job(folder, "images", "tesseract", output_dir=None)
        await _transcribe(
            root, TesseractTranscriber(retry), folder, "images", "tesseract"
        )
        assert ocr.names == ["p2_tess_preprocessed.png"]
        assert [row["status"] for row in _units(root)] == ["done"] * 3
        _no_side_files(inbox / "scans", {"p1.png", "p2.png", "p10.png"}, tmp_path)
        assert {path.name for path in inbox.iterdir()} == {"scans"}

    async def test_force_transcribes_every_selected_page_again(
        self, tmp_path: Path, root: Path, ocr: FakeOcr
    ) -> None:
        scan = _blank_pdf(tmp_path / "inbox" / "scan.pdf", 2)
        job = _job(scan, "pdf", "tesseract", output_dir=None)
        await _transcribe(root, TesseractTranscriber(job), scan, "pdf", "tesseract")
        ocr.seen.clear()

        forced = _job(scan, "pdf", "tesseract", output_dir=None, force=True)
        await _transcribe(root, TesseractTranscriber(forced), scan, "pdf", "tesseract")

        assert len(ocr.seen) == 2
        assert [row["status"] for row in _units(root)] == ["done", "done"]

    async def test_preprocessed_images_exist_only_while_the_item_runs(
        self, tmp_path: Path, root: Path, ocr: FakeOcr
    ) -> None:
        inbox = tmp_path / "inbox"
        scan = _blank_pdf(inbox / "scan.pdf", 2)
        job = _job(scan, "pdf", "tesseract", output_dir=tmp_path / "out")

        await _transcribe(root, TesseractTranscriber(job), scan, "pdf", "tesseract")

        assert ocr.existed == [True, True]
        assert all(not path.exists() for path in ocr.seen)
        assert not any(
            {inbox, tmp_path / "out", root} & set(path.parents) for path in ocr.seen
        )
        assert not (tmp_path / "out").exists()
        _no_side_files(inbox, {"scan.pdf"}, tmp_path)

    async def test_keep_working_files_keeps_them_in_the_item_folder(
        self, tmp_path: Path, root: Path, ocr: FakeOcr
    ) -> None:
        scan = _blank_pdf(tmp_path / "inbox" / "scan.pdf", 2)
        job = _job(
            scan,
            "pdf",
            "tesseract",
            output_dir=tmp_path / "out",
            keep_working_files=True,
        )

        await _transcribe(root, TesseractTranscriber(job), scan, "pdf", "tesseract")

        kept = job.locate(scan).folder / PREPROCESSED_FOLDER
        assert sorted(path.name for path in kept.iterdir()) == ocr.names
        assert all(path.parent == kept for path in ocr.seen)


class TestNativeUnits:
    async def test_a_native_pdf_is_one_text_unit_per_page_rendered_as_before(
        self, tmp_path: Path, root: Path
    ) -> None:
        inbox = tmp_path / "inbox"
        pdf = _text_pdf(inbox / "book.pdf", ["Erste Seite", "Zweite Seite"])
        job = _job(pdf, "pdf", "native", output_dir=None)

        assert (
            await _transcribe(root, NativeTranscriber(job), pdf, "pdf", "native") == []
        )

        units = _units(root)
        assert [
            (unit["kind"], unit["idx"], unit["source_index"], unit["status"])
            for unit in units
        ] == [("text", 0, 0, "done"), ("text", 1, 1, "done")]
        assert {(unit["provider"], unit["model"]) for unit in units} == {
            ("local", "native")
        }
        assert [unit["content"] for unit in units] == [
            native_extract_pdf_text(pdf, [0]),
            native_extract_pdf_text(pdf, [1]),
        ]
        assert {unit["outcome"] for unit in units} == {"ok"}
        assert _status(root) == "complete"
        _no_side_files(inbox, {"book.pdf"}, tmp_path)
        rendered = await _render(root, "book", tmp_path / "rendered" / "book.txt")
        assert rendered == _old_bytes(native_extract_pdf_text(pdf))

    async def test_an_epub_is_a_header_unit_and_one_unit_per_section(
        self, tmp_path: Path, root: Path
    ) -> None:
        inbox = tmp_path / "inbox"
        book = _epub(inbox / "kochbuch.epub")
        job = _job(book, "epub", "native", output_dir=None)

        await _transcribe(root, NativeTranscriber(job), book, "epub", "native")

        units = _units(root)
        extraction = native_method.EPUBProcessor(book).extract_text()
        text = extraction.to_plain_text()
        assert "Suppe" in text
        assert len(units) == len(extraction.sections) + 1
        assert {unit["kind"] for unit in units} == {"text"}
        assert units[0]["content"] == "# Title: Kochbuch\n# Author(s): Anna Muster"
        assert [unit["source_index"] for unit in units[1:]] == list(
            range(len(extraction.sections))
        )
        _no_side_files(inbox, {"kochbuch.epub"}, tmp_path)
        rendered = await _render(root, "kochbuch", tmp_path / "rendered" / "k.txt")
        assert rendered == _old_bytes(text)

    async def test_a_mobi_is_a_header_unit_and_one_unit_per_section(
        self, tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(native_method._EBOOKS, "mobi", (_FakeMobiProcessor, "MOBI"))
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        book = inbox / "roman.mobi"
        book.write_bytes(b"BOOKMOBI")
        job = _job(book, "mobi", "native", output_dir=None)

        await _transcribe(root, NativeTranscriber(job), book, "mobi", "native")

        assert [(unit["idx"], unit["content"]) for unit in _units(root)] == [
            (0, "# Title: Mobi"),
            (1, "Erster Abschnitt"),
            (2, "Zweiter Abschnitt"),
        ]
        text = _FakeMobiExtraction().to_plain_text()
        _no_side_files(inbox, {"roman.mobi"}, tmp_path)
        rendered = await _render(root, "roman", tmp_path / "rendered" / "roman.txt")
        assert rendered == _old_bytes(text)

    async def test_an_ebook_whose_extraction_fails_fails_the_item(
        self, tmp_path: Path, root: Path
    ) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        book = inbox / "broken.epub"
        book.write_bytes(b"not an epub")
        job = _job(book, "epub", "native", output_dir=None)

        with pytest.raises(RuntimeError, match="broken.epub"):
            await _transcribe(root, NativeTranscriber(job), book, "epub", "native")

        assert _units(root) == []

    async def test_a_pdf_without_a_text_layer_goes_to_tesseract(
        self, tmp_path: Path, root: Path, ocr: FakeOcr
    ) -> None:
        inbox = tmp_path / "inbox"
        scan = _blank_pdf(inbox / "scan.pdf", 2)
        job = _job(scan, "pdf", "native", output_dir=None)

        await _transcribe(root, NativeTranscriber(job), scan, "pdf", "native")

        assert ocr.names == [
            "page_0001_tess_preprocessed.png",
            "page_0002_tess_preprocessed.png",
        ]
        rows = _units(root)
        assert [(row["kind"], row["status"]) for row in rows] == [("page", "done")] * 2
        events = job.events
        assert isinstance(events, RecordingEvents)
        assert any("not searchable" in message for message in events.of("warning"))
        _no_side_files(inbox, {"scan.pdf"}, tmp_path)

    async def test_without_tesseract_a_scanned_pdf_fails(
        self, tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tesseract_method, "ensure_tesseract_available", lambda: False
        )
        scan = _blank_pdf(tmp_path / "inbox" / "scan.pdf", 1)
        job = _job(scan, "pdf", "native", output_dir=None)

        with pytest.raises(tesseract_method.TesseractMissingError):
            await _transcribe(root, NativeTranscriber(job), scan, "pdf", "native")

        assert _units(root) == []


class _SectionExtraction:
    title = None
    authors = ()

    def __init__(self, sections: list[str]) -> None:
        self.sections = sections


def _section_processor(count: int) -> type[Any]:
    class _SectionProcessor:
        def __init__(self, path: Path) -> None:
            self.path = path

        def extract_text(self) -> _SectionExtraction:
            return _SectionExtraction([f"section {i}" for i in range(count)])

    return _SectionProcessor


async def _ebook_text(
    tmp_path: Path,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    pages: str,
    sections: int,
) -> tuple[list[str] | None, RecordingEvents]:
    """Run the native method on a fake EPUB; return its stored sections and events."""
    monkeypatch.setitem(
        native_method._EBOOKS, "epub", (_section_processor(sections), "EPUB")
    )
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    book = inbox / "book.epub"
    book.write_bytes(b"epub")
    job = _job(book, "epub", "native", output_dir=None)

    await _transcribe(root, NativeTranscriber(job), book, "epub", "native", pages)

    _no_side_files(inbox, {"book.epub"}, tmp_path)
    events = job.events
    assert isinstance(events, RecordingEvents)
    units = _units(root)
    if not units:
        return None, events
    assert len(units) == sections + 1
    assert {unit["kind"] for unit in units} == {"text"}
    header, *rest = units
    assert (header["idx"], header["status"], header["content"]) == (0, "done", "")
    return [str(unit["content"]) for unit in rest if unit["status"] == "done"], events


class TestEbookPageRangeResolvesAgainstRealCount:
    async def test_last_n_selects_final_sections_not_empty(
        self, tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text, _events = await _ebook_text(tmp_path, root, monkeypatch, "last:5", 10)

        assert text == [f"section {i}" for i in range(5, 10)]
        assert _status(root) == "partial"

    async def test_open_span_clamps_to_real_count(
        self, tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text, _events = await _ebook_text(tmp_path, root, monkeypatch, "3-", 10)

        assert text == [f"section {i}" for i in range(2, 10)]

    async def test_out_of_range_skips_and_writes_nothing(
        self, tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(NothingSelectedError, match="selected no EPUB sections"):
            await _ebook_text(tmp_path, root, monkeypatch, "50-60", 3)

        assert _units(root) == []
