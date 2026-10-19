"""The Tesseract method: a missing engine fails the item; folder preprocessing
runs off the event loop; page events follow the stored unit; the absolute page
order survives a page range; pages dropped in preprocessing are reported."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import fitz  # PyMuPDF
import pytest
from PIL import Image

import chronotranscriber.methods.tesseract as tesseract_method
import chronotranscriber.run as core
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.images.tesseract_runtime import TESSERACT_MISSING
from chronotranscriber.methods import Item, PageCounter
from chronotranscriber.methods.tesseract import (
    TesseractMissingError,
    TesseractTranscriber,
)
from chronotranscriber.methods.units import DocumentHandle
from chronotranscriber.output.store import database_path, open_store
from tests.conftest import make_job, make_settings, make_spec
from tests.recording_events import RecordingEvents


def _scan(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    document.new_page(width=200, height=300)
    document.save(path)
    document.close()
    return path


def _image_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "page1.png").write_bytes(b"png")
    return folder


def _method(tmp_path: Path, folder: Path) -> TesseractTranscriber:
    spec = make_spec(folder, type="images", method="tesseract")
    job = make_job(spec, output_dir=tmp_path / "out", input_root=tmp_path)
    return TesseractTranscriber(job)


@pytest.fixture
def no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tesseract_method, "ensure_tesseract_available", lambda: False)


@pytest.mark.usefixtures("no_engine")
def test_a_missing_engine_fails_a_pdf_item(tmp_path: Path) -> None:
    scan = _scan(tmp_path / "in" / "scan.pdf")
    events = RecordingEvents()
    run_plan = core.plan(
        make_spec(scan, type="pdf", method="tesseract", output=tmp_path / "out"),
        make_settings(),
    )

    result = asyncio.run(core.run(run_plan, events))

    assert (result.complete, result.failed) == (0, 1)
    (finished,) = events.of("item_finished")
    assert finished.success is False
    assert finished.error == TESSERACT_MISSING
    assert not list((tmp_path / "out").rglob("preprocessed_images_tesseract"))


@pytest.mark.usefixtures("no_engine")
def test_a_missing_engine_fails_an_image_folder(tmp_path: Path) -> None:
    folder = _image_folder(tmp_path)
    method = _method(tmp_path, folder)

    with pytest.raises(TesseractMissingError, match="Tesseract is not installed"):
        asyncio.run(method.transcribe(Item(folder, "images", "tesseract")))


def test_folder_preprocessing_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _image_folder(tmp_path)
    threads: list[int] = []
    written = [tmp_path / "pre" / "page1_tess_preprocessed.png"]

    def preprocess(*_args: Any, **_kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return written, {written[0].name: 0}

    monkeypatch.setattr(tesseract_method, "ensure_tesseract_available", lambda: True)
    monkeypatch.setattr(
        tesseract_method.ImageProcessor,
        "process_and_save_images_for_tesseract",
        staticmethod(preprocess),
    )
    method = _method(tmp_path, folder)
    ocr = AsyncMock(return_value=None)
    monkeypatch.setattr(method, "_ocr", ocr)

    async def transcribe() -> int:
        async with open_store(tmp_path) as store:
            await store.begin_run("transcribe")
            doc_id = await store.ensure_document("scans", input_path=folder)
            lease = await store.acquire(doc_id)
            assert lease is not None
            handle = DocumentHandle(store, doc_id, "scans", lease, "m", "p")
            await method.transcribe(
                Item(folder, "images", "tesseract", document=handle)
            )
        return threading.get_ident()

    loop_thread = asyncio.run(transcribe())

    assert len(threads) == 1
    assert threads[0] != loop_thread
    ocr.assert_awaited_once()


def _units(root: Path) -> list[dict[str, Any]]:
    return read_rows(database_path(root), "units", order_by=["idx"])


async def _run_item(root: Path, method: TesseractTranscriber, item: Item) -> None:
    """Transcribe *item* under a leased store document, as the run core does."""
    async with open_store(root) as store:
        await store.begin_run("transcribe")
        doc_id = await store.ensure_document(item.name, input_path=item.path)
        lease = await store.acquire(doc_id)
        assert lease is not None
        handle = DocumentHandle(store, doc_id, item.name, lease, "m", "p")
        try:
            await method.transcribe(
                Item(item.path, item.source, item.method, document=handle)
            )
        finally:
            await store.release(lease)


class TestPageEvents:
    async def test_a_tesseract_pass_reports_each_page_after_its_unit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "src"
        folder.mkdir()
        Image.new("RGB", (40, 60), "white").save(folder / "page1.png")
        events = RecordingEvents()
        spec = make_spec(folder, type="images", method="tesseract")
        job = make_job(spec, output_dir=None, input_root=tmp_path, events=events)
        root = tmp_path / "store"
        root.mkdir()
        stored: list[list[str]] = []
        original = PageCounter.__call__

        def counting(self: PageCounter, index: int, failed: bool) -> None:
            stored.append([str(unit["status"]) for unit in _units(root)])
            original(self, index, failed)

        monkeypatch.setattr(PageCounter, "__call__", counting)
        monkeypatch.setattr(
            tesseract_method, "ensure_tesseract_available", lambda: True
        )
        monkeypatch.setattr(tesseract_method, "perform_ocr", lambda p, c: "ok text")

        await _run_item(
            root, TesseractTranscriber(job), Item(folder, "images", "tesseract")
        )

        (done,) = events.of("page_done")
        assert (done.item, done.status, done.done, done.total, done.unit) == (
            "src",
            "ok",
            1,
            1,
            "image",
        )
        assert stored == [["done"]]


def _run_tess_preprocess(
    tmp_path: Path, filenames: list[str], page_indices: list[int] | None
) -> tuple[list[Path], dict[str, int]]:
    from chronotranscriber.images.pipeline import ImageProcessor

    source = tmp_path / "src"
    source.mkdir()
    for name in filenames:
        (source / name).write_bytes(b"fake image data")
    out_dir = tmp_path / "pre"

    def fake_task(img_path: Path, out_path: Path, settings: Any) -> str:
        out_path.write_bytes(b"out")
        return str(out_path)

    with patch.object(
        ImageProcessor, "_tesseract_preprocess_image_task", staticmethod(fake_task)
    ):
        return ImageProcessor.process_and_save_images_for_tesseract(
            source, out_dir, page_indices=page_indices
        )


class TestTesseractOrderMap:
    @pytest.mark.unit
    def test_order_map_uses_absolute_indices_under_page_range(
        self, tmp_path: Path
    ) -> None:
        """With a page subset, each output maps to its full-listing index."""
        files, order_map = _run_tess_preprocess(
            tmp_path,
            ["page_1.png", "page_2.png", "page_3.png", "page_4.png"],
            page_indices=[2, 3],
        )
        assert order_map == {
            "page_3_tess_preprocessed.png": 2,
            "page_4_tess_preprocessed.png": 3,
        }
        assert len(files) == 2

    @pytest.mark.unit
    def test_order_map_covers_full_listing_without_range(self, tmp_path: Path) -> None:
        _files, order_map = _run_tess_preprocess(
            tmp_path, ["a.png", "b.png"], page_indices=None
        )
        assert order_map == {
            "a_tess_preprocessed.png": 0,
            "b_tess_preprocessed.png": 1,
        }


class _Ocr:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, path: Path, _config: str) -> str:
        self.seen.append(path.name)
        return f"text of {path.name}"


@pytest.fixture
def scans(tmp_path: Path) -> Path:
    folder = tmp_path / "scans"
    folder.mkdir()
    for i in range(1, 5):
        (folder / f"page_{i}.png").write_bytes(b"")
    return folder


@pytest.fixture
def ocr(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Ocr]:
    fake = _Ocr()
    monkeypatch.setattr(tesseract_method, "ensure_tesseract_available", lambda: True)
    monkeypatch.setattr(tesseract_method, "perform_ocr", fake)
    yield fake


def _dropping_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: Path, written: list[int]
) -> TesseractTranscriber:
    """A Tesseract method whose preprocessing wrote the pages *written* of four."""
    spec = make_spec(folder, type="images", method="tesseract")
    job = make_job(spec, output_dir=tmp_path / "out", input_root=tmp_path)
    order_map = {f"page_{i}_tess_preprocessed.png": i - 1 for i in range(1, 5)}

    def fake_pre(src: Any, dst: Path, page_indices: Any = None, **_kw: Any) -> Any:
        dst.mkdir(parents=True, exist_ok=True)
        files = [dst / f"page_{i}_tess_preprocessed.png" for i in written]
        for path in files:
            path.write_bytes(b"")
        return files, order_map

    monkeypatch.setattr(
        tesseract_method.ImageProcessor,
        "process_and_save_images_for_tesseract",
        staticmethod(fake_pre),
    )
    return TesseractTranscriber(job)


class TestDroppedTesseractPagesGuard:
    async def test_excessive_drop_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        scans: Path,
        ocr: _Ocr,
    ) -> None:
        """4 attempted, 1 written: above the 50% threshold, so it raises."""
        method = _dropping_tesseract(tmp_path, monkeypatch, scans, written=[1])
        root = tmp_path / "store"
        root.mkdir()

        with pytest.raises(RuntimeError, match="failed for 3/4 pages"):
            await _run_item(root, method, Item(scans, "images", "tesseract"))

        assert ocr.seen == []
        assert [unit["status"] for unit in _units(root)] == ["pending"] * 4

    async def test_minor_drop_proceeds_and_leaves_the_page_pending(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        scans: Path,
        ocr: _Ocr,
    ) -> None:
        """4 attempted, 3 written: below the threshold, so it proceeds."""
        method = _dropping_tesseract(tmp_path, monkeypatch, scans, written=[1, 2, 4])
        root = tmp_path / "store"
        root.mkdir()

        await _run_item(root, method, Item(scans, "images", "tesseract"))

        assert sorted(ocr.seen) == [
            "page_1_tess_preprocessed.png",
            "page_2_tess_preprocessed.png",
            "page_4_tess_preprocessed.png",
        ]
        rows = _units(root)
        assert [(row["idx"], row["status"]) for row in rows] == [
            (0, "done"),
            (1, "done"),
            (2, "pending"),
            (3, "done"),
        ]
        assert rows[3]["content"] == "text of page_4_tess_preprocessed.png"
