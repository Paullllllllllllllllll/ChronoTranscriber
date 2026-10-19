"""Page sources: ChronoTranscriber's page names, page units and file payloads."""

from __future__ import annotations

import base64
import hashlib
import random
import subprocess
import tracemalloc
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest
from PIL import Image

from chronotranscriber.methods.pages import transcribe_payload
from chronotranscriber.sources import images
from chronotranscriber.sources.audio import ffmpeg
from chronotranscriber.sources.images import (
    FolderPages,
    PageImage,
    encode_image_to_base64,
    list_folder_images,
    load_image,
    stream_pages,
)
from chronotranscriber.sources.pdf import PdfPages, render_pdf_page

IMG_CFG: dict[str, Any] = {
    "target_dpi": 72,
    "grayscale_conversion": True,
    "handle_transparency": True,
    "jpeg_quality": 90,
    "llm_detail": "high",
    "high_target_box": [64, 128],
    "resize_profile": "auto",
}


def make_pdf(path: Path, pages: int = 3) -> Path:
    with fitz.open() as doc:
        for i in range(pages):
            page = doc.new_page(width=200, height=100)
            page.insert_text((20, 50), f"Page {i + 1}")
        doc.save(str(path))
    return path


def make_folder(folder: Path, count: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (60, 40), (i * 20, i * 20, i * 20)).save(
            folder / f"scan_{i:02d}.png"
        )
    return folder


async def collect(source: Any, indices: list[int] | None = None) -> list[PageImage]:
    return [page async for page in stream_pages(source, indices)]


class TestNames:
    def test_folder_listing_is_natural_and_skips_other_files(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        for name in ("page_10.png", "page_2.jp2", "notes.txt", "page_1.tif"):
            (folder / name).write_bytes(b"")
        assert [p.name for p in list_folder_images(folder)] == [
            "page_1.tif",
            "page_2.jp2",
            "page_10.png",
        ]


class TestPdfStream:
    async def test_yields_named_pages_with_provenance(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf")
        pages = await collect(PdfPages(pdf, IMG_CFG, "openai"))

        assert [p.index for p in pages] == [0, 1, 2]
        assert [p.image_name for p in pages] == [
            "page_0001_pre_processed.jpg",
            "page_0002_pre_processed.jpg",
            "page_0003_pre_processed.jpg",
        ]
        for page in pages:
            assert page.render_error is None
            assert page.source_file == str(pdf)
            assert page.page_index == page.index
            data = base64.b64decode(page.base64)
            assert page.mime_type == "image/jpeg"
            provenance = page.provenance()
            assert provenance["byte_size"] == len(data)
            assert provenance["sha256"] == hashlib.sha256(data).hexdigest()
            # The direct strategy never renders above the target resolution.
            assert 0 < provenance["effective_dpi"] <= 72

    async def test_streams_only_the_selected_pages(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=5)
        pages = await collect(PdfPages(pdf, IMG_CFG, "openai"), [1, 3, 99])
        assert [p.index for p in pages] == [1, 3]

    async def test_memory_stays_bounded(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=40)
        tracemalloc.start()
        try:
            count = 0
            async for _page in stream_pages(
                PdfPages(pdf, {**IMG_CFG, "target_dpi": 150}, "openai")
            ):
                count += 1
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert count == 40
        # One page at a time: far below an all-pages accumulation.
        assert peak < 80 * 1024 * 1024

    async def test_the_source_is_closed_afterwards(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        source = PdfPages(pdf, IMG_CFG, "openai")
        await collect(source)
        assert len(source) == 0

    def test_supersample_renders_at_the_target(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        cfg = {**IMG_CFG, "render_strategy": "supersample"}
        page = render_pdf_page(pdf, 0, cfg, "openai")
        assert page.provenance()["effective_dpi"] == 72

    def test_direct_renders_below_the_target_for_a_box_fit(
        self, tmp_path: Path
    ) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        page = render_pdf_page(pdf, 0, IMG_CFG, "openai")
        assert 0 < page.provenance()["effective_dpi"] < 72

    def test_a_single_page_keeps_its_name(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=2)
        page = render_pdf_page(pdf, 1, IMG_CFG, "openai")
        assert (page.index, page.page_index) == (1, 1)
        assert page.image_name == "page_0002_pre_processed.jpg"
        with pytest.raises(IndexError, match="out of range"):
            render_pdf_page(pdf, 2, IMG_CFG, "openai")


class TestFolderStream:
    async def test_yields_sorted_images_with_page_names(self, tmp_path: Path) -> None:
        folder = make_folder(tmp_path / "scans")
        pages = await collect(FolderPages(folder, IMG_CFG, "openai"))

        assert [p.image_name for p in pages] == [
            "scan_00.png_pre_processed.jpg",
            "scan_01.png_pre_processed.jpg",
            "scan_02.png_pre_processed.jpg",
        ]
        assert [p.index for p in pages] == [0, 1, 2]
        assert all(p.page_index is None for p in pages)
        assert pages[0].source_file == str(folder / "scan_00.png")

    def test_a_single_image_takes_the_given_index(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.png"
        Image.new("RGB", (30, 30), "white").save(path)
        page = load_image(path, 4, IMG_CFG, "openai")
        assert page.image_name == "scan.png_pre_processed.jpg"
        assert page.index == 4

    def test_undecodable_jpeg2000_is_decoded_with_ffmpeg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "page_7.jp2").write_bytes(b"not a codestream")
        calls: list[Path] = []

        def decode(source: Path, target: Path) -> None:
            calls.append(source)
            Image.new("L", (40, 60), 255).save(target)

        monkeypatch.setattr(
            "chronotranscriber.sources.images._decode_with_ffmpeg", decode
        )
        payload = FolderPages(folder, IMG_CFG, "openai").build_payload(0)

        assert calls == [folder / "page_7.jp2"]
        assert payload.image_name == "page_7.jp2_pre_processed.jpg"
        assert payload.source_file == str(folder / "page_7.jp2")
        assert payload.number == 7

    def test_the_ffmpeg_decoder_runs_the_configured_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        monkeypatch.setattr(ffmpeg, "_FFMPEG_CMD", "/opt/ffmpeg/bin/ffmpeg")
        monkeypatch.setattr(images.subprocess, "run", run)
        images._decode_with_ffmpeg(tmp_path / "page.jp2", tmp_path / "page.png")

        assert [argv[0] for argv in calls] == ["/opt/ffmpeg/bin/ffmpeg"]


class TestRenderErrors:
    async def test_an_oversized_page_becomes_an_error_page(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        Image.new("L", (40, 60), 200).save(folder / "page_1.png")
        noise = random.Random(7).randbytes(300 * 300 * 3)
        Image.frombytes("RGB", (300, 300), noise).save(folder / "page_2.png")
        cfg = {
            "payload_format": "jpeg",
            "jpeg_quality": 95,
            "grayscale_conversion": False,
            "max_image_bytes": 20000,
            "resize_profile": "none",
        }

        pages = await collect(FolderPages(folder, cfg, "custom"))

        assert [p.render_error is None for p in pages] == [True, False]
        failed = pages[1]
        assert failed.image_name == "page_2.png_pre_processed.jpg"
        assert failed.source_file == str(folder / "page_2.png")
        assert "max_image_bytes" in failed.provenance()["render_error"]
        transcriber = MagicMock()
        transcriber.transcribe_image_from_base64 = AsyncMock()
        result = await transcribe_payload(failed, transcriber)
        assert result[2] == f"[transcription error: {failed.image_name}]"
        transcriber.transcribe_image_from_base64.assert_not_called()

    async def test_too_many_failures_fail_the_item(self, tmp_path: Path) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        for i in range(3):
            (folder / f"page_{i}.png").write_bytes(b"broken")

        with pytest.raises(RuntimeError, match="3/3 pages"):
            await collect(FolderPages(folder, IMG_CFG, "openai"))


class TestImageFiles:
    def test_png_is_encoded_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.png"
        Image.new("RGB", (1, 1), (255, 0, 0)).save(path, format="PNG")
        data, mime_type = encode_image_to_base64(path)
        assert mime_type == "image/png"
        assert base64.b64decode(data) == path.read_bytes()

    def test_mime_type_follows_the_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.jpg"
        Image.new("RGB", (2, 2), (0, 255, 0)).save(path, format="JPEG")
        assert encode_image_to_base64(path)[1] == "image/jpeg"

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "file.xyz"
        path.write_bytes(b"nothing")
        with pytest.raises(ValueError, match="Unsupported image format"):
            encode_image_to_base64(path)
