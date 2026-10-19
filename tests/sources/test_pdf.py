"""PDF source: processor lifecycle, native-text detection and extraction, and the
pixel-budget clamp on the render resolution.

The tests write tiny PDFs with PyMuPDF, so the PDF library is not mocked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import fitz
import pytest

from chronotranscriber.sources.pdf import (
    PDFProcessor,
    _get_effective_dpi,
    native_extract_pdf_text,
)


def _write_text_pdf(path: Path, text: str = "Hello PDF") -> None:
    """Write a minimal searchable PDF containing *text* to *path*."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=300, height=400)
        page.insert_text((72, 72), text)
        doc.save(str(path))
    finally:
        doc.close()


def _write_image_only_pdf(path: Path) -> None:
    """Write a PDF with a blank page and no text layer."""
    doc = fitz.open()
    try:
        doc.new_page(width=200, height=200)  # no insert_text call
        doc.save(str(path))
    finally:
        doc.close()


class TestPDFProcessorContextManager:
    """Context manager opens the PDF on enter and closes it on exit."""

    @pytest.mark.unit
    def test_enter_opens_document(self, tmp_path: Path) -> None:
        pdf = tmp_path / "sample.pdf"
        _write_text_pdf(pdf)

        with PDFProcessor(pdf) as proc:
            assert proc.doc is not None
            assert proc.doc.page_count == 1

    @pytest.mark.unit
    def test_exit_closes_document(self, tmp_path: Path) -> None:
        pdf = tmp_path / "sample.pdf"
        _write_text_pdf(pdf)

        proc = PDFProcessor(pdf)
        with proc:
            assert proc.doc is not None
        # After exit, doc should be set back to None
        assert proc.doc is None

    @pytest.mark.unit
    def test_close_pdf_is_idempotent(self, tmp_path: Path) -> None:
        """Calling close_pdf on an already-closed processor should not raise."""
        pdf = tmp_path / "sample.pdf"
        _write_text_pdf(pdf)

        proc = PDFProcessor(pdf)
        proc.open_pdf()
        proc.close_pdf()
        # Second close should be a no-op
        proc.close_pdf()
        assert proc.doc is None


class TestIsNativePDF:
    @pytest.mark.unit
    def test_returns_true_for_searchable_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "searchable.pdf"
        _write_text_pdf(pdf, "Some searchable text")

        assert PDFProcessor(pdf).is_native_pdf() is True

    @pytest.mark.unit
    def test_returns_false_for_image_only_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "blank.pdf"
        _write_image_only_pdf(pdf)

        assert PDFProcessor(pdf).is_native_pdf() is False

    @pytest.mark.unit
    def test_returns_false_on_open_error(self, tmp_path: Path) -> None:
        """An unreadable PDF path returns False rather than propagating."""
        missing = tmp_path / "does_not_exist.pdf"
        assert PDFProcessor(missing).is_native_pdf() is False


class TestNativeExtractPDFText:
    @pytest.mark.unit
    def test_returns_text_for_searchable_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "searchable.pdf"
        _write_text_pdf(pdf, "Native content")

        text = native_extract_pdf_text(pdf)
        assert "Native content" in text

    @pytest.mark.unit
    def test_returns_empty_for_image_only_pdf(self, tmp_path: Path) -> None:
        pdf = tmp_path / "blank.pdf"
        _write_image_only_pdf(pdf)

        text = native_extract_pdf_text(pdf)
        assert text.strip() == ""

    @pytest.mark.unit
    def test_page_indices_filter(self, tmp_path: Path) -> None:
        """When page_indices is given, only those pages' text is returned."""
        pdf = tmp_path / "two_pages.pdf"
        doc = fitz.open()
        try:
            page1 = doc.new_page(width=300, height=300)
            page1.insert_text((72, 72), "PageOneText")
            page2 = doc.new_page(width=300, height=300)
            page2.insert_text((72, 72), "PageTwoText")
            doc.save(str(pdf))
        finally:
            doc.close()

        only_second = native_extract_pdf_text(pdf, page_indices=[1])
        assert "PageTwoText" in only_second
        assert "PageOneText" not in only_second


class TestPageCount:
    @pytest.mark.unit
    def test_page_count_closes_the_document(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        _write_text_pdf(pdf)
        proc = PDFProcessor(pdf)
        assert proc.page_count() >= 1
        assert proc.doc is None


class TestTesseractPreprocessing:
    @pytest.mark.unit
    def test_a_failed_page_is_not_replaced_by_a_stale_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = tmp_path / "doc.pdf"
        doc = fitz.open()
        try:
            for _ in range(3):
                doc.new_page(width=72, height=72)
            doc.save(str(pdf))
        finally:
            doc.close()
        out = tmp_path / "work"
        out.mkdir()
        for n in (2, 3):  # left behind by another PDF or an earlier run
            (out / f"page_{n:04d}_tess_preprocessed.png").write_bytes(b"stale")

        def fake_page(
            _pdf: Path, page_index: int, _dpi: int, out_path: Path, *_: Any
        ) -> bool:
            if page_index == 1:
                raise RuntimeError("render failed")
            if page_index == 2:
                return False
            out_path.write_bytes(b"page")
            return True

        monkeypatch.setattr(
            PDFProcessor, "_process_single_page_tesseract", staticmethod(fake_page)
        )
        files = asyncio.run(PDFProcessor(pdf).process_images_for_tesseract(out, 300))

        assert [p.name for p in files] == ["page_0001_tess_preprocessed.png"]


class TestGetEffectiveDPI:
    @pytest.mark.unit
    def test_returns_unchanged_when_no_budget(self, tmp_path: Path) -> None:
        pdf = tmp_path / "p.pdf"
        _write_text_pdf(pdf)
        with fitz.open(str(pdf)) as doc:
            page = doc[0]
            assert _get_effective_dpi(page, 300, 0) == 300

    @pytest.mark.unit
    def test_returns_unchanged_when_under_budget(self, tmp_path: Path) -> None:
        """A small page well under the pixel budget keeps its target DPI."""
        pdf = tmp_path / "small.pdf"
        doc = fitz.open()
        try:
            doc.new_page(width=72, height=72)  # 1 inch x 1 inch
            doc.save(str(pdf))
        finally:
            doc.close()

        with fitz.open(str(pdf)) as d:
            page = d[0]
            # At 300 DPI this is 300x300 = 90_000 pixels, well under 10 MP
            assert _get_effective_dpi(page, 300, 10_000_000) == 300

    @pytest.mark.unit
    def test_clamps_dpi_when_over_budget(self, tmp_path: Path) -> None:
        """A large page above the pixel budget gets a reduced DPI."""
        pdf = tmp_path / "large.pdf"
        doc = fitz.open()
        try:
            # Create a large page: 20x20 inches at 300 DPI -> 36 MP
            doc.new_page(width=72 * 20, height=72 * 20)
            doc.save(str(pdf))
        finally:
            doc.close()

        with fitz.open(str(pdf)) as d:
            page = d[0]
            budget = 4_000_000  # cap at 4 MP
            effective = _get_effective_dpi(page, 300, budget)
            assert effective < 300
            # Verify the rendered pixel count really is within budget
            rendered_pixels = (page.rect.width / 72 * effective) * (
                page.rect.height / 72 * effective
            )
            assert rendered_pixels <= budget * 1.01  # tolerate tiny rounding
