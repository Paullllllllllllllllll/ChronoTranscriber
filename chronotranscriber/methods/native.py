"""The native method: the text layer of a PDF, EPUB or MOBI file.

Every page of a searchable PDF is one ``text`` unit of its store document
(``idx`` and ``source_index`` are the 0-based page index). An EPUB or MOBI
has its metadata header as unit 0 and section *i* as unit *i + 1*
(``source_index`` *i*). All units are planned as ``pending``; only the
selected units still to do are extracted, so a ``--pages`` run leaves the
document partial. The run core renders the units as one text. A PDF
without a text layer is handed to Tesseract instead, and fails when
Tesseract is missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from chronotranscriber.common.store import PlannedUnit
from chronotranscriber.llm.response_parsing import detect_transcription_cause
from chronotranscriber.methods import Item, Job, NothingSelectedError
from chronotranscriber.methods.tesseract import (
    PROVIDER,
    TesseractTranscriber,
    store_document,
    units_to_do,
)
from chronotranscriber.methods.units import DocumentHandle
from chronotranscriber.output.store import KIND_TEXT
from chronotranscriber.sources._text import metadata_header, normalize_text
from chronotranscriber.sources.epub import EPUBProcessor
from chronotranscriber.sources.mobi import MOBIProcessor
from chronotranscriber.sources.pdf import PDFProcessor, native_extract_pdf_pages

__all__ = ["MODEL", "NativeTranscriber"]

MODEL = "native"
"""The model stamped on native text units."""

_EBOOKS: dict[str, tuple[type[Any], str]] = {
    "epub": (EPUBProcessor, "EPUB"),
    "mobi": (MOBIProcessor, "MOBI"),
}


class NativeTranscriber:
    """Take the text of PDFs, EPUB and MOBI files as it is stored."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self._tesseract = TesseractTranscriber(job)

    async def open(self) -> None:
        """Nothing to open."""

    async def close(self) -> None:
        """Nothing to close."""

    async def transcribe(self, item: Item) -> list[Path]:
        """Store the text of one file as its units; return []."""
        if item.source in _EBOOKS:
            return await self._ebook(item)
        return await self._pdf(item)

    def _handle(self, item: Item) -> DocumentHandle:
        return store_document(item, KIND_TEXT, provider=PROVIDER, model=MODEL)

    async def _pdf(self, item: Item) -> list[Path]:
        indices = self._tesseract.page_indices(item)
        if not PDFProcessor(item.path).is_native_pdf():
            self.job.events.warning(
                f"PDF '{item.name}' is not searchable. Switching to tesseract method."
            )
            self._tesseract.require_engine()
            return await self._tesseract.pdf(item, indices)
        handle = self._handle(item)
        total = PDFProcessor(item.path).page_count()
        units = [PlannedUnit(index, source_index=index) for index in range(total)]
        selected = indices if indices is not None else list(range(total))
        work = await units_to_do(self.job, handle, selected, units)
        if not work:
            return []
        pages = await asyncio.to_thread(native_extract_pdf_pages, item.path, work)
        for index in work:
            if index in pages:
                await _write(handle, index, pages[index], source_index=index)
        return []

    async def _ebook(self, item: Item) -> list[Path]:
        processor_cls, label = _EBOOKS[item.source]
        pages = item.pages
        try:
            extraction = await asyncio.to_thread(processor_cls(item.path).extract_text)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract text from {label} '{item.name}': {exc}"
            ) from exc

        sections: list[str] = list(extraction.sections)
        keep = list(range(len(sections)))
        if pages is not None:
            keep = pages.resolve(len(sections))
            if not keep:
                raise NothingSelectedError(
                    f"Page range '{pages.describe()}' selected no {label} sections "
                    f"for '{item.name}' ({len(sections)} section(s)). Skipping."
                )

        handle = self._handle(item)
        units = [PlannedUnit(0)] + [
            PlannedUnit(i + 1, source_index=i) for i in range(len(sections))
        ]
        work = await units_to_do(self.job, handle, [0] + [i + 1 for i in keep], units)
        if not work:
            return []
        header = metadata_header(extraction.title, extraction.authors)
        await _write(handle, 0, header)
        for index in work:
            if index > 0:
                text = normalize_text(sections[index - 1])
                await _write(handle, index, text, source_index=index - 1)
        return []


async def _write(
    handle: DocumentHandle,
    index: int,
    text: str,
    *,
    source_index: int | None = None,
) -> None:
    """Commit *text* as ``text`` unit *index*."""
    cause = detect_transcription_cause(text)
    await handle.write(
        index,
        status="done",
        content=text,
        outcome="ok" if cause == "api_error" else cause,
        source_index=source_index,
    )
