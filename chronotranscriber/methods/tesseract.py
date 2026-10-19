"""The Tesseract method: local OCR of PDF pages and folder images.

Every PDF page or folder image is one ``page`` unit of the item's store
document, planned as ``pending`` under its absolute index. Only the selected
units still to do are rendered (PDF) or read (folder), preprocessed for OCR
into a ``preprocessed_images_tesseract`` folder of the item's working
directory and transcribed concurrently; each result is committed as its
unit. The working directory is transient unless ``--keep-working-files`` is
set. The run core renders the item's files.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path

from chronotranscriber.common.images import check_failure_rate
from chronotranscriber.common.store import PlannedUnit
from chronotranscriber.images.pipeline import (
    ImageProcessor,
    TesseractSettings,
    tesseract_settings,
)
from chronotranscriber.images.tesseract_runtime import (
    TESSERACT_MISSING,
    ensure_tesseract_available,
    perform_ocr,
    use_tesseract_command,
)
from chronotranscriber.methods import Item, Job, NothingSelectedError, PageCounter
from chronotranscriber.methods.units import (
    DocumentHandle,
    PayloadResult,
    run_unit_files,
)
from chronotranscriber.output.paths import natural_sort_key
from chronotranscriber.output.store import KIND_PAGE
from chronotranscriber.sources.pdf import PDFProcessor

__all__ = [
    "MODEL",
    "PREPROCESSED_FOLDER",
    "PROVIDER",
    "TesseractMissingError",
    "TesseractTranscriber",
    "store_document",
]

logger = logging.getLogger(__name__)

PREPROCESSED_FOLDER = "preprocessed_images_tesseract"

PROVIDER = "local"
"""The provider stamped on the units of the local methods."""
MODEL = "tesseract"
"""The model stamped on Tesseract units."""


class TesseractMissingError(RuntimeError):
    """Tesseract is not installed or its configured executable does not run."""


def store_document(
    item: Item, kind: str, *, provider: str, model: str
) -> DocumentHandle:
    """Return the item's store document for units of *kind*.

    Raises:
        RuntimeError: The run core opened no store document for the item.
    """
    if item.document is None:
        raise RuntimeError(
            f"No store document is open for '{item.name}'; the run core leases "
            "one before a method runs."
        )
    return dataclasses.replace(item.document, kind=kind, provider=provider, model=model)


async def units_to_do(
    job: Job, handle: DocumentHandle, selected: list[int]
) -> list[int]:
    """Return the selected units still to do, after ``--force`` resets them.

    ``--force`` deletes the units another method stored under another kind
    and returns every unit of the handle's kind to ``pending``; the
    responses stay. Failed units are to do with ``--retry-errors``.
    """
    if job.force:
        await handle.discard()
        await handle.reset()
    todo = set(handle.todo(retry_errors=job.retry_errors))
    return [index for index in selected if index in todo]


class TesseractTranscriber:
    """Transcribe PDFs and image folders with Tesseract."""

    def __init__(self, job: Job) -> None:
        self.job = job

    @property
    def settings(self) -> TesseractSettings:
        """The preprocessing of the run's ``--ocr-preprocess`` preset."""
        return tesseract_settings(
            self.job.spec.ocr.preprocess,
            self.job.settings.concurrency.image_processing,
        )

    async def open(self) -> None:
        """Use the settings' Tesseract executable; availability is checked
        per item."""
        use_tesseract_command(self.job.settings.tools.tesseract_cmd)

    async def close(self) -> None:
        """Nothing to close."""

    async def transcribe(self, item: Item) -> list[Path]:
        """Transcribe one PDF or image folder into its units; return []."""
        if item.source == "images":
            return await self._folder(item)
        indices = self.page_indices(item)
        self.require_engine()
        return await self.pdf(item, indices)

    @staticmethod
    def require_engine() -> None:
        """Raise :class:`TesseractMissingError` when Tesseract does not run."""
        if not ensure_tesseract_available():
            raise TesseractMissingError(TESSERACT_MISSING)

    def page_indices(self, item: Item) -> list[int] | None:
        """Return the selected page indices of a PDF, None for every page.

        Raises:
            NothingSelectedError: The range selects no page.
        """
        if item.pages is None:
            return None
        total = PDFProcessor(item.path).page_count()
        indices = item.pages.resolve(total)
        if not indices:
            raise NothingSelectedError(
                f"Page range '{item.pages.describe()}' yielded no pages for "
                f"'{item.name}' ({total} pages). Skipping."
            )
        return indices

    def _handle(self, item: Item) -> DocumentHandle:
        return store_document(item, KIND_PAGE, provider=PROVIDER, model=MODEL)

    async def pdf(self, item: Item, indices: list[int] | None) -> list[Path]:
        """OCR the selected pages of a PDF; the caller checks the engine."""
        handle = self._handle(item)
        settings = self.settings
        total = PDFProcessor(item.path).page_count()
        await handle.plan(
            [
                PlannedUnit(
                    index,
                    label=f"page_{index + 1:04d}_tess_preprocessed{settings.suffix}",
                    source_index=index,
                )
                for index in range(total)
            ]
        )
        selected = indices if indices is not None else list(range(total))
        work = await units_to_do(self.job, handle, selected)
        if not work:
            return []
        with self.job.working_dir(item.path) as base:
            folder = base / PREPROCESSED_FOLDER
            processor = PDFProcessor(item.path, settings)
            try:
                files = await processor.process_images_for_tesseract(
                    folder, settings.target_dpi, page_indices=work
                )
            finally:
                processor.close_pdf()
            check_failure_rate(item.name, len(work), len(work) - len(files))
            await self._ocr(
                item,
                handle,
                files,
                is_folder=False,
                already_complete=len(selected) - len(work),
            )
        return []

    async def _folder(self, item: Item) -> list[Path]:
        settings = self.settings
        names = ImageProcessor.tesseract_image_names(item.path, settings)
        selected = list(range(len(names)))
        if item.pages is not None:
            selected = item.pages.resolve(len(names))
            if not selected:
                raise NothingSelectedError(
                    f"Page range '{item.pages.describe()}' yielded no images for "
                    f"'{item.name}' ({len(names)} images). Skipping."
                )
        self.require_engine()
        if not names:
            self.job.events.warning(f"No images found or processed in {item.path}.")
            return []
        handle = self._handle(item)
        await handle.plan(
            [
                PlannedUnit(index, label=name, source_index=name)
                for index, name in enumerate(names)
            ]
        )
        work = await units_to_do(self.job, handle, selected)
        if not work:
            return []
        with self.job.working_dir(item.path) as base:
            files, order = await asyncio.to_thread(
                ImageProcessor.process_and_save_images_for_tesseract,
                item.path,
                base / PREPROCESSED_FOLDER,
                page_indices=work,
                settings=settings,
            )
            check_failure_rate(item.name, len(order), len(order) - len(files))
            if not files:
                self.job.events.warning(f"No images found or processed in {item.path}.")
                return []
            files.sort(key=lambda path: natural_sort_key(path.name))
            await self._ocr(
                item,
                handle,
                files,
                is_folder=True,
                order=order,
                already_complete=len(selected) - len(work),
            )
        return []

    async def _ocr(
        self,
        item: Item,
        handle: DocumentHandle,
        files: list[Path],
        *,
        is_folder: bool,
        order: dict[str, int] | None = None,
        already_complete: int = 0,
    ) -> None:
        counter = PageCounter(
            self.job.events,
            item.name,
            total=len(files),
            unit="image" if is_folder else "page",
            already_complete=already_complete,
        )
        config = self.job.spec.ocr.config

        async def ocr(path: Path, index: int) -> PayloadResult:
            # OCR blocks (a subprocess and PIL); run it off the event loop.
            text = await asyncio.to_thread(perform_ocr, path, config)
            return (str(path), path.name, text, None, index)

        await run_unit_files(
            handle,
            files,
            ocr,
            order=order,
            concurrency=self.job.spec.run.concurrency,
            delay=self.job.settings.concurrency.delay_between_tasks,
            on_unit=counter,
        )
