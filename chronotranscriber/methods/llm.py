"""The LLM method: PDF pages and folder images sent to a vision model.

Every page of the item is a ``page`` unit of the item's store document,
planned as ``pending`` before the first call, so a ``--pages`` run leaves
the document partial. The pages still to do (pending, plus failed with
``--retry-errors``) are rendered and encoded in memory and transcribed
concurrently through
:class:`~chronotranscriber.llm.transcription.PageTranscriber`, each unit
committed to the store as it finishes; under ``--batch`` they are submitted
to the provider's batch API instead. The method writes no file: the run core
renders the item from the store.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from chronotranscriber.common.native import changed_image_settings
from chronotranscriber.common.store import PlannedUnit
from chronotranscriber.llm.transcription import ROLE, PageTranscriber
from chronotranscriber.methods import Item, Job, NothingSelectedError, PageCounter
from chronotranscriber.methods.pages import transcribe_payload
from chronotranscriber.methods.units import DocumentHandle, run_unit_stream
from chronotranscriber.output.store import image_settings_record
from chronotranscriber.sources.images import (
    FolderPages,
    PageImage,
    folder_image_name,
    list_folder_images,
    resolve_image_settings,
    stream_pages,
)
from chronotranscriber.sources.pdf import PDF_PAGE_NAME, PdfPages, PDFProcessor
from chronotranscriber.spec import RunSpec

__all__ = [
    "ROLE",
    "BatchSubmissionFailedError",
    "ImageSettingsChangedError",
    "LlmTranscriber",
    "image_record",
]

logger = logging.getLogger(__name__)


class ImageSettingsChangedError(RuntimeError):
    """The document's stored pages were produced with other image settings."""


class BatchSubmissionFailedError(RuntimeError):
    """A batch submission was refused without a fallback, or left unresolved."""


def image_record(
    spec: RunSpec, original_max_side_px: int | None = None
) -> dict[str, Any]:
    """Return the image-settings record of *spec*'s model and image options.

    *original_max_side_px* is the ``images`` setting of that name.

    Raises:
        ValueError: An image setting is invalid.
    """
    model = spec.model
    settings, model_type, _section = resolve_image_settings(
        model.provider, model.model, spec.images, original_max_side_px
    )
    return image_settings_record(settings, model_type)


def _recorded(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


class LlmTranscriber:
    """Transcribe PDFs and image folders with the run's vision model.

    In batch mode the pages go to the batch API. A client is opened there
    only with ``--sync-fallback``, so that the pages of a refused submission
    are transcribed synchronously.
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self._client: PageTranscriber | None = None

    async def open(self) -> None:
        """Build the transcription client; in batch mode only for the fallback."""
        if self.job.batch and not self.job.spec.run.sync_fallback:
            return
        spec = self.job.spec
        self._client = PageTranscriber(
            spec.model,
            self.job.env,
            schema_path=self.job.schema_path,
            prompt_path=spec.prompt.prompt,
            image_detail=spec.images.detail,
            role=ROLE,
        )

    async def close(self) -> None:
        """Close the client."""
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    async def transcribe(self, item: Item) -> list[Path]:
        """Transcribe one PDF or image folder into its store document.

        Returns no paths: the run core renders the item.

        Raises:
            RuntimeError: The run core opened no store document for the item.
            NothingSelectedError: The page range selects no page.
            ImageSettingsChangedError: The stored pages used other image
                settings.
            UnitsFailedError: A page failed or could not be stored.
        """
        if item.document is None:
            raise RuntimeError(f"No store document is open for '{item.name}'.")
        handle = item.document
        if self._client is not None:
            self._client.update_context(item.context)
            self._client.update_context_image(item.context_image)
        is_folder = item.source == "images"
        units = self._units(item.path, is_folder)
        if not units:
            noun = "images" if is_folder else "pages"
            self.job.events.warning(f"No {noun} found in '{item.name}'.")
            return []
        selected = list(range(len(units)))
        if item.pages is not None:
            selected = item.pages.resolve(len(units))
            if not selected:
                noun = "images" if is_folder else "pages"
                raise NothingSelectedError(
                    f"Page range '{item.pages.describe()}' yielded no {noun} for "
                    f"'{item.name}' ({len(units)} {noun}). Skipping."
                )
        model = self.job.spec.model
        img_cfg, model_type, section = resolve_image_settings(
            model.provider,
            model.model,
            self.job.spec.images,
            self.job.settings.images.original_max_side_px,
        )
        if self._client is not None:
            self._client.page_detail = str(img_cfg.get("resolved_detail") or "") or None
        await self._record_settings(handle, image_settings_record(img_cfg, model_type))
        await handle.plan(units)
        if self.job.force:
            await handle.discard()
            await handle.reset()
        todo = set(handle.todo(retry_errors=self.job.retry_errors))
        needed = [index for index in selected if index in todo]
        already = len(selected) - len(needed)
        if already:
            logger.info("Skipped %d pages already stored for %s", already, item.name)
        if not needed:
            return []

        async def source(indices: list[int]) -> AsyncIterator[PageImage]:
            pages_type = FolderPages if is_folder else PdfPages
            pages = pages_type(item.path, img_cfg, model_type, section=section)
            async for page in stream_pages(pages, indices):
                yield page

        if self.job.batch:
            payloads = [payload async for payload in source(needed)]
            if await self._submit(item, handle, payloads):
                return []
            # Parts accepted before the refused one stay held by their batch.
            todo = set(handle.todo(retry_errors=self.job.retry_errors))
            needed = [index for index in needed if index in todo]

        counter = PageCounter(
            self.job.events,
            item.name,
            total=len(needed),
            unit="image" if is_folder else "page",
            already_complete=already,
        )
        await run_unit_stream(
            handle,
            source(needed),
            self._client,
            transcribe_payload,
            concurrency=self.job.spec.run.concurrency,
            delay=self.job.settings.concurrency.delay_between_tasks,
            on_unit=counter,
        )
        return []

    @staticmethod
    def _units(path: Path, is_folder: bool) -> list[PlannedUnit]:
        """Return every page of the item as an expected unit."""
        if is_folder:
            return [
                PlannedUnit(idx, name, name)
                for idx, name in enumerate(
                    folder_image_name(image) for image in list_folder_images(path)
                )
            ]
        total = PDFProcessor(path).page_count()
        return [
            PlannedUnit(idx, PDF_PAGE_NAME.format(number=idx + 1), idx)
            for idx in range(total)
        ]

    async def _record_settings(
        self, handle: DocumentHandle, record: dict[str, Any]
    ) -> None:
        """Compare the document's image settings with *record*, then store it.

        A changed model alone passes; a changed user setting refuses
        the item while it holds stored pages, unless ``--force``.

        Raises:
            ImageSettingsChangedError: A user image setting changed.
        """
        row = handle.store.document(handle.doc_id)
        recorded = _recorded(row.image_settings) if row is not None else None
        stored = row is not None and (row.units_done + row.units_failed) > 0
        if recorded is not None and stored and not self.job.force:
            changed = changed_image_settings(recorded, record)
            if changed:
                raise ImageSettingsChangedError(
                    f"Image settings changed for '{handle.name}': "
                    f"{', '.join(changed)}. Skipping it; use --force to replace "
                    "the existing run."
                )
        if recorded != record:
            await handle.store.set_image_settings(handle.lease, record)

    async def _submit(
        self, item: Item, handle: DocumentHandle, payloads: list[PageImage]
    ) -> bool:
        """Submit the pages; False when they are to be transcribed synchronously.

        A refused submission fails the item unless ``--sync-fallback`` is
        set; an unresolved one always fails it, its pages left blocked.

        Raises:
            BatchSubmissionFailedError: The submission was not accepted.
        """
        submit = self.job.submit
        if submit is None:
            raise RuntimeError("The run core set no batch submitter.")
        fallback = self.job.spec.run.sync_fallback
        outcome = await submit(
            payloads,
            document=handle,
            spec=self.job.spec,
            settings=self.job.settings,
            schema_path=self.job.schema_path,
            context=item.context,
            context_image=item.context_image,
        )
        if outcome.status == "submitted":
            return True
        if outcome.status == "refused":
            detail = outcome.detail or "the provider refused the batch"
            if not fallback:
                raise BatchSubmissionFailedError(detail)
            self.job.events.warning(
                f"Batch submission failed for '{item.name}' ({detail}); falling "
                "back to synchronous processing (--sync-fallback enabled)."
            )
            return False
        detail = f" ({outcome.detail})" if outcome.detail else ""
        raise BatchSubmissionFailedError(
            f"The batch submission of '{item.name}' is unresolved{detail}: the "
            "provider may have created the batch. Its pages stay blocked; run "
            "'chronotranscriber batch check' to reconcile it."
        )
