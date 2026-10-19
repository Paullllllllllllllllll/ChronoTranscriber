"""Transcription methods behind one interface.

Each method (LLM, Tesseract, native text, Whisper, audio API) implements
:class:`Transcriber`: the run opens it once, hands it the items routed to
it and closes it. :data:`VALID_PAIRS` lists the methods each source type
takes; :mod:`chronotranscriber.run` routes every item by its pair.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from chronotranscriber.common.selection import PageRange
from chronotranscriber.common.store import Store
from chronotranscriber.events import PageDone, RunEvents
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.methods.units import DocumentHandle
from chronotranscriber.output.paths import ItemPaths, item_paths
from chronotranscriber.output.render import text_formats
from chronotranscriber.settings import Settings
from chronotranscriber.spec import METHODS_BY_TYPE, RunSpec

__all__ = [
    "VALID_PAIRS",
    "BatchSubmitter",
    "Item",
    "Job",
    "NothingSelectedError",
    "PageCounter",
    "SubmitOutcome",
    "Transcriber",
]

VALID_PAIRS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {kind: methods for kind, methods in METHODS_BY_TYPE.items() if kind != "auto"}
)
"""The methods each source type can be transcribed with."""


class NothingSelectedError(Exception):
    """``--pages`` selects no unit of the item; the message names the range.

    The run core skips the item: nothing is reset, called or rendered.
    """


class SubmitOutcome(Protocol):
    """The outcome of a batch submission.

    *status* is ``submitted`` (the units wait for ``batch check``),
    ``refused`` (the provider certainly made no batch; the units stay
    pending) or ``unresolved`` (the create's outcome is unknown; the units
    stay blocked). *detail* says why a submission was not accepted.
    """

    @property
    def status(self) -> str: ...

    @property
    def detail(self) -> str | None: ...


class BatchSubmitter(Protocol):
    """Submit one document's page payloads to the provider's batch API."""

    async def __call__(
        self,
        payloads: Sequence[Any],
        *,
        document: DocumentHandle,
        spec: RunSpec,
        settings: Settings,
        schema_path: Path | None,
        context: str | None = None,
        context_image: Path | None = None,
    ) -> SubmitOutcome: ...


@dataclass(frozen=True)
class Job:
    """What the items of one run share.

    *output_dir* None writes beside each input; *input_root* is the run's
    input, against which hash keys and the mirror layout are taken.
    *schema_path* is the resolved schema of LLM items. *batch* submits LLM
    items to the provider's batch API through *submit*. *store* is the
    run's open store; the run core sets it and *submit*.
    """

    spec: RunSpec
    settings: Settings
    events: RunEvents
    env: CallEnv
    output_dir: Path | None
    input_root: Path
    schema_path: Path | None = None
    batch: bool = False
    store: Store | None = None
    submit: BatchSubmitter | None = None

    @property
    def force(self) -> bool:
        """Whether earlier results are discarded instead of resumed."""
        return self.spec.run.force

    @property
    def retry_errors(self) -> bool:
        """Whether resume transcribes failed pages again."""
        return self.spec.run.retry_errors

    @property
    def formats(self) -> tuple[str, ...]:
        """The text formats the run writes; the first names the main output."""
        return text_formats(self.spec.output.formats)

    @property
    def keep_preprocessed(self) -> bool:
        """Whether preprocessed images and audio chunks are kept."""
        return self.spec.run.keep_working_files

    def locate(self, item: Path, *, ebook: bool = False) -> ItemPaths:
        """Return the paths of *item* without creating anything."""
        return item_paths(
            item,
            output_dir=self.output_dir,
            layout=self.spec.output.layout,
            input_root=self.input_root,
            output_format=self.formats[0],
            mirror=not ebook,
        )

    @contextlib.contextmanager
    def working_dir(self, item: Path) -> Iterator[Path]:
        """Yield a folder for the transient files of *item*.

        The folder is a temporary directory, removed on exit; with
        ``--keep-working-files`` it is the item's folder, created and kept.
        """
        if self.keep_preprocessed:
            folder = self.locate(item).folder
            folder.mkdir(parents=True, exist_ok=True)
            yield folder
            return
        with tempfile.TemporaryDirectory(
            prefix="chronotranscriber-", ignore_cleanup_errors=True
        ) as temp:
            yield Path(temp)


@dataclass(frozen=True)
class Item:
    """One item as a method transcribes it.

    *pages* selects pages, images or audio chunks; *context* and
    *context_image* are the item's resolved prompt context (LLM items).
    *document* is the item's store document, set by the run core.
    """

    path: Path
    source: str
    method: str
    pages: PageRange | None = None
    context: str | None = None
    context_image: Path | None = None
    document: DocumentHandle | None = None

    @property
    def name(self) -> str:
        """The item's file or folder name."""
        return self.path.name


class Transcriber(Protocol):
    """One transcription method of a run.

    ``open`` builds what every item shares (a model client) before the
    first item; ``transcribe`` writes one item's outputs and returns their
    paths, raising when the item fails; ``close`` releases the resources.
    """

    async def open(self) -> None: ...

    async def transcribe(self, item: Item) -> list[Path]: ...

    async def close(self) -> None: ...


@dataclass
class PageCounter:
    """Report each finished unit of an item as a :class:`PageDone` event."""

    events: RunEvents
    item: str
    total: int
    unit: str = "page"
    already_complete: int = 0
    done: int = field(default=0, init=False)

    def __call__(self, index: int, failed: bool) -> None:
        self.done += 1
        self.events.page_done(
            PageDone(
                item=self.item,
                page=index,
                status="failed" if failed else "ok",
                done=self.done,
                total=self.total,
                already_complete=self.already_complete,
                unit=self.unit,
            )
        )
