"""The txt and md views of a document, rendered from the store.

A document's done and failed units are written in order; pending units (not
selected by a ``--pages`` run) are left out. Each unit becomes one page line
(:func:`~chronotranscriber.llm.response_parsing.format_page_line`): the text
as stored, or for a placeholder its page image name and the placeholder, so
the ``.txt`` keeps lines such as ``page_0002_pre_processed.jpg: [transcription
error]``. The ``text`` units of a native item form one line together: their
contents in order, joined directly for a PDF and by a blank line over the
non-empty ones for an EPUB or MOBI. The txt format joins the lines with a
newline, the md format with a blank line, and every file ends with exactly
one newline.

:func:`render_item` writes the files through the store's render driver: each
file is replaced atomically under the document's lease and recorded in
``outputs``; a file changed outside the tool is kept unless forced, and a
file that cannot be replaced is marked stale while the units stay stored.
:func:`render_database` renders every document of a database that way (the
``render`` subcommand) and :func:`preview_database` reports what it would
write without writing.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from chronotranscriber.common.sqlite import read
from chronotranscriber.common.store import (
    DocumentRow,
    Lease,
    LeaseLostError,
    OutputRow,
    Renderer,
    RenderReport,
    RenderTarget,
    Store,
    file_sha256,
)
from chronotranscriber.llm.response_parsing import format_page_line
from chronotranscriber.output.store import (
    DATABASE_NAME,
    KIND_TEXT,
    open_store,
    read_only_connection,
    read_store,
)

__all__ = [
    "FAILED_PLACEHOLDER",
    "RENDERERS",
    "TEXT_FORMATS",
    "DocumentRender",
    "document_targets",
    "find_databases",
    "format_path",
    "preview_database",
    "render_database",
    "render_item",
    "render_md",
    "render_targets",
    "render_txt",
    "rendered_units",
    "text_formats",
]

TEXT_FORMATS: Final = ("txt", "md")
FAILED_PLACEHOLDER: Final = "[transcription error]"
"""The page text of a failed unit that stored no placeholder of its own."""

_SEPARATORS: Final[Mapping[str, str]] = {"txt": "\n", "md": "\n\n"}
_EBOOK_KINDS: Final = ("epub", "mobi")
_UNITS = (
    "SELECT kind, idx, status, label, content FROM units "
    "WHERE doc_id = ? AND status != 'pending' ORDER BY idx, kind"
)
_COUNT = "SELECT count(*) FROM units WHERE doc_id = ? AND status != 'pending'"


def text_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Return the text formats among *formats*; txt when there is none."""
    chosen = tuple(name for name in formats if name in TEXT_FORMATS)
    return chosen or ("txt",)


def format_path(output: Path, output_format: str) -> Path:
    """Return *output* with the suffix of *output_format* (txt or md)."""
    return output.with_suffix(f".{output_format}")


def _text_block(connection: sqlite3.Connection, doc_id: int, texts: list[str]) -> str:
    """Join the ``text`` units of a document into its one text."""
    row = connection.execute(
        "SELECT kind FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if row is not None and row[0] in _EBOOK_KINDS:
        return "\n\n".join(text for text in texts if text)
    return "".join(texts)


def _page_lines(connection: sqlite3.Connection, doc_id: int) -> list[str]:
    lines: list[str] = []
    texts: list[str] = []
    text_line: int | None = None
    for kind, idx, status, label, content in connection.execute(_UNITS, (doc_id,)):
        text = content
        if text is None:
            text = FAILED_PLACEHOLDER if status == "failed" else ""
        if kind == KIND_TEXT:
            if text_line is None:
                text_line = len(lines)
                lines.append("")
            texts.append(str(text))
            continue
        lines.append(format_page_line(str(text), int(idx) + 1, label))
    if text_line is not None:
        block = _text_block(connection, doc_id, texts)
        lines[text_line] = format_page_line(block, None, None)
    return lines


def _render(connection: sqlite3.Connection, doc_id: int, output_format: str) -> str:
    body = _SEPARATORS[output_format].join(_page_lines(connection, doc_id))
    return body.rstrip("\n") + "\n"


def render_txt(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the txt view of document *doc_id*: page lines joined by newline."""
    return _render(connection, doc_id, "txt")


def render_md(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the md view of document *doc_id*: page lines joined by a blank line."""
    return _render(connection, doc_id, "md")


RENDERERS: Final[Mapping[str, Renderer]] = {"txt": render_txt, "md": render_md}


def render_targets(output: Path, formats: Sequence[str]) -> list[RenderTarget]:
    """Return the render targets of an item, one per text format in *formats*.

    *output* is the item's output path; each format takes its own suffix.
    """
    return [
        RenderTarget(format_path(output, name), name, RENDERERS[name])
        for name in text_formats(formats)
    ]


def rendered_units(store: Store, doc_id: int) -> int:
    """Return the number of a document's units that a rendering shows."""
    (count,) = read(
        store.path, lambda connection: connection.execute(_COUNT, (doc_id,)).fetchone()
    )
    return int(count)


async def render_item(
    store: Store,
    document: int | str,
    formats: Sequence[str],
    output: Path,
    lease: Lease,
    *,
    force: bool = False,
) -> RenderReport:
    """Render document *document* (its id or name) into its text formats.

    *output* is the item's output path and *lease* the document's lease of
    this run. A document without a done or failed unit gets no file. A file
    that cannot be replaced is marked stale and reported in the warnings; a
    later call replaces it.
    """
    doc_id = document if isinstance(document, int) else _doc_id(store, document)
    if doc_id != lease.doc_id:
        raise ValueError(f"The lease names document {lease.doc_id}, not {document!r}.")
    if not rendered_units(store, doc_id):
        return RenderReport()
    return await store.render(lease, render_targets(output, formats), force=force)


def _doc_id(store: Store, name: str) -> int:
    row = store.document(name)
    if row is None:
        raise KeyError(f"No document named {name!r} in {store.path}")
    return row.doc_id


@dataclass
class DocumentRender:
    """What a render did, or would do, with one document's files.

    *skipped* says why the document was not rendered (its lease is held by
    another run, or it was lost); *kept* lists files changed outside the
    tool and left alone, *stale* files that could not be replaced.
    """

    database: Path
    name: str
    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)
    stale: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: str | None = None

    @property
    def failed(self) -> bool:
        """Whether a file could not be rendered."""
        return bool(self.stale) or self.skipped is not None


def find_databases(path: Path) -> list[Path]:
    """Return the tool's databases: *path* itself when it is a file, else below it."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(found for found in path.rglob(DATABASE_NAME) if found.is_file())


def _registered_path(root: Path, row: OutputRow) -> Path:
    path = Path(row.path)
    return path if path.is_absolute() else root / path


def document_targets(
    root: Path,
    name: str,
    registered: Sequence[OutputRow],
    formats: Sequence[str] | None = None,
) -> list[RenderTarget]:
    """Return the render targets of document *name* of output root *root*.

    Without *formats* the document's registered text formats are rendered,
    txt when none is registered. A format keeps its registered path; another
    takes the registered base name, else ``<root>/<name>``, with its suffix.
    """
    paths = {
        row.format: _registered_path(root, row)
        for row in registered
        if row.format in TEXT_FORMATS
    }
    chosen = text_formats(formats) if formats else tuple(sorted(paths)) or ("txt",)
    base = next(iter(paths.values()), None)
    output = base if base is not None else Path(f"{root / name}.txt")
    return [
        RenderTarget(paths.get(fmt) or format_path(output, fmt), fmt, RENDERERS[fmt])
        for fmt in chosen
    ]


def _preview_document(
    connection: sqlite3.Connection,
    database: Path,
    document: DocumentRow,
    registered: Sequence[OutputRow],
    formats: Sequence[str] | None,
    force: bool,
) -> DocumentRender:
    result = DocumentRender(database, document.name)
    recorded = {
        _registered_path(database.parent, row): row.sha256 for row in registered
    }
    for target in document_targets(database.parent, document.name, registered, formats):
        text = target.renderer(connection, document.doc_id)
        data = text if isinstance(text, bytes) else text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        try:
            current = file_sha256(target.path)
        except OSError as exc:
            result.stale.append(target.path)
            result.warnings.append(f"Could not read {target.path}: {exc}")
            continue
        if current == digest:
            result.unchanged.append(target.path)
        elif current is not None and current != recorded.get(target.path) and not force:
            result.kept.append(target.path)
        else:
            result.written.append(target.path)
    return result


def _renderable_documents(root: Path) -> tuple[Store, list[DocumentRow]] | None:
    reader = read_store(root)
    if reader is None:
        return None
    documents = [
        doc for doc in reader.documents() if rendered_units(reader, doc.doc_id)
    ]
    return reader, documents


def preview_database(
    database: Path,
    *,
    formats: Sequence[str] | None = None,
    force: bool = False,
) -> list[DocumentRender]:
    """Return what :func:`render_database` would do; write nothing.

    A target that cannot be read is listed as stale with a warning.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    found = _renderable_documents(root)
    if found is None:
        return []
    reader, documents = found
    outputs = {doc.doc_id: reader.outputs(doc.doc_id) for doc in documents}
    with read_only_connection(root) as connection:
        return [
            _preview_document(
                connection, database, doc, outputs[doc.doc_id], formats, force
            )
            for doc in documents
        ]


async def _render_document(
    store: Store,
    document: DocumentRow,
    formats: Sequence[str] | None,
    force: bool,
) -> DocumentRender:
    result = DocumentRender(store.path, document.name)
    targets = document_targets(
        store.path.parent, document.name, store.outputs(document.doc_id), formats
    )
    lease = await store.acquire(document.doc_id)
    if lease is None:
        result.skipped = "another run holds its lease"
        await store.add_note(
            f"render skipped '{document.name}': another run holds its lease",
            doc_id=document.doc_id,
        )
        return result
    try:
        report = await store.render(lease, targets, force=force)
    except (LeaseLostError, OSError, sqlite3.Error) as exc:
        result.skipped = f"not rendered: {exc}"
        return result
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    result.written = report.written
    result.unchanged = report.unchanged
    result.kept = report.kept
    result.stale = report.stale
    result.warnings = report.warnings
    return result


async def render_database(
    database: Path,
    *,
    formats: Sequence[str] | None = None,
    force: bool = False,
) -> list[DocumentRender]:
    """Render every document of *database* from the store, without model calls.

    Each document is rendered under its lease; a document another run holds
    is skipped with a note. A file changed outside the tool is left alone
    unless *force*; a file that cannot be replaced stays stale for a later
    render. Documents without a done or failed unit are left out.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    found = _renderable_documents(root)
    if found is None or not found[1]:
        return []
    documents = found[1]
    results: list[DocumentRender] = []
    async with open_store(root) as store:
        await store.begin_run(
            "render",
            spec={
                "formats": {"value": list(formats or ()), "source": "flag"},
                "force": {"value": force, "source": "flag"},
            },
        )
        try:
            for document in documents:
                results.append(await _render_document(store, document, formats, force))
        finally:
            await store.end_run(1 if any(result.failed for result in results) else 0)
    return results
