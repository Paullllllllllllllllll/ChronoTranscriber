"""ChronoTranscriber's store: its schema component, read views and resume.

Every output root (the output folder, or the input's folder when writing
beside the input) holds one ``chronotranscriber.sqlite``. Its core tables come
from :mod:`chronotranscriber.common.store`; this module adds the tool's
component (the ``page_types`` table of the decision pre-pass and the views
below) and registers the shared ``classifications`` component of
:mod:`chronotranscriber.common.classify_store` beside it, whose rows the
``classify`` commands write.

A document is one input item, named by its rendered output path relative to
the output root, without suffix (``doc-139d544b/doc``). Its units are:

- ``page``: a PDF page or a folder image; ``idx`` is the page's position in
  the item, ``label`` the page image name and ``source_index`` the PDF page
  index or the image file name;
- ``chunk``: an audio chunk; ``label`` is the chunk file name and
  ``source_index`` its time range in seconds (``start-end``);
- ``text``: native text. For a PDF, one unit per page (``idx`` and
  ``source_index`` are the 0-based page index); for an EPUB or MOBI, the
  metadata header lines as ``idx`` 0 and section *i* as ``idx`` *i + 1*
  (``source_index`` *i*).

A failed unit keeps its placeholder text (``[transcription error]``) as its
content, as the rendered files show it.

Read contract
=============

Outside code reads the store only through these views; the tables behind
them may change between versions, the views stay stable.

``page_text``
    One row per unit, ordered by document and ``order_index``: ``document``
    (the document name), ``kind`` (page, chunk or text), ``page_index`` (the
    0-based PDF page of a page unit, else NULL), ``image_name`` (page image
    or chunk file name), ``order_index`` (position in the item; the page or
    section position of a text unit), ``text``, ``status``
    (pending, done or failed), ``model``, ``usage`` (JSON token counts),
    ``outcome`` (ok, no_text, not_possible or failed), ``error``,
    ``provider``, ``run_id`` and ``page_type`` (the type the decision
    pre-pass gave a page unit, else NULL). Units a ``--pages`` run did not
    select are pending, with NULL text. A page the pre-pass skipped under
    ``--decisions apply`` is done with outcome ``skipped_no_text`` and a
    ``[No transcribable text: <type>]`` placeholder.

``document_status``
    One row per document: ``document``, ``kind`` (pdf, images, epub, mobi or
    audio), ``input_path``, ``status`` (pending, partial, complete or
    failed), ``units_total``, ``units_done``, ``units_failed``,
    ``units_pending``, ``last_run_id`` and ``updated_at``.

``page_attempts``
    One row per provider attempt in the order they were made: ``document``,
    ``kind``, ``order_index``, ``response_id``, ``run_id``, ``provider``,
    ``model``, ``attempt``, ``submission_id``, ``custom_id``, ``raw`` (the
    response body as received), ``raw_kind``, ``usage``, ``error``,
    ``http_status`` and ``created_at``.

``classifications``
    The latest answer per document, unit, classifier and question, as
    :mod:`chronotranscriber.common.classify_store` documents it. A page row
    joins ``page_text`` on ``document`` and ``ordinal`` = ``order_index``,
    whatever method transcribed the page; its ``locator`` is the page-image
    name, for information. A document row has the locator ``document``.

Open the database read-only and close it after the query, so that a running
transcription never waits on the reader::

    import sqlite3
    from pathlib import Path

    database = Path("out") / "chronotranscriber.sqlite"
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        rows = connection.execute(
            "SELECT order_index, image_name, status, text FROM page_text "
            "WHERE document = ? ORDER BY order_index",
            ("doc-139d544b/doc",),
        ).fetchall()
    finally:
        connection.close()
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from chronotranscriber.common.classify_store import COMPONENT as CLASSIFICATIONS
from chronotranscriber.common.classify_store import VIEW as CLASSIFICATIONS_VIEW
from chronotranscriber.common.native import changed_image_settings
from chronotranscriber.common.sqlite import read, read_only
from chronotranscriber.common.store import (
    BLOCKING_SUBMISSIONS,
    Component,
    PlannedUnit,
    SchemaState,
    Store,
    SubmissionStateError,
    UnitKey,
    WriterOptions,
    check_schema,
    refresh_documents,
)

__all__ = [
    "COMPONENT",
    "DATABASE_NAME",
    "KIND_CHUNK",
    "KIND_PAGE",
    "KIND_TEXT",
    "TOOL",
    "UNIT_KINDS",
    "VIEWS",
    "Classification",
    "ItemState",
    "changed_input",
    "classify",
    "database_path",
    "document_name",
    "forced_replacement",
    "identity_write",
    "image_settings_record",
    "lease_reason",
    "open_store",
    "page_type_rows",
    "read_only_connection",
    "read_store",
    "tool_version",
]

TOOL: Final = "chronotranscriber"
DATABASE_NAME: Final = "chronotranscriber.sqlite"

KIND_PAGE: Final = "page"
KIND_CHUNK: Final = "chunk"
KIND_TEXT: Final = "text"
UNIT_KINDS: Final = (KIND_PAGE, KIND_CHUNK, KIND_TEXT)

VIEWS: Final = ("page_text", "document_status", "page_attempts", CLASSIFICATIONS_VIEW)

_VIEWS_V1 = """
CREATE VIEW page_text AS
SELECT
    d.name AS document,
    u.kind AS kind,
    CASE WHEN u.kind = 'page' AND typeof(u.source_index) = 'integer'
        THEN u.source_index END AS page_index,
    u.label AS image_name,
    u.idx AS order_index,
    u.content AS text,
    u.status AS status,
    u.model AS model,
    u.usage AS usage,
    u.outcome AS outcome,
    u.error AS error,
    u.provider AS provider,
    u.run_id AS run_id
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
ORDER BY d.name, u.idx, u.kind;
CREATE VIEW document_status AS
SELECT
    name AS document,
    kind,
    input_path,
    status,
    units_total,
    units_done,
    units_failed,
    units_pending,
    last_run_id,
    updated_at
FROM documents
ORDER BY name;
CREATE VIEW page_attempts AS
SELECT
    d.name AS document,
    r.kind AS kind,
    r.idx AS order_index,
    r.response_id AS response_id,
    r.run_id AS run_id,
    r.provider AS provider,
    r.model AS model,
    r.attempt AS attempt,
    r.submission_id AS submission_id,
    r.custom_id AS custom_id,
    r.raw AS raw,
    r.raw_kind AS raw_kind,
    r.usage AS usage,
    r.error AS error,
    r.http_status AS http_status,
    r.created_at AS created_at
FROM responses AS r
LEFT JOIN documents AS d ON d.doc_id = r.doc_id
ORDER BY r.response_id;
"""

_PAGE_TYPES_V2 = """
CREATE TABLE page_types (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    idx INTEGER NOT NULL,
    page_type TEXT NOT NULL,
    probabilities TEXT NOT NULL,
    taxonomy TEXT NOT NULL,
    profile TEXT NOT NULL,
    model TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
DROP VIEW page_text;
CREATE VIEW page_text AS
SELECT
    d.name AS document,
    u.kind AS kind,
    CASE WHEN u.kind = 'page' AND typeof(u.source_index) = 'integer'
        THEN u.source_index END AS page_index,
    u.label AS image_name,
    u.idx AS order_index,
    u.content AS text,
    u.status AS status,
    u.model AS model,
    u.usage AS usage,
    u.outcome AS outcome,
    u.error AS error,
    u.provider AS provider,
    u.run_id AS run_id,
    pt.page_type AS page_type
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
LEFT JOIN page_types AS pt
    ON pt.doc_id = u.doc_id AND pt.idx = u.idx AND u.kind = 'page'
ORDER BY d.name, u.idx, u.kind;
"""


def _classifications_v3(connection: sqlite3.Connection) -> None:
    """Change nothing: version 3 marks a store that holds classifications.

    The ``classifications`` component adds the tables; this step makes a
    tool that knows version 2 only refuse such a store as newer.
    """


COMPONENT: Final = Component(TOOL, (_VIEWS_V1, _PAGE_TYPES_V2, _classifications_v3))
"""The tool's schema component: version 1 adds the views, version 2 the
``page_types`` table and the ``page_type`` column of ``page_text``, version 3
nothing of its own (it comes with the ``classifications`` component)."""

_COMPONENTS: Final = (COMPONENT, CLASSIFICATIONS)


def tool_version() -> str:
    """Return the installed version of the tool, or ``unknown``."""
    try:
        return metadata.version(TOOL)
    except metadata.PackageNotFoundError:
        return "unknown"


def database_path(root: Path) -> Path:
    """Return the database file of output root *root*."""
    return Path(root) / DATABASE_NAME


def open_store(
    root: Path,
    *,
    clock: Callable[[], float] = time.time,
    writer: WriterOptions | None = None,
) -> Store:
    """Return the store of output root *root*, not yet opened.

    ``async with open_store(root) as store`` opens its writer, creating the
    database when missing.
    """
    return Store(
        database_path(root),
        TOOL,
        tool_version=tool_version(),
        components=_COMPONENTS,
        clock=clock,
        writer=writer,
    )


def _schema(root: Path) -> SchemaState:
    """Return the schema state of *root*'s database; refuse it as the store does."""
    return check_schema(database_path(root), _COMPONENTS, tool=TOOL)


def read_store(root: Path) -> Store | None:
    """Return the store of *root* for reads only; None without a database.

    The store is not opened: its read methods use short read-only
    connections. A versionless, newer or unreadable database raises the
    :class:`~chronotranscriber.common.store.StoreError` that refuses it.
    """
    state = _schema(root)
    if not state.exists or state.empty:
        return None
    return Store(state.path, TOOL, components=_COMPONENTS)


@contextlib.contextmanager
def read_only_connection(root: Path) -> Iterator[sqlite3.Connection]:
    """Yield a ``mode=ro`` connection to *root*'s database; close it after.

    The schema is checked first, as for :func:`read_store`;
    ``FileNotFoundError`` without a database.
    """
    state = _schema(root)
    if not state.exists:
        raise FileNotFoundError(f"No database at {state.path}")
    with read_only(state.path) as connection:
        yield connection


def document_name(output: Path, root: Path) -> str:
    """Return the document name of an item: *output* relative to *root*.

    *output* is the item's rendered output path; the name has no suffix.
    Outside *root*, the file name without suffix is used.
    """
    unsuffixed = output.with_suffix("")
    try:
        return unsuffixed.relative_to(root).as_posix()
    except ValueError:
        return unsuffixed.name


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return *value* as JSON would store it (tuples become lists)."""
    plain = json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))
    return plain if isinstance(plain, dict) else {}


def image_settings_record(
    settings: Mapping[str, Any], model_type: str, max_pixels: int | None = None
) -> dict[str, Any]:
    """Return the image-settings record a document's pages are produced with.

    *settings* are the resolved image settings of the model's section,
    *model_type* its model family. The record holds the values the image
    settings fingerprint covers, in their stored JSON form.
    """
    record = dict(settings)
    if max_pixels is not None:
        record["max_pixels_per_page"] = max_pixels
    record["model_type"] = model_type
    return _plain(record)


class ItemState(StrEnum):
    """What the store holds for an item, as resume sees it."""

    NONE = "none"
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSTORED = "unstored"
    BLOCKED = "blocked"
    REFUSED = "refused"


@dataclass(frozen=True)
class Classification:
    """The resume state of one item.

    *units* are the units to process of a partial item. *render* asks a
    complete item to re-render its files without model calls; *missing*
    lists those files absent on disk. *note* records a model change, which
    resumes. *total* counts the stored units of a complete or partial item.
    *leased_by* names the run whose lease blocks the item; it is None for an
    item blocked by a batch submission.
    """

    state: ItemState
    name: str
    doc_id: int | None = None
    units: tuple[UnitKey, ...] = ()
    render: bool = False
    missing: tuple[Path, ...] = ()
    reason: str = ""
    note: str | None = None
    total: int = 0
    leased_by: str | None = None

    @property
    def indices(self) -> tuple[int, ...]:
        """Return the indices of the units to process."""
        return tuple(unit.idx for unit in self.units)


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _path_key(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _details(
    connection: sqlite3.Connection, doc_id: int
) -> tuple[list[str], dict[str, int], str | None]:
    (input_sha256,) = connection.execute(
        "SELECT input_sha256 FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    models = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT model FROM units WHERE doc_id = ? "
            "AND status != 'pending' ORDER BY model",
            (doc_id,),
        )
    ]
    outputs = {
        str(path): int(stale)
        for path, stale in connection.execute(
            "SELECT path, stale FROM outputs WHERE doc_id = ?", (doc_id,)
        )
    }
    return models, outputs, input_sha256


def _model_note(
    name: str,
    earlier: Sequence[str],
    model: str | None,
    recorded: Mapping[str, Any] | None,
) -> str:
    previous = ", ".join(earlier)
    if not previous and recorded is not None:
        previous = str(recorded.get("model_name") or "")
    return (
        f"'{name}' holds units transcribed with {previous or 'another model'}; "
        f"this run uses {model or 'another model'}."
    )


def changed_input(name: str, stored: str, current: str) -> str:
    """Return the refusal of item *name* whose input changed since it was stored.

    *stored* and *current* are the recorded and the current input identity.
    """
    return (
        f"Input changed for '{name}' since it was transcribed (stored "
        f"{stored[:12]}..., now {current[:12]}...). Skipping it; use --force to "
        "transcribe it from scratch."
    )


def lease_reason(holder: str | None, until: float | None) -> str:
    """Name the run that leases a document and when its lease expires.

    *until* is the expiry in seconds since the epoch, shown in local time.
    """
    reason = f"leased by run {holder or 'unknown'}"
    if until is not None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until))
        reason += f" until {stamp}"
    return reason


def classify(  # noqa: PLR0913 - one keyword per resume input
    root: Path,
    name: str,
    *,
    outputs: Sequence[Path] = (),
    image_settings: Mapping[str, Any] | None = None,
    model: str | None = None,
    retry_errors: bool = False,
    holder: str | None = None,
    now: float | None = None,
    input_sha256: str | None = None,
) -> Classification:
    """Classify item *name* of output root *root* from the store; never write.

    *outputs* are the item's rendered files for this run's formats.
    *image_settings* is this run's image-settings record (None for methods
    that render no page images) and *model* its model. Failed units count
    as done unless *retry_errors*; pending units make the item partial. A
    lease held by another run than *holder* that has not expired at *now*
    blocks the item, as an open or unresolved batch submission does. An
    item with stored units whose recorded input identity differs from
    *input_sha256*, the current one, is refused, partial or complete; a
    document without a recorded identity is trusted. A changed user image
    setting refuses a partial item; a changed model alone resumes with a
    note. Raises the store's errors for a versionless, newer or unreadable
    database.
    """
    path = database_path(root)
    present = tuple(output for output in outputs if _nonempty(output))
    state = _schema(root)
    store = None
    if state.exists and not state.empty:
        store = Store(path, TOOL, components=_COMPONENTS)
    resume = store.resume_state(name) if store is not None else None
    doc_id = resume.doc_id if resume is not None else None
    if resume is None or not resume.counts:
        if present:
            return Classification(
                ItemState.UNSTORED,
                name,
                doc_id,
                reason=(
                    f"{present[0].name} exists without units in {DATABASE_NAME}; "
                    "--force transcribes it again"
                ),
            )
        return Classification(ItemState.NONE, name, doc_id, reason="no stored units")
    assert doc_id is not None

    moment = time.time() if now is None else now
    if (
        resume.lease_run_id is not None
        and resume.lease_run_id != holder
        and resume.lease_until is not None
        and resume.lease_until > moment
    ):
        return Classification(
            ItemState.BLOCKED,
            name,
            doc_id,
            reason=lease_reason(resume.lease_run_id, resume.lease_until),
            leased_by=resume.lease_run_id,
        )
    if resume.blocked:
        return Classification(
            ItemState.BLOCKED,
            name,
            doc_id,
            reason=(
                f"{len(resume.blocked)} unit(s) await an open or unresolved batch "
                "submission; run batch check"
            ),
        )

    models, registered, stored_sha256 = read(
        path, lambda connection: _details(connection, doc_id)
    )
    if (
        stored_sha256 is not None
        and input_sha256 is not None
        and stored_sha256 != input_sha256
    ):
        return Classification(
            ItemState.REFUSED,
            name,
            doc_id,
            reason=changed_input(name, stored_sha256, input_sha256),
        )
    recorded = resume.image_settings
    recorded = _plain(recorded) if isinstance(recorded, Mapping) else None
    current = _plain(image_settings) if image_settings is not None else None
    changed: list[str] = []
    model_changed = False
    if recorded is not None and current is not None:
        changed = changed_image_settings(recorded, current)
        model_changed = not changed and recorded != current
    earlier = [used for used in models if model is not None and used != model]
    note = (
        _model_note(name, earlier, model, recorded)
        if earlier or model_changed
        else None
    )

    total = sum(resume.counts.values())
    work = resume.pending + (resume.failed if retry_errors else ())
    if not work:
        missing = tuple(output for output in outputs if not output.is_file())
        stale = any(registered.get(_path_key(output, root)) == 1 for output in outputs)
        render = bool(missing) or stale
        failed = len(resume.failed)
        reason = "all units stored"
        if failed:
            reason += f", {failed} failed (use --retry-errors to retry them)"
        if render:
            reason += "; its files are rendered again"
        return Classification(
            ItemState.COMPLETE,
            name,
            doc_id,
            render=render,
            missing=missing,
            reason=reason,
            note=note,
            total=total,
        )
    if changed:
        return Classification(
            ItemState.REFUSED,
            name,
            doc_id,
            reason=(
                f"Image settings changed for '{name}': {', '.join(changed)}. "
                "Skipping it; use --force to replace the existing run."
            ),
        )
    units = tuple(sorted(work))
    return Classification(
        ItemState.PARTIAL,
        name,
        doc_id,
        units=units,
        reason=f"{len(units)} of {total} unit(s) to process",
        note=note,
        total=total,
    )


def identity_write(
    doc_id: int, input_sha256: str
) -> Callable[[sqlite3.Connection], None]:
    """Return a write that records a document's input identity.

    Pass it to ``Store.write`` so it runs under the document's lease.
    """

    def write(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE documents SET input_sha256 = ? WHERE doc_id = ?",
            (input_sha256, doc_id),
        )

    return write


_HOLDERS: Final = (
    "SELECT DISTINCT r.submission_id FROM submission_requests AS r "
    "JOIN submissions AS s ON s.submission_id = r.submission_id "
    "WHERE s.status IN (SELECT value FROM json_each(:blocking)) "
    "AND r.doc_id = :doc ORDER BY r.submission_id"
)
_REPLACED: Final = (
    "doc_id = :doc AND (kind != :kind OR idx NOT IN "
    "(SELECT value FROM json_each(:idxs)))"
)
_REPLAN_UNIT: Final = """
INSERT INTO units (
    doc_id, kind, idx, status, source_index, label, tool, model, provider,
    run_id, attempts, stale, updated_at
) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, 0, 0, ?)
ON CONFLICT (doc_id, kind, idx) DO UPDATE SET
    status = 'pending', outcome = NULL, content = NULL, usage = NULL,
    error = NULL, error_type = NULL, provenance = NULL, response_id = NULL,
    attempts = 0, stale = 0, source_index = excluded.source_index,
    label = excluded.label, run_id = excluded.run_id,
    updated_at = excluded.updated_at
"""


def forced_replacement(  # noqa: PLR0913 - one keyword per part of the new plan
    store: Store,
    doc_id: int,
    kind: str,
    units: Sequence[PlannedUnit],
    *,
    model: str,
    provider: str,
    input_sha256: str | None = None,
    image_settings: Mapping[str, Any] | None = None,
) -> Callable[[sqlite3.Connection], None]:
    """Return the one write that starts document *doc_id* over for ``--force``.

    Pass it to ``Store.write`` so it runs under the document's lease, in one
    transaction: it deletes the units of other kinds and those of *kind*
    that *units* (the new plan) does not hold, with their settled batch
    requests; creates the planned units, or returns them to ``pending``
    with the planned label and source index; deletes the decision
    pre-pass's ``page_types`` rows; marks every registered output stale;
    and records *input_sha256* and *image_settings* (None keeps the
    recorded value). The responses stay.

    Raises:
        SubmissionStateError: An open or unresolved batch submission holds
            a unit of the document; nothing is changed.
    """
    run_id = store.run_id
    if run_id is None:
        raise RuntimeError("Begin a run before replacing a document's units.")
    now = datetime.fromtimestamp(store.clock(), tz=UTC).isoformat(
        timespec="milliseconds"
    )
    values = {
        "doc": doc_id,
        "kind": kind,
        "idxs": json.dumps([unit.idx for unit in units]),
        "blocking": json.dumps(list(BLOCKING_SUBMISSIONS)),
    }
    rows = [
        (
            doc_id,
            kind,
            unit.idx,
            unit.source_index,
            unit.label,
            store.tool,
            model,
            provider,
            run_id,
            now,
        )
        for unit in units
    ]

    def write(connection: sqlite3.Connection) -> None:
        held = [str(row[0]) for row in connection.execute(_HOLDERS, values)]
        if held:
            raise SubmissionStateError(
                f"Submission {', '.join(held)} still holds units of document "
                f"{doc_id}; run batch check, or abandon it, first."
            )
        connection.execute(f"DELETE FROM submission_requests WHERE {_REPLACED}", values)
        connection.execute(f"DELETE FROM units WHERE {_REPLACED}", values)
        connection.executemany(_REPLAN_UNIT, rows)
        connection.execute("DELETE FROM page_types WHERE doc_id = ?", (doc_id,))
        connection.execute("UPDATE outputs SET stale = 1 WHERE doc_id = ?", (doc_id,))
        if input_sha256 is not None:
            identity_write(doc_id, input_sha256)(connection)
        if image_settings is not None:
            connection.execute(
                "UPDATE documents SET image_settings = ? WHERE doc_id = ?",
                (json.dumps(image_settings, ensure_ascii=False, default=str), doc_id),
            )
        refresh_documents(connection, doc_id)
        connection.execute(
            "UPDATE documents SET last_run_id = COALESCE(?, last_run_id), "
            "updated_at = ? WHERE doc_id = ?",
            (run_id, now, doc_id),
        )

    return write


def page_type_rows(
    document: str,
    decisions: Sequence[tuple[int, str, Mapping[str, float]]],
    *,
    taxonomy: str,
    profile: str,
    model: str,
    fingerprint: str,
    run_id: str | None,
) -> Callable[[sqlite3.Connection], None]:
    """Return a write that replaces the page types of *document*'s pages.

    *decisions* holds ``(idx, page_type, probabilities)``; pass the write to
    ``Store.write`` so it runs under the document's lease. The rows belong to
    the document's id, so dropping the document removes them.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = [
        (
            document,
            idx,
            page_type,
            json.dumps(dict(probabilities), sort_keys=True),
            taxonomy,
            profile,
            model,
            fingerprint,
            run_id,
            now,
        )
        for idx, page_type, probabilities in decisions
    ]

    def write(connection: sqlite3.Connection) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO page_types (doc_id, idx, page_type, "
            "probabilities, taxonomy, profile, model, fingerprint, run_id, "
            "created_at) SELECT doc_id, ?, ?, ?, ?, ?, ?, ?, ?, ? "
            "FROM documents WHERE name = ?",
            [(*row[1:], row[0]) for row in rows],
        )

    return write
