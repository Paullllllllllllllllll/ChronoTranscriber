"""Resume classification from the store.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_ct_resume.py -q

Each case seeds an output root through the store API (raw SQL only where a
case needs a state no run leaves: a live foreign lease, a stale output, a
newer schema), classifies the item and checks the state, the units to
process, the render flag, the reason and the note. Every case also checks
that classification left the database bytes and the folder unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.store import (
    Lease,
    NewerSchemaError,
    Store,
    StoreError,
    UnitKey,
    UnreadableDatabaseError,
    VersionlessDatabaseError,
)
from chronotranscriber.output.render import render_item
from chronotranscriber.output.store import (
    DATABASE_NAME,
    ItemState,
    classify,
    image_settings_record,
    open_store,
)
from tests.output.test_ct_store import MODEL, NAME, Unit, page, seed_document

HOLDER = "this-run"
OTHER_MODEL = "gpt-5.6-terra"

_SETTINGS = {
    "grayscale_conversion": True,
    "high_target_box": (768, 1536),
    "jpeg_quality": 95,
    "llm_detail": "original",
    "model_name": MODEL,
    "provider": "openai",
    "resolved_detail": "original",
    "cap_policy": "openai-original-30k-v1",
    "image_original_patch_cap_30k": True,
}
SETTINGS = image_settings_record(_SETTINGS, "openai", 24_000_000)
OTHER_MODEL_SETTINGS = image_settings_record(
    {
        **_SETTINGS,
        "model_name": OTHER_MODEL,
        "resolved_detail": "high",
        "cap_policy": "profile-v1",
        "image_original_patch_cap_30k": False,
    },
    "openai",
    24_000_000,
)
CHANGED_SETTINGS = image_settings_record(
    {**_SETTINGS, "jpeg_quality": 80, "grayscale_conversion": False},
    "openai",
    24_000_000,
)

After = Callable[[Store, Lease], Awaitable[None]]
Setup = Callable[[Path], None]


def _output(root: Path) -> Path:
    return root / "doc-139d544b" / "doc.txt"


def seeded(
    units: Sequence[Unit],
    *,
    render: bool = True,
    after: After | None = None,
    sql: str | None = None,
) -> Setup:
    """Return a setup that stores *units*, renders them and runs *after*.

    *sql* runs on the closed database, for states no run leaves.
    """

    def setup(root: Path) -> None:
        async def main() -> None:
            async with open_store(root) as store:
                await store.begin_run("transcribe")
                lease = await seed_document(store, units, image_settings=SETTINGS)
                if render:
                    await render_item(store, NAME, ("txt",), _output(root), lease)
                if after is not None:
                    await after(store, lease)
                await store.end_run(0)

        asyncio.run(main())
        if sql is not None:
            _execute(root, sql)

    return setup


def _execute(root: Path, sql: str) -> None:
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute(sql)
        connection.commit()
    finally:
        connection.close()


def unstored_output(root: Path) -> None:
    _output(root).parent.mkdir(parents=True)
    _output(root).write_text("Rendered outside the store\n", encoding="utf-8")


def other_document_and_unstored_output(root: Path) -> None:
    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            await seed_document(store, [page(0)], name="other-1/other")

    asyncio.run(main())
    unstored_output(root)


async def open_submission(store: Store, lease: Lease) -> None:
    await store.open_submission(
        provider="openai", model=MODEL, units=[UnitKey(lease.doc_id, "page", 1)]
    )


async def unresolved_submission(store: Store, lease: Lease) -> None:
    submission = await store.open_submission(
        provider="anthropic", model=MODEL, units=[UnitKey(lease.doc_id, "page", 1)]
    )
    await store.mark_unresolved(submission.submission_id, "create response lost")


def _lease(holder: str, offset: float) -> str:
    return (
        f"UPDATE documents SET lease_run_id = '{holder}', "
        f"lease_until = {time.time() + offset}, "
        "lease_generation = lease_generation + 1"
    )


DONE = [page(0), page(1), page(2)]
WITH_FAILED = [page(0), page(1, "failed"), page(2)]
PAGES_RUN = [page(0, "pending"), page(1), page(2), page(3, "pending")]


@dataclass(frozen=True)
class Case:
    """One classification: the setup, the call and what it must return."""

    setup: Setup | None
    state: ItemState
    indices: tuple[int, ...] = ()
    render: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    note: bool = False


CASES = {
    "no_database": Case(None, ItemState.NONE),
    "all_done": Case(seeded(DONE), ItemState.COMPLETE),
    "failed_count_as_done": Case(
        seeded(WITH_FAILED), ItemState.COMPLETE, reason="1 failed"
    ),
    "failed_retried": Case(
        seeded(WITH_FAILED),
        ItemState.PARTIAL,
        (1,),
        options={"retry_errors": True},
    ),
    "pages_run_leaves_pending_units": Case(
        seeded(PAGES_RUN), ItemState.PARTIAL, (0, 3)
    ),
    "pending_and_failed_retried": Case(
        seeded([page(0, "failed"), page(1), page(2, "pending")]),
        ItemState.PARTIAL,
        (0, 2),
        options={"retry_errors": True},
    ),
    "unstored_output_without_database": Case(
        unstored_output, ItemState.UNSTORED, reason="--force"
    ),
    "unstored_output_without_row": Case(
        other_document_and_unstored_output, ItemState.UNSTORED, reason="--force"
    ),
    "missing_file_renders_again": Case(
        seeded(DONE, render=False), ItemState.COMPLETE, render=True
    ),
    "stale_output_renders_again": Case(
        seeded(DONE, sql="UPDATE outputs SET stale = 1"),
        ItemState.COMPLETE,
        render=True,
    ),
    "open_submission": Case(
        seeded(PAGES_RUN, after=open_submission),
        ItemState.BLOCKED,
        reason="batch",
    ),
    "unresolved_submission": Case(
        seeded(PAGES_RUN, after=unresolved_submission),
        ItemState.BLOCKED,
        reason="batch",
    ),
    "leased_by_another_live_holder": Case(
        seeded(PAGES_RUN, sql=_lease("other-run", 600)),
        ItemState.BLOCKED,
        reason="other-run",
    ),
    "expired_foreign_lease": Case(
        seeded(PAGES_RUN, sql=_lease("other-run", -1)), ItemState.PARTIAL, (0, 3)
    ),
    "own_lease": Case(
        seeded(PAGES_RUN, sql=_lease(HOLDER, 600)), ItemState.PARTIAL, (0, 3)
    ),
    "same_image_settings": Case(
        seeded(PAGES_RUN),
        ItemState.PARTIAL,
        (0, 3),
        options={"image_settings": SETTINGS, "model": MODEL},
    ),
    "changed_user_setting_refused": Case(
        seeded(PAGES_RUN),
        ItemState.REFUSED,
        options={"image_settings": CHANGED_SETTINGS, "model": MODEL},
        reason="grayscale_conversion, jpeg_quality",
    ),
    "changed_setting_on_complete_item": Case(
        seeded(DONE),
        ItemState.COMPLETE,
        options={"image_settings": CHANGED_SETTINGS, "model": MODEL},
    ),
    "changed_model_resumes_with_note": Case(
        seeded(PAGES_RUN),
        ItemState.PARTIAL,
        (0, 3),
        options={"image_settings": OTHER_MODEL_SETTINGS, "model": OTHER_MODEL},
        note=True,
    ),
    "changed_model_on_complete_item": Case(
        seeded(DONE),
        ItemState.COMPLETE,
        options={"image_settings": OTHER_MODEL_SETTINGS, "model": OTHER_MODEL},
        note=True,
    ),
    "changed_model_without_images": Case(
        seeded(PAGES_RUN), ItemState.PARTIAL, (0, 3), options={"model": OTHER_MODEL},
        note=True,
    ),
}  # fmt: skip


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("name", list(CASES))
def test_classify(tmp_path: Path, name: str) -> None:
    case = CASES[name]
    if case.setup is not None:
        case.setup(tmp_path)
    before = _fingerprint(tmp_path)

    result = classify(
        tmp_path,
        NAME,
        outputs=[_output(tmp_path)],
        holder=HOLDER,
        **case.options,
    )

    assert result.state is case.state, result.reason
    assert result.indices == case.indices
    assert result.render is case.render
    if case.reason is not None:
        assert case.reason in result.reason
    if case.note:
        assert result.note is not None
        assert MODEL in result.note and OTHER_MODEL in result.note
    else:
        assert result.note is None
    assert _fingerprint(tmp_path) == before


def test_a_missing_file_is_listed_for_rendering(tmp_path: Path) -> None:
    seeded(DONE, render=False)(tmp_path)
    md = _output(tmp_path).with_suffix(".md")

    result = classify(tmp_path, NAME, outputs=[_output(tmp_path), md])

    assert result.missing == (_output(tmp_path), md)
    assert result.doc_id is not None


def _versionless(root: Path) -> None:
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute(
            "CREATE TABLE documents (name TEXT PRIMARY KEY, complete INTEGER)"
        )
        connection.execute(
            "CREATE TABLE pages (document TEXT, page_index INTEGER, text TEXT)"
        )
        connection.execute("INSERT INTO documents VALUES (?, 1)", (NAME,))
        connection.commit()
    finally:
        connection.close()


def _newer(root: Path) -> None:
    seeded(DONE)(root)
    _execute(root, "UPDATE meta SET version = 99 WHERE component = 'chronotranscriber'")


def _unreadable(root: Path) -> None:
    (root / DATABASE_NAME).write_bytes(b"not a database, " * 256)


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (_versionless, VersionlessDatabaseError),
        (_newer, NewerSchemaError),
        (_unreadable, UnreadableDatabaseError),
    ],
)
def test_refused_databases_raise_and_stay_unchanged(
    tmp_path: Path, setup: Setup, error: type[StoreError]
) -> None:
    setup(tmp_path)
    before = _fingerprint(tmp_path)

    with pytest.raises(error) as raised:
        classify(tmp_path, NAME, outputs=[_output(tmp_path)])

    assert raised.value.exit_code == 2
    assert _fingerprint(tmp_path) == before
