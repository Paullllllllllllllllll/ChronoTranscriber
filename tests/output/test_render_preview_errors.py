"""The render subcommand and its preview survive an unreadable target.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/output/test_render_preview_errors.py -q

Two documents are seeded into one store and rendered; then the first one's
``.txt`` cannot be read: it is held open with an exclusive Windows handle
(share mode 0, skipped elsewhere), or reading or rendering it is patched to
raise ``PermissionError``. Oracle: ``preview_database`` lists the target as
stale with a warning naming it instead of raising, and ``render_database``
reports the first document as failed while it writes the second one's
deleted file again.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.output.render as render_module
from chronotranscriber.common.store import Store
from chronotranscriber.output.render import (
    DocumentRender,
    preview_database,
    render_database,
    render_item,
)
from chronotranscriber.output.store import database_path, open_store
from tests.output.test_ct_store import NAME, page, seed_document

OTHER = "other-0badf00d/other"

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="exclusive file handles are Windows-only"
)


@contextlib.contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold *path* open for reading with share mode 0."""
    import _winapi

    handle = _winapi.CreateFile(
        str(path), _winapi.GENERIC_READ, 0, 0, _winapi.OPEN_EXISTING, 0, 0
    )
    try:
        yield
    finally:
        _winapi.CloseHandle(handle)


def _seeded(root: Path) -> tuple[Path, Path]:
    """Seed and render two documents; return their txt files."""
    first = root / "doc-139d544b" / "doc.txt"
    second = root / "other-0badf00d" / "other.txt"

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            lease = await seed_document(store, [page(0), page(1)])
            await render_item(store, NAME, ("txt",), first, lease)
            lease = await seed_document(store, [page(0)], name=OTHER)
            await render_item(store, OTHER, ("txt",), second, lease)

    asyncio.run(main())
    return first, second


def _by_name(results: list[DocumentRender]) -> dict[str, DocumentRender]:
    return {result.name: result for result in results}


def _assert_preview(results: list[DocumentRender], held: Path) -> None:
    by_name = _by_name(results)
    first = by_name[NAME]
    assert first.stale == [held]
    assert any(str(held) in warning for warning in first.warnings)
    assert first.failed
    assert not by_name[OTHER].failed


def _assert_render(results: list[DocumentRender], held: Path, other: Path) -> None:
    by_name = _by_name(results)
    assert by_name[NAME].failed
    assert held not in by_name[NAME].written
    assert other in by_name[OTHER].written
    assert other.read_bytes() == b"Page 1 text\n"


@windows_only
def test_preview_lists_an_exclusively_held_target_as_stale(tmp_path: Path) -> None:
    held, _ = _seeded(tmp_path)

    with _exclusive(held):
        results = preview_database(database_path(tmp_path))

    _assert_preview(results, held)


def test_preview_lists_a_target_that_raises_permission_error_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held, _ = _seeded(tmp_path)
    original = render_module.file_sha256

    def file_sha256(path: Path) -> str | None:
        if Path(path) == held:
            raise PermissionError(13, "Permission denied", str(path))
        return original(path)

    monkeypatch.setattr(render_module, "file_sha256", file_sha256)

    results = preview_database(database_path(tmp_path))

    _assert_preview(results, held)


@windows_only
def test_render_reports_an_exclusively_held_target_and_renders_the_rest(
    tmp_path: Path,
) -> None:
    held, other = _seeded(tmp_path)
    other.unlink()

    with _exclusive(held):
        results = asyncio.run(render_database(database_path(tmp_path), force=True))

    _assert_render(results, held, other)


def test_render_reports_a_render_that_raises_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held, other = _seeded(tmp_path)
    other.unlink()
    original = Store.render

    async def render(self: Store, lease: Any, targets: Any, **kwargs: Any) -> Any:
        if any(target.path == held for target in targets):
            raise PermissionError(13, "Permission denied", str(held))
        return await original(self, lease, targets, **kwargs)

    monkeypatch.setattr(Store, "render", render)

    results = asyncio.run(render_database(database_path(tmp_path), force=True))

    _assert_render(results, held, other)
