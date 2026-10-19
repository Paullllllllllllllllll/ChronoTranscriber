"""The drop subcommand: exit codes, the JSON line, dry run and refusals.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_drop_command.py -q

CLI oracle: ``drop --dry-run`` leaves the directory listing and the database
bytes unchanged and lists what a drop would delete; a real drop deletes the
selected document's rows and its unedited rendered file, keeps and names a
hand-edited one and leaves the other document intact; a document another run
leases is refused with exit 1; a missing store or a selection that is
malformed or names nothing exits 2. ``--json`` writes exactly one line on
every exit. The commands run in-process through the console entry point.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber.output.store import DATABASE_NAME, open_store
from tests.cli.conftest import json_lines
from tests.output.test_ct_store import page, seed_document

ALPHA = "alpha-11111111/alpha"
BETA = "beta-22222222/beta"
DROP_KEYS = {"dry_run", "command", "dropped", "refused", "deleted_files", "kept_files"}


def seed_two_documents(root: Path, *, render: bool = True) -> tuple[Path, Path]:
    """Seed the documents alpha and beta, rendered unless not *render*;
    return their txt files."""

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("transcribe")
            await seed_document(store, [page(0), page(1)], name=ALPHA)
            await seed_document(store, [page(0)], name=BETA)

    asyncio.run(main())
    alpha, beta = root / f"{ALPHA}.txt", root / f"{BETA}.txt"
    if render:
        assert entry.main(["render", "--input", str(root)]) == 0
        assert alpha.is_file() and beta.is_file()
    return alpha, beta


def _listing(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _count(root: Path, sql: str) -> int:
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        return int(connection.execute(sql).fetchone()[0])
    finally:
        connection.close()


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    code = entry.main(["drop", *argv, "--json"])
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    assert set(payload) == DROP_KEYS
    assert payload["command"] == "drop"
    return code, payload, captured.err


def test_a_dry_run_lists_the_drop_and_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    alpha, _beta = seed_two_documents(root)
    capsys.readouterr()
    before = _listing(root)

    code, payload, err = _run(
        capsys, "--input", str(root), "--select", "alpha", "--dry-run"
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["dropped"] == [ALPHA]
    assert payload["deleted_files"] == [str(alpha)]
    assert (payload["refused"], payload["kept_files"]) == ([], [])
    assert "[DRY RUN] would drop alpha" in err
    assert _listing(root) == before


def test_a_drop_deletes_the_document_and_keeps_an_edited_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    alpha, beta = seed_two_documents(root)
    alpha.write_text("edited by hand\n", encoding="utf-8")
    capsys.readouterr()

    code, payload, err = _run(
        capsys, "--input", str(root / DATABASE_NAME), "--select", "1"
    )

    assert code == 0
    assert payload["dropped"] == [ALPHA]
    assert payload["deleted_files"] == []
    assert payload["kept_files"] == [
        {"path": str(alpha), "reason": "it was changed outside the tool"}
    ]
    assert f"kept {alpha}" in err
    assert "skip that item as unstored" in err
    assert alpha.read_text(encoding="utf-8") == "edited by hand\n"
    assert beta.is_file()
    assert _count(root, "SELECT COUNT(*) FROM documents") == 1
    assert _count(root, f"SELECT COUNT(*) FROM documents WHERE name = '{BETA}'") == 1
    assert _count(root, "SELECT COUNT(*) FROM units") == 1


def test_a_drop_of_every_document_deletes_their_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    alpha, beta = seed_two_documents(root)
    capsys.readouterr()

    code, payload, _err = _run(capsys, "--input", str(root), "--all")

    assert code == 0
    assert payload["dropped"] == [ALPHA, BETA]
    assert sorted(payload["deleted_files"]) == sorted([str(alpha), str(beta)])
    assert not alpha.exists() and not beta.exists()
    assert _count(root, "SELECT COUNT(*) FROM documents") == 0


def test_a_leased_document_is_refused_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    alpha, beta = seed_two_documents(root)
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'other-run', lease_until = ? "
            "WHERE name = ?",
            (time.time() + 3600, ALPHA),
        )
        connection.commit()
    finally:
        connection.close()
    capsys.readouterr()

    code, payload, err = _run(
        capsys, "--input", str(root), "--files", f"alpha.pdf,{BETA}"
    )

    assert code == 1
    assert payload["dropped"] == [BETA]
    (refusal,) = payload["refused"]
    assert refusal["name"] == ALPHA
    assert "run other-run leases it" in refusal["reason"]
    assert "refused alpha" in err
    assert alpha.is_file() and not beta.exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--select", "1"], "--input"),
        (["--input", "{root}"], "exactly one of"),
        (["--input", "{root}", "--all", "--select", "1"], "not allowed with"),
        (["--input", "{root}", "--select", "gamma"], "No stored document matches"),
        (["--input", "{root}", "--files", "gamma"], "names no stored document"),
        (["--input", "{missing}", "--all"], "does not exist"),
        (["--input", "{empty}", "--all"], f"No {DATABASE_NAME}"),
        (["--input", "{other}", "--all"], f"is not a {DATABASE_NAME} file"),
    ],
    ids=[
        "no input",
        "no selection",
        "two selections",
        "select matches nothing",
        "files names nothing",
        "missing input",
        "no store",
        "another file",
    ],
)
def test_usage_errors_exit_2_with_one_line(
    argv: list[str], message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "out"
    seed_two_documents(root)
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.sqlite").write_bytes(b"")
    paths = {
        "root": root,
        "missing": tmp_path / "missing",
        "empty": tmp_path / "empty",
        "other": tmp_path / "notes.sqlite",
    }
    capsys.readouterr()
    before = _listing(root)

    code, payload, err = _run(capsys, *(arg.format(**paths) for arg in argv))

    assert code == 2
    assert payload["dropped"] == []
    assert message in err
    assert _listing(root) == before
