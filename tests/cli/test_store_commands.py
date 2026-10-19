"""The render subcommand: plans, exit codes and the JSON line.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_store_commands.py -q

CLI oracle: ``render --dry-run`` leaves the directory listing and the
database bytes unchanged and prints its plan on stderr; ``--json`` writes
exactly one line on every exit (0, 2 and 130); ``render`` re-renders a stale
output and leaves a hand-edited file alone without ``--force``; a database
without a schema version makes ``render`` exit 2 as not a store. The commands
run in-process through the console entry point.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.cli.store as store_cli
from chronotranscriber import __main__ as entry
from chronotranscriber.output.store import DATABASE_NAME, open_store
from tests.cli.conftest import json_lines
from tests.output.test_ct_store import NAME, page, seed_document

RENDER_KEYS = {
    "dry_run",
    "command",
    "databases",
    "documents",
    "written",
    "unchanged",
    "kept",
    "stale",
    "skipped",
    "outputs",
}


def _store(tmp_path: Path) -> Path:
    """Seed one document of two pages and render it; return the output root."""
    out = tmp_path / "out"

    async def main() -> None:
        async with open_store(out) as store:
            await store.begin_run("transcribe")
            await seed_document(store, [page(0), page(1)])

    asyncio.run(main())
    assert entry.main(["render", "--input", str(out)]) == 0
    return out


def _listing(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    code = entry.main(list(argv))
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return code, payload, captured.err


def test_render_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _store(tmp_path)
    capsys.readouterr()
    rendered = out / f"{NAME}.txt"
    rendered.unlink()
    before = _listing(out)

    code, payload, err = _run(
        capsys, "render", "--input", str(out), "--dry-run", "--json"
    )

    assert code == 0
    assert set(payload) == RENDER_KEYS
    assert (payload["dry_run"], payload["written"]) == (True, 1)
    assert "[DRY RUN]" in err and "would write" in err
    assert _listing(out) == before


def _mark_stale(out: Path) -> None:
    connection = sqlite3.connect(out / DATABASE_NAME)
    try:
        connection.execute("UPDATE outputs SET stale = 1")
        connection.execute("UPDATE units SET content = 'corrected' WHERE idx = 0")
        connection.commit()
    finally:
        connection.close()


def test_render_rerenders_a_stale_output_and_keeps_an_edited_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _store(tmp_path)
    capsys.readouterr()
    rendered = out / f"{NAME}.txt"
    _mark_stale(out)

    code, payload, _err = _run(capsys, "render", "--input", str(out), "--json")

    assert code == 0
    assert payload["written"] == 1 and payload["outputs"] == [str(rendered)]
    assert rendered.read_text(encoding="utf-8").startswith("corrected\n")

    rendered.write_text("edited by hand\n", encoding="utf-8")
    code, payload, err = _run(capsys, "render", "--input", str(out), "--json")

    assert code == 0
    assert (payload["written"], payload["kept"]) == (0, 1)
    assert "--force" in err
    assert rendered.read_text(encoding="utf-8") == "edited by hand\n"

    code, payload, _err = _run(
        capsys, "render", "--input", str(out), "--force", "--json"
    )
    assert (code, payload["written"]) == (0, 1)
    assert rendered.read_text(encoding="utf-8").startswith("corrected\n")


def test_render_refuses_a_database_without_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    connection = sqlite3.connect(out / DATABASE_NAME)
    try:
        connection.execute("CREATE TABLE documents (name TEXT)")
        connection.commit()
    finally:
        connection.close()
    before = _listing(out)

    code, payload, err = _run(capsys, "render", "--input", str(out), "--json")

    assert code == 2
    assert payload["written"] == 0
    assert "not a chronotranscriber store" in err
    assert "migrate" not in err
    assert _listing(out) == before


@pytest.mark.parametrize(
    "argv",
    [
        ["render", "--formats", "pdf", "--json"],
        ["render", "--formats", "", "--json"],
    ],
    ids=["bad format", "no format"],
)
def test_usage_errors_exit_2_with_one_line(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, err = _run(capsys, *argv)
    assert code == 2
    assert payload["command"] == argv[0]
    assert "Invalid arguments" in err


def test_a_missing_input_path_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, _err = _run(
        capsys, "render", "--input", str(tmp_path / "missing"), "--json"
    )
    assert code == 2
    assert payload["command"] == "render"


def test_an_interrupt_exits_130_with_one_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(store_cli, "find_databases", interrupted)

    code, payload, _err = _run(capsys, "render", "--input", str(tmp_path), "--json")

    assert code == 130
    assert payload["command"] == "render"


def test_without_json_stdout_stays_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _store(tmp_path)
    capsys.readouterr()

    assert entry.main(["render", "--input", str(out), "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RENDER" in captured.err
