"""The store across runs: one database per output folder, read back read-only.

Each case runs ``chronotranscriber transcribe`` more than once over the same
output folder through ``run_cli`` and reads ``chronotranscriber.sqlite``
back through short ``mode=ro`` connections, as outside readers do: the
``page_text`` and ``document_status`` views and the ``responses`` table. A
complete item whose rendered file is gone is rendered again from the store
without a model call; an output without a store row is unstored and skipped.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.output.store import DATABASE_NAME
from tests.characterization.conftest import GoldenCheck
from tests.characterization.fakes import FakeLLM
from tests.characterization.inputs import make_scanned_pdf
from tests.characterization.runner import RunResult, run_cli

pytestmark = pytest.mark.integration


def _query(database: Path, sql: str, *params: Any) -> list[dict[str, Any]]:
    """Run *sql* on a read-only connection to *database* and close it."""
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()


def _page_text(database: Path) -> list[dict[str, Any]]:
    return _query(
        database,
        "SELECT document, kind, page_index, image_name, order_index, text, status, "
        "outcome, model FROM page_text",
    )


def _run(llm: FakeLLM, root: Path, input_path: Path, **options: Any) -> RunResult:
    return run_cli(
        "transcribe",
        llm=llm,
        root=root,
        input=input_path,
        output=root / "out",
        type="pdf",
        method="llm",
        **options,
    )


def _counts(result: RunResult) -> tuple[int, ...]:
    assert result.summary is not None, result.stdout + result.stderr
    keys = ("total", "complete", "failed", "skipped")
    return tuple(result.summary[f"items_{key}"] for key in keys)


def _name(result: RunResult, item: Path) -> str:
    """Return the document name of *item*: its output below the output folder."""
    assert result.output_dir is not None
    output = result.output(item).with_suffix("")
    return output.relative_to(result.output_dir).as_posix()


def test_page_text_reads_back_what_the_txt_renders(
    case_root: Callable[[str], Path], fake_llm: FakeLLM
) -> None:
    root = case_root("case")
    pdf = make_scanned_pdf(root / "docs" / "doc.pdf", 3)

    result = _run(fake_llm, root, pdf)

    assert result.exit_code == 0, result.stdout + result.stderr
    database = root / "out" / DATABASE_NAME
    name = _name(result, pdf)
    rows = _page_text(database)
    assert [
        (row["document"], row["kind"], row["page_index"], row["order_index"])
        for row in rows
    ] == [(name, "page", index, index) for index in range(3)]
    assert {(row["status"], row["outcome"]) for row in rows} == {("done", "ok")}
    assert [row["image_name"] for row in rows] == [
        f"page_{index:04d}_pre_processed.jpg" for index in range(1, 4)
    ]
    text = result.output(pdf).read_text(encoding="utf-8")
    assert all(row["text"] for row in rows)
    assert text == "\n".join(row["text"] for row in rows) + "\n"
    (status,) = _query(database, "SELECT * FROM document_status")
    assert (status["document"], status["status"]) == (name, "complete")
    assert (status["units_total"], status["units_done"]) == (3, 3)


def test_two_items_share_one_database_and_a_rerun_keeps_history(
    case_root: Callable[[str], Path], fake_llm: FakeLLM, golden: GoldenCheck
) -> None:
    root = case_root("case")
    docs = root / "docs"
    make_scanned_pdf(docs / "alpha.pdf", 2)
    make_scanned_pdf(docs / "beta.pdf", 3)
    database = root / "out" / DATABASE_NAME

    first = _run(fake_llm, root, docs)

    assert first.exit_code == 0, first.stdout + first.stderr
    assert [path.name for path in (root / "out").glob("*.sqlite*")] == [DATABASE_NAME]
    alpha, beta = (_name(first, docs / name) for name in ("alpha.pdf", "beta.pdf"))
    documents = _query(database, "SELECT name, status FROM documents ORDER BY name")
    assert documents == [
        {"name": alpha, "status": "complete"},
        {"name": beta, "status": "complete"},
    ]
    pages = _page_text(database)
    assert [(row["document"], row["page_index"]) for row in pages] == [
        (alpha, 0),
        (alpha, 1),
        (beta, 0),
        (beta, 1),
        (beta, 2),
    ]
    golden("sqlite/two_items_one_database", first)

    second = _run(fake_llm, root, docs, force=True)

    assert second.exit_code == 0, second.stdout + second.stderr
    assert len(second.calls) == 5
    assert _page_text(database) == pages
    runs = _query(database, "SELECT run_id FROM runs ORDER BY rowid")
    assert len(runs) == 2
    last = {row["last_run_id"] for row in _query(database, "SELECT * FROM documents")}
    assert last == {runs[1]["run_id"]}
    attempts = _query(
        database, "SELECT run_id, count(*) AS n FROM responses GROUP BY run_id"
    )
    assert sorted(row["n"] for row in attempts) == [5, 5]
    golden("sqlite/rerun_upserts", second)


def test_a_complete_item_without_its_file_is_rendered_without_a_model_call(
    case_root: Callable[[str], Path], fake_llm: FakeLLM, golden: GoldenCheck
) -> None:
    root = case_root("case")
    pdf = make_scanned_pdf(root / "docs" / "doc.pdf", 3)
    first = _run(fake_llm, root, pdf)
    assert first.exit_code == 0, first.stdout + first.stderr
    output = first.output(pdf)
    rendered = output.read_bytes()
    output.unlink()

    again = _run(fake_llm, root, pdf)

    assert again.exit_code == 0, again.stdout + again.stderr
    assert again.calls == []
    assert again.created == [output]
    assert output.read_bytes() == rendered
    golden("sqlite/missing_file_rendered", again)


def test_an_output_without_store_rows_is_unstored_and_skipped(
    case_root: Callable[[str], Path], fake_llm: FakeLLM
) -> None:
    root = case_root("case")
    pdf = make_scanned_pdf(root / "docs" / "doc.pdf", 2)
    first = _run(fake_llm, root, pdf)
    assert first.exit_code == 0, first.stdout + first.stderr
    output = first.output(pdf)
    rendered = output.read_bytes()
    (root / "out" / DATABASE_NAME).unlink()

    second = _run(fake_llm, root, pdf)

    assert second.calls == []
    assert _counts(second)[1:3] == (0, 0)
    assert "(unstored)" in second.stderr
    assert "--force transcribes it again" in second.stderr
    assert output.read_bytes() == rendered
