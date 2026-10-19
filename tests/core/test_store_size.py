"""Database size of a 200-page fake LLM run.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/core/test_store_size.py -q -s

A scripted chat model answers each page with a schema response whose
transcription holds about 2,000 characters, a dense printed page. Every
attempt is stored verbatim in ``responses`` and every page as a unit. The
test prints the database size and the bytes per page (compression is worth
adding only if this measurement calls for it) and bounds the size.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.run as core
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.output.store import DATABASE_NAME
from tests.conftest import make_settings, make_spec
from tests.llm import helpers

PAGES = 200
LINE = "Nimm ein Pfund Mehl, drey Eyer und ein wenig Salz, rühre es wohl. "
TEXT = (LINE * 31)[:2000]
PAGE: dict[str, Any] = {
    "image_analysis": "A printed page of a cookbook, one column, no images.",
    "transcription": TEXT,
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}
MAX_BYTES_PER_PAGE = 16_384
"""Upper bound per page: the unit text, the stored response and the indexes."""


def _pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=100, height=150)
    document.save(path)
    document.close()
    return path


def test_the_database_of_a_200_page_run_stays_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    script = helpers.install(monkeypatch, [PAGE])
    out = tmp_path / "out"
    spec = make_spec(
        _pdf(tmp_path / "in" / "book.pdf", PAGES), type="pdf", output=out, dpi=72
    )

    result = asyncio.run(
        core.run(core.plan(spec, make_settings(rate_limits=((100_000, 1),))))
    )

    db = out / DATABASE_NAME
    size = db.stat().st_size
    responses = read_rows(db, "responses")
    raw = sum(len(str(row["raw"]).encode("utf-8")) for row in responses)
    print(  # noqa: T201 - the measurement is the intended output
        f"\nstore size of a {PAGES}-page run: {size:,} bytes, "
        f"{size / PAGES:,.0f} bytes per page "
        f"(raw responses {raw / PAGES:,.0f} bytes per page)"
    )
    assert result.complete == 1
    assert len(script.requests) == PAGES
    assert len(responses) == PAGES
    assert len(read_rows(db, "units")) == PAGES
    assert size <= PAGES * MAX_BYTES_PER_PAGE
