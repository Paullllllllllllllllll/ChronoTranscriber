"""Page answers join ``page_text`` on document and ordinal for every method.

One born-digital PDF is transcribed with Tesseract (``page`` units, through
``fakes_ocr.FakeTesseract``) and with its native text layer (``text`` units
without an image name), each into its own output folder, and then
classified page by page into the same folder against the fake decision
endpoint of ``tests/cli/test_classify_command.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from chronotranscriber import __main__ as entry
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_ocr import (
    FakeTesseract,
    block_real_tesseract,
    tesseract_cmd_settings,
)
from tests.characterization.inputs import make_born_digital_pdf
from tests.characterization.runner import run_cli
from tests.cli.test_classify_command import FakeDecisions, decisions  # noqa: F401

PAGES = 3


@pytest.mark.parametrize(
    ("method", "kind"), [("tesseract", "page"), ("native", "text")]
)
def test_page_answers_join_the_transcribed_units_on_their_ordinal(
    method: str,
    kind: str,
    tmp_path: Path,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    decisions: FakeDecisions,  # noqa: F811
) -> None:
    FakeTesseract().install(monkeypatch)
    missing = block_real_tesseract(monkeypatch, tmp_path)
    pdf = make_born_digital_pdf(tmp_path / "docs" / "born.pdf", PAGES)
    out = tmp_path / "out"
    transcribed = run_cli(
        "transcribe",
        llm=fake_llm,
        root=tmp_path,
        settings=tesseract_cmd_settings(missing),
        input=pdf,
        output=out,
        type="pdf",
        method=method,
    )
    assert transcribed.exit_code == 0, transcribed.stderr

    code = entry.main(
        [
            "classify",
            "pages",
            "--input",
            str(pdf),
            "--output",
            str(out),
            "--classifier",
            "layout",
        ]
    )
    assert code == 0

    connection = sqlite3.connect(out / "chronotranscriber.sqlite")
    try:
        joined = connection.execute(
            "SELECT c.ordinal, p.kind FROM classifications AS c JOIN page_text AS p "
            "ON p.document = c.document AND p.order_index = c.ordinal "
            "ORDER BY c.ordinal"
        ).fetchall()
    finally:
        connection.close()
    assert joined == [(n, kind) for n in range(PAGES)]
