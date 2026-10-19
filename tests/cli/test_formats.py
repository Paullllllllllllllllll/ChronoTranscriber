"""Output formats and working files at the command line.

The formats are txt and md; json and sqlite are refused, and the database
is written whatever the formats. A ``keep_preprocessed_images`` settings key
is refused; ``--keep-working-files`` keeps the working files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber.output.store import DATABASE_NAME
from tests.characterization.fakes_ocr import FakeTesseract, block_real_tesseract
from tests.characterization.inputs import make_scanned_pdf
from tests.cli.conftest import MakePdf, json_lines


def _transcribe(*argv: str) -> int:
    return entry.main(["transcribe", *argv])


def _one_line(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return payload, captured.err


@pytest.fixture(autouse=True)
def dummy_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy key variables so that a plan never fails on a missing key."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, "test-key")


@pytest.mark.parametrize("formats", ["json", "sqlite", "txt,sqlite", "md,json"])
def test_removed_formats_exit_2_with_one_json_line(
    formats: str, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--input", str(make_pdf()), "--formats", formats, "--dry-run", "--json"]

    assert _transcribe(*argv) == 2

    payload, err = _one_line(capsys)
    assert payload["items_total"] == 0
    assert "is not one of txt, md" in err


def test_a_tesseract_run_in_md_writes_the_database_beside_the_md(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeTesseract().install(monkeypatch)
    missing = block_real_tesseract(monkeypatch, tmp_path)
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        f"tools:\n  tesseract_cmd: '{missing.as_posix()}'\n", encoding="utf-8"
    )
    make_scanned_pdf(tmp_path / "docs" / "scan.pdf", 2)
    argv = [
        "--input",
        str(tmp_path / "docs"),
        "--type",
        "pdf",
        "--method",
        "tesseract",
        "--formats",
        "md",
        "--layout",
        "mirror",
        "--output",
        str(tmp_path / "out"),
        "--settings",
        str(settings),
        "--json",
    ]

    assert _transcribe(*argv) == 0

    payload, _err = _one_line(capsys)
    database = (tmp_path / "out" / DATABASE_NAME).resolve()
    markdown = (tmp_path / "out" / "scan.md").resolve()
    assert payload["outputs"] == [str(markdown), str(database)]
    assert markdown.parent == database.parent
    assert database.is_file()
    assert not list((tmp_path / "out").rglob("*.txt"))


def test_keep_preprocessed_images_in_the_settings_exits_2(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("keep_preprocessed_images: true\n", encoding="utf-8")
    argv = ["--input", str(make_pdf()), "--settings", str(settings)]

    assert _transcribe(*argv, "--dry-run", "--json") == 2

    payload, err = _one_line(capsys)
    assert payload["items_total"] == 0
    assert "keep_preprocessed_images" in err
