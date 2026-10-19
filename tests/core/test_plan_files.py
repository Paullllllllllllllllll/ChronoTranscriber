"""Names given with --files are checked when the run is planned."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber.spec import to_argv
from tests.conftest import make_settings, make_spec


def _pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
    document.save(path)
    document.close()


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """An input folder with two PDFs and one image folder."""
    folder = tmp_path / "in"
    _pdf(folder / "a.pdf")
    _pdf(folder / "b.pdf")
    (folder / "scans").mkdir()
    (folder / "scans" / "p1.png").write_bytes(b"png")
    return folder


CASES: dict[str, tuple[dict[str, Any], tuple[str, ...], str]] = {
    "pdf": (
        {"type": "pdf", "method": "native"},
        ("a.pdf", "missing.pdf", "gone.pdf"),
        "'missing.pdf', 'gone.pdf'",
    ),
    "images": (
        {"type": "images", "method": "tesseract"},
        ("scans", "a.pdf", "nope"),
        "'a.pdf', 'nope'",
    ),
}


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_the_cli_exits_2_on_missing_files(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    dry_run: bool,
) -> None:
    flags, files, named = CASES[case]
    out = tmp_path / "out"
    spec = make_spec(docs, files=files, output=out, **flags)
    argv = ["transcribe", *to_argv(spec), "--json"]
    if dry_run:
        argv.append("--dry-run")
    before = sorted(docs.rglob("*"))

    assert entry.main(argv) == 2

    assert named in capsys.readouterr().err
    assert not out.exists()
    assert sorted(docs.rglob("*")) == before


def test_existing_names_select_their_items(tmp_path: Path, docs: Path) -> None:
    pdfs = core.plan(
        make_spec(docs, type="pdf", method="native", files=("b.pdf",), dry_run=True),
        make_settings(),
    )
    images = core.plan(
        make_spec(
            docs, type="images", method="tesseract", files=("scans",), dry_run=True
        ),
        make_settings(),
    )

    assert [item.name for item in pdfs.to_process] == ["b.pdf"]
    assert [item.name for item in images.to_process] == ["scans"]
