"""A folder with several items needs --all, --select or --files."""

from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF
import pytest

import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber.spec import resolve_spec
from tests.conftest import make_settings, make_spec


def _pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
    document.save(path)
    document.close()


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """An input folder with the PDFs p1, p2, p10 and ledger."""
    folder = tmp_path / "in"
    for name in ("p10", "ledger", "p2", "p1"):
        _pdf(folder / f"{name}.pdf")
    return folder


def _names(run_plan: core.RunPlan) -> list[str]:
    return sorted(item.name for item in run_plan.to_process)


@pytest.mark.parametrize(
    ("kind", "hint"),
    [
        ("pdf", "--all, --select or --files"),
        # auto mode refuses --files, so its hint leaves it out.
        ("auto", "neither --all nor --select"),
    ],
)
def test_several_items_without_a_selection_exit_2(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    hint: str,
) -> None:
    out = tmp_path / "out"

    code = entry.main(
        [
            "transcribe",
            "--input",
            str(docs),
            "--type",
            kind,
            "--output",
            str(out),
            "--dry-run",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert hint in captured.err
    assert ("--files" in captured.err) is (kind != "auto")
    assert json.loads(captured.out)["items_total"] == 0
    assert not out.exists()


def test_one_item_needs_no_selection(tmp_path: Path) -> None:
    _pdf(tmp_path / "in" / "only.pdf")

    run_plan = core.plan(make_spec(tmp_path / "in", dry_run=True), make_settings())

    assert _names(run_plan) == ["only.pdf"]


def test_all_takes_every_item(docs: Path) -> None:
    run_plan = core.plan(make_spec(docs, all=True, dry_run=True), make_settings())

    assert _names(run_plan) == ["ledger.pdf", "p1.pdf", "p10.pdf", "p2.pdf"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # Natural path order: ledger, p1, p2, p10.
        ("1", ["ledger.pdf"]),
        ("2-3", ["p1.pdf", "p2.pdf"]),
        ("4,1", ["ledger.pdf", "p10.pdf"]),
        ("ledg", ["ledger.pdf"]),
        ("P1", ["p1.pdf", "p10.pdf"]),
    ],
)
def test_select_numbers_items_in_natural_path_order(
    docs: Path, expression: str, expected: list[str]
) -> None:
    run_plan = core.plan(
        make_spec(docs, select=expression, dry_run=True), make_settings()
    )

    assert _names(run_plan) == expected


def test_select_keeps_the_discovery_order(docs: Path) -> None:
    every = core.plan(make_spec(docs, all=True, dry_run=True), make_settings())
    chosen = core.plan(make_spec(docs, select="2-4", dry_run=True), make_settings())

    order = [item.name for item in every.to_process]
    picked = [item.name for item in chosen.to_process]
    assert picked == [name for name in order if name in picked]


def test_out_of_range_parts_warn(docs: Path) -> None:
    run_plan = core.plan(make_spec(docs, select="1,9", dry_run=True), make_settings())

    assert _names(run_plan) == ["ledger.pdf"]
    assert any("9" in warning and "1-4" in warning for warning in run_plan.warnings)


@pytest.mark.parametrize("expression", ["9", "nothing-like-this"])
def test_select_matching_nothing_exits_2(
    docs: Path, capsys: pytest.CaptureFixture[str], expression: str
) -> None:
    code = entry.main(
        ["transcribe", "--input", str(docs), "--select", expression, "--dry-run"]
    )

    assert code == 2
    assert "matches --select" in capsys.readouterr().err


def test_the_selection_options_exclude_each_other(
    docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = entry.main(
        ["transcribe", "--input", str(docs), "--all", "--select", "1", "--dry-run"]
    )

    assert code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_a_settings_default_all_yields_to_a_select_flag(docs: Path) -> None:
    flags = {"input": docs, "select": "p2", "dry_run": True}
    spec = resolve_spec(flags, {"all": True}).spec

    run_plan = core.plan(spec, make_settings())

    assert spec.input.all is False
    assert _names(run_plan) == ["p2.pdf"]


def test_selection_order_sorts_numbers_by_value(docs: Path) -> None:
    found = core.discover(make_spec(docs, all=True))

    ordered = [item.name for item in core.selection_order(found)]

    assert ordered == ["ledger.pdf", "p1.pdf", "p2.pdf", "p10.pdf"]
