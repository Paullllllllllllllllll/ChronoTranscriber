"""Characterization of run control: resume, selection, layouts, pages, exit codes.

Each case builds its inputs, runs one or more stages through ``run_tool`` in
the same root and snapshots every stage. The case runs twice in fresh roots;
both snapshots must match before each stage is compared with
``golden/run_control/<case>/<stage>/``. Besides the standard snapshot, a stage
may record ``outputs/`` (the current content of named output files, for files
an earlier stage created) and ``report.txt`` (selected console lines, for
counts and errors the JSON line does not carry).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import WriteGuard
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import FakeLLM, Reply, permanent_error
from tests.characterization.golden import (
    check_files,
    compare,
    normalizer_for,
    snapshot,
)
from tests.characterization.inputs import (
    make_born_digital_pdf,
    make_image_folder,
    page_text,
)

pytestmark = pytest.mark.integration

CASE_PREFIX = "run_control"
PAGES = 2
RANGE_PAGES = 6
REVISED = "Revised transcription of the first page."

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ============================================================================
# Stages and case runner
# ============================================================================
@dataclasses.dataclass
class Stage:
    """One ``run_tool`` call of a case and its golden files."""

    result: RunResult
    files: dict[str, str]


Build = Callable[[Path], dict[str, Stage]]


def stage(
    result: RunResult,
    *,
    outputs: Iterable[Path] = (),
    report: Sequence[str] = (),
) -> Stage:
    """Snapshot *result*; add the content of *outputs* and the *report* lines.

    *report* lists substrings; every console line (stdout and stderr) that
    contains one of them is recorded in ``report.txt``.
    """
    files = snapshot(result)
    kept = sorted(outputs)
    if kept:
        current = snapshot(dataclasses.replace(result, created=kept))
        files.update(
            {
                "outputs/" + name.removeprefix("files/"): content
                for name, content in current.items()
                if name.startswith("files/")
            }
        )
    if report:
        files["report.txt"] = _report(result, report)
    return Stage(result, files)


def _report(result: RunResult, needles: Sequence[str]) -> str:
    norm = normalizer_for(result)
    lines = [
        norm.text(_ANSI_RE.sub("", line).rstrip())
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if any(needle in line for needle in needles)
    ]
    return "".join(f"{line}\n" for line in lines)


@pytest.fixture
def run_case(
    case_root: Callable[[str], Path], write_guard: WriteGuard
) -> Callable[[str, Build], dict[str, Stage]]:
    """Return ``run(case, build)``: run twice, compare, check every stage."""

    def run(case: str, build: Build) -> dict[str, Stage]:
        first = build(case_root("run1"))
        second = build(case_root("run2"))
        problems = [
            f"[{name}] {problem}"
            for name in sorted(set(first) | set(second))
            for problem in compare(
                first[name].files if name in first else {},
                second[name].files if name in second else {},
            )
        ]
        assert not problems, "two identical runs differ:\n" + "\n".join(problems)
        for name, current in second.items():
            check_files(
                f"{CASE_PREFIX}/{case}/{name}",
                current.files,
                on_write=write_guard.consume,
            )
        return second

    return run


# ============================================================================
# Commands
# ============================================================================
def _transcribe(
    llm: FakeLLM,
    root: Path,
    *,
    input: Path,
    type: str,
    method: str,
    config: Mapping[str, Mapping[str, Any]] | None = None,
    **flags: Any,
) -> RunResult:
    return run_tool(
        "transcribe",
        llm=llm,
        root=root,
        config=config,
        input=input,
        output=root / "out",
        type=type,
        method=method,
        **flags,
    )


def _general(**values: Any) -> dict[str, dict[str, Any]]:
    return {"paths_config": {"general": values}}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pdf_folder(folder: Path, names: Sequence[str]) -> list[Path]:
    """Write one born-digital PDF per relative name under *folder*."""
    return [make_born_digital_pdf(folder / name, PAGES) for name in names]


# ============================================================================
# Resume, --force and resume_mode
# ============================================================================
@pytest.mark.parametrize(
    ("case", "flags", "config", "reprocessed"),
    [
        ("resume_skip", {"resume": True}, None, False),
        ("force", {"force": True}, None, True),
        (
            "resume_mode_overwrite_config",
            {},
            _general(resume_mode="overwrite"),
            True,
        ),
    ],
)
def test_second_run_on_completed_item(
    case: str,
    flags: Mapping[str, Any],
    config: Mapping[str, Mapping[str, Any]] | None,
    reprocessed: bool,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_llm.pages.clear()
        pdf = make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)
        first = _transcribe(fake_llm, root, input=pdf, type="pdfs", method="gpt")
        assert first.exit_code == 0, first.stdout + first.stderr
        fake_llm.script(0, transcription=REVISED)
        second = _transcribe(
            fake_llm,
            root,
            input=pdf,
            type="pdfs",
            method="gpt",
            config=config,
            **flags,
        )
        return {
            "first": stage(first),
            "second": stage(
                second,
                outputs=[second.output(pdf), second.paths(pdf).database],
                report=("Resume", "Beginning processing"),
            ),
        }

    second = run_case(case, build)["second"].result

    assert second.exit_code == 0
    assert second.created == []
    assert len(second.calls) == (PAGES if reprocessed else 0)
    assert second.summary is not None
    count = 1 if reprocessed else 0
    assert (second.summary["items_total"], second.summary["items_complete"]) == (
        count,
        count,
    )
    output = second.output(second.root / "docs" / "doc.pdf")
    assert (REVISED in _text(output)) == reprocessed


# ============================================================================
# Dry run
# ============================================================================
def test_dry_run_on_partly_completed_folder(
    fake_llm: FakeLLM, run_case: Callable[[str, Build], dict[str, Stage]]
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        docs = root / "docs"
        _pdf_folder(docs, ("a.pdf", "b.pdf", "c.pdf"))
        done = _transcribe(
            fake_llm, root, input=docs, type="pdfs", method="native", files=["a.pdf"]
        )
        assert done.exit_code == 0, done.stdout + done.stderr
        completed = stage(done)
        planned = _transcribe(
            fake_llm, root, input=docs, type="pdfs", method="native", dry_run=True
        )
        return {
            "completed_one": completed,
            "dry_run": stage(planned, report=("[DRY RUN]", "Resume mode")),
        }

    planned = run_case("dry_run_partly_completed", build)["dry_run"].result

    assert planned.exit_code == 0
    assert planned.created == []
    assert planned.calls == []
    assert planned.summary is not None
    assert planned.summary["dry_run"] is True
    assert planned.summary["items_total"] == 2
    assert "3 discovered; 2 would be processed, 1 skipped" in planned.stderr


# ============================================================================
# Exit codes
# ============================================================================
def _exit_success(llm: FakeLLM, root: Path) -> RunResult:
    pdf = make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)
    return _transcribe(llm, root, input=pdf, type="pdfs", method="native")


def _exit_item_fails(llm: FakeLLM, root: Path) -> RunResult:
    llm.script(1, always=Reply(permanent_error()))
    pdf = make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)
    return _transcribe(llm, root, input=pdf, type="pdfs", method="gpt")


def _exit_missing_input(llm: FakeLLM, root: Path) -> RunResult:
    return _transcribe(
        llm, root, input=root / "docs" / "missing.pdf", type="pdfs", method="native"
    )


def _exit_invalid_pages(llm: FakeLLM, root: Path) -> RunResult:
    pdf = make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)
    return _transcribe(llm, root, input=pdf, type="pdfs", method="native", pages="3-1")


@pytest.mark.parametrize(
    ("case", "run", "exit_code"),
    [
        ("exit_0_success", _exit_success, 0),
        ("exit_1_item_fails", _exit_item_fails, 1),
        ("exit_2_missing_input", _exit_missing_input, 2),
        ("exit_2_invalid_pages", _exit_invalid_pages, 2),
    ],
)
def test_exit_code(
    case: str,
    run: Callable[[FakeLLM, Path], RunResult],
    exit_code: int,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_llm.pages.clear()
        return {"run": stage(run(fake_llm, root), report=("[ERROR]",))}

    result = run_case(case, build)["run"].result

    assert result.exit_code == exit_code, result.stdout + result.stderr
    assert result.summary is not None
    if exit_code == 2:
        assert result.created == []
        assert result.calls == []
        assert "Invalid arguments" in result.stderr
        assert result.stdout.count("\n") == 1
    else:
        assert result.summary["items_failed"] == exit_code


# ============================================================================
# Item selection and output layout
# ============================================================================
NESTED = ("a.pdf", "sub/b.pdf", "sub/deep/c.pdf")


def test_files_selection(
    fake_llm: FakeLLM, run_case: Callable[[str, Build], dict[str, Stage]]
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        docs = root / "docs"
        _pdf_folder(docs, ("a.pdf", "b.pdf", "c.pdf"))
        result = _transcribe(
            fake_llm,
            root,
            input=docs,
            type="pdfs",
            method="native",
            files=["a.pdf", "c.pdf"],
        )
        return {"run": stage(result)}

    result = run_case("files_selection", build)["run"].result

    assert result.exit_code == 0
    docs = result.root / "docs"
    assert result.output(docs / "a.pdf") in result.created
    assert result.output(docs / "c.pdf") in result.created
    assert not result.output(docs / "b.pdf").exists()
    assert result.summary is not None
    assert result.summary["items_complete"] == 2


@pytest.mark.parametrize(
    ("case", "recursive", "output_mode", "selected"),
    [
        ("non_recursive_pdfs", False, None, ("a.pdf",)),
        ("recursive_pdfs_hash", True, None, NESTED),
        ("recursive_pdfs_mirror", True, "mirror", NESTED),
    ],
)
def test_nested_pdf_folder(
    case: str,
    recursive: bool,
    output_mode: str | None,
    selected: tuple[str, ...],
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        docs = root / "docs"
        _pdf_folder(docs, NESTED)
        result = _transcribe(
            fake_llm,
            root,
            input=docs,
            type="pdfs",
            method="native",
            recursive=recursive,
            output_mode=output_mode,
        )
        return {"run": stage(result)}

    result = run_case(case, build)["run"].result

    assert result.exit_code == 0
    docs = result.root / "docs"
    for name in NESTED:
        output = result.output(docs / name)
        assert (output in result.created) == (name in selected)
    if output_mode == "mirror":
        nested = result.output(docs / "sub" / "deep" / "c.pdf")
        assert nested.relative_to(result.root / "out").parts[:2] == ("sub", "deep")


def test_recursive_image_folders(
    fake_llm: FakeLLM, run_case: Callable[[str, Build], dict[str, Stage]]
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_llm.pages.clear()
        scans = root / "scans"
        for volume in ("vol1", "vol2"):
            make_image_folder(scans / volume, PAGES)
        result = _transcribe(
            fake_llm, root, input=scans, type="images", method="gpt", recursive=True
        )
        return {"run": stage(result)}

    result = run_case("recursive_image_folders", build)["run"].result

    assert result.exit_code == 0
    assert len(result.calls) == 2 * PAGES
    scans = result.root / "scans"
    for volume in ("vol1", "vol2"):
        assert result.output(scans / volume) in result.created


def test_outputs_beside_input(
    fake_llm: FakeLLM, run_case: Callable[[str, Build], dict[str, Stage]]
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        pdf = make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)
        result = _transcribe(
            fake_llm,
            root,
            input=pdf,
            type="pdfs",
            method="native",
            config=_general(input_paths_is_output_path=True),
        )
        return {"run": stage(result)}

    result = run_case("input_paths_is_output_path", build)["run"].result

    assert result.exit_code == 0
    docs = result.root / "docs"
    output = result.output(docs / "doc.pdf")
    assert output.parent == docs.resolve()
    assert output in result.created
    assert not any(path.is_relative_to(result.root / "out") for path in result.created)


# ============================================================================
# Page ranges
# ============================================================================
PAGE_FORMS: dict[str, tuple[str, tuple[int, ...]]] = {
    "5": ("5", (0, 1, 2, 3, 4)),
    "first:2": ("first_2", (0, 1)),
    "last:1": ("last_1", (5,)),
    "2-3": ("2_to_3", (1, 2)),
    "3-": ("3_to_end", (2, 3, 4, 5)),
    "-2": ("to_2", (0, 1)),
    "1,3": ("1_and_3", (0, 2)),
}
"""``--pages`` value: case suffix and selected 0-based pages of six."""


@pytest.mark.parametrize("pages", list(PAGE_FORMS))
def test_pages_on_pdf(
    pages: str,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    suffix, selected = PAGE_FORMS[pages]

    def build(root: Path) -> dict[str, Stage]:
        pdf = make_born_digital_pdf(root / "docs" / "doc.pdf", RANGE_PAGES)
        result = _transcribe(
            fake_llm, root, input=pdf, type="pdfs", method="native", pages=pages
        )
        return {"run": stage(result)}

    result = run_case(f"pages_pdf_{suffix}", build)["run"].result

    assert result.exit_code == 0
    text = _text(result.output(result.root / "docs" / "doc.pdf"))
    for index in range(RANGE_PAGES):
        assert (page_text(index) in text) == (index in selected)


@pytest.mark.parametrize("pages", list(PAGE_FORMS))
def test_pages_on_image_folder(
    pages: str,
    fake_llm: FakeLLM,
    run_case: Callable[[str, Build], dict[str, Stage]],
) -> None:
    suffix, selected = PAGE_FORMS[pages]

    def build(root: Path) -> dict[str, Stage]:
        fake_llm.pages.clear()
        folder = make_image_folder(root / "scans", RANGE_PAGES)
        result = _transcribe(
            fake_llm, root, input=folder, type="images", method="gpt", pages=pages
        )
        return {"run": stage(result)}

    result = run_case(f"pages_images_{suffix}", build)["run"].result

    assert result.exit_code == 0
    assert [call["page"] for call in result.calls] == list(selected)
