"""Characterization of resume for image folders, EPUB, MOBI, audio and auto mode.

Each case runs four stages through ``run_tool`` in one root: a full run
(``first``), a resumed run on the completed item (``second``), then, after
the test recreates the files of an interrupted run, a ``--dry-run``
(``dry_run``) and a resumed run (``resumed``). An interrupted image folder,
audio recording or PDF keeps its first page or chunk done in the store, has
its other units returned to pending and loses its rendered file; an
interrupted EPUB or MOBI has its single unit returned to pending. The case
runs twice in fresh roots; both snapshots must match before each stage is
compared with ``golden/resume/<case>/<stage>/``. Stages record the console
lines about resume in ``report.txt`` and the current content of the outputs
and, after the interruption, of the store under ``outputs/``; audio stages
add ``ffmpeg.json``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import WriteGuard
from chronotranscriber.output.store import document_name, open_store
from tests.characterization.adapter import ItemPaths, RunResult, run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_audio import FakeAudio
from tests.characterization.fakes_ocr import (
    FakeTesseract,
    block_real_tesseract,
    tesseract_cmd_config,
)
from tests.characterization.golden import check_files, compare, normalizer_for
from tests.characterization.inputs import make_born_digital_pdf, make_image_folder
from tests.characterization.inputs_audio import make_wav
from tests.characterization.inputs_ebooks import (
    CHAPTERS,
    MOBI_PARAGRAPHS,
    make_epub,
    make_mobi,
)
from tests.characterization.test_audio import transcribe_audio
from tests.characterization.test_run_control import Stage, stage

pytestmark = pytest.mark.integration

CASE_PREFIX = "resume"
PAGES = 3
LONG_SECONDS = 75
CHUNK_STARTS = [0, 30, 60]
OVERSIZED = {"chunking": {"max_request_bytes": 12_000}}
"""Audio config that splits the long recording into three chunks."""

RUN_LINES = ("Resume", "Beginning processing", "Processing item", "Nothing to do")
DRY_RUN_LINES = ("[DRY RUN]", "Resume")

Run = Callable[..., tuple[RunResult, dict[str, str]]]
"""Run the case's command with extra flags; return the result and extra files."""

Build = Callable[[Path], dict[str, Stage]]
CaseRunner = Callable[[str, Build], dict[str, Stage]]


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture(autouse=True)
def missing_tesseract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point pytesseract at a missing executable for every case."""
    return block_real_tesseract(monkeypatch, tmp_path)


@pytest.fixture(autouse=True)
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> FakeTesseract:
    """Install the fake Tesseract engine."""
    return FakeTesseract().install(monkeypatch)


@pytest.fixture
def fake_audio(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeAudio]:
    """Install the audio fakes over the provider fakes and return them."""
    controller = FakeAudio()
    controller.install(monkeypatch)
    controller.reset_records()
    yield controller
    controller.reset_records()


@pytest.fixture
def run_case(case_root: Callable[[str], Path], write_guard: WriteGuard) -> CaseRunner:
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
# Stages
# ============================================================================
INTERRUPTED = "interrupted"
"""Command recorded by the run that :func:`interrupt` leaves unfinished."""


async def _reset_units(database: Path, name: str) -> None:
    async with open_store(database.parent) as store:
        await store.begin_run(INTERRUPTED)
        document = store.document(name)
        assert document is not None, f"no document {name!r} in {database}"
        lease = await store.acquire(document.doc_id)
        assert lease is not None
        units = store.units(document.doc_id)
        idxs = [unit.idx for unit in units if unit.idx > 0] or [0]
        assert await store.reset_units(lease, idxs=idxs) == len(idxs)


def interrupt(paths: ItemPaths) -> None:
    """Leave *paths* as an interrupted run does.

    The rendered file goes, and a run that never ends returns every unit
    after the first page or chunk to pending through the store; an item of
    one unit (EPUB, MOBI) has that unit returned.
    """
    name = document_name(paths.output, paths.database.parent)
    paths.output.unlink()
    asyncio.run(_reset_units(paths.database, name))


def resume_stages(
    run: Run, items: Sequence[Path], partial: Sequence[Path]
) -> dict[str, Stage]:
    """Run the four stages of a case on *items*, interrupting *partial*.

    Each stage is snapshotted right after its run, so ``outputs/`` shows the
    files as that run left them.
    """

    def staged(
        flags: dict[str, Any],
        outputs: Sequence[Path] = (),
        report: Sequence[str] = (),
    ) -> Stage:
        result, extra = run(**flags)
        current = stage(result, outputs=outputs, report=report)
        current.files.update(extra)
        return current

    first = staged({})
    assert first.result.exit_code == 0, first.result.stdout + first.result.stderr
    paths = {item: first.result.paths(item) for item in items}
    outputs = [entry.output for entry in paths.values()]
    second = staged({"resume": True}, outputs, RUN_LINES)
    for item in partial:
        interrupt(paths[item])
    stores = sorted({paths[item].database for item in partial})
    planned = staged({"resume": True, "dry_run": True}, stores, DRY_RUN_LINES)
    resumed = staged({"resume": True}, stores, RUN_LINES)
    return {"first": first, "second": second, "dry_run": planned, "resumed": resumed}


def _document_run(llm: FakeLLM, root: Path, missing: Path, **base: Any) -> Run:
    def run(**flags: Any) -> tuple[RunResult, dict[str, str]]:
        result = run_tool(
            "transcribe",
            llm=llm,
            root=root,
            config={"image_processing_config": tesseract_cmd_config(missing)},
            output=root / "out",
            **base,
            **flags,
        )
        return result, {}

    return run


def _counts(result: RunResult) -> tuple[int, int]:
    assert result.summary is not None, result.stdout + result.stderr
    return result.summary["items_total"], result.summary["items_complete"]


def _heading(page: int) -> str:
    """Return the heading the fake chat model writes on 0-based *page*."""
    return f"Market Records, Part {page + 1:02d}"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _check_completed(second: RunResult) -> None:
    assert second.exit_code == 0
    assert second.created == []
    assert second.calls == []


# ============================================================================
# Image folder
# ============================================================================
def test_resume_image_folder_gpt(
    fake_llm: FakeLLM,
    missing_tesseract: Path,
    run_case: CaseRunner,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_llm.pages.clear()
        folder = make_image_folder(root / "scans", PAGES)
        run = _document_run(
            fake_llm, root, missing_tesseract, input=folder, type="images", method="gpt"
        )
        return resume_stages(run, [folder], [folder])

    stages = run_case("image_folder_gpt", build)
    first, second = stages["first"].result, stages["second"].result
    planned, resumed = stages["dry_run"].result, stages["resumed"].result

    assert [call["page"] for call in first.calls] == list(range(PAGES))
    _check_completed(second)
    assert _counts(second) == (0, 0)
    assert planned.calls == []
    assert planned.created == []
    assert _counts(planned) == (1, 0)
    assert "1 discovered; 1 would be processed, 0 skipped" in planned.stderr
    assert [call["page"] for call in resumed.calls] == [1, 2]
    output = resumed.output(resumed.root / "scans")
    assert resumed.created == [output]
    assert _counts(resumed) == (1, 1)
    assert all(_heading(page) in _text(output) for page in range(PAGES))


# ============================================================================
# EPUB and MOBI
# ============================================================================
@pytest.mark.parametrize(
    ("case", "name", "kind", "make", "expected"),
    [
        ("epub_native", "annals.epub", "epubs", make_epub, CHAPTERS[0][1]),
        ("mobi_native", "notes.mobi", "mobis", make_mobi, MOBI_PARAGRAPHS[0]),
    ],
    ids=["epub", "mobi"],
)
def test_resume_ebook_native(
    case: str,
    name: str,
    kind: str,
    make: Callable[[Path], Path],
    expected: str,
    fake_llm: FakeLLM,
    missing_tesseract: Path,
    run_case: CaseRunner,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        book = make(root / "books" / name)
        run = _document_run(
            fake_llm, root, missing_tesseract, input=book, type=kind, method="native"
        )
        return resume_stages(run, [book], [book])

    stages = run_case(case, build)
    second, planned = stages["second"].result, stages["dry_run"].result
    resumed = stages["resumed"].result

    _check_completed(second)
    assert _counts(second) == (0, 0)
    assert _counts(planned) == (1, 0)
    assert "1 discovered; 1 would be processed, 0 skipped" in planned.stderr
    paths = resumed.paths(resumed.root / "books" / name)
    assert resumed.created == [paths.output]
    assert _counts(resumed) == (1, 1)
    assert expected in " ".join(_text(paths.output).split())


# ============================================================================
# Audio
# ============================================================================
def test_resume_audio_chunked(
    fake_llm: FakeLLM,
    fake_audio: FakeAudio,
    run_case: CaseRunner,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        audio = make_wav(root / "audio" / "talk.wav", LONG_SECONDS)

        def run(**flags: Any) -> tuple[RunResult, dict[str, str]]:
            outcome = transcribe_audio(
                fake_llm,
                fake_audio,
                root,
                input=audio,
                method="audio-api",
                audio_config=OVERSIZED,
                **flags,
            )
            norm = normalizer_for(outcome.result)
            ffmpeg = json.dumps(
                norm.data(outcome.ffmpeg), indent=2, sort_keys=True, ensure_ascii=False
            )
            return outcome.result, {"ffmpeg.json": ffmpeg + "\n"}

        return resume_stages(run, [audio], [audio])

    stages = run_case("audio_chunked", build)
    first, second = stages["first"].result, stages["second"].result
    planned, resumed = stages["dry_run"].result, stages["resumed"].result

    def starts(result: RunResult) -> list[int]:
        return [call["chunk"]["start_second"] for call in result.calls]

    assert starts(first) == CHUNK_STARTS
    _check_completed(second)
    assert _counts(second) == (0, 0)
    assert planned.calls == []
    assert _counts(planned) == (1, 0)
    assert "talk.wav (3 chunk(s))" in planned.stderr
    assert starts(resumed) == CHUNK_STARTS[1:]
    output = resumed.output(resumed.root / "audio" / "talk.wav")
    assert resumed.created == [output]
    assert _counts(resumed) == (1, 1)
    text = _text(output)
    assert all(f"Recording from second {start} " in text for start in CHUNK_STARTS)


# ============================================================================
# Auto mode
# ============================================================================
AUTO_ITEMS = ("born.pdf", "scans", "annals.epub", "notes.mobi")
AUTO_PARTIAL = ("born.pdf", "scans")


def test_resume_auto_mixed_folder(
    fake_llm: FakeLLM,
    missing_tesseract: Path,
    run_case: CaseRunner,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_llm.pages.clear()
        mixed = root / "mixed"
        make_born_digital_pdf(mixed / "born.pdf", PAGES)
        make_image_folder(mixed / "scans", PAGES)
        make_epub(mixed / "annals.epub")
        make_mobi(mixed / "notes.mobi")
        run = _document_run(fake_llm, root, missing_tesseract, input=mixed, auto=True)
        return resume_stages(
            run,
            [mixed / name for name in AUTO_ITEMS],
            [mixed / name for name in AUTO_PARTIAL],
        )

    stages = run_case("auto_mixed_folder", build)
    first, second = stages["first"].result, stages["second"].result
    planned, resumed = stages["dry_run"].result, stages["resumed"].result
    mixed = resumed.root / "mixed"

    assert len(first.calls) == 2 * PAGES
    assert _counts(first) == (4, 4)
    _check_completed(second)
    assert "Resume: skipping 4 already-completed file(s) (0 remaining)" in (
        second.stderr
    )
    assert planned.calls == []
    assert _counts(planned) == (2, 0)
    assert "Resume: skipping 2 already-completed file(s) (2 remaining)" in (
        planned.stderr
    )
    assert sorted(call["page"] for call in resumed.calls) == [1, 1, 2, 2]
    assert sorted(resumed.created) == sorted(
        resumed.output(mixed / name) for name in AUTO_PARTIAL
    )
    assert _counts(resumed) == (2, 2)
    for name in AUTO_PARTIAL:
        text = _text(resumed.output(mixed / name))
        assert all(_heading(page) in text for page in range(PAGES))
