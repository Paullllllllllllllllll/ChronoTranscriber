"""Characterization of repair and ``--retry-errors``.

Each case first transcribes a four-page scanned PDF with the ``llm`` method.
The fake LLM gives page 2 a permanent error, page 3 the no-text flag and
page 4 the not-possible flag, so the output holds one placeholder line of
each kind. The scripts are then healed, so every page re-sent afterwards
gets its transcription, and the stage under test runs: a resume without and
with ``--retry-errors``, ``repair`` in synchronous mode with each failure
selection, and ``repair --batch``, which submits, waits for and reads a batch
from the fake batch service in one command. The batch case runs once per
provider (OpenAI, Anthropic and Google); both of its commands use that
provider.

Every stage is snapshotted: the files the run created, the files it changed
(under ``changed/``), the JSON summary and ``requests.json``; the batch case
adds the batch service's client constructions, API calls and submitted
request bodies under ``batch``. Each case runs twice in fresh roots; both
snapshots must match before they are compared with ``golden/repair/``.
The batch service reads each OpenAI request's nested body, so a request
with a response format gets schema JSON back.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import WriteGuard
from tests.characterization.fakes import FakeLLM, Reply, permanent_error
from tests.characterization.fakes_batch import FakeBatchService, Provider
from tests.characterization.golden import (
    check_files,
    compare,
    normalizer_for,
    snapshot,
)
from tests.characterization.inputs import make_scanned_pdf
from tests.characterization.runner import Command, RunResult, run_cli

pytestmark = pytest.mark.integration

PAGES = 4
ERROR_PAGE = 1
NO_TEXT_PAGE = 2
NOT_POSSIBLE_PAGE = 3
FAILED_PAGES = (ERROR_PAGE, NO_TEXT_PAGE, NOT_POSSIBLE_PAGE)

ERROR_MARK = "[transcription error"
NO_TEXT_MARK = "[No transcribable text"
NOT_POSSIBLE_MARK = "[Transcription not possible"


@dataclasses.dataclass
class Stage:
    """One command of a case: its result, snapshot and changed files."""

    result: RunResult
    files: dict[str, str]
    changed: list[Path]
    batch_calls: list[dict[str, Any]]


@pytest.fixture(autouse=True)
def _empty_project_context(
    tmp_path: Path, project_context: Callable[[Path], None]
) -> None:
    """Replace the repository's context folder with an empty one."""
    folder = tmp_path / "project_context"
    folder.mkdir()
    project_context(folder)


@pytest.fixture
def batch_service(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> FakeBatchService:
    """Install the fake batch service."""
    return FakeBatchService(fake_llm).install(monkeypatch)


# ============================================================================
# Stages and snapshots
# ============================================================================
def _contents(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _stage_files(
    result: RunResult,
    changed: list[Path],
    service: FakeBatchService | None,
) -> dict[str, str]:
    replacements = service.id_replacements() if service is not None else {}
    files = snapshot(result, replacements=replacements)
    if changed:
        edited = snapshot(
            dataclasses.replace(result, created=changed), replacements=replacements
        )
        files.update(
            {
                "changed/" + name.removeprefix("files/"): content
                for name, content in edited.items()
                if name.startswith("files/")
            }
        )
    if service is not None:
        requests = json.loads(files["requests.json"])
        records = service.request_records()
        requests["batch"] = normalizer_for(result, replacements).data(records)
        files["requests.json"] = (
            json.dumps(requests, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
    return files


def run_stage(
    command: Command,
    *,
    service: FakeBatchService | None = None,
    **kwargs: Any,
) -> Stage:
    """Run one command and snapshot what it created and changed in the root."""
    root: Path = kwargs["root"]
    before = _contents(root)
    if service is not None:
        service.reset_records()
    result = run_cli(command, **kwargs)
    after = _contents(root)
    changed = sorted(
        path for path, data in after.items() if path in before and before[path] != data
    )
    batch_calls = [dict(call) for call in service.calls] if service else []
    return Stage(result, _stage_files(result, changed, service), changed, batch_calls)


def run_twice(
    case_root: Callable[[str], Path],
    build: Callable[[Path], Mapping[str, Stage]],
) -> dict[str, Stage]:
    """Run *build* in two fresh roots and return the stages of the second.

    Fails when the snapshots of the two runs differ.
    """
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
    return dict(second)


def check_case(case: str, stages: Mapping[str, Stage], guard: WriteGuard) -> None:
    """Compare every stage with ``golden/repair/<case>/<stage>/``."""
    for name, stage in stages.items():
        check_files(f"repair/{case}/{name}", stage.files, on_write=guard.consume)


# ============================================================================
# Commands
# ============================================================================
def _pdf(root: Path) -> Path:
    return make_scanned_pdf(root / "docs" / "doc.pdf", PAGES)


def _script_failures(llm: FakeLLM) -> None:
    llm.pages.clear()
    llm.script(ERROR_PAGE, always=Reply(exception=permanent_error()))
    llm.script(NO_TEXT_PAGE, form="no_text")
    llm.script(NOT_POSSIBLE_PAGE, form="not_possible")


def _heal(llm: FakeLLM) -> None:
    llm.script(ERROR_PAGE, always=None)
    llm.script(NO_TEXT_PAGE, form="auto")
    llm.script(NOT_POSSIBLE_PAGE, form="auto")


def transcribe(
    llm: FakeLLM,
    root: Path,
    *,
    retry_errors: bool = False,
    **model: Any,
) -> Stage:
    """Run ``transcribe --method llm`` on the case's PDF; a rerun resumes.

    *model* holds the ``provider`` and ``model`` options, if any.
    """
    pdf = root / "docs" / "doc.pdf"
    if not pdf.exists():
        _pdf(root)
    return run_stage(
        "transcribe",
        llm=llm,
        root=root,
        input=pdf,
        output=root / "out",
        type="pdf",
        method="llm",
        retry_errors=retry_errors,
        **model,
    )


def first_run(llm: FakeLLM, root: Path, **model: Any) -> Stage:
    """Transcribe with one page of each failure kind, then heal the scripts."""
    _script_failures(llm)
    stage = transcribe(llm, root, **model)
    result = stage.result
    assert result.exit_code == 1, result.stdout + result.stderr
    assert [call["page"] for call in result.calls] == list(range(PAGES))
    text = _output(result).read_text(encoding="utf-8")
    for mark in (ERROR_MARK, NO_TEXT_MARK, NOT_POSSIBLE_MARK):
        assert mark in text
    _heal(llm)
    return stage


def _output(result: RunResult) -> Path:
    return result.output(result.root / "docs" / "doc.pdf")


def repair(
    llm: FakeLLM,
    root: Path,
    first: Stage,
    *,
    service: FakeBatchService | None = None,
    **options: Any,
) -> Stage:
    """Run ``repair --input`` on the first run's output."""
    return run_stage(
        "repair",
        service=service,
        llm=llm,
        root=root,
        input=_output(first.result),
        **options,
    )


# ============================================================================
# Assertions
# ============================================================================
def _pages(stage: Stage) -> list[int | None]:
    return [call["page"] for call in stage.result.calls]


def _part(page: int) -> str:
    return f"Market Records, Part {page + 1:02d}"


def _assert_repaired(
    stages: Mapping[str, Stage],
    repaired: tuple[int, ...],
    marks_left: tuple[str, ...],
) -> None:
    stage = stages["repair"]
    result = stage.result
    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.summary is not None
    assert (result.summary["repaired"], result.summary["failed"]) == (
        len(repaired),
        0,
    )
    output = _output(stages["first"].result)
    assert output in stage.changed
    text = output.read_text(encoding="utf-8")
    for page in repaired:
        assert _part(page) in text
    for mark in marks_left:
        assert mark in text
    positions = [text.index(_part(page)) for page in (0, *repaired)]
    assert positions == sorted(positions)
    assert result.created == []
    assert sorted(path.name for path in stage.changed) == sorted(
        [output.name, "chronotranscriber.sqlite"]
    )


# ============================================================================
# Cases
# ============================================================================
def test_retry_errors(
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        stages = {"first": first_run(fake_llm, root)}
        stages["resume"] = transcribe(fake_llm, root)
        stages["retry_errors"] = transcribe(fake_llm, root, retry_errors=True)
        return stages

    stages = run_twice(case_root, build)

    resumed = stages["resume"]
    assert resumed.result.exit_code == 0, resumed.result.stdout
    assert resumed.result.calls == []
    assert resumed.result.created == []
    assert resumed.changed == []

    retried = stages["retry_errors"]
    assert retried.result.exit_code == 0, retried.result.stdout
    assert _pages(retried) == [ERROR_PAGE]
    output = _output(retried.result)
    assert output in retried.changed
    text = output.read_text(encoding="utf-8")
    assert ERROR_MARK not in text
    assert NO_TEXT_MARK in text
    assert NOT_POSSIBLE_MARK in text
    assert text.index(_part(0)) < text.index(_part(ERROR_PAGE))
    assert text.index(_part(ERROR_PAGE)) < text.index(NO_TEXT_MARK)
    check_case("retry_errors", stages, write_guard)


def _sync_case(
    llm: FakeLLM,
    case_root: Callable[[str], Path],
    options: Callable[[Stage], dict[str, Any]],
) -> dict[str, Stage]:
    def build(root: Path) -> dict[str, Stage]:
        first = first_run(llm, root)
        return {"first": first, "repair": repair(llm, root, first, **options(first))}

    return run_twice(case_root, build)


def check_repair(case: str, stages: Mapping[str, Stage], guard: WriteGuard) -> None:
    """Compare the repair stage with its golden; the first run has its own."""
    check_case(case, {"repair": stages["repair"]}, guard)


def test_sync_errors_only(
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    stages = _sync_case(fake_llm, case_root, lambda _first: {"errors_only": True})

    assert _pages(stages["repair"]) == [ERROR_PAGE]
    _assert_repaired(
        stages, (ERROR_PAGE,), marks_left=(NO_TEXT_MARK, NOT_POSSIBLE_MARK)
    )
    check_repair("sync_errors_only", stages, write_guard)


def test_sync_indices(
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    stages = _sync_case(
        fake_llm,
        case_root,
        lambda _first: {"all_failures": True, "indices": str(NOT_POSSIBLE_PAGE)},
    )

    assert _pages(stages["repair"]) == [NOT_POSSIBLE_PAGE]
    _assert_repaired(
        stages, (NOT_POSSIBLE_PAGE,), marks_left=(ERROR_MARK, NO_TEXT_MARK)
    )
    check_repair("sync_indices", stages, write_guard)


def test_sync_not_possible_and_no_text(
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    stages = _sync_case(
        fake_llm, case_root, lambda _first: {"not_possible": True, "no_text": True}
    )

    assert _pages(stages["repair"]) == [NO_TEXT_PAGE, NOT_POSSIBLE_PAGE]
    _assert_repaired(
        stages, (NO_TEXT_PAGE, NOT_POSSIBLE_PAGE), marks_left=(ERROR_MARK,)
    )
    check_repair("sync_not_possible_no_text", stages, write_guard)


def test_sync_all_failures(
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    stages = _sync_case(fake_llm, case_root, lambda _first: {"all_failures": True})

    assert _pages(stages["repair"]) == list(FAILED_PAGES)
    _assert_repaired(stages, FAILED_PAGES, marks_left=())
    check_repair("sync_all_failures", stages, write_guard)


@dataclasses.dataclass(frozen=True)
class BatchCase:
    """Provider, model, golden name and expected batch API calls of a case."""

    provider: Provider
    model: dict[str, str]
    golden: str
    calls: tuple[str, ...]


BATCH_CASES = (
    BatchCase(
        "openai",
        {},
        "batch_all_failures",
        (
            "openai.files.create",
            "openai.batches.create",
            "openai.batches.retrieve",
            "openai.batches.retrieve",
            "openai.batches.retrieve",
            "openai.files.content",
            "openai.batches.retrieve",
            "openai.files.delete",
            "openai.files.delete",
        ),
    ),
    BatchCase(
        "anthropic",
        {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        "batch_all_failures_anthropic",
        (
            "anthropic.messages.batches.create",
            "anthropic.messages.batches.retrieve",
            "anthropic.messages.batches.retrieve",
            "anthropic.messages.batches.retrieve",
            "anthropic.messages.batches.results",
        ),
    ),
    BatchCase(
        "google",
        {"provider": "google", "model": "gemini-3-flash-preview"},
        "batch_all_failures_google",
        (
            "google.files.upload",
            "google.batches.create",
            "google.batches.get",
            "google.batches.get",
            "google.batches.get",
            "google.files.download",
            "google.batches.get",
            "google.files.delete",
            "google.files.delete",
        ),
    ),
)
"""Both commands of a case run on its provider; openai keeps the default model.

After a successful finalization the provider files are deleted: the
OpenAI case retrieves the batch once more and deletes its input and output
files, the Google case gets the batch once more and finds no file to delete.
"""


@pytest.mark.parametrize("case", BATCH_CASES, ids=lambda case: case.provider)
def test_batch_all_failures(
    case: BatchCase,
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        batch_service.reset()
        first = first_run(fake_llm, root, **case.model)
        return {
            "first": first,
            "repair": repair(
                fake_llm,
                root,
                first,
                service=batch_service,
                all_failures=True,
                batch=True,
                **case.model,
            ),
        }

    stages = run_twice(case_root, build)

    stage = stages["repair"]
    assert stage.result.calls == []
    assert [call["call"] for call in stage.batch_calls] == list(case.calls)
    (batch,) = batch_service.batches.values()
    assert batch.provider == case.provider
    assert [entry.page for entry in batch.entries] == list(FAILED_PAGES)
    _assert_repaired(stages, FAILED_PAGES, marks_left=())
    check_repair(case.golden, stages, write_guard)
