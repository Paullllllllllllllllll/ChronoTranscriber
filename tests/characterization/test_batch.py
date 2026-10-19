"""Characterization of batch transcription through fake batch services.

Each case submits with ``transcribe --batch --method llm`` and finalizes with
``batch check --input``; the real OpenAI, Anthropic and Google backends
run against the fake SDK clients of ``fakes_batch``. Every stage is
snapshotted: the files the run created, the files it changed (under
``changed/``), the JSON summary and ``requests.json``, whose ``batch`` entry
holds the client constructions, the API calls and the submitted request
bodies. Each case runs twice in fresh roots; both snapshots must match
before they are compared with the golden files under ``golden/batch/``.
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
from tests.characterization.inputs import make_image_folder, make_scanned_pdf
from tests.characterization.runner import Command, RunResult, run_cli

pytestmark = pytest.mark.integration

PAGES = 2
PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "google")
MODELS: dict[str, str | None] = {
    "openai": None,
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-3-flash-preview",
}
"""Model per provider; None keeps the configured default."""


@dataclasses.dataclass
class Stage:
    """One command of a case: its result and its normalized snapshot."""

    result: RunResult
    files: dict[str, str]
    calls: list[dict[str, Any]]
    changed: list[Path]


@pytest.fixture
def batch_service(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> FakeBatchService:
    """Install the fake batch service over ``fake_llm``'s page scripts."""
    return FakeBatchService(fake_llm).install(monkeypatch)


# ============================================================================
# Stages and snapshots
# ============================================================================
def _contents(dirs: tuple[Path, ...]) -> dict[Path, bytes]:
    found: dict[Path, bytes] = {}
    for directory in dirs:
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    found[path] = path.read_bytes()
    return found


def _stage_files(
    result: RunResult,
    service: FakeBatchService,
    changed: list[Path],
    *,
    with_requests: bool,
) -> dict[str, str]:
    ids = service.id_replacements()
    files = snapshot(result, replacements=ids)
    if changed:
        edited = snapshot(
            dataclasses.replace(result, created=changed), replacements=ids
        )
        files.update(
            {
                "changed/" + name.removeprefix("files/"): content
                for name, content in edited.items()
                if name.startswith("files/")
            }
        )
    records = service.request_records()
    if not with_requests:
        records.pop("requests")
        records.pop("prompts")
    requests = json.loads(files["requests.json"])
    requests["batch"] = normalizer_for(result, ids).data(records)
    files["requests.json"] = (
        json.dumps(requests, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return files


def run_stage(
    service: FakeBatchService,
    command: Command,
    *,
    with_requests: bool = False,
    **kwargs: Any,
) -> Stage:
    """Run one command with fresh service records and snapshot it."""
    root: Path = kwargs["root"]
    folder = kwargs.get("input" if command == "batch check" else "output")
    watched = (folder,) if isinstance(folder, Path) else (root,)
    before = _contents(watched)
    service.reset_records()
    result = run_cli(command, **kwargs)
    after = _contents(watched)
    changed = sorted(
        path for path, data in after.items() if path in before and before[path] != data
    )
    calls = [dict(call) for call in service.calls]
    files = _stage_files(result, service, changed, with_requests=with_requests)
    return Stage(result, files, calls, changed)


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
    """Compare every stage with ``golden/batch/<case>/<stage>/``."""
    for name, stage in stages.items():
        check_files(f"batch/{case}/{name}", stage.files, on_write=guard.consume)


# ============================================================================
# Commands
# ============================================================================
def _pdf(root: Path) -> Path:
    return make_scanned_pdf(root / "docs" / "doc.pdf", PAGES)


def submit(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    provider: Provider,
    *,
    source: Path | None = None,
    type: str = "pdf",
    sync_fallback: bool = False,
) -> Stage:
    """Run ``transcribe --batch --method llm`` for *provider* on *source*."""
    return run_stage(
        service,
        "transcribe",
        with_requests=True,
        llm=llm,
        root=root,
        input=source if source is not None else _pdf(root),
        output=root / "out",
        type=type,
        method="llm",
        batch=True,
        provider=provider,
        model=MODELS[provider],
        sync_fallback=sync_fallback,
    )


def check(llm: FakeLLM, service: FakeBatchService, root: Path) -> Stage:
    """Run ``batch check`` on the output folder."""
    return run_stage(service, "batch check", llm=llm, root=root, input=root / "out")


def round_trip(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    provider: Provider,
    **kwargs: Any,
) -> dict[str, Stage]:
    """Submit, finalize and check again; return the three stages."""
    service.reset()
    stages = {"submit": submit(llm, service, root, provider, **kwargs)}
    stages["finalize"] = check(llm, service, root)
    stages["recheck"] = check(llm, service, root)
    return stages


def _output_text(stage: Stage) -> str:
    outputs = [path for path in stage.result.created if path.suffix == ".txt"]
    assert len(outputs) == 1, stage.result.created
    return outputs[0].read_text(encoding="utf-8")


def _assert_round_trip(stages: Mapping[str, Stage], *, failed: bool = False) -> None:
    """*failed*: a page fails in the batch, so the finalizing check exits 1."""
    submitted, finalized, again = (
        stages["submit"],
        stages["finalize"],
        stages["recheck"],
    )
    assert submitted.result.exit_code == 0, submitted.result.stdout
    assert submitted.result.calls == []
    assert not [p for p in submitted.result.created if p.suffix == ".txt"]
    assert finalized.result.exit_code == int(failed), finalized.result.stdout
    assert finalized.result.summary is not None
    assert finalized.result.summary["finalized"] == 1, finalized.result.stdout
    assert again.result.exit_code == 0
    assert again.result.summary is not None
    assert again.result.summary["finalized"] == 0
    assert again.result.created == []
    assert again.changed == []
    assert again.calls == []


# ============================================================================
# Cases
# ============================================================================
@pytest.mark.parametrize("provider", PROVIDERS)
def test_pdf_round_trip(
    provider: Provider,
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    stages = run_twice(
        case_root,
        lambda root: round_trip(fake_llm, batch_service, root, provider),
    )

    _assert_round_trip(stages)
    text = _output_text(stages["finalize"])
    assert "Market Records, Part 01" in text
    assert "Market Records, Part 02" in text
    assert text.index("Part 01") < text.index("Part 02")
    check_case(f"{provider}_pdf", stages, write_guard)


def test_image_folder_round_trip(
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        folder = make_image_folder(root / "scans" / "ledger", PAGES)
        return round_trip(
            fake_llm, batch_service, root, "openai", source=folder, type="images"
        )

    stages = run_twice(case_root, build)

    _assert_round_trip(stages)
    assert "Market Records, Part 02" in _output_text(stages["finalize"])
    check_case("openai_image_folder", stages, write_guard)


@pytest.mark.parametrize(
    ("provider", "error_form"),
    [
        ("openai", "error_file"),
        ("openai", "output"),
        ("anthropic", "errored"),
        ("google", "inline_error"),
    ],
)
def test_failed_request(
    provider: Provider,
    error_form: str,
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    fake_llm.script(1, always=Reply(exception=permanent_error()))
    if error_form == "output":
        batch_service.openai_error_form = "output"

    stages = run_twice(
        case_root,
        lambda root: round_trip(fake_llm, batch_service, root, provider),
    )

    _assert_round_trip(stages, failed=True)
    text = _output_text(stages["finalize"])
    assert "Market Records, Part 01" in text
    assert "[transcription error" in text
    assert "Market Records, Part 02" not in text
    check_case(f"{provider}_failed_request_{error_form}", stages, write_guard)


def test_missing_result(
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    batch_service.drop_pages = {1}

    stages = run_twice(
        case_root,
        lambda root: round_trip(fake_llm, batch_service, root, "openai"),
    )

    _assert_round_trip(stages, failed=True)
    assert "no result returned" in _output_text(stages["finalize"])
    check_case("openai_missing_result", stages, write_guard)


@pytest.mark.parametrize("sync_fallback", [False, True], ids=["strict", "fallback"])
def test_submission_failure(
    sync_fallback: bool,
    fake_llm: FakeLLM,
    batch_service: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    batch_service.fail_on["openai.batches.create"] = permanent_error()

    def build(root: Path) -> dict[str, Stage]:
        batch_service.reset()
        return {
            "submit": submit(
                fake_llm, batch_service, root, "openai", sync_fallback=sync_fallback
            )
        }

    stages = run_twice(case_root, build)

    result = stages["submit"].result
    assert [call["call"] for call in stages["submit"].calls] == [
        "openai.files.create",
        "openai.batches.create",
        "openai.files.delete",
    ]
    if sync_fallback:
        assert result.exit_code == 0, result.stdout
        assert len(result.calls) == PAGES
        assert "falling back to synchronous processing" in result.stderr
        text = _output_text(stages["submit"])
        assert "[transcription error" not in text
        assert text.index("Part 01") < text.index("Part 02")
    else:
        assert result.exit_code == 1
        assert result.calls == []
        assert not [p for p in result.created if p.suffix == ".txt"]
    case = "fallback" if sync_fallback else "strict"
    check_case(f"openai_submission_failure_{case}", stages, write_guard)
