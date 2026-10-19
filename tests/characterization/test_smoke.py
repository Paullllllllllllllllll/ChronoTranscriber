"""Smoke cases of the characterization harness, one per command.

Each case runs twice in fresh roots; both snapshots must match before the
run is compared with its golden files under ``golden/smoke/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.cli.parser import build_parser
from tests.characterization.conftest import GoldenCheck
from tests.characterization.fakes import FakeLLM
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import make_born_digital_pdf, page_text
from tests.characterization.runner import (
    Command,
    RunResult,
    build_argv,
    option_table,
    run_cli,
)

pytestmark = pytest.mark.integration

PAGES = 3
_SAMPLES: dict[Command, dict[str, Any]] = {
    "transcribe": {
        "input": "doc.pdf",
        "output": "out",
        "type": "pdf",
        "method": "llm",
        "searchable_pdfs": "text",
        "sync_fallback": True,
        "schema": "markdown_transcription_schema",
        "context": "context.txt",
        "fallback_context": "fallback.txt",
        "context_image": "title.png",
        "model": "gpt-5.6-terra",
        "provider": "openai",
        "reasoning_effort": "low",
        "verbosity": "low",
        "max_output_tokens": 4096,
        "service_tier": "priority",
        "response_format": "json",
        "endpoint": "fake",
        "force": True,
        "files": ["a.pdf", "b.pdf"],
        "recursive": True,
        "layout": "mirror",
        "pages": "1-2",
        "dry_run": True,
        "retry_errors": True,
        "batch": True,
        "keep_working_files": True,
        "formats": "md",
        "concurrency": 4,
        "ocr_config": "--psm 6",
        "language": "de",
        "audio_chunk_seconds": 30,
    },
    "repair": {
        "input": "doc.txt",
        "errors_only": True,
        "not_possible": True,
        "no_text": True,
        "all_failures": True,
        "indices": "0,2",
        "batch": True,
        "model": "gpt-5.6-terra",
        "provider": "openai",
    },
    "batch check": {"input": "out", "formats": "md"},
    "batch cancel": {"batch_ids": ["batch_1", "msgbatch_2"]},
}
"""An example value of each option the characterization cases pass."""

_REQUIRED: dict[Command, dict[str, Any]] = {
    "transcribe": {"input": "doc.pdf"},
    "repair": {"input": "doc.txt"},
    "batch check": {},
    "batch cancel": {},
}


def run_twice(
    case_root: Callable[[str], Path], build: Callable[[Path], RunResult]
) -> RunResult:
    """Run *build* in two fresh roots and return the second result.

    Fails when the two snapshots differ, so a golden never records output
    that changes between identical runs.
    """
    first = snapshot(build(case_root("run1")))
    result = build(case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)
    return result


def _pdf(root: Path) -> Path:
    return make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)


def _transcribe_pdf(llm: FakeLLM, root: Path, method: str) -> RunResult:
    return run_cli(
        "transcribe",
        llm=llm,
        root=root,
        input=_pdf(root),
        output=root / "out",
        type="pdf",
        method=method,
    )


def test_every_sample_option_becomes_parsed_arguments() -> None:
    parser = build_parser()
    for command, samples in _SAMPLES.items():
        assert set(samples) <= set(option_table(command).names)
        for name, value in samples.items():
            argv = build_argv(command, {**_REQUIRED[command], name: value})
            parser.parse_args(argv)


def test_an_option_the_command_lacks_raises() -> None:
    with pytest.raises(TypeError, match="indices"):
        build_argv("transcribe", {"input": "doc.pdf", "indices": "1"})


def test_an_option_the_runner_passes_itself_raises() -> None:
    with pytest.raises(TypeError, match="json"):
        build_argv("transcribe", {"input": "doc.pdf", "json": True})


def test_native_pdf(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    result = run_twice(
        case_root, lambda root: _transcribe_pdf(fake_llm, root, "native")
    )

    assert result.exit_code == 0
    assert result.calls == []
    output = result.output(result.root / "docs" / "doc.pdf")
    assert output in result.created
    text = output.read_text(encoding="utf-8")
    assert all(page_text(index) in text for index in range(PAGES))
    golden("smoke/native_pdf", result)


def test_llm_pdf_default_model(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    result = run_twice(case_root, lambda root: _transcribe_pdf(fake_llm, root, "llm"))

    assert result.exit_code == 0
    assert [call["page"] for call in result.calls] == list(range(PAGES))
    assert [model["constructor"] for model in result.models] == ["ChatOpenAI"]
    structured = result.calls[0]["structured"]
    assert structured["method"] == "json_schema"
    assert structured["schema"]["name"] == "markdown_transcription_schema"
    output = result.output(result.root / "docs" / "doc.pdf")
    assert "Market Records, Part 01" in output.read_text(encoding="utf-8")
    golden("smoke/llm_pdf_default_model", result)


def test_batch_check_without_submissions(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    def build(root: Path) -> RunResult:
        folder = root / "out"
        folder.mkdir()
        return run_cli("batch check", llm=fake_llm, root=root, input=folder)

    result = run_twice(case_root, build)

    assert result.exit_code == 0
    assert result.summary is not None
    assert result.summary["finalized"] == 0
    assert result.created == []
    golden("smoke/batch_check_without_submissions", result)


def test_repair_clean_llm_transcription(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    def build(root: Path) -> RunResult:
        transcribed = _transcribe_pdf(fake_llm, root, "llm")
        assert transcribed.exit_code == 0
        return run_cli(
            "repair",
            llm=fake_llm,
            root=root,
            input=transcribed.output(root / "docs" / "doc.pdf"),
        )

    result = run_twice(case_root, build)

    assert result.exit_code == 0
    assert result.summary is not None
    assert (result.summary["repaired"], result.summary["failed"]) == (0, 0)
    assert result.calls == []
    golden("smoke/repair_clean_llm_transcription", result)
