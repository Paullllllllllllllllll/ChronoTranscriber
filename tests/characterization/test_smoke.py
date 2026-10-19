"""Smoke cases of the characterization harness, one per command.

Each case runs twice in fresh roots; both snapshots must match before the
run is compared with its golden files under ``golden/smoke/``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.cli.parser import build_parser
from tests.characterization.adapter import (
    RUN_TOOL_KEYWORDS,
    RunResult,
    build_argv,
    run_tool,
)
from tests.characterization.conftest import GoldenCheck
from tests.characterization.fakes import FakeLLM
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import make_born_digital_pdf, page_text

pytestmark = pytest.mark.integration

PAGES = 3
_RUN_TOOL_OWN = {"command", "llm", "root", "config_dir", "config", "env"}
_SAMPLES: dict[str, Any] = {
    "input": "doc.pdf",
    "output": "out",
    "type": "pdfs",
    "method": "gpt",
    "auto": True,
    "sync_fallback": True,
    "schema": "markdown_transcription_schema",
    "context": "context.txt",
    "fallback_context": "fallback.txt",
    "context_image": "title.png",
    "model": "gpt-5.6-terra",
    "provider": "openai",
    "reasoning_effort": "low",
    "model_verbosity": "concise",
    "max_output_tokens": 4096,
    "service_tier": "priority",
    "response_format": "json",
    "endpoint": "fake",
    "resume": True,
    "force": True,
    "files": ["a.pdf", "b.pdf"],
    "recursive": True,
    "output_mode": "mirror",
    "pages": "1-2",
    "dry_run": True,
    "retry_errors": True,
    "batch": True,
    "keep_working_files": True,
    "output_format": "md",
    "transcription": "doc.txt",
    "errors_only": True,
    "not_possible": True,
    "no_text": True,
    "all_failures": True,
    "indices": "0,2",
    "directory": "out",
    "no_diagnostics": True,
    "batch_ids": ["batch_1", "msgbatch_2"],
}
_REQUIRED: dict[str, dict[str, Any]] = {
    "transcribe": {"input": "doc.pdf"},
    "repair": {"transcription": "doc.txt"},
    "check_batches": {},
    "cancel_batches": {},
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
    return run_tool(
        "transcribe",
        llm=llm,
        root=root,
        input=_pdf(root),
        output=root / "out",
        type="pdfs",
        method=method,
    )


def test_every_run_tool_keyword_becomes_parsed_arguments() -> None:
    params = set(inspect.signature(run_tool).parameters) - _RUN_TOOL_OWN
    assert params == set().union(*RUN_TOOL_KEYWORDS.values()) == set(_SAMPLES)

    parser = build_parser()
    for command, names in RUN_TOOL_KEYWORDS.items():
        for name in sorted(names):
            keywords = {**_REQUIRED[command], name: _SAMPLES[name]}
            argv, _settings = build_argv(command, keywords)  # type: ignore[arg-type]
            parser.parse_args(argv)


def test_a_keyword_the_command_lacks_raises() -> None:
    with pytest.raises(TypeError, match="indices"):
        build_argv("transcribe", {"input": "doc.pdf", "indices": "1"})


def test_an_override_without_a_flag_or_setting_raises() -> None:
    config = {"paths_config": {"general": {"retain_temporary_jsonl": False}}}
    with pytest.raises(ValueError, match="retain_temporary_jsonl"):
        build_argv("transcribe", {"input": "doc.pdf"}, config)


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


def test_gpt_pdf_default_model(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    result = run_twice(case_root, lambda root: _transcribe_pdf(fake_llm, root, "gpt"))

    assert result.exit_code == 0
    assert [call["page"] for call in result.calls] == list(range(PAGES))
    assert [model["constructor"] for model in result.models] == ["ChatOpenAI"]
    structured = result.calls[0]["structured"]
    assert structured["method"] == "json_schema"
    assert structured["schema"]["name"] == "markdown_transcription_schema"
    output = result.output(result.root / "docs" / "doc.pdf")
    assert "Market Records, Part 01" in output.read_text(encoding="utf-8")
    golden("smoke/gpt_pdf_default_model", result)


def test_check_batches_without_batch_files(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    def build(root: Path) -> RunResult:
        folder = root / "out"
        folder.mkdir()
        return run_tool(
            "check_batches",
            llm=fake_llm,
            root=root,
            directory=folder,
            no_diagnostics=True,
        )

    result = run_twice(case_root, build)

    assert result.exit_code == 0
    assert result.summary is not None
    assert result.summary["finalized"] == 0
    assert result.created == []
    golden("smoke/check_batches_without_batch_files", result)


def test_repair_clean_gpt_transcription(
    fake_llm: FakeLLM, case_root: Callable[[str], Path], golden: GoldenCheck
) -> None:
    def build(root: Path) -> RunResult:
        transcribed = _transcribe_pdf(fake_llm, root, "gpt")
        assert transcribed.exit_code == 0
        return run_tool(
            "repair",
            llm=fake_llm,
            root=root,
            transcription=transcribed.output(root / "docs" / "doc.pdf"),
        )

    result = run_twice(case_root, build)

    assert result.exit_code == 0
    assert result.summary is not None
    assert (result.summary["repaired"], result.summary["failed"]) == (0, 0)
    assert result.calls == []
    golden("smoke/repair_clean_gpt_transcription", result)
