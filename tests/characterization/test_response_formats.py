"""Characterization of ``--response-format`` json and text on every provider.

Each case transcribes a three-page PDF or image folder with the LLM method
and compares the run with ``golden/formats/<case>/``. Case names follow
``<format>_<provider>_<source>``. ``json`` sends the schema prompt without
enforcement and validates the answer: page 2 first answers truncated JSON
and is asked again, since the case allows two validation attempts. ``text``
sends the plain-text prompt and takes the answer as the transcription; page
3 answers the no-text marker. ``schema_custom_images`` has the custom
endpoint enforce a schema other than the default one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.characterization.conftest import GoldenCheck
from tests.characterization.fakes import NO_TEXT_MARKER, FakeLLM, Reply
from tests.characterization.inputs import make_born_digital_pdf, make_image_folder
from tests.characterization.runner import (
    RunResult,
    custom_endpoint_options,
    custom_endpoint_settings,
    run_cli,
)

pytestmark = pytest.mark.integration

PAGES = 3
RETRIED_PAGE = 1
NO_TEXT_PAGE = 2
MARKERS_SCHEMA = "plain_text_transcription_with_markers_schema"
VALIDATION = {"retry": {"validation_attempts": 2}}

PROVIDERS: Mapping[str, tuple[str, str]] = {
    "openai": ("gpt-5.6-luna", "pdf"),
    "anthropic": ("claude-sonnet-4-5", "images"),
    "google": ("gemini-3-flash", "images"),
    "openrouter": ("anthropic/claude-sonnet-4.5", "pdf"),
}
"""Model and source of each provider's cases."""


@dataclass(frozen=True)
class FormatCase:
    """One response-format case; ``options`` and ``settings`` go to ``run_cli``."""

    name: str
    fmt: str
    source: str
    options: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)


def _registry_case(fmt: str, provider: str) -> FormatCase:
    model, source = PROVIDERS[provider]
    return FormatCase(
        f"{fmt}_{provider}_{source}",
        fmt,
        source,
        options={"provider": provider, "model": model, "response_format": fmt},
        settings=VALIDATION if fmt == "json" else {},
    )


CASES: tuple[FormatCase, ...] = (
    *(
        _registry_case(fmt, provider)
        for fmt in ("json", "text")
        for provider in PROVIDERS
    ),
    FormatCase(
        "schema_custom_images",
        "schema",
        "images",
        options={"schema": MARKERS_SCHEMA, **custom_endpoint_options("schema")},
        settings=custom_endpoint_settings("schema"),
    ),
)


def _source(case: FormatCase, root: Path) -> tuple[str, Path]:
    if case.source == "pdf":
        return "pdf", make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)
    return "images", make_image_folder(root / "scans" / "pages", PAGES)


def _script(case: FormatCase, llm: FakeLLM) -> None:
    llm.pages.clear()
    if case.fmt == "json":
        llm.reply(RETRIED_PAGE, 1, Reply(form="malformed"))
    elif case.fmt == "text":
        llm.script(NO_TEXT_PAGE, form="no_text")


def _system_text(call: Mapping[str, Any]) -> str:
    message = call["messages"][0]
    return "".join(block.get("text", "") for block in message["blocks"])


def _attempts(result: RunResult) -> dict[int, int]:
    counts: dict[int, int] = {}
    for call in result.calls:
        counts[call["page"]] = counts.get(call["page"], 0) + 1
    return counts


def _check(case: FormatCase, result: RunResult) -> None:
    structured = [call["structured"] for call in result.calls]
    text = result.output(result.input_root or result.root).read_text(encoding="utf-8")
    for page in range(PAGES):
        if case.fmt == "text" and page == NO_TEXT_PAGE:
            assert NO_TEXT_MARKER in text
        else:
            assert f"Market Records, Part {page + 1:02d}" in text
    assert "```" not in text
    if case.fmt == "schema":
        assert all(
            entry and entry["schema"]["name"] == MARKERS_SCHEMA for entry in structured
        )
        return
    assert structured == [None] * len(result.calls)
    if case.fmt == "json":
        assert "The JSON schema:" in _system_text(result.calls[0])
        assert _attempts(result) == {0: 1, 1: 2, 2: 1}
    else:
        assert "JSON" not in _system_text(result.calls[0])
        assert _attempts(result) == dict.fromkeys(range(PAGES), 1)


@pytest.fixture(autouse=True)
def _llm_setup(
    fake_llm: FakeLLM, tmp_path: Path, project_context: Callable[[Path], None]
) -> None:
    """Use SDK structured-output defaults and an empty project context folder."""
    fake_llm.sdk_defaults = True
    folder = tmp_path / "project_context"
    folder.mkdir()
    project_context(folder)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_response_format_case(
    case: FormatCase,
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    golden: GoldenCheck,
) -> None:
    _script(case, fake_llm)
    root = case_root("run")
    kind, source = _source(case, root)

    result = run_cli(
        "transcribe",
        llm=fake_llm,
        root=root,
        settings=case.settings,
        input=source,
        output=root / "out",
        type=kind,
        method="llm",
        **dict(case.options),
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert {model["provider"] for model in result.models} == {
        "custom" if case.fmt == "schema" else case.options["provider"]
    }
    _check(case, result)
    golden(f"formats/{case.name}", result)


def test_case_names_are_unique() -> None:
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
