"""Characterization of response formats, providers, the custom endpoint and context.

Each case transcribes a three-page PDF or image folder with the ``llm``
method through ``run_cli``, twice in fresh roots, and compares the second
run with ``golden/llm/<case>/``. Case names follow
``<source>_<provider>_<format>[_variant]``, where the format is the response
format: ``schema`` (enforced through structured output), ``json`` (schema in
the prompt only) or ``text`` (plain-text prompt).

The project-level context folder is replaced by an empty folder, so a prompt
with a context section shows only the context a case supplies; the
``context_fallback`` case passes the public general context of the test
context folder as ``--fallback-context``. Clients are built with their
provider class's structured-output defaults (``FakeLLM.sdk_defaults``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.characterization.conftest import CONTEXT_DIR, GoldenCheck
from tests.characterization.fakes import (
    PREAMBLE,
    FakeLLM,
    PageScript,
    Reply,
    permanent_error,
    transient_error,
)
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import (
    make_born_digital_pdf,
    make_image_folder,
    make_named_images,
    make_scanned_pdf,
)
from tests.characterization.runner import (
    RunResult,
    custom_endpoint_options,
    custom_endpoint_settings,
    run_cli,
)

pytestmark = pytest.mark.integration

PAGES = 3
SCHEMAS = {
    "markdown": "markdown_transcription_schema",
    "plain_text": "plain_text_transcription_schema",
    "plain_text_markers": "plain_text_transcription_with_markers_schema",
}
"""Case variant name of each public schema."""

PROJECT_CONTEXT_DIR = CONTEXT_DIR
"""The test context folder, which holds the public general fallback context."""

GLOBAL_CONTEXT = "Global context: municipal grain registers, Lyon, 1601-1603."
SIDECAR_CONTEXT = "File context: the ledger of the city market of Lyon."
FOLDER_CONTEXT = "Folder context: scanned pages of a bakers' guild register."

Setup = Callable[[Path], dict[str, Any]]
Script = Callable[[FakeLLM], None]
Check = Callable[[RunResult], None]


# ============================================================================
# Inputs
# ============================================================================
def _pdf(root: Path) -> Path:
    return make_born_digital_pdf(root / "docs" / "doc.pdf", PAGES)


def _scanned_pdf(root: Path) -> Path:
    return make_scanned_pdf(root / "docs" / "scan.pdf", PAGES)


def _images(root: Path) -> Path:
    return make_image_folder(root / "scans" / "pages", PAGES)


SOURCES: dict[str, tuple[str, Callable[[Path], Path]]] = {
    "pdf": ("pdf", _pdf),
    "scanned": ("pdf", _scanned_pdf),
    "images": ("images", _images),
}


# ============================================================================
# Request digests
# ============================================================================
def _system_text(call: Mapping[str, Any]) -> str:
    """Return the full system prompt of a recorded call."""
    message = call["messages"][0]
    assert message["role"] == "system"
    return "".join(block.get("text", "") for block in message["blocks"])


def _human_blocks(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = call["messages"][-1]
    assert message["role"] == "human"
    return list(message["blocks"])


def _structured(result: RunResult) -> list[dict[str, Any] | None]:
    return [call["structured"] for call in result.calls]


def _output_text(result: RunResult, source: Path) -> str:
    return result.output(source).read_text(encoding="utf-8")


def _source(result: RunResult) -> Path:
    return result.input_root if result.input_root is not None else result.root


# ============================================================================
# Cases
# ============================================================================
@dataclass(frozen=True)
class Case:
    """One characterization case.

    ``options`` and ``settings`` go to ``run_cli``; ``setup`` writes extra
    files into the case root and returns more options; ``script`` scripts the
    fake LLM before each run; ``check`` asserts the behavior the golden pins.
    """

    name: str
    source: str
    options: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    setup: Setup | None = None
    script: Script | None = None
    check: Check | None = None
    exit_code: int = 0


def _schema_check(schema: str) -> Check:
    def check(result: RunResult) -> None:
        for structured in _structured(result):
            assert structured is not None
            assert structured["method"] == "json_schema"
            assert structured["schema"]["name"] == schema
        assert [model["constructor"] for model in result.models] == ["ChatOpenAI"]

    return check


def _check_anthropic(result: RunResult) -> None:
    assert [model["constructor"] for model in result.models] == ["ChatAnthropic"]
    for structured in _structured(result):
        assert structured is not None
        assert structured["method"] == "json_schema"
        assert structured["schema"]["title"] == "TranscriptionSchema"
    assert all(_human_blocks(call)[-1]["type"] == "image" for call in result.calls)


def _check_google(result: RunResult) -> None:
    assert [model["constructor"] for model in result.models] == [
        "ChatGoogleGenerativeAI"
    ]
    for structured in _structured(result):
        assert structured is not None
        assert structured["method"] == "json_schema"
    assert [call["page"] for call in result.calls] == list(range(PAGES))


def _check_openrouter(result: RunResult) -> None:
    assert [model["provider"] for model in result.models] == ["openrouter"]
    for structured in _structured(result):
        assert structured is not None
        assert structured["method"] == "json_schema"
        assert structured["schema"]["title"] == SCHEMAS["markdown"]


def _custom_schema_check(schema: str) -> Check:
    def check(result: RunResult) -> None:
        assert [model["provider"] for model in result.models] == ["custom"]
        for structured in _structured(result):
            assert structured is not None
            assert structured["method"] == "json_schema"
            assert structured["schema"]["name"] == schema

    return check


def _check_custom_json(result: RunResult) -> None:
    assert [model["provider"] for model in result.models] == ["custom"]
    assert _structured(result) == [None] * PAGES
    assert "The JSON schema:" in _system_text(result.calls[0])
    text = _output_text(result, _source(result))
    assert "Market Records, Part 01" in text
    assert "Market Records, Part 02" in text
    assert PREAMBLE not in text
    assert "```" not in text


def _check_custom_text(result: RunResult) -> None:
    assert [model["provider"] for model in result.models] == ["custom"]
    assert _structured(result) == [None] * PAGES
    assert "JSON" not in _system_text(result.calls[0])
    assert "[No transcribable text]" in _output_text(result, _source(result))


def _check_prompted_json(result: RunResult) -> None:
    assert [call["model"] for call in result.calls] == ["o3-deep-research"] * PAGES
    assert _structured(result) == [None] * PAGES
    assert "The JSON schema:" in _system_text(result.calls[0])


def _output_format_check(suffix: str) -> Check:
    def check(result: RunResult) -> None:
        output = result.output(_source(result))
        assert output.suffix == suffix
        assert output in result.created

    return check


def _context_check(expected: str, absent: tuple[str, ...] = ()) -> Check:
    def check(result: RunResult) -> None:
        for call in result.calls:
            system = _system_text(call)
            assert expected in system
            assert all(text not in system for text in absent)

    return check


def _check_fallback_context(result: RunResult) -> None:
    fallback = PROJECT_CONTEXT_DIR / "transcr_context.txt"
    first_line = fallback.read_text(encoding="utf-8").strip().splitlines()[0]
    for call in result.calls:
        assert first_line in _system_text(call)


def _context_image_check(result: RunResult) -> None:
    for call in result.calls:
        blocks = _human_blocks(call)
        assert [block["type"] for block in blocks] == [
            "text",
            "image_url",
            "text",
            "image_url",
        ]
        assert blocks[0]["text"] == "Context image:"


def _check_overrides(result: RunResult) -> None:
    (model,) = result.models
    kwargs = model["kwargs"]
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["verbosity"] == "low"
    assert kwargs["max_completion_tokens"] == 4096
    assert kwargs["service_tier"] == "priority"


def _check_scanned(result: RunResult) -> None:
    sizes = {
        (block["width"], block["height"])
        for call in result.calls
        for block in _human_blocks(call)
        if block["type"] == "image_url"
    }
    assert sizes == {(500, 750)}


def _check_transient(result: RunResult) -> None:
    attempts = [(call["page"], call["attempt"]) for call in result.calls]
    assert attempts == [(0, 1), (0, 2), (0, 3), (1, 1), (2, 1)]
    assert "[transcription error]" not in _output_text(result, _source(result))


def _check_permanent(result: RunResult) -> None:
    assert [call["page"] for call in result.calls] == [0, 1, 2]
    assert "[transcription error]" in _output_text(result, _source(result))


# -- setups ------------------------------------------------------------------
def _global_context(root: Path) -> dict[str, Any]:
    context = root / "ctx" / "global_context.txt"
    context.parent.mkdir(parents=True)
    context.write_text(GLOBAL_CONTEXT, encoding="utf-8")
    sidecar = root / "docs" / "doc_transcr_context.txt"
    sidecar.write_text(SIDECAR_CONTEXT, encoding="utf-8")
    return {"context": context}


def _sidecar_context(root: Path) -> dict[str, Any]:
    sidecar = root / "docs" / "doc_transcr_context.txt"
    sidecar.write_text(SIDECAR_CONTEXT, encoding="utf-8")
    return {}


def _fallback_context(root: Path) -> dict[str, Any]:
    fallback = root / "fallback_context.txt"
    text = (PROJECT_CONTEXT_DIR / "transcr_context.txt").read_text(encoding="utf-8")
    fallback.write_text(text, encoding="utf-8")
    return {"fallback_context": fallback}


def _folder_context(root: Path) -> dict[str, Any]:
    sidecar = root / "scans" / "pages_transcr_context.txt"
    sidecar.write_text(FOLDER_CONTEXT, encoding="utf-8")
    return {}


def _context_image_flag(root: Path) -> dict[str, Any]:
    folder = make_named_images(root / "ctx", ["title.png"], size=(300, 200))
    return {"context_image": folder / "title.png"}


def _context_image_sidecar(root: Path) -> dict[str, Any]:
    make_named_images(root / "docs", ["doc_transcr_context_image.png"], size=(300, 200))
    return {}


# -- scripts -----------------------------------------------------------------
def _loose_answer(script: PageScript) -> str:
    """Return a schema answer whose transcription is a list of lines."""
    payload = script.payload("schema")
    payload["transcription"] = script.transcription.splitlines()
    return json.dumps(payload, ensure_ascii=False)


def _script_custom_json(llm: FakeLLM) -> None:
    llm.script(0, form="fenced")
    first = llm.page(1)
    fenced = first.content("fenced", json_answer=True)
    llm.script(1, always=Reply(text=f"{PREAMBLE}\n\n{fenced}"))
    llm.script(2, always=Reply(text=_loose_answer(llm.page(2))))


def _script_custom_text(llm: FakeLLM) -> None:
    llm.script(2, form="no_text")


def _script_transient(llm: FakeLLM) -> None:
    llm.script(0, replies={1: Reply(transient_error()), 2: Reply(transient_error())})


def _script_permanent(llm: FakeLLM) -> None:
    llm.script(1, always=Reply(permanent_error()))


CASES: tuple[Case, ...] = (
    *(
        Case(
            f"pdf_openai_schema_{variant}",
            "pdf",
            options={"schema": schema},
            check=_schema_check(schema),
        )
        for variant, schema in SCHEMAS.items()
    ),
    Case(
        "pdf_anthropic_schema",
        "pdf",
        options={"provider": "anthropic", "model": "claude-sonnet-4-5"},
        check=_check_anthropic,
    ),
    Case(
        "images_google_schema",
        "images",
        options={"provider": "google", "model": "gemini-3-flash"},
        check=_check_google,
    ),
    Case(
        "pdf_openrouter_schema",
        "pdf",
        options={"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"},
        check=_check_openrouter,
    ),
    Case(
        "pdf_custom_schema",
        "pdf",
        options=custom_endpoint_options("schema"),
        settings=custom_endpoint_settings("schema"),
        check=_custom_schema_check(SCHEMAS["markdown"]),
    ),
    Case(
        "pdf_custom_schema_plain_text",
        "pdf",
        options={"schema": SCHEMAS["plain_text"], **custom_endpoint_options("schema")},
        settings=custom_endpoint_settings("schema"),
        check=_custom_schema_check(SCHEMAS["plain_text"]),
    ),
    Case(
        "pdf_custom_json",
        "pdf",
        options=custom_endpoint_options("json"),
        settings=custom_endpoint_settings("json"),
        script=_script_custom_json,
        check=_check_custom_json,
    ),
    Case(
        "pdf_custom_text",
        "pdf",
        options=custom_endpoint_options("text"),
        settings=custom_endpoint_settings("text"),
        script=_script_custom_text,
        check=_check_custom_text,
    ),
    Case(
        "pdf_openai_json_no_structured_outputs",
        "pdf",
        # OpenAI documents no structured outputs for o3-deep-research; o1-pro,
        # the earlier model of this case, has them.
        options={"provider": "openai", "model": "o3-deep-research"},
        check=_check_prompted_json,
    ),
    *(
        Case(
            f"pdf_openai_schema_formats_{suffix}",
            "pdf",
            options={"formats": suffix},
            check=_output_format_check(f".{suffix}"),
        )
        for suffix in ("txt", "md")
    ),
    Case(
        "pdf_openai_schema_context_global",
        "pdf",
        setup=_global_context,
        check=_context_check(GLOBAL_CONTEXT, absent=(SIDECAR_CONTEXT,)),
    ),
    Case(
        "pdf_openai_schema_context_sidecar",
        "pdf",
        setup=_sidecar_context,
        check=_context_check(SIDECAR_CONTEXT),
    ),
    Case(
        "images_openai_schema_context_folder",
        "images",
        setup=_folder_context,
        check=_context_check(FOLDER_CONTEXT),
    ),
    Case(
        "pdf_openai_schema_context_fallback",
        "pdf",
        setup=_fallback_context,
        check=_check_fallback_context,
    ),
    Case(
        "pdf_openai_schema_context_image",
        "pdf",
        setup=_context_image_flag,
        check=_context_image_check,
    ),
    Case(
        "pdf_openai_schema_context_image_sidecar",
        "pdf",
        setup=_context_image_sidecar,
        check=_context_image_check,
    ),
    Case(
        "pdf_openai_schema_overrides",
        "pdf",
        options={
            "reasoning_effort": "low",
            "verbosity": "low",
            "max_output_tokens": 4096,
            "service_tier": "priority",
        },
        check=_check_overrides,
    ),
    Case("scanned_openai_schema", "scanned", check=_check_scanned),
    Case(
        "pdf_openai_schema_transient_error",
        "pdf",
        script=_script_transient,
        check=_check_transient,
    ),
    Case(
        "pdf_openai_schema_permanent_error",
        "pdf",
        script=_script_permanent,
        check=_check_permanent,
        exit_code=1,
    ),
)


# ============================================================================
# Runner
# ============================================================================
@pytest.fixture(autouse=True)
def _llm_setup(
    fake_llm: FakeLLM, tmp_path: Path, project_context: Callable[[Path], None]
) -> None:
    """Use SDK structured-output defaults and an empty project context folder."""
    fake_llm.sdk_defaults = True
    folder = tmp_path / "project_context"
    folder.mkdir()
    project_context(folder)


def _run(case: Case, llm: FakeLLM, root: Path) -> RunResult:
    llm.pages.clear()
    if case.script is not None:
        case.script(llm)
    kind, make = SOURCES[case.source]
    source = make(root)
    extra = case.setup(root) if case.setup is not None else {}
    return run_cli(
        "transcribe",
        llm=llm,
        root=root,
        settings=case.settings,
        input=source,
        output=root / "out",
        type=kind,
        method="llm",
        **{**dict(case.options), **extra},
    )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_llm_case(
    case: Case,
    fake_llm: FakeLLM,
    case_root: Callable[[str], Path],
    golden: GoldenCheck,
) -> None:
    first = snapshot(_run(case, fake_llm, case_root("run1")))
    result = _run(case, fake_llm, case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)

    assert result.exit_code == case.exit_code, result.stdout + result.stderr
    assert len({call["page"] for call in result.calls}) == PAGES
    if case.check is not None:
        case.check(result)
    golden(f"llm/{case.name}", result)


def test_case_names_are_unique() -> None:
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
