"""The transcribe review: lines, markers, the plan, Force and the command."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import ext
from chronotranscriber import run as core
from chronotranscriber.common.wizard import Part, Review, State, render_review
from chronotranscriber.settings import resolve
from chronotranscriber.spec import OPTIONS, parse
from chronotranscriber.ui.review import (
    FORCE_ACTION,
    PROGRAM,
    build_review,
    equivalent_command,
)
from chronotranscriber.ui.session import Session, absolute
from tests.characterization.inputs import make_born_digital_pdf
from tests.ui.conftest import DUMMY_KEYS, SessionMaker


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the dummy provider keys the plan checks."""
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)


def _state(session: Session, **answers: Any) -> State:
    return State(OPTIONS, session.settings.defaults, answers)


def _lines(review: Review) -> dict[str, list[str]]:
    return {
        line.label: [
            part.text if isinstance(part, Part) else part for part in line.parts
        ]
        for line in review.lines
    }


def _marked(review: Review) -> set[str]:
    return {
        part.text
        for line in review.lines
        for part in line.parts
        if isinstance(part, Part) and part.from_settings
    }


@pytest.mark.usefixtures("keys")
def test_the_lines_of_an_llm_run(
    make_session: SessionMaker, scanned_pdf: Path, tmp_path: Path
) -> None:
    session = make_session({"reasoning_effort": "medium", "formats": ["md"]})
    out = tmp_path / "out"
    state = _state(
        session,
        input=scanned_pdf,
        type="pdf",
        method="llm",
        output=out,
        pages="2-3",
        concurrency=4,
    )
    review = build_review(state, session)
    lines = _lines(review)
    assert list(lines) == [
        "input",
        "method",
        "output",
        "model",
        "access",
        "response",
        "images",
        "pages",
        "run",
    ]
    assert lines["access"] == ["API key OPENAI_API_KEY"]
    assert lines["input"] == [f"{scanned_pdf}  (1 PDF, 3 pages)"]
    assert lines["method"] == ["type pdf", "llm"]
    assert lines["output"] == [
        str(out),
        "layout hash",
        "formats md (+ chronotranscriber.sqlite)",
    ]
    assert lines["model"][:2] == ["openai / gpt-6-luna", "reasoning medium"]
    assert lines["pages"] == ["pages 2-3"]
    assert lines["run"] == ["sync", "resume", "retry errors", "concurrency 4"]
    assert _marked(review) == {
        "reasoning medium",
        "formats md (+ chronotranscriber.sqlite)",
    }
    assert review.withheld is None
    assert review.notes[:2] == [
        "Plan: 1 to process, 0 complete",
        "  book.pdf: new (llm)",
    ]
    screen = render_review(review)
    assert screen.splitlines()[0].endswith("(s) = from settings defaults")
    assert "reasoning medium (s)" in screen


@pytest.mark.usefixtures("keys")
def test_the_lines_of_local_engines(
    session: Session, image_folder: Path, mixed_folder: Path
) -> None:
    tesseract = build_review(
        _state(session, input=image_folder, type="images", method="tesseract"),
        session,
    )
    lines = _lines(tesseract)
    assert lines["engine"] == ["Tesseract (local OCR)"]
    assert lines["ocr"] == ["config --oem 3 --psm 6", "preprocess default"]
    assert lines["pages"] == ["all images"]
    whisper = build_review(
        _state(
            session,
            input=mixed_folder / "interview.wav",
            type="audio",
            method="whisper",
            language="de",
        ),
        session,
    )
    lines = _lines(whisper)
    assert lines["engine"] == ["Whisper large-v3"]
    assert lines["audio"] == ["language de", "chunks of 600 s"]
    assert lines["pages"] == ["all audio chunks"]
    assert "response" not in lines
    assert "access" not in lines


@pytest.mark.usefixtures("keys")
def test_the_method_line_names_how_searchable_pdfs_are_read(
    session: Session, mixed_folder: Path
) -> None:
    answers = {"input": mixed_folder, "type": "auto", "method": "llm"}
    lines = _lines(build_review(_state(session, **answers), session))
    assert lines["method"] == [
        "type auto",
        "llm for scans and image folders",
        "llm for searchable PDFs",
    ]
    answers["searchable_pdfs"] = "text"
    lines = _lines(build_review(_state(session, **answers), session))
    assert lines["method"][2] == "text layer for searchable PDFs"


@pytest.mark.usefixtures("keys")
def test_the_access_line_names_batch_and_fallback_calls(
    session: Session, scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = {"input": scanned_pdf, "type": "pdf", "method": "llm", "batch": True}
    review = build_review(_state(session, **answers), session)
    assert _lines(review)["access"] == ["API key OPENAI_API_KEY"]

    def describe(provider: str, endpoint: str | None = None, **kw: Any) -> str:
        return f"{kw['kind']} route"

    monkeypatch.setattr(ext, "describe_access", describe)
    answers["sync_fallback"] = True
    review = build_review(_state(session, **answers), session)
    assert _lines(review)["access"] == [
        "batch: batch route",
        "sync fallback: chat route",
    ]


def test_run_is_withheld_while_a_key_is_missing(
    session: Session, scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    review = build_review(_state(session, input=scanned_pdf), session)
    assert review.withheld == "set OPENAI_API_KEY in the environment"
    assert any(
        note.startswith("Missing API key: OPENAI_API_KEY is not set (provider openai")
        for note in review.notes
    )
    assert any(note.startswith("Plan: 1 to process") for note in review.notes)
    assert "Run unavailable" in render_review(review)
    dry = build_review(_state(session, input=scanned_pdf, dry_run=True), session)
    assert dry.withheld is None


def test_unresolvable_answers_withhold_run(session: Session, scanned_pdf: Path) -> None:
    review = build_review(
        _state(session, input=scanned_pdf, type="epub", method="tesseract"), session
    )
    assert review.lines == ()
    assert review.withheld is not None
    assert "--type epub" in review.withheld


def test_a_force_action_is_offered_for_complete_items(
    session: Session, tmp_path: Path
) -> None:
    pdf = make_born_digital_pdf(tmp_path / "digital" / "letters.pdf", pages=2)
    answers = {"input": pdf, "type": "pdf", "method": "native"}
    resolved = resolve(session.settings, answers)
    first = core.plan(resolved.spec, session.settings, sources=resolved.sources)
    assert asyncio.run(core.run(first)).failed == 0

    state = _state(session, **answers)
    review = build_review(state, session)
    assert review.notes[:2] == [
        "Plan: 0 to process, 1 complete",
        "  letters.pdf: complete (skipped)",
    ]
    [action] = review.actions
    assert action.key == FORCE_ACTION
    assert action.title == "Transcribe 1 complete item again (--force)"
    action.apply(state)
    assert state.value("force") is True
    forced = build_review(state, session)
    assert forced.actions == []
    assert _lines(forced)["run"][1] == "force"


@pytest.mark.parametrize(
    "defaults",
    [
        {},
        {"model": "claude-sonnet-5", "formats": ["md", "txt"], "concurrency": 3},
        {"top_p": 0.5, "layout": "mirror", "recursive": True},
    ],
)
def test_the_equivalent_command_parses_back_to_the_spec(
    make_session: SessionMaker,
    mixed_folder: Path,
    tmp_path: Path,
    defaults: Mapping[str, Any],
) -> None:
    session = make_session(defaults)
    state = _state(
        session,
        input=mixed_folder,
        type="pdf",
        method="llm",
        output=tmp_path / "out",
        model="gpt-5.6-terra",
        pages="first:2",
        force=True,
    )
    words = equivalent_command(state, session)
    assert tuple(words[:2]) == PROGRAM
    assert words[-2:] == ["--settings", str(absolute(session.settings_path or ""))]
    spec = resolve(session.settings, state.answers).spec
    assert parse(words[2:], session.settings.defaults) == spec
    assert parse(words[2:-2]) == spec


def test_the_equivalent_command_names_the_service_tier_only_for_openai(
    make_session: SessionMaker, mixed_folder: Path, tmp_path: Path
) -> None:
    session = make_session({"service_tier": "priority"})
    answers = {
        "input": mixed_folder,
        "type": "pdf",
        "method": "llm",
        "output": tmp_path / "out",
    }
    openai = equivalent_command(_state(session, **answers), session)
    assert openai[openai.index("--service-tier") + 1] == "priority"
    answers["model"] = "claude-sonnet-5"
    anthropic = equivalent_command(_state(session, **answers), session)
    assert not any(word.startswith("--service-tier") for word in anthropic)
    spec = resolve(session.settings, answers).spec
    assert parse(anthropic[2:], session.settings.defaults) == spec
