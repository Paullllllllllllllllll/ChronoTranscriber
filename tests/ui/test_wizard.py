"""The transcribe wizard steps: options, valid pairs, Back, skipping, markers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    Choice,
    Decision,
    State,
    StepContext,
)
from chronotranscriber.spec import METHODS_BY_TYPE, OPTIONS, SOURCE_TYPES
from chronotranscriber.ui.session import Session, build_wizard, new_state
from chronotranscriber.ui.transcribe import flow
from chronotranscriber.ui.wizard import (
    build_steps,
    file_names,
    offered_types,
    unit_word,
)
from tests.characterization.inputs import make_image_folder
from tests.characterization.inputs_ebooks import make_mobi
from tests.ui.conftest import SessionMaker


class RecordingPrompter(ScriptedPrompter):
    """A scripted prompter that keeps the values each select offered."""

    def __init__(self, answers: Sequence[Any]) -> None:
        super().__init__(answers)
        self.offered: dict[str, list[Any]] = {}

    def select(self, message: str, choices: Sequence[Choice[Any]], **kw: Any) -> Any:
        self.offered[message] = [choice.value for choice in choices]
        return super().select(message, choices, **kw)


def _step(session: Session, name: str) -> Any:
    return next(step for step in build_steps(session) if step.name == name)


def _ask(session: Session, state: State, name: str, answers: Sequence[Any]) -> Any:
    prompter = RecordingPrompter(answers)
    with prompter:
        result = _step(session, name).ask(StepContext(state, prompter))
    return result, prompter


@pytest.fixture
def every_type(mixed_folder: Path) -> Path:
    """The mixed folder with a MOBI book added: every source type."""
    make_mobi(mixed_folder / "tale.mobi")
    return mixed_folder


def test_the_steps_set_every_option_but_json_and_settings(session: Session) -> None:
    steps = build_steps(session)
    assert set().union(*(step.sets for step in steps)) == set(OPTIONS.names) - {
        "json",
        "settings",
    }


def test_the_decision_step_asks_the_model_only_when_on(session: Session) -> None:
    state = new_state(session, OPTIONS)
    result, prompter = _ask(session, state, "decisions", ["off"])
    assert result is DONE
    assert list(prompter.offered) == [
        "Decision-model pre-pass (page types before transcription):"
    ]
    result, prompter = _ask(
        session, state, "decisions", ["shadow", "openai/gpt-6-luna", None, ""]
    )
    assert result is DONE
    assert state.value("decisions") == "shadow"
    assert state.value("decision_model") == "openai/gpt-6-luna"
    assert state.value("decision_provider") is None
    assert state.value("decision_endpoint") is None


def test_the_type_question_offers_auto_and_the_types_found(
    session: Session, every_type: Path
) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", every_type)
    assert offered_types(state) == list(SOURCE_TYPES)
    state.set("recursive", True)
    assert "auto" not in offered_types(state)


@pytest.mark.parametrize("kind", SOURCE_TYPES)
def test_the_method_question_offers_the_methods_of_the_type(
    session: Session, every_type: Path, kind: str
) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", every_type)
    result, prompter = _ask(session, state, "method", [kind, BACK, BACK])
    assert result is BACK
    message = "Method:"
    if kind == "auto":
        message = "Method for scanned PDFs and image folders:"
    assert prompter.offered["Input type:"] == list(SOURCE_TYPES)
    assert prompter.offered[message] == list(METHODS_BY_TYPE[kind])


def test_image_subfolders_are_named_without_the_subfolder_search(
    session: Session, tmp_path: Path
) -> None:
    folder = tmp_path / "archive"
    make_image_folder(folder / "volume_1", pages=2)
    make_image_folder(folder / "volume_2", pages=2)
    state = new_state(session, OPTIONS)
    state.set("input", folder)
    state.set("type", "images")
    assert file_names(state) == (["volume_1", "volume_2"], True)
    _ask(session, state, "method", ["images", "tesseract", "list", ["volume_2"]])
    assert state.value("files") == ("volume_2",)
    _ask(session, state, "method", ["images", "tesseract", "all"])
    assert state.value("files") == ("volume_1", "volume_2")


def _llm_pdf_answers(pdf: Path, end: Sequence[Any]) -> list[Any]:
    """Answers of a sync LLM run on one PDF, ending with the review answers."""
    return [
        KEEP,  # task
        str(pdf),
        "pdf",
        "llm",
        "beside",
        ["md"],
        KEEP,  # provider
        KEEP,  # model
        KEEP,  # reasoning
        KEEP,  # model parameters
        KEEP,  # response format
        KEEP,  # schema
        KEEP,  # prompt
        KEEP,  # context
        KEEP,  # items without a sidecar
        KEEP,  # context image
        KEEP,  # page images
        KEEP,  # decision pre-pass
        KEEP,  # pages
        KEEP,  # submission
        [],  # run options
        KEEP,  # concurrency
        *end,
    ]


def test_parallel_requests_are_asked_only_where_calls_are_synchronous(
    session: Session, scanned_pdf: Path
) -> None:
    state = new_state(session, OPTIONS)
    for name, value in (("input", scanned_pdf), ("type", "pdf"), ("method", "llm")):
        state.set(name, value)
    _, batch = _ask(session, state, "run", [True, False, []])
    assert "? Submission:" in batch.transcript
    assert "Parallel requests:" not in batch.transcript
    _, fallback = _ask(session, state, "run", [True, True, [], KEEP])
    assert "Parallel requests:" in fallback.transcript


def test_back_returns_to_the_previous_step_and_keeps_its_answers(
    session: Session, scanned_pdf: Path
) -> None:
    answers = _llm_pdf_answers(scanned_pdf, ["quit"])
    where = answers.index("beside")
    prompter = ScriptedPrompter(
        [
            *answers[:where],
            BACK,  # on "Where should the outputs go?"
            KEEP,  # type: pdf stays preselected
            "tesseract",
            "beside",
            ["md"],
            KEEP,  # ocr config
            KEEP,  # ocr preprocess
            KEEP,  # pages
            [],  # run options
            KEEP,  # concurrency
            "quit",
        ]
    )
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.action is Decision.QUIT
    assert outcome.answers["method"] == "tesseract"
    assert outcome.answers["type"] == "pdf"
    assert prompter.transcript.count("? Input type:") == 2
    assert "? Provider:" not in prompter.transcript


def test_steps_of_other_methods_are_skipped(
    session: Session, image_folder: Path, mixed_folder: Path
) -> None:
    wizard = build_wizard(flow(), session, ScriptedPrompter([]))
    state = wizard.state

    def applicable() -> list[str]:
        return [step.name for step in wizard.applicable()]

    state.set("input", image_folder)
    state.set("type", "images")
    state.set("method", "tesseract")
    assert applicable() == ["task", "input", "method", "output", "ocr", "pages", "run"]
    state.set("method", "llm")
    assert applicable() == [
        "task",
        "input",
        "method",
        "output",
        "model",
        "response",
        "images",
        "decisions",
        "pages",
        "run",
    ]
    state.set("input", mixed_folder / "interview.wav")
    state.set("type", "audio")
    state.set("method", "whisper")
    assert applicable() == [
        "task",
        "input",
        "method",
        "output",
        "audio",
        "pages",
        "run",
    ]
    assert unit_word(state) == "audio chunks"
    state.set("method", "audio-api")
    assert "model" in applicable()
    assert "audio" in applicable()


def test_an_inapplicable_step_drops_its_answers(
    session: Session, scanned_pdf: Path
) -> None:
    answers = _llm_pdf_answers(scanned_pdf, [])
    prompter = ScriptedPrompter(
        [
            *answers[:11],  # up to the schema
            "plain_text_transcription_schema",
            *answers[12:],
            "edit",
            "method",
            KEEP,  # type
            "native",
            "quit",
        ]
    )
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.answers["method"] == "native"
    assert "schema" not in outcome.answers


def test_a_step_applying_again_after_an_edit_gets_its_answers_back(
    session: Session, scanned_pdf: Path
) -> None:
    answers = _llm_pdf_answers(scanned_pdf, [])
    prompter = ScriptedPrompter(
        [
            *answers[:5],
            ["txt", "md"],  # formats
            *answers[6:11],
            "plain_text_transcription_schema",
            *answers[12:],
            "edit",
            "method",
            KEEP,  # type
            "native",
            "edit",
            "method",
            KEEP,  # type
            "llm",
            "quit",
        ]
    )
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.answers["schema"] == "plain_text_transcription_schema"
    assert outcome.answers["formats"] == ("txt", "md")
    assert "method" not in outcome.answers
    assert prompter.transcript.count("Edit which step?") == 2


def test_settings_defaults_are_preselected_and_marked(
    make_session: SessionMaker, scanned_pdf: Path
) -> None:
    session = make_session({"formats": ["md"], "reasoning_effort": "medium"})
    answers = _llm_pdf_answers(scanned_pdf, ["quit"])
    answers[answers.index(["md"])] = KEEP
    prompter = ScriptedPrompter(answers)
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.action is Decision.QUIT
    lines = prompter.lines
    assert "   [x] Markdown (.md) (s)" in lines
    assert " > medium (s)" in lines
    assert "formats" not in outcome.answers
    assert "(s) = from settings defaults" in prompter.transcript
    assert "reasoning medium (s)" in prompter.transcript


def test_the_pages_step_checks_the_range(session: Session, scanned_pdf: Path) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", scanned_pdf)
    state.set("type", "pdf")
    _ask(session, state, "pages", ["range", "2-3"])
    assert state.value("pages") == "2-3"
    with pytest.raises(ScriptError, match="rejected"):
        _ask(session, state, "pages", ["range", "3-1"])
    _ask(session, state, "pages", ["all"])
    assert state.value("pages") is None
