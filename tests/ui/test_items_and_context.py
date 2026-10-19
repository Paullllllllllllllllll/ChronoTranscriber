"""The wizard's item selection, context list, input check and key review."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import KEEP, ScriptedPrompter
from chronotranscriber.common.wizard import BACK, DONE, State, StepContext
from chronotranscriber.spec import CONTEXT_NONE, OPTIONS
from chronotranscriber.ui.review import build_review
from chronotranscriber.ui.session import Session, new_state
from chronotranscriber.ui.wizard import build_steps, run_items
from tests.characterization.inputs import make_scanned_pdf


def _ask(
    session: Session, state: State, name: str, answers: Sequence[Any]
) -> tuple[Any, ScriptedPrompter]:
    step = next(step for step in build_steps(session) if step.name == name)
    prompter = ScriptedPrompter(answers)
    with prompter:
        result = step.ask(StepContext(state, prompter))
    return result, prompter


@pytest.fixture
def pdfs(tmp_path: Path) -> Path:
    """A folder with the scanned PDFs p1, p2 and p10."""
    folder = tmp_path / "pdfs"
    for name in ("p10", "p2", "p1"):
        make_scanned_pdf(folder / f"{name}.pdf", pages=1)
    return folder


def _state(session: Session, location: Path) -> State:
    state = new_state(session, OPTIONS)
    state.set("input", location)
    return state


def test_the_items_are_listed_in_selection_order(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)
    state.set("type", "pdf")

    assert [item.name for item in run_items(state)] == ["p1.pdf", "p2.pdf", "p10.pdf"]


def test_all_items_set_all(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    result, prompter = _ask(session, state, "method", ["pdf", "llm", "all"])

    assert result is DONE
    assert "? Which items?" in prompter.transcript
    assert state.value("all") is True
    assert state.value("select") is None


def test_items_by_number_set_select(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    _result, prompter = _ask(session, state, "method", ["pdf", "llm", "range", "2-3"])

    assert state.value("select") == "2-3"
    assert state.value("all") is False
    assert "   3  p10.pdf" in prompter.transcript


def test_items_by_name_set_select(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    _result, prompter = _ask(
        session, state, "method", ["auto", "llm", KEEP, "search", "p1"]
    )

    assert state.value("select") == "p1"
    assert "2 items: p1.pdf, p10.pdf" in prompter.transcript


def test_the_checklist_numbers_auto_mode_items(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    _ask(session, state, "method", ["auto", "llm", KEEP, "list", [1, 3]])

    assert state.value("select") == "1,3"
    assert state.value("files") == ()


def test_the_checklist_names_files_of_one_type(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    _ask(session, state, "method", ["pdf", "llm", "list", ["p2.pdf"]])

    assert state.value("files") == ("p2.pdf",)
    assert state.value("select") is None


def test_one_item_asks_nothing(session: Session, tmp_path: Path) -> None:
    pdf = make_scanned_pdf(tmp_path / "one" / "book.pdf", pages=1)
    state = _state(session, pdf.parent)

    _result, prompter = _ask(session, state, "method", ["pdf", "llm"])

    assert "Which items?" not in prompter.transcript


def test_an_input_with_nothing_is_asked_again(
    session: Session, tmp_path: Path, pdfs: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    state = new_state(session, OPTIONS)

    _result, prompter = _ask(session, state, "input", [str(empty), str(pdfs), KEEP])

    assert "nothing to process; choose another input" in prompter.transcript
    assert state.value("input") == pdfs


def _context_answers(*context: Any) -> list[Any]:
    """Answers of the response step on an LLM run, with the context answers."""
    return [KEEP, KEEP, KEEP, *context, KEEP]


def test_topics_as_text_set_the_context(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    _ask(session, state, "response", _context_answers("text", "Basel ledgers"))

    assert state.value("context") == "Basel ledgers"


def test_a_text_file_sets_the_context(
    session: Session, pdfs: Path, tmp_path: Path
) -> None:
    topics = tmp_path / "topics.txt"
    topics.write_text("Ledgers", encoding="utf-8")
    state = _state(session, pdfs)

    _ask(session, state, "response", _context_answers("file", str(topics)))

    assert state.value("context") == str(topics)


def test_automatic_asks_the_fallback(session: Session, pdfs: Path) -> None:
    state = _state(session, pdfs)

    _result, prompter = _ask(
        session, state, "response", _context_answers("auto", "text", "Fallback topics")
    )

    assert "? Items without a sidecar:" in prompter.transcript
    assert state.value("context") is None
    assert state.value("fallback_context") == "Fallback topics"


def test_none_sends_no_context_and_skips_the_fallback(
    session: Session, pdfs: Path
) -> None:
    state = _state(session, pdfs)

    _result, prompter = _ask(session, state, "response", _context_answers("none"))

    assert state.value("context") == CONTEXT_NONE
    assert "Items without a sidecar" not in prompter.transcript


def test_back_from_the_fallback_returns_to_the_context(
    session: Session, pdfs: Path
) -> None:
    state = _state(session, pdfs)

    _result, _prompter = _ask(
        session, state, "response", _context_answers("auto", BACK, "none")
    )

    assert state.value("context") == CONTEXT_NONE


@pytest.mark.parametrize("dry_run", [False, True])
def test_the_review_names_a_missing_key(
    session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=1)
    state = _state(session, pdf)
    state.set("dry_run", dry_run)

    review = build_review(state, session)

    notes = "\n".join(review.notes)
    if dry_run:
        assert review.withheld is None
        assert "OPENAI_API_KEY" in notes and "Warning:" in notes
    else:
        assert review.withheld == "set OPENAI_API_KEY in the environment"
        assert "Missing API key: OPENAI_API_KEY is not set" in notes
