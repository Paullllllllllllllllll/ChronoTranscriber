"""Every type and method the wizard offers for an input plans without error."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import ScriptedPrompter
from chronotranscriber.common.wizard import BACK, DONE, Choice, StepContext
from chronotranscriber.run import plan
from chronotranscriber.spec import OPTIONS
from chronotranscriber.ui.questions import resolution_of
from chronotranscriber.ui.session import Session, new_state
from chronotranscriber.ui.wizard import build_steps
from tests.characterization.inputs import make_image_folder, make_scanned_pdf
from tests.characterization.inputs_audio import make_wav
from tests.ui.conftest import DUMMY_KEYS

TYPE_MESSAGE = "Input type:"
METHOD_MESSAGES = ("Method:", "Method for scanned PDFs and image folders:")


class OfferPrompter(ScriptedPrompter):
    """A scripted prompter that records offered values and takes every file."""

    def __init__(self, answers: Sequence[Any]) -> None:
        super().__init__(answers)
        self.offered: dict[str, list[Any]] = {}

    def select(self, message: str, choices: Sequence[Choice[Any]], **kw: Any) -> Any:
        self.offered[message] = [choice.value for choice in choices]
        if message.startswith("Which "):
            return "all"
        return super().select(message, choices, **kw)


def _lone_wav(root: Path) -> Path:
    return make_wav(root / "talk.wav", seconds=1)


def _audio_folder(root: Path) -> Path:
    folder = root / "recordings"
    make_wav(folder / "one.wav", seconds=1)
    make_wav(folder / "two.wav", seconds=1)
    return folder


def _pdf_and_loose_image(root: Path) -> Path:
    folder = make_image_folder(root / "loose", pages=1)
    make_scanned_pdf(folder / "book.pdf", pages=1)
    return folder


INPUTS: dict[str, Callable[[Path, dict[str, Path]], tuple[Path, bool]]] = {
    "mixed": lambda root, fx: (fx["mixed"], False),
    "lone-wav": lambda root, fx: (_lone_wav(root), False),
    "audio-folder": lambda root, fx: (_audio_folder(root), False),
    "image-folder": lambda root, fx: (fx["images"], False),
    "single-pdf": lambda root, fx: (fx["pdf"], False),
    "pdf-and-loose-image": lambda root, fx: (_pdf_and_loose_image(root), False),
    "mixed-recursive": lambda root, fx: (fx["mixed"], True),
}
AUTO_OFFERED = {
    "mixed": True,
    "lone-wav": False,
    "audio-folder": False,
    "image-folder": False,
    "single-pdf": True,
    "pdf-and-loose-image": True,
    "mixed-recursive": False,
}


def _ask_method(
    session: Session, location: Path, recursive: bool, answers: Sequence[Any]
) -> tuple[Any, OfferPrompter, Any]:
    state = new_state(session, OPTIONS)
    state.set("input", location)
    if recursive:
        state.set("recursive", True)
    step = next(step for step in build_steps(session) if step.name == "method")
    prompter = OfferPrompter(answers)
    with prompter:
        result = step.ask(StepContext(state, prompter))
    return result, prompter, state


def _offered_methods(prompter: OfferPrompter) -> list[str]:
    return next(
        prompter.offered[message]
        for message in METHOD_MESSAGES
        if message in prompter.offered
    )


@pytest.fixture
def located(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    mixed_folder: Path,
    image_folder: Path,
    scanned_pdf: Path,
) -> tuple[str, Path, bool]:
    name: str = request.param
    fixtures = {"mixed": mixed_folder, "images": image_folder, "pdf": scanned_pdf}
    location, recursive = INPUTS[name](tmp_path / "extra", fixtures)
    return name, location, recursive


@pytest.mark.parametrize("located", list(INPUTS), indirect=True)
def test_every_offered_type_and_method_plans(
    located: tuple[str, Path, bool],
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    label, location, recursive = located
    result, prompter, _state = _ask_method(session, location, recursive, [BACK])
    assert result is BACK
    types = prompter.offered[TYPE_MESSAGE]
    assert types
    if label in AUTO_OFFERED:
        assert ("auto" in types) is AUTO_OFFERED[label]
    for kind in types:
        _result, prompter, _state = _ask_method(
            session, location, recursive, [kind, BACK, BACK]
        )
        searchable = ("ocr", "text") if kind == "auto" else (None,)
        for method in _offered_methods(prompter):
            for mode in searchable:
                answers = [kind, method] + ([mode] if mode else [])
                result, _prompter, state = _ask_method(
                    session, location, recursive, answers
                )
                assert result is DONE, (kind, method, mode)
                spec = resolution_of(state, session).spec
                plan(spec, session.settings)
