"""The extension's arguments belong to the commands that call providers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chronotranscriber import ext
from chronotranscriber.cli import parser as parser_module
from chronotranscriber.settings import Settings
from chronotranscriber.spec import OPTIONS, RENDER_OPTIONS
from chronotranscriber.ui.session import Session, command_words

PROVIDER_COMMANDS = (
    ("transcribe",),
    ("repair",),
    ("batch", "check"),
    ("batch", "cancel"),
    ("ui",),
)
OFFLINE_COMMANDS = (("render",), ("drop",))


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the extension register ``--probe`` and spell it in commands."""

    def register(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--probe", action="store_true")

    monkeypatch.setattr(ext, "register_args", register)
    monkeypatch.setattr(ext, "extra_argv", lambda args: ("--probe",))


def _help(words: tuple[str, ...]) -> str:
    parser = parser_module.build_parser()
    for word in words:
        actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        parser = actions[0].choices[word]
    return parser.format_help()


@pytest.mark.usefixtures("probe")
@pytest.mark.parametrize("words", PROVIDER_COMMANDS)
def test_provider_commands_take_the_extension_arguments(words: tuple[str, ...]) -> None:
    assert "--probe" in _help(words)


@pytest.mark.usefixtures("probe")
@pytest.mark.parametrize("words", OFFLINE_COMMANDS)
def test_offline_commands_take_none(words: tuple[str, ...]) -> None:
    assert "--probe" not in _help(words)


@pytest.mark.usefixtures("probe")
def test_the_wizard_appends_them_only_where_accepted() -> None:
    session = Session(Settings(), None, {}, argparse.Namespace())

    transcribe = command_words(
        ("chronotranscriber", "transcribe"), ["--input", "x"], session, OPTIONS
    )
    render = command_words(
        ("chronotranscriber", "render"),
        ["--input", str(Path("out"))],
        session,
        RENDER_OPTIONS,
    )

    assert transcribe[-1] == "--probe"
    assert "--probe" not in render
