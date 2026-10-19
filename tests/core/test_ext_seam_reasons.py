"""Extension seams of the front ends: refusal reasons and copied commands.

``ext.batch_allowed`` may raise a ``SettingsError`` whose message is the
reason a batch is refused; ``ext.extra_argv`` returns the words of the
extension's options so a command copied from the ui reproduces them. Every
test replaces the hooks with a scratch plugin; no call leaves the process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber import ext
from chronotranscriber.batch.repair import RepairPlanError, plan_repair
from chronotranscriber.cli.parser import build_parser
from chronotranscriber.common.wizard import State
from chronotranscriber.settings import Settings, SettingsError, resolve
from chronotranscriber.spec import OPTIONS, REPAIR_OPTIONS
from chronotranscriber.ui import repair as repair_ui
from chronotranscriber.ui import wizard
from chronotranscriber.ui.review import PROGRAM, equivalent_command
from chronotranscriber.ui.session import Session, new_state
from tests.batch.test_repair_store import spec_for, store_document
from tests.characterization.inputs import make_scanned_pdf
from tests.conftest import make_settings, make_spec

FLAG = "--scratch-access"
NAME = "alpha"
REASON = "scratch refusal reason"


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a plugin that registers ``--scratch-access`` and copies it."""

    def register_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(FLAG, dest="scratch_access", metavar="NAME")

    def extra_argv(args: argparse.Namespace | None) -> tuple[str, ...]:
        value = getattr(args, "scratch_access", None)
        return (FLAG, value) if value else ()

    monkeypatch.setattr(ext, "register_args", register_args)
    monkeypatch.setattr(ext, "extra_argv", extra_argv)


class Refusal:
    """Replace ``ext.batch_allowed``; refuse with a reason or with False."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, reason: bool) -> None:
        self.calls = 0

        def batch_allowed(
            spec: Any,
            *,
            settings: Settings,
            args: argparse.Namespace | None = None,
        ) -> bool:
            self.calls += 1
            if reason:
                raise SettingsError(REASON)
            return False

        monkeypatch.setattr(ext, "batch_allowed", batch_allowed)


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _ui_session() -> Session:
    args = build_parser().parse_args(["ui", FLAG, NAME])
    return Session(Settings(), None, {}, args)


def test_the_public_default_gives_no_extra_words() -> None:
    assert ext.extra_argv(None) == ()
    assert ext.extra_argv(argparse.Namespace(command="ui")) == ()


# Copied commands


@pytest.mark.usefixtures("scratch")
def test_the_review_command_ends_with_the_extension_flag(tmp_path: Path) -> None:
    session = _ui_session()
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    state = State(OPTIONS, {}, {"input": pdf, "type": "pdf", "method": "llm"})

    words = equivalent_command(state, session)

    assert tuple(words[:2]) == PROGRAM
    assert words[-2:] == [FLAG, NAME]
    args = build_parser().parse_args(words[1:])
    assert args.scratch_access == NAME
    expected = resolve(session.settings, state.answers).spec
    assert resolve(Settings(), OPTIONS.values(args)).spec == expected


@pytest.mark.usefixtures("scratch")
def test_the_repair_command_ends_with_the_extension_flag(tmp_path: Path) -> None:
    session = _ui_session()
    target = tmp_path / "doc.txt"
    target.write_text("text", encoding="utf-8")
    flow = repair_ui.flow()
    state = State(REPAIR_OPTIONS, {}, {"input": target, "errors_only": True})

    words = flow.command(state, session)

    assert words[-2:] == [FLAG, NAME]
    args = build_parser().parse_args(words[1:])
    assert args.scratch_access == NAME
    reparsed = State(REPAIR_OPTIONS, {}, REPAIR_OPTIONS.values(args))
    assert flow.resolve(reparsed, session).spec == flow.resolve(state, session).spec


def test_without_extension_flags_the_command_is_unchanged(tmp_path: Path) -> None:
    session = Session(Settings(), None, {}, argparse.Namespace(command="ui"))
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    state = State(OPTIONS, {}, {"input": pdf, "type": "pdf"})

    words = equivalent_command(state, session)

    assert FLAG not in words
    assert words[-1] != "--settings"


# Batch refusal with a reason


@pytest.mark.usefixtures("key")
def test_the_run_plan_raises_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Refusal(monkeypatch, reason=True)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)

    with pytest.raises(core.ConfigurationError, match=REASON) as raised:
        core.plan(make_spec(pdf, type="pdf", batch=True), make_settings())

    assert raised.value.exit_code == 2


@pytest.mark.usefixtures("key")
def test_the_transcribe_command_exits_2_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    Refusal(monkeypatch, reason=True)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)
    argv = ["transcribe", "--input", str(pdf), "--type", "pdf", "--batch"]

    assert entry.main([*argv, "--dry-run", "--json"]) == 2

    assert REASON in capsys.readouterr().err


@pytest.mark.usefixtures("key")
async def test_the_repair_plan_raises_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Refusal(monkeypatch, reason=True)
    stored = await store_document(tmp_path, ["failed"])

    with pytest.raises(RepairPlanError, match=REASON):
        plan_repair(spec_for(stored.output, "batch"), Settings())


@pytest.mark.parametrize("reason", [True, False])
def test_the_wizards_do_not_offer_a_refused_batch(
    monkeypatch: pytest.MonkeyPatch, reason: bool
) -> None:
    refusal = Refusal(monkeypatch, reason=reason)
    session = Session(Settings(), None, {}, argparse.Namespace(command="ui"))

    assert not wizard.batch_available(new_state(session, OPTIONS), session)
    assert not repair_ui._batch_available(new_state(session, REPAIR_OPTIONS), session)
    assert refusal.calls == 2


# Refusal without a reason


@pytest.mark.usefixtures("key")
def test_false_keeps_the_generic_run_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Refusal(monkeypatch, reason=False)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", pages=1)

    with pytest.raises(
        core.ConfigurationError,
        match="^Batch submission is not available for this run.$",
    ):
        core.plan(make_spec(pdf, type="pdf", batch=True), make_settings())


@pytest.mark.usefixtures("key")
async def test_false_keeps_the_generic_repair_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Refusal(monkeypatch, reason=False)
    stored = await store_document(tmp_path, ["failed"])

    with pytest.raises(
        RepairPlanError,
        match="^Batch submission is not available for this repair.$",
    ):
        plan_repair(spec_for(stored.output, "batch"), Settings())
