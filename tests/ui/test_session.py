"""The session layer: paths, tasks, the wizard frame and the input inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import ext
from chronotranscriber.common.options import Option, OptionTable
from chronotranscriber.common.testing import KEEP, ScriptedPrompter
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    Decision,
    Line,
    Nav,
    Review,
    State,
    Step,
    StepContext,
)
from chronotranscriber.spec import (
    BATCH_CHECK_OPTIONS,
    OPTIONS,
    REPAIR_OPTIONS,
    resolve_spec,
)
from chronotranscriber.ui.session import (
    INVENTORY,
    TASKS,
    Inventory,
    RunRequest,
    Session,
    TaskFlow,
    TaskSwitch,
    absolute,
    build_wizard,
    can_reset,
    command_words,
    has_settings_default,
    inventory_of,
    new_state,
    take_inventory,
    task_step,
)
from tests.characterization.inputs import make_scanned_pdf
from tests.ui.conftest import SessionMaker


@pytest.mark.parametrize("quote", ['"', "'"])
def test_a_pasted_quoted_path_resolves_to_the_file(
    scanned_pdf: Path, quote: str
) -> None:
    pasted = f"  {quote}{scanned_pdf}{quote} "
    assert absolute(pasted) == scanned_pdf
    assert absolute(pasted).is_file()


def test_absolute_keeps_unmatched_quotes_and_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert absolute("scans") == tmp_path / "scans"
    assert absolute("\"scans'") == tmp_path / "\"scans'"
    assert absolute(Path("a") / "b.pdf") == tmp_path / "a" / "b.pdf"


def _ask_task(current: str, answer: Any) -> tuple[Nav, ScriptedPrompter]:
    prompter = ScriptedPrompter([answer])
    step = task_step(current)
    ctx = StepContext(State(OPTIONS), prompter)
    with prompter:
        result = step.ask(ctx)
    return result, prompter


def test_task_step_returns_done_for_the_current_task() -> None:
    result, prompter = _ask_task("repair", KEEP)
    assert result is DONE
    assert " > repair a transcription" in prompter.lines
    step = task_step("repair")
    assert (step.name, step.sets) == ("task", ())


def test_task_step_returns_back_on_back() -> None:
    result, _prompter = _ask_task("transcribe", BACK)
    assert result is BACK


def test_task_step_raises_task_switch_for_another_task() -> None:
    with pytest.raises(TaskSwitch) as raised:
        _ask_task("transcribe", "batch-check")
    assert raised.value.task == "batch-check"


def _flow(table: OptionTable, ran: list[RunRequest] | None = None) -> TaskFlow:
    def ask(ctx: StepContext) -> Nav:
        ctx.prompter.show("  own step")
        return DONE

    def review(state: State, session: Session) -> Review:
        return Review([Line("input", ["nothing"])])

    def command(state: State, session: Session) -> list[str]:
        return command_words(("chronotranscriber", "x"), ["--y"], session, table)

    def run(request: RunRequest) -> int:
        (ran if ran is not None else []).append(request)
        return 0

    return TaskFlow(
        name="transcribe",
        title=TASKS["transcribe"],
        program=("chronotranscriber", "x"),
        table=table,
        steps=lambda session: [Step("own", "Own", ask=ask)],
        review=review,
        command=command,
        resolve=lambda state, session: resolve_spec({"input": Path()}),
        run=run,
    )


def _extension_step() -> Step:
    return Step("extension", "Extension", ask=lambda ctx: DONE)


def test_build_wizard_puts_the_task_step_first_and_adds_extension_steps(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "wizard_steps", lambda: [_extension_step()])
    prompter = ScriptedPrompter([KEEP, "quit"])
    wizard = build_wizard(_flow(OPTIONS), session, prompter, width=100)
    assert [step.name for step in wizard.steps] == ["task", "own", "extension"]
    assert wizard.title == "ChronoTranscriber"
    with prompter:
        outcome = wizard.run()
    assert outcome.action is Decision.QUIT
    assert prompter.lines[1] == "ChronoTranscriber | new transcription   step 1"
    settings = str(absolute(session.settings_path or ""))
    assert outcome.command == ("chronotranscriber", "x", "--y", "--settings", settings)


def test_build_wizard_adds_no_extension_steps_to_other_tables(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "wizard_steps", lambda: [_extension_step()])
    wizard = build_wizard(_flow(REPAIR_OPTIONS), session, ScriptedPrompter([]))
    assert [step.name for step in wizard.steps] == ["task", "own"]
    assert wizard.state.table is REPAIR_OPTIONS


def test_new_state_keeps_the_settable_defaults_of_the_table(
    make_session: SessionMaker,
) -> None:
    session = make_session({"model": "gpt-5.6-terra", "formats": ["md"]})
    transcribe = new_state(session, OPTIONS)
    assert transcribe.defaults == {"model": "gpt-5.6-terra", "formats": ("md",)}
    assert transcribe.from_settings("model")
    assert new_state(session, REPAIR_OPTIONS).defaults == {"model": "gpt-5.6-terra"}
    assert new_state(session, BATCH_CHECK_OPTIONS).defaults == {}
    assert new_state(session, REPAIR_OPTIONS).answers == {}


def test_command_words_add_the_settings_file_only_where_the_table_has_it(
    session: Session,
) -> None:
    program = ("chronotranscriber", "batch", "check")
    settings = str(absolute(session.settings_path or ""))
    assert command_words(program, ["--json"], session, BATCH_CHECK_OPTIONS) == [
        *program,
        "--json",
        "--settings",
        settings,
    ]
    bare = OptionTable([Option("input", "--input", "input")])
    assert command_words(program, [], session, bare) == list(program)
    started_without = Session(session.settings, None, {})
    assert command_words(program, [], started_without, BATCH_CHECK_OPTIONS) == list(
        program
    )


def test_reset_is_offered_only_without_a_settings_default(
    make_session: SessionMaker,
) -> None:
    state = new_state(make_session({"max_output_tokens": 4000}), OPTIONS)
    assert has_settings_default(state, "max_output_tokens")
    assert not can_reset(state, "max_output_tokens")
    assert not has_settings_default(state, "top_p")
    assert can_reset(state, "top_p")


def test_inventory_describes_a_folder_of_every_kind(mixed_folder: Path) -> None:
    inventory = take_inventory(mixed_folder)
    assert inventory.describe() == (
        "1 PDF, 3 pages; 1 image folder, 2 images; 1 EPUB; 1 audio file"
    )
    assert [item.type for item in inventory.items] == [
        "pdf",
        "images",
        "epub",
        "audio",
    ]
    assert inventory.label(0) == "   1  book.pdf  (PDF, 3 pages)"
    assert inventory.label(1) == "   2  scans  (image folder, 2 images)"
    assert inventory.label(2) == "   3  novel.epub  (EPUB)"
    assert inventory.label(3) == "   4  interview.wav  (audio file)"


def test_inventory_of_single_inputs(
    scanned_pdf: Path, image_folder: Path, mixed_folder: Path
) -> None:
    assert take_inventory(scanned_pdf).describe() == "1 PDF, 3 pages"
    assert take_inventory(image_folder).describe() == "1 image folder, 3 images"
    audio = mixed_folder / "interview.wav"
    assert take_inventory(audio).describe() == "1 audio file"
    assert take_inventory(mixed_folder / "missing").describe() == "nothing to process"


def test_recursive_inventory_finds_files_in_subfolders(tmp_path: Path) -> None:
    make_scanned_pdf(tmp_path / "archive" / "1791" / "book.pdf", pages=2)
    assert take_inventory(tmp_path / "archive").describe() == "nothing to process"
    found = take_inventory(tmp_path / "archive", recursive=True)
    assert found.describe() == "1 PDF, 2 pages"
    assert found.recursive


def test_inventory_of_caches_by_path_and_recursion(
    mixed_folder: Path, image_folder: Path
) -> None:
    state = State(OPTIONS)
    first = inventory_of(state, mixed_folder)
    assert state.data[INVENTORY] is first
    assert inventory_of(state, mixed_folder) is first
    other = inventory_of(state, image_folder)
    assert isinstance(other, Inventory)
    assert other.path == image_folder
    state.set("recursive", True)
    assert inventory_of(state, image_folder).recursive
