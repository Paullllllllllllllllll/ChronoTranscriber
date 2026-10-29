"""Parity between the CLI and the wizard for the seven tasks.

Every option of a task's table must be an argument of its subcommand and must
be set by some scripted wizard session, through a step that lists it, apart
from the exemptions below. Each session's printed command, parsed by the real
parser and resolved with the settings file it names, must resolve to the spec
the session's answers resolve to; it must also do so with an empty defaults
block. The service tier of a provider other than OpenAI is left out of the
command and of the comparison. Seeded random sessions keep that property. The round trip
``parse(to_argv(spec)) == spec`` lives in ``tests/core/test_spec_roundtrip.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import pytest

from chronotranscriber import ext
from chronotranscriber import run as core
from chronotranscriber.cli.parser import build_parser
from chronotranscriber.common.options import Option, OptionTable
from chronotranscriber.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronotranscriber.common.wizard import (
    BACK,
    NO_DEFAULT,
    Choice,
    Decision,
    Nav,
    Outcome,
    State,
    Step,
    StepContext,
    Validator,
    command_display,
    default_index,
)
from chronotranscriber.settings import Settings, load
from chronotranscriber.spec import (
    DEFAULT_SERVICE_TIER,
    METHODS,
    OPTIONS,
    SOURCE_TYPES,
    RunSpec,
)
from chronotranscriber.ui import batch, repair, store, transcribe
from chronotranscriber.ui.review import FORCE_ACTION
from chronotranscriber.ui.session import (
    TASKS,
    Session,
    TaskFlow,
    absolute,
    build_wizard,
    new_state,
)
from tests.characterization.inputs import make_image_folder, make_scanned_pdf
from tests.characterization.inputs_audio import make_wav
from tests.characterization.inputs_ebooks import make_epub, make_mobi
from tests.cli.test_drop_command import ALPHA, seed_two_documents
from tests.ui.conftest import SettingsWriter

EXEMPT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "json": "agent-contract output (one JSON line on stdout); no use in the UI",
        "settings": "given to the ui itself (ui --settings) and added to the command",
        "wait": "the guided batch check makes one pass; waiting serves scripts",
        "poll_minutes": "applies with --wait only, which the UI does not offer",
        "wait_hours": "applies with --wait only, which the UI does not offer",
        "decisions": "decision-model pre-pass; set by flag or settings default",
        "decision_model": "decision-model pre-pass; set by flag or settings default",
        "decision_provider": "decision-model pre-pass; set by flag or settings default",
        "decision_endpoint": "decision-model pre-pass; set by flag or settings default",
    }
)
WIDTH: Final = 100
_MISSING: Final = object()

FLOWS: Final[Mapping[str, Callable[[], TaskFlow]]] = MappingProxyType(
    {
        "transcribe": transcribe.flow,
        "repair": repair.flow,
        "batch-check": batch.check_flow,
        "batch-cancel": batch.cancel_flow,
        "render": store.render_flow,
        "drop": store.drop_flow,
    }
)


def unreached(names: Collection[str], reached: Collection[str]) -> set[str]:
    """Return the options of *names* that no wizard answer reached."""
    return {name for name in names if name not in EXEMPT and name not in reached}


class Workspace:
    """The inputs, files and settings of one test."""

    def __init__(
        self,
        tmp_path: Path,
        write_settings: SettingsWriter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.tmp = tmp_path
        self.write_settings = write_settings
        self.monkeypatch = monkeypatch
        self.pdf = make_scanned_pdf(tmp_path / "inputs" / "book.pdf", pages=2)
        self.pages = make_image_folder(tmp_path / "pages", pages=2)
        self.archive = tmp_path / "my archive"
        make_image_folder(self.archive / "volume_1", pages=1)
        make_image_folder(self.archive / "volume_2", pages=1)
        self.mixed = tmp_path / "mixed"
        make_scanned_pdf(self.mixed / "book.pdf", pages=2)
        make_image_folder(self.mixed / "scans", pages=1)
        make_epub(self.mixed / "novel.epub")
        make_mobi(self.mixed / "tale.mobi")
        make_wav(self.mixed / "interview.wav", seconds=1)
        self.wav = make_wav(tmp_path / "audio" / "talk.wav", seconds=1)
        files = tmp_path / "files"
        files.mkdir()
        self.prompt = self._file(files / "prompt.txt", "Transcribe the page.\n")
        self.context = self._file(files / "context.txt", "Early modern recipes.\n")
        self.context_image = self._file(files / "context image.png", "")
        self.schema = self._file(files / "schema.json", "{}\n")
        self.transcription = self._file(files / "book.txt", "Page one\n")
        self.out = tmp_path / "out dir"
        self.jobs = tmp_path / "jobs"
        self.jobs.mkdir()
        self.store = tmp_path / "store"
        seed_two_documents(self.store, render=False)

    @staticmethod
    def _file(path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def session(
        self, defaults: Mapping[str, Any] | None = None, *, named: bool = True
    ) -> Session:
        """Return a session over a settings file with *defaults*.

        Without *named* the session has the file's settings but no path, as
        the ui started without ``--settings`` has.
        """
        path = self.write_settings(defaults)
        settings = load(path, blocks=ext.settings_blocks())
        return Session(settings, path if named else None, {})


@dataclasses.dataclass(frozen=True)
class Script:
    """A wizard session: task, settings and the answers after the task step."""

    task: str
    session: Session
    answers: Sequence[Any]


@dataclasses.dataclass
class Played:
    """A finished session: outcome, transcript and what each step set."""

    flow: TaskFlow
    session: Session
    outcome: Outcome
    prompter: ScriptedPrompter
    changed: dict[str, set[str]]
    reached: dict[str, set[str]]

    @property
    def command(self) -> tuple[str, ...]:
        command = self.outcome.command
        assert command is not None, self.prompter.transcript
        return command

    def spec(self) -> Any:
        """Resolve the outcome's answers as the task does."""
        defaults = _defaults(self.session, self.flow)
        state = State(self.flow.table, defaults, self.outcome.answers)
        return self.flow.resolve(state, self.session).spec


def _defaults(session: Session, flow: TaskFlow) -> dict[str, Any]:
    return new_state(session, flow.table).defaults


def _recording(
    steps: Iterable[Step],
    table: OptionTable,
    changed: dict[str, set[str]],
    reached: dict[str, set[str]],
) -> list[Step]:
    """Wrap *steps* to record the answers each one changes and reaches."""

    def wrap(step: Step) -> Step:
        def ask(ctx: StepContext) -> Nav:
            before = ctx.state.answers
            result = step.ask(ctx)
            after = ctx.state.answers
            names = {
                name
                for name in before.keys() | after.keys()
                if before.get(name, _MISSING) != after.get(name, _MISSING)
            }
            changed.setdefault(step.name, set()).update(names)
            reached.setdefault(step.name, set()).update(
                name
                for name in names
                if name in after and after[name] != table[name].default
            )
            return result

        return dataclasses.replace(step, ask=ask)

    return [wrap(step) for step in steps]


def play(
    task: str, session: Session, prompter: ScriptedPrompter, *, check: bool = True
) -> Played:
    """Run the wizard of *task* with *prompter* and return what it produced."""
    flow = FLOWS[task]()
    changed: dict[str, set[str]] = {}
    reached: dict[str, set[str]] = {}
    recorded = dataclasses.replace(
        flow,
        steps=lambda s: _recording(flow.steps(s), flow.table, changed, reached),
    )
    wizard = build_wizard(recorded, session, prompter, width=WIDTH)
    try:
        outcome = wizard.run()
        if check:
            prompter.finish()
    except ScriptError as exc:
        raise ScriptError(f"{exc}\n{prompter.transcript}") from exc
    return Played(flow, session, outcome, prompter, changed, reached)


def play_script(script: Script) -> Played:
    """Play *script*, its task step answered with the preselected task."""
    return play(script.task, script.session, ScriptedPrompter([KEEP, *script.answers]))


MODEL_KEEPS: Final = [KEEP] * 4
"""Provider, model, reasoning effort and the recommended tuning."""
RESPONSE_KEEPS: Final = [KEEP] * 6
"""Response format, schema, prompt, context, its fallback and context image."""


def llm_pdf_customized(ws: Workspace) -> Script:
    """An LLM run on a PDF with every model, response and image value set."""
    answers = [
        str(ws.pdf),
        "pdf",
        "llm",
        "folder",
        str(ws.out),
        "mirror",
        ["txt", "md"],
        "openai",
        "<other>",
        "gpt-4o",
        "customize",
        "4000",
        "0.3",
        "0.5",
        "priority",
        "json",
        "plain_text_transcription_schema",
        "file",
        str(ws.prompt),
        "file",
        str(ws.context),
        "file",
        str(ws.context_image),
        "customize",
        "200",
        "high",
        "jpeg",
        "80",
        False,
        "range",
        "1-2",
        True,
        True,
        ["retry_errors"],
        "4",
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def llm_reasoning_and_schema_file(ws: Workspace) -> Script:
    """Reasoning, verbosity, a schema file and no context, as a dry run."""
    answers = [
        str(ws.pdf),
        "pdf",
        KEEP,
        "beside",
        KEEP,
        "openai",
        "gpt-5.6-terra",
        "low",
        "customize",
        "low",
        "",
        KEEP,
        "schema",
        "<other>",
        str(ws.schema),
        KEEP,
        "none",
        "none",
        "recommended",
        KEEP,
        KEEP,
        ["force", "dry_run"],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def llm_named_endpoint(ws: Workspace) -> Script:
    """The settings' named endpoint, in a folder run of image subfolders."""
    answers = [
        str(ws.archive),
        KEEP,
        "images",
        "llm",
        "list",
        ["volume_2"],
        "beside",
        KEEP,
        "endpoint:local",
        "qwen-vl",
        KEEP,
        "text",
        *RESPONSE_KEEPS[:4],
        "recommended",
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def llm_provider_without_implied_model(ws: Workspace) -> Script:
    """OpenRouter with a typed model name that implies no provider, topics as
    the fallback context."""
    answers = [
        str(ws.pdf),
        "pdf",
        KEEP,
        "beside",
        KEEP,
        "openrouter",
        "<other>",
        "qwen-vl",
        KEEP,
        *RESPONSE_KEEPS[:4],
        "text",
        "Recipes, menus",
        KEEP,
        KEEP,
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def tesseract_images_with_back(ws: Workspace) -> Script:
    """Tesseract on an image folder, with Back within and across steps."""
    answers = [
        str(ws.pages),
        KEEP,
        "images",
        "tesseract",
        "beside",
        ["txt"],
        BACK,
        KEEP,
        ["md"],
        "--psm 4",
        "sauvola",
        "all",
        ["keep_working_files"],
        BACK,
        BACK,
        "range",
        "first:1",
        ["keep_working_files"],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def auto_folder_tesseract(ws: Workspace) -> Script:
    """Auto mode on a mixed folder, the text layer of searchable PDFs."""
    answers = [
        str(ws.mixed),
        False,
        "auto",
        "tesseract",
        "text",
        "all",
        "folder",
        str(ws.out),
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def recursive_images(ws: Workspace) -> Script:
    """The subfolder search over image folders, one of them chosen by number."""
    answers = [
        str(ws.archive),
        True,
        "images",
        "tesseract",
        "range",
        "2",
        "beside",
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def native_ebooks(ws: Workspace) -> Script:
    """MOBI then, edited from the review, EPUB, both native."""
    answers = [
        str(ws.mixed),
        False,
        "mobi",
        "native",
        "beside",
        KEEP,
        KEEP,
        [],
        KEEP,
        "edit",
        "method",
        "epub",
        "native",
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def native_mobi(ws: Workspace) -> Script:
    """One MOBI file, native."""
    mobi = ws.mixed / "tale.mobi"
    answers = [str(mobi), "mobi", "native", "beside", KEEP, KEEP, [], KEEP, "copy"]
    return Script("transcribe", ws.session(), answers)


def native_pdf(ws: Workspace) -> Script:
    """The text layer of a PDF."""
    answers = [str(ws.pdf), "pdf", "native", "beside", KEEP, KEEP, [], KEEP, "copy"]
    return Script("transcribe", ws.session(), answers)


def whisper_audio(ws: Workspace) -> Script:
    """Whisper with a model, a language, a chunk length and a range."""
    answers = [
        str(ws.wav),
        "audio",
        "whisper",
        "beside",
        KEEP,
        "small",
        "de",
        "60",
        "range",
        "2-",
        ["keep_working_files"],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def audio_api_google(ws: Workspace) -> Script:
    """The audio API of Google, Back from the model to the provider."""
    answers = [
        str(ws.wav),
        "audio",
        "audio-api",
        "beside",
        KEEP,
        "openai",
        BACK,
        "google",
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


TRANSCRIBE_DEFAULTS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "formats": ["md"],
        "reasoning_effort": "medium",
        "concurrency": 4,
        "service_tier": "priority",
        "layout": "mirror",
    }
)


def transcribe_settings_defaults(ws: Workspace) -> Script:
    """Settings defaults kept; the output folder of the settings replaced."""
    defaults = {**TRANSCRIBE_DEFAULTS, "output": str(ws.tmp / "default out")}
    answers = [
        str(ws.pdf),
        "pdf",
        KEEP,
        "beside",
        KEEP,
        *MODEL_KEEPS,
        *RESPONSE_KEEPS,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "copy",
    ]
    return Script("transcribe", ws.session(defaults), answers)


def transcribe_defaults_unnamed(ws: Workspace) -> Script:
    """Settings defaults of a ui started without ``--settings``."""
    answers = [
        str(ws.pdf),
        "pdf",
        "tesseract",
        "folder",
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "copy",
    ]
    defaults = {**TRANSCRIBE_DEFAULTS, "output": str(ws.tmp / "default out")}
    return Script("transcribe", ws.session(defaults, named=False), answers)


def _complete_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every planned item complete unless the run forces it."""
    real = core.plan

    def plan(spec: RunSpec, settings: Settings, **kwargs: Any) -> core.RunPlan:
        result = real(spec, settings, **kwargs)
        if spec.run.force:
            return result
        return dataclasses.replace(result, to_process=(), skipped=result.to_process)

    monkeypatch.setattr(core, "plan", plan)


def transcribe_force_action(ws: Workspace) -> Script:
    """The review's force action for complete items."""
    _complete_plan(ws.monkeypatch)
    answers = [
        str(ws.pdf),
        "pdf",
        "tesseract",
        "beside",
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        [],
        KEEP,
        FORCE_ACTION,
        "copy",
    ]
    return Script("transcribe", ws.session(), answers)


def repair_customized(ws: Workspace) -> Script:
    """One cause, indices, every model, response and image value, a batch."""
    answers = [
        str(ws.transcription),
        ["not_possible"],
        "0,2-3",
        "openai",
        "<other>",
        "gpt-4o",
        "customize",
        "4000",
        "0.3",
        "0.5",
        "priority",
        "json",
        "plain_text_transcription_schema",
        "file",
        str(ws.prompt),
        "file",
        str(ws.context),
        "file",
        str(ws.context_image),
        "customize",
        "200",
        "high",
        "jpeg",
        "80",
        False,
        True,
        True,
        "copy",
    ]
    return Script("repair", ws.session(), answers)


def repair_endpoint_with_back(ws: Workspace) -> Script:
    """Errors and one other cause at the named endpoint, with Back."""
    answers = [
        str(ws.transcription),
        ["api_error", "no_text"],
        BACK,
        ["api_error", "no_text"],
        "",
        "endpoint:local",
        "qwen-vl",
        KEEP,
        "text",
        *RESPONSE_KEEPS[:4],
        "recommended",
        KEEP,
        "copy",
    ]
    return Script("repair", ws.session(), answers)


def repair_reasoning_defaults(ws: Workspace) -> Script:
    """Every cause, reasoning and verbosity, over settings defaults."""
    defaults = {"reasoning_effort": "low", "service_tier": "priority"}
    answers = [
        str(ws.transcription),
        ["api_error", "not_possible", "no_text", "skipped"],
        KEEP,
        "openai",
        "gpt-5.6-terra",
        "medium",
        "customize",
        "high",
        "",
        KEEP,
        "schema",
        "<other>",
        str(ws.schema),
        KEEP,
        "none",
        "none",
        "recommended",
        KEEP,
        KEEP,
        "copy",
    ]
    return Script("repair", ws.session(defaults), answers)


def repair_provider(ws: Workspace) -> Script:
    """OpenRouter with a typed model name, edited from the review; a fallback
    context file."""
    answers = [
        str(ws.transcription),
        KEEP,
        KEEP,
        *MODEL_KEEPS,
        *RESPONSE_KEEPS[:4],
        "file",
        str(ws.context),
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "edit",
        "model",
        "openrouter",
        "<other>",
        "qwen-vl",
        KEEP,
        "copy",
    ]
    return Script("repair", ws.session(named=False), answers)


def check_adopt(ws: Workspace) -> Script:
    """Named batches, an adopted batch and a format, with Back."""
    answers = [
        str(ws.jobs),
        "b1, b2",
        BACK,
        "b1,b2",
        "adopt",
        "batch_7",
        "chosen",
        ["md"],
        "copy",
    ]
    return Script("batch-check", ws.session(), answers)


def check_abandon_defaults(ws: Workspace) -> Script:
    """An abandoned submission, edited from the review, over a formats default."""
    answers = [
        str(ws.jobs),
        "",
        "adopt",
        "batch_7",
        KEEP,
        "edit",
        "unresolved",
        "abandon",
        "sub-1",
        "copy",
    ]
    return Script("batch-check", ws.session({"formats": ["md"]}), answers)


def cancel_ids_with_back(ws: Workspace) -> Script:
    """Named batch IDs after Back from the folder."""
    answers = ["dir", BACK, "ids", "batch_1, -batch_2", "copy"]
    return Script("batch-cancel", ws.session(), answers)


def cancel_all(ws: Workspace) -> Script:
    """Every unfinished batch."""
    return Script("batch-cancel", ws.session(named=False), ["all", "copy"])


def cancel_folder(ws: Workspace) -> Script:
    """The open submissions below a folder, over settings defaults."""
    answers = ["dir", str(ws.jobs), "copy"]
    return Script("batch-cancel", ws.session(TRANSCRIBE_DEFAULTS), answers)


def render_chosen(ws: Workspace) -> Script:
    """Chosen formats, force and a dry run, with Back."""
    answers = [
        str(ws.out),
        "choose",
        BACK,
        "choose",
        ["txt", "md"],
        True,
        True,
        "copy",
    ]
    return Script("render", ws.session(), answers)


def render_kept(ws: Workspace) -> Script:
    """The formats rendered before, over settings defaults."""
    answers = [str(ws.out), KEEP, KEEP, KEEP, "copy"]
    return Script("render", ws.session(TRANSCRIBE_DEFAULTS), answers)


def drop_by_number(ws: Workspace) -> Script:
    """A document chosen by number, as a dry run."""
    answers = [str(ws.store), "range", "2", True, "copy"]
    return Script("drop", ws.session(), answers)


def drop_from_list_with_back(ws: Workspace) -> Script:
    """A document from the checklist, with Back to the question."""
    answers = [str(ws.store), "list", BACK, "list", [ALPHA], KEEP, "copy"]
    return Script("drop", ws.session(), answers)


def drop_all_over_defaults(ws: Workspace) -> Script:
    """Every document, over settings defaults that set the transcribe --all."""
    answers = [str(ws.store), "all", KEEP, "copy"]
    return Script("drop", ws.session({**TRANSCRIBE_DEFAULTS, "all": True}), answers)


SESSIONS: Final[Mapping[str, Callable[[Workspace], Script]]] = MappingProxyType(
    {
        builder.__name__: builder
        for builder in (
            llm_pdf_customized,
            llm_reasoning_and_schema_file,
            llm_named_endpoint,
            llm_provider_without_implied_model,
            tesseract_images_with_back,
            auto_folder_tesseract,
            recursive_images,
            native_ebooks,
            native_mobi,
            native_pdf,
            whisper_audio,
            audio_api_google,
            transcribe_settings_defaults,
            transcribe_defaults_unnamed,
            transcribe_force_action,
            repair_customized,
            repair_endpoint_with_back,
            repair_reasoning_defaults,
            repair_provider,
            check_adopt,
            check_abandon_defaults,
            cancel_ids_with_back,
            cancel_all,
            cancel_folder,
            render_chosen,
            render_kept,
            drop_by_number,
            drop_from_list_with_back,
            drop_all_over_defaults,
        )
    }
)


@pytest.fixture
def workspace(
    tmp_path: Path, write_settings: SettingsWriter, monkeypatch: pytest.MonkeyPatch
) -> Workspace:
    return Workspace(tmp_path, write_settings, monkeypatch)


def _subparser(words: Sequence[str]) -> argparse.ArgumentParser:
    """Return the parser of the subcommand *words*, such as batch check."""
    parser: argparse.ArgumentParser = build_parser()
    for word in words:
        (commands,) = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        parser = commands.choices[word]
    return parser


def _arguments(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction | argparse._SubParsersAction)
    }


def _extension_dests() -> set[str]:
    parser = argparse.ArgumentParser(add_help=False)
    ext.register_args(parser)
    return {action.dest for action in parser._actions}


@pytest.mark.parametrize("task", list(FLOWS))
def test_every_option_is_an_argument_of_the_subcommand(task: str) -> None:
    flow = FLOWS[task]()
    arguments = _arguments(_subparser(flow.program[1:]))
    # Only the commands that call providers take the extension's options.
    extension = set() if task in ("render", "drop") else _extension_dests()
    assert set(arguments) == {*flow.table.names, *extension}
    for option in flow.table:
        strings = set(arguments[option.name].option_strings)
        expected = {option.flag}
        if option.kind.value == "switch":
            expected.add(option.negative_flag)
        assert strings == expected, option.name


def test_the_ui_parser_accepts_the_settings_option(tmp_path: Path) -> None:
    try:
        arguments = _arguments(_subparser(("ui",)))
    except KeyError:
        pytest.skip("no ui subcommand")
    assert set(arguments) == {"settings", *_extension_dests()}
    path = tmp_path / "other.yaml"
    args = build_parser().parse_args(["ui", "--settings", str(path)])
    assert args.settings == path


def test_the_tasks_are_the_flows() -> None:
    assert list(FLOWS) == list(TASKS)
    for task, make in FLOWS.items():
        flow = make()
        assert flow.name == task
        assert flow.program[0] == "chronotranscriber"


def test_the_exemptions_are_options_no_step_sets(workspace: Workspace) -> None:
    session = workspace.session()
    assert all(reason.strip() for reason in EXEMPT.values())
    for make in FLOWS.values():
        flow = make()
        assert "json" in flow.table
        declared = {name for step in flow.steps(session) for name in step.sets}
        assert not declared & set(EXEMPT), flow.name


def test_an_option_missing_from_every_step_is_reported() -> None:
    names = set(OPTIONS.names)
    extra = Option("extra", "--extra", "an option no step sets")
    table = OptionTable([*OPTIONS, extra])
    assert unreached(names, names) == set()
    assert unreached(table.names, names) == {"extra"}
    assert unreached(names, names - {"top_p", "json"}) == {"top_p"}


@pytest.mark.parametrize("task", list(FLOWS))
def test_the_sessions_reach_every_option_from_their_steps(
    workspace: Workspace, task: str
) -> None:
    changed: dict[str, set[str]] = {}
    reached: dict[str, set[str]] = {}
    for build in SESSIONS.values():
        script = build(workspace)
        if script.task != task:
            continue
        played = play_script(script)
        for name, names in played.changed.items():
            changed.setdefault(name, set()).update(names)
        for name, names in played.reached.items():
            reached.setdefault(name, set()).update(names)
    flow = FLOWS[task]()
    every = {name for names in reached.values() for name in names}
    assert unreached(flow.table.names, every) == set()
    for step in flow.steps(workspace.session()):
        assert changed.get(step.name, set()) <= set(step.sets), step.name
        assert set(step.sets) <= reached.get(step.name, set()), step.name


def test_the_transcribe_sessions_cover_every_type_and_method(
    workspace: Workspace,
) -> None:
    pairs: set[tuple[str, str]] = set()
    for build in SESSIONS.values():
        script = build(workspace)
        if script.task == "transcribe":
            spec = play_script(script).spec()
            pairs.add((spec.input.type, spec.method.name))
    assert {kind for kind, _ in pairs} == set(SOURCE_TYPES)
    assert {method for _, method in pairs} == set(METHODS)


def command_spec(played: Played, *, empty: bool = False) -> Any:
    """Parse the printed command with the real parser and resolve it.

    The settings are those of the file the command names, else the
    session's; *empty* drops their defaults block.
    """
    flow, words = played.flow, played.command
    assert tuple(words[: len(flow.program)]) == flow.program
    args = build_parser().parse_args(list(words[1:]))
    flags = flow.table.values(args)
    path = flags.pop("settings", None)
    settings = played.session.settings
    if path is not None:
        settings = load(path, blocks=ext.settings_blocks())
    if empty:
        settings = dataclasses.replace(settings, defaults=MappingProxyType({}))
    session = Session(settings)
    state = State(flow.table, _defaults(session, flow), flags)
    return flow.resolve(state, session).spec


def _tier_neutral(spec: Any) -> Any:
    """Return *spec* without the service tier of a provider other than OpenAI.

    The printed command leaves that tier out, since only OpenAI calls send it.
    """
    model = getattr(spec, "model", None)
    if model is None or model.provider == "openai":
        return spec
    return dataclasses.replace(
        spec, model=dataclasses.replace(model, service_tier=DEFAULT_SERVICE_TIER)
    )


def check_command(played: Played) -> None:
    """The printed command yields the session's spec, with and without defaults."""
    spec = _tier_neutral(played.spec())
    words = played.command
    assert "--json" not in words
    path = played.session.settings_path
    if path is not None and "settings" in played.flow.table:
        assert words[-2:] == ("--settings", str(absolute(path)))
    else:
        assert "--settings" not in words
    assert _tier_neutral(command_spec(played)) == spec, played.prompter.transcript
    assert _tier_neutral(command_spec(played, empty=True)) == spec, (
        played.prompter.transcript
    )
    shown = command_display(list(words), width=WIDTH)
    lines = played.prompter.lines
    start = len(lines) - lines[::-1].index("Equivalent command:")
    assert lines[start : start + len(shown)] == shown


@pytest.mark.parametrize("name", list(SESSIONS))
def test_the_printed_command_yields_the_wizard_spec(
    workspace: Workspace, name: str
) -> None:
    played = play_script(SESSIONS[name](workspace))
    assert played.outcome.action is Decision.COPY
    check_command(played)


def _flag_value(words: Sequence[str], flag: str) -> str:
    return words[list(words).index(flag) + 1]


def test_settings_defaults_are_kept_and_spelled_out(workspace: Workspace) -> None:
    played = play_script(transcribe_settings_defaults(workspace))
    spec = played.spec()
    assert spec.output.directory is None
    assert spec.output.formats == ("md",)
    assert spec.model.reasoning_effort == "medium"
    words = played.command
    assert "--beside-input" in words and "--output" not in words
    assert _flag_value(words, "--formats") == "md"
    assert _flag_value(words, "--reasoning-effort") == "medium"
    assert _flag_value(words, "--concurrency") == "4"
    assert _flag_value(words, "--service-tier") == "priority"
    assert _flag_value(words, "--layout") == "mirror"
    assert "(s)" in played.prompter.transcript


def test_the_force_action_and_the_dry_run_reach_the_command(
    workspace: Workspace,
) -> None:
    forced = play_script(transcribe_force_action(workspace))
    assert forced.outcome.answers["force"] is True
    assert "--force" in forced.command
    dry = play_script(llm_reasoning_and_schema_file(workspace))
    run = dry.spec().run
    assert (run.dry_run, run.force) == (True, True)
    assert {"--dry-run", "--force"} <= set(dry.command)


RANDOM_SESSIONS: Final = 300
PATIENCE: Final = 120
"""Questions a random session answers before it stops going back."""
_BACK_RATE: Final = 0.08


def _texts(ws: Workspace) -> list[tuple[str, list[str]]]:
    """Candidate texts by question prefix; the first matching prefix wins."""
    inputs = [ws.pdf, ws.pages, ws.archive, ws.mixed, ws.wav, ws.mixed / "tale.mobi"]
    return [
        ("Input (file or folder):", [str(path) for path in inputs]),
        ("Output folder:", [str(ws.out), str(ws.tmp / "other")]),
        ("Model name:", ["gpt-4o", "claude-sonnet-5", "gemini-3.1-pro", "qwen-vl"]),
        ("Model at endpoint", ["qwen-vl", "llava"]),
        ("Max output tokens", ["", "4000"]),
        ("Temperature", ["0.3", "1.5"]),
        ("Top p", ["0.5", "1"]),
        ("Schema file:", [str(ws.schema)]),
        ("Prompt file:", [str(ws.prompt)]),
        ("Context file:", [str(ws.context)]),
        ("Context topics:", ["Recipes, menus", "-fish"]),
        ("Fallback context file:", [str(ws.context)]),
        ("Fallback context topics:", ["Early modern cookery"]),
        ("Context image file:", [str(ws.context_image)]),
        ("Items (", ["1", "1-2", "2,1"]),
        ("Name contains:", ["book", "vol", "e"]),
        ("DPI", ["", "native", "200"]),
        ("JPEG quality", ["", "70"]),
        ("Tesseract configuration:", ["--psm 4", "--oem 1 --psm 6"]),
        ("Spoken language", ["", "de", "auto"]),
        ("Chunk length", ["60", "600"]),
        ("Range:", ["1", "first:1", "2-", "1,2"]),
        ("Parallel requests:", ["1", "16", "4"]),
        ("Transcription file", [str(ws.transcription), str(ws.pdf)]),
        ("Page indices", ["", "0", "0,2-3"]),
        ("Folder with", [str(ws.jobs), str(ws.tmp)]),
        ("Batch IDs to finalize", ["", "b1", "b1, -b2"]),
        ("Batch IDs (comma-separated):", ["batch_1", "b1, -b2"]),
        ("Provider batch ID:", ["batch_7"]),
        ("Submission ID:", ["sub-1", "-sub"]),
        ("Input (folder or", [str(ws.out), str(ws.jobs), str(ws.tmp / "missing")]),
        ("Store (", [str(ws.store), str(ws.store / "chronotranscriber.sqlite")]),
        ("Documents (", ["1", "1-2", "2,1"]),
    ]


class RandomPrompter(ScriptedPrompter):
    """Answer from the offered choices and from valid candidate texts.

    The task step keeps its task and the quit question stays. After
    :data:`PATIENCE` questions no answer is Back and the review copies the
    command; a review that offers no command quits after four times as
    many. Each answer is recorded as a scripted answer, so the transcript
    and the checks of :class:`ScriptedPrompter` apply.
    """

    def __init__(self, rng: random.Random, texts: list[tuple[str, list[str]]]):
        super().__init__([])
        self.rng = rng
        self.texts = texts
        self.asked = 0

    def _back(self) -> bool:
        return self.asked < PATIENCE and self.rng.random() < _BACK_RATE

    def _answer(self, answer: Any) -> None:
        self.asked += 1
        self._answers.append(answer)

    def select[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        default: Any = NO_DEFAULT,
        marker: str | None = None,
    ) -> T | Literal[Nav.BACK]:
        values = [choice.value for choice in choices]
        answer: Any
        if message == "Task:":
            answer = default
        elif message == "Quit without running?":
            answer = "stay"
        elif message == "Start:":
            answer = self._start(values)
        elif self._back():
            answer = BACK
        elif self.rng.random() < 0.3 and default_index(choices, default) is not None:
            answer = KEEP
        else:
            answer = self.rng.choice(values)
        self._answer(answer)
        return super().select(message, choices, default=default, marker=marker)

    def _start(self, values: list[Any]) -> Any:
        if self.asked >= PATIENCE and Decision.COPY.value in values:
            return Decision.COPY.value
        roll = self.rng.random()
        if roll < 0.15 and self.asked < PATIENCE:
            return BACK
        if roll < 0.35 and self.asked < PATIENCE:
            return "edit"
        if FORCE_ACTION in values and roll < 0.5:
            return FORCE_ACTION
        if Decision.COPY.value in values:
            return Decision.COPY.value
        return "edit" if self.asked < 4 * PATIENCE else Decision.QUIT.value

    def checkbox[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        checked: Collection[T] = (),
        marker: str | None = None,
    ) -> list[T] | Literal[Nav.BACK]:
        answer: Any
        if self._back():
            answer = BACK
        else:
            answer = [choice.value for choice in choices if self.rng.random() < 0.5]
            if not answer and choices and self.asked >= PATIENCE:
                answer = [choices[0].value]
        self._answer(answer)
        return super().checkbox(message, choices, checked=checked, marker=marker)

    def _typed(
        self,
        message: str,
        default: str,
        validate: Validator | None,
        marker: str | None,
    ) -> str | Literal[Nav.BACK]:
        if self._back():
            self._answer(BACK)
        else:
            self._answer(self._text(message, default, validate))
        return super()._typed(message, default, validate, marker)

    def _text(self, message: str, default: str, validate: Validator | None) -> str:
        pool = next(
            (texts for prefix, texts in self.texts if message.startswith(prefix)), []
        )
        candidates = [*pool, default] if default else list(pool)
        valid = [
            text for text in candidates if validate is None or validate(text) is None
        ]
        if not valid:
            raise ScriptError(f"no valid candidate text for {message!r}")
        return self.rng.choice(valid)


_DEFAULT_POOL: Final[Mapping[str, Sequence[Any]]] = MappingProxyType(
    {
        "type": ["pdf", "images", "audio", "auto"],
        "method": ["llm", "tesseract", "whisper"],
        "recursive": [True, False],
        "searchable_pdfs": ["text"],
        "layout": ["mirror"],
        "formats": [["md"], ["txt", "md"]],
        "schema": ["plain_text_transcription_schema"],
        "context": ["none"],
        "context_image": ["none"],
        "provider": ["anthropic", "google", "openai"],
        "model": ["gpt-5.6-terra", "claude-sonnet-5", "gemini-3.1-pro", "gpt-4o"],
        "endpoint": ["local"],
        "reasoning_effort": ["low", "medium"],
        "verbosity": ["low"],
        "max_output_tokens": [3000],
        "temperature": [0.4],
        "top_p": [0.7],
        "service_tier": ["priority", "auto"],
        "response_format": ["json", "text", "schema"],
        "dpi": [200, "native"],
        "image_detail": ["low", "auto"],
        "image_format": ["png", "jpeg"],
        "jpeg_quality": [70],
        "grayscale": [True, False],
        "ocr_config": ["--psm 4"],
        "ocr_preprocess": ["none"],
        "whisper_model": ["small"],
        "language": ["de"],
        "audio_chunk_seconds": [120],
        "batch": [True, False],
        "sync_fallback": [True],
        "retry_errors": [True],
        "concurrency": [4],
        "keep_working_files": [True],
    }
)


def _random_defaults(rng: random.Random, ws: Workspace) -> dict[str, Any] | None:
    if rng.random() < 0.3:
        return None
    pool = {
        **_DEFAULT_POOL,
        "output": [str(ws.tmp / "default out")],
        "prompt": [str(ws.prompt)],
        "fallback_context": [str(ws.context), "Early modern cookery"],
        "all": [True],
    }
    keys = rng.sample(sorted(pool), rng.randint(1, 10))
    return {key: rng.choice(pool[key]) for key in keys}


@pytest.mark.parametrize("task", list(FLOWS))
def test_random_sessions_print_the_command_of_their_spec(
    workspace: Workspace, task: str
) -> None:
    texts = _texts(workspace)
    for seed in range(RANDOM_SESSIONS // len(FLOWS)):
        rng = random.Random(f"{task}-{seed}")
        defaults = _random_defaults(rng, workspace)
        session = workspace.session(defaults, named=rng.random() < 0.5)
        prompter = RandomPrompter(rng, texts)
        played = play(task, session, prompter, check=False)
        assert played.outcome.action is Decision.COPY, (seed, prompter.transcript)
        check_command(played)
