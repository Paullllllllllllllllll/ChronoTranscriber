"""The ui subcommand: a guided run in an interactive terminal.

:func:`execute` is what ``chronotranscriber ui`` dispatches to; :func:`run_app`
does the same with an injectable prompter, console, command stream,
clipboard runner and environment, for tests and recorded sessions. Once the
settings load, log records are held until the wizard ends; they then go to
the settings' log file, except in a dry run, which writes no log file. The
session starts with the transcribe task; choosing another task on the task step rebuilds
the wizard for it, starting at that task's first step. Exit codes are those
of the chosen task's command: 0 when every item completed or nothing ran, 1
when an item failed, 2 for usage and configuration errors, 130 on Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, TextIO

from rich.console import Console

from chronotranscriber import ext
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.log_setup import configure_logging
from chronotranscriber.common.prompts import QuestionaryPrompter
from chronotranscriber.common.wizard import (
    ClipboardRunner,
    Decision,
    Outcome,
    Prompter,
    command_line,
    copy_to_clipboard,
)
from chronotranscriber.settings import Settings, SettingsError, load
from chronotranscriber.ui import batch, repair, store, transcribe
from chronotranscriber.ui.session import (
    RunRequest,
    Session,
    TaskFlow,
    TaskSwitch,
    build_wizard,
    new_state,
)

__all__ = ["FLOWS", "NOT_INTERACTIVE", "execute", "run_app"]

logger = logging.getLogger(__name__)

LOGGER_NAME: Final = "chronotranscriber"

NOT_INTERACTIVE: Final = (
    "chronotranscriber ui needs an interactive terminal (stdin and stdout). "
    "For scripts and agents, use 'chronotranscriber transcribe' with flags; "
    "see 'chronotranscriber transcribe --help'."
)

FLOWS: dict[str, Callable[[], TaskFlow]] = {
    "transcribe": transcribe.flow,
    "repair": repair.flow,
    "batch-check": batch.check_flow,
    "batch-cancel": batch.cancel_flow,
    "render": store.render_flow,
    "drop": store.drop_flow,
}
"""The task flows by their :data:`chronotranscriber.ui.session.TASKS` names."""

FIRST_TASK: Final = "transcribe"


def execute(args: argparse.Namespace) -> int:
    """Run the guided UI for the parsed ``ui`` arguments; return the exit code."""
    settings_path: Path | None = getattr(args, "settings", None)
    return run_app(settings_path, args=args)


def _interactive() -> bool:
    streams = (sys.stdin, sys.stdout)
    return all(stream is not None and stream.isatty() for stream in streams)


def _say(console: Console, text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


class _HeldLog(logging.Handler):
    """Hold the tool's log records until the session knows its log file.

    While it is installed, records reach no file and the console handler
    shows only critical ones. :meth:`settle` writes the held records to the
    settings' log file and keeps logging there, or, for a dry run, drops
    them and writes no log file.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(logging.INFO)
        self.settings = settings
        self.records: list[logging.LogRecord] = []
        self.settled = False
        configure_logging(logger_name=LOGGER_NAME, level=logging.CRITICAL)
        tool = logging.getLogger(LOGGER_NAME)
        tool.addHandler(self)
        tool.setLevel(logging.INFO)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def settle(self, *, to_file: bool) -> None:
        """Stop holding; with *to_file*, log to the file and write the records."""
        if self.settled:
            return
        self.settled = True
        tool = logging.getLogger(LOGGER_NAME)
        tool.removeHandler(self)
        configure_logging(
            logger_name=LOGGER_NAME,
            level=logging.CRITICAL,
            log_file=self.settings.log_file() if to_file else None,
        )
        if not to_file:
            return
        files = [h for h in tool.handlers if isinstance(h, logging.FileHandler)]
        for record in self.records:
            for handler in files:
                if record.levelno >= handler.level:
                    handler.handle(record)


def _dry_run(flow: TaskFlow, session: Session, outcome: Outcome) -> bool:
    """Whether the wizard's outcome runs *flow* as a dry run."""
    if outcome.action is not Decision.RUN or "dry_run" not in flow.table:
        return False
    defaults = new_state(session, flow.table).defaults
    return bool(flow.table.resolve(outcome.answers, defaults)["dry_run"].value)


def run_app(
    settings_path: Path | None = None,
    *,
    args: argparse.Namespace | None = None,
    prompter: Prompter | None = None,
    console: Console | None = None,
    stdout: TextIO | None = None,
    clipboard: ClipboardRunner | None = None,
    environ: Mapping[str, str] | None = None,
    require_tty: bool = True,
) -> int:
    """Run the wizard, then the chosen task, and return the exit code.

    *args* holds the parsed ``ui`` arguments for the extension (default:
    none); *console* shows messages, the run screen and the summary
    (default: stderr); *stdout* receives the copied command; *clipboard*
    runs the clipboard program; *environ* is where the review looks up API
    key variables. ``require_tty=False`` skips the terminal check.
    """
    if require_tty and not _interactive():
        sys.stderr.write(NOT_INTERACTIVE + "\n")
        return int(ExitCode.USAGE)
    screen = console if console is not None else Console(stderr=True)
    held: _HeldLog | None = None
    try:
        try:
            settings = load(settings_path, blocks=ext.settings_blocks())
        except SettingsError as exc:
            _say(screen, f"Error: {exc}")
            return int(ExitCode.USAGE)
        held = _HeldLog(settings)
        session = Session(
            settings,
            settings_path,
            environ if environ is not None else os.environ,
            args,
        )
        return _guided(
            session,
            prompter if prompter is not None else QuestionaryPrompter(),
            screen,
            stdout if stdout is not None else sys.stdout,
            clipboard,
            args if args is not None else argparse.Namespace(),
            held,
        )
    except KeyboardInterrupt:
        if held is not None:
            held.settle(to_file=True)
        _say(screen, "Interrupted by the user.")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:
        if held is not None:
            held.settle(to_file=True)
        logger.exception("Critical error in the guided run: %s", exc)
        _say(screen, f"Unexpected error: {exc}")
        return int(ExitCode.FAILURE)


def _wizard_outcome(
    session: Session, prompter: Prompter, width: int
) -> tuple[TaskFlow, Outcome]:
    """Run the wizard, rebuilding it for each task the user switches to.

    A rebuilt wizard starts at the new task's first step.
    """
    flow = FLOWS[FIRST_TASK]()
    chosen = False
    while True:
        wizard = build_wizard(flow, session, prompter, width=width, chosen=chosen)
        try:
            return flow, wizard.run()
        except TaskSwitch as switch:
            flow = FLOWS[switch.task]()
            chosen = True


def _guided(
    session: Session,
    prompter: Prompter,
    console: Console,
    stdout: TextIO,
    clipboard: ClipboardRunner | None,
    args: argparse.Namespace,
    held: _HeldLog,
) -> int:
    flow, outcome = _wizard_outcome(session, prompter, console.width)
    held.settle(to_file=not _dry_run(flow, session, outcome))
    if outcome.action is Decision.QUIT:
        return int(ExitCode.OK)
    if outcome.action is Decision.COPY and outcome.command is not None:
        line = command_line(outcome.command)
        stdout.write(line + "\n")
        stdout.flush()
        if copy_to_clipboard(line, runner=clipboard):
            _say(console, "Copied the command to the clipboard.")
        else:
            _say(console, "Clipboard unavailable; copy the command above.")
        return int(ExitCode.OK)
    return int(flow.run(RunRequest(session, outcome.answers, console, args)))
