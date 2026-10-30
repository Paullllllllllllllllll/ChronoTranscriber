"""Argument parsing: the top-level parser and one parser per subcommand."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, NoReturn

from chronotranscriber import ext
from chronotranscriber.common.options import HelpFormatter, OptionTable
from chronotranscriber.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    CLASSIFY_OPTIONS,
    DROP_OPTIONS,
    OPTIONS,
    RENDER_OPTIONS,
    REPAIR_OPTIONS,
)

__all__ = [
    "SUBCOMMANDS",
    "UI_DESCRIPTION",
    "Parser",
    "UsageError",
    "add_ui_command",
    "build_parser",
]

PROG = "chronotranscriber"

DESCRIPTION = (
    "Transcribe PDFs, image folders, EPUB and MOBI files and audio recordings "
    "with vision models, Tesseract, native text layers or speech-to-text."
)

TOP_EPILOG = "Run 'chronotranscriber <command> --help' for the options of a command."

PRECEDENCE = """\
Precedence: a flag wins over the settings defaults: block, which wins over the \
code default. --settings FILE reads another settings file instead of \
config/settings.yaml.

"""

JSON_LINE = """
With --json, stdout carries exactly one JSON summary line on every exit; \
progress, warnings and errors go to stderr.
"""

TRANSCRIBE_EPILOG = (
    PRECEDENCE
    + """\
Exit codes:
  0    every item completed or, under --batch, was submitted, or nothing
       was left to do; a dry run that planned, also without API keys
  1    an item failed, or another run leases one (a killed run keeps its
       lease until it expires)
  2    usage or configuration error: no item found, several items without
       --all, --select or --files, --select matching nothing, a database
       the store refuses, or an unset API key variable
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
    + """
Examples:
  Transcribe one PDF with the default vision model:
    chronotranscriber transcribe --input book.pdf --output out
  Every item of a folder, the method chosen per file:
    chronotranscriber transcribe --input scans --type auto --all --output out
  Items 1 to 3 of a folder, and those whose name holds "ledger":
    chronotranscriber transcribe --input scans --select 1-3 --output out
    chronotranscriber transcribe --input scans --select ledger --output out
  Tesseract on an image folder, Markdown output:
    chronotranscriber transcribe --input pages --type images --method tesseract
      --formats md
  Plan a run without API calls and print the summary as JSON:
    chronotranscriber transcribe --input scans --all --dry-run --json
"""
)

REPAIR_EPILOG = (
    PRECEDENCE
    + """\
Exit codes:
  0    every selected page was repaired, or no page was left to repair; a
       dry run that planned, also without API keys
  1    a selected page stayed unrepaired, or the repair work failed
  2    usage or configuration error: a path no store records, a database
       the store refuses, --indices with several documents, or an unset
       API key variable
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
    + """
Examples:
  Repair the transcription errors of one file:
    chronotranscriber repair --input out/book-1a2b3c4d/book.txt
  Repair every failure cause of the documents below a folder through the
  batch API:
    chronotranscriber repair --input out --all-failures --batch
"""
)

BATCH_CHECK_EPILOG = (
    PRECEDENCE
    + """\
Exit codes:
  0    every submission was finalized or is still running, or there was
       none
  1    a submission failed or waits unresolved; under --wait also when
       --wait-hours passed first
  2    usage error, a missing folder, a database the store refuses, or
       --adopt or --abandon naming nothing to act on
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
)

BATCH_CANCEL_EPILOG = (
    PRECEDENCE
    + """\
Exit codes:
  0    every batch was cancelled
  1    a cancellation or a provider listing failed
  2    usage error, a missing folder or a database the store refuses
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
)

RENDER_EPILOG = (
    """\
Exit codes:
  0    every file holds its rendering or was left as edited
  1    a file could not be replaced, or another run holds a document
  2    usage error, or a database the store refuses
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
    + """
Examples:
  Render every document below a folder again:
    chronotranscriber render --input out
  Write Markdown views too, replacing files edited by hand:
    chronotranscriber render --input out --formats txt,md --force
"""
)

DROP_EPILOG = (
    """\
Exit codes:
  0    every selected document was dropped, or in a dry run would be
  1    a document was refused: another run leases it, or a batch
       submission that is open or unresolved holds its pages
  2    usage error: no store at the input, a database the store refuses,
       not exactly one of --select, --files and --all, or a selection
       that matches no stored document
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
    + """
drop deletes each selected document with its pages, responses and render \
records, so the next transcribe run processes it from scratch. A rendered \
file is deleted only while it matches the hash recorded when it was written; \
a file changed outside the tool stays and is named, and while it stays the \
next run skips the item as unstored.

Examples:
  List what dropping the documents whose name holds "ledger" would delete:
    chronotranscriber drop --input out --select ledger --dry-run
  Drop documents 2 and 3 of the store:
    chronotranscriber drop --input out --select 2,3
"""
)

CLASSIFY_EPILOG = (
    PRECEDENCE
    + """\
Exit codes:
  0    every item was classified, its answers asked or reused; a dry run
       that planned, also without API keys
  1    a request failed after its retries, or an item was refused because
       the store holds its document with another input file
  2    usage or configuration error: no classifier, a classifier of another
       unit, no item found, several items without --all, --select or
       --files, a database the store refuses, or an unset API key variable
  130  interrupted (Ctrl+C or Ctrl+Break)
"""
    + JSON_LINE
    + """
Classification only records answers in the store's classifications view; it
changes no transcription. Answers stored for the same pages, classifier and
decision model are reused unless --force is given.
"""
)

CLASSIFY_PAGES_EPILOG = (
    CLASSIFY_EPILOG
    + """
Examples:
  The script and hand of every page of one PDF:
    chronotranscriber classify pages --input book.pdf --output out
      --classifier script_and_hand
  The page kinds of the first 20 pages of every item of a folder:
    chronotranscriber classify pages --input scans --all --output out
      --classifier page_kind --pages first:20
"""
)

CLASSIFY_DOCUMENTS_EPILOG = (
    CLASSIFY_EPILOG
    + """
Examples:
  The document type of every item of a folder, as a dry run:
    chronotranscriber classify documents --input scans --all --output out
      --classifier document_type --dry-run --json
"""
)

UI_DESCRIPTION = (
    "Guided run: choose a task (a new transcription, a repair, batch check or "
    "cancel, render or drop), answer its steps with Back navigation, then "
    "review the values, marked where they come from the settings defaults, "
    "with the equivalent command, and run it, copy the command or quit. Needs "
    "an interactive terminal."
)

UI_EPILOG = (
    "Exit codes are those of the chosen task's command. For scripts and "
    "agents, use 'chronotranscriber transcribe' and the other commands with "
    "flags."
)


class UsageError(Exception):
    """The command line is invalid; *usage* is the parser's usage text."""

    def __init__(self, message: str, *, prog: str, usage: str) -> None:
        super().__init__(message)
        self.prog = prog
        self.usage = usage


class Parser(argparse.ArgumentParser):
    """An argument parser that raises :class:`UsageError` instead of exiting.

    Subcommand parsers created from it share the class, so every parse
    error reaches the caller, which can still write the JSON summary.
    """

    def error(self, message: str) -> NoReturn:
        """Raise :class:`UsageError` with *message* and this parser's usage."""
        raise UsageError(message, prog=self.prog, usage=self.format_usage())


# The action ``add_subparsers`` returns; its class is private to argparse.
Commands = Any


def _add_command(  # noqa: PLR0913 - the parser's texts and the extension switch
    commands: Commands,
    name: str,
    table: OptionTable,
    *,
    help: str,
    description: str,
    epilog: str,
    usage: str | None = None,
    calls_providers: bool = True,
) -> argparse.ArgumentParser:
    """Add one subcommand to *commands* and return its parser.

    The extension's arguments are added only when *calls_providers* is true.
    """
    parser: argparse.ArgumentParser = commands.add_parser(
        name,
        help=help,
        description=description,
        epilog=epilog,
        usage=usage,
        formatter_class=HelpFormatter,
    )
    table.add_arguments(parser)
    if calls_providers:
        ext.register_args(parser)
    return parser


def add_transcribe_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the transcribe subcommand."""
    return _add_command(
        commands,
        "transcribe",
        OPTIONS,
        help="transcribe documents, image folders and recordings",
        description=(
            "Transcribe PDFs, image folders, EPUB and MOBI files and audio "
            "recordings; resumes earlier runs unless --force is given."
        ),
        epilog=TRANSCRIBE_EPILOG,
        usage="%(prog)s --input PATH [options]",
    )


def add_repair_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the repair subcommand."""
    return _add_command(
        commands,
        "repair",
        REPAIR_OPTIONS,
        help="transcribe the failed pages of transcriptions again",
        description=(
            "Find the failed pages of a transcription file, of a database or "
            "of every database at or below a folder, transcribe them again, "
            "store the new results in the database and render the documents' "
            "text files from it again."
        ),
        epilog=REPAIR_EPILOG,
        usage="%(prog)s [--input PATH] [options]",
    )


def add_batch_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the batch subcommand with its check and cancel subcommands."""
    batch: argparse.ArgumentParser = commands.add_parser(
        "batch",
        help="check, finalize or cancel batch submissions",
        description="Check and finalize, or cancel, batch submissions.",
        formatter_class=HelpFormatter,
    )
    actions = batch.add_subparsers(
        dest="batch_command", metavar="COMMAND", title="commands"
    )
    _add_command(
        actions,
        "check",
        BATCH_CHECK_OPTIONS,
        help="check batches and write the outputs of finished ones",
        description=(
            "Check the batch submissions found below a folder and write the "
            "outputs of every finished batch, in the formats of the run that "
            "submitted it unless --formats names others."
        ),
        epilog=BATCH_CHECK_EPILOG,
    )
    _add_command(
        actions,
        "cancel",
        BATCH_CANCEL_OPTIONS,
        help="cancel batches",
        description="Cancel the named batches, or every unfinished batch.",
        epilog=BATCH_CANCEL_EPILOG,
    )
    batch.set_defaults(batch_parser=batch)
    return batch


def add_render_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the render subcommand."""
    return _add_command(
        commands,
        "render",
        RENDER_OPTIONS,
        help="render the txt and md views from the store again",
        description=(
            "Render the files of every document in the databases at or below "
            "a folder from the store, without model calls."
        ),
        epilog=RENDER_EPILOG,
        calls_providers=False,
    )


def add_drop_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the drop subcommand."""
    return _add_command(
        commands,
        "drop",
        DROP_OPTIONS,
        help="remove documents from the store, to transcribe them from scratch",
        description=(
            "Delete the selected documents of a store with all their stored "
            "rows and the rendered files that were not changed outside the "
            "tool, without model calls."
        ),
        epilog=DROP_EPILOG,
        usage=(
            "%(prog)s --input PATH\n"
            "                              (--select EXPR | --files LIST | --all) "
            "[options]"
        ),
        calls_providers=False,
    )


def add_classify_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the classify subcommand with its pages and documents subcommands."""
    classify: argparse.ArgumentParser = commands.add_parser(
        "classify",
        help="record a classifier's answers about pages or documents",
        description=(
            "Ask a decision model a classifier's questions about the pages or "
            "the documents of PDFs and image folders and record the answers "
            "in the store, without transcribing."
        ),
        formatter_class=HelpFormatter,
    )
    actions = classify.add_subparsers(
        dest="classify_command", metavar="COMMAND", title="commands"
    )
    _add_command(
        actions,
        "pages",
        CLASSIFY_OPTIONS["page"],
        help="answer a page classifier about every selected page",
        description=(
            "Render the selected pages of each item, ask a classifier of unit "
            "page about them in groups and record one answer per page and "
            "question."
        ),
        epilog=CLASSIFY_PAGES_EPILOG,
        usage="%(prog)s --input PATH --classifier NAME|PATH [options]",
    )
    _add_command(
        actions,
        "documents",
        CLASSIFY_OPTIONS["document"],
        help="answer a document classifier about each item",
        description=(
            "Send the classifier's selection of each item's pages (the first "
            "10 by default) in one request and record one answer per document "
            "and question."
        ),
        epilog=CLASSIFY_DOCUMENTS_EPILOG,
        usage="%(prog)s --input PATH --classifier NAME|PATH [options]",
    )
    classify.set_defaults(classify_parser=classify)
    return classify


def add_ui_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the ui subcommand: the settings option and the extension arguments."""
    ui: argparse.ArgumentParser = commands.add_parser(
        "ui",
        help="guided run in an interactive terminal",
        description=UI_DESCRIPTION,
        epilog=UI_EPILOG,
        formatter_class=HelpFormatter,
    )
    OPTIONS.subset(("settings",)).add_arguments(ui)
    ext.register_args(ui)
    return ui


SUBCOMMANDS: tuple[Callable[[Commands], argparse.ArgumentParser], ...] = (
    add_transcribe_command,
    add_repair_command,
    add_batch_command,
    add_render_command,
    add_drop_command,
    add_classify_command,
    add_ui_command,
)


def build_parser() -> Parser:
    """Return the top-level parser with every subcommand."""
    parser = Parser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=TOP_EPILOG,
        formatter_class=HelpFormatter,
    )
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", title="commands"
    )
    for add_command in SUBCOMMANDS:
        add_command(commands)
    return parser
