"""Dispatch the console subcommand; bare ``chronotranscriber`` prints help.

On Windows, Ctrl+Break (``SIGBREAK``) is handled like Ctrl+C
(:mod:`chronotranscriber.common.interrupt`), so an interrupted run releases
its leases and records its end.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from chronotranscriber.cli.parser import UsageError, build_parser
from chronotranscriber.common.interrupt import install_break_handler

# Subcommand path -> (module, function) that runs it.
_HANDLERS: dict[tuple[str, ...], tuple[str, str]] = {
    ("transcribe",): ("chronotranscriber.cli.transcribe", "execute"),
    ("repair",): ("chronotranscriber.cli.repair", "execute"),
    ("batch", "check"): ("chronotranscriber.cli.batch", "execute_check"),
    ("batch", "cancel"): ("chronotranscriber.cli.batch", "execute_cancel"),
    ("render",): ("chronotranscriber.cli.store", "execute_render"),
    ("drop",): ("chronotranscriber.cli.store", "execute_drop"),
    ("ui",): ("chronotranscriber.ui.app", "execute"),
}


def _command(namespace: argparse.Namespace) -> tuple[str, ...]:
    if namespace.command == "batch":
        return ("batch", namespace.batch_command or "")
    return (namespace.command or "",)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the subcommand in *argv* (``sys.argv[1:]`` when None)."""
    install_break_handler()
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help()
        return 0
    try:
        namespace = parser.parse_args(args)
    except UsageError as error:
        from chronotranscriber.cli.command import report_usage_error

        return report_usage_error(error, args)
    handler = _HANDLERS.get(_command(namespace))
    if handler is None:
        help_parser = getattr(namespace, "batch_parser", None) or parser
        help_parser.print_help()
        return 0
    module_name, function = handler
    execute = getattr(importlib.import_module(module_name), function)
    return int(execute(namespace))


if __name__ == "__main__":
    raise SystemExit(main())
