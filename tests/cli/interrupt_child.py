"""Child process of the Ctrl+Break test.

Usage, from the repository root:

    python tests/cli/interrupt_child.py PDF OUT SETTINGS

Runs ``chronotranscriber transcribe --input PDF --type pdf --method native
--output OUT --settings SETTINGS --json`` through the console entry point.
After the first page
is stored the child prints READY and waits, so the parent can send Ctrl+Break
while the document is leased. The exit code is the command's.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chronotranscriber.methods.native as native  # noqa: E402
from chronotranscriber.__main__ import main  # noqa: E402

_real_write = native._write


async def _write_then_wait(*args: Any, **kwargs: Any) -> None:
    await _real_write(*args, **kwargs)
    print("READY", flush=True)  # noqa: T201 - the parent reads stdout
    await asyncio.sleep(300)


def run(argv: list[str]) -> int:
    native._write = _write_then_wait
    pdf, out, settings = argv
    return main(
        [
            "transcribe",
            "--input",
            pdf,
            "--type",
            "pdf",
            "--method",
            "native",
            "--output",
            out,
            "--settings",
            settings,
            "--json",
        ]
    )


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
