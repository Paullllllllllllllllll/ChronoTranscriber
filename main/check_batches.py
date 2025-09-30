# check_batches.py
"""
CLI script to check whether batch jobs have finished successfully and
download and process completed batches.

This is a thin delegator to the operations.batch.check module.
"""

from __future__ import annotations

from modules.operations.batch.check import run_batch_finalization


def main() -> None:
    """
    Entrypoint for checking all directories for batch outputs and finalizing them.
    """
    run_batch_finalization(run_diagnostics=True)


if __name__ == "__main__":
    main()