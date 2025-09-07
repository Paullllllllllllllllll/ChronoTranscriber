"""Operational orchestration modules.

This package contains high-level operations used by entry-point scripts,
organized for reusability and testability. The entry-point scripts in
`main/` import and delegate to these modules, keeping the CLI layers thin.
"""

from .batch_check import process_all_batches, run_batch_finalization
from .repair import main as repair_main
