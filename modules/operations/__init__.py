"""Operational orchestration modules.

This package contains high-level operations used by entry-point scripts,
organized for reusability and testability. The entry-point scripts in
`main/` import and delegate to these modules, keeping the CLI layers thin.
"""

# Avoid circular imports - use direct imports instead of re-exporting
__all__ = [
    "process_all_batches",
    "run_batch_finalization",
    "repair_main",
]
