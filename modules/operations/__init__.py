"""Operational orchestration modules.

This package contains high-level operations used by entry-point scripts,
organized for reusability and testability. The entry-point scripts in
`main/` import and delegate to these modules, keeping the CLI layers thin.
"""

# Avoid circular imports - use direct imports instead of re-exporting
# Note: get_openai_client and validate_api_key moved to modules.llm.openai_sdk_utils
__all__ = [
    "process_all_batches",
    "run_batch_finalization",
    "repair_main",
    # JSONL utilities
    "read_jsonl_records",
    "write_jsonl_record",
    "extract_image_metadata",
    "extract_batch_ids",
    "is_batch_jsonl",
    "backup_file",
    "find_companion_files",
    "ImageMetadata",
]
