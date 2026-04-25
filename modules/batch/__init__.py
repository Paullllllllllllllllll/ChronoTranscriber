"""Provider-agnostic batch operations: submit, check, cancel, repair.

Single deep module covering every batch concern across providers
(OpenAI, Anthropic, Google): request assembly, provider backends,
job submission and status, result download, cancellation, and
failed-transcription repair.

Public surface below; prefer importing from here rather than submodules.
"""

from modules.batch.backends import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    clear_backend_cache,
    get_batch_backend,
    supports_batch,
)
from modules.batch.cancel import cancel_batch_by_id
from modules.batch.check import process_all_batches, run_batch_finalization
from modules.batch.jsonl import (
    ImageMetadata,
    backup_file,
    extract_batch_ids,
    extract_image_metadata,
    extract_transcription_records,
    find_companion_files,
    get_processed_image_names,
    is_batch_jsonl,
    read_jsonl_records,
    write_jsonl_record,
)
from modules.batch.mapping import (
    diagnose_batch_failure,
    extract_custom_id_mapping,
)
from modules.batch.requests import (
    DEFAULT_BATCH_CHUNK_SIZE,
    create_batch_request_line,
    encode_image_to_data_url,
    get_batch_chunk_size,
    process_batch_transcription,
    submit_batch as submit_batch_legacy,
    write_batch_file,
)
from modules.batch.submission import submit_batch

__all__ = [
    # Backends
    "BatchBackend", "BatchHandle", "BatchRequest", "BatchResultItem",
    "BatchStatus", "BatchStatusInfo", "get_batch_backend",
    "supports_batch", "clear_backend_cache",
    # Cancel
    "cancel_batch_by_id",
    # Check
    "process_all_batches", "run_batch_finalization",
    # JSONL
    "ImageMetadata", "read_jsonl_records", "write_jsonl_record",
    "extract_image_metadata", "extract_batch_ids",
    "extract_transcription_records", "get_processed_image_names",
    "is_batch_jsonl", "backup_file", "find_companion_files",
    # Mapping
    "diagnose_batch_failure", "extract_custom_id_mapping",
    # Requests
    "DEFAULT_BATCH_CHUNK_SIZE", "get_batch_chunk_size",
    "encode_image_to_data_url", "create_batch_request_line",
    "write_batch_file", "submit_batch_legacy",
    "process_batch_transcription",
    # Submission
    "submit_batch",
]
