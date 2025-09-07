# check_batches.py
"""
Script to check whether batch jobs have finished successfully (i.e., are
marked as completed) and—if so—to download and process them. Temporary .jsonl
files and image folders will only be deleted if the final output is
successfully written and all batches in a JSONL file are completed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from modules.logger import setup_logger
from modules.operations.batch_check import run_batch_finalization

logger = setup_logger(__name__)


def load_config() -> Tuple[List[Path], Dict[str, Any]]:
    """
    Load and parse configuration YAML files. Identify directories to scan and
    retrieve general processing settings (e.g., concurrency limits, keep_raw_images flag).
    """
    from modules.operations.batch_check import load_config as _impl
    return _impl()


def diagnose_batch_failure(batch_id: str, client: OpenAI) -> str:
    # Delegated to centralized utility in modules.batch_utils
    from modules.batch_utils import diagnose_batch_failure as _diag
    return _diag(batch_id, client)


def extract_custom_id_mapping(
    temp_file: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    # Delegated to centralized utility in modules.batch_utils
    from modules.batch_utils import extract_custom_id_mapping as _extract
    return _extract(temp_file)


def process_all_batches(
    root_folder: Path, processing_settings: Dict[str, Any], client: Any
) -> None:
    """
    Scans the root folder for *_transcription.jsonl files, locates batch IDs
    within those files, checks if ALL batches for a file are completed, and if so,
    downloads the results and writes them to a final text file while preserving
    the original image order.
    """
    from modules.operations.batch_check import process_all_batches as _impl
    return _impl(root_folder, processing_settings, client)


def diagnose_api_issues() -> None:
    """
    Provide diagnostics on common API issues.
    """
    from modules.operations.batch_check import diagnose_api_issues as _impl
    return _impl()


def main() -> None:
    """
    Entrypoint for checking all directories for batch outputs and finalizing them.
    """
    # Delegate to centralized operations module to perform diagnostics,
    # scan directories, and finalize batch outputs.
    run_batch_finalization(run_diagnostics=True)


if __name__ == "__main__":
    main()