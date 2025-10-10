"""CLI argument parsing utilities for non-interactive script execution.

This module provides argument parsers for each main script when running in CLI mode
(interactive_mode: false in paths_config.yaml).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def create_transcriber_parser() -> argparse.ArgumentParser:
    """Create argument parser for unified_transcriber.py in CLI mode.
    
    Returns:
        Configured ArgumentParser for transcription operations
    """
    parser = argparse.ArgumentParser(
        description="ChronoTranscriber - Historical Document Transcription (CLI Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe images with GPT
  python main/unified_transcriber.py --input images/my_folder --output results --type images --method gpt

  # Transcribe PDFs with Tesseract OCR
  python main/unified_transcriber.py --input pdfs --output output --type pdfs --method tesseract

  # Batch processing with custom schema
  python main/unified_transcriber.py --input images/docs --output output --type images --method gpt --batch --schema my_schema
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input path (relative or absolute). For images: folder containing subfolders. For PDFs: folder with PDF files."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path (relative or absolute) where results will be saved."
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["images", "pdfs", "epubs"],
        required=True,
        help="Type of documents to process: 'images', 'pdfs', or 'epubs'."
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["native", "tesseract", "gpt"],
        required=True,
        help="Transcription method. Use 'native' for PDFs/EPUBs, 'tesseract' for OCR, or 'gpt' for AI transcription."
    )
    
    # Optional arguments for GPT
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch processing (only for GPT method). More cost-effective for large jobs."
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        help="Name of the transcription schema to use (for GPT method). If not specified, uses default schema."
    )
    
    parser.add_argument(
        "--context",
        type=str,
        help="Path to additional context file (for GPT method). If not specified, no additional context is used."
    )
    
    # File selection options
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to process (relative to input path). If not specified, processes all files."
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process files recursively in subdirectories."
    )
    
    return parser


def create_repair_parser() -> argparse.ArgumentParser:
    """Create argument parser for repair_transcriptions.py in CLI mode.
    
    Returns:
        Configured ArgumentParser for repair operations
    """
    parser = argparse.ArgumentParser(
        description="ChronoTranscriber Repair Tool - Fix failed transcriptions (CLI Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repair all failures in a transcription
  python main/repair_transcriptions.py --transcription results/my_doc_transcription.txt

  # Repair only API errors with batch processing
  python main/repair_transcriptions.py --transcription results/doc.txt --errors-only --batch

  # Repair specific line indices
  python main/repair_transcriptions.py --transcription results/doc.txt --indices 0,5,12
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--transcription", "-t",
        type=str,
        required=True,
        help="Path to the transcription file to repair (relative or absolute)."
    )
    
    # Failure type selection
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only repair '[transcription error]' lines."
    )
    
    parser.add_argument(
        "--not-possible",
        action="store_true",
        help="Include '[Transcription not possible]' lines for repair."
    )
    
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Include '[No transcribable text]' lines for repair."
    )
    
    parser.add_argument(
        "--all-failures",
        action="store_true",
        help="Repair all failure types (equivalent to setting all flags)."
    )
    
    # Line selection
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated line indices to repair (e.g., '0,5,12'). If not specified, repairs all detected failures."
    )
    
    # Processing mode
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch processing mode (waits for completion). Default is synchronous mode."
    )
    
    return parser


def create_check_batches_parser() -> argparse.ArgumentParser:
    """Create argument parser for check_batches.py in CLI mode.
    
    Returns:
        Configured ArgumentParser for batch checking operations
    """
    parser = argparse.ArgumentParser(
        description="ChronoTranscriber Batch Checker - Check and finalize batch jobs (CLI Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check batches in specific directory
  python main/check_batches.py --directory results

  # Check batches without diagnostics
  python main/check_batches.py --directory results --no-diagnostics
        """
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Directory to scan for batch files (relative or absolute). If not specified, uses configured output directories."
    )
    
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip running API diagnostics."
    )
    
    return parser


def create_cancel_batches_parser() -> argparse.ArgumentParser:
    """Create argument parser for cancel_batches.py in CLI mode.
    
    Returns:
        Configured ArgumentParser for batch cancellation operations
    """
    parser = argparse.ArgumentParser(
        description="ChronoTranscriber Batch Canceller - Cancel non-terminal batch jobs (CLI Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cancel all non-terminal batches
  python main/cancel_batches.py

  # Cancel specific batches by ID
  python main/cancel_batches.py --batch-ids batch_123 batch_456
        """
    )
    
    parser.add_argument(
        "--batch-ids",
        nargs="+",
        help="Specific batch IDs to cancel. If not specified, attempts to cancel all non-terminal batches."
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with caution)."
    )
    
    return parser


def resolve_path(path_str: str, base_path: Optional[Path] = None) -> Path:
    """Resolve a path string to an absolute Path object.
    
    Args:
        path_str: Path string (relative or absolute)
        base_path: Base path for relative paths (defaults to cwd)
        
    Returns:
        Absolute Path object
    """
    path = Path(path_str)
    
    if path.is_absolute():
        return path
    
    # Resolve relative to base_path or cwd
    if base_path:
        return (base_path / path).resolve()
    return path.resolve()


def validate_input_path(path: Path, must_exist: bool = True) -> None:
    """Validate an input path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must already exist
        
    Raises:
        ValueError: If path is invalid
    """
    if must_exist and not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    
    if must_exist and not (path.is_file() or path.is_dir()):
        raise ValueError(f"Input path is neither a file nor directory: {path}")


def validate_output_path(path: Path, create_parents: bool = True) -> None:
    """Validate and prepare an output path.
    
    Args:
        path: Path to validate
        create_parents: Whether to create parent directories
        
    Raises:
        ValueError: If path is invalid
    """
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {path}")
    
    if create_parents:
        path.mkdir(parents=True, exist_ok=True)


def parse_indices(indices_str: str) -> List[int]:
    """Parse a comma-separated string of indices.
    
    Args:
        indices_str: String like "0,5,12" or "1-5,10"
        
    Returns:
        List of integer indices
        
    Raises:
        ValueError: If string format is invalid
    """
    result = set()
    
    for part in indices_str.split(","):
        part = part.strip()
        if not part:
            continue
            
        if "-" in part:
            # Range: "1-5"
            try:
                start, end = part.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                result.update(range(start_idx, end_idx + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Single index: "5"
            try:
                result.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid index: {part}")
    
    return sorted(result)
