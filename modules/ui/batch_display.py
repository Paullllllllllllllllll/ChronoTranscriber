"""Batch processing display utilities.

Provides specialized display functions for batch operations, including
batch summaries, progress indicators, and transcription error reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.ui.prompts import (
    ui_print,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_separator,
    PromptStyle,
)


def _format_page_image(page_number: Optional[int], image_name: str) -> str:
    """Format page/image identifier for display."""
    if page_number is None:
        return f"({image_name})"
    return f"page {page_number} ({image_name})"


def display_batch_summary(batches: List[Dict[str, Any]]) -> None:
    """Display a summary of batch jobs grouped by status.
    
    Args:
        batches: List of batch objects (dicts or SDK objects)
    """
    if not batches:
        print_info("No batches found.")
        return
    
    status_groups: Dict[str, List[Any]] = {}
    for batch in batches:
        status_val = getattr(batch, "status", None)
        if status_val is None and isinstance(batch, dict):
            status_val = batch.get("status")
        status = (status_val or "").lower()
        status_groups.setdefault(status, []).append(batch)
    
    ui_print("\n" + PromptStyle.DOUBLE_LINE * 80, PromptStyle.HEADER)
    ui_print("  BATCH SUMMARY", PromptStyle.HEADER)
    ui_print(PromptStyle.DOUBLE_LINE * 80, PromptStyle.HEADER)
    ui_print(f"\n  Total batches: {len(batches)}", PromptStyle.HIGHLIGHT)
    
    for status, batch_list in sorted(status_groups.items()):
        status_color = (
            PromptStyle.SUCCESS if status == "completed" else
            PromptStyle.ERROR if status in ["failed", "cancelled"] else
            PromptStyle.WARNING if status in ["validating", "in_progress", "finalizing"] else
            PromptStyle.INFO
        )
        ui_print(f"    • {status.capitalize()}: {len(batch_list)} batch(es)", status_color)
    
    in_progress_statuses = {"validating", "in_progress", "finalizing"}
    for status in in_progress_statuses:
        if status in status_groups and status_groups[status]:
            ui_print(f"\n  {status.capitalize()} Batches:", PromptStyle.WARNING)
            print_separator(PromptStyle.LIGHT_LINE, 80)
            for batch in status_groups[status]:
                batch_id = getattr(batch, "id", None)
                batch_status = getattr(batch, "status", None)
                if isinstance(batch, dict):
                    batch_id = batch.get("id", batch_id)
                    batch_status = batch.get("status", batch_status)
                ui_print(f"    • Batch ID: {batch_id} | Status: {batch_status}", PromptStyle.DIM)


def display_batch_processing_progress(
    temp_file: Path, batch_ids: List[str], completed_count: int, missing_count: int
) -> None:
    """Display progress information for batch file processing.
    
    Args:
        temp_file: The temporary file being processed
        batch_ids: List of batch IDs found in the file
        completed_count: Number of completed batches
        missing_count: Number of missing batches
    """
    ui_print(f"\n{PromptStyle.SINGLE_LINE * 80}", PromptStyle.DIM)
    ui_print(f"  Processing: {temp_file.name}", PromptStyle.HIGHLIGHT)
    ui_print(f"{PromptStyle.SINGLE_LINE * 80}", PromptStyle.DIM)
    ui_print(f"  Found {len(batch_ids)} batch ID(s)", PromptStyle.INFO)
    
    if completed_count == len(batch_ids):
        print_success("All batches completed!")
    else:
        in_progress = len(batch_ids) - completed_count - missing_count
        ui_print("  Completed: ", PromptStyle.INFO, end="")
        ui_print(f"{completed_count}", PromptStyle.SUCCESS, end="")
        ui_print(" | Pending: ", PromptStyle.INFO, end="")
        ui_print(f"{in_progress}", PromptStyle.WARNING, end="")
        ui_print(" | Missing: ", PromptStyle.INFO, end="")
        ui_print(f"{missing_count}", PromptStyle.ERROR if missing_count > 0 else PromptStyle.DIM)
        if missing_count > 0:
            print_warning(f"{missing_count} batch ID(s) were not found in the API response")
        if completed_count < len(batch_ids) - missing_count:
            print_info("Some batches are still processing. Try again later.")


def display_batch_cancellation_results(
    cancelled_batches: List[Tuple[str, str, bool]], skipped_batches: List[Tuple[str, str]]
) -> None:
    """Display results of batch cancellation operations.
    
    Args:
        cancelled_batches: List of (batch_id, status, success) tuples
        skipped_batches: List of (batch_id, status) tuples for skipped batches
    """
    success_count = sum(1 for _, _, success in cancelled_batches if success)
    fail_count = len(cancelled_batches) - success_count
    
    ui_print("\n" + PromptStyle.DOUBLE_LINE * 80, PromptStyle.HEADER)
    ui_print("  CANCELLATION SUMMARY", PromptStyle.HEADER)
    ui_print(PromptStyle.DOUBLE_LINE * 80, PromptStyle.HEADER)
    ui_print(f"\n  Total batches found: {len(cancelled_batches) + len(skipped_batches)}", PromptStyle.INFO)
    ui_print(f"  Skipped (terminal status): {len(skipped_batches)}", PromptStyle.DIM)
    ui_print(f"  Attempted to cancel: {len(cancelled_batches)}", PromptStyle.INFO)
    print_success(f"Successfully cancelled: {success_count}")
    
    if fail_count > 0:
        print_error(f"Failed to cancel: {fail_count}")
        ui_print("\n  Failed Cancellations:", PromptStyle.ERROR)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        for batch_id, status, success in cancelled_batches:
            if not success:
                ui_print(f"    • Batch {batch_id} (status: '{status}')", PromptStyle.DIM)
    
    if success_count > 0:
        ui_print("\n  Successfully Cancelled:", PromptStyle.SUCCESS)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        for batch_id, status, success in cancelled_batches:
            if success:
                ui_print(f"    • Batch {batch_id} (was: '{status}')", PromptStyle.DIM)


def print_transcription_item_error(
    image_name: str,
    page_number: Optional[int] = None,
    status_code: Optional[int] = None,
    err_code: Optional[str] = None,
    err_message: Optional[str] = None,
) -> None:
    """Print an error for a single transcription item.
    
    Args:
        image_name: Name of the image file
        page_number: Optional page number
        status_code: Optional HTTP status code
        err_code: Optional error code
        err_message: Optional error message
    """
    label = _format_page_image(page_number, image_name)
    parts: List[str] = []
    if status_code is not None:
        parts.append(f"status={status_code}")
    if err_code:
        parts.append(f"code={err_code}")
    if err_message:
        parts.append(f"message={err_message}")
    detail = " ".join(parts)
    print_error(f"{label} failed in batch" + (f": {detail}" if detail else ""))


def print_transcription_not_possible(image_name: str, page_number: Optional[int] = None) -> None:
    """Print a warning that transcription was not possible for an item.
    
    Args:
        image_name: Name of the image file
        page_number: Optional page number
    """
    label = _format_page_image(page_number, image_name)
    print_warning(f"Model reported transcription not possible for {label}.")


def print_no_transcribable_text(image_name: str, page_number: Optional[int] = None) -> None:
    """Print info that no transcribable text was detected.
    
    Args:
        image_name: Name of the image file
        page_number: Optional page number
    """
    label = _format_page_image(page_number, image_name)
    print_info(f"No transcribable text detected for {label}.")


def display_page_error_summary(error_entries: List[Dict[str, Any]]) -> None:
    """Display a summary of page-level errors from batch processing.
    
    Args:
        error_entries: List of error entry dictionaries
    """
    if not error_entries:
        return
    
    print_warning(f"{len(error_entries)} page(s) failed during batch processing:")
    for e in error_entries:
        img = (e.get("image_info", {}) or {}).get("image_name") or e.get("custom_id", "[unknown image]")
        page = (e.get("image_info", {}) or {}).get("page_number")
        det = (e.get("error_details", {}) or {})
        status = det.get("status_code")
        code = det.get("code")
        msg = det.get("message")
        label = _format_page_image(page, img)
        parts: List[str] = []
        if status is not None:
            parts.append(f"status={status}")
        if code:
            parts.append(f"code={code}")
        if msg:
            parts.append(f"message={msg}")
        ui_print("  • " + label + (": " + " ".join(parts) if parts else ""), PromptStyle.DIM)


def display_transcription_not_possible_summary(count: int) -> None:
    """Display summary count of pages where transcription was not possible.
    
    Args:
        count: Number of pages with transcription not possible
    """
    if count > 0:
        print_info(f"{count} page(s) reported 'transcription not possible' by the model.")
