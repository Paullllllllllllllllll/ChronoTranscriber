# modules/ui/core.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import sys

from modules.logger import setup_logger
from modules.utils import console_print, check_exit, safe_input
from modules.schema_utils import list_schema_options
from modules.config_loader import PROJECT_ROOT

logger = setup_logger(__name__)


@dataclass
class UserConfiguration:
    """
    Stores user's processing preferences to avoid re-prompting during workflow.
    """

    processing_type: Optional[str] = None  # "images" or "pdfs"
    transcription_method: Optional[str] = None  # "native", "tesseract", or "gpt"
    use_batch_processing: bool = False
    selected_items: List[Path] = None  # files or folders to process
    process_all: bool = False  # flag to process all files/folders
    # Schema selection (GPT only)
    selected_schema_name: Optional[str] = None
    selected_schema_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.selected_items is None:
            self.selected_items = []

    def __str__(self) -> str:
        method_name = {
            "native": "Native PDF extraction",
            "tesseract": "Tesseract OCR",
            "gpt": "GPT-based transcription",
        }.get(self.transcription_method or "", self.transcription_method)
        batch_text = " with batch processing" if self.use_batch_processing else ""
        schema_text = (
            f", Schema: {self.selected_schema_name}"
            if self.transcription_method == "gpt" and self.selected_schema_name
            else ""
        )
        return (
            f"Processing type: {self.processing_type}, "
            f"Method: {method_name}{batch_text}{schema_text}, "
            f"Process all: {self.process_all}, "
            f"Selected items: {len(self.selected_items)}"
        )


class UserPrompt:
    """
    Centralized class for all user interaction and prompting functionality.
    """

    # --- Top-level banners and summaries ---

    @staticmethod
    def display_banner() -> None:
        console_print("\n" + "=" * 80)
        console_print("  CHRONO TRANSCRIBER - Historical Document Digitization Tool")
        console_print("=" * 80)
        console_print(
            "  This tool helps you convert historical documents to searchable text using"
        )
        console_print(
            "  multiple transcription methods tailored to different document types."
        )
        console_print("=" * 80 + "\n")

    @staticmethod
    def display_processing_summary(user_config: UserConfiguration) -> bool:
        """
        Display processing summary and ask for confirmation.
        Returns True if user confirms, False otherwise.
        """
        console_print("\n" + "=" * 80)
        console_print("  PROCESSING SUMMARY")
        console_print("=" * 80)

        item_type = "image folder(s)" if user_config.processing_type == "images" else "PDF file(s)"
        console_print(
            f"\nReady to process {len(user_config.selected_items)} {item_type} with the following settings:"
        )
        console_print(f"  - Document type: {user_config.processing_type.capitalize()}")
        console_print(f"  - Transcription method: {user_config.transcription_method.capitalize()}")
        if user_config.transcription_method == "gpt":
            batch_mode = (
                "Batch (asynchronous)" if user_config.use_batch_processing else "Synchronous"
            )
            console_print(f"  - Processing mode: {batch_mode}")
            if user_config.selected_schema_name:
                console_print(f"  - Schema: {user_config.selected_schema_name}")

        console_print("\nSelected items (first 5 shown):")
        for i, item in enumerate(user_config.selected_items[:5]):
            console_print(f"  {i + 1}. {item.name}")
        if len(user_config.selected_items) > 5:
            console_print(f"  ... and {len(user_config.selected_items) - 5} more")

        while True:
            choice = safe_input("\nProceed with processing? (Y/n): ").strip().lower()
            check_exit(choice)
            if choice in ("", "y", "yes"):
                return True
            if choice in ("n", "no"):
                return False
            console_print("[ERROR] Please enter 'y' to proceed, 'n' to cancel, or 'q' to quit.")

    # --- Option helpers ---

    @staticmethod
    def enhanced_select_option(
        prompt: str, options: List[Tuple[str, str]], allow_quit: bool = True
    ) -> str:
        """
        Display a prompt with detailed options and return the user's choice value.
        """
        console_print(f"\n{prompt}")
        console_print("-" * 80)
        for idx, (value, description) in enumerate(options, 1):
            console_print(f"  {idx}. {description}")
        if allow_quit:
            console_print("\n  (Type 'q' to exit the application)")
        while True:
            choice = safe_input("\nEnter your choice: ").strip()
            if allow_quit:
                check_exit(choice)
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1][0]
            console_print("[ERROR] Invalid selection. Please try again.")

    @staticmethod
    def select_option(prompt: str, options: List[str]) -> str:
        console_print(prompt)
        for idx, option in enumerate(options, 1):
            console_print(f"{idx}. {option}")
        choice = safe_input("Enter the number of your choice (or type 'q' to exit): ")
        check_exit(choice)
        return choice.strip()

    # --- Static option models ---

    @staticmethod
    def get_processing_type_options() -> List[Tuple[str, str]]:
        return [
            ("images", "Image Folders - Process collections of image files organized in folders"),
            ("pdfs", "PDF Documents - Process PDF files containing documents or scanned pages"),
        ]

    @staticmethod
    def get_method_options(processing_type: str) -> List[Tuple[str, str]]:
        if processing_type == "pdfs":
            return [
                ("native", "Native PDF extraction - Fast extraction from searchable PDFs (text-based, not scanned)"),
                ("tesseract", "Tesseract OCR - Open-source OCR suited for printed text and scanned documents"),
                ("gpt", "GPT Transcription - High-quality AI-powered transcription ideal for complex documents"),
            ]
        return [
            ("tesseract", "Tesseract OCR - Open-source OCR suited for printed text and straightforward layouts"),
            ("gpt", "GPT Transcription - High-quality AI-powered transcription ideal for handwriting and complex layouts"),
        ]

    @staticmethod
    def get_batch_options() -> List[Tuple[str, str]]:
        return [
            ("yes", "Yes - Batch processing (for large jobs, processes asynchronously, lower cost)"),
            ("no", "No - Synchronous processing (for small jobs, immediate results)"),
        ]

    @staticmethod
    def get_pdf_scope_options() -> List[Tuple[str, str]]:
        return [
            ("all", "Process all PDFs in the input directory and its subfolders"),
            ("subfolders", "Process PDFs from specific subfolders"),
            ("specific", "Process specific PDF files"),
        ]

    # --- Configuration helpers ---

    @staticmethod
    def configure_processing_type(user_config: UserConfiguration) -> None:
        user_config.processing_type = UserPrompt.enhanced_select_option(
            "What type of documents would you like to process?",
            UserPrompt.get_processing_type_options(),
        )

    @staticmethod
    def configure_transcription_method(user_config: UserConfiguration) -> None:
        user_config.transcription_method = UserPrompt.enhanced_select_option(
            f"Which transcription method would you like to use for {user_config.processing_type}?",
            UserPrompt.get_method_options(user_config.processing_type),
        )

    @staticmethod
    def configure_batch_processing(user_config: UserConfiguration) -> None:
        if user_config.transcription_method == "gpt":
            batch_choice = UserPrompt.enhanced_select_option(
                "Would you like to use batch processing for GPT transcription?",
                UserPrompt.get_batch_options(),
            )
            user_config.use_batch_processing = batch_choice == "yes"

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                console_print(
                    "[ERROR] OPENAI_API_KEY is required for GPT transcription. Please set it and try again."
                )
                sys.exit(1)

            # After enabling GPT, select a transcription schema
            UserPrompt.configure_schema_selection(user_config)

    @staticmethod
    def configure_schema_selection(user_config: UserConfiguration) -> None:
        """
        Let the user pick a transcription schema by its `name` from schemas/.
        Stores both the selected name and the Path.
        """
        if user_config.transcription_method != "gpt":
            return

        options = list_schema_options()
        if not options:
            # Fallback to default markdown schema path
            default_schema = (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json").resolve()
            user_config.selected_schema_name = "markdown_transcription_schema"
            user_config.selected_schema_path = default_schema
            console_print(
                f"[INFO] No schemas discovered under schemas/. Falling back to default: {default_schema.name}"
            )
            return

        if len(options) == 1:
            only_name, only_path = options[0]
            user_config.selected_schema_name = only_name
            user_config.selected_schema_path = only_path
            console_print(f"[INFO] Using the only available schema: {only_name} ({only_path.name})")
            return

        # Build UI choices: (value, description)
        value_to_path: Dict[str, Path] = {name: path for name, path in options}
        choices: List[Tuple[str, str]] = [
            (name, f"{name} ({path.name})") for name, path in options
        ]
        selected_name = UserPrompt.enhanced_select_option(
            "Which transcription schema would you like to use?",
            choices,
        )
        user_config.selected_schema_name = selected_name
        user_config.selected_schema_path = value_to_path[selected_name]

    # --- Selection flows ---

    @staticmethod
    def select_pdfs_or_folders(
        directory: Path, selection_type: str = "folders", process_subfolders: bool = False
    ) -> List[Path]:
        if selection_type == "folders":
            items = [d for d in directory.iterdir() if d.is_dir()]
            item_description = "folders"
        else:
            items = (
                list(directory.rglob("*.pdf")) if process_subfolders else [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
            )
            item_description = "PDF files"

        if not items:
            console_print(f"[INFO] No {item_description} found in {directory}")
            return []

        console_print(f"\nAvailable {item_description} ({len(items)}):")
        console_print("-" * 80)
        for idx, item in enumerate(items, 1):
            name = item.name
            if len(name) > 70:
                name = name[:67] + "..."
            console_print(f"  {idx}. {name}")

        console_print("\nSelection options:")
        console_print("  - Enter numbers separated by commas (e.g., '1,3,5')")
        console_print("  - Enter 'all' to select all items")
        console_print("  - Type 'q' to exit")

        while True:
            selection = safe_input("\nMake your selection: ").strip().lower()
            check_exit(selection)
            if selection == "all":
                return items
            try:
                indices = [int(i.strip()) - 1 for i in selection.split(",") if i.strip().isdigit()]
                selected_items = [items[i] for i in indices if 0 <= i < len(items)]
                if not selected_items:
                    console_print("[ERROR] No valid items selected. Please try again.")
                    continue
                return selected_items
            except (ValueError, IndexError):
                console_print("[ERROR] Invalid selection format. Please try again.")

    @staticmethod
    def select_files(directory: Path, extension: str) -> List[Path]:
        files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == extension.lower()]
        if not files:
            console_print(f"No files with extension '{extension}' found in {directory}.")
            logger.info(f"No files with extension '{extension}' found in {directory}.")
            return []

        console_print(f"Files with extension '{extension}' found in {directory}:")
        for idx, file in enumerate(files, 1):
            console_print(f"{idx}. {file.name}")

        selected = safe_input(
            "Enter the numbers of the files to select, separated by commas (or type 'q' to exit): "
        ).strip()
        check_exit(selected)
        try:
            indices = [int(i.strip()) - 1 for i in selected.split(",") if i.strip().isdigit()]
            selected_files = [files[i] for i in indices if 0 <= i < len(files)]
            if not selected_files:
                console_print("No valid files selected.")
                logger.info("No valid files selected by the user.")
            return selected_files
        except ValueError:
            console_print("Invalid input. Please enter numbers separated by commas.")
            logger.error("User entered invalid file selection input.")
            return []

    @staticmethod
    def select_pdfs_workflow(user_config: UserConfiguration, pdf_input_dir: Path) -> None:
        pdf_scope = UserPrompt.enhanced_select_option(
            "How would you like to select PDFs for processing?",
            UserPrompt.get_pdf_scope_options(),
        )

        if pdf_scope == "all":
            all_pdfs = list(pdf_input_dir.rglob("*.pdf"))
            if not all_pdfs:
                console_print(f"[ERROR] No PDF files found in {pdf_input_dir} or its subfolders.")
                return
            console_print(f"[INFO] Found {len(all_pdfs)} PDF file(s) to process.")
            user_config.selected_items = all_pdfs
            user_config.process_all = True
            return

        if pdf_scope == "subfolders":
            subfolders = [d for d in pdf_input_dir.iterdir() if d.is_dir()]
            if not subfolders:
                console_print(f"[ERROR] No subfolders found in {pdf_input_dir}")
                return
            console_print(f"[INFO] Found {len(subfolders)} subfolder(s) in the input directory.")
            user_config.selected_items = UserPrompt.select_pdfs_or_folders(pdf_input_dir, "folders")
            if not user_config.selected_items:
                console_print("[INFO] No folders selected. Exiting.")
                return
            pdf_files: List[Path] = []
            for folder in user_config.selected_items:
                folder_pdfs = list(folder.glob("*.pdf"))
                if not folder_pdfs:
                    console_print(f"[WARN] No PDF files found in {folder.name}.")
                else:
                    pdf_files.extend(folder_pdfs)
            if not pdf_files:
                console_print("[ERROR] No PDF files found in the selected folders.")
                return
            console_print(f"[INFO] Found {len(pdf_files)} PDF file(s) in the selected folders.")
            user_config.selected_items = pdf_files
            return

        # specific PDFs
        all_pdfs = list(pdf_input_dir.glob("*.pdf"))
        if not all_pdfs:
            console_print(f"[ERROR] No PDF files found directly in {pdf_input_dir}")
            return
        user_config.selected_items = UserPrompt.select_pdfs_or_folders(pdf_input_dir, "pdfs")
        if not user_config.selected_items:
            console_print("[INFO] No PDFs selected. Exiting.")

    @staticmethod
    def select_images_workflow(user_config: UserConfiguration, image_input_dir: Path) -> None:
        from modules.image_utils import SUPPORTED_IMAGE_EXTENSIONS

        source_dir = image_input_dir
        subfolders = [f for f in source_dir.iterdir() if f.is_dir()]
        direct_images = [
            f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        if not subfolders and not direct_images:
            console_print(f"[ERROR] No image folders or images found in {source_dir}")
            console_print("[INFO] Please add image folders or images to the input directory and try again.")
            return

        if not subfolders and direct_images:
            console_print(
                f"[INFO] No subfolders found, but {len(direct_images)} images detected directly in the input directory."
            )
            process_option = UserPrompt.enhanced_select_option(
                "How would you like to proceed?",
                [("process_direct", "Process these images directly"), ("cancel", "Cancel and create proper subfolders first")],
            )
            if process_option == "process_direct":
                user_config.selected_items = [source_dir]
            else:
                console_print("[INFO] Operation cancelled. Please organize your images into subfolders and try again.")
            return

        console_print(f"[INFO] Found {len(subfolders)} image folder(s) in the input directory.")
        selection_mode = UserPrompt.enhanced_select_option(
            "Would you like to process all folders or select specific ones?",
            [("specific", "Select specific folders to process"), ("all", "Process all folders")],
        )
        if selection_mode == "all":
            user_config.selected_items = subfolders
            user_config.process_all = True
        else:
            user_config.selected_items = UserPrompt.select_pdfs_or_folders(source_dir, "folders")
            if not user_config.selected_items:
                console_print("[INFO] No folders selected. Exiting.")

    # --- Batch related displays ---

    @staticmethod
    def display_batch_summary(batches: List[Dict[str, Any]]) -> None:
        if not batches:
            console_print("No batches found.")
            return
        status_groups: Dict[str, List[Any]] = {}
        for batch in batches:
            status_val = getattr(batch, "status", None)
            if status_val is None and isinstance(batch, dict):
                status_val = batch.get("status")
            status = (status_val or "").lower()
            status_groups.setdefault(status, []).append(batch)
        console_print("\n===== Batch Summary =====")
        console_print(f"Total batches: {len(batches)}")
        for status, batch_list in sorted(status_groups.items()):
            console_print(f"{status.capitalize()}: {len(batch_list)} batch(es)")
        in_progress_statuses = {"validating", "in_progress", "finalizing"}
        for status in in_progress_statuses:
            if status in status_groups and status_groups[status]:
                console_print(f"\n----- {status.capitalize()} Batches -----")
                for batch in status_groups[status]:
                    batch_id = getattr(batch, "id", None)
                    batch_status = getattr(batch, "status", None)
                    if isinstance(batch, dict):
                        batch_id = batch.get("id", batch_id)
                        batch_status = batch.get("status", batch_status)
                    console_print(f"  Batch ID: {batch_id} | Status: {batch_status}")

    @staticmethod
    def _format_page_image(page_number: Optional[int], image_name: str) -> str:
        if page_number is None:
            return f"({image_name})"
        return f"page {page_number} ({image_name})"

    @staticmethod
    def print_transcription_item_error(
        image_name: str,
        page_number: Optional[int] = None,
        status_code: Optional[int] = None,
        err_code: Optional[str] = None,
        err_message: Optional[str] = None,
    ) -> None:
        label = UserPrompt._format_page_image(page_number, image_name)
        parts: List[str] = []
        if status_code is not None:
            parts.append(f"status={status_code}")
        if err_code:
            parts.append(f"code={err_code}")
        if err_message:
            parts.append(f"message={err_message}")
        detail = " ".join(parts)
        console_print(f"[ERROR] {label} failed in batch" + (f": {detail}" if detail else ""))

    @staticmethod
    def print_transcription_not_possible(image_name: str, page_number: Optional[int] = None) -> None:
        label = UserPrompt._format_page_image(page_number, image_name)
        console_print(f"[WARN] Model reported transcription not possible for {label}.")

    @staticmethod
    def print_no_transcribable_text(image_name: str, page_number: Optional[int] = None) -> None:
        label = UserPrompt._format_page_image(page_number, image_name)
        console_print(f"[INFO] No transcribable text detected for {label}.")

    @staticmethod
    def display_page_error_summary(error_entries: List[Dict[str, Any]]) -> None:
        if not error_entries:
            return
        console_print(f"[WARN] {len(error_entries)} page(s) failed during batch processing:")
        for e in error_entries:
            img = (e.get("image_info", {}) or {}).get("image_name") or e.get("custom_id", "[unknown image]")
            page = (e.get("image_info", {}) or {}).get("page_number")
            det = (e.get("error_details", {}) or {})
            status = det.get("status_code")
            code = det.get("code")
            msg = det.get("message")
            label = UserPrompt._format_page_image(page, img)
            parts: List[str] = []
            if status is not None:
                parts.append(f"status={status}")
            if code:
                parts.append(f"code={code}")
            if msg:
                parts.append(f"message={msg}")
            console_print("  - " + label + (": " + " ".join(parts) if parts else ""))

    @staticmethod
    def display_transcription_not_possible_summary(count: int) -> None:
        if count > 0:
            console_print(f"[INFO] {count} page(s) reported 'transcription not possible' by the model.")

    @staticmethod
    def display_batch_processing_progress(
        temp_file: Path, batch_ids: List[str], completed_count: int, missing_count: int
    ) -> None:
        console_print(f"\n----- Processing File: {temp_file.name} -----")
        console_print(f"Found {len(batch_ids)} batch ID(s)")
        if completed_count == len(batch_ids):
            console_print("[SUCCESS] All batches completed!")
        else:
            in_progress = len(batch_ids) - completed_count - missing_count
            console_print(
                f"Completed: {completed_count} | Pending: {in_progress} | Missing: {missing_count}"
            )
            if missing_count > 0:
                console_print(f"[WARN] {missing_count} batch ID(s) were not found in the API response")
            if completed_count < len(batch_ids) - missing_count:
                console_print(f"[INFO] Some batches are still processing. Try again later.")

    @staticmethod
    def display_batch_cancellation_results(
        cancelled_batches: List[Tuple[str, str, bool]], skipped_batches: List[Tuple[str, str]]
    ) -> None:
        success_count = sum(1 for _, _, success in cancelled_batches if success)
        fail_count = len(cancelled_batches) - success_count
        console_print("\n===== Cancellation Summary =====")
        console_print(f"Total batches found: {len(cancelled_batches) + len(skipped_batches)}")
        console_print(f"Skipped (terminal status): {len(skipped_batches)}")
        console_print(f"Attempted to cancel: {len(cancelled_batches)}")
        console_print(f"Successfully cancelled: {success_count}")
        if fail_count > 0:
            console_print(f"Failed to cancel: {fail_count}")
            console_print("\n----- Failed Cancellations -----")
            for batch_id, status, success in cancelled_batches:
                if not success:
                    console_print(f"  Batch {batch_id} (status: '{status}')")
        if success_count > 0:
            console_print("\n----- Successfully Cancelled -----")
            for batch_id, status, success in cancelled_batches:
                if success:
                    console_print(f"  Batch {batch_id} (previous status: '{status}')")
