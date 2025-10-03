"""Enhanced workflow UI components using the new prompt system.

This module provides workflow-specific UI components that use the improved
prompting utilities for a consistent and navigable user experience.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.ui.prompts import (
    NavigationAction,
    PromptResult,
    prompt_select,
    prompt_yes_no,
    prompt_multiselect,
    print_header,
    print_separator,
    print_info,
    print_success,
    print_warning,
    print_error,
    ui_print,
    PromptStyle,
)
from modules.ui.core import UserConfiguration
from modules.infra.logger import setup_logger
from modules.llm.schema_utils import list_schema_options
from modules.config.config_loader import PROJECT_ROOT

logger = setup_logger(__name__)


class WorkflowUI:
    """Enhanced workflow UI with navigation support."""
    
    @staticmethod
    def display_welcome() -> None:
        """Display welcome banner."""
        print_header(
            "CHRONO TRANSCRIBER",
            "Historical Document Digitization Tool"
        )
        ui_print("  Transform historical documents into searchable text using", PromptStyle.INFO)
        ui_print("  state-of-the-art transcription methods tailored to your needs.\n", PromptStyle.INFO)
    
    @staticmethod
    def get_processing_type_options() -> List[Tuple[str, str]]:
        """Get processing type options."""
        return [
            ("images", "Image Folders — Process collections of images organized in folders"),
            ("pdfs", "PDF Documents — Process PDF files or scanned documents"),
        ]
    
    @staticmethod
    def get_method_options(processing_type: str) -> List[Tuple[str, str]]:
        """Get transcription method options based on processing type."""
        if processing_type == "pdfs":
            return [
                ("native", "Native PDF Extraction — Fast extraction from searchable PDFs"),
                ("tesseract", "Tesseract OCR — Open-source OCR for printed text"),
                ("gpt", "GPT Transcription — AI-powered transcription for complex documents"),
            ]
        return [
            ("tesseract", "Tesseract OCR — Open-source OCR for printed text"),
            ("gpt", "GPT Transcription — AI-powered transcription for handwriting & complex layouts"),
        ]
    
    @staticmethod
    def get_batch_options() -> List[Tuple[str, str]]:
        """Get batch processing options."""
        return [
            ("yes", "Batch Processing — Asynchronous processing for large jobs (lower cost)"),
            ("no", "Synchronous Processing — Immediate results for smaller jobs"),
        ]
    
    @staticmethod
    def configure_processing_type(config: UserConfiguration) -> bool:
        """Configure processing type with navigation.
        
        Returns:
            True if configured successfully, False if user wants to go back/quit
        """
        result = prompt_select(
            "What type of documents would you like to process?",
            WorkflowUI.get_processing_type_options(),
            allow_back=False
        )
        
        if result.action == NavigationAction.CONTINUE:
            config.processing_type = result.value
            logger.info(f"User selected processing type: {result.value}")
            return True
        
        return False
    
    @staticmethod
    def configure_transcription_method(config: UserConfiguration) -> bool:
        """Configure transcription method with navigation.
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        result = prompt_select(
            f"Which transcription method would you like to use for {config.processing_type}?",
            WorkflowUI.get_method_options(config.processing_type),
            allow_back=True
        )
        
        if result.action == NavigationAction.CONTINUE:
            config.transcription_method = result.value
            logger.info(f"User selected transcription method: {result.value}")
            return True
        elif result.action == NavigationAction.BACK:
            return False
        
        return False
    
    @staticmethod
    def configure_batch_processing(config: UserConfiguration) -> bool:
        """Configure batch processing with navigation.
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        if config.transcription_method != "gpt":
            return True
        
        # Check API key first
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_error("OPENAI_API_KEY environment variable is required for GPT transcription.")
            print_info("Please set your API key and try again.")
            sys.exit(1)
        
        # Ask about batch processing
        result = prompt_select(
            "Would you like to use batch processing for GPT transcription?",
            WorkflowUI.get_batch_options(),
            allow_back=True
        )
        
        if result.action == NavigationAction.BACK:
            return False
        
        config.use_batch_processing = (result.value == "yes")
        logger.info(f"User selected batch processing: {config.use_batch_processing}")
        
        # Configure schema
        if not WorkflowUI.configure_schema_selection(config):
            return False
        
        # Configure additional context
        if not WorkflowUI.configure_additional_context(config):
            return False
        
        return True
    
    @staticmethod
    def configure_schema_selection(config: UserConfiguration) -> bool:
        """Configure schema selection with navigation.
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        if config.transcription_method != "gpt":
            return True
        
        options = list_schema_options()
        
        if not options:
            # Fallback to default
            default_schema = (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json").resolve()
            config.selected_schema_name = "markdown_transcription_schema"
            config.selected_schema_path = default_schema
            print_info(f"Using default schema: {default_schema.name}")
            return True
        
        if len(options) == 1:
            # Only one option
            only_name, only_path = options[0]
            config.selected_schema_name = only_name
            config.selected_schema_path = only_path
            print_info(f"Using schema: {only_name} ({only_path.name})")
            return True
        
        # Multiple options - let user choose
        value_to_path: Dict[str, Path] = {name: path for name, path in options}
        choices: List[Tuple[str, str]] = [
            (name, f"{name} — {path.name}") for name, path in options
        ]
        
        result = prompt_select(
            "Which transcription schema would you like to use?",
            choices,
            allow_back=True
        )
        
        if result.action == NavigationAction.CONTINUE:
            config.selected_schema_name = result.value
            config.selected_schema_path = value_to_path[result.value]
            logger.info(f"User selected schema: {result.value}")
            return True
        
        return False
    
    @staticmethod
    def configure_additional_context(config: UserConfiguration) -> bool:
        """Configure additional context with navigation.
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        if config.transcription_method != "gpt":
            return True
        
        context_file = PROJECT_ROOT / "additional_context" / "additional_context.txt"
        
        if not context_file.exists():
            print_info(f"No additional context file found. Skipping.")
            config.additional_context_path = None
            return True
        
        result = prompt_select(
            "Would you like to use additional context to guide transcription?",
            [
                ("yes", "Yes — Use domain-specific guidance from additional_context.txt"),
                ("no", "No — Proceed without additional context"),
            ],
            allow_back=True
        )
        
        if result.action == NavigationAction.BACK:
            return False
        
        if result.value == "yes":
            config.additional_context_path = context_file
            print_info(f"Additional context loaded from: {context_file.name}")
        else:
            config.additional_context_path = None
            print_info("Proceeding without additional context.")
        
        logger.info(f"Additional context: {config.additional_context_path}")
        return True
    
    @staticmethod
    def select_items_for_processing(config: UserConfiguration, base_dir: Path) -> bool:
        """Select items for processing based on configuration.
        
        Returns:
            True if items selected successfully, False if user wants to go back
        """
        print_header(
            f"DOCUMENT SELECTION — {config.processing_type.upper()}",
            "Choose which items to process"
        )
        
        # Display current configuration
        ui_print("  Current Settings:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        ui_print(f"    • Document type: {config.processing_type.capitalize()}", PromptStyle.INFO)
        ui_print(f"    • Transcription method: {config.transcription_method.capitalize()}", PromptStyle.INFO)
        if config.transcription_method == "gpt":
            mode = "Batch (asynchronous)" if config.use_batch_processing else "Synchronous"
            ui_print(f"    • Processing mode: {mode}", PromptStyle.INFO)
            if config.selected_schema_name:
                ui_print(f"    • Schema: {config.selected_schema_name}", PromptStyle.INFO)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        ui_print("")
        
        if config.processing_type == "images":
            return WorkflowUI._select_image_folders(config, base_dir)
        else:
            return WorkflowUI._select_pdf_files(config, base_dir)
    
    @staticmethod
    def _select_image_folders(config: UserConfiguration, image_dir: Path) -> bool:
        """Select image folders for processing."""
        from modules.processing.image_utils import SUPPORTED_IMAGE_EXTENSIONS
        
        subfolders = [f for f in image_dir.iterdir() if f.is_dir()]
        direct_images = [
            f for f in image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        if not subfolders and not direct_images:
            print_error(f"No image folders or images found in {image_dir}")
            print_info("Please add image folders and try again.")
            sys.exit(1)
        
        if not subfolders and direct_images:
            print_info(f"Found {len(direct_images)} images directly in the input directory.")
            result = prompt_select(
                "How would you like to proceed?",
                [
                    ("process", "Process these images directly"),
                    ("cancel", "Cancel — organize images into subfolders first")
                ],
                allow_back=True
            )
            
            if result.action == NavigationAction.BACK:
                return False
            
            if result.value == "process":
                config.selected_items = [image_dir]
                return True
            else:
                print_info("Please organize your images into subfolders and try again.")
                sys.exit(0)
        
        # Select folders
        print_info(f"Found {len(subfolders)} image folder(s) in the input directory.")
        
        result = prompt_select(
            "Would you like to process all folders or select specific ones?",
            [
                ("specific", "Select specific folders"),
                ("all", "Process all folders")
            ],
            allow_back=True
        )
        
        if result.action == NavigationAction.BACK:
            return False
        
        if result.value == "all":
            config.selected_items = subfolders
            config.process_all = True
            print_success(f"Selected all {len(subfolders)} folders for processing.")
            return True
        
        # Multi-select folders
        folder_items = [(str(f), f.name) for f in subfolders]
        selection_result = prompt_multiselect(
            f"Select image folders to process ({len(subfolders)} available):",
            folder_items,
            allow_all=True,
            allow_back=True
        )
        
        if selection_result.action == NavigationAction.BACK:
            return False
        
        selected_paths = [Path(p) for p in selection_result.value]
        config.selected_items = selected_paths
        print_success(f"Selected {len(selected_paths)} folder(s) for processing.")
        return True
    
    @staticmethod
    def _select_pdf_files(config: UserConfiguration, pdf_dir: Path) -> bool:
        """Select PDF files for processing."""
        result = prompt_select(
            "How would you like to select PDFs for processing?",
            [
                ("all", "Process all PDFs (including subfolders)"),
                ("subfolders", "Process PDFs from specific subfolders"),
                ("specific", "Select specific PDF files")
            ],
            allow_back=True
        )
        
        if result.action == NavigationAction.BACK:
            return False
        
        if result.value == "all":
            all_pdfs = list(pdf_dir.rglob("*.pdf"))
            if not all_pdfs:
                print_error(f"No PDF files found in {pdf_dir} or its subfolders.")
                sys.exit(1)
            config.selected_items = all_pdfs
            config.process_all = True
            print_success(f"Selected all {len(all_pdfs)} PDF(s) for processing.")
            return True
        
        if result.value == "subfolders":
            subfolders = [d for d in pdf_dir.iterdir() if d.is_dir()]
            if not subfolders:
                print_error(f"No subfolders found in {pdf_dir}")
                sys.exit(1)
            
            folder_items = [(str(f), f.name) for f in subfolders]
            selection_result = prompt_multiselect(
                f"Select subfolders containing PDFs ({len(subfolders)} available):",
                folder_items,
                allow_all=True,
                allow_back=True
            )
            
            if selection_result.action == NavigationAction.BACK:
                return False
            
            selected_folders = [Path(p) for p in selection_result.value]
            pdf_files: List[Path] = []
            for folder in selected_folders:
                folder_pdfs = list(folder.glob("*.pdf"))
                if not folder_pdfs:
                    print_warning(f"No PDFs found in {folder.name}")
                else:
                    pdf_files.extend(folder_pdfs)
            
            if not pdf_files:
                print_error("No PDF files found in selected folders.")
                sys.exit(1)
            
            config.selected_items = pdf_files
            print_success(f"Selected {len(pdf_files)} PDF(s) from {len(selected_folders)} folder(s).")
            return True
        
        # Specific PDFs
        all_pdfs = list(pdf_dir.glob("*.pdf"))
        if not all_pdfs:
            print_error(f"No PDF files found in {pdf_dir}")
            sys.exit(1)
        
        pdf_items = [(str(f), f.name) for f in all_pdfs]
        selection_result = prompt_multiselect(
            f"Select PDF files to process ({len(all_pdfs)} available):",
            pdf_items,
            allow_all=True,
            allow_back=True
        )
        
        if selection_result.action == NavigationAction.BACK:
            return False
        
        selected_pdfs = [Path(p) for p in selection_result.value]
        config.selected_items = selected_pdfs
        print_success(f"Selected {len(selected_pdfs)} PDF(s) for processing.")
        return True
    
    @staticmethod
    def display_processing_summary(config: UserConfiguration) -> bool:
        """Display processing summary and confirm.
        
        Returns:
            True if user confirms, False if they want to go back
        """
        print_header("PROCESSING SUMMARY", "Review your selections")
        
        item_type = "image folder(s)" if config.processing_type == "images" else "PDF file(s)"
        ui_print(f"  Ready to process ", PromptStyle.INFO, end="")
        ui_print(f"{len(config.selected_items)}", PromptStyle.HIGHLIGHT, end="")
        ui_print(f" {item_type}\n", PromptStyle.INFO)
        
        ui_print("  Configuration:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        ui_print(f"    • Document type: {config.processing_type.capitalize()}", PromptStyle.INFO)
        ui_print(f"    • Transcription method: {config.transcription_method.capitalize()}", PromptStyle.INFO)
        
        if config.transcription_method == "gpt":
            mode = "Batch (asynchronous)" if config.use_batch_processing else "Synchronous"
            ui_print(f"    • Processing mode: {mode}", PromptStyle.INFO)
            if config.selected_schema_name:
                ui_print(f"    • Schema: {config.selected_schema_name}", PromptStyle.INFO)
        
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        ui_print("\n  Selected items (first 5 shown):", PromptStyle.INFO)
        for i, item in enumerate(config.selected_items[:5], 1):
            ui_print(f"    {i}. {item.name}", PromptStyle.DIM)
        
        if len(config.selected_items) > 5:
            ui_print(f"    ... and {len(config.selected_items) - 5} more", PromptStyle.DIM)
        
        ui_print("")
        
        result = prompt_yes_no(
            "Proceed with processing?",
            default=True,
            allow_back=True
        )
        
        if result.action == NavigationAction.CONTINUE:
            return result.value
        
        return False
    
    @staticmethod
    def display_completion_summary(config: UserConfiguration) -> None:
        """Display completion summary."""
        print_header("PROCESSING COMPLETE", "")
        
        if config.use_batch_processing and config.transcription_method == "gpt":
            print_success("Batch processing jobs have been submitted!")
            ui_print("")
            ui_print("  Next steps:", PromptStyle.HIGHLIGHT)
            ui_print("    • Check batch status: ", PromptStyle.DIM, end="")
            ui_print("python main/check_batches.py", PromptStyle.INFO)
            ui_print("    • Cancel pending batches: ", PromptStyle.DIM, end="")
            ui_print("python main/cancel_batches.py", PromptStyle.INFO)
        else:
            print_success("All selected items have been processed.")
        
        ui_print("\n  Thank you for using ChronoTranscriber!\n", PromptStyle.HIGHLIGHT)
