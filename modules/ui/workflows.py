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
    prompt_text,
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
            ("auto", "Auto Mode — Automatically detect and process mixed file types"),
            ("images", "Image Folders — Process collections of images organized in folders"),
            ("pdfs", "PDF Documents — Process PDF files or scanned documents"),
            ("epubs", "EPUB Documents — Extract text directly from EPUB ebooks"),
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
        if processing_type == "epubs":
            return [
                ("native", "Native EPUB Extraction — Extract XHTML text from EPUB chapters"),
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
        
        Args:
            config: UserConfiguration object
        
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
            WorkflowUI.get_method_options(config.processing_type or "pdfs"),
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
        
        # Check API key based on configured provider
        import os
        from modules.config.service import get_config_service
        
        config_service = get_config_service()
        model_config = config_service.get_model_config()
        provider = model_config.get("transcription_model", {}).get("provider", "openai")
        
        # Map provider to environment variable
        provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        
        env_var = provider_env_vars.get(provider, "OPENAI_API_KEY")
        api_key = os.getenv(env_var)
        
        if not api_key:
            print_error(f"{env_var} environment variable is required for {provider.upper()} transcription.")
            print_info(f"Please set your API key and try again.")
            print_info(f"  (Configured provider: {provider} in model_config.yaml)")
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
        
        Context resolution follows a hierarchy (most specific wins):
        1. File-specific: {input_stem}_transcr_context.txt next to input file
        2. Folder-specific: {parent_folder}_transcr_context.txt next to parent folder
        3. General fallback: context/transcr_context.txt in project root
        
        The user can choose to:
        - Use hierarchical resolution (auto-detect file/folder-specific context)
        - Use the global additional_context.txt (overrides hierarchy)
        - Use no context at all
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        if config.transcription_method != "gpt":
            return True
        
        context_file = PROJECT_ROOT / "additional_context" / "additional_context.txt"
        global_context_exists = context_file.exists()
        
        # Build options based on available context sources
        options = [
            ("hierarchical", "Auto — Use file/folder-specific context if available (recommended)"),
        ]
        if global_context_exists:
            options.append(
                ("global", f"Global — Use {context_file.name} for all files (overrides file-specific)")
            )
        options.append(
            ("none", "None — Proceed without any additional context")
        )
        
        result = prompt_select(
            "How should additional context be resolved?",
            options,
            allow_back=True
        )
        
        if result.action == NavigationAction.BACK:
            return False
        
        if result.value == "hierarchical":
            config.additional_context_path = None  # Triggers hierarchical resolution at runtime
            config.use_hierarchical_context = True
            print_info("Using hierarchical context resolution (file > folder > general fallback).")
        elif result.value == "global":
            config.additional_context_path = context_file
            config.use_hierarchical_context = False
            print_info(f"Using global context from: {context_file.name}")
        else:
            config.additional_context_path = None
            config.use_hierarchical_context = False
            print_info("Proceeding without additional context.")
        
        logger.info(f"Additional context mode: {result.value}, path: {config.additional_context_path}")
        return True
    
    @staticmethod
    def configure_auto_mode_schema(config: UserConfiguration) -> bool:
        """Configure schema and context for auto mode when GPT files are detected.
        
        This method checks if any GPT transcription will occur in auto mode
        and prompts for schema selection if so.
        
        Args:
            config: UserConfiguration object with auto_decisions populated
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        # Check if any decisions will use GPT
        decisions = config.auto_decisions or []
        gpt_decisions = [d for d in decisions if d.method == "gpt"]
        
        if not gpt_decisions:
            # No GPT processing, skip schema selection
            return True
        
        print_header("GPT TRANSCRIPTION SETTINGS", f"{len(gpt_decisions)} file(s) will use GPT transcription")
        
        # configure_schema_selection and configure_additional_context gate on
        # config.transcription_method == "gpt".  In auto mode the field is
        # None, so temporarily set it so the prompts actually appear.
        prev_method = config.transcription_method
        config.transcription_method = "gpt"
        
        # Configure schema
        if not WorkflowUI.configure_schema_selection(config):
            config.transcription_method = prev_method
            return False
        
        # Configure additional context
        if not WorkflowUI.configure_additional_context(config):
            config.transcription_method = prev_method
            return False
        
        config.transcription_method = prev_method
        return True

    @staticmethod
    def configure_page_range(config: UserConfiguration) -> bool:
        """Optionally configure page-range filtering.

        Args:
            config: UserConfiguration object

        Returns:
            True if configured (or skipped), False if user wants to go back
        """
        result = prompt_select(
            "Would you like to limit which pages/sections are processed?",
            [
                ("no", "No — Process all pages (default)"),
                ("yes", "Yes — Specify a page range"),
            ],
            allow_back=True,
        )

        if result.action == NavigationAction.BACK:
            return False

        if result.value == "no":
            config.page_range = None
            return True

        # Prompt for the page-range string
        from modules.core.page_range import parse_page_range

        def _validate_range(value: str) -> bool:
            try:
                parse_page_range(value)
                return True
            except ValueError:
                return False

        text_result = prompt_text(
            "Enter page range (e.g. '5', 'first:5', 'last:5', '3-7', '1,3,5-8'):",
            allow_back=True,
            validator=_validate_range,
            error_message=(
                "Invalid page range. Use formats like '5', 'first:5', 'last:5', "
                "'3-7', '3-', '-7', or '1,3,5-8'."
            ),
        )

        if text_result.action == NavigationAction.BACK:
            return False

        config.page_range = parse_page_range(text_result.value)
        print_info(f"Page range set: {config.page_range.describe()}")
        logger.info(f"User configured page range: {config.page_range.describe()}")
        return True

    @staticmethod
    def configure_resume_mode(config: UserConfiguration) -> bool:
        """Configure resume/overwrite behavior for existing output.

        Args:
            config: UserConfiguration object

        Returns:
            True if configured successfully, False if user wants to go back
        """
        result = prompt_select(
            "How should existing output files be handled?",
            [
                ("skip", "Skip — Resume processing, skip files with existing output (default)"),
                ("overwrite", "Overwrite — Reprocess all files, overwriting existing output"),
            ],
            allow_back=True,
        )

        if result.action == NavigationAction.BACK:
            return False

        config.resume_mode = result.value or "skip"
        if config.resume_mode == "overwrite":
            print_info("Overwrite mode: all files will be reprocessed.")
        else:
            print_info("Resume mode: files with existing output will be skipped.")
        logger.info(f"User selected resume mode: {config.resume_mode}")
        return True

    @staticmethod
    def select_items_for_processing(
        config: UserConfiguration,
        base_dir: Path,
        paths_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Select items for processing based on configuration.
        
        Returns:
            True if items selected successfully, False if user wants to go back
        """
        # Handle auto mode separately
        if config.processing_type == "auto":
            return WorkflowUI._configure_auto_mode(config, base_dir, paths_config)
        
        print_header(
            f"DOCUMENT SELECTION — {(config.processing_type or 'UNKNOWN').upper()}",
            "Choose which items to process"
        )
        
        # Display current configuration
        ui_print("  Current Settings:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        ui_print(f"    • Document type: {(config.processing_type or 'unknown').capitalize()}", PromptStyle.INFO)
        ui_print(f"    • Transcription method: {(config.transcription_method or 'unknown').capitalize()}", PromptStyle.INFO)
        if config.transcription_method == "gpt":
            mode = "Batch (asynchronous)" if config.use_batch_processing else "Synchronous"
            ui_print(f"    • Processing mode: {mode}", PromptStyle.INFO)
            if config.selected_schema_name:
                ui_print(f"    • Schema: {config.selected_schema_name}", PromptStyle.INFO)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        ui_print("")
        
        if config.processing_type == "images":
            return WorkflowUI._select_image_folders(config, base_dir)
        if config.processing_type == "pdfs":
            return WorkflowUI._select_pdf_files(config, base_dir)
        return WorkflowUI._select_epub_files(config, base_dir)

    @staticmethod
    def _select_epub_files(config: UserConfiguration, epub_dir: Path) -> bool:
        """Select EPUB files for processing."""
        result = prompt_select(
            "How would you like to select EPUBs for processing?",
            [
                ("all", "Process all EPUBs (including subfolders)"),
                ("specific", "Select specific EPUB files"),
            ],
            allow_back=True
        )

        if result.action == NavigationAction.BACK:
            return False

        if result.value == "all":
            all_epubs = list(epub_dir.rglob("*.epub"))
            if not all_epubs:
                print_error(f"No EPUB files found in {epub_dir} or its subfolders.")
                sys.exit(1)
            config.selected_items = all_epubs
            config.process_all = True
            print_success(f"Selected all {len(all_epubs)} EPUB(s) for processing.")
            return True

        # Select specific EPUB files
        epub_files = list(epub_dir.glob("*.epub"))
        if not epub_files:
            print_error(f"No EPUB files found in {epub_dir}.")
            return False

        file_items = [(str(f), f.name) for f in epub_files]
        selection_result = prompt_multiselect(
            f"Select EPUB files to process ({len(epub_files)} available):",
            file_items,
            allow_all=True,
            allow_back=True
        )

        if selection_result.action == NavigationAction.BACK:
            return False

        selected_paths = [Path(p) for p in selection_result.value]
        if not selected_paths:
            print_warning("No EPUB files selected. Nothing to process.")
            return False

        config.selected_items = selected_paths
        config.process_all = len(selected_paths) == len(epub_files)
        print_success(f"Selected {len(selected_paths)} EPUB file(s) for processing.")
        return True
    
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
    def _configure_auto_mode(
        config: UserConfiguration,
        base_dir: Path,
        paths_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Configure auto mode processing.
        
        Args:
            config: UserConfiguration object
            base_dir: Base directory to scan (from Auto paths config)
        
        Returns:
            True if configured successfully, False if user wants to go back
        """
        from modules.core.auto_selector import AutoSelector
        from modules.config.service import get_config_service

        print_header("AUTO MODE CONFIGURATION", "Automatic file detection and method selection")

        selector = config.auto_selector
        if selector is None:
            if paths_config is None:
                paths_config = get_config_service().get_paths_config()
            selector = AutoSelector(paths_config or {})
            config.auto_selector = selector
        
        # Scan directory and create decisions
        print_info(f"Scanning directory: {base_dir}")
        decisions = selector.create_decisions(base_dir)

        if not decisions:
            print_error(f"No processable files found in {base_dir}")
            print_info("Please add files and try again.")
            return False
        
        # Display decision summary
        selector.print_decision_summary(decisions)
        
        # Confirm with user
        result = prompt_yes_no(
            f"Proceed with processing {len(decisions)} file(s) using auto mode?",
            default=True,
            allow_back=True
        )
        
        if result.action == NavigationAction.BACK:
            return False
        
        if result.action == NavigationAction.CONTINUE and result.value:
            # Store decisions and output directory
            config.auto_decisions = decisions
            config.selected_items = [base_dir]  # Will be used as output dir
            return True
        
        return False
    
    @staticmethod
    def display_processing_summary(config: UserConfiguration) -> bool:
        """Display processing summary and confirm.
        
        Returns:
            True if user confirms, False if they want to go back
        """
        # Load configs for display
        from modules.config.service import get_config_service
        config_service = get_config_service()
        model_config = config_service.get_model_config()
        paths_config = config_service.get_paths_config()
        concurrency_config = config_service.get_concurrency_config()
        
        is_auto = config.processing_type == "auto"
        decisions = config.auto_decisions or []
        
        # Determine whether any GPT processing will occur
        if is_auto:
            has_gpt = any(d.method == "gpt" for d in decisions)
        else:
            has_gpt = config.transcription_method == "gpt"
        
        print_header("PROCESSING SUMMARY", "Review your selections before processing")
        
        # === Item count ===
        if is_auto:
            ui_print(f"  Ready to process ", PromptStyle.INFO, end="")
            ui_print(f"{len(decisions)}", PromptStyle.HIGHLIGHT, end="")
            ui_print(f" file(s) in auto mode\n", PromptStyle.INFO)
        else:
            if config.processing_type == "images":
                item_type = "image folder(s)"
            elif config.processing_type == "pdfs":
                item_type = "PDF file(s)"
            elif config.processing_type == "epubs":
                item_type = "EPUB file(s)"
            else:
                item_type = "file(s)"
            ui_print(f"  Ready to process ", PromptStyle.INFO, end="")
            ui_print(f"{len(config.selected_items or [])}", PromptStyle.HIGHLIGHT, end="")
            ui_print(f" {item_type}\n", PromptStyle.INFO)
        
        # === Processing Configuration ===
        ui_print("  Processing Configuration:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        if is_auto:
            ui_print(f"    • Document type: Auto (mixed)", PromptStyle.INFO)
            # Show method breakdown
            from collections import Counter
            method_counts = Counter(d.method for d in decisions)
            for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
                ui_print(f"    • {method.upper()}: {count} file(s)", PromptStyle.INFO)
            ui_print(f"    • Processing mode: Synchronous", PromptStyle.INFO)
        else:
            ui_print(f"    • Document type: {(config.processing_type or 'unknown').capitalize()}", PromptStyle.INFO)
            ui_print(f"    • Transcription method: {(config.transcription_method or 'unknown').upper()}", PromptStyle.INFO)
            if config.transcription_method == "gpt":
                mode = "Batch (asynchronous)" if config.use_batch_processing else "Synchronous"
                ui_print(f"    • Processing mode: {mode}", PromptStyle.INFO)
        
        # Schema and context (shown when GPT is involved)
        if has_gpt:
            if config.selected_schema_name:
                ui_print(f"    • Schema: {config.selected_schema_name}", PromptStyle.INFO)
            # Display context resolution mode accurately
            if config.additional_context_path:
                ui_print(f"    • Additional context: Global ({config.additional_context_path.name})", PromptStyle.INFO)
            elif getattr(config, 'use_hierarchical_context', False):
                ui_print(f"    • Additional context: Hierarchical (file/folder-specific)", PromptStyle.INFO)
            else:
                ui_print(f"    • Additional context: None", PromptStyle.DIM)

        # Show page range if configured
        if config.page_range is not None:
            ui_print(f"    • Page range: {config.page_range.describe()}", PromptStyle.INFO)
        
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        # === Model Configuration (when GPT is involved) ===
        if has_gpt:
            ui_print("\n  Model Configuration:", PromptStyle.HIGHLIGHT)
            print_separator(PromptStyle.LIGHT_LINE, 80)
            
            tm = model_config.get("transcription_model", {})
            provider = tm.get("provider", "openai")
            model_name = tm.get("name", "gpt-4o")
            ui_print(f"    • Provider: {provider.upper()}", PromptStyle.INFO)
            ui_print(f"    • Model: {model_name}", PromptStyle.INFO)
            
            # Show key model parameters
            temperature = tm.get("temperature")
            max_tokens = tm.get("max_output_tokens") or tm.get("max_tokens", 20480)
            if temperature is not None:
                ui_print(f"      - Temperature: {temperature}", PromptStyle.DIM)
            ui_print(f"      - Max output tokens: {max_tokens:,}", PromptStyle.DIM)
            
            # Show reasoning configuration if present
            reasoning = tm.get("reasoning", {})
            if reasoning:
                effort = reasoning.get("effort", "medium")
                ui_print(f"      - Reasoning effort: {effort}", PromptStyle.DIM)
            
            # Show text verbosity if present (GPT-5 specific)
            text_config = tm.get("text", {})
            if text_config:
                verbosity = text_config.get("verbosity", "medium")
                ui_print(f"      - Text verbosity: {verbosity}", PromptStyle.DIM)
            
            print_separator(PromptStyle.LIGHT_LINE, 80)
            
            # === Concurrency Configuration ===
            ui_print("\n  Concurrency Configuration:", PromptStyle.HIGHLIGHT)
            print_separator(PromptStyle.LIGHT_LINE, 80)
            
            # API request concurrency
            trans_cfg = concurrency_config.get("concurrency", {}).get("transcription", {})
            trans_concurrency = trans_cfg.get("concurrency_limit", 5)
            trans_service_tier = trans_cfg.get("service_tier", "default")
            ui_print(f"    • Transcription API: {trans_concurrency} concurrent requests", PromptStyle.INFO)
            ui_print(f"      - Service tier: {trans_service_tier}", PromptStyle.DIM)
            
            # Retry configuration
            retry_config = trans_cfg.get("retry", {})
            max_attempts = retry_config.get("attempts", 5)
            ui_print(f"      - Max retry attempts: {max_attempts}", PromptStyle.DIM)
            
            # Daily token limit
            daily_limit_cfg = concurrency_config.get("daily_token_limit", {})
            if daily_limit_cfg.get("enabled", False):
                daily_tokens = daily_limit_cfg.get("daily_tokens", 0)
                ui_print(f"    • Daily token limit: {daily_tokens:,}", PromptStyle.INFO)
            
            print_separator(PromptStyle.LIGHT_LINE, 80)
        
        # === Output Location ===
        ui_print("\n  Output Location:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        use_input_as_output = paths_config.get('general', {}).get('input_paths_is_output_path', False)
        if use_input_as_output:
            ui_print("    • Output: Same directory as input files", PromptStyle.INFO)
        else:
            file_paths = paths_config.get('file_paths', {})
            if is_auto:
                output_dir = file_paths.get('Auto', {}).get('output', 'auto_out')
            elif config.processing_type == "images":
                output_dir = file_paths.get('Images', {}).get('output', 'images_out')
            elif config.processing_type == "pdfs":
                output_dir = file_paths.get('PDFs', {}).get('output', 'pdfs_out')
            elif config.processing_type == "epubs":
                output_dir = file_paths.get('EPUBs', {}).get('output', 'epubs_out')
            else:
                output_dir = "configured output directory"
            ui_print(f"    • Output directory: {output_dir}", PromptStyle.INFO)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        # === Selected Items ===
        if is_auto:
            # Show first 5 files from auto decisions
            ui_print("\n  Selected Files (first 5 shown):", PromptStyle.HIGHLIGHT)
            for i, decision in enumerate(decisions[:5], 1):
                ui_print(f"    {i}. {decision.file_path.name}", PromptStyle.DIM)
            if len(decisions) > 5:
                ui_print(f"    ... and {len(decisions) - 5} more", PromptStyle.DIM)
            selected_file_paths = [d.file_path for d in decisions]
        else:
            ui_print("\n  Selected Items (first 5 shown):", PromptStyle.HIGHLIGHT)
            selected = config.selected_items or []
            for i, item in enumerate(selected[:5], 1):
                ui_print(f"    {i}. {item.name}", PromptStyle.DIM)
            if len(selected) > 5:
                ui_print(f"    ... and {len(selected) - 5} more", PromptStyle.DIM)
            selected_file_paths = list(selected)
        
        # === Resume Information ===
        from modules.core.resume import ResumeChecker, ProcessingState
        resume_checker = ResumeChecker(
            resume_mode=config.resume_mode,
            paths_config=paths_config,
            use_input_as_output=use_input_as_output,
        )
        # For auto mode, derive processing_type from each decision's file_type;
        # use "pdfs" as a reasonable default since most auto items are PDFs.
        resume_processing_type = config.processing_type or "pdfs"
        _, skipped = resume_checker.filter_items(
            selected_file_paths, resume_processing_type
        )
        # Always show resume information
        total_count = len(selected_file_paths)
        ui_print("")
        ui_print("  Resume Information:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        ui_print(f"    • Resume mode: {config.resume_mode}", PromptStyle.INFO)
        if skipped and config.resume_mode == "skip":
            ui_print(
                f"    • {len(skipped)} of {total_count} item(s) already have output and will be skipped",
                PromptStyle.WARNING,
            )
            new_count = total_count - len(skipped)
            ui_print(f"    • {new_count} item(s) will be processed", PromptStyle.INFO)
        elif config.resume_mode == "overwrite":
            ui_print(f"    • All {total_count} item(s) will be (re)processed", PromptStyle.INFO)
        else:
            ui_print(f"    • {total_count} item(s) will be processed", PromptStyle.INFO)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        if skipped and config.resume_mode == "skip":
            new_count = total_count - len(skipped)
            if new_count == 0:
                print_warning("All items already processed. Nothing to do.")
                result = prompt_yes_no(
                    "Force reprocess all items?",
                    default=False,
                    allow_back=True,
                )
                if result.action == NavigationAction.CONTINUE and result.value:
                    config.resume_mode = "overwrite"
                    return True
                return False
        
        ui_print("")
        
        result = prompt_yes_no(
            "Proceed with processing?",
            default=True,
            allow_back=True
        )
        
        if result.action == NavigationAction.CONTINUE:
            return bool(result.value)
        
        return False
    
    @staticmethod
    def display_completion_summary(
        config: UserConfiguration,
        processed_count: int = 0,
        failed_count: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Display detailed completion summary.
        
        Args:
            config: User configuration object
            processed_count: Number of successfully processed items
            failed_count: Number of failed items
            duration_seconds: Total processing duration in seconds
        """
        from modules.config.service import get_config_service
        config_service = get_config_service()
        paths_config = config_service.get_paths_config()
        
        print_header("PROCESSING COMPLETE", "")
        
        total_count = processed_count + failed_count
        
        # === Results Section ===
        ui_print("  Results:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        if config.use_batch_processing and config.transcription_method == "gpt":
            print_success("Batch processing jobs have been submitted!")
            ui_print(f"    • Jobs submitted: {total_count}", PromptStyle.INFO)
        else:
            if failed_count == 0 and processed_count > 0:
                print_success(f"All {processed_count} item(s) processed successfully!")
            elif processed_count > 0:
                ui_print(f"    • Processed: {processed_count}/{total_count} item(s)", PromptStyle.INFO)
                if failed_count > 0:
                    ui_print(f"    • Failed: {failed_count} item(s)", PromptStyle.WARNING)
            else:
                ui_print("    • No items were processed.", PromptStyle.WARNING)
        
        # Duration
        if duration_seconds > 0:
            if duration_seconds >= 3600:
                hours = duration_seconds / 3600
                ui_print(f"    • Duration: {hours:.1f} hours", PromptStyle.INFO)
            elif duration_seconds >= 60:
                minutes = duration_seconds / 60
                ui_print(f"    • Duration: {minutes:.1f} minutes", PromptStyle.INFO)
            else:
                ui_print(f"    • Duration: {duration_seconds:.1f} seconds", PromptStyle.INFO)
        
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        # === Output Location ===
        ui_print("\n  Output:", PromptStyle.HIGHLIGHT)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        use_input_as_output = paths_config.get('general', {}).get('input_paths_is_output_path', False)
        if use_input_as_output:
            ui_print("    • Location: Same directory as input files", PromptStyle.INFO)
        else:
            file_paths = paths_config.get('file_paths', {})
            if config.processing_type == "images":
                output_dir = file_paths.get('Images', {}).get('output', 'images_out')
            elif config.processing_type == "pdfs":
                output_dir = file_paths.get('PDFs', {}).get('output', 'pdfs_out')
            elif config.processing_type == "epubs":
                output_dir = file_paths.get('EPUBs', {}).get('output', 'epubs_out')
            else:
                output_dir = "configured output directory"
            ui_print(f"    • Location: {output_dir}", PromptStyle.INFO)
        ui_print("    • Transcriptions: .txt files", PromptStyle.INFO)
        print_separator(PromptStyle.LIGHT_LINE, 80)
        
        # === Next Steps (for batch mode) ===
        if config.use_batch_processing and config.transcription_method == "gpt":
            ui_print("\n  Next steps:", PromptStyle.HIGHLIGHT)
            print_separator(PromptStyle.LIGHT_LINE, 80)
            ui_print("    • Check batch status: ", PromptStyle.DIM, end="")
            ui_print("python main/check_batches.py", PromptStyle.INFO)
            ui_print("    • Cancel pending batches: ", PromptStyle.DIM, end="")
            ui_print("python main/cancel_batches.py", PromptStyle.INFO)
            print_separator(PromptStyle.LIGHT_LINE, 80)
        
        ui_print("\n  Thank you for using ChronoTranscriber!\n", PromptStyle.HIGHLIGHT)
