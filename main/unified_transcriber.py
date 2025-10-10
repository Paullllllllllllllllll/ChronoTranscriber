# unified_transcriber.py
"""
Main CLI script for the ChronoTranscriber application.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import sys
import os
import asyncio
import logging
import traceback
from pathlib import Path

from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
from modules.infra.logger import setup_logger
from modules.llm.openai_utils import open_transcriber
from modules.ui import (
    UserConfiguration,
    WorkflowUI,
    print_info,
    print_success,
    print_error,
    ui_print,
    PromptStyle,
)
from modules.io.path_utils import validate_paths
from modules.core.workflow import WorkflowManager
from modules.core.cli_args import (
    create_transcriber_parser,
    resolve_path,
    validate_input_path,
    validate_output_path,
)
from modules.core.mode_selector import run_with_mode_detection
from modules.llm.schema_utils import list_schema_options

logger = setup_logger(__name__)


def create_config_from_cli_args(args, base_input_dir: Path, base_output_dir: Path) -> UserConfiguration:
    """Create UserConfiguration from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        base_input_dir: Base directory for relative input paths
        base_output_dir: Base directory for relative output paths
        
    Returns:
        UserConfiguration object
    """
    config = UserConfiguration()
    
    # Set processing type and method
    config.processing_type = args.type
    config.transcription_method = args.method

    # EPUBs currently support native extraction only
    if config.processing_type == "epubs" and config.transcription_method != "native":
        raise ValueError("EPUB processing only supports the 'native' method.")

    # Set batch processing (only for GPT)
    if args.method == "gpt":
        config.use_batch_processing = args.batch
        
        # Handle schema selection
        if args.schema:
            # Find schema by name
            options = list_schema_options()
            schema_dict = {name: path for name, path in options}
            if args.schema in schema_dict:
                config.selected_schema_name = args.schema
                config.selected_schema_path = schema_dict[args.schema]
            else:
                raise ValueError(f"Schema '{args.schema}' not found. Available: {list(schema_dict.keys())}")
        else:
            # Use default schema
            default_schema = (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json").resolve()
            config.selected_schema_name = "markdown_transcription_schema"
            config.selected_schema_path = default_schema
        
        # Handle additional context
        if args.context:
            context_path = resolve_path(args.context, PROJECT_ROOT)
            validate_input_path(context_path)
            config.additional_context_path = context_path
        else:
            config.additional_context_path = None
    
    # Resolve input path
    input_path = resolve_path(args.input, base_input_dir)
    validate_input_path(input_path)
    
    # Resolve output path
    output_path = resolve_path(args.output, base_output_dir)
    validate_output_path(output_path)
    
    # Determine selected items based on input type
    if config.processing_type == "images":
        # For images, expect input to be a folder containing subfolders
        if input_path.is_dir():
            if args.files:
                # Specific subfolders
                config.selected_items = [input_path / f for f in args.files]
            elif args.recursive:
                # All subfolders
                config.selected_items = [d for d in input_path.iterdir() if d.is_dir()]
            else:
                # Direct processing of the folder
                config.selected_items = [input_path]
        else:
            raise ValueError(f"For image processing, input must be a directory: {input_path}")
    elif config.processing_type == "pdfs":
        # For PDFs, collect PDF files
        if input_path.is_dir():
            if args.files:
                # Specific files
                config.selected_items = [input_path / f for f in args.files]
            elif args.recursive:
                # All PDFs recursively
                config.selected_items = list(input_path.rglob("*.pdf"))
            else:
                # PDFs in the directory
                config.selected_items = list(input_path.glob("*.pdf"))
        elif input_path.is_file():
            # Single PDF file
            config.selected_items = [input_path]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    else:
        # For EPUBs, collect EPUB files
        if input_path.is_dir():
            if args.files:
                config.selected_items = [input_path / f for f in args.files]
            elif args.recursive:
                config.selected_items = list(input_path.rglob("*.epub"))
            else:
                config.selected_items = list(input_path.glob("*.epub"))
        elif input_path.is_file():
            config.selected_items = [input_path]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    
    if not config.selected_items:
        raise ValueError(f"No items found to process in: {input_path}")
    
    return config


async def configure_user_workflow_interactive(
    pdf_input_dir: Path,
    image_input_dir: Path,
    epub_input_dir: Path,
) -> UserConfiguration:
    """
    Guide user through configuration with navigation support (interactive mode).
    
    Returns:
        UserConfiguration object with all settings
    """
    config = UserConfiguration()
    
    # Display welcome banner
    WorkflowUI.display_welcome()
    
    # Navigation state machine
    current_step = "processing_type"
    
    while True:
        if current_step == "processing_type":
            if WorkflowUI.configure_processing_type(config):
                current_step = "transcription_method"
            else:
                current_step = "processing_type"
        
        elif current_step == "transcription_method":
            if WorkflowUI.configure_transcription_method(config):
                current_step = "batch_processing"
            else:
                current_step = "processing_type"
        
        elif current_step == "batch_processing":
            if WorkflowUI.configure_batch_processing(config):
                current_step = "item_selection"
            else:
                current_step = "transcription_method"
        
        elif current_step == "item_selection":
            if config.processing_type == "images":
                base_dir = image_input_dir
            elif config.processing_type == "pdfs":
                base_dir = pdf_input_dir
            else:
                base_dir = epub_input_dir

            if WorkflowUI.select_items_for_processing(config, base_dir):
                current_step = "summary"
            else:
                current_step = "batch_processing"
        
        elif current_step == "summary":
            confirmed = WorkflowUI.display_processing_summary(config)
            if confirmed:
                return config
            else:
                current_step = "item_selection"


async def process_documents(
    user_config: UserConfiguration,
    paths_config: dict,
    model_config: dict,
    concurrency_config: dict,
    image_processing_config: dict,
) -> None:
    """
    Process documents based on user configuration.
    
    Args:
        user_config: User's processing preferences
        paths_config: Paths configuration
        model_config: Model configuration
        concurrency_config: Concurrency configuration
        image_processing_config: Image processing configuration
    """
    print_info("PROCESSING", "Starting document processing...")
    
    # Create workflow manager
    workflow_manager = WorkflowManager(
        user_config,
        paths_config,
        model_config,
        concurrency_config,
        image_processing_config
    )
    
    # Initialize transcriber if needed for synchronous GPT processing
    transcriber = None
    if user_config.transcription_method == "gpt" and not user_config.use_batch_processing:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print_error("OPENAI_API_KEY is required for GPT transcription.")
            sys.exit(1)
        
        async with open_transcriber(
            api_key=api_key,
            model=model_config.get("transcription_model", {}).get("name", "gpt-4o"),
            schema_path=user_config.selected_schema_path,
            additional_context_path=user_config.additional_context_path,
        ) as t:
            transcriber = t
            await workflow_manager.process_selected_items(transcriber)
    else:
        # For non-GPT methods or batch processing, no transcriber needed
        await workflow_manager.process_selected_items()


async def transcribe_interactive() -> None:
    """Handle interactive mode transcription workflow."""
    # Load configurations
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config = config_loader.get_paths_config()
    
    # Validate paths
    validate_paths(paths_config)
    
    # Get directory paths from config
    pdf_input_dir = Path(
        paths_config.get('file_paths', {}).get('PDFs', {}).get('input', 'pdfs_in')
    )
    image_input_dir = Path(
        paths_config.get('file_paths', {}).get('Images', {}).get('input', 'images_in')
    )
    epub_input_dir = Path(
        paths_config.get('file_paths', {}).get('EPUBs', {}).get('input', 'epubs_in')
    )
    
    # Ensure directories exist for interactive mode
    pdf_input_dir.mkdir(parents=True, exist_ok=True)
    image_input_dir.mkdir(parents=True, exist_ok=True)
    epub_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Create user configuration through interactive workflow
    user_config = await configure_user_workflow_interactive(pdf_input_dir, image_input_dir, epub_input_dir)
    
    # Process documents
    await process_documents(
        user_config,
        paths_config,
        config_loader.get_model_config(),
        config_loader.get_concurrency_config(),
        config_loader.get_image_processing_config()
    )
    
    # Display completion summary
    WorkflowUI.display_completion_summary(user_config)


async def transcribe_cli(args, paths_config: dict) -> None:
    """Handle CLI mode transcription workflow.
    
    Args:
        args: Parsed command-line arguments
        paths_config: Paths configuration dictionary
    """
    # Load additional configurations
    config_loader = ConfigLoader()
    config_loader.load_configs()
    
    # Validate paths
    validate_paths(paths_config)
    
    # Get directory paths from config
    pdf_input_dir = Path(
        paths_config.get('file_paths', {}).get('PDFs', {}).get('input', 'pdfs_in')
    )
    image_input_dir = Path(
        paths_config.get('file_paths', {}).get('Images', {}).get('input', 'images_in')
    )
    pdf_output_dir = Path(
        paths_config.get('file_paths', {}).get('PDFs', {}).get('output', 'pdfs_out')
    )
    image_output_dir = Path(
        paths_config.get('file_paths', {}).get('Images', {}).get('output', 'images_out')
    )
    epub_input_dir = Path(
        paths_config.get('file_paths', {}).get('EPUBs', {}).get('input', 'epubs_in')
    )
    epub_output_dir = Path(
        paths_config.get('file_paths', {}).get('EPUBs', {}).get('output', 'epubs_out')
    )
    
    # Determine base directories based on processing type
    if args.type == "images":
        base_input_dir = image_input_dir
        base_output_dir = image_output_dir
    elif args.type == "pdfs":
        base_input_dir = pdf_input_dir
        base_output_dir = pdf_output_dir
    else:
        base_input_dir = epub_input_dir
        base_output_dir = epub_output_dir
    
    # Create configuration from CLI arguments
    user_config = create_config_from_cli_args(args, base_input_dir, base_output_dir)
    
    # Process documents
    await process_documents(
        user_config,
        paths_config,
        config_loader.get_model_config(),
        config_loader.get_concurrency_config(),
        config_loader.get_image_processing_config()
    )
    
    print_success("Processing complete!")


async def main() -> None:
    """Main entry point supporting both interactive and CLI modes."""
    try:
        # Use centralized mode detection
        config_loader, interactive_mode, args, paths_config = run_with_mode_detection(
            interactive_handler=transcribe_interactive,
            cli_handler=transcribe_cli,
            parser_factory=create_transcriber_parser,
            script_name="unified_transcriber"
        )
        
        # Route to appropriate handler
        if interactive_mode:
            await transcribe_interactive()
        else:
            await transcribe_cli(args, paths_config)
            
    except KeyboardInterrupt:
        print_info("\nProcessing interrupted by user.")
        sys.exit(0)
    except ValueError as e:
        print_error(f"Invalid arguments: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print_error(f"An unexpected error occurred: {e}")
        print_info("Check the logs for more details.")
        if logger.isEnabledFor(logging.DEBUG):
            ui_print(f"\nTraceback:\n{traceback.format_exc()}", PromptStyle.DIM)
        sys.exit(1)


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())