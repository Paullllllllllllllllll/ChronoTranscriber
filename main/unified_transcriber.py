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
from copy import deepcopy

from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
from modules.infra.logger import setup_logger
from modules.llm.openai_utils import open_transcriber
from modules.ui import (
    UserConfiguration,
    WorkflowUI,
    print_info,
    print_success,
    print_warning,
    print_error,
    ui_print,
    PromptStyle,
)
from modules.io.path_utils import validate_paths
from modules.core.workflow import WorkflowManager
from modules.core.auto_selector import AutoSelector
from modules.core.cli_args import (
    create_transcriber_parser,
    resolve_path,
    validate_input_path,
    validate_output_path,
)
from modules.core.mode_selector import run_with_mode_detection
from modules.llm.schema_utils import list_schema_options

logger = setup_logger(__name__)


def create_config_from_cli_args(args, base_input_dir: Path, base_output_dir: Path, paths_config: dict) -> UserConfiguration:
    """Create UserConfiguration from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        base_input_dir: Base directory for relative input paths
        base_output_dir: Base directory for relative output paths
        paths_config: Paths configuration dictionary
        
    Returns:
        UserConfiguration object
    """
    config = UserConfiguration()
    config.auto_selector = AutoSelector(paths_config)

    # Handle auto mode
    if args.auto:
        # Use Auto input/output paths from config
        auto_input = Path(paths_config.get('file_paths', {}).get('Auto', {}).get('input', base_input_dir))
        auto_output = Path(paths_config.get('file_paths', {}).get('Auto', {}).get('output', base_output_dir))
        
        # Override with CLI args if provided
        if args.input:
            auto_input = resolve_path(args.input, auto_input)
        if args.output:
            auto_output = resolve_path(args.output, auto_output)
        
        validate_input_path(auto_input)
        validate_output_path(auto_output)
        
        # Generate decisions using shared selector
        decisions = config.auto_selector.create_decisions(auto_input)
        
        if not decisions:
            raise ValueError(f"No processable files found in auto mode input directory: {auto_input}")
        
        # Store decisions in config
        config.processing_type = "auto"
        config.auto_decisions = decisions
        
        # Set output directory for auto mode
        config.selected_items = [auto_output]  # Store output path
        
        return config
    
    # Validate non-auto mode has required arguments
    if not args.type or not args.method:
        raise ValueError("--type and --method are required unless using --auto mode")
    
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
    auto_input_dir: Path,
    paths_config: dict,
) -> UserConfiguration:
    """
    Guide user through configuration with navigation support (interactive mode).
    
    Args:
        pdf_input_dir: PDF input directory
        image_input_dir: Image input directory
        epub_input_dir: EPUB input directory
        auto_input_dir: Auto mode input directory
    
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
                # Auto mode skips method/batch selection
                if config.processing_type == "auto":
                    current_step = "item_selection"
                else:
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
            if config.processing_type == "auto":
                base_dir = auto_input_dir
            elif config.processing_type == "images":
                base_dir = image_input_dir
            elif config.processing_type == "pdfs":
                base_dir = pdf_input_dir
            else:
                base_dir = epub_input_dir

            if WorkflowUI.select_items_for_processing(config, base_dir, paths_config):
                current_step = "summary"
            else:
                # Auto mode goes back to processing_type, others to batch_processing
                current_step = "processing_type" if config.processing_type == "auto" else "batch_processing"
        
        elif current_step == "summary":
            confirmed = WorkflowUI.display_processing_summary(config)
            if confirmed:
                return config
            else:
                current_step = "item_selection"


async def process_auto_mode(
    user_config: UserConfiguration,
    paths_config: dict,
    model_config: dict,
    concurrency_config: dict,
    image_processing_config: dict,
) -> None:
    """Process documents in auto mode with per-file method decisions.
    
    Args:
        user_config: User configuration with auto_decisions
        paths_config: Paths configuration
        model_config: Model configuration
        concurrency_config: Concurrency configuration
        image_processing_config: Image processing configuration
    """
    from modules.core.auto_selector import AutoSelector, FileDecision
    
    print_info("AUTO MODE", "Processing files with automatic method selection...")

    decisions = user_config.auto_decisions or []
    if not decisions:
        print_warning("No auto mode decisions available. Nothing to process.")
        return

    output_dir = user_config.selected_items[0]  # Default output directory

    selector = user_config.auto_selector or AutoSelector(paths_config)
    user_config.auto_selector = selector
    selector.print_decision_summary(decisions)
    
    # Check if we should use input paths as output paths
    use_input_as_output = paths_config.get('general', {}).get('input_paths_is_output_path', False)
    
    # Group decisions by method for efficient processing
    by_method = {}
    for decision in decisions:
        if decision.method not in by_method:
            by_method[decision.method] = []
        by_method[decision.method].append(decision)
    
    # Process each method group
    for method, items in by_method.items():
        print_info(f"Processing {len(items)} file(s) with {method.upper()} method...")

        # Create a temporary UserConfiguration for this method
        temp_config = UserConfiguration()
        temp_config.transcription_method = method
        temp_config.use_batch_processing = False  # Auto mode uses synchronous
        temp_config.selected_items = [d.file_path for d in items]

        first_item = items[0]
        if first_item.file_type == "pdf":
            temp_config.processing_type = "pdfs"
        elif first_item.file_type in ("image", "image_folder"):
            temp_config.processing_type = "images"
        else:
            temp_config.processing_type = "epubs"

        # Determine per-group paths configuration
        group_paths_config = paths_config
        override_with_auto_output = not use_input_as_output

        if use_input_as_output:
            parents: set[Path] = set()
            for decision in items:
                if temp_config.processing_type == "images" and decision.file_path.is_dir():
                    parents.add(decision.file_path)
                else:
                    parents.add(decision.file_path.parent)

            if len(parents) == 1:
                parent_dir = next(iter(parents))
                group_paths_config = deepcopy(paths_config)
                file_paths_cfg = group_paths_config.setdefault('file_paths', {})

                if temp_config.processing_type == "pdfs":
                    pdf_cfg = file_paths_cfg.setdefault('PDFs', {})
                    pdf_cfg['input'] = str(parent_dir)
                    pdf_cfg['output'] = str(parent_dir)
                elif temp_config.processing_type == "images":
                    image_cfg = file_paths_cfg.setdefault('Images', {})
                    image_cfg['input'] = str(parent_dir)
                    image_cfg['output'] = str(parent_dir)
                elif temp_config.processing_type == "epubs":
                    epub_cfg = file_paths_cfg.setdefault('EPUBs', {})
                    epub_cfg['input'] = str(parent_dir)
                    epub_cfg['output'] = str(parent_dir)

                override_with_auto_output = False
            else:
                override_with_auto_output = True

        # Create workflow manager for this batch
        workflow_manager = WorkflowManager(
            temp_config,
            group_paths_config,
            model_config,
            concurrency_config,
            image_processing_config
        )

        if override_with_auto_output:
            workflow_manager.pdf_output_dir = output_dir
            workflow_manager.image_output_dir = output_dir
            workflow_manager.epub_output_dir = output_dir
        transcriber = None
        if method == "gpt":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print_error("OPENAI_API_KEY required for GPT transcription. Skipping GPT files.")
                continue
            
            # Use default schema for auto mode
            from modules.config.config_loader import PROJECT_ROOT
            default_schema = (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json").resolve()
            
            async with open_transcriber(
                api_key=api_key,
                model=model_config.get("transcription_model", {}).get("name", "gpt-4o"),
                schema_path=default_schema,
                additional_context_path=None,
            ) as t:
                transcriber = t
                await workflow_manager.process_selected_items(transcriber)
        else:
            await workflow_manager.process_selected_items()
    
    print_success("Auto mode processing complete!")


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
    auto_input_dir = Path(
        paths_config.get('file_paths', {}).get('Auto', {}).get('input', 'auto_in')
    )
    auto_output_dir = Path(
        paths_config.get('file_paths', {}).get('Auto', {}).get('output', 'auto_out')
    )
    
    # Ensure directories exist for interactive mode
    pdf_input_dir.mkdir(parents=True, exist_ok=True)
    image_input_dir.mkdir(parents=True, exist_ok=True)
    epub_input_dir.mkdir(parents=True, exist_ok=True)
    auto_input_dir.mkdir(parents=True, exist_ok=True)
    auto_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create user configuration through interactive workflow
    user_config = await configure_user_workflow_interactive(
        pdf_input_dir,
        image_input_dir,
        epub_input_dir,
        auto_input_dir,
        paths_config,
    )
    
    # Process documents
    if user_config.processing_type == "auto":
        # Override output directory with Auto output
        user_config.selected_items = [auto_output_dir]
        await process_auto_mode(
            user_config,
            paths_config,
            config_loader.get_model_config(),
            config_loader.get_concurrency_config(),
            config_loader.get_image_processing_config()
        )
    else:
        await process_documents(
            user_config,
            paths_config,
            config_loader.get_model_config(),
            config_loader.get_concurrency_config(),
            config_loader.get_image_processing_config()
        )
    
    # Display completion summary
    if user_config.processing_type != "auto":  # Auto mode prints its own summary
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
    if args.auto:
        # Auto mode uses Auto paths from config
        base_input_dir = Path(paths_config.get('file_paths', {}).get('Auto', {}).get('input', 'auto_in'))
        base_output_dir = Path(paths_config.get('file_paths', {}).get('Auto', {}).get('output', 'auto_out'))
    elif args.type == "images":
        base_input_dir = image_input_dir
        base_output_dir = image_output_dir
    elif args.type == "pdfs":
        base_input_dir = pdf_input_dir
        base_output_dir = pdf_output_dir
    else:
        base_input_dir = epub_input_dir
        base_output_dir = epub_output_dir
    
    # Create configuration from CLI arguments
    user_config = create_config_from_cli_args(args, base_input_dir, base_output_dir, paths_config)
    
    # Process documents
    if user_config.processing_type == "auto":
        await process_auto_mode(
            user_config,
            paths_config,
            config_loader.get_model_config(),
            config_loader.get_concurrency_config(),
            config_loader.get_image_processing_config()
        )
    else:
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