# unified_transcriber.py
"""
Main CLI script for the ChronoTranscriber application.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import sys
from pathlib import Path
from copy import deepcopy
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from modules.config.config_loader import PROJECT_ROOT
from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.llm import open_transcriber
from modules.ui import (
    UserConfiguration,
    WorkflowUI,
    print_info,
    print_success,
    print_warning,
    print_error,
)
from modules.core.workflow import WorkflowManager
from modules.core.auto_selector import AutoSelector
from modules.core.cli_args import (
    create_transcriber_parser,
    resolve_path,
    validate_input_path,
    validate_output_path,
)
from modules.core.execution_framework import AsyncDualModeScript
from modules.core.path_config import PathConfig
from modules.llm.schema_utils import list_schema_options

logger = setup_logger(__name__)


async def _open_transcriber_from_config(
    user_config: UserConfiguration,
    model_config: dict[str, Any],
):
    """Create an LLM transcriber context manager from config (shared helper).

    Returns an async context manager that yields the transcriber instance.
    Centralises the identical ``open_transcriber`` call used in both
    :func:`process_auto_mode` and :func:`process_documents`.
    """
    return open_transcriber(
        api_key=None,
        model=model_config.get("transcription_model", {}).get("name", "gpt-4o"),
        schema_path=user_config.selected_schema_path,
        additional_context_path=user_config.additional_context_path,
        use_hierarchical_context=getattr(user_config, 'use_hierarchical_context', True),
    )


def _resolve_schema(args_schema: str | None, config: UserConfiguration) -> None:
    """Resolve and set schema on config from CLI --schema argument.
    
    If args_schema is provided, looks it up in available schemas.
    Otherwise, sets the default markdown_transcription_schema.
    """
    if args_schema:
        options = list_schema_options()
        schema_dict = {name: path for name, path in options}
        if args_schema in schema_dict:
            config.selected_schema_name = args_schema
            config.selected_schema_path = schema_dict[args_schema]
        else:
            raise ValueError(f"Schema '{args_schema}' not found. Available: {list(schema_dict.keys())}")
    else:
        default_schema = (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json").resolve()
        config.selected_schema_name = "markdown_transcription_schema"
        config.selected_schema_path = default_schema


def _resolve_context(args_context: str | None, config: UserConfiguration) -> None:
    """Resolve and set additional context path on config from CLI --context argument."""
    if args_context:
        context_path = resolve_path(args_context, PROJECT_ROOT)
        validate_input_path(context_path)
        config.additional_context_path = context_path
    else:
        config.additional_context_path = None


def _collect_files_for_type(
    input_path: Path,
    processing_type: str,
    args: Any,
) -> list[Path]:
    """Collect files/folders to process based on processing type and CLI args.
    
    Args:
        input_path: Validated input path (directory or single file).
        processing_type: One of 'images', 'pdfs', 'epubs', 'mobis'.
        args: Parsed CLI args (uses .files, .recursive).
    
    Returns:
        List of paths to process.
    
    Raises:
        ValueError: If input path is invalid for the processing type.
    """
    if processing_type == "images":
        if not input_path.is_dir():
            raise ValueError(f"For image processing, input must be a directory: {input_path}")
        if args.files:
            return [input_path / f for f in args.files]
        elif args.recursive:
            return [d for d in input_path.iterdir() if d.is_dir()]
        else:
            return [input_path]

    # File-based types: pdfs, epubs, mobis
    ext_map: dict[str, list[str]] = {
        "pdfs": [".pdf"],
        "epubs": [".epub"],
        "mobis": [".mobi", ".azw", ".azw3", ".kfx"],
    }
    extensions = ext_map.get(processing_type, [])

    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    # Directory
    if args.files:
        return [input_path / f for f in args.files]
    elif args.recursive:
        return [f for ext in extensions for f in input_path.rglob(f"*{ext}")]
    else:
        return [f for ext in extensions for f in input_path.glob(f"*{ext}")]


def create_config_from_cli_args(args: Any, base_input_dir: Path, base_output_dir: Path, paths_config: dict[str, Any]) -> UserConfiguration:
    """Create UserConfiguration from CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        base_input_dir: Base directory for relative input paths
        base_output_dir: Base directory for relative output paths
        paths_config: Paths configuration dictionary
        
    Returns:
        UserConfiguration object
    """
    from modules.core.page_range import parse_page_range

    config = UserConfiguration()
    config.auto_selector = AutoSelector(paths_config)

    # Resume mode: default is 'skip'; --force/--overwrite switches to 'overwrite'
    if getattr(args, "force", None):
        config.resume_mode = "overwrite"

    # Parse page range if provided
    if getattr(args, "pages", None):
        config.page_range = parse_page_range(args.pages)

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
        
        # Apply resume filtering to exclude already-completed items
        if config.resume_mode == "skip":
            from modules.core.resume import ResumeChecker, ProcessingState
            from modules.core.path_config import PathConfig

            pc = PathConfig.from_paths_config(paths_config)
            checker = ResumeChecker(
                resume_mode=config.resume_mode,
                paths_config=paths_config,
                use_input_as_output=pc.use_input_as_output,
                pdf_output_dir=pc.pdf_output_dir,
                image_output_dir=pc.image_output_dir,
                epub_output_dir=pc.epub_output_dir,
                mobi_output_dir=pc.mobi_output_dir,
            )
            total_before = len(decisions)
            decisions = [
                d for d in decisions
                if checker.should_skip(d.file_path, "auto").state != ProcessingState.COMPLETE
            ]
            skipped_count = total_before - len(decisions)
            if skipped_count:
                print_info(
                    f"Resume: skipping {skipped_count} already-completed file(s) "
                    f"({len(decisions)} remaining)"
                )
            if not decisions:
                print_info("All files already processed. Nothing to do.")
                return config

        # Store decisions in config
        config.processing_type = "auto"
        config.auto_decisions = decisions
        
        # Set output directory for auto mode
        config.selected_items = [auto_output]  # Store output path
        
        _resolve_schema(args.schema, config)
        _resolve_context(args.context, config)
        
        return config
    
    # Validate non-auto mode has required arguments
    if not args.type or not args.method:
        raise ValueError("--type and --method are required unless using --auto mode")
    
    # Set processing type and method
    config.processing_type = args.type
    config.transcription_method = args.method

    # EPUBs and MOBIs currently support native extraction only
    if config.processing_type == "epubs" and config.transcription_method != "native":
        raise ValueError("EPUB processing only supports the 'native' method.")
    if config.processing_type == "mobis" and config.transcription_method != "native":
        raise ValueError("MOBI processing only supports the 'native' method.")

    # Set batch processing (only for GPT)
    if args.method == "gpt":
        config.use_batch_processing = args.batch
        
        _resolve_schema(args.schema, config)
        _resolve_context(args.context, config)
    
    # Resolve input path
    input_path = resolve_path(args.input, base_input_dir)
    validate_input_path(input_path)
    
    # Resolve output path
    output_path = resolve_path(args.output, base_output_dir)
    validate_output_path(output_path)
    
    # Determine selected items based on input type
    config.selected_items = _collect_files_for_type(input_path, config.processing_type, args)
    
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
                # Auto mode needs schema selection if GPT files detected
                if config.processing_type == "auto":
                    current_step = "auto_schema_selection"
                else:
                    current_step = "page_range"
            else:
                # Auto mode goes back to processing_type, others to batch_processing
                current_step = "processing_type" if config.processing_type == "auto" else "batch_processing"
        
        elif current_step == "auto_schema_selection":
            # Schema selection for auto mode (when GPT files are detected)
            if WorkflowUI.configure_auto_mode_schema(config):
                current_step = "page_range"
            else:
                current_step = "item_selection"

        elif current_step == "page_range":
            if WorkflowUI.configure_page_range(config):
                current_step = "resume_mode"
            else:
                # Go back to the previous step
                if config.processing_type == "auto":
                    current_step = "auto_schema_selection"
                else:
                    current_step = "item_selection"

        elif current_step == "resume_mode":
            if WorkflowUI.configure_resume_mode(config):
                current_step = "summary"
            else:
                current_step = "page_range"

        elif current_step == "summary":
            confirmed = WorkflowUI.display_processing_summary(config)
            if confirmed:
                return config
            else:
                current_step = "resume_mode"


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

    selected = user_config.selected_items or []
    output_dir = selected[0] if selected else Path.cwd()  # Default output directory

    selector = user_config.auto_selector or AutoSelector(paths_config)
    user_config.auto_selector = selector
    selector.print_decision_summary(decisions)
    
    pc = PathConfig.from_paths_config(paths_config)
    
    # Group decisions by method only.  All file types within the same method
    # share a single processing queue so that PDFs and image folders are
    # interleaved, preventing one type from exhausting the daily token budget
    # before the other is reached.  Per-item routing (PDF vs image folder vs
    # ebook) happens inside process_selected_items via processing_type="auto".
    by_method: dict[str, list[Any]] = {}
    for decision in decisions:
        by_method.setdefault(decision.method, []).append(decision)
    
    # Process each method group
    for method, items in by_method.items():
        print_info(f"Processing {len(items)} file(s) with {method.upper()} method...")

        # Create a temporary UserConfiguration for this method
        temp_config = UserConfiguration()
        temp_config.transcription_method = method
        temp_config.use_batch_processing = False  # Auto mode uses synchronous
        temp_config.selected_items = [d.file_path for d in items]
        temp_config.resume_mode = user_config.resume_mode
        temp_config.processing_type = "auto"  # per-item routing in process_selected_items
        temp_config.page_range = user_config.page_range

        # Create workflow manager
        workflow_manager = WorkflowManager(
            temp_config,
            paths_config,
            model_config,
            concurrency_config,
            image_processing_config
        )

        # When output is NOT co-located with input, redirect all output
        # to the configured Auto output directory.
        if not pc.use_input_as_output:
            workflow_manager.pdf_output_dir = output_dir
            workflow_manager.image_output_dir = output_dir
            workflow_manager.epub_output_dir = output_dir
            workflow_manager.mobi_output_dir = output_dir

        transcriber = None
        if method == "gpt":
            # Ensure schema is set (fall back to default if not specified)
            if user_config.selected_schema_path is None:
                from modules.config.config_loader import PROJECT_ROOT
                user_config.selected_schema_path = (
                    PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
                ).resolve()

            async with await _open_transcriber_from_config(user_config, model_config) as t:
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
        async with await _open_transcriber_from_config(user_config, model_config) as t:
            transcriber = t
            await workflow_manager.process_selected_items(transcriber)
    else:
        # For non-GPT methods or batch processing, no transcriber needed
        await workflow_manager.process_selected_items()


async def transcribe_interactive() -> None:
    """Handle interactive mode transcription workflow."""
    import time
    
    # Load configurations
    config_service = get_config_service()
    paths_config = config_service.get_paths_config()
    pc = PathConfig.from_paths_config(paths_config)
    pc.ensure_input_dirs()
    
    # Create user configuration through interactive workflow
    user_config = await configure_user_workflow_interactive(
        pc.pdf_input_dir,
        pc.image_input_dir,
        pc.epub_input_dir,
        pc.auto_input_dir,
        paths_config,
    )
    
    # Track processing time
    start_time = time.time()
    processed_count = len(user_config.selected_items or [])
    failed_count = 0
    
    # Process documents
    if user_config.processing_type == "auto":
        # Override output directory with Auto output
        user_config.selected_items = [pc.auto_output_dir]
        await process_auto_mode(
            user_config,
            paths_config,
            config_service.get_model_config(),
            config_service.get_concurrency_config(),
            config_service.get_image_processing_config()
        )
    else:
        await process_documents(
            user_config,
            paths_config,
            config_service.get_model_config(),
            config_service.get_concurrency_config(),
            config_service.get_image_processing_config()
        )
    
    # Calculate duration
    duration_seconds = time.time() - start_time
    
    # Display completion summary
    if user_config.processing_type != "auto":  # Auto mode prints its own summary
        WorkflowUI.display_completion_summary(
            user_config,
            processed_count=processed_count,
            failed_count=failed_count,
            duration_seconds=duration_seconds,
        )


async def transcribe_cli(args: Any, paths_config: dict[str, Any]) -> None:
    """Handle CLI mode transcription workflow.
    
    Args:
        args: Parsed command-line arguments
        paths_config: Paths configuration dictionary
    """
    # Load additional configurations
    config_service = get_config_service()
    pc = PathConfig.from_paths_config(paths_config)
    
    # Determine base directories based on processing type
    ptype = "auto" if args.auto else (args.type or "pdfs")
    base_input_dir, base_output_dir = pc.base_dirs_for_type(ptype)
    
    # Create configuration from CLI arguments
    user_config = create_config_from_cli_args(args, base_input_dir, base_output_dir, paths_config)

    effective_paths_config = paths_config
    if not args.auto:
        output_path = resolve_path(args.output, base_output_dir)
        validate_output_path(output_path)

        effective_paths_config = deepcopy(paths_config)
        file_paths_cfg = effective_paths_config.setdefault("file_paths", {})
        _TYPE_TO_SECTION = {"images": "Images", "pdfs": "PDFs", "epubs": "EPUBs", "mobis": "MOBIs"}
        section = _TYPE_TO_SECTION.get(args.type, "PDFs")
        file_paths_cfg.setdefault(section, {})["output"] = str(output_path)
    
    # Log resume mode for CLI awareness
    if user_config.resume_mode == "skip":
        print_info("Resume mode: skip (use --force to reprocess all)")
    else:
        print_info("Resume mode: overwrite (all files will be reprocessed)")

    # Log context resolution for CLI awareness
    if user_config.transcription_method == "gpt" or (
        user_config.processing_type == "auto"
        and user_config.auto_decisions
        and any(d.method == "gpt" for d in user_config.auto_decisions)
    ):
        if user_config.additional_context_path:
            print_info(f"Additional context: Global ({user_config.additional_context_path.name})")
        elif getattr(user_config, "use_hierarchical_context", False):
            print_info("Additional context: Hierarchical (file/folder-specific)")
        else:
            print_info("Additional context: None")

    # Process documents
    if user_config.processing_type == "auto":
        await process_auto_mode(
            user_config,
            effective_paths_config,
            config_service.get_model_config(),
            config_service.get_concurrency_config(),
            config_service.get_image_processing_config()
        )
    else:
        await process_documents(
            user_config,
            effective_paths_config,
            config_service.get_model_config(),
            config_service.get_concurrency_config(),
            config_service.get_image_processing_config()
        )
    
    print_success("Processing complete!")


class UnifiedTranscriberScript(AsyncDualModeScript):
    """Main script for the ChronoTranscriber application."""
    
    def __init__(self) -> None:
        super().__init__("unified_transcriber")
    
    def create_argument_parser(self) -> Any:
        """Create argument parser for CLI mode."""
        from argparse import ArgumentParser
        return create_transcriber_parser()
    
    async def run_interactive(self) -> None:
        """Run transcription in interactive mode."""
        await transcribe_interactive()
    
    async def run_cli(self, args: Any) -> None:
        """Run transcription in CLI mode."""
        try:
            await transcribe_cli(args, self.paths_config)
        except ValueError as e:
            print_error(f"Invalid arguments: {e}")
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    UnifiedTranscriberScript().execute()


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()