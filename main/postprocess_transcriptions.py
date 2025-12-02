#!/usr/bin/env python3
"""
postprocess_transcriptions.py

Post-process transcription output files to clean up OCR artifacts,
normalize whitespace, optionally merge hyphenated words, and wrap long lines.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.core.mode_selector import run_sync_with_mode_detection
from modules.core.cli_args import create_postprocess_parser, resolve_path
from modules.processing.postprocess import (
    postprocess_transcription,
    postprocess_file,
)
from modules.ui import (
    print_info,
    print_success,
    print_warning,
    print_error,
    prompt_select,
    prompt_yes_no,
    prompt_text,
    NavigationAction,
)

logger = setup_logger(__name__)


def collect_transcription_files(
    input_path: Path,
    recursive: bool = False,
) -> List[Path]:
    """Collect transcription files from a path.
    
    Args:
        input_path: File or directory path
        recursive: Whether to search recursively
        
    Returns:
        List of transcription file paths
    """
    if input_path.is_file():
        return [input_path]
    
    if not input_path.is_dir():
        return []
    
    pattern = "**/*_transcription.txt" if recursive else "*_transcription.txt"
    return sorted(input_path.glob(pattern))


def build_config_from_args(args, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build post-processing config from CLI arguments and base config.
    
    Args:
        args: Parsed CLI arguments
        base_config: Base configuration from YAML
        
    Returns:
        Combined configuration dictionary
    """
    config = dict(base_config) if args.use_config else {}
    
    # Always enable when called from CLI
    config["enabled"] = True
    
    # Override with explicit CLI arguments
    if args.merge_hyphenation:
        config["merge_hyphenation"] = True
    
    if args.no_collapse_spaces:
        config["collapse_internal_spaces"] = False
    elif "collapse_internal_spaces" not in config:
        config["collapse_internal_spaces"] = True
    
    if args.max_blank_lines is not None:
        config["max_blank_lines"] = args.max_blank_lines
    elif "max_blank_lines" not in config:
        config["max_blank_lines"] = 2
    
    if args.tab_size is not None:
        config["tab_size"] = args.tab_size
    elif "tab_size" not in config:
        config["tab_size"] = 4
    
    if args.wrap_width is not None:
        config["wrap_lines"] = True
        config["wrap_width"] = args.wrap_width
        config["auto_wrap"] = False
    elif args.auto_wrap:
        config["wrap_lines"] = True
        config["auto_wrap"] = True
    
    return config


def postprocess_cli(args) -> int:
    """Handle CLI mode post-processing.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Load base config if requested
    config_service = get_config_service()
    paths_config = config_service.get_paths_config()
    base_config = paths_config.get("postprocessing", {})
    
    # Resolve input path
    input_path = resolve_path(args.input)
    if not input_path.exists():
        print_error(f"Input path does not exist: {input_path}")
        return 1
    
    # Collect files to process
    files = collect_transcription_files(input_path, args.recursive)
    if not files:
        print_warning(f"No transcription files found at: {input_path}")
        return 1
    
    # Build config from args
    config = build_config_from_args(args, base_config)
    
    # Determine output handling
    if args.in_place:
        # In-place processing
        print_info(f"Processing {len(files)} file(s) in-place...")
        for file_path in files:
            try:
                postprocess_file(file_path, config=config, in_place=True)
                print_success(f"Processed: {file_path.name}")
            except Exception as e:
                print_error(f"Failed to process {file_path.name}: {e}")
                logger.exception(f"Error processing {file_path}")
        print_success(f"Post-processing complete. {len(files)} file(s) processed.")
        return 0
    
    elif args.output:
        # Output to specified path
        output_path = resolve_path(args.output)
        
        if len(files) == 1:
            # Single file -> single output
            if output_path.is_dir() or not output_path.suffix:
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / files[0].name
            try:
                postprocess_file(files[0], output_path=output_path, config=config)
                print_success(f"Processed: {files[0].name} -> {output_path}")
            except Exception as e:
                print_error(f"Failed to process: {e}")
                return 1
        else:
            # Multiple files -> output directory
            output_path.mkdir(parents=True, exist_ok=True)
            print_info(f"Processing {len(files)} file(s) to {output_path}...")
            for file_path in files:
                try:
                    out_file = output_path / file_path.name
                    postprocess_file(file_path, output_path=out_file, config=config)
                    print_success(f"Processed: {file_path.name}")
                except Exception as e:
                    print_error(f"Failed to process {file_path.name}: {e}")
                    logger.exception(f"Error processing {file_path}")
        
        print_success(f"Post-processing complete. Files saved to: {output_path}")
        return 0
    
    else:
        # Output to stdout (single file only)
        if len(files) > 1:
            print_error("Cannot output multiple files to stdout. Use --output or --in-place.")
            return 1
        
        try:
            text = files[0].read_text(encoding="utf-8", errors="replace")
            processed = postprocess_transcription(text, config)
            sys.stdout.write(processed)
            return 0
        except Exception as e:
            print_error(f"Failed to process: {e}")
            return 1


def postprocess_interactive() -> int:
    """Handle interactive mode post-processing.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print_info("=== Post-Processing Transcription Files ===")
    print_info("")
    print_info("This tool cleans up transcription output by:")
    print_info("  - Normalizing Unicode characters")
    print_info("  - Removing control characters and artifacts")
    print_info("  - Normalizing whitespace")
    print_info("  - Optionally merging hyphenated line breaks")
    print_info("  - Optionally wrapping long lines")
    print_info("")
    
    # Load config
    config_service = get_config_service()
    paths_config = config_service.get_paths_config()
    base_config = paths_config.get("postprocessing", {})
    
    # Step 1: Get input path
    while True:
        result = prompt_text(
            "Enter the path to a transcription file or directory:",
            allow_back=False,
        )
        if result.action == NavigationAction.QUIT:
            print_info("Cancelled.")
            return 0
        
        input_path = Path(result.value).resolve()
        if input_path.exists():
            break
        print_error(f"Path does not exist: {input_path}")
    
    # Step 2: Collect files
    is_dir = input_path.is_dir()
    if is_dir:
        result = prompt_yes_no(
            "Search subdirectories recursively?",
            default=False,
            allow_back=True,
        )
        if result.action == NavigationAction.QUIT:
            print_info("Cancelled.")
            return 0
        recursive = result.value
        
        files = collect_transcription_files(input_path, recursive)
        if not files:
            print_warning(f"No *_transcription.txt files found in: {input_path}")
            return 1
        
        print_info(f"Found {len(files)} transcription file(s):")
        for f in files[:10]:
            print_info(f"  - {f.name}")
        if len(files) > 10:
            print_info(f"  ... and {len(files) - 10} more")
    else:
        files = [input_path]
        print_info(f"Processing: {input_path.name}")
    
    # Step 3: Configure options
    print_info("")
    print_info("=== Post-Processing Options ===")
    
    # Use config as base or start fresh
    result = prompt_yes_no(
        "Use settings from config file as base?",
        default=bool(base_config.get("enabled", False)),
        allow_back=True,
    )
    if result.action == NavigationAction.QUIT:
        print_info("Cancelled.")
        return 0
    
    config: Dict[str, Any] = dict(base_config) if result.value else {}
    config["enabled"] = True
    
    # Hyphenation merging
    result = prompt_yes_no(
        "Merge hyphenated line breaks? (e.g., 'politi-\\nche' -> 'politiche')",
        default=config.get("merge_hyphenation", False),
        allow_back=True,
    )
    if result.action == NavigationAction.QUIT:
        print_info("Cancelled.")
        return 0
    config["merge_hyphenation"] = result.value
    
    # Line wrapping
    wrap_options = [
        ("no", "No line wrapping"),
        ("auto", "Auto-detect width from text"),
        ("manual", "Specify wrap width manually"),
    ]
    result = prompt_select(
        "Line wrapping mode:",
        wrap_options,
        allow_back=True,
    )
    if result.action == NavigationAction.QUIT:
        print_info("Cancelled.")
        return 0
    
    if result.value == "no":
        config["wrap_lines"] = False
    elif result.value == "auto":
        config["wrap_lines"] = True
        config["auto_wrap"] = True
    else:
        config["wrap_lines"] = True
        config["auto_wrap"] = False
        result = prompt_text(
            "Enter wrap width (characters):",
            default="80",
            allow_back=True,
        )
        if result.action == NavigationAction.QUIT:
            print_info("Cancelled.")
            return 0
        try:
            config["wrap_width"] = int(result.value)
        except ValueError:
            config["wrap_width"] = 80
    
    # Step 4: Output mode
    print_info("")
    output_options = [
        ("in_place", "Modify files in-place"),
        ("new_dir", "Save to a new directory"),
    ]
    if len(files) == 1:
        output_options.append(("new_file", "Save to a new file"))
    
    result = prompt_select(
        "Output mode:",
        output_options,
        allow_back=True,
    )
    if result.action == NavigationAction.QUIT:
        print_info("Cancelled.")
        return 0
    
    output_mode = result.value
    output_path: Optional[Path] = None
    
    if output_mode == "new_dir":
        result = prompt_text(
            "Enter output directory path:",
            allow_back=True,
        )
        if result.action == NavigationAction.QUIT:
            print_info("Cancelled.")
            return 0
        output_path = Path(result.value).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    elif output_mode == "new_file":
        result = prompt_text(
            "Enter output file path:",
            default=str(files[0].with_suffix(".cleaned.txt")),
            allow_back=True,
        )
        if result.action == NavigationAction.QUIT:
            print_info("Cancelled.")
            return 0
        output_path = Path(result.value).resolve()
    
    # Step 5: Confirm and process
    print_info("")
    print_info("=== Summary ===")
    print_info(f"Files to process: {len(files)}")
    print_info(f"Merge hyphenation: {config.get('merge_hyphenation', False)}")
    print_info(f"Line wrapping: {config.get('wrap_lines', False)}")
    if config.get("wrap_lines"):
        if config.get("auto_wrap"):
            print_info("  Mode: Auto-detect width")
        else:
            print_info(f"  Width: {config.get('wrap_width', 80)} characters")
    print_info(f"Output: {'In-place' if output_mode == 'in_place' else output_path}")
    print_info("")
    
    result = prompt_yes_no("Proceed with post-processing?", default=True, allow_back=True)
    if result.action == NavigationAction.QUIT or not result.value:
        print_info("Cancelled.")
        return 0
    
    # Process files
    print_info("")
    print_info("Processing...")
    
    success_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            if output_mode == "in_place":
                postprocess_file(file_path, config=config, in_place=True)
            elif output_mode == "new_file":
                postprocess_file(file_path, output_path=output_path, config=config)
            else:
                # new_dir
                out_file = output_path / file_path.name
                postprocess_file(file_path, output_path=out_file, config=config)
            
            print_success(f"Processed: {file_path.name}")
            success_count += 1
        except Exception as e:
            print_error(f"Failed: {file_path.name} - {e}")
            logger.exception(f"Error processing {file_path}")
            error_count += 1
    
    # Summary
    print_info("")
    if error_count == 0:
        print_success(f"Post-processing complete! {success_count} file(s) processed successfully.")
    else:
        print_warning(f"Completed with errors: {success_count} succeeded, {error_count} failed.")
    
    return 0 if error_count == 0 else 1


def main() -> None:
    """Main entry point with dual-mode support."""
    # Detect mode and parse arguments
    _config_service, interactive_mode, args, _paths_config = run_sync_with_mode_detection(
        interactive_handler=postprocess_interactive,
        cli_handler=lambda a, p: postprocess_cli(a),
        parser_factory=create_postprocess_parser,
        script_name="postprocess_transcriptions.py",
    )
    
    if interactive_mode:
        sys.exit(postprocess_interactive())
    else:
        sys.exit(postprocess_cli(args))


if __name__ == "__main__":
    main()
