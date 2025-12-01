# main/cost_analysis.py

"""
Script for analyzing token costs from temporary .jsonl files.

Supports two execution modes:
1. Interactive Mode: User-friendly prompts and selections via UI
2. CLI Mode: Command-line arguments for automation and scripting

The mode is controlled by the 'interactive_mode' setting in config/paths_config.yaml
or by providing command-line arguments.

Workflow:
 1. Load configuration and determine input/output paths
 2. Scan for temporary .jsonl files
 3. Extract token usage data from each file
 4. Calculate costs based on model pricing
 5. Display statistics with standard and 50% discounted pricing
 6. Optionally save results as CSV
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.operations.cost_analysis import (
    find_jsonl_files,
    perform_cost_analysis,
    save_analysis_to_csv,
)
from modules.ui import (
    print_info,
    print_warning,
    print_success,
    print_error,
    prompt_yes_no,
    PromptResult,
)
from modules.ui.workflows import WorkflowUI
from modules.ui.cost_display import display_analysis
from modules.core.mode_selector import run_sync_with_mode_detection

# Initialize logger
logger = setup_logger(__name__)


def _run_interactive_mode() -> None:
    """Run cost analysis in interactive mode."""
    WorkflowUI.display_welcome()
    
    print_info("Loading configuration...")
    logger.info("Starting cost analysis (Interactive Mode)")
    
    # Load configuration
    paths_config = get_config_service().get_paths_config()
    # Use file_paths as schemas_paths (PDFs and Images)
    schemas_paths = paths_config.get("file_paths", {})
    
    # Find .jsonl files
    print_info("Scanning for temporary .jsonl files...")
    jsonl_files = find_jsonl_files(paths_config, schemas_paths)
    
    if not jsonl_files:
        print_warning("No temporary .jsonl files found.")
        logger.warning("No .jsonl files found for analysis")
        return
    
    print_success(f"Found {len(jsonl_files)} file(s) to analyze")
    
    # Perform analysis
    print_info("Analyzing token usage...")
    analysis = perform_cost_analysis(jsonl_files)
    
    # Display results
    display_analysis(analysis, interactive_mode=True)
    
    # Ask to save CSV
    if analysis.file_stats:
        result = prompt_yes_no("Save results as CSV?", default=True)
        
        if result == PromptResult.YES:
            # Determine output directory (use first file's directory)
            output_dir = analysis.file_stats[0].file_path.parent
            output_path = output_dir / "cost_analysis.csv"
            
            try:
                save_analysis_to_csv(analysis, output_path)
                print_success(f"Saved to: {output_path}")
            except Exception as e:
                print_error(f"Failed to save CSV: {e}")
    
    print_info("\nAnalysis complete!\n")


def _create_parser() -> ArgumentParser:
    """Create argument parser for CLI mode."""
    parser = ArgumentParser(
        description="Analyze token costs from temporary .jsonl files"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results as CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for CSV file (default: cost_analysis.csv in first file's directory)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    return parser


def _run_cli_mode(args, paths_config) -> None:
    """Run cost analysis in CLI mode."""
    logger.info("Starting cost analysis (CLI Mode)")
    
    # Use file_paths as schemas_paths (PDFs and Images)
    schemas_paths = paths_config.get("file_paths", {})
    
    # Find .jsonl files
    if not args.quiet:
        print("[INFO] Scanning for temporary .jsonl files...")
    
    jsonl_files = find_jsonl_files(paths_config, schemas_paths)
    
    if not jsonl_files:
        print("[WARNING] No temporary .jsonl files found.")
        logger.warning("No .jsonl files found for analysis")
        sys.exit(0)
    
    if not args.quiet:
        print(f"[INFO] Found {len(jsonl_files)} file(s) to analyze")
    
    # Perform analysis
    if not args.quiet:
        print("[INFO] Analyzing token usage...")
    
    analysis = perform_cost_analysis(jsonl_files)
    
    # Display results
    if not args.quiet:
        display_analysis(analysis, interactive_mode=False)
    
    # Save CSV if requested
    if args.save_csv and analysis.file_stats:
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = analysis.file_stats[0].file_path.parent
            output_path = output_dir / "cost_analysis.csv"
        
        try:
            save_analysis_to_csv(analysis, output_path)
            print(f"[SUCCESS] Saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
            sys.exit(1)
    
    logger.info("Cost analysis complete")


def main() -> None:
    """Main entry point."""
    try:
        # Use mode detection helper
        config_loader, interactive_mode, args, paths_config = run_sync_with_mode_detection(
            interactive_handler=_run_interactive_mode,
            cli_handler=_run_cli_mode,
            parser_factory=_create_parser,
            script_name="cost_analysis"
        )
        
        if interactive_mode:
            _run_interactive_mode()
        else:
            _run_cli_mode(args, paths_config)
    
    except KeyboardInterrupt:
        logger.info("Cost analysis interrupted by user")
        print("\n[INFO] Operation cancelled by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error in cost analysis", exc_info=exc)
        print(f"[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
