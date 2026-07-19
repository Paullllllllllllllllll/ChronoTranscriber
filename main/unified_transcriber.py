# unified_transcriber.py
"""
Main CLI script for the ChronoTranscriber application.

Supports two modes:
1. Interactive mode: Guided workflow with user prompts (interactive_mode: true)
2. CLI mode: Command-line arguments for automation (interactive_mode: false)
"""

from __future__ import annotations

import dataclasses
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from modules.config.service import get_config_service  # noqa: E402
from modules.core.cli_args import (  # noqa: E402
    create_transcriber_parser,
    resolve_path,
    validate_output_path,
)
from modules.documents.auto_selector import AutoSelector  # noqa: E402
from modules.infra.logger import setup_logger  # noqa: E402
from modules.infra.paths import PathConfig  # noqa: E402
from modules.llm import open_transcriber  # noqa: E402
from modules.llm.schema_utils import list_schema_options  # noqa: E402, F401
from modules.transcribe.config_builder import (  # noqa: E402
    _resolve_model_config_from_cli,
    create_config_from_cli_args,
)
from modules.transcribe.dual_mode import AsyncDualModeScript  # noqa: E402
from modules.transcribe.manager import (  # noqa: E402
    ProcessingSummary,
    WorkflowManager,
)
from modules.transcribe.user_config import UserConfiguration  # noqa: E402
from modules.ui import (  # noqa: E402
    WorkflowUI,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)

logger = setup_logger(__name__)


async def _open_transcriber_from_config(
    user_config: UserConfiguration,
    model_config: dict[str, Any],
) -> Any:
    """Create an LLM transcriber context manager from config (shared helper).

    Returns an async context manager that yields the transcriber instance.
    Centralises the identical ``open_transcriber`` call used in both
    :func:`process_auto_mode` and :func:`process_documents`.
    """
    tm = model_config.get("transcription_model", {})

    return open_transcriber(
        api_key=None,
        model=tm.get("name", "gpt-4o"),
        provider=tm.get("provider"),
        schema_path=user_config.selected_schema_path,
        additional_context_path=user_config.additional_context_path,
        use_hierarchical_context=getattr(user_config, "use_hierarchical_context", True),
        max_output_tokens=tm.get("max_output_tokens"),
        reasoning_config=tm.get("reasoning"),
        text_config=tm.get("text"),
    )


# Config-synthesis helpers live in modules.transcribe.config_builder and
# are re-imported at the top of this module so existing callers that do
# ``from main.unified_transcriber import create_config_from_cli_args``
# continue to work.


async def configure_user_workflow_interactive(
    pdf_input_dir: Path,
    image_input_dir: Path,
    epub_input_dir: Path,
    mobi_input_dir: Path,
    auto_input_dir: Path,
    paths_config: dict[str, Any],
) -> UserConfiguration:
    """
    Guide user through configuration with navigation support (interactive mode).

    Args:
        pdf_input_dir: PDF input directory
        image_input_dir: Image input directory
        epub_input_dir: EPUB input directory
        mobi_input_dir: MOBI input directory
        auto_input_dir: Auto mode input directory

    Returns:
        UserConfiguration object with all settings
    """
    config = UserConfiguration()
    # Mirror the CLI's config-fallback for output_format (interactive mode has
    # no output-format prompt). Set here — rather than after the wizard — so the
    # auto-mode resume pre-filter inside _configure_auto_mode checks the correct
    # output extension instead of the "txt" default.
    config.output_format = paths_config.get("general", {}).get("output_format", "txt")

    # Display welcome banner
    WorkflowUI.display_welcome()

    # Navigation state machine.
    #
    # Auto flow:     processing_type -> resume_mode -> item_selection ->
    #                auto_schema_selection -> page_range -> summary
    # Non-auto flow: processing_type -> transcription_method -> batch_processing ->
    #                item_selection -> page_range -> resume_mode -> summary
    #
    # Auto mode chooses resume_mode BEFORE item_selection so that the resume
    # pre-filter in _configure_auto_mode uses the real mode instead of the
    # dataclass default "skip" (files dropped under "skip" cannot be restored
    # by a later "Overwrite" choice).
    current_step = "processing_type"

    while True:
        if current_step == "processing_type":
            if WorkflowUI.configure_processing_type(config):
                if config.processing_type == "auto":
                    current_step = "resume_mode"
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

        elif current_step == "resume_mode":
            if WorkflowUI.configure_resume_mode(config):
                # Auto: resume precedes item selection; non-auto: precedes summary.
                if config.processing_type == "auto":
                    current_step = "item_selection"
                else:
                    current_step = "summary"
            else:
                # Back target differs by flow order.
                if config.processing_type == "auto":
                    current_step = "processing_type"
                else:
                    current_step = "page_range"

        elif current_step == "item_selection":
            if config.processing_type == "auto":
                base_dir = auto_input_dir
            elif config.processing_type == "images":
                base_dir = image_input_dir
            elif config.processing_type == "pdfs":
                base_dir = pdf_input_dir
            elif config.processing_type == "mobis":
                base_dir = mobi_input_dir
            else:
                base_dir = epub_input_dir

            if WorkflowUI.select_items_for_processing(config, base_dir, paths_config):
                # Auto mode needs schema selection if GPT files detected
                if config.processing_type == "auto":
                    current_step = "auto_schema_selection"
                else:
                    current_step = "page_range"
            else:
                # Back target: auto -> resume_mode (new order); gpt ->
                # batch_processing; other methods -> transcription_method (a
                # non-gpt method makes configure_batch_processing a no-op that
                # would bounce straight back here).
                if config.processing_type == "auto":
                    current_step = "resume_mode"
                elif config.transcription_method == "gpt":
                    current_step = "batch_processing"
                else:
                    current_step = "transcription_method"

        elif current_step == "auto_schema_selection":
            # Schema selection for auto mode (when GPT files are detected)
            if WorkflowUI.configure_auto_mode_schema(config):
                current_step = "page_range"
            else:
                current_step = "item_selection"

        elif current_step == "page_range":
            if WorkflowUI.configure_page_range(config):
                # Auto: page_range precedes summary; non-auto: precedes resume.
                if config.processing_type == "auto":
                    current_step = "summary"
                else:
                    current_step = "resume_mode"
            else:
                # Back target. For auto, skip auto_schema_selection when it has
                # no GPT decisions to show (it returns immediately and would
                # otherwise bounce straight back here).
                if config.processing_type == "auto":
                    decisions = config.auto_decisions or []
                    if any(d.method == "gpt" for d in decisions):
                        current_step = "auto_schema_selection"
                    else:
                        current_step = "item_selection"
                else:
                    current_step = "item_selection"

        elif current_step == "summary":
            confirmed = WorkflowUI.display_processing_summary(config)
            if confirmed:
                return config
            else:
                # Back to the step immediately preceding summary in each flow.
                if config.processing_type == "auto":
                    current_step = "page_range"
                else:
                    current_step = "resume_mode"


async def process_auto_mode(
    user_config: UserConfiguration,
    paths_config: dict[str, Any],
    model_config: dict[str, Any],
    concurrency_config: dict[str, Any],
    image_processing_config: dict[str, Any],
    *,
    print_decision_summary: bool = True,
) -> ProcessingSummary:
    """Process documents in auto mode with per-file method decisions.

    Args:
        user_config: User configuration with auto_decisions
        paths_config: Paths configuration
        model_config: Model configuration
        concurrency_config: Concurrency configuration
        image_processing_config: Image processing configuration
        print_decision_summary: When True, print the AUTO MODE DECISIONS table.
            The interactive wizard already prints it during selection, so it
            passes False here to avoid a duplicate; CLI mode keeps it True.
    """

    print_header("AUTO MODE", "Processing files with automatic method selection...")

    summary = ProcessingSummary()
    decisions = user_config.auto_decisions or []
    if not decisions:
        print_warning("No auto mode decisions available. Nothing to process.")
        return summary

    selected = user_config.selected_items or []
    output_dir = selected[0] if selected else Path.cwd()  # Default output directory

    selector = user_config.auto_selector or AutoSelector(paths_config)
    user_config.auto_selector = selector
    if print_decision_summary:
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

        # Full copy of the user's config, overriding only the fields the auto
        # method-grouping decides. A field-by-field rebuild previously dropped
        # output_format, output_mode, input_root, the two context paths, and
        # sync_fallback (e.g. `--auto --output-format md` wrote .txt, and an
        # explicit --context was clobbered per file). dataclasses.replace carries
        # every other field through unchanged (CT-9).
        temp_config = dataclasses.replace(
            user_config,
            transcription_method=method,
            use_batch_processing=False,  # Auto mode uses synchronous
            selected_items=[d.file_path for d in items],
            processing_type="auto",  # per-item routing in process_selected_items
        )

        # Create workflow manager
        workflow_manager = WorkflowManager(
            temp_config,
            paths_config,
            model_config,
            concurrency_config,
            image_processing_config,
        )

        # When output is NOT co-located with input, redirect all output
        # to the configured Auto output directory.
        if not pc.use_input_as_output:
            workflow_manager.pdf_output_dir = output_dir
            workflow_manager.image_output_dir = output_dir
            workflow_manager.epub_output_dir = output_dir
            workflow_manager.mobi_output_dir = output_dir
            # The ResumeChecker was built in WorkflowManager.__init__ from the
            # ORIGINAL output dirs; re-point it at the redirected auto output dir
            # so item-level resume actually finds completed items (CT-9). Without
            # this the checker looks in the wrong directory and re-processes
            # everything.
            rc = workflow_manager.resume_checker
            rc.pdf_output_dir = output_dir
            rc.image_output_dir = output_dir
            rc.epub_output_dir = output_dir
            rc.mobi_output_dir = output_dir

        transcriber = None
        if method == "gpt":
            # Ensure schema is set (fall back to default if not specified)
            if user_config.selected_schema_path is None:
                from modules.config.config_loader import PROJECT_ROOT

                user_config.selected_schema_path = (
                    PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
                ).resolve()

            async with await _open_transcriber_from_config(
                user_config, model_config
            ) as t:
                transcriber = t
                group_summary = await workflow_manager.process_selected_items(
                    transcriber
                )
        else:
            group_summary = await workflow_manager.process_selected_items()

        summary.processed += group_summary.processed
        summary.failed += group_summary.failed
        summary.total += group_summary.total
        summary.skipped += group_summary.skipped

    print_success("Auto mode processing complete!")
    return summary


async def process_documents(
    user_config: UserConfiguration,
    paths_config: dict[str, Any],
    model_config: dict[str, Any],
    concurrency_config: dict[str, Any],
    image_processing_config: dict[str, Any],
) -> ProcessingSummary:
    """
    Process documents based on user configuration.

    Args:
        user_config: User's processing preferences
        paths_config: Paths configuration
        model_config: Model configuration
        concurrency_config: Concurrency configuration
        image_processing_config: Image processing configuration

    Returns:
        A `ProcessingSummary` with the real success/failure counts.
    """
    print_info("Starting document processing...")

    # Create workflow manager
    workflow_manager = WorkflowManager(
        user_config,
        paths_config,
        model_config,
        concurrency_config,
        image_processing_config,
    )

    # Initialize transcriber if needed for synchronous GPT processing
    if (
        user_config.transcription_method == "gpt"
        and not user_config.use_batch_processing
    ):
        async with await _open_transcriber_from_config(user_config, model_config) as t:
            return await workflow_manager.process_selected_items(t)
    # For non-GPT methods or batch processing, no transcriber needed
    return await workflow_manager.process_selected_items()


async def transcribe_interactive() -> None:
    """Handle interactive mode transcription workflow."""
    import time

    # Load configurations
    config_service = get_config_service()
    paths_config = config_service.get_paths_config()
    pc = PathConfig.from_paths_config(paths_config)
    pc.ensure_input_dirs()
    # ensure_input_dirs() does not cover the MOBI input directory; create it so
    # interactive MOBI selection can scan an existing folder.
    pc.mobi_input_dir.mkdir(parents=True, exist_ok=True)

    concurrency_config = config_service.get_concurrency_config()

    # Create user configuration through interactive workflow
    user_config = await configure_user_workflow_interactive(
        pc.pdf_input_dir,
        pc.image_input_dir,
        pc.epub_input_dir,
        pc.mobi_input_dir,
        pc.auto_input_dir,
        paths_config,
    )

    # Track processing time
    start_time = time.time()
    summary: ProcessingSummary | None = None

    # Process documents
    if user_config.processing_type == "auto":
        # Override output directory with Auto output
        user_config.selected_items = [pc.auto_output_dir]
        summary = await process_auto_mode(
            user_config,
            paths_config,
            config_service.get_model_config(),
            concurrency_config,
            config_service.get_image_processing_config(),
            # The wizard already displayed the AUTO MODE DECISIONS table during
            # selection; suppress the duplicate here.
            print_decision_summary=False,
        )
    else:
        summary = await process_documents(
            user_config,
            paths_config,
            config_service.get_model_config(),
            concurrency_config,
            config_service.get_image_processing_config(),
        )

    # Calculate duration
    duration_seconds = time.time() - start_time

    # Display a comprehensive completion summary with real counts for both the
    # auto and non-auto paths.
    if summary is not None:
        WorkflowUI.display_completion_summary(
            user_config,
            processed_count=summary.processed,
            failed_count=summary.failed,
            duration_seconds=duration_seconds,
            skipped=summary.skipped,
            concurrency_config=concurrency_config,
        )


def _emit_json_summary(summary: ProcessingSummary, *, dry_run: bool = False) -> None:
    """Print a single machine-readable JSON summary line on stdout."""
    payload = {
        "tool": "chronotranscriber",
        "dry_run": dry_run,
        "items_total": summary.total,
        "items_processed": summary.processed,
        "items_failed": summary.failed,
        "exit_code": 0 if summary.failed == 0 else 1,
    }
    print(json.dumps(payload, ensure_ascii=False))


def _dry_run_report(
    user_config: UserConfiguration,
    paths_config: dict[str, Any],
) -> ProcessingSummary:
    """Report discovery + resume classification without any API calls.

    Returns a ProcessingSummary whose ``total`` is the number of items that
    would be processed after resume filtering.
    """
    if user_config.processing_type == "auto":
        decisions = user_config.auto_decisions or []
        planned = len(decisions)
        print_info(f"[DRY RUN] Auto mode: {planned} file(s) would be processed.")
        return ProcessingSummary(processed=0, failed=0, total=planned)

    from modules.infra.paths import PathConfig
    from modules.transcribe.resume import ResumeChecker

    pc = PathConfig.from_paths_config(paths_config)
    checker = ResumeChecker(
        resume_mode=user_config.resume_mode,
        paths_config=paths_config,
        use_input_as_output=pc.use_input_as_output,
        pdf_output_dir=pc.pdf_output_dir,
        image_output_dir=pc.image_output_dir,
        epub_output_dir=pc.epub_output_dir,
        mobi_output_dir=pc.mobi_output_dir,
        output_format=user_config.output_format,
        output_mode=user_config.output_mode,
        input_root=user_config.input_root,
        retry_errors=user_config.retry_errors,
    )
    items = list(user_config.selected_items or [])
    to_process, skipped = checker.filter_items(items, user_config.processing_type or "")
    print_info(
        f"[DRY RUN] {len(items)} discovered; {len(to_process)} would be processed, "
        f"{len(skipped)} skipped (already complete)."
    )
    for item in to_process:
        print_info(f"[DRY RUN]   would process: {item.name}")
    return ProcessingSummary(processed=0, failed=0, total=len(to_process))


async def transcribe_cli(args: Any, paths_config: dict[str, Any]) -> int:
    """Handle CLI mode transcription workflow.

    Args:
        args: Parsed command-line arguments
        paths_config: Paths configuration dictionary

    Returns:
        Process exit code: 0 = success, 1 = one or more items failed.
    """
    # Load additional configurations
    config_service = get_config_service()
    pc = PathConfig.from_paths_config(paths_config)

    # Determine base directories based on processing type
    ptype = "auto" if args.auto else (args.type or "pdfs")
    base_input_dir, base_output_dir = pc.base_dirs_for_type(ptype)

    # Create configuration from CLI arguments
    user_config = create_config_from_cli_args(
        args, base_input_dir, base_output_dir, paths_config
    )

    # Resolve model config from base config + optional CLI overrides
    effective_model_config, applied_model_overrides = _resolve_model_config_from_cli(
        config_service.get_model_config(),
        args,
    )
    if applied_model_overrides:
        print_info(f"CLI model overrides: {', '.join(applied_model_overrides)}")

    effective_paths_config = paths_config
    if not args.auto:
        output_path = resolve_path(args.output, base_output_dir)
        validate_output_path(output_path)

        effective_paths_config = deepcopy(paths_config)
        file_paths_cfg = effective_paths_config.setdefault("file_paths", {})
        _TYPE_TO_SECTION = {
            "images": "Images",
            "pdfs": "PDFs",
            "epubs": "EPUBs",
            "mobis": "MOBIs",
        }
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
            ctx_name = user_config.additional_context_path.name
            print_info(f"Additional context: Global ({ctx_name})")
        elif getattr(user_config, "use_hierarchical_context", False):
            print_info("Additional context: Hierarchical (file/folder-specific)")
        else:
            print_info("Additional context: None")

    json_summary = bool(getattr(args, "json_summary", False))

    # Dry run: discovery + resume classification only, no API calls / side effects.
    if getattr(args, "dry_run", False):
        summary = _dry_run_report(user_config, effective_paths_config)
        if json_summary:
            _emit_json_summary(summary, dry_run=True)
        return 0

    # Process documents
    if user_config.processing_type == "auto":
        summary = await process_auto_mode(
            user_config,
            effective_paths_config,
            effective_model_config,
            config_service.get_concurrency_config(),
            config_service.get_image_processing_config(),
        )
    else:
        summary = await process_documents(
            user_config,
            effective_paths_config,
            effective_model_config,
            config_service.get_concurrency_config(),
            config_service.get_image_processing_config(),
        )

    if summary.failed:
        print_warning(
            f"Processing complete with {summary.failed} failure(s) "
            f"of {summary.total} item(s)."
        )
    else:
        print_success("Processing complete!")

    if json_summary:
        _emit_json_summary(summary)

    # Exit code 1 when any item failed/partial (CLI agent contract).
    return 1 if summary.failed else 0


class UnifiedTranscriberScript(AsyncDualModeScript):
    """Main script for the ChronoTranscriber application."""

    def __init__(self) -> None:
        super().__init__("unified_transcriber")

    def create_argument_parser(self) -> Any:
        """Create argument parser for CLI mode."""
        return create_transcriber_parser()

    async def run_interactive(self) -> None:
        """Run transcription in interactive mode."""
        await transcribe_interactive()

    async def run_cli(self, args: Any) -> None:
        """Run transcription in CLI mode.

        Exit codes: 0 = success; 1 = one or more items failed/partial;
        2 = usage/config error (CLI agent contract).
        """
        try:
            code = await transcribe_cli(args, self.paths_config)
        except ValueError as e:
            # Usage / configuration error.
            print_error(f"Invalid arguments: {e}")
            sys.exit(2)
        if code:
            sys.exit(code)


def main() -> None:
    """Main entry point."""
    UnifiedTranscriberScript().execute()


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()
