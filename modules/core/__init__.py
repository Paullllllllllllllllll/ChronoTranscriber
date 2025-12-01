"""Core utilities package.

Provides CLI argument parsing, mode selection, workflow management, token guards,
and utility functions.

Submodules:
- cli_args: CLI argument parsers (create_transcriber_parser, etc.)
- mode_selector: Dual-mode execution (run_with_mode_detection, run_sync_with_mode_detection)
- workflow: WorkflowManager for document processing
- token_guard: Token limit management (check_and_wait_for_token_limit)
- utils: Basic utilities (console_print, check_exit, safe_input)
- auto_selector: Auto mode file detection
- safe_paths: Windows MAX_PATH handling

Note: To avoid circular imports, use direct imports from submodules:
    from modules.core.cli_args import create_transcriber_parser
    from modules.core.workflow import WorkflowManager
"""
