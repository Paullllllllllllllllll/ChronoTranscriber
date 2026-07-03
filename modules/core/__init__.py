"""Core utilities package.

Provides CLI argument parsing for the entry points in ``main/``.

Submodules:
- cli_args: CLI argument parsers (create_transcriber_parser, etc.)

Related modules living outside this package:
- modules.transcribe.dual_mode: AsyncDualModeScript base (interactive vs CLI)
- modules.transcribe.manager: WorkflowManager for document processing
- modules.infra.token_budget: daily token limit + check_and_wait_for_token_limit
- modules.documents.auto_selector: auto-mode file detection

Note: To avoid circular imports, use direct imports from submodules:
    from modules.core.cli_args import create_transcriber_parser
    from modules.transcribe.manager import WorkflowManager
"""
