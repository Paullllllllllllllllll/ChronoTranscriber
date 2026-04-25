"""CLI → UserConfiguration translation.

Extracts the configuration-synthesis helpers that previously lived in
``main/unified_transcriber.py``. Keeps the CLI entry point thin and
centralises every place where CLI arguments become a validated
:class:`UserConfiguration`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from modules.config.config_loader import PROJECT_ROOT
from modules.core.cli_args import (
    resolve_path,
    validate_input_path,
    validate_output_path,
)
from modules.documents.auto_selector import AutoSelector
from modules.llm.schema_utils import list_schema_options
from modules.transcribe.user_config import UserConfiguration
from modules.ui import print_info, print_warning


def _resolve_schema(args_schema: str | None, config: UserConfiguration) -> None:
    """Resolve and set schema on config from CLI ``--schema`` argument.

    If ``args_schema`` is provided, looks it up in available schemas;
    otherwise sets the default markdown_transcription_schema.
    """
    if args_schema:
        options = list_schema_options()
        schema_dict = {name: path for name, path in options}
        if args_schema in schema_dict:
            config.selected_schema_name = args_schema
            config.selected_schema_path = schema_dict[args_schema]
        else:
            raise ValueError(
                f"Schema '{args_schema}' not found. "
                f"Available: {list(schema_dict.keys())}"
            )
    else:
        default_schema = (
            PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
        ).resolve()
        config.selected_schema_name = "markdown_transcription_schema"
        config.selected_schema_path = default_schema


def _resolve_context(
    args_context: str | None, config: UserConfiguration
) -> None:
    """Resolve and set additional context path from CLI ``--context``."""
    if args_context:
        context_path = resolve_path(args_context, PROJECT_ROOT)
        validate_input_path(context_path)
        config.additional_context_path = context_path
    else:
        config.additional_context_path = None


def _resolve_model_config_from_cli(
    base_model_config: dict[str, Any],
    args: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Apply supported model-related CLI overrides onto a model-config copy."""
    effective_model_config = deepcopy(base_model_config)
    tm = effective_model_config.setdefault("transcription_model", {})
    applied_overrides: list[str] = []

    model = getattr(args, "model", None)
    if model:
        tm["name"] = model
        applied_overrides.append(f"model={model}")

    provider = getattr(args, "provider", None)
    if provider:
        tm["provider"] = provider
        applied_overrides.append(f"provider={provider}")
    elif model:
        from modules.llm.providers.factory import detect_provider_from_model

        inferred_provider = detect_provider_from_model(model).value
        tm["provider"] = inferred_provider
        applied_overrides.append(f"provider={inferred_provider} (auto)")

    max_output_tokens = getattr(args, "max_output_tokens", None)
    if max_output_tokens is not None:
        if int(max_output_tokens) <= 0:
            raise ValueError("--max-output-tokens must be a positive integer")
        tm["max_output_tokens"] = int(max_output_tokens)
        applied_overrides.append(f"max_output_tokens={int(max_output_tokens)}")

    reasoning_effort = getattr(args, "reasoning_effort", None)
    if reasoning_effort:
        reasoning_cfg = tm.get("reasoning")
        if not isinstance(reasoning_cfg, dict):
            reasoning_cfg = {}
        reasoning_cfg["effort"] = reasoning_effort
        tm["reasoning"] = reasoning_cfg
        applied_overrides.append(f"reasoning.effort={reasoning_effort}")

    model_verbosity = getattr(args, "model_verbosity", None)
    if model_verbosity:
        text_cfg = tm.get("text")
        if not isinstance(text_cfg, dict):
            text_cfg = {}
        text_cfg["verbosity"] = model_verbosity
        tm["text"] = text_cfg
        applied_overrides.append(f"text.verbosity={model_verbosity}")

        effective_provider = str(tm.get("provider", "")).lower()
        effective_model = str(tm.get("name", "")).lower()
        if not (
            effective_provider == "openai"
            and effective_model.startswith("gpt-5")
        ):
            print_warning(
                "--model-verbosity is currently effective only for "
                "OpenAI GPT-5 models (gpt-5, gpt-5-mini, gpt-5.1, "
                "gpt-5.2)."
            )

    return effective_model_config, applied_overrides


def _collect_files_for_type(
    input_path: Path,
    processing_type: str,
    args: Any,
) -> list[Path]:
    """Collect files/folders to process based on processing type and args."""
    if processing_type == "images":
        if not input_path.is_dir():
            raise ValueError(
                f"For image processing, input must be a directory: "
                f"{input_path}"
            )
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
    if args.files:
        return [input_path / f for f in args.files]
    elif args.recursive:
        return [f for ext in extensions for f in input_path.rglob(f"*{ext}")]
    else:
        return [f for ext in extensions for f in input_path.glob(f"*{ext}")]


def create_config_from_cli_args(
    args: Any,
    base_input_dir: Path,
    base_output_dir: Path,
    paths_config: dict[str, Any],
) -> UserConfiguration:
    """Build a :class:`UserConfiguration` from parsed CLI arguments."""
    from modules.documents.page_range import parse_page_range

    config = UserConfiguration()
    config.auto_selector = AutoSelector(paths_config)

    # Resume mode: CLI flags take precedence; fall back to paths_config
    config_resume_default = paths_config.get("general", {}).get(
        "resume_mode", "skip"
    )
    if getattr(args, "force", None):
        config.resume_mode = "overwrite"
    elif getattr(args, "resume", None):
        config.resume_mode = "skip"
    else:
        config.resume_mode = config_resume_default

    # Output format: CLI flag takes precedence; fall back to paths_config
    config_default = paths_config.get("general", {}).get(
        "output_format", "txt"
    )
    config.output_format = (
        getattr(args, "output_format", None) or config_default
    )

    # Parse page range if provided
    if getattr(args, "pages", None):
        config.page_range = parse_page_range(args.pages)

    # Handle auto mode
    if args.auto:
        auto_input = Path(
            paths_config.get("file_paths", {})
            .get("Auto", {})
            .get("input", base_input_dir)
        )
        auto_output = Path(
            paths_config.get("file_paths", {})
            .get("Auto", {})
            .get("output", base_output_dir)
        )

        if args.input:
            auto_input = resolve_path(args.input, auto_input)
        if args.output:
            auto_output = resolve_path(args.output, auto_output)

        validate_input_path(auto_input)
        validate_output_path(auto_output)

        decisions = config.auto_selector.create_decisions(auto_input)

        if not decisions:
            raise ValueError(
                f"No processable files found in auto mode input directory: "
                f"{auto_input}"
            )

        # Apply resume filtering to exclude already-completed items
        if config.resume_mode == "skip":
            from modules.infra.paths import PathConfig
            from modules.transcribe.resume import (
                ProcessingState,
                ResumeChecker,
            )

            pc = PathConfig.from_paths_config(paths_config)
            checker = ResumeChecker(
                resume_mode=config.resume_mode,
                paths_config=paths_config,
                use_input_as_output=pc.use_input_as_output,
                pdf_output_dir=pc.pdf_output_dir,
                image_output_dir=pc.image_output_dir,
                epub_output_dir=pc.epub_output_dir,
                mobi_output_dir=pc.mobi_output_dir,
                output_format=config.output_format,
            )
            total_before = len(decisions)
            decisions = [
                d
                for d in decisions
                if checker.should_skip(d.file_path, "auto").state
                != ProcessingState.COMPLETE
            ]
            skipped_count = total_before - len(decisions)
            if skipped_count:
                print_info(
                    f"Resume: skipping {skipped_count} already-completed "
                    f"file(s) ({len(decisions)} remaining)"
                )
            if not decisions:
                print_info("All files already processed. Nothing to do.")
                return config

        config.processing_type = "auto"
        config.auto_decisions = decisions

        config.selected_items = [auto_output]

        _resolve_schema(args.schema, config)
        _resolve_context(args.context, config)

        return config

    # Validate non-auto mode has required arguments
    if not args.type or not args.method:
        raise ValueError(
            "--type and --method are required unless using --auto mode"
        )

    config.processing_type = args.type
    config.transcription_method = args.method

    if (
        config.processing_type == "epubs"
        and config.transcription_method != "native"
    ):
        raise ValueError(
            "EPUB processing only supports the 'native' method."
        )
    if (
        config.processing_type == "mobis"
        and config.transcription_method != "native"
    ):
        raise ValueError(
            "MOBI processing only supports the 'native' method."
        )

    if args.method == "gpt":
        config.use_batch_processing = args.batch

        _resolve_schema(args.schema, config)
        _resolve_context(args.context, config)

    input_path = resolve_path(args.input, base_input_dir)
    validate_input_path(input_path)

    output_path = resolve_path(args.output, base_output_dir)
    validate_output_path(output_path)

    config.selected_items = _collect_files_for_type(
        input_path, config.processing_type, args
    )

    if not config.selected_items:
        raise ValueError(f"No items found to process in: {input_path}")

    return config


__all__ = [
    "_resolve_schema",
    "_resolve_context",
    "_resolve_model_config_from_cli",
    "_collect_files_for_type",
    "create_config_from_cli_args",
]
