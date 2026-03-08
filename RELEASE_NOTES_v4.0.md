# ChronoTranscriber v4.0 Release Notes

**Release date:** March 2026

## Overview

Version 4.0 streamlines ChronoTranscriber by removing ancillary tooling that
sat outside the core transcription pipeline, and introduces multi-format output
as the headline new feature. All core transcription, batch, repair, and
post-processing functionality remains unchanged.

## New features

### Multi-format output (`--output-format`)

Transcription results can now be written in three formats, selectable via the
`--output-format` CLI flag:

- **`txt`** (default) -- Plain text with pages joined by newline. Backward
  compatible with all prior versions.
- **`md`** -- Markdown with `## Page N` or `## image_name` headers before each
  page block.
- **`json`** -- Structured JSON array of page objects containing `page_number`,
  `image_name`, and `transcription` fields. No post-processing is applied,
  preserving the raw LLM output.

The format setting is threaded through the entire pipeline: CLI entry points,
`WorkflowManager`, `ResumeChecker`, transcription pipeline, and batch
finalization.

### Centralized output writer

A new `modules/processing/output_writer.py` module provides
`write_transcription_output()` and `resolve_output_path()`, replacing six
inline write-to-disk code paths with a single, format-aware writer.

## Internal changes

### Renamed module

`modules/processing/text_processing.py` has been renamed to
`modules/processing/response_parsing.py` to better reflect its purpose (API
response parsing and page formatting, not general text processing). All imports
throughout the codebase have been updated; the test file was likewise renamed to
`tests/unit/test_response_parsing.py`.

### Extension-aware ResumeChecker

`ResumeChecker` now accepts an `output_format` parameter and looks for the
correct file extension (`.txt`, `.md`, or `.json`) when determining whether an
item has already been processed. This ensures resume/skip behavior works
correctly for all output formats.

## Removed components

The following non-vital components have been deleted to reduce maintenance
burden:

| Component | Reason |
|-----------|--------|
| `main/cost_analysis.py` | Standalone cost analysis CLI tool |
| `modules/operations/cost_analysis.py` | Backend for cost analysis |
| `modules/ui/cost_display.py` | Display functions for cost analysis |
| `main/prepare_ground_truth.py` | Ground truth preparation CLI tool |
| `fine_tuning/` (entire directory) | Fine-tuning dataset preparation |
| `modules/diagnostics/` (entire directory) | System diagnostics |
| `modules/testing/fixtures.py` | Unused test fixtures |
| `tests/unit/test_cost_analysis.py` | Tests for removed module |
| `tests/unit/test_cost_display.py` | Tests for removed module |
| `tests/unit/test_system_check.py` | Tests for removed module |
| `RELEASE_NOTES_v2.0.md` | Legacy release notes |

**Kept:** The `eval/` directory is retained for replication purposes.

## Migration notes

- If you relied on `cost_analysis.py` for token cost tracking, note that the
  daily token tracker (`modules/infra/token_tracker.py`) remains available and
  provides runtime usage statistics.
- Import paths for `extract_transcribed_text`, `format_page_line`,
  `detect_transcription_cause`, and `process_batch_output` have changed from
  `modules.processing.text_processing` to `modules.processing.response_parsing`.
- The `--output-format` flag defaults to `txt`, so existing scripts and
  workflows require no changes.
