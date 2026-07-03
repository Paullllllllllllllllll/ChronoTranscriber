# ChronoTranscriber v1.16.0

A Python-based document transcription tool for researchers, archivists,
and digital humanities projects. ChronoTranscriber transforms historical
documents, academic papers, and ebooks into searchable, structured text
using state-of-the-art AI models or local OCR.

Designed to integrate with
[ChronoMiner](https://github.com/Paullllllllllllllllll/ChronoMiner) and
[ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader)
for a complete document retrieval, transcription, and data extraction
pipeline.

> **Work in Progress** -- ChronoTranscriber is under active development.
> If you encounter any issues, please
> [report them on GitHub](https://github.com/Paullllllllllllllllll/ChronoTranscriber/issues).

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Providers and Models](#supported-providers-and-models)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Batch Processing](#batch-processing)
- [Utilities](#utilities)
- [Architecture](#architecture)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [Development](#development)
- [Changelog](#changelog)
- [License](#license)

## Overview

ChronoTranscriber enables researchers and archivists to transcribe
historical documents at scale with minimal cost and effort. It supports
multiple AI providers through a unified LangChain-based architecture,
local OCR via Tesseract, and fine-grained control over image
preprocessing.

**Execution modes:**

- **Interactive** -- guided terminal wizard with back/quit navigation.
  Ideal for first-time users and exploratory workflows.
- **CLI** -- headless automation for scripting and CI/CD pipelines.
  Set `interactive_mode: false` in `config/paths_config.yaml` or pass
  arguments directly.

**Supported document types:**

- **PDFs** -- native text extraction or page-to-image OCR
- **Image folders** -- PNG, JPEG, WEBP, BMP, TIFF
- **EPUBs** -- native extraction from EPUB 2.0/3.0
- **MOBI/Kindle** -- unencrypted MOBI, AZW, AZW3, KFX
- **Auto mode** -- scan mixed directories and select the best method
  per file

## Key Features

- **Multi-provider LLM support** via LangChain (OpenAI, Anthropic,
  Google, OpenRouter, custom OpenAI-compatible endpoints)
- **Tesseract local OCR** -- fully offline processing with configurable
  preprocessing (grayscale, deskew, denoise, binarization)
- **Centralized capability registry** -- single source of truth for all
  provider/model capabilities; unsupported parameters filtered
  automatically before API calls
- **Hierarchical context resolution** -- file-specific, folder-specific,
  or project-wide transcription context
  (`{name}_transcr_context.txt` convention)
- **Context image support** -- include a reference image (title page,
  TOC, column headers) alongside each page image to improve
  transcription quality (`{name}_transcr_context_image.{ext}`
  convention; OpenAI provider)
- **Batch processing** -- async batch APIs for OpenAI, Anthropic, and
  Google with smart chunking (150 MB per chunk) and 50% cost savings
  on OpenAI
- **Multi-tier retry** -- exponential backoff for network errors;
  validation retries for malformed structured output; content-quality
  retries for hallucination loops, truncation, system-prompt bleed,
  and excessive line repetition
- **Daily token budget** -- configurable per-day limits with automatic
  midnight reset
- **Three output formats** -- `txt`, `md` (with page headers), `json`
  (structured per-page array)
- **Resume and repair** -- skip already-transcribed pages; repair
  individual failed pages after the fact
- **Custom transcription schemas** -- JSON schemas controlling output
  structure; four included, custom schemas supported

## Supported Providers and Models

Set the provider in `config/model_config.yaml` or let the system
auto-detect from the model name.

| Provider | Notable model families | Env variable | Batch |
|----------|----------------------|--------------|-------|
| OpenAI | GPT-5.4, GPT-5.3, GPT-5.2, GPT-5.1, GPT-5, o-series, GPT-4.1, GPT-4o | `OPENAI_API_KEY` | Yes |
| Anthropic | Claude 4.7, 4.6, 4.5, 4.1, 4, 3.7, 3.5 | `ANTHROPIC_API_KEY` | Yes |
| Google | Gemini 3.1, 3, 2.5, 2.0, 1.5; Gemma 4 | `GOOGLE_API_KEY` | Yes |
| OpenRouter | 200+ models via unified API | `OPENROUTER_API_KEY` | No |
| Custom | Any OpenAI-compatible endpoint | User-configured | No |

### Custom OpenAI-Compatible Endpoint

Connect to any self-hosted or third-party endpoint implementing the
OpenAI Chat Completions API. Set `provider: custom` in
`model_config.yaml` and configure the `custom_endpoint` block:

```yaml
transcription_model:
  provider: custom
  name: "org/model-name"
  custom_endpoint:
    base_url: "https://your-endpoint.example.com/v1"
    api_key_env_var: "CUSTOM_API_KEY"
    use_plain_text_prompt: false
    capabilities:
      supports_vision: true
      supports_structured_output: false
```

Three operating modes are available, controlled by
`supports_structured_output` and `use_plain_text_prompt`:

| Mode | Configuration | Use when |
|------|--------------|----------|
| Structured | `supports_structured_output: true` | Endpoint supports JSON schema enforcement |
| JSON-instructed (default) | `supports_structured_output: false`, `use_plain_text_prompt: false` | Model follows JSON instructions without API-level enforcement |
| Plain text | `supports_structured_output: false`, `use_plain_text_prompt: true` | Model works best with simple text instructions |

## System Requirements

- **Python** 3.10+ (3.13 recommended)
- **Tesseract OCR** (optional) -- required only for local OCR
- **FFmpeg** (optional) -- required for JPEG2000 bilevel codestreams
- At least one API key (see provider table above)

All Python dependencies are declared in `pyproject.toml` and locked
in `uv.lock`.

## Installation

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
cd ChronoTranscriber

# Install uv if not already available
pip install uv

# Runtime dependencies only
uv sync

# Include development and test tools
uv sync --extra dev

# Include evaluation notebook dependencies
uv sync --extra eval
```

**Install Tesseract** (optional, for local OCR):

- Windows: [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki);
  configure path in `image_processing_config.yaml`
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

**Install FFmpeg** (optional, for JPEG2000 bilevel codestreams):

- Windows: [ffmpeg.org](https://ffmpeg.org/download.html), add `bin/`
  to PATH
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

**Configure API keys:**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# Linux/macOS
export OPENAI_API_KEY="your_key_here"
```

For persistent configuration, add to system environment variables or
shell profile.

**Configure your settings (optional for a quick start):**

The `config/` directory ships with scrubbed `*.example.yaml` templates.
On a fresh clone, the loader reads those templates automatically and
prints a one-line notice. To set your own paths and model:

```bash
cp config/model_config.example.yaml config/model_config.yaml
cp config/paths_config.example.yaml config/paths_config.yaml
# edit both files
```

The real `*.yaml` files are gitignored and never pushed; only the
`*.example.yaml` templates are tracked.

## Quick Start

### Your First Transcription

**Interactive mode** (recommended for new users):

```bash
python main/unified_transcriber.py
```

The wizard guides you through document type, method, processing options,
and file selection. Press `b` to go back, `q` to quit at any time.

**CLI mode:**

```bash
# Transcribe a PDF with AI
python main/unified_transcriber.py --type pdfs --method gpt \
    --input ./documents/my_doc.pdf --output ./results

# Process images with Tesseract (offline)
python main/unified_transcriber.py --type images --method tesseract \
    --input ./scans --output ./results

# Batch process PDFs (50% cheaper)
python main/unified_transcriber.py --type pdfs --method gpt --batch \
    --input ./archive --output ./results
```

### Common Workflows

**Large-scale batch processing:**

```bash
# Submit
python main/unified_transcriber.py --type pdfs --method gpt --batch \
    --input ./archive --output ./results
# Monitor (run periodically; auto-downloads on completion)
python main/check_batches.py
```

**Mixed document types (auto mode):**

```bash
python main/unified_transcriber.py --auto \
    --input ./mixed_documents --output ./results
```

**EPUB and MOBI extraction:**

```bash
python main/unified_transcriber.py --type epubs --method native \
    --input ./ebooks --output ./results
python main/unified_transcriber.py --type mobis --method native \
    --input ./kindle_books --output ./results
```

**Repair failed pages:**

```bash
python main/repair_transcriptions.py \
    --transcription ./results/document_transcription.txt --errors-only
```

### CLI Reference

```
--input / --output         Input and output paths
--type                     pdfs | images | epubs | mobis
--method                   native | tesseract | gpt
--auto                     Auto mode (bypasses --type/--method)
--batch                    Use async batch API
--schema NAME              JSON schema selection
--context PATH             Override context file
--context-image PATH       Context image for each page
--model ID                 Override model
--provider NAME            openai | anthropic | google | openrouter
--reasoning-effort LEVEL   none | low | medium | high | xhigh
--output-format FORMAT     txt | md | json
--pages RANGE              e.g., '3-7', 'first:5', '1,3,5-8'
--resume / --force         Skip vs overwrite existing output
--retry-errors             Re-process pages left as '[transcription error]'
--sync-fallback            On batch-submit failure, fall back to sync (default off)
--files FILE ...           Process specific files
--recursive                Recurse into subdirectories
--interactive / --non-interactive   Override the config-file mode
--dry-run                  Report planned actions; no API calls or writes
--json                     Emit a machine-readable JSON summary line on stdout
```

Run `python main/unified_transcriber.py --help` for the full list.

### Exit Codes and Automation

All primary entry points follow a uniform CLI agent contract:

- `0` full success; `1` one or more items failed or partial; `2` usage or
  configuration error; `130` interrupted by the user.
- `--json` prints one JSON summary line on stdout (items total / processed /
  failed) for machine consumption.
- In interactive mode without a TTY the tool exits `2` with a clear message
  rather than hanging or reporting a false success; drive it with
  `--non-interactive` plus CLI arguments instead.
- `check_batches` exits non-zero when a batch reached a terminal failure;
  `cancel_batches` exits non-zero when any cancellation failed.

### Batch Submission Splitting

Batch jobs are automatically split into parts under each provider's request-count
and byte limits, so a large book is never submitted as one oversized batch. Every
part's id is recorded for retrieval. A failed submission exits non-zero rather
than silently reprocessing the whole job synchronously at full price; pass
`--sync-fallback` to opt into the old fall-back behavior.

## Configuration

ChronoTranscriber uses four YAML files in `config/`, plus one optional
fifth file (`api_keys_config.yaml`). The config directory can be
overridden via the `CHRONO_CONFIG_DIR` environment variable.

**Example/real split.** Every config file has a tracked, scrubbed
`<name>.example.yaml` sibling. The loader resolves config in this order:

1. Load `<name>.yaml` if present (your private settings, gitignored).
2. Fall back to `<name>.example.yaml` with a one-line INFO notice
   telling you to copy and customize the file.
3. Raise a clear error if neither file exists.

A fresh clone therefore runs with sane defaults instead of crashing.
Copy the example files to their real names only when you need to
override the defaults.

### 1. Model Configuration (`model_config.yaml`)

```yaml
transcription_model:
  provider: openai       # openai | anthropic | google | openrouter | custom
  name: gpt-5-mini
  max_output_tokens: 128000
  reasoning:
    effort: medium       # Cross-provider preset (low | medium | high)
  temperature: 0.01
  top_p: 1.0
  user_instruction: "The image:"          # text block before page image
  context_image_instruction: "Context image:"  # text block before context image
```

Key parameters: `provider` (auto-detected if omitted), `name` (model
identifier), `max_output_tokens` (must cover reasoning tokens on
reasoning models), `reasoning.effort` (OpenAI also supports `none`,
`minimal`, `xhigh`), `temperature`/`top_p` (applied only when the
model supports them), `user_instruction` (text sent alongside each
page image; set to `""` to omit the text block entirely for models
that expect image-only input), `context_image_instruction` (label
for the optional context image; independently configurable).

### 2. Paths Configuration (`paths_config.yaml`)

```yaml
general:
  interactive_mode: true
  output_format: 'txt'        # txt | md | json
  resume_mode: 'skip'         # skip | overwrite
file_paths:
  PDFs:
    input: './input/pdfs'
    output: './output/pdfs'
  # Images, EPUBs, MOBIs, Auto sections follow the same pattern
```

Controls execution mode, output format, resume behavior, and per-type
input/output directories. Auto-mode settings
(`auto_mode_pdf_use_ocr_for_scanned`, etc.) configure per-file method
selection.

### 3. Image Processing Configuration (`image_processing_config.yaml`)

Provider-aware sections (`api_image_processing`, `google_image_processing`,
`anthropic_image_processing`, `tesseract_image_processing`,
`custom_image_processing`) configure preprocessing per backend: target
DPI, grayscale conversion, transparency handling, resize profiles,
JPEG quality, and detail/resolution levels.

The `postprocessing` block controls text cleanup after transcription:
Unicode normalization, optional hyphenation merging, whitespace
collapsing, blank-line capping, and line wrapping.

### 4. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  transcription:
    concurrency_limit: 20
    retry:
      attempts: 5              # Network retries with exponential backoff
      validation_attempts: 3   # Retries for malformed output + quality
      min_input_tokens: 500    # Cross-contamination detection threshold
      content_quality:
        enabled: true          # Hallucination, truncation, bleed, loop detection
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

Controls concurrency limits, retry strategy (network and
validation/quality retries share separate budgets), content-quality
validators with configurable thresholds, service tier, batch chunk
size, and daily token budgets.

### 5. API Keys Configuration (Optional) (`api_keys_config.yaml`)

```yaml
openai: OPENAI_API_KEY
anthropic: ANTHROPIC_API_KEY
google: GOOGLE_API_KEY
openrouter: OPENROUTER_API_KEY
```

Maps each provider to the name of the environment variable holding its
API key, letting you swap keys between runs by editing one file (for
example `openai: OPENAI_API_KEY_2`) instead of changing the environment.
The values are environment variable names, never the secret keys
themselves. This file is entirely optional and backward-compatible: when
it is absent, or when a provider entry is omitted, the default env var
name shown above applies. The remap is honored everywhere a key is read,
including batch mode. The custom provider's env var name is configured
separately via `custom_endpoint.api_key_env_var` in `model_config.yaml`.

### Context Resolution

Hierarchical context resolution automatically selects the most
specific transcription guidance available:

1. **File-specific**: `{input_stem}_transcr_context.txt` next to the
   input file
2. **Folder-specific**: `{parent_folder}_transcr_context.txt` next to
   the input's parent folder
3. **General fallback**: `context/transcr_context.txt` in the project
   root

Context files should be plain text describing the document type,
expected content, formatting conventions, and any domain-specific
terminology. Keep under 4,000 characters.

**Context images** follow the same hierarchy but use image files:

1. **File-specific**: `{input_stem}_transcr_context_image.{ext}`
2. **Folder-specific**: `{parent_folder}_transcr_context_image.{ext}`
3. **General fallback**: `context/transcr_context_image.{ext}`

A context image (e.g., a title page, table of contents, or column
headers) is sent alongside each page image in the user message,
giving the LLM visual reference material. Supported on the OpenAI
provider; other providers accept the parameter but ignore it.
Use `--context-image PATH` to override with a specific file.

### Custom Transcription Schemas

Place JSON schemas in `schemas/`. Included schemas:

- `markdown_transcription_schema.json` (default) -- Markdown with LaTeX
- `plain_text_transcription_schema.json` -- plain text
- `plain_text_transcription_with_markers_schema.json` -- plain text
  with `<page_number>` tags

All schemas require `transcription`, `no_transcribable_text`, and
`transcription_not_possible` fields. Select with `--schema` or via
the interactive wizard.

## Output Formats

Three formats via `--output-format` (default set in `paths_config.yaml`):

- `txt` -- plain text, one page per block
- `md` -- Markdown with `## Page N` headers
- `json` -- structured JSON array with per-page metadata

Output files are named `<original_name>_transcription.{ext}`.

## Batch Processing

Async batch APIs for OpenAI, Anthropic, and Google. OpenAI offers
50% cost savings. OpenRouter and custom endpoints do not support
batch mode.

**How it works:**

1. Images are base64-encoded as data URLs
2. Requests are split into chunks (max 150 MB each)
3. Chunks are submitted as separate batch jobs with metadata tracking
4. A debug artifact (`*_batch_submission_debug.json`) is saved for
   repair and status recovery

**Monitoring and cancellation:**

```bash
# Check status, auto-download completed results
python main/check_batches.py

# Cancel all non-terminal batch jobs
python main/cancel_batches.py
```

Batch processing typically completes within 24 hours.

## Utilities

### Repair Transcriptions

Re-transcribe failed or selected pages within an existing output:

```bash
# Repair API errors only
python main/repair_transcriptions.py \
    --transcription ./results/doc_transcription.txt --errors-only

# Repair specific page indices
python main/repair_transcriptions.py \
    --transcription ./results/doc_transcription.txt --indices 5,12,18
```

### Post-process Transcriptions

Run the text cleanup pipeline (Unicode normalization, hyphenation
merging, whitespace normalization, line wrapping) on existing output:

```bash
python main/postprocess_transcriptions.py \
    --input-dir ./results
```

### Daily Token Budget

Enable in `concurrency_config.yaml` to cap daily API usage. Tracks
total tokens per call, resets at local midnight. The counter is persisted
under a user-level state directory (`~/.chronotranscriber/token_state.json`
by default), so it is shared across runs regardless of the working
directory. Override the location with `general.state_dir` in
`paths_config.yaml`; a legacy per-directory
`.chronotranscriber_token_state.json` is adopted once if present.

## Architecture

ChronoTranscriber follows a deep-module architecture: nine packages
under `modules/`, each with a narrow public surface, composed by CLI
entry points in `main/`.

```
modules/
+-- batch/         Provider-agnostic batch operations
+-- config/        YAML config, capability registry, context resolution
|   +-- capabilities/
+-- core/          CLI parser factories
+-- documents/     PDF / EPUB / MOBI loaders, auto-selector, PageRange
+-- images/        Image preprocessing pipeline, encoding, Tesseract
+-- infra/         Logging, token budget, paths, concurrency, progress
+-- llm/           Provider abstraction, transcriber, schemas, quality
+-- postprocess/   Text cleanup, output writer (txt/md/json)
+-- transcribe/    Workflow manager, pipeline, resume, dual-mode script
+-- ui/            Interactive prompts, batch display, workflow wizard

main/
+-- unified_transcriber.py      Primary entry point
+-- check_batches.py            Monitor and finalize batch jobs
+-- cancel_batches.py           Cancel non-terminal batch jobs
+-- repair_transcriptions.py    Re-transcribe failed pages
+-- postprocess_transcriptions.py  Standalone post-processing
```

Provider integration flows through `modules/llm/providers/` (factory
with auto-detection, per-provider implementations) and
`modules/config/capabilities/` (registry, detection, parameter
gating).

## Frequently Asked Questions

**Which AI provider should I choose?**
Depends on priorities. OpenAI `gpt-5-mini` offers the best
cost/quality balance with a 50% batch discount. Google Gemini Flash
is fastest and cheapest. Anthropic Claude excels with complex layouts.
OpenRouter provides access to 200+ models with a single key. Start
with OpenAI `gpt-5-mini` at low reasoning effort.

**How much does transcription cost?**
With OpenAI `gpt-5-mini`: roughly $0.01--0.02 per page (sync),
$0.005--0.01 per page (batch). A 100-page PDF costs around $1--2
synchronous or $0.50--1 in batch mode.

**Batch or synchronous?**
Use batch for 50+ pages when you can wait up to 24 hours. Use
synchronous for immediate results, small jobs, or testing.

**Can I process documents offline?**
Yes, use `--method tesseract`. Quality is generally lower than AI
models but requires no API key or internet connection.

**How do I switch providers?**
Edit `config/model_config.yaml` and set the appropriate environment
variable. Provider can also be auto-detected from the model name.

**What happens when pages fail?**
Failed pages are marked with error placeholders in the output.
Use `repair_transcriptions.py --errors-only` to re-transcribe only
the failures.

**Can I process password-protected PDFs?**
No. Decrypt them first using external tools.

**How do I integrate into existing pipelines?**
Use CLI mode (`interactive_mode: false`). All scripts return proper
exit codes suitable for shell scripting and CI/CD.

**I'm experiencing issues not covered here.**
Check logs in the configured `logs_dir` and validate configuration
files. For
persistent issues, open a
[GitHub issue](https://github.com/Paullllllllllllllllll/ChronoTranscriber/issues)
with error details and relevant config sections.

## Contributing

Contributions are welcome. When reporting issues, include: a clear
description, steps to reproduce, expected vs. actual behavior, your
environment (OS, Python version), relevant config sections (remove
sensitive data), and log excerpts.

For code contributions: fork the repository, create a feature branch,
follow the existing code style, add tests, and submit a pull request.
Test with both Tesseract and at least one AI backend.

## Development

Install dev dependencies:

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run python -m pytest -v
```

The suite contains 1,250+ tests (unit and integration) covering all
modules, providers, batch backends, and CLI parsers.

## Versioning

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The commit history was squashed to
a single baseline commit at v1.0.0 on 25 April 2026; version numbers before
v1.0.0 do not exist.

## Changelog

- **v1.16.0** (3 July 2026) -- Concurrency and token-budget hardening.
    Gate the synchronous repair path on the daily token budget with the same
    drain/wait/re-pass behavior as the main pipeline; move token-state
    persistence to a debounced background writer with per-process-unique
    temp files and race-tolerant retries (no more disk I/O or sleeps on the
    event loop); make the tenacity loop the single retry authority (SDK
    retries disabled, status-code-first classification, HTTP `Retry-After`
    honored, default 8 attempts with a 120 s cap); count Anthropic
    prompt-cache creation and read tokens at full weight in the daily budget
    and recover token usage from failed attempts; add a per-provider
    multi-window rate limiter with adaptive backoff
    (`modules/infra/rate_limit.py`, `concurrency.rate_limits`); run
    Tesseract OCR off the event loop; replace eager task creation with
    bounded lazy submission and make streaming failures cancel producer and
    workers cleanly; re-read `daily_token_limit.daily_tokens` during the
    wait-at-limit loop; implement real provider client teardown; document
    `image_processing.concurrency_limit` and fix stale module references in
    the example configs.

- **v1.15.0** (2 July 2026) -- Hardening release closing the silent-page-loss
    and batch-integrity defects found in a full production audit. OpenAI batch
    error files are now always parsed and reconciled against the submitted
    custom_id map, so failed pages surface as explicit `[transcription error]`
    placeholders instead of vanishing from final outputs; expired and cancelled
    batches are treated as terminal instead of polling forever; batch repair
    correlates results by request index rather than position. The Tesseract
    pipeline gains the absolute-page-order and regenerate-from-JSONL resume
    semantics the GPT streaming path received in v1.7.0, image folders sort
    naturally (`page_2` before `page_10`) via one shared key, and temp JSONLs
    carry a resume-format version that refuses incompatible pre-fix artifacts.
    Oversized batch submissions are split into provider-limited parts, and the
    silent full-price synchronous fallback is now opt-in via `--sync-fallback`.
    Content-quality validators no longer flag the schema's own `![Image: ...]`
    markers, run inside the retry loop, and default to the conservative example
    thresholds. All entry points adopt an agent-friendly CLI contract: exit
    codes 0/1/2/130, a `--json` run summary, `--dry-run`, `--interactive`/
    `--non-interactive` overrides, a non-TTY guard, and a `--retry-errors`
    resume mode. Token-budget state moves to a user-level directory
    (configurable via `general.state_dir`) with one-time legacy adoption; EXIF
    orientation, palette-PNG transparency, embedded DPI after downscaling, and
    EPUB spine ordering are fixed; JSON artifacts write `ensure_ascii=False`;
    the ruff backlog is cleared.

- **v1.14.0** (28 June 2026) -- Ship scrubbed `*.example.yaml` config templates
    with conservative OpenAI defaults and a real->example loader fallback, so a
    fresh clone runs with clear guidance instead of crashing on missing config.
    Each of the five config files now has a tracked `<name>.example.yaml` sibling
    in `config/`; the real `*.yaml` files remain gitignored. The loader tries the
    real file first, falls back to the example with a one-line INFO notice if it is
    absent, and raises a clear error only when neither file exists. The
    `api_keys_config` loader retains its non-raising behavior (returns `{}` when
    both files are absent). The `.gitignore` pattern is updated from `/config/` to
    `/config/*` plus `!/config/*.example.yaml` so examples are tracked while real
    configs stay private.

- **v1.13.0** (28 June 2026) -- Add optional `api_keys_config.yaml` for
    per-provider API-key environment-variable remapping. Each provider can be
    pointed at a custom env var name (for example `openai: OPENAI_API_KEY_2`) to
    swap keys between runs by editing one file; a missing file or omitted
    provider entry falls back to the existing default env var name, so behavior
    is unchanged for current setups. The remap is honored uniformly across the
    sync pipeline, the wizard validation gate, repair, diagnostics, and the
    batch backends, so it applies in batch mode too.

- **v1.12.0** (24 June 2026) -- The daily token limit is now enforced at the
    page level, not just between files. When the limit is enabled, the
    synchronous (GPT) streaming pipeline reserves a self-calibrating estimate
    of per-page token usage before each page, so concurrent workers cannot
    collectively overshoot; once the budget is exhausted mid-file it drains
    in-flight pages, waits for the daily reset, and re-streams the still-pending
    pages from the JSONL resume record. Configured concurrency and per-task
    delay are unchanged when budget is plentiful. Batch mode is now fully exempt
    from token limiting (it is pre-priced and submitted whole). Two optional
    `daily_token_limit` settings tune the estimate (`chunk_estimate_seed`,
    `estimate_smoothing`). All 1287 tests pass.

- **v1.11.0** (21 June 2026) -- Adopted mypy 2.x for static type checking and made
    `mypy .` runnable. Raised the dev pin to `mypy>=2.1`; fixed the config so the
    `__init__.py`-less `main/` no longer resolves twice (`explicit_package_bases`,
    `namespace_packages`, `mypy_path`, and an `exclude` scoping checks to source).
    Added one missing return annotation and three targeted `arg-type` ignores for
    the langchain `HumanMessage` content stub. The source type-checks clean under
    mypy 2.1.0 and all 1,279 tests pass.

- **v1.10.0** (21 June 2026) -- Adopted the google-genai 2.x SDK major.
    Raised the runtime pin from `google-genai>=1.73` to `google-genai>=2` and
    refreshed the lockfile (`google-genai` 1.73.1 -> 2.9.0;
    `langchain-google-genai` unchanged). The Google batch backend imports clean
    and all 1,279 tests pass. Live Google batch API calls are not exercised by
    the test suite; validate a real Google run before relying on it.

- **v1.9.0** (20 June 2026) -- Consolidated six within-module duplication clusters
    behind new private helpers, leaving every public interface and runtime behavior
    unchanged. In `modules/llm/providers/base.py` the three retry-config loaders
    and the content-quality config getter now share a single `_load_retry_config`
    helper. `main/cancel_batches.py` extracts the repeated batch id/status
    normalization into `_extract_batch_id_and_status`.
    `modules/batch/requests.py` folds the two identical submit-and-cleanup blocks
    into `_submit_and_cleanup_batch_file`. `modules/images/pipeline.py` shares its
    longest-side downscale logic via `_cap_longest_side`.
    `modules/batch/backends/google_backend.py` routes both JSONL and inline result
    branches through `_apply_json_content`. `modules/llm/transcriber.py`
    centralizes the common provider transcribe keyword arguments in
    `_transcribe_kwargs`. The empty confirmed dead-code list left nothing to remove.

- **v1.8.0** (20 June 2026) -- Refreshed dependencies under the conservative,
    majors-gated policy. Added `httpx>=0.28` as an explicit runtime dependency,
    since `modules/llm/providers/base.py` imports it directly for the connection
    and timeout exceptions in its retry logic while it was previously only
    transitive. Upgraded the LangChain stack (`langchain-core` to 1.4.8,
    `langchain-openai` to 1.3.2, `langchain-anthropic` to 1.4.6,
    `langchain-google-genai` to 4.2.5), the direct SDKs `openai` (2.43.0) and
    `anthropic` (0.111.0), plus `deskew` (1.6.1), `numpy` (2.4.6), and `lxml`
    (6.1.1) on the runtime side. In the dev and eval groups, raised `ruff`
    (0.15.18), `pytest` (9.1.1), `pytest-asyncio` (1.4.0), `coverage` (7.14.1),
    the type stubs for aiofiles and PyYAML, `pandas` (3.0.3), and `matplotlib`
    (3.11.0). Held two major bumps: `google-genai` stays on 1.73.1 (2.9.0
    withheld) and `mypy` stays on 1.20.2 (2.1.0 withheld), as each had no
    within-major release available. No dependencies were removed, since the deptry
    unused flags are all package-versus-module name-mapping false positives.

- **v1.7.0** (10 June 2026) -- Introduced a streaming in-memory image pipeline for
    all GPT paths (synchronous and batch, PDFs and image folders): pages are
    rendered, preprocessed, and base64-encoded fully in memory, and the
    `preprocessed_images/` folder is no longer written for GPT runs (Tesseract is
    unchanged), with peak memory being one raw page plus the payloads in flight.
    Page-level resume and page-range slicing are now applied before any rendering,
    so resuming a mostly-complete PDF no longer re-renders every page; virtual
    image names keep the historical `*_pre_processed.jpg` pattern so old partial
    JSONLs resume cleanly. Reproducibility provenance was added: each transcription
    record carries `image_provenance` (SHA-256 of the sent JPEG bytes, dimensions,
    byte size, effective DPI) plus `source_file`/`page_index`, and each run writes
    a `file_provenance` record (source SHA-256, PyMuPDF/Pillow versions,
    image-config snapshot). Repair gains an in-memory re-render fallback: when no
    preprocessed image exists on disk, failed pages are re-rendered from the
    recorded source PDF page or source image and repaired from base64 (sync and
    batch modes). Final output for streaming runs is regenerated from the complete
    JSONL so pages completed in earlier resumed runs are included; `order_index` is
    now the absolute page index, fixing page renumbering under page ranges and
    resume. `max_pixels_per_page` default was lowered from 150,000,000 to
    24,000,000, bounding the worst-case raw page to roughly 72 MB RGB while
    staying 2.3x above the 10.24 MP `original_max_pixels` send cap. The now-dead
    disk pipeline was removed: `PDFProcessor.process_images` / `extract_images`,
    `ImageProcessor.process_image` / `process_and_save_images` /
    `process_images_multiprocessing`; `keep_preprocessed_images` now affects only
    Tesseract folders.

- **v1.6.0** (30 May 2026) -- Correctness fixes from a full code review: the
    completion summary now reports real success/failure counts,
    `process_selected_items` returns a `ProcessingSummary`, failed items are no
    longer also counted as processed, and interactive mode stops hardcoding zero
    failures. Both PDF extraction paths now raise when the per-page failure rate
    exceeds the existing image threshold instead of silently returning a short,
    possibly page-misaligned image list. The interactive resume preview now passes
    `output_format` and the resolved output directories to `ResumeChecker` so skip
    counts are accurate for `md`/`json` output rather than always assuming `.txt`.
    The Anthropic batch backend now uses the capability registry
    (`detect_capabilities`) to decide whether to send `temperature`, replacing
    brittle model-name substring matching that drifted from the sync providers.
    Failures are no longer silently swallowed: provider `_invoke_llm` handlers log
    full tracebacks, the JSONL diagnostic-context builder logs instead of passing,
    the JPEG draft fast-path and Tesseract checks use narrowed exceptions, and the
    token-state retry backoff is honored instead of skipped inside a running event
    loop. `parse_indices` rejects negative/open-ended tokens with a clear message
    instead of a confusing range-parse error. Dead-code and duplication cleanup:
    removed the unused `ru_*` aliases and pass-through wrappers in batch repair,
    consolidated the three per-backend image-encoding helpers onto the shared
    `modules.images.encoding` functions, extracted the duplicated EPUB and MOBI
    text-normalization helper into `modules.documents._text`, and removed a dead
    `media_resolution` expression in the Google provider.

- **v1.5.0** (21 May 2026) -- Added configurable `user_instruction` and
    `context_image_instruction` keys under `transcription_model` in
    `model_config.yaml`; when set to an empty string the text block is omitted
    entirely from the user message, sending image-only input, which is required for
    models like `churro-3B` (Qwen2.5-VL fine-tune) that expect no accompanying
    text. Both sync and batch paths respect the new keys across all five providers
    (OpenAI, Anthropic, Google, OpenRouter, Custom) and all four batch backends.
    Fixed pre-existing test failures in `TestCacheTokenExtraction` where the mock
    provider's content quality validator received a MagicMock config instead of a
    dict, triggering false hallucination-loop detection on short test content.

- **v1.4.0** (20 May 2026) -- Added `--output-mode {hash,mirror}` CLI flag: mirror
    mode replicates the input directory hierarchy under the output root, preserving
    edition/page structure for downstream consumers. Fixed a hash collision in
    non-colocated output mode: the directory hash now incorporates the full
    relative path from the input root instead of just the leaf folder name,
    preventing overwrites when multiple editions share page numbers. The resume
    checker supports both mirror mode and relative-path-aware hash lookups.

- **v1.3.1** (19 May 2026) -- Dependency refresh from an environment-wide CVE audit:
    bumped `langchain-core` 1.3.2 -> 1.4.0 (RCE on deserialization); `langsmith`
    0.7.36 -> 0.8.5 (unsafe deserialization; full fix to 1.0.x deferred pending
    upstream constraint relaxation); `pillow` 12.1.1 -> 12.2.0 (FITS GZIP
    decompression bomb); `jupyterlab` 4.5.6 -> 4.5.7 and `notebook` 7.5.5 ->
    7.5.6 (one-click command execution chain); `jupyter-server` 2.17.0 -> 2.18.2
    (persistent cookie secret); `urllib3` 2.6.3 -> 2.7.0 (audit-surface
    consolidation); `deskew` downgraded 1.6.0 -> 1.5.3 as a side effect of
    relaxing the `pillow<12.2` peer constraint. Fixed
    `tests/integration/test_live_api.py` scripted-input drift introduced by
    v1.3.0: inserted one response for the new `configure_additional_context_image`
    prompt so the `GPT_PDF_RESPONSES` sequence matches the post-v1.3.0 workflow.

- **v1.3.0** (5 May 2026) -- Added context image support: a reference image (title
    page, table of contents, column headers) can now be included alongside each
    page image to improve transcription quality, using the same hierarchical
    resolution as text context (`{name}_transcr_context_image.{ext}` convention).
    Added a `--context-image` CLI flag to override the context image path and an
    interactive wizard prompt for context image selection. Supported on the OpenAI
    provider (sync and batch paths); other providers accept the parameter for
    interface compatibility.

- **v1.2.1** (5 May 2026) -- Fixed a circular import cycle between
    `modules.documents`, `modules.images`, `modules.ui`, and `modules.transcribe`
    that prevented startup: `WorkflowUI` is now lazily imported in
    `modules.ui.__init__` and the `AutoSelector` import in `config_builder.py` is
    deferred to function scope. Fixed test-induced directory pollution:
    `WorkflowManager` integration tests now provide `tmp_path`-based `file_paths`
    instead of relying on relative-path defaults that created `epubs_out`,
    `images_out`, `mobis_out`, `pdfs_out` in the project root.

- **v1.2.0** (4 May 2026) -- Applied ruff linter and formatter across entire
    codebase.

- **v1.1.0** (4 May 2026) -- Version bump consolidating post-baseline development.

- **v1.0.1** (25 April 2026) -- Migrated to `pyproject.toml` and updated
    dependencies; fixed test artifacts polluting the project root (initial pass).

- **v1.0.0** (25 April 2026) -- Repository baseline: squashed history into single
    commit.

## License

MIT License. Copyright (c) 2025 Paul Goetz. See
[LICENSE](LICENSE) for details.
