# ChronoTranscriber v1.0.0

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

All Python dependencies are listed in `requirements.txt`.

## Installation

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
cd ChronoTranscriber

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# For development and tests
pip install -r requirements-dev.txt
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
shell profile. Edit `config/paths_config.yaml` to set input/output
directories.

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
--model ID                 Override model
--provider NAME            openai | anthropic | google | openrouter
--reasoning-effort LEVEL   none | low | medium | high | xhigh
--output-format FORMAT     txt | md | json
--pages RANGE              e.g., '3-7', 'first:5', '1,3,5-8'
--resume / --force         Skip vs overwrite existing output
--files FILE ...           Process specific files
--recursive                Recurse into subdirectories
```

Run `python main/unified_transcriber.py --help` for the full list.

## Configuration

ChronoTranscriber uses four YAML files in `config/`. The config
directory can be overridden via the `CHRONO_CONFIG_DIR` environment
variable.

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
```

Key parameters: `provider` (auto-detected if omitted), `name` (model
identifier), `max_output_tokens` (must cover reasoning tokens on
reasoning models), `reasoning.effort` (OpenAI also supports `none`,
`minimal`, `xhigh`), `temperature`/`top_p` (applied only when the
model supports them).

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
total tokens per call, resets at local midnight. Add
`.chronotranscriber_token_state.json` to `.gitignore`.

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
Check logs in the configured `logs_dir`, validate configuration
files, and review `requirements.txt` for version mismatches. For
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
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m pytest -v
```

The suite contains 1,200+ tests (unit and integration) covering all
modules, providers, batch backends, and CLI parsers.

## Versioning

This project uses semantic versioning. The commit history was
squashed to a single baseline commit at v1.0.0 on 25 April 2026.
All prior development history was consolidated; version numbers
before v1.0.0 do not exist.

## License

MIT License. Copyright (c) 2025 Paul Goetz. See
[LICENSE](LICENSE) for details.
