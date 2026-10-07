# ChronoTranscriber v4.4.2

A Python-based document transcription tool for researchers, archivists,
and digital humanities projects. ChronoTranscriber transforms historical
documents, academic papers, ebooks, and audio recordings into searchable,
structured text using state-of-the-art AI models or local OCR.

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
- [Audio Transcription](#audio-transcription)
- [Batch Processing](#batch-processing)
- [Utilities](#utilities)
- [Architecture](#architecture)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [Development](#development)
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
- **Audio recordings** -- MP3, WAV, M4A, MP4, FLAC, OGG, and more, via a
  speech-to-text API or a local Whisper model
- **Auto mode** -- scan mixed directories and select the best method
  per file

## Key Features

- **Native scan resolution** -- renders each scanned PDF page at the density of
  its scan image and sizes payloads to each model's documented limits, with
  optional lossless PNG, payload size guards, and recorded image settings
- **Multi-provider LLM support** via LangChain (OpenAI, Anthropic,
  Google, OpenRouter, custom OpenAI-compatible endpoints)
- **Tesseract local OCR** -- fully offline processing with configurable
  preprocessing (grayscale, deskew, denoise, binarization)
- **Audio transcription** -- speech recordings to plain text via the
  OpenAI audio API, Google Gemini, or a local faster-whisper model;
  long recordings are chunked automatically with per-chunk resume
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
  Google with smart chunking under each provider's request-count and
  byte limits, and 50% cost savings on OpenAI
- **Multi-tier retry** -- exponential backoff for network errors;
  validation retries for malformed structured output; content-quality
  retries for hallucination loops, truncation, system-prompt bleed,
  and excessive line repetition
- **Daily token budget** -- configurable per-day limits that reset at
  00:01 UTC (one minute after OpenAI's 00:00 UTC free-tier reset)
- **Three output formats** -- `txt`, `md` (with page headers), `json`
  (structured per-page array)
- **Resume and repair** -- skip already-transcribed pages; repair
  individual failed pages after the fact
- **Custom transcription schemas** -- JSON schemas controlling output
  structure; three included, custom schemas supported

## Supported Providers and Models

Set the provider in `config/model_config.yaml` or let the system
auto-detect from the model name.

| Provider | Notable model families | Env variable | Batch |
|----------|----------------------|--------------|-------|
| OpenAI | GPT-5.6, GPT-5.5, GPT-5.4, GPT-5.3, GPT-5.2, GPT-5.1, GPT-5, o-series, GPT-4.1, GPT-4o | `OPENAI_API_KEY` | Yes |
| Anthropic | Claude 5 (Fable, Sonnet), 4.8, 4.7, 4.6, 4.5, 4.1, 4, 3.7, 3.5 | `ANTHROPIC_API_KEY` | Yes |
| Google | Gemini 3.5, 3.1, 3, 2.5, 2.0, 1.5; Gemma 4 | `GOOGLE_API_KEY` | Yes |
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

- **Python** 3.13+ (matches `requires-python` in `pyproject.toml`)
- **Tesseract OCR** (optional) -- required only for local OCR
- **FFmpeg** (optional) -- required for JPEG2000 bilevel codestreams and
  for chunking audio recordings that exceed the provider size limit
- **faster-whisper** (optional) -- the `audio` extra, required only for
  local Whisper transcription
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

# Include local Whisper audio transcription (faster-whisper)
uv sync --extra audio
```

**Install Tesseract** (optional, for local OCR):

- Windows: [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki);
  configure path in `image_processing_config.yaml`
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

**Install FFmpeg** (optional, for JPEG2000 bilevel codestreams and for
chunking long audio recordings):

- Windows: `winget install Gyan.FFmpeg`, or
  [ffmpeg.org](https://ffmpeg.org/download.html) with `bin/` added to
  PATH
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
python main/transcribe.py
```

The wizard guides you through document type, method, processing options,
and file selection. Press `b` to go back, `q` to quit at any time.

**CLI mode:**

```bash
# Transcribe a PDF with AI
python main/transcribe.py --type pdfs --method gpt \
    --input ./documents/my_doc.pdf --output ./results

# Process images with Tesseract (offline)
python main/transcribe.py --type images --method tesseract \
    --input ./scans --output ./results

# Batch process PDFs (50% cheaper)
python main/transcribe.py --type pdfs --method gpt --batch \
    --input ./archive --output ./results
```

### Common Workflows

**Large-scale batch processing:**

```bash
# Submit
python main/transcribe.py --type pdfs --method gpt --batch \
    --input ./archive --output ./results
# Monitor (run periodically; auto-downloads on completion)
python main/check_batches.py
```

**Mixed document types (auto mode):**

```bash
python main/transcribe.py --auto \
    --input ./mixed_documents --output ./results
```

**EPUB and MOBI extraction:**

```bash
python main/transcribe.py --type epubs --method native \
    --input ./ebooks --output ./results
python main/transcribe.py --type mobis --method native \
    --input ./kindle_books --output ./results
```

**Audio recordings:**

```bash
# Remote speech-to-text (venue set in config/audio_config.yaml)
python main/transcribe.py --type audio --method audio-api \
    --input ./recordings --output ./transcripts

# Local faster-whisper (offline)
python main/transcribe.py --type audio --method whisper \
    --input ./recordings --output ./transcripts
```

**Repair failed pages:**

```bash
python main/repair.py \
    --transcription ./results/document_transcription.txt --errors-only
```

### CLI Reference

```
--input / --output         Input and output paths
--type                     pdfs | images | epubs | mobis | audio
--method                   native | tesseract | gpt | audio-api | whisper
--auto                     Auto mode (bypasses --type/--method)
--batch                    Use async batch API
--schema NAME              JSON schema selection
--context PATH             Override context file
--context-image PATH       Context image for each page
--model ID                 Override model
--provider NAME            openai | anthropic | google | openrouter
--reasoning-effort LEVEL   none | low | medium | high | xhigh
--model-verbosity LEVEL    concise | medium | verbose (OpenAI GPT-5 family)
--max-output-tokens N      Override the max output token limit
--service-tier TIER        auto | default | flex | priority (OpenAI; overrides
                            concurrency_config.yaml for this run)
--output-format FORMAT     txt | md | json
--output-mode MODE         hash | mirror (mirror replicates input hierarchy)
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

With `--type audio`, `--provider` and `--model` configure the audio venue
declared in `audio_config.yaml` (`openai` or `google`) instead of the image
transcription model, and `--batch` is refused: audio runs synchronously only.

Run `python main/transcribe.py --help` for the full list.

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

ChronoTranscriber uses five YAML files in `config/`, plus one optional
sixth file (`api_keys_config.yaml`). The config directory can be
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

Provider sections are `api_image_processing`, `anthropic_image_processing`,
`google_image_processing`, and `custom_image_processing`. Each accepts:

- `target_dpi`: a positive integer or `native`. Native detects the densest image
  covering at least half the page, including rotated, cropped, and layered scans.
  It renders the composite without upsampling the scan. Pages without a usable
  scan use `native_fallback_dpi` (300 by default).
- `payload_format`: `jpeg` (default) or lossless `png`. `max_image_bytes` limits
  base64 bytes (0 disables it). An oversized PNG falls back to JPEG with a warning;
  a still-oversized payload becomes a page error.
- Grayscale, transparency, JPEG quality, detail/resolution, and resize profiles.
  New keys are read only from the provider section. Tesseract requires numeric DPI.

Fresh clones default to native JPEG at quality 95 for OpenAI and Anthropic.
Google uses 300 DPI, quality 95, and `media_resolution: high`; custom uses 300 DPI
and quality 90. Per-image guards are 50 MB, 10 MB, 20 MB, and disabled respectively.

OpenAI `original` uses a 30,000-patch (32 px) limit and 65,535 px edge for GPT-5.6
and GPT-6; other original-detail models use 10,000 patches and 6,000 px.
Anthropic high-resolution models use 4,784 patches (28 px) and a 2,576 px edge;
standard models use 1,568 patches and a 1,568 px edge. Optional
`original_max_side_px`, `original_max_pixels`, and Anthropic `high_max_side_px`
only tighten model caps. Caps apply to numeric runs too.

Other detail levels, Google, and custom use the configured resize profile;
native mode warns when such a profile discards native resolution. OpenRouter
sizing follows the detail actually sent. Native with an unbounded `none` profile
is rejected. Top-level `render_strategy` selects `direct` or `supersample` for
numeric runs; native always renders directly to the target. The top-level
`max_pixels_per_page` guard (24 MP) applies only to numeric runs.

JSONL provenance records source and render density, sent dimensions, downscale
reason, encoding, model policy, and a settings fingerprint. Resume rejects changed
settings with an error naming the differences; use `--overwrite` to start again.
Legacy JSONLs resume with a warning. Repair reuses recorded settings and detail,
including for raw source images; legacy repair falls back to current settings.

To restore the previous configuration, explicitly set `target_dpi: 300` and
`jpeg_quality: 100` for API, Anthropic, and Google; keep custom quality 90.
Set API `original_max_side_px: 6000`, `original_max_pixels: 10240000`,
Anthropic `high_max_side_px: 2576`, and the previous `llm_detail` / `resize_profile`
values (`original` / `high` for API, `auto` for Anthropic, `high` for Google).
Registry patch caps still apply, so oversized numeric pages may change.

The `postprocessing` block controls text cleanup, hyphenation merging,
whitespace, blank lines, and wrapping.

### 4. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  transcription:
    concurrency_limit: 20
    request_timeout: 900       # Read timeout per attempt (seconds)
    page_timeout: auto         # Wall-clock ceiling per page across all retries
    connect_timeout: 10        # Per-phase HTTP timeouts (OpenAI-family only)
    write_timeout: 30
    pool_timeout: 30
    retry:
      attempts: 8              # Network retries with exponential backoff
      timeout_attempts: 3      # Smaller budget for request-timeout failures
      validation_attempts: 3   # Retries for malformed output + quality
      min_input_tokens: 500    # Cross-contamination detection threshold
      content_quality:
        enabled: true          # Hallucination, truncation, bleed, loop detection
daily_token_limit:
  enabled: true
  daily_tokens: 9000000   # combined cap across tools (secondary guard)
  scope: pooled           # pooled = cap only calls in a defined pool; all = legacy
  per_key_pool_caps:      # per-(API key, pool) daily caps (primary gate)
    enabled: true
    openai:
      small: 9750000      # bare int: cap; model list from built-in defaults
      large:
        cap: 975000       # mapping form: custom cap and/or model prefixes
        # models: ["gpt-5", "o3"]
    # any provider can define its own named pools:
    # myhost:
    #   standard:
    #     cap: 5000000
    #     models: ["my-model"]
```

Controls concurrency limits, retry strategy (network and
validation/quality retries share separate budgets), content-quality
validators with configurable thresholds, service tier, batch chunk
size, and daily token budgets.

HTTP timeouts are per phase for the OpenAI-family clients (openai,
openrouter, custom, and the OpenAI audio backend): `request_timeout` is
the read budget, while `connect_timeout`, `write_timeout`, and
`pool_timeout` (defaults 10/30/30 s) bound the other phases. On Windows
the kernel dead-peer detection Linux gets from `TCP_USER_TIMEOUT` is
unavailable, so the read timeout and the `page_timeout` watchdog are the
only stall detectors -- keep `request_timeout` tight (120-300 s) outside
the `flex` service tier.

### 5. Audio Configuration (`audio_config.yaml`)

```yaml
audio_transcription:
  provider: openai        # openai | google
  concurrency_limit: 4
  openai:
    model: gpt-transcribe # gpt-4o-transcribe | gpt-4o-mini-transcribe | whisper-1
  google:
    model: gemini-3.6-flash
chunking:
  target_seconds: 600     # nominal chunk length
  overlap_seconds: 0
  chunk_format: mp3
local_whisper:
  model_size: large-v3
  device: auto            # auto | cpu | cuda
ffmpeg:
  ffmpeg_cmd: ''          # empty = look up on PATH
```

Selects the remote venue for `--method audio-api`, the request parameters
of each venue (model, prompt, language hints, temperature), how long
recordings are cut into chunks, the local faster-whisper runtime for
`--method whisper`, the ffmpeg/ffprobe executables, and a speech-tuned
postprocessing profile. Every key is optional; each reader falls back to a
built-in default. See [Audio Transcription](#audio-transcription) for the
workflow.

### 6. API Keys Configuration (Optional) (`api_keys_config.yaml`)

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

## Audio Transcription

Speech recordings are a first-class processing type: `--type audio` treats
each file as a flat input, like a PDF, and writes a plain-text transcript
through the same output writer, resume logic, and postprocessing as the
document paths.

**Supported containers:** `mp3`, `wav`, `m4a`, `mp4`, `mpga`, `mpeg`,
`webm`, `flac`, `ogg`, `aac`, `aiff`. Each remote venue accepts a subset:
OpenAI takes `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, and `webm`;
Gemini takes `wav`, `mp3`, `aiff`, `aac`, `ogg`, and `flac`. Chunking
re-encodes to the configured `chunk_format` (`mp3` by default), so a
container outside a venue's list still works once it is chunked.

### Choosing a Venue

| Venue | How to select | Notes |
|-------|--------------|-------|
| OpenAI audio API | `--method audio-api` with `provider: openai` | Default `gpt-transcribe`; also `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1`. 25 MB per request |
| Google Gemini | `--method audio-api` with `provider: google` | Any audio-capable Gemini model; 20 MB per inline request |
| Local faster-whisper | `--method whisper` | Fully offline, no API key, no ffmpeg; handles long files natively |

The remote venue is set by `audio_transcription.provider` in
`config/audio_config.yaml` (or overridden per run with `--provider`);
`--method whisper` bypasses it entirely and runs locally.

### Installation

```bash
# Local Whisper only
uv sync --extra audio

# FFmpeg (Windows) -- needed only for chunking long recordings or page ranges
winget install Gyan.FFmpeg
```

FFmpeg is looked up at call time and is required only when a recording
exceeds the venue's per-request limit or when `--pages` selects a span of
a recording. Short files are sent whole and never touch ffmpeg; local
Whisper ships its own decoder (PyAV) and needs no ffmpeg at all.

### Usage

```bash
# Remote speech-to-text
python main/transcribe.py --type audio --method audio-api \
    --input ./recordings --output ./transcripts

# Local faster-whisper (offline)
python main/transcribe.py --type audio --method whisper \
    --input ./recordings --output ./transcripts
```

The interactive wizard offers audio alongside the document types: choose
"Audio" as the processing type and then the remote or local method; the
remaining prompts (output format, resume mode, file selection) are the
usual ones.

### Chunking and Resume

Recordings longer than a venue's per-request limit are cut into
deterministic segments of `chunking.target_seconds` (600 s by default,
with optional `overlap_seconds`). Each chunk becomes one JSONL record, so
resume, `--retry-errors`, and the evaluation tooling work exactly as they
do for pages.

### Caveats

- Output is plain text only: no timestamps, no speaker diarization.
- Synchronous only. No provider offers a batch API for audio, so
  `--batch` is refused for `--type audio`.
- Chunk boundaries can fall mid-sentence. Raise
  `chunking.overlap_seconds` if boundary artifacts matter.
- Changing `chunking.target_seconds` invalidates an in-progress resume:
  chunk indices shift and already-transcribed chunks no longer align.
- `repair.py` does not cover audio. Re-run the main tool
  with `--retry-errors` to redo failed chunks.

## Batch Processing

Async batch APIs for OpenAI, Anthropic, and Google. OpenAI offers
50% cost savings. OpenRouter and custom endpoints do not support
batch mode.

**How it works:**

1. Images are base64-encoded as data URLs
2. Requests are split into parts under each provider's request-count
   and byte limits (OpenAI 150 MB, Anthropic 224 MB, Google 2 GB)
3. Parts are submitted as separate batch jobs with metadata tracking
4. A debug artifact (`*_batch_submission_debug.json`) is saved for
   repair and status recovery

**Monitoring and cancellation:**

```bash
# Check status and finalize completed results (OpenAI, Anthropic, Google)
python main/check_batches.py

# Cancel non-terminal batch jobs
python main/cancel_batches.py

# Cancel specific non-OpenAI batches by id
python main/cancel_batches.py --batch-ids msgbatch_... batches/...
```

`check_batches` finalizes batches for every supported provider by reading
the provider recorded in each local tracking artifact. `cancel_batches`
without arguments auto-lists and cancels only OpenAI batches; to cancel
Anthropic (`msgbatch_...`) or Google (`batches/...`) batches, pass their ids
via `--batch-ids`, which are routed to the correct provider by id shape.

Batch processing typically completes within 24 hours.

## Utilities

### Repair Transcriptions

Re-transcribe failed or selected pages within an existing output:

```bash
# Repair API errors only
python main/repair.py \
    --transcription ./results/doc_transcription.txt --errors-only

# Repair specific page indices
python main/repair.py \
    --transcription ./results/doc_transcription.txt --indices 5,12,18
```

Audio transcripts are not covered; re-run `transcribe.py` with
`--retry-errors` to redo failed chunks.

### Post-process Transcriptions

Run the text cleanup pipeline (Unicode normalization, hyphenation
merging, whitespace normalization, line wrapping) on existing output:

```bash
python main/postprocess.py \
    --input-dir ./results
```

### Daily Token Budget

Enable in `concurrency_config.yaml` to cap daily API usage. Tracks
total tokens per call, resets at 00:01 UTC (one minute after OpenAI's
00:00 UTC free-tier reset). The counter is persisted
under a user-level state directory (`~/.chronotranscriber/token_state.json`
by default), so it is shared across runs regardless of the working
directory. Override the location with `general.state_dir` in
`paths_config.yaml`; a legacy per-directory
`.chronotranscriber_token_state.json` is adopted once if present.

#### Shared Cross-Tool Token Budget (optional)

ChronoTranscriber can share ONE combined daily budget with its sibling tools
(ChronoMiner, AutoExcerpter) instead of enforcing its cap in isolation. Off
by default; single-tool installations need not care. Enable it in
`concurrency_config.yaml`:

```yaml
shared_token_budget:
  enabled: true
  ledger_dir: ''   # empty = ~/.chronopipeline; or an absolute path
```

When enabled, every participating tool merges its usage into one shared
ledger (`token_ledger.json`, schema v2) guarded by an OS file lock, and
`daily_token_limit.daily_tokens` is enforced against the COMBINED total, so
several tools running concurrently cannot collectively overshoot the budget.
Usage is merged as deltas under the lock (concurrent processes lose nothing);
the hot path stays in memory with the debounced background writer, plus
forced refreshes near the cap and while waiting at the limit. If the ledger
is ever unavailable, the tool degrades to its private counter with a single
warning and never crashes. Keep `daily_tokens` identical across
participating tools; the strictest value simply stops its tool first.
Editing `daily_tokens` while a tool waits at the limit lifts the cap within
a poll cycle, no restart needed.

#### Per-Key-Pool Accounting and Caps

A "pool" is a named set of models that share one daily token allowance per
API key. Pools are defined per provider in `per_key_pool_caps` — each entry
gives a cap and, optionally, a model prefix list — and built-in defaults
mirroring OpenAI's complimentary daily token program apply when a provider
has no configured model lists, so zero-config installs keep working. Every
API call's usage is stamped with its provider, the NAME of the environment
variable that served it (key values are never stored or logged), and the
pool derived from the model name. The shared ledger records a per-(tool,
provider, key env, pool) breakdown alongside the per-tool totals, so you
can always tell how much of a daily allowance remains on any key you use.
Enforcement is two-tier: `per_key_pool_caps` gates each key's own pool (set
your own caps and pools, or disable the gate, to match your account's
terms), and `daily_tokens` remains a combined secondary guard. Under the
default `scope: pooled`, calls whose model belongs to no pool — local or
self-hosted endpoints, providers without an allowance program — are counted
but never blocked, so a free endpoint can never be starved by pooled usage
(with `scope: all` the combined cap applies to every call, the legacy
behavior). When a key's pool cap is reached, the wait message names the
exhausted key and reports the remaining pool of any other keys visible in
the ledger; note that this tool constructs its provider once per run, so
remapping a provider to a different key env var takes effect on the next
run. Usage that predates the upgrade (or arrives from un-stamped paths) is
kept under an "unattributed" row and counts toward the combined total only;
a v1 ledger is adopted in place without losing the day's count.

## Architecture

ChronoTranscriber follows a deep-module architecture: eleven packages
under `modules/`, each with a narrow public surface, composed by CLI
entry points in `main/`.

```
modules/
+-- audio/         Chunk planning, ffmpeg cutting, speech-to-text backends
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
+-- transcribe.py               Primary entry point
+-- check_batches.py            Monitor and finalize batch jobs
+-- cancel_batches.py           Cancel non-terminal batch jobs
+-- repair.py                   Re-transcribe failed pages
+-- postprocess.py              Standalone post-processing
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
Use `repair.py --errors-only` to re-transcribe only
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

The suite contains roughly 2,000 tests (unit and integration) covering
all modules, providers, batch backends, audio, and CLI parsers. Live API smoke
tests are marked `api` and deselected by default; run them explicitly
with `pytest -m api`.

## Versioning

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The commit history was squashed to
a single baseline commit at v1.0.0 on 25 April 2026; version numbers before
v1.0.0 do not exist.

Release notes are in [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT License. Copyright (c) 2025 Paul Goetz. See
[LICENSE](LICENSE) for details.
