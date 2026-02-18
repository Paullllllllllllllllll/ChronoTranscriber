# ChronoTranscriber

A Python-based document transcription tool for researchers, archivists, and digital humanities projects. ChronoTranscriber transforms historical documents, academic papers, and ebooks into searchable, structured text using state-of-the-art AI models or local OCR.

Designed to integrate with [ChronoMiner](https://github.com/Paullllllllllllllllll/ChronoMiner) and [ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader) for a complete document retrieval, transcription, and data extraction pipeline.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Providers and Models](#supported-providers-and-models)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [First-Time Setup](#first-time-setup)
  - [Your First Transcription](#your-first-transcription)
  - [Common Workflows](#common-workflows)
- [Configuration](#configuration)
- [Usage](#usage)
- [Batch Processing](#batch-processing)
- [Fine-Tuning Dataset Preparation](#fine-tuning-dataset-preparation)
- [Utilities](#utilities)
- [Architecture](#architecture)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Contributing](#contributing)
- [Development](#development)
- [License](#license)

## Overview

ChronoTranscriber enables researchers and archivists to transcribe historical documents at scale with minimal cost and effort. The tool supports multiple AI providers through a unified LangChain-based architecture, local OCR via Tesseract, and provides fine-grained control over image preprocessing with reproducible results through structured JSON outputs.

### Execution Modes

ChronoTranscriber supports two execution modes:

- **Interactive Mode**: Guided UI with step-by-step prompts, navigation support (press 'b' to go back, 'q' to quit), and visual feedback. Ideal for first-time users and exploratory workflows.
- **CLI Mode**: Command-line arguments for automation, scripting, and CI/CD pipelines. Set `interactive_mode: false` in `config/paths_config.yaml` or provide arguments directly.

### Document Processing

- **PDF Transcription**: Automatic text extraction or page-to-image rendering for OCR
- **EPUB Extraction**: Native text extraction from EPUB 2.0/3.0 ebooks without OCR
- **MOBI/Kindle Extraction**: Text extraction from unencrypted MOBI, AZW, AZW3, and KFX files
- **Image Folder Transcription**: Process directories of scanned page images (PNG, JPEG, WEBP, BMP, TIFF)
- **Auto Mode**: Scan mixed directories and automatically choose the best method per file
- **Preprocessing Pipeline**: Configurable image enhancement (grayscale, transparency, deskewing, denoising, binarization)
- **Post-processing Pipeline**: Optional text cleanup (merge hyphenation, normalize whitespace, wrap lines)

## Key Features

### Multi-Provider LLM Support

- **OpenAI**: GPT-5.2, GPT-5.1, GPT-5, GPT-4.1, GPT-4o, o3, o4-mini and variants
- **Anthropic**: Claude Opus/Sonnet 4.6, Opus/Sonnet/Haiku 4.5, 4.1, 4, 3.5
- **Google**: Gemini 3.0 Flash Preview, Gemini 3.0 Preview, Gemini 3 Pro, Gemini 2.5 Pro/Flash, Gemini 2.0/1.5
- **OpenRouter**: Access 200+ models through unified API
- **LangChain Integration**: Unified interface with automatic capability detection and parameter filtering

### Tesseract Local OCR

Fully offline processing with configurable engine modes, page segmentation, and advanced preprocessing pipeline for improved accuracy without external API calls.

### Intelligent Capability Management

Automatic model capability detection. OpenAI reasoning models (GPT-5 and o-series) do not support sampler controls like `temperature`/`top_p`, and ChronoTranscriber filters unsupported parameters automatically. LangChain handles retry logic, token tracking, and structured output parsing.

### Hierarchical Context Resolution

Automatically selects the most appropriate contextual guidance:

- File-specific context (e.g., `document_transcr_context.txt` for `document.pdf`)
- Folder-specific context (e.g., `military_records_transcr_context.txt` for `military_records/` folder)
- General fallback context (`context/transcr_context.txt`)

Enables processing mixed document collections with different transcription requirements.

### Batch Processing

- **Multi-Provider Support**: OpenAI, Anthropic, and Google batch APIs
- **Smart Chunking**: Automatic request splitting (150 MB chunks, below 180 MB API limit)
- **Data URL Encoding**: Base64-encoded images embedded directly in requests
- **Metadata Tracking**: Image name, page number, order index for reliable reconstruction
- **Cost Savings**: 50% reduction with OpenAI batch processing

### Reliability Features

- **Multi-tier Retry Strategy**: Automatic exponential backoff for API errors (429, 5xx, timeouts) with jitter
- **Transcription-aware Retries**: Optional retries for `no_transcribable_text` or `transcription_not_possible`
- **Daily Token Budget**: Configurable per-day token limits with automatic midnight reset
- **Comprehensive Logging**: Detailed logs for troubleshooting and observability

## Supported Providers and Models

ChronoTranscriber supports four AI providers through LangChain integration. Set the provider in `config/model_config.yaml` or let the system auto-detect from the model name.

### OpenAI

| Model Family | Models | Key Features |
|--------------|--------|--------------|
| GPT-5.2 | gpt-5.2, gpt-5.2-pro | Flagship reasoning, 400K context, 128K output |
| GPT-5.1 | gpt-5.1, gpt-5.1-mini, gpt-5.1-nano | Adaptive thinking, 400K context |
| GPT-5 | gpt-5, gpt-5-mini, gpt-5-nano | Reasoning, 400K context |
| o-series | o4-mini, o3, o3-pro, o3-mini, o1, o1-pro, o1-mini | Advanced reasoning |
| GPT-4.1 | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | 1M context, sampler controls |
| GPT-4o | gpt-4o, gpt-4o-mini | Multimodal, fast |

Environment variable: `OPENAI_API_KEY`

### Anthropic

| Model Family | Models | Key Features |
|--------------|--------|--------------|
| Claude 4.6 | claude-opus-4-6, claude-sonnet-4-6 | Adaptive/extended thinking; 128K / 64K output |
| Claude 4.5 | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 | Extended thinking, 200K context |
| Claude 4.1 | claude-opus-4-1 | Reasoning support |
| Claude 4 | claude-sonnet-4, claude-opus-4 | Vision, structured output |
| Claude 3.5 | claude-3-5-sonnet, claude-3-5-haiku | 200K context |

Environment variable: `ANTHROPIC_API_KEY`

### Google

| Model Family | Models | Key Features |
|--------------|--------|--------------|
| Gemini 3.0 Flash Preview | gemini-3-flash-preview, gemini-3.0-flash-preview | Thinking, near-Pro speed, 1M context |
| Gemini 3.0 Preview | gemini-3-preview, gemini-3.0-preview | Full reasoning, 1M context |
| Gemini 3 Pro | gemini-3-pro | State-of-the-art reasoning, 1M context |
| Gemini 2.5 | gemini-2.5-pro, gemini-2.5-flash | Adaptive thinking |
| Gemini 2.0 | gemini-2.0-flash | Fast, 1M context |
| Gemini 1.5 | gemini-1.5-pro, gemini-1.5-flash | 2M context |

Environment variable: `GOOGLE_API_KEY`

### OpenRouter

Access 200+ models from multiple providers. Model names use provider prefix (e.g., `openai/gpt-5.1`, `anthropic/claude-sonnet-4-5`, `meta/llama-3.2-90b-vision`).

Environment variable: `OPENROUTER_API_KEY`

**Note**: OpenRouter does not support batch processing.

### Processing Modes

- **Synchronous**: Real-time responses for interactive workflows
- **Batch Processing**: Asynchronous processing for large jobs (OpenAI, Anthropic, Google only)

### Model-Specific Features

- **Reasoning / thinking controls**: Provider-specific controls for internal reasoning (OpenAI `reasoning.effort`, Anthropic extended thinking via `thinking.budget_tokens`, Google `thinkingConfig` / thinking level). Availability and valid values differ by model.
- **Sampler controls**: `temperature`/`top_p` are supported on most non-reasoning models. OpenAI reasoning models don’t support these controls. For Anthropic, extended thinking isn’t compatible with changing `temperature` or `top_k`.
- **Automatic Capability Detection**: Unsupported parameters filtered before API calls

## System Requirements

### Software Dependencies

- **Python**: 3.10 or higher (3.13 recommended)
- **Tesseract OCR** (optional): Required only for local OCR backend
  - Windows: Install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), configure path in `image_processing_config.yaml`
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

### API Requirements

At least one API key required for AI-powered transcription:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

For OpenAI batch processing, ensure your account has Batch API access.

### Python Packages

All dependencies listed in `requirements.txt` (updated December 2025). Key packages:

- LangChain: `langchain==1.2.0`, `langchain-core==1.2.5`
- Providers: `langchain-openai==1.1.6`, `langchain-anthropic==1.3.0`, `langchain-google-genai==4.1.2`
- APIs: `openai==2.14.0`, `anthropic==0.75.0`, `google-genai==1.56.0`
- Processing: `PyMuPDF==1.26.7`, `pillow==12.0.0`, `pytesseract==0.3.13`, `opencv-python==4.12.0.88`
- Data: `pydantic==2.12.5`, `pyyaml==6.0.3`, `numpy==2.2.6`

## Installation

### Clone the Repository

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
cd ChronoTranscriber
```

### Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

For development and running tests:

```bash
pip install -r requirements-dev.txt
```

### Install Tesseract (Optional)

For local OCR:

- **Windows**: Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Configure API Keys

Set API keys for your chosen provider(s):

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
$env:ANTHROPIC_API_KEY="your_api_key_here"
$env:GOOGLE_API_KEY="your_api_key_here"
$env:OPENROUTER_API_KEY="your_api_key_here"

# Linux/macOS
export OPENAI_API_KEY=your_api_key_here
export ANTHROPIC_API_KEY=your_api_key_here
export GOOGLE_API_KEY=your_api_key_here
export OPENROUTER_API_KEY=your_api_key_here
```

For persistent configuration, add to your system environment variables or shell profile.

### Configure File Paths

Edit `config/paths_config.yaml` to specify input/output directories.

## Quick Start

### First-Time Setup

**Step 1: Install Dependencies**

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
cd ChronoTranscriber

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt

# For development and running tests
pip install -r requirements-dev.txt
```

**Step 2: Set Up API Key**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"

# Linux/macOS
export OPENAI_API_KEY="your_key_here"
```

**Step 3: Configure Model (optional)**

Edit `config/model_config.yaml`:

```yaml
transcription_model:
  provider: openai  # Options: openai, anthropic, google, openrouter
  name: gpt-5-mini
  reasoning:
    effort: low  # Options: low, medium, high
```

**Step 4: Configure Paths (optional)**

Edit `config/paths_config.yaml`:

```yaml
file_paths:
  PDFs:
    input: './input/pdfs'
    output: './output/pdfs'
  Images:
    input: './input/images'
    output: './output/images'
```

### Your First Transcription

**Interactive Mode (Recommended)**

```bash
python main/unified_transcriber.py
```

The interface guides you through:
1. Document type selection (PDFs, images, EPUBs, auto mode)
2. Transcription method (native extraction, Tesseract OCR, AI)
3. Processing options (batch vs synchronous, schema selection)
4. File selection
5. Review and confirmation

Press 'b' to go back, 'q' to quit at any time.

**CLI Mode (Automation)**

```bash
# Transcribe PDF with AI
python main/unified_transcriber.py --type pdfs --method gpt --input ./documents/my_document.pdf --output ./results

# Process images with Tesseract (offline)
python main/unified_transcriber.py --type images --method tesseract --input ./scans --output ./results

# Batch process multiple PDFs (cost-effective)
python main/unified_transcriber.py --type pdfs --method gpt --batch --input ./archive --output ./results
```

### Common Workflows

**Workflow 1: Quick Test with Single PDF**

```bash
# Using AI
python main/unified_transcriber.py --type pdfs --method gpt --input ./test.pdf --output ./results

# Using Tesseract (offline)
python main/unified_transcriber.py --type pdfs --method tesseract --input ./test.pdf --output ./results
```

**Workflow 2: Large-Scale Batch Processing**

```bash
# Step 1: Submit batch (50% cost reduction)
python main/unified_transcriber.py --type pdfs --method gpt --batch --input ./archive --output ./results

# Step 2: Check status (run periodically)
python main/check_batches.py

# Results automatically downloaded when complete
```

Batch processing typically completes within 24 hours.

**Workflow 3: Mixed Document Types (Auto Mode)**

```bash
python main/unified_transcriber.py --auto --input ./mixed_documents --output ./results
```

Auto mode intelligently selects the best method for each file type.

**Workflow 4: Process Image Collections**

```bash
# Process all image folders
python main/unified_transcriber.py --type images --method gpt --input ./scans --output ./results

# With custom schema
python main/unified_transcriber.py --type images --method gpt --schema plain_text_transcription_schema --input ./scans --output ./results
```

**Workflow 5: EPUB and MOBI Ebook Extraction**

```bash
# Extract from EPUB
python main/unified_transcriber.py --type epubs --method native --input ./ebooks --output ./results

# Extract from MOBI/Kindle
python main/unified_transcriber.py --type mobis --method native --input ./kindle_books --output ./results
```

**Workflow 6: Repair Failed Transcriptions**

```bash
# Repair all failures
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt

# Repair only API errors
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt --errors-only

# Repair specific pages
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt --indices 5,12,18
```

## Configuration

ChronoTranscriber uses four YAML configuration files in the `config/` directory.

### 1. Model Configuration (`model_config.yaml`)

```yaml
transcription_model:
  provider: openai  # Options: openai, anthropic, google, openrouter (auto-detected from model name)
  name: gpt-5-mini
  max_output_tokens: 128000
  
  # Reasoning / thinking controls.
  # OpenAI: sent as `reasoning: { effort: ... }` in the Responses API (and as `reasoning_effort` in Chat Completions); supported values are model-dependent.
  # Anthropic: mapped to `thinking: { type: "enabled", budget_tokens: ... }`.
  # Google: mapped to the Gemini thinking configuration (thinking level/budget).
  reasoning:
    effort: medium  # Default cross-provider preset. For OpenAI, valid values also include `none`, `minimal`, and `xhigh` (model-dependent).
  
  # OpenAI GPT-5 family only (Responses API): controls verbosity of the model's response.
  text:
    verbosity: medium  # Options: low, medium, high
  
  # Sampler controls (only applied when supported by the selected provider/model).
  # OpenAI: temperature range 0.0–2.0.
  # Anthropic: temperature range 0.0–1.0.
  # Google: temperature range 0.0–2.0.
  temperature: 0.01
  top_p: 1.0
  # OpenAI only (Chat Completions): frequency/presence penalties range -2.0–2.0.
  frequency_penalty: 0.01
  presence_penalty: 0.01
```

**Key Parameters**:
- `provider`: AI provider (auto-detected if not specified)
- `name`: Model identifier
- `max_output_tokens`: Maximum tokens per request. For OpenAI reasoning models, this budget must cover both visible output and internal reasoning tokens (insufficient budget can yield an `incomplete` response).
- `reasoning.effort`: Cross-provider reasoning preset used by ChronoTranscriber.
  - OpenAI supports additional values such as `none`, `minimal`, and `xhigh` depending on the model.
- `text.verbosity`: OpenAI GPT-5 family output verbosity (`low` | `medium` | `high`).
- `temperature`, `top_p`: Sampling controls (only used when supported by the selected provider/model).
- `frequency_penalty`, `presence_penalty`: OpenAI Chat Completions sampling penalties (range -2.0–2.0).

References:
- OpenAI reasoning: https://platform.openai.com/docs/guides/reasoning
- OpenAI Chat Completions parameters (incl. `verbosity` / `reasoning_effort` / penalties): https://platform.openai.com/docs/api-reference/chat/create
- Anthropic Messages + extended thinking: https://platform.claude.com/docs/en/api/messages/create and https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- Google GenerationConfig (temperature 0.0–2.0, etc.): https://ai.google.dev/api/rest/v1/GenerationConfig

### 2. Paths Configuration (`paths_config.yaml`)

```yaml
general:
  interactive_mode: true  # true=interactive prompts, false=CLI mode
  retain_temporary_jsonl: false
  input_paths_is_output_path: true
  logs_dir: './logs'
  keep_preprocessed_images: false
  auto_mode_pdf_use_ocr_for_scanned: true
  auto_mode_pdf_use_ocr_for_searchable: false
  auto_mode_pdf_ocr_method: 'gpt'
  auto_mode_image_ocr_method: 'gpt'

file_paths:
  PDFs:
    input: './input/pdfs'
    output: './output/pdfs'
  Images:
    input: './input/images'
    output: './output/images'
  EPUBs:
    input: './input/epubs'
    output: './output/epubs'
  MOBIs:
    input: './input/mobis'
    output: './output/mobis'
  Auto:
    input: './input/auto'
    output: './output/auto'
```

**Key Parameters**:
- `interactive_mode`: Controls operation mode
- `retain_temporary_jsonl`: Keep temporary files after completion
- `input_paths_is_output_path`: Write outputs alongside inputs
- `keep_preprocessed_images`: Retain preprocessed images
- Auto mode settings control OCR method selection

### 3. Image Processing Configuration (`image_processing_config.yaml`)

Provider-aware preprocessing automatically detects model type and applies appropriate settings.

**OpenAI Image Processing**:

```yaml
api_image_processing:
  target_dpi: 300
  grayscale_conversion: true
  handle_transparency: true
  llm_detail: high  # Options: low, high, auto
  jpeg_quality: 100
  resize_profile: high  # Options: auto, none
  low_max_side_px: 512
  high_target_box: [768, 1536]
```

**Google Gemini Image Processing**:

```yaml
google_image_processing:
  target_dpi: 300
  grayscale_conversion: true
  handle_transparency: true
  media_resolution: high  # Options: low, medium, high, ultra_high, auto
  jpeg_quality: 100
  resize_profile: high
  low_max_side_px: 512
  high_target_box: [768, 1536]
```

**Anthropic Claude Image Processing**:

```yaml
anthropic_image_processing:
  target_dpi: 300
  grayscale_conversion: true
  handle_transparency: true
  jpeg_quality: 100
  resize_profile: auto
  low_max_side_px: 512
  high_max_side_px: 1568
```

**Tesseract Image Processing**:

```yaml
tesseract_image_processing:
  target_dpi: 300
  ocr:
    tesseract_config: "--oem 3 --psm 6"
    tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
  
  preprocessing:
    flatten_alpha: true
    grayscale: true
    invert_to_dark_on_light: auto  # Options: auto, always, never
    deskew: true
    denoise: median  # Options: none, median, bilateral
    binarization: sauvola  # Options: sauvola, adaptive, otsu
    sauvola_window: 25
    sauvola_k: 0.2
    morphology: none  # Options: none, open, close, erode, dilate
    morph_kernel: 3
    border_px: 10
    output_format: png
    preserve_resolution: true
    embed_dpi_metadata: true
```

**Post-processing (Text Cleanup)**:

```yaml
postprocessing:
  enabled: true
  merge_hyphenation: false
  collapse_internal_spaces: true
  max_blank_lines: 2
  tab_size: 4
  wrap_lines: true
  auto_wrap: true
  wrap_width: null
```

### 4. Concurrency Configuration (`concurrency_config.yaml`)

```yaml
concurrency:
  transcription:
    concurrency_limit: 1500
    delay_between_tasks: 0.05
    service_tier: default  # Options: auto, default, flex, priority
    batch_chunk_size: 50
    
    retry:
      attempts: 10
      wait_min_seconds: 1
      wait_max_seconds: 30
      jitter_max_seconds: 0.5
      
      transcription_failures:
        no_transcribable_text_retries: 0
        transcription_not_possible_retries: 3
        wait_min_seconds: 1
        wait_max_seconds: 30
        jitter_max_seconds: 0.5
  
  image_processing:
    concurrency_limit: 24
    delay_between_tasks: 0.0005
```

**Key Parameters**:
- `concurrency_limit`: Maximum concurrent tasks
- `service_tier`: OpenAI service tier (synchronous only; automatically omitted for batch)
- Retry settings: Exponential backoff configuration
- `transcription_failures`: Optional retries for specific transcription outcomes

### Additional Context Guidance

Hierarchical context resolution automatically selects the most appropriate guidance
using the `_transcr_context` filename suffix (most specific wins):

**Context Resolution Order** (applies to all input types):
1. **File-specific**: `{input_stem}_transcr_context.txt` next to the input file
2. **Folder-specific**: `{parent_folder}_transcr_context.txt` next to the input's parent folder
3. **General fallback**: `context/transcr_context.txt` in the project root

**Examples**:
- For `archive/document.pdf`: looks for `archive/document_transcr_context.txt`, then `archive_transcr_context.txt`, then `context/transcr_context.txt`
- For `scans/` image folder: looks for `scans_transcr_context.txt`, then `context/transcr_context.txt`
- For `scans/page001.png`: looks for `scans/page001_transcr_context.txt`, then `scans_transcr_context.txt`, then `context/transcr_context.txt`

**Context File Format**:

```text
You are transcribing historical military service records from Brazilian archives.

Typical content includes:
- Personal details: full name, birthplace, birth date
- Service history: enlistment date, units served, ranks
- Campaigns and actions: battles, decorations, wounds

Formatting conventions:
- Dates in day/month/year format
- Abbreviations: Tte. (Teniente), Cap. (Capitán)
- Preserve original orthography and accents
```

Keep context files under 4000 characters for optimal performance.

### Custom Transcription Schemas

ChronoTranscriber supports custom JSON schemas for controlling output format.

**Included Schemas**:
- `markdown_transcription_schema.json` (default): Markdown with LaTeX equations
- `plain_text_transcription_schema.json`: Plain text without formatting
- `plain_text_transcription_with_markers_schema.json`: Plain text with page markers
- `swiss_address_book_schema.json`: Specialized address book extraction

**Schema Structure**:

```json
{
  "name": "schema_name",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "image_analysis": { "type": "string" },
      "transcription": { "type": ["string", "null"] },
      "no_transcribable_text": { "type": "boolean" },
      "transcription_not_possible": { "type": "boolean" }
    },
    "required": ["image_analysis", "transcription", "no_transcribable_text", "transcription_not_possible"],
    "additionalProperties": false
  }
}
```

Place custom schemas in `schemas/` directory.

## Usage

### Main Transcription Workflow

**Interactive Mode**:

```bash
python main/unified_transcriber.py
```

Guides you through document type, OCR backend, processing mode, schema selection, context configuration, and file selection.

**CLI Mode**:

```bash
# Process directory of PDFs with GPT batch mode
python main/unified_transcriber.py --type pdfs --method gpt --batch --input ./input/pdfs --output ./output/pdfs

# Process image folder with Tesseract (offline)
python main/unified_transcriber.py --type images --method tesseract --input ./input/images --output ./output/images

# Extract from EPUB ebooks
python main/unified_transcriber.py --type epubs --method native --input ./input/epubs --output ./output/epubs

# Extract from MOBI/Kindle ebooks
python main/unified_transcriber.py --type mobis --method native --input ./input/mobis --output ./output/mobis

# View all options
python main/unified_transcriber.py --help
```

### Output Files

Outputs saved to configured directories:

- `<original_name>_transcription.txt`: Final transcription
- `<original_name>_temporary.jsonl`: Batch tracking (deleted after completion unless `retain_temporary_jsonl: true`)
- `<original_name>_batch_submission_debug.json`: Batch metadata for tracking and repair

## Batch Processing

Batch processing enables asynchronous transcription at 50% lower cost (OpenAI). Supported providers: OpenAI, Anthropic, Google. Not supported: OpenRouter.

### How Batch Processing Works

1. **Image Encoding**: Images base64-encoded as data URLs
2. **Request Chunking**: Split into chunks (≤150 MB per chunk)
3. **Metadata Tagging**: custom_id, image name, page number, order index
4. **Batch Submission**: Chunks submitted as separate jobs
5. **Debug Artifact**: Metadata saved in `<job>_batch_submission_debug.json`

### Monitoring Batch Jobs

```bash
python main/check_batches.py
```

- Scans directories for temporary JSONL files
- Checks status of all batch jobs
- Repairs missing batch IDs using debug artifacts
- Downloads results when all batches complete
- Merges outputs using multi-level ordering strategy
- Diagnoses API/model errors

### Cancelling Batch Jobs

```bash
python main/cancel_batches.py
```

- Lists all batch jobs with status summary
- Identifies terminal batches (completed, expired, cancelled, failed)
- Cancels all non-terminal batches
- Shows detailed cancellation results

## Fine-Tuning Dataset Preparation

ChronoTranscriber includes a workflow for preparing OpenAI-compatible vision fine-tuning datasets where the assistant response is a structured JSON string matching your transcription schema.

### Overview

1. Produce transcriptions with `main/unified_transcriber.py`
2. Export editable correction file: `main/prepare_ground_truth.py --extract`
3. Apply corrections into ground truth JSONL: `main/prepare_ground_truth.py --apply`
4. Build OpenAI vision fine-tuning JSONL: `fine_tuning/build_openai_vision_sft_jsonl.py`

### Ground Truth Input

Expects ground truth JSONL from `main/prepare_ground_truth.py --apply`:

- `page_index` (0-based)
- `transcription` (string or null)
- `no_transcribable_text` (boolean)
- `transcription_not_possible` (boolean)

### Manifest Input (JSONL)

JSONL file listing pages for training. Requires:

- `page_index` or `order_index` (0-based)
- One of: `pre_processed_image`, `image_path`, or `image_url`

### Build Command

```bash
.venv\Scripts\python fine_tuning\build_openai_vision_sft_jsonl.py --ground-truth eval\test_data\ground_truth\address_books --manifest path\to\manifest.jsonl --output fine_tuning\training.jsonl
```

**Optional flags**:
- `--schema <schema_name_or_path>`
- `--system-prompt <path>`
- `--additional-context <path>`
- `--image-detail low|high|auto`
- `--strict` (fail on first skipped entry)

**Image constraints**:
- Allowed: `.jpg`, `.jpeg`, `.png`, `.webp`
- Maximum file size: 10 MB

## Utilities

### Token Cost Analysis

Inspects preserved `.jsonl` files and produces detailed cost estimates.

**When to Run**:
- Preserve temporary files: `retain_temporary_jsonl: true` in `paths_config.yaml`
- After processing completes
- To validate budgeting assumptions

**Execution Modes**:

```bash
# Interactive UI
python main/cost_analysis.py

# CLI Mode
python main/cost_analysis.py --save-csv --output path/to/report.csv --quiet
```

**Output Features**:
- Aggregated totals (uncached input, cached, output, reasoning tokens)
- Dual pricing (standard + 50% discount for batch/flex)
- Model normalization (date-stamped variants mapped to parent profiles)
- CSV export with per-file ledger and summary row

See [OpenAI Pricing](https://platform.openai.com/docs/pricing) for current rates.

### Daily Token Limit

Automatically track daily Responses API usage and pause when budget is exhausted.

**Configuration** (`config/concurrency_config.yaml`):

```yaml
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

**How It Works**:
- Automatic tracking of `usage.total_tokens` from each API call
- Midnight reset at local midnight
- Pre-flight checks before each GPT document
- Live telemetry in logs and console

**Recommended Usage**:
1. Enable and set `daily_tokens` to match allocation
2. Add `.chronotranscriber_token_state.json` to `.gitignore`
3. Delete state file to manually reset counts

### System Diagnostics

Built into `check_batches.py`:

```bash
python main/check_batches.py
```

**Checks**:
1. API Key Presence
2. Model Listing Access
3. Batch API Access

## Architecture

ChronoTranscriber follows a modular architecture with clear separation of concerns.

### Directory Structure

```
ChronoTranscriber/
├── config/                    # Configuration files
│   ├── concurrency_config.yaml
│   ├── image_processing_config.yaml
│   ├── model_config.yaml
│   └── paths_config.yaml
├── fine_tuning/               # Fine-tuning dataset preparation
│   └── build_openai_vision_sft_jsonl.py
├── main/                      # CLI entry points
│   ├── cancel_batches.py
│   ├── check_batches.py
│   ├── cost_analysis.py
│   ├── repair_transcriptions.py
│   └── unified_transcriber.py
├── modules/                   # Core application modules
│   ├── config/               # Configuration loading and service
│   ├── core/                 # Core utilities, workflow, CLI args
│   ├── diagnostics/          # System health checks
│   ├── infra/                # Logging, concurrency, async tasks
│   ├── io/                   # File I/O and path utilities
│   ├── llm/                  # LLM integration
│   │   ├── providers/        # LangChain provider implementations
│   │   ├── batch/            # Batch API processing
│   │   └── ...               # Transcriber, schemas, capabilities
│   ├── operations/           # High-level operations
│   ├── processing/           # Image and PDF processing
│   └── ui/                   # User interface and prompts
├── schemas/                   # JSON schemas for structured outputs
├── system_prompt/             # System prompt templates
├── context/                   # Hierarchical context directory
│   └── transcr_context.txt   # General fallback context
├── LICENSE
├── README.md
└── requirements.txt
```

### Module Structure

- **modules/config/**: Configuration loading and centralized ConfigService
- **modules/core/**: Core utilities (console, workflow, path handling, CLI args)
- **modules/infra/**: Infrastructure (logging, concurrency, async tasks)
- **modules/io/**: File I/O operations
- **modules/llm/**: LLM integration layer
  - **providers/**: LangChain-based implementations (OpenAI, Anthropic, Google, OpenRouter)
  - **batch/**: Batch API processing with multi-provider backends
  - **transcriber.py**: High-level transcription interface
  - **model_capabilities.py**: Capability detection and parameter filtering
  - **schemas.py**: Pydantic models for structured outputs
- **modules/operations/**: High-level operations (batch checking, repair, cost analysis)
- **modules/processing/**: Document processing (PDF, image preprocessing, text formatting)
- **modules/ui/**: User interface (interactive prompts, styled output, navigation)
- **modules/diagnostics/**: System health checks
- **modules/testing/**: Test fixtures and utilities

### LangChain Provider Architecture

Unified multi-provider interface:

```
modules/llm/providers/
├── base.py               # Abstract BaseProvider class
├── factory.py            # Provider factory with auto-detection
├── openai_provider.py    # OpenAI (GPT-5, o-series, GPT-4o)
├── anthropic_provider.py # Anthropic (Claude family)
├── google_provider.py    # Google (Gemini family)
└── openrouter_provider.py # OpenRouter (200+ models)
```

Each provider handles capability detection, parameter validation, retry logic, token tracking, and structured output parsing.

### Operations Layer

High-level operations in `modules/operations/` (batch checking, repair workflows, cost analysis) are separated from CLI entry points in `main/` for testability and reusability.

## Frequently Asked Questions

### General Questions

**Q: Which AI provider should I choose?**

A: Depends on your priorities:
- **OpenAI (gpt-5-mini)**: Best cost/quality balance. 50% batch discount.
- **Anthropic (Claude 3.5 Sonnet)**: Superior for complex layouts and academic papers.
- **Google (Gemini 2.0 Flash)**: Fastest processing, lowest cost.
- **OpenRouter**: Access 200+ models with single API key.

Start with OpenAI gpt-5-mini with low reasoning effort.

**Q: How much does transcription cost?**

A: With OpenAI gpt-5-mini:
- Single page (300 DPI): ~$0.01-0.02
- 100-page PDF (synchronous): ~$1-2
- 100-page PDF (batch): ~$0.50-1 (50% discount)
- 1000-page archive (batch): ~$5-10

Use `python main/cost_analysis.py` to track actual spending.

**Q: Should I use batch or synchronous mode?**

A: Use batch for:
- More than 50 pages
- Cost priority (50% cheaper)
- Can wait 24 hours
- Large archives

Use synchronous for:
- Immediate results needed
- Fewer than 50 pages
- Testing/debugging

**Q: Can I process documents offline?**

A: Yes, use Tesseract OCR:

```bash
python main/unified_transcriber.py --type pdfs --method tesseract --input ./docs --output ./results
```

Tesseract is free but generally lower quality than AI models.

### Configuration Questions

**Q: How do I switch providers?**

A: Edit `config/model_config.yaml`:

```yaml
transcription_model:
  provider: anthropic
  name: claude-3-5-sonnet
```

Then set the appropriate API key environment variable.

**Q: How do I control costs with daily token limits?**

A: Enable in `config/concurrency_config.yaml`:

```yaml
daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

Processing pauses when limit is reached and resumes next day.

**Q: What's the difference between reasoning effort levels?**

A: For reasoning models (GPT-5, o-series, Claude 4.5, Gemini 3):
- **Low**: Fastest, cheapest, good for straightforward documents
- **Medium**: Balanced quality/cost (recommended default)
- **High**: Best quality for complex documents, slower/more expensive

For most historical documents, low or medium is sufficient.

**Q: How do I customize output format?**

A: Use different schemas:
- `markdown_transcription_schema.json`: Formatted text with LaTeX
- `plain_text_transcription_schema.json`: Simple plain text
- `plain_text_transcription_with_markers_schema.json`: Plain text with page markers

Specify with `--schema` flag or select in interactive mode.

### Processing Questions

**Q: What happens if pages fail to transcribe?**

A: Failed pages are marked in output:
- `[transcription error: page_name]`: API/processing error
- `[No transcribable text]`: Model found no readable text
- `[Transcription not possible]`: Model couldn't process

Retry with repair tool:

```bash
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt --errors-only
```

**Q: How do I process mixed content (PDFs, images, ebooks)?**

A: Use auto mode:

```bash
python main/unified_transcriber.py --auto --input ./mixed_folder --output ./results
```

Auto mode intelligently selects the best method for each file type.

**Q: Can I process password-protected PDFs?**

A: No, ChronoTranscriber does not support encrypted PDFs. Decrypt them first using external tools.

**Q: How do I preserve page numbers in output?**

A: Use a schema with page markers:

```bash
python main/unified_transcriber.py --type pdfs --method gpt --schema plain_text_transcription_with_markers_schema --input ./docs --output ./results
```

Page numbers appear as `<page_number>5</page_number>`.

### Batch Processing Questions

**Q: How do I check if batch jobs are complete?**

A:

```bash
python main/check_batches.py
```

Shows status and automatically downloads completed results.

**Q: Can I cancel a batch job?**

A: Yes:

```bash
python main/cancel_batches.py
```

Note: You're charged for processing before cancellation.

**Q: Where are batch results stored?**

A: In the output directory specified when submitting. Temporary tracking files (`*_temporary.jsonl`) kept until results are downloaded.

**Q: What if batch processing fails?**

A: ChronoTranscriber automatically falls back to synchronous processing if batch submission fails. Check logs for details. Common causes:
- Provider doesn't support batch (OpenRouter)
- API key lacks batch access
- Network issues

### Technical Questions

**Q: What image formats are supported?**

A: PNG, JPEG, JPG, WEBP, BMP, TIFF. Images automatically preprocessed based on selected provider.

**Q: What's the maximum PDF size?**

A: No hard limit, but very large PDFs (1000+ pages) should use batch processing. Individual page images limited by provider constraints (typically 20MB after preprocessing).

**Q: Can I run multiple jobs simultaneously?**

A: Yes, but be mindful of:
- API rate limits (configure in `concurrency_config.yaml`)
- Daily token budgets
- System memory for image preprocessing

Each job runs independently.

**Q: How do I integrate into existing pipelines?**

A: Use CLI mode with `interactive_mode: false`:

```bash
# In your script or CI/CD pipeline
python main/unified_transcriber.py --type pdfs --method gpt --input "$INPUT_DIR" --output "$OUTPUT_DIR"

# Check exit code
if [ $? -eq 0 ]; then
    echo "Transcription successful"
else
    echo "Transcription failed"
fi
```

All scripts return proper exit codes.

**Q: I'm experiencing issues not covered here**

A: Check logs in configured `logs_dir`, run `python main/check_batches.py` for diagnostics, validate configuration files, verify directory permissions, and review dependencies. For persistent issues, please open a GitHub issue with detailed error information.

## Contributing

Contributions are welcome!

### Reporting Issues

Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, package versions)
- Relevant config sections (remove sensitive info)
- Log excerpts

### Suggesting Features

Provide:
- Use case description
- Proposed solution
- Alternatives considered
- Impact assessment

### Code Contributions

1. Fork repository and create feature branch
2. Follow existing code style and architecture
3. Add tests for new functionality
4. Update documentation
5. Test with both Tesseract and AI backends
6. Submit pull request with clear description

### Development Guidelines

- **Modularity**: Keep functions focused, modules organized
- **Error Handling**: Use try-except with informative messages
- **Logging**: Use logger for debugging information
- **Configuration**: Use YAML files, avoid hardcoding
- **User Experience**: Clear prompts and feedback
- **Documentation**: Update docstrings and README

### Areas for Contribution

- Additional OCR backends
- Enhanced preprocessing algorithms
- New output formats
- Testing (unit and integration tests)
- Documentation (tutorials, examples)
- Performance optimization
- Error recovery mechanisms

## Development

### Development Dependencies

For development and running tests, install the dev requirements file:

```bash
pip install -r requirements-dev.txt
```

### Running Tests

```bash
.venv\Scripts\python.exe -m pytest -v
```

### Recent Updates

For complete release history, see [RELEASE_NOTES_v3.0.md](RELEASE_NOTES_v3.0.md) and [RELEASE_NOTES_v2.0.md](RELEASE_NOTES_v2.0.md).

**December 2025**: Dependency updates - All packages updated to latest stable versions with full backward compatibility verified.

**November 2025**: Version 3.0 - Multi-provider LangChain integration, EPUB/MOBI support, auto mode, cost analysis utilities, daily token budgets.

**October 2025**: Version 2.0 - Dual-mode operation (interactive/CLI), Windows Unicode encoding fix, batch API compatibility fix.

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
