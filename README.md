# ChronoTranscriber

A comprehensive Python-based document transcription tool designed for researchers, archivists, and digital humanities projects. ChronoTranscriber transforms historical documents, academic papers, and ebooks into searchable, structured text using state-of-the-art AI models or local OCR.

## What Makes ChronoTranscriber Different

ChronoTranscriber is built specifically for large-scale document digitization with features that matter for serious research projects:

**Multi-Provider AI Support**: Choose from OpenAI (GPT-5, o-series, GPT-4o), Anthropic (Claude 4.5, Claude 3.5), Google (Gemini 3, Gemini 2.5), or access 200+ models via OpenRouter. Switch providers without changing your workflow.

**Cost-Effective Batch Processing**: Submit hundreds or thousands of pages for asynchronous processing at 50% lower cost. Batch support for OpenAI, Anthropic, and Google with automatic chunking and progress tracking.

**Flexible Operation Modes**: Run interactively with guided prompts and navigation, or use CLI mode for automation, scripting, and CI/CD pipelines. The same tool adapts to your workflow.

**Production-Ready Reliability**: Automatic retry logic with exponential backoff, daily token budget management, comprehensive error handling, and detailed logging for troubleshooting.

**Multiple Input Formats**: Process PDFs (native text extraction or OCR), image folders (PNG, JPEG), EPUB ebooks, and MOBI/Kindle files. Auto mode intelligently selects the best method for each file.

**Local OCR Option**: Use Tesseract OCR for completely offline processing without API costs or internet dependency.

**Structured Outputs**: JSON schema enforcement ensures consistent, parseable results. Built-in schemas for markdown, plain text, and specialized formats like Swiss address books.

**Hierarchical Context System**: Provide domain-specific guidance at file, folder, or global level. The system automatically selects the most appropriate context for each document, supporting mixed collections with different transcription requirements.

## Table of Contents

-   [What Makes ChronoTranscriber Different](#what-makes-chronotranscriber-different)
-   [Overview](#overview)
-   [Key Features](#key-features)
-   [Supported Providers and Models](#supported-providers-and-models)
-   [System Requirements](#system-requirements)
-   [Installation](#installation)
-   [Quick Start](#quick-start)
    -   [First-Time Setup](#first-time-setup)
    -   [Your First Transcription](#your-first-transcription)
    -   [Common Workflows](#common-workflows)
-   [Configuration](#configuration)
-   [Usage](#usage)
-   [Batch Processing](#batch-processing)
-   [Fine-Tuning Dataset Preparation](#fine-tuning-dataset-preparation)
-   [Utilities](#utilities)
-   [Architecture](#architecture)
-   [Frequently Asked Questions](#frequently-asked-questions)
-   [Troubleshooting](#troubleshooting)
-   [Contributing](#contributing)
-   [Development](#development)
-   [License](#license)

## Overview

ChronoTranscriber enables researchers and archivists to transcribe historical documents at scale with minimal cost and effort. The tool supports multiple AI providers through a unified LangChain-based architecture, local OCR via Tesseract, and provides fine-grained control over image preprocessing with reproducible results through structured JSON outputs.

The application works equally well with modern documents including academic papers, books, and ebooks. It is designed to integrate with [ChronoMiner](https://github.com/Paullllllllllllllllll/ChronoMiner) and [ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader) for a complete historical document retrieval, transcription, and data extraction pipeline.

### Execution Modes

ChronoTranscriber supports two execution modes to accommodate different workflows and user preferences:

-   **Interactive Workflow:** Launch `python main/unified_transcriber.py` with no arguments or keep `general.interactive_mode: true` in `config/paths_config.yaml` to step through a guided UI powered by `modules/ui/workflows.py`. The prompts surface available models, schemas, and preprocessing options while letting you review every choice before execution. Navigation features allow you to go back to previous steps (press 'b') or quit at any time (press 'q').
-   **CLI / Automation Mode:** Provide command-line arguments (see `python main/unified_transcriber.py --help`) or set `general.interactive_mode: false` to run unattended jobs. Argument parsing is handled by `modules/core/cli_args.py`, giving you scriptable control for cron jobs, CI pipelines, and large batch workflows.

### Document Processing

-   PDF Transcription: Process PDF documents with automatic text extraction fallback or page-to-image rendering
-   EPUB Extraction: Natively extract structured text chapters from EPUB ebooks without OCR
    -   Supports EPUB 2.0 and EPUB 3.0
    -   Automatically detects and extracts text from EPUB chapters
    -   Preserves EPUB chapter structure and metadata
-   MOBI/Kindle Extraction: Natively extract text from unencrypted MOBI, AZW, AZW3, and KFX ebooks by unpacking to EPUB/HTML/PDF and normalizing the resulting text
-   Image Folder Transcription: Process directories containing scanned page images (PNG, JPEG)
-   Auto Mode Discovery: Scan a mixed directory and automatically choose native, Tesseract, or GPT transcription per file
-   Multi-Page Support: Preserve page ordering and handle documents with hundreds or thousands of pages
-   Preprocessing Pipeline: Configurable image enhancement including grayscale conversion, transparency handling, deskewing, denoising, and binarization
-   Post-processing Pipeline: Optional text cleanup that merges hyphenated line breaks, normalizes whitespace, limits blank lines, and wraps output via `config/image_processing_config.yaml`

## Key Features

### Multi-Provider LLM Support

ChronoTranscriber uses LangChain to provide a unified interface across multiple AI providers. Select your preferred provider and model through configuration or at runtime.

### Tesseract Local OCR

For fully offline processing, Tesseract OCR provides configurable engine modes, page segmentation options, and an advanced preprocessing pipeline for improved accuracy without external API calls.

### Intelligent Capability Management

The system automatically detects model capabilities and adjusts parameters accordingly. For reasoning models (GPT-5, o-series, Claude 4.5), unsupported parameters like temperature are automatically filtered. LangChain handles retry logic with exponential backoff, token usage tracking, and structured output parsing.

### Hierarchical Context Resolution

ChronoTranscriber automatically selects the most appropriate contextual guidance for each file based on a hierarchical search:

-   File-specific context files (e.g., `document.txt` for `document.pdf`)
-   Folder-specific context files (e.g., `military_records.txt` for files in `military_records/`)
-   Global fallback context (`additional_context/additional_context.txt`)

This enables processing mixed document collections where different file types require different transcription guidance. Context files are validated for size and automatically optimized to reduce API token usage when empty.

## Supported Providers and Models

ChronoTranscriber supports four AI providers through LangChain integration. Set the provider in `config/model_config.yaml` or let the system auto-detect from the model name.

### OpenAI

Model Family

Models

Key Features

GPT-5.1

gpt-5.1, gpt-5.1-mini, gpt-5.1-nano

Adaptive thinking, 256K context

GPT-5

gpt-5, gpt-5-mini, gpt-5-nano

Reasoning, 256K context

o-series

o4-mini, o3, o3-pro, o3-mini, o1, o1-pro, o1-mini

Advanced reasoning

GPT-4.1

gpt-4.1, gpt-4.1-mini, gpt-4.1-nano

1M context, sampler controls

GPT-4o

gpt-4o, gpt-4o-mini

Multimodal, fast

Environment variable: `OPENAI_API_KEY`

### Anthropic

Model Family

Models

Key Features

Claude 4.5

claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5

Extended thinking

Claude 4.1

claude-opus-4-1

Reasoning support

Claude 4

claude-sonnet-4, claude-opus-4

Vision, structured output

Claude 3.5

claude-3-5-sonnet, claude-3-5-haiku

200K context

Environment variable: `ANTHROPIC_API_KEY`

### Google

Model Family

Models

Key Features

Gemini 3

gemini-3-pro

State-of-the-art reasoning, 2M context

Gemini 2.5

gemini-2.5-pro, gemini-2.5-flash

Adaptive thinking

Gemini 2.0

gemini-2.0-flash

Fast, 1M context

Gemini 1.5

gemini-1.5-pro, gemini-1.5-flash

2M context

Environment variable: `GOOGLE_API_KEY`

### OpenRouter

Access 200+ models from multiple providers through a unified API. Model names use provider prefix format (e.g., `openai/gpt-5.1`, `anthropic/claude-sonnet-4-5`, `meta/llama-3.2-90b-vision`).

Environment variable: `OPENROUTER_API_KEY`

### Processing Modes

-   **Synchronous**: Real-time responses for interactive workflows
-   **Batch Processing**: Asynchronous processing for large jobs via provider-native batch APIs (OpenAI, Anthropic, Google). OpenRouter does not support batch processing.

### Model-Specific Features

-   **Reasoning Models** (GPT-5, o-series, Claude 4.5, Gemini 3): Support reasoning effort controls; temperature and sampling parameters are automatically disabled
-   **Classic Models** (GPT-4o, GPT-4.1, Claude 3.5, Gemini 2.0): Support temperature, top_p, frequency_penalty, and presence_penalty controls
-   **Automatic Capability Detection**: Parameters incompatible with selected models are filtered before API calls

### Structured Outputs

-   JSON Schema Enforcement: Models return structured responses conforming to predefined schemas
-   Multiple Schema Support: Choose between markdown formatting or plain text output
-   Custom Schemas: Create your own schemas for specialized transcription needs
-   Fallback Parsing: Graceful handling of malformed responses with automatic fallback to raw text

### Reliability and Retry Control

-   Multi-tier Retry Strategy: Automatic exponential backoff for transient API errors (429, 5xx, network timeouts) with jitter to avoid synchronized retries
-   Transcription-aware Retries: Optional, per-condition retries when the model returns `no_transcribable_text` or `transcription_not_possible`, configurable in `config/concurrency_config.yaml`
-   Retry Hints: Honors server-provided `Retry-After` headers and logs each attempt for clear observability
-   Safe Defaults: Sensible baseline retry settings ship out of the box, with the ability to disable or tighten retries per project requirements

### Batch Processing

-   Scalable Submission: Submit large document sets as provider-native batch jobs (OpenAI Batch API, Anthropic Message Batches API, Google Gemini Batch API)
-   Smart Chunking: Automatic request splitting with 150 MB chunk size limit (below the 180 MB API limit)
-   Data URL Encoding: Images are base64-encoded and embedded directly in requests (no external hosting required)
-   Metadata Tracking: Each request includes image name, page number, and order index for reliable reconstruction
-   Debug Artifacts: Submission metadata saved for batch tracking and repair operations

## System Requirements

### Software Dependencies

-   Python: 3.10 or higher (3.13 recommended)
-   Tesseract OCR (optional): Required only if using local OCR backend
    -   Windows: Install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and configure path in `image_processing_config.yaml`
    -   Linux: `sudo apt-get install tesseract-ocr`
    -   macOS: `brew install tesseract`

### API Requirements

At least one API key is required for AI-powered transcription. Set environment variables for your preferred provider(s):

Provider

Environment Variable

OpenAI

`OPENAI_API_KEY`

Anthropic

`ANTHROPIC_API_KEY`

Google

`GOOGLE_API_KEY`

OpenRouter

`OPENROUTER_API_KEY`

For OpenAI batch processing, ensure your account has access to the Batch API.

### Python Packages

All Python dependencies are listed in `requirements.txt` (updated December 2025). Key packages include:

-   `langchain==1.2.0`, `langchain-core==1.2.5`: Core LangChain framework
-   `langchain-openai==1.1.6`, `langchain-anthropic==1.3.0`, `langchain-google-genai==4.1.2`: Provider integrations
-   `openai==2.14.0`: OpenAI SDK for batch processing
-   `anthropic==0.75.0`: Anthropic SDK
-   `google-genai==1.56.0`: Google Gemini SDK
-   `PyMuPDF==1.26.7`: PDF processing
-   `pillow==12.0.0`: Image manipulation
-   `pytesseract==0.3.13`: Tesseract OCR wrapper
-   `opencv-python==4.12.0.88`: Advanced image processing
-   `scikit-image==0.26.0`: Scientific image processing
-   `numpy==2.2.6`: Numerical computing (constrained for opencv compatibility)
-   `pydantic==2.12.5`: Data validation and structured outputs
-   `pyyaml==6.0.3`: Configuration file parsing
-   `aiohttp==3.13.2`, `aiofiles==25.1.0`: Asynchronous I/O

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
.venvScriptsactivate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install Tesseract (Optional)

If you plan to use local OCR, install Tesseract:

-   Windows: Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
-   Linux: `sudo apt-get install tesseract-ocr`
-   macOS: `brew install tesseract`

### Configure OpenAI API Key (Optional)

If using vision-language models, set your API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Windows Command Prompt
set OPENAI_API_KEY=your_api_key_here

# Linux/macOS
export OPENAI_API_KEY=your_api_key_here
```

For persistent configuration, add the environment variable to your system settings or shell profile.

### Configure File Paths

Edit `config/paths_config.yaml` to specify your input and output directories for PDFs, EPUBs, and images.

## Quick Start

This guide will get you transcribing documents in under 5 minutes.

### First-Time Setup

**Step 1: Install Dependencies**

```bash
# Clone the repository
git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
cd ChronoTranscriber

# Create and activate virtual environment
python -m venv .venv
.venvScriptsactivate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install Python packages
pip install -r requirements.txt
```

**Step 2: Set Up API Key (for AI transcription)**

Choose one provider and set its API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key_here"
$env:ANTHROPIC_API_KEY="your_key_here"
$env:GOOGLE_API_KEY="your_key_here"

# Linux/macOS
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
```

For persistent configuration, add to your system environment variables or shell profile.

**Step 3: Configure Your Model (optional)**

Edit `config/model_config.yaml` to select your preferred provider and model:

```yaml
transcription_model:
  provider: openai          # Options: openai, anthropic, google, openrouter
  name: gpt-5-mini          # Model identifier
  reasoning:
    effort: low             # Options: low, medium, high (for reasoning models)
```

The default configuration uses OpenAI's gpt-5-mini with medium reasoning effort, which provides a good balance of quality and cost.

**Step 4: Configure Input/Output Paths (optional)**

Edit `config/paths_config.yaml` to set your default directories:

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

**Interactive Mode (Recommended for First-Time Users)**

Run the transcriber without arguments for a guided experience:

```bash
python main/unified_transcriber.py
```

The interactive interface will guide you through:

1.  Selecting document type (PDFs, images, EPUBs, or auto mode)
2.  Choosing transcription method (native extraction, Tesseract OCR, or AI)
3.  Configuring processing options (batch vs synchronous, schema selection)
4.  Selecting files to process
5.  Reviewing and confirming your choices

You can press 'b' to go back to previous steps or 'q' to quit at any time.

**CLI Mode (For Automation)**

For scripting and automation, use command-line arguments:

```bash
# Transcribe a PDF with AI
python main/unified_transcriber.py --type pdfs --method gpt --input ./documents/my_document.pdf --output ./results

# Process a folder of images with Tesseract (offline)
python main/unified_transcriber.py --type images --method tesseract --input ./scans --output ./results

# Batch process multiple PDFs (cost-effective for large jobs)
python main/unified_transcriber.py --type pdfs --method gpt --batch --input ./archive --output ./results
```

To enable CLI mode by default, set `interactive_mode: false` in `config/paths_config.yaml`.

### Common Workflows

**Workflow 1: Quick Test with a Single PDF**

Best for: Testing the system or processing a single document

```bash
# Using AI (requires API key)
python main/unified_transcriber.py --type pdfs --method gpt --input ./test.pdf --output ./results

# Using Tesseract (offline, no API key needed)
python main/unified_transcriber.py --type pdfs --method tesseract --input ./test.pdf --output ./results
```

**Workflow 2: Large-Scale Batch Processing**

Best for: Processing hundreds or thousands of pages cost-effectively

```bash
# Step 1: Submit batch job (50% cost reduction)
python main/unified_transcriber.py --type pdfs --method gpt --batch --input ./archive --output ./results

# Step 2: Check status (run periodically or when you expect completion)
python main/check_batches.py

# Step 3: Results are automatically downloaded when batches complete
```

Batch processing typically completes within 24 hours. You can close the terminal after submission and check status later.

**Workflow 3: Mixed Document Types (Auto Mode)**

Best for: Folders containing PDFs, images, and ebooks

```bash
# Auto mode intelligently selects the best method for each file
python main/unified_transcriber.py --auto --input ./mixed_documents --output ./results
```

Auto mode will:

-   Use native extraction for searchable PDFs and EPUBs
-   Apply OCR to scanned PDFs and images
-   Choose between Tesseract and AI based on your configuration

**Workflow 4: Processing Image Collections**

Best for: Scanned documents organized in folders

```bash
# Process all image folders
python main/unified_transcriber.py --type images --method gpt --input ./scans --output ./results

# Process with custom schema
python main/unified_transcriber.py --type images --method gpt --schema plain_text_transcription_schema --input ./scans --output ./results
```

**Workflow 5: EPUB and MOBI Ebook Extraction**

Best for: Extracting text from ebooks (no OCR needed)

```bash
# Extract from EPUB files
python main/unified_transcriber.py --type epubs --method native --input ./ebooks --output ./results

# Extract from MOBI/Kindle files
python main/unified_transcriber.py --type mobis --method native --input ./kindle_books --output ./results
```

**Workflow 6: Repairing Failed Transcriptions**

Best for: Fixing pages that failed during initial processing

```bash
# Repair all failures in a transcription
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt

# Repair only API errors (skip "no text" and "not possible" pages)
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt --errors-only

# Repair specific page indices
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt --indices 5,12,18
```

## Configuration

ChronoTranscriber uses four YAML configuration files located in the `config/` directory. Each file controls a specific aspect of the pipeline.

### 1. Model Configuration (`model_config.yaml`)

Controls which provider and model to use with behavioral parameters.

```yaml
transcription_model:
  # Provider selection: openai, anthropic, google, openrouter
  # Auto-detected from model name if not specified
  provider: openai
  
  # Model name (see Supported Providers section for options)
  name: gpt-5-mini
  max_output_tokens: 128000
  
  # Reasoning models (GPT-5, o-series, Claude 4.5, Gemini 3)
  reasoning:
    effort: medium  # Options: low, medium, high
  
  # GPT-5 series only
  text:
    verbosity: medium  # Options: low, medium, high
  
  # Classic models only (GPT-4o, GPT-4.1, Claude 3.5, Gemini 2.0)
  # These are automatically disabled for reasoning models
  temperature: 0.01
  top_p: 1.0
  frequency_penalty: 0.01
  presence_penalty: 0.01
```

Key Parameters:

-   `provider`: AI provider (openai, anthropic, google, openrouter); auto-detected if not specified
-   `name`: Model identifier (provider-specific)
-   `max_output_tokens`: Maximum tokens the model can generate per request
-   `reasoning.effort`: Controls reasoning depth for reasoning models (low, medium, high)
-   `text.verbosity`: Controls response verbosity for GPT-5 models (low, medium, high)
-   `temperature`: Controls randomness (automatically disabled for reasoning models)
-   `top_p`: Nucleus sampling probability (automatically disabled for reasoning models)
-   `frequency_penalty`: Penalizes token repetition (automatically disabled for reasoning models)
-   `presence_penalty`: Penalizes repeated topics (automatically disabled for reasoning models)

### 2. Paths Configuration (`paths_config.yaml`)

Defines input/output directories, operation mode, and path resolution behavior.

```yaml
general:
  interactive_mode: true  # Toggle between interactive prompts (true) or CLI mode (false)
  retain_temporary_jsonl: false  # Keep temporary JSONL files for debugging
  input_paths_is_output_path: true  # Write outputs alongside inputs
  logs_dir: './logs'
  keep_preprocessed_images: false  # Retain preprocessed images after processing
  # Auto mode: PDF processing settings
  auto_mode_pdf_use_ocr_for_scanned: true  # Force OCR for scanned PDFs
  auto_mode_pdf_use_ocr_for_searchable: false  # Force OCR even for searchable PDFs
  auto_mode_pdf_ocr_method: 'gpt'  # OCR method: 'tesseract' or 'gpt'
  # Auto mode: Image processing settings
  auto_mode_image_ocr_method: 'gpt'  # OCR method for images: 'tesseract' or 'gpt'

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
    input: './input/mobis'   # MOBI/Kindle files (.mobi, .azw, .azw3, .kfx)
    output: './output/mobis'
  Auto:
    input: './input/auto'
    output: './output/auto'
```

Key Parameters:

-   `interactive_mode`: Controls operation mode (true for interactive prompts, false for CLI mode)
-   `retain_temporary_jsonl`: Keep temporary JSONL files after batch processing completes
-   `input_paths_is_output_path`: Write outputs to the same directory as inputs (auto mode honors this when all files share a parent)
-   `logs_dir`: Directory for log files
-   `keep_preprocessed_images`: Retain preprocessed images after transcription
-   `auto_mode_pdf_use_ocr_for_scanned`: Force OCR for scanned/non-searchable PDFs (if false, attempts native extraction)
-   `auto_mode_pdf_use_ocr_for_searchable`: Force OCR even for searchable PDFs, bypassing native text extraction
-   `auto_mode_pdf_ocr_method`: OCR method to use when OCR is forced for PDFs (`tesseract` or `gpt`)
-   `auto_mode_image_ocr_method`: OCR method for images (`tesseract` or `gpt`; automatically falls back if unavailable)
-   `transcription_prompt_path` (optional): Custom system prompt file path
-   `transcription_schema_path` (optional): Custom JSON schema file path

### 3. Image Processing Configuration (`image_processing_config.yaml`)

Controls image preprocessing for API providers and Tesseract backends. ChronoTranscriber automatically selects the appropriate preprocessing configuration based on your chosen provider and model, ensuring optimal image preparation for each AI service.

#### Provider-Aware Preprocessing

The system automatically detects the underlying model type and applies provider-specific preprocessing, even when using models through OpenRouter. For example, using `google/gemini-2.5-flash` via OpenRouter will apply Google-specific preprocessing settings.

Provider

Model Detection

Config Section

OpenAI

Direct or `openai/`, `gpt-`, `o1`, `o3`, `o4`

`api_image_processing`

Google

Direct or `google/`, `gemini`

`google_image_processing`

Anthropic

Direct or `anthropic/`, `claude`

`anthropic_image_processing`

OpenRouter

Auto-detected from model name

Varies by underlying model

#### OpenAI Image Processing

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

Key Parameters:

-   `target_dpi`: DPI for rendering PDF pages to images
-   `grayscale_conversion`: Convert images to grayscale to reduce noise
-   `handle_transparency`: Flatten transparent images onto white background
-   `llm_detail`: Controls the OpenAI `detail` parameter (high: better accuracy, higher token usage; low: faster and cheaper; auto: let model decide)
-   `jpeg_quality`: JPEG compression quality (1-100) for processed images
-   `resize_profile`: Image resizing strategy before API submission
-   `low_max_side_px`: Maximum side length in pixels for low-detail resizing
-   `high_target_box`: Target dimensions [width, height] for high-detail resizing with white padding

#### Google Gemini Image Processing

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

Key Parameters:

-   `media_resolution`: Controls the Google `media_resolution` parameter
    -   `high`: MEDIA_RESOLUTION_HIGH (recommended for OCR and dense documents)
    -   `medium`: MEDIA_RESOLUTION_MEDIUM (balanced quality and cost)
    -   `low`: MEDIA_RESOLUTION_LOW (minimal tokens, fast and cheap)
    -   `ultra_high`: MEDIA_RESOLUTION_ULTRA_HIGH (Gemini 3 only, highest quality)
    -   `auto`: MEDIA_RESOLUTION_UNSPECIFIED (let model decide)
-   `high_target_box`: Google uses 768x768 tile-based token calculation; box dimensions are optimized for this

#### Anthropic Claude Image Processing

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

Key Parameters:

-   `high_max_side_px`: Anthropic recommends images with longest edge up to 1568 pixels for optimal latency
-   `resize_profile`: Set to `auto` to apply Anthropic's recommended resizing (no padding, aspect ratio preserved)

Anthropic-specific considerations:

-   Maximum image size: 8000x8000 pixels (2000x2000 if more than 20 images per request)
-   Minimum recommended size: 200 pixels on any edge
-   Token calculation: `tokens = (width * height) / 750`
-   Pre-resizing reduces time-to-first-token latency

#### Tesseract Image Processing

```yaml
tesseract_image_processing:
  target_dpi: 300
  ocr:
    tesseract_config: "--oem 3 --psm 6"
    tesseract_cmd: 'C:Program Files (x86)Tesseract-OCRtesseract.exe'
  
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
    output_format: png  # Options: png, tiff
    preserve_resolution: true
    embed_dpi_metadata: true
```

Preprocessing Pipeline:

1.  Flatten Alpha: Remove transparency by compositing onto white background
2.  Grayscale Conversion: Convert to grayscale for consistent processing
3.  Invert Colors: Ensure dark text on light background (auto/always/never)
4.  Deskew: Detect and correct page rotation
5.  Denoise: Remove noise using median or bilateral filtering
6.  Binarization: Convert to black and white using Sauvola, adaptive, or Otsu thresholding
7.  Morphology: Apply morphological operations to enhance text (optional)
8.  Border Addition: Add white border to prevent text cropping at edges

#### Post-processing (Text Cleanup)

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

Key Behaviors:

-   **Single Source of Truth:** Located in `config/image_processing_config.yaml` so visual preprocessing and textual cleanup live together.
-   **Whitespace & Hyphen Control:** Removes stray control characters, collapses long space runs, and optionally rejoins hyphenated line breaks.
-   **Blank-Line Limits:** Caps consecutive blank lines to keep transcripts tidy while preserving intentional paragraph spacing.
-   **Adaptive Wrapping:** When `wrap_lines` and `auto_wrap` are enabled, the pipeline infers a natural width per document; set `wrap_width` for explicit lengths or disable wrapping entirely.
-   **Safe Defaults:** Leave `enabled: false` to emit raw model output verbatim; toggle on when preparing publication-ready transcripts.

### 4. Concurrency Configuration (`concurrency_config.yaml`)

Controls parallel processing and retry behavior.

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

Key Parameters:

-   `concurrency_limit`: Maximum number of concurrent tasks
-   `delay_between_tasks`: Delay in seconds between starting tasks
-   `service_tier`: OpenAI service tier for rate limiting and processing speed (auto, default, flex, priority)
    -   Note: Service tiers apply only to synchronous API calls; batch processing automatically omits this parameter
-   `batch_chunk_size`: Number of requests per batch part file (affects chunking)
-   Retry settings: Exponential backoff configuration for transient API failures
    -   `attempts`: Maximum retry attempts per request
    -   `wait_min_seconds`: Minimum wait time before retry
    -   `wait_max_seconds`: Maximum wait time before retry
    -   `jitter_max_seconds`: Random jitter to prevent synchronized retries
-   `transcription_failures`: Optional retries applied when the model responds with `no_transcribable_text` or `transcription_not_possible`
    -   Individual counters let you disable or tighten retries per condition (set to 0 to accept the first response)
    -   Independent wait controls mirror the primary retry strategy

### Additional Context Guidance

ChronoTranscriber provides hierarchical context resolution, allowing you to provide domain-specific guidance at multiple levels of specificity. The system automatically selects the most appropriate context for each file being processed.

#### Hierarchical Context Resolution

The system searches for context files in order of specificity, using the first match found:

**For PDF and EPUB files:**

1.  File-specific context: `document.pdf` looks for `document.txt` in the same directory
2.  Folder-specific context: Files in `archive/` look for `archive.txt` in the parent directory
3.  Global fallback: `additional_context/additional_context.txt` in the project root

**For image folders:**

1.  Folder-specific context: Folder `scans/` looks for `scans.txt` in the parent directory
2.  In-folder context: `scans/context.txt` inside the folder itself
3.  Global fallback: `additional_context/additional_context.txt` in the project root

**For individual images:**

1.  Image-specific context: `page001.png` looks for `page001.txt` in the same directory
2.  Folder-specific context: Images in `scans/` look for `scans.txt` in the parent directory
3.  In-folder context: `scans/context.txt` inside the image's directory
4.  Global fallback: `additional_context/additional_context.txt` in the project root

#### Context File Format

Context files are plain text files containing instructions or domain knowledge to guide transcription:

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

#### Context Examples

**Example 1: Mixed Document Collection**

For a folder containing both military records and civil documents:

```
input/
  military_records/
    service_card_001.pdf
    service_card_002.pdf
  civil_registry/
    birth_certificate_001.pdf
    marriage_record_001.pdf
  military_records.txt   # Context for military records
  civil_registry.txt     # Context for civil documents
```

**Example 2: Page-Specific Context**

For documents with different content types on different pages:

```
scans/
  page001.png
  page001.txt      # "This page contains a title page with decorative elements"
  page002.png
  page002.txt      # "This page contains a table of contents"
  page003.png
  page003.txt      # "This page begins the main text in two-column format"
```

**Example 3: Auto Mode with Context**

Auto mode now automatically applies the appropriate context for each file type:

```
input/auto/
  documents/
    report.pdf
  documents.txt    # Applied to all PDFs in documents/
  scans/
    image001.png
  scans.txt        # Applied to all images in scans/
  books/
    novel.epub
  books.txt        # Applied to all EPUBs in books/
```

#### Context in Interactive Mode

When running in interactive mode, you will be prompted whether to use additional context if a global `additional_context.txt` file exists. File-specific and folder-specific contexts are automatically detected and applied without prompting.

#### Context in CLI Mode

In CLI mode, specify a global context file using the `--context` flag:

```bash
python main/unified_transcriber.py --type pdfs --method gpt --context ./my_context.txt --input ./documents --output ./results
```

File-specific and folder-specific context files are always automatically detected and take precedence over the global context.

#### Context Size Recommendations

The system automatically validates context files and warns if they exceed 4000 characters. For optimal performance:

-   Keep context files focused and concise
-   Use bullet points and clear formatting
-   Include only essential domain knowledge
-   Test with sample documents to refine instructions

#### Empty Context Handling

When no context is available (no matching files found), the "Additional context:" section is automatically removed from the prompt to save tokens and avoid model confusion. This optimization reduces API costs without affecting transcription quality.

### Custom Transcription Schemas

ChronoTranscriber supports custom JSON schemas for controlling transcription output format.

Some schemas include explicit page number markers in the transcription output. When enabled, page numbers appearing on headers and footers should be written like this: `<page_number>9</page_number>`.

#### Included Schemas

-   `markdown_transcription_schema.json` (default): Produces markdown-formatted transcriptions with LaTeX equations, headings, and formatting
-   `plain_text_transcription_schema.json`: Produces plain text transcriptions without formatting
-   `plain_text_transcription_with_markers_schema.json`: Produces plain text transcriptions with explicit page number markers

#### Schema Structure

Each schema file should include:

```json
{
  "name": "schema_name",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "image_analysis": { "type": "string", "description": "..." },
      "transcription": { "type": ["string", "null"], "description": "..." },
      "no_transcribable_text": { "type": "boolean", "description": "..." },
      "transcription_not_possible": { "type": "boolean", "description": "..." }
    },
    "required": ["image_analysis", "transcription", "no_transcribable_text", "transcription_not_possible"],
    "additionalProperties": false
  }
}
```

Available Schemas:

-   `markdown_transcription_schema.json` (default): Produces markdown-formatted transcriptions with LaTeX equations, headings, and formatting
-   `plain_text_transcription_schema.json`: Produces plain text transcriptions without formatting
-   `plain_text_transcription_with_markers_schema.json`: Produces plain text transcriptions with explicit page number markers
-   `swiss_address_book_schema.json`: Specialized schema for extracting structured address book entries from historical Swiss documents

Creating Custom Schemas:

1.  Place your schema file in the `schemas/` directory
2.  Follow the structure above with your custom field descriptions
3.  The `name` field will appear in the interactive schema selection menu
4.  Field descriptions instruct the model on how to format its output

## Usage

### Main Transcription Workflow

The primary entry point is `main/unified_transcriber.py`, which provides a workflow for transcribing documents.

#### Interactive Mode

Run the transcriber without arguments:

```bash
python main/unified_transcriber.py
```

The interactive interface guides you through:

1.  Select Processing Type (PDF or Images)
2.  Select OCR Backend (Tesseract or GPT)
3.  Select Processing Mode (Synchronous or Batch, GPT only)
4.  Select JSON Schema (GPT only)
5.  Configure Additional Context (GPT only)
6.  Select Files/Folders
7.  Review and Confirm

#### CLI Mode

Process documents with command-line arguments:

```bash
# Process a directory of PDFs with GPT in batch mode
python main/unified_transcriber.py --type pdfs --method gpt --batch --input ./input/pdfs --output ./output/pdfs

# Process an image folder with Tesseract (offline)
python main/unified_transcriber.py --type images --method tesseract --input ./input/images --output ./output/images

# Extract text from EPUB ebooks (native, no OCR)
python main/unified_transcriber.py --type epubs --method native --input ./input/epubs --output ./output/epubs

# Extract text from MOBI/Kindle ebooks (native unpack and text extraction)
python main/unified_transcriber.py --type mobis --method native --input ./input/mobis --output ./output/mobis

# View all options
python main/unified_transcriber.py --help
```

#### Additional Context in Workflows

Context files are automatically detected and applied during processing. See the "Additional Context Guidance" section in Configuration for complete documentation on hierarchical context resolution, including file-specific and folder-specific context support.

#### Retry Behavior and Observability

ChronoTranscriber applies a layered retry strategy automatically during GPT-based transcription:

-   General API errors (429, 5xx, network timeouts) retry with exponential backoff and jitter
-   Transcription outcomes flagged as `no_transcribable_text` or `transcription_not_possible` can trigger additional retries when enabled
-   Logs emitted in the console and `logs_dir` include attempt counts and wait durations
-   Tuning: Increase `wait_min_seconds`/`wait_max_seconds` for aggressive rate limits, or set individual retry counters to 0 to accept the first model response

### Output Files

Transcription outputs are saved to the configured output directories:

-   PDFs: `file_paths.PDFs.output` (or input directory if `input_paths_is_output_path: true`)
-   Images: `file_paths.Images.output` (or input directory if `input_paths_is_output_path: true`)
-   EPUBs: `file_paths.EPUBs.output` (or input directory if `input_paths_is_output_path: true`)
-   MOBIs: `file_paths.MOBIs.output` (or input directory if `input_paths_is_output_path: true`)

Output Naming Convention:

-   `<original_name>_transcription.txt`: Final transcription text file
-   `<original_name>_temporary.jsonl`: Temporary batch tracking file (deleted after successful completion unless `retain_temporary_jsonl: true`)
-   `<original_name>_batch_submission_debug.json`: Batch metadata for tracking and repair

## Batch Processing

Batch processing allows you to submit hundreds or thousands of pages for asynchronous transcription via provider-native batch APIs.

Supported batch providers:

-   OpenAI (Batch API)
-   Anthropic (Message Batches API)
-   Google (Gemini Batch API)

Not supported:

-   OpenRouter (no native batch API)

### How Batch Processing Works

1.  Image Encoding: Images are base64-encoded as data URLs
2.  Request Chunking: Requests are split into chunks (≤150 MB per chunk)
3.  Metadata Tagging: Each request includes custom_id, image name, page number, and order index
4.  Batch Submission: Chunks are submitted as separate batch jobs
5.  Debug Artifact: Submission metadata is saved in `<job>_batch_submission_debug.json`

### Monitoring Batch Jobs

Use `check_batches.py` to monitor batch job status and download completed results.

```bash
python main/check_batches.py
```

What It Does:

-   Scans all configured directories for temporary JSONL files
-   Checks status of all batch jobs (in_progress, completed, failed, etc.)
-   Repairs missing batch IDs using debug artifacts
-   Downloads results when all batches in a job are complete
-   Merges outputs using multi-level ordering strategy
-   Diagnoses API and model errors with helpful messages
-   Cleans up temporary files after successful completion (optional)

Multi-Level Ordering Strategy:

1.  Explicit order info from request metadata
2.  Custom ID index from batch responses
3.  Embedded page number from structured outputs
4.  Page number parsed from filename
5.  Stable fallback index

### Cancelling Batch Jobs

Use `cancel_batches.py` to cancel pending or in-progress batch jobs.

```bash
python main/cancel_batches.py
```

What It Does:

-   Lists all batch jobs in your account (with pagination)
-   Displays summary by status (completed, in_progress, failed, etc.)
-   Identifies terminal batches (completed, expired, cancelled, failed)
-   Cancels all non-terminal batches
-   Shows detailed summary of cancellation results

## Fine-Tuning Dataset Preparation

ChronoTranscriber includes a workflow for preparing supervised fine-tuning datasets (one example per page) using corrected ground truth. The dataset builder outputs OpenAI-compatible JSONL for vision fine-tuning where the assistant response is a structured JSON string matching your transcription schema.

### Overview

1.  Produce transcriptions (synchronous or batch) with `main/unified_transcriber.py`.
2.  Export an editable correction file with `main/prepare_ground_truth.py --extract`.
3.  Apply corrections into ground truth JSONL with `main/prepare_ground_truth.py --apply`.
4.  Build an OpenAI vision fine-tuning JSONL with `fine_tuning/build_openai_vision_sft_jsonl.py`.

### Ground truth input

The fine-tuning builder expects ground truth JSONL produced by `main/prepare_ground_truth.py --apply` (or equivalent). Each line represents one page and includes:

-   `page_index` (0-based)
-   `transcription` (string or null)
-   `no_transcribable_text` (boolean)
-   `transcription_not_possible` (boolean)

### Manifest input (JSONL)

The manifest is a JSONL file listing the pages you want included in training. To match existing repo conventions, the builder accepts the same page ordering fields used in ChronoTranscriber artifacts:

-   Use `page_index` or `order_index` (0-based)
-   Provide one of:
    -   `pre_processed_image` (path to an image file)
    -   `image_path` (path to an image file)
    -   `image_url` (URL to a hosted image)

You can also pass a filtered transcription JSONL as the manifest. The builder supports nested `image_metadata` records as written by batch-mode JSONL files.

### Build command

```bash
.venvScriptspython fine_tuningbuild_openai_vision_sft_jsonl.py --ground-truth evaltest_dataground_truthaddress_books --manifest pathtomanifest.jsonl --output fine_tuningtraining.jsonl
```

Optional flags:

-   `--schema <schema_name_or_path>`
-   `--system-prompt <path>`
-   `--additional-context <path>`
-   `--image-detail low|high|auto`
-   `--strict` (fail immediately on the first skipped manifest entry)

Image constraints enforced by the builder:

-   Allowed extensions: `.jpg`, `.jpeg`, `.png`, `.webp`
-   Maximum file size: 10 MB

## Utilities

### Token Cost Analysis

ChronoMiner bundles a lightweight analytics utility that inspects preserved temporary `.jsonl` responses and produces detailed cost estimates for every processed file. The workflow is implemented in `main/cost_analysis.py`, which orchestrates helper logic contained in `modules/operations/cost_analysis.py` and formatted output helpers in `modules/ui/cost_display.py`.

#### When to Run It

-   Preserve temporary `.jsonl` files (`retain_temporary_jsonl: true` in `config/paths_config.yaml`).
-   After processing is complete, run the analysis to quantify spend across synchronous and batch jobs.
-   Use the report to validate budgeting assumptions or to decide whether to switch schemas or models.

#### Execution Modes

-   **Interactive UI:** `python -m main.cost_analysis`
    -   Mirrors the standard UI look and feel.
    -   Automatically locates `.jsonl` files based on schema path configuration.
    -   Displays aggregated token totals, per-file summaries, and optional CSV export prompts.
-   **CLI Mode:** `python -m main.cost_analysis --save-csv --output path/to/report.csv`
    -   Suitable for automation or scheduled reporting.
    -   Flags:
        -   `--save-csv`: Persist results to CSV (defaults to the folder that contains the first `.jsonl`).
        -   `--output`: Override the target CSV path.
        -   `--quiet`: Suppress console breakdown and emit only essential status messages.

#### Output Features

-   Aggregated totals for uncached input tokens, cached tokens, output tokens, reasoning tokens, and overall totals.
-   Dual pricing: standard per-million-token rates and an automatic 50% discount column that models batched/flex billing tiers.
-   Model normalization: date-stamped variants (e.g., `gpt-5-mini-2025-08-07`) are mapped to their parent pricing profile before calculations.
-   CSV export includes a per-file ledger plus a consolidated summary row that mirrors the on-screen totals.

#### Supported Pricing Profiles (USD per 1M tokens)

Model

Input

Cached Input

Output

gpt-5

1.25

0.125

10.00

gpt-5-mini

0.25

0.025

2.00

gpt-5-nano

0.05

0.005

0.40

gpt-5-chat-latest

1.25

0.125

10.00

gpt-5-codex

1.25

0.125

10.00

gpt-4.1

2.00

0.50

8.00

gpt-4.1-mini

0.40

0.10

1.60

gpt-4.1-nano

0.10

0.025

0.40

gpt-4o

2.50

1.25

10.00

gpt-4o-2024-05-13

5.00

-

15.00

gpt-4o-mini

0.15

0.075

0.60

gpt-4o-realtime-preview

5.00

2.50

20.00

gpt-4o-mini-realtime-preview

0.60

0.30

2.40

gpt-4o-audio-preview

2.50

-

10.00

gpt-4o-mini-audio-preview

0.15

-

0.60

gpt-4o-search-preview

2.50

-

10.00

gpt-4o-mini-search-preview

0.15

-

0.60

gpt-audio

2.50

0.00

10.00

o1

15.00

7.50

60.00

o1-pro

150.00

-

600.00

o1-mini

1.10

0.55

4.40

o3

2.00

0.50

8.00

o3-pro

20.00

0.00

80.00

o3-mini

1.10

0.55

4.40

o3-deep-research

10.00

2.50

40.00

o4-mini

1.10

0.275

4.40

o4-mini-deep-research

2.00

0.50

8.00

codex-mini-latest

1.50

0.375

6.00

computer-use-preview

3.00

-

12.00

gpt-image-1

5.00

1.25

-

> See: [https://platform.openai.com/docs/pricing](https://platform.openai.com/docs/pricing) for more information. **Note:** Cached input pricing is denoted with `-` wherever OpenAI has not published a discounted tier. The analytics tool automatically treats missing values as zero in the CSV export.

### Daily Token Limit

ChronoTranscriber can automatically track daily OpenAI Responses usage and pause processing when a configurable token budget is exhausted. The feature is disabled by default and is controlled via `config/concurrency_config.yaml`:

```yaml
daily_token_limit:
  enabled: true          # Set to true to activate enforcement
  daily_tokens: 9000000  # Maximum tokens allowed per calendar day
```

#### How It Works

-   **Automatic tracking:** Every successful Responses API call reports `usage.total_tokens`, which are summed in `modules/token_tracker.py` with thread-safe persistence.
-   **Midnight reset:** The tracker stores state in `.chronotranscriber_token_state.json` and resets at local midnight without manual intervention.
-   **Pre-flight checks:** `WorkflowManager.process_selected_items()` verifies the remaining budget before each GPT-driven document. When the limit is exhausted, the workflow waits until the next reset or until the operator cancels with `Ctrl+C`.
-   **Live telemetry:** Running totals are emitted to the log and console at the start of processing, after each item, and once the session completes, helping teams monitor burn-in-place.

#### Recommended Usage

1.  Enable the block above and adjust `daily_tokens` to match your allocation.
2.  Keep `.chronotranscriber_token_state.json` under version-control ignore lists so local usage does not pollute repositories.
3.  Delete the state file or edit the JSON manually if you need to reset counts ahead of the daily rollover.

### System Diagnostics

ChronoTranscriber includes a diagnostics module (`modules/diagnostics/system_check.py`) that verifies system requirements and API connectivity:

System Requirement Checks:

-   **Python Version**: Verifies Python 3.8+ is installed
-   **Tesseract OCR**: Checks if Tesseract is available and configured
-   **API Key**: Validates that the required API key environment variable is set
-   **Configuration Files**: Confirms all required YAML configuration files exist

API Connectivity Diagnostics:

-   Verifies API key is properly formatted
-   Tests connectivity to the OpenAI API
-   Reports the number of accessible models
-   Provides detailed error messages for troubleshooting

You can access diagnostics programmatically:

```python
from modules.diagnostics.system_check import generate_diagnostic_report
print(generate_diagnostic_report())
```

### API Diagnostics via Batch Checker

Built into `check_batches.py`, the diagnostics tool verifies your API configuration.

```bash
python main/check_batches.py
```

Diagnostic Checks:

1.  API Key Presence: Verifies `OPENAI_API_KEY` environment variable is set
2.  Model Listing: Attempts to list available models
3.  Batch API Access: Verifies access to the Batch API endpoint

## Architecture

ChronoTranscriber follows a modular architecture that separates concerns and promotes maintainability.

### Directory Structure

```
ChronoTranscriber/
├── config/                    # Configuration files
│   ├── concurrency_config.yaml
│   ├── image_processing_config.yaml
│   ├── model_config.yaml
│   └── paths_config.yaml
├── fine_tuning/                # Fine-tuning dataset preparation tools
│   └── build_openai_vision_sft_jsonl.py
├── main/                      # CLI entry points
│   ├── cancel_batches.py      # Cancel pending batch jobs
│   ├── check_batches.py       # Monitor and download batch results
│   ├── cost_analysis.py       # Token usage and cost reporting
│   ├── repair_transcriptions.py # Repair failed transcriptions
│   └── unified_transcriber.py # Main transcription workflow
├── modules/                   # Core application modules
│   ├── config/               # Configuration loading and service
│   ├── core/                 # Core utilities, workflow, CLI args
│   ├── infra/                # Logging, concurrency, async tasks
│   ├── io/                   # File I/O and path utilities
│   ├── llm/                  # LLM integration
│   │   ├── providers/        # LangChain provider implementations
│   │   ├── batch/            # OpenAI Batch API processing
│   │   └── ...               # Transcriber, schemas, capabilities
│   ├── operations/           # High-level operations
│   ├── processing/           # Image and PDF processing
│   └── ui/                   # User interface and prompts
├── schemas/                   # JSON schemas for structured outputs
│   ├── markdown_transcription_schema.json
│   ├── plain_text_transcription_schema.json
│   ├── plain_text_transcription_with_markers_schema.json
│   └── swiss_address_book_schema.json
├── system_prompt/             # System prompt templates
│   └── system_prompt.txt
├── additional_context/        # Optional domain context
│   └── additional_context.txt
├── LICENSE
├── README.md
└── requirements.txt
```

### Module Structure

ChronoTranscriber follows a modular architecture with clear separation of concerns:

-   `modules/config/`: Configuration loading, validation, and centralized ConfigService
-   `modules/core/`: Core utilities including console printing, workflow management, path handling, and CLI argument parsing
-   `modules/infra/`: Infrastructure layer providing logging, concurrency control, and async task management
-   `modules/io/`: File I/O operations including path validation, directory scanning, and output management
-   `modules/llm/`: LLM integration layer with the following submodules:
    -   `providers/`: LangChain-based provider implementations (OpenAI, Anthropic, Google, OpenRouter)
    -   `batch/`: OpenAI Batch API processing
    -   `transcriber.py`: High-level transcription interface
    -   `model_capabilities.py`: Model capability detection and parameter filtering
    -   `schemas.py`: Pydantic models for structured outputs
-   `modules/operations/`: High-level operation orchestration (batch checking, repair workflows, cost analysis)
-   `modules/processing/`: Document processing including PDF rendering, image preprocessing, and text formatting
-   `modules/ui/`: User interface components including interactive prompts, styled output, and navigation
-   `modules/diagnostics/`: System health checks including Python version, Tesseract availability, API key validation, and configuration file verification
-   `modules/testing/`: Test fixtures and utilities for development and validation

### LangChain Provider Architecture

The LLM integration uses LangChain for a unified multi-provider interface:

```
modules/llm/providers/
├── base.py              # Abstract BaseProvider class and common interfaces
├── factory.py           # Provider factory with auto-detection
├── openai_provider.py   # OpenAI implementation (GPT-5, o-series, GPT-4o)
├── anthropic_provider.py # Anthropic implementation (Claude family)
├── google_provider.py   # Google implementation (Gemini family)
└── openrouter_provider.py # OpenRouter implementation (200+ models)
```

Each provider handles:

-   Model capability detection and parameter validation
-   Automatic filtering of unsupported parameters via LangChain's `disabled_params`
-   Retry logic with exponential backoff (via LangChain's `max_retries`)
-   Token usage tracking (via LangChain's `response_metadata`)
-   Structured output parsing (via LangChain's `with_structured_output`)

### Windows Path Length Handling

ChronoTranscriber implements robust handling for Windows MAX_PATH (260 character) limitations:

-   Safe Directory Naming: Long document names are automatically truncated with content-based hashes for directory structures
-   Preserved File Names: Output files retain their original proper names without hash suffixes
-   Automatic Path Resolution: The system uses extended-length path syntax when needed on Windows 10 1607+
-   Hybrid Strategy: Directories use hash-based safe names (e.g., `Long_document_name-a3f8d9e2/`) while files preserve original names (e.g., `Long document name_transcription.txt`)

This approach ensures reliable processing of documents with long filenames while maintaining human-readable output files. The implementation follows production-ready patterns similar to npm and Git for handling filesystem limitations.

### Operations Layer

ChronoTranscriber separates orchestration logic from CLI entry points to improve testability and maintainability:

-   High-level operations live in `modules/operations/` (e.g., batch checking, repair workflows)
-   CLI entry points in `main/` are thin wrappers that delegate to operations modules
-   This design pattern allows operations to be reused, tested independently, and invoked programmatically

## Frequently Asked Questions

### General Questions

**Q: Which AI provider should I choose?**

A: It depends on your priorities:

-   **OpenAI (GPT-5-mini)**: Best balance of cost and quality. Excellent for general transcription. Supports batch processing for 50% cost reduction.
-   **Anthropic (Claude 3.5 Sonnet)**: Superior for complex layouts and academic papers. Better at preserving formatting and structure.
-   **Google (Gemini 2.0 Flash)**: Fastest processing and lowest cost. Good for straightforward documents.
-   **OpenRouter**: Access to 200+ models. Useful for trying different models without managing multiple API keys.

Start with OpenAI's gpt-5-mini with low reasoning effort for cost-effective results.

**Q: How much does it cost to transcribe documents?**

A: Costs vary by provider and model. Examples with OpenAI gpt-5-mini:

-   Single page (300 DPI image): ~$0.01-0.02
-   100-page PDF (synchronous): ~$1-2
-   100-page PDF (batch mode): ~$0.50-1 (50% discount)
-   1000-page archive (batch): ~$5-10

Use the cost analysis tool to track actual spending: `python main/cost_analysis.py`

**Q: Should I use batch processing or synchronous mode?**

A: Use batch processing when:

-   Processing more than 50 pages
-   Cost is a priority (50% cheaper)
-   You can wait 24 hours for results
-   Processing large archives

Use synchronous mode when:

-   You need results immediately
-   Processing fewer than 50 pages
-   Testing or debugging

**Q: Can I process documents offline without an API key?**

A: Yes. Use Tesseract OCR for completely offline processing:

```bash
python main/unified_transcriber.py --type pdfs --method tesseract --input ./docs --output ./results
```

Tesseract is free but generally produces lower quality results than AI models.

### Configuration Questions

**Q: How do I switch between providers?**

A: Edit `config/model_config.yaml`:

```yaml
transcription_model:
  provider: anthropic  # Change to: openai, anthropic, google, or openrouter
  name: claude-3-5-sonnet  # Update model name for the provider
```

Then set the appropriate API key environment variable.

**Q: How do I control costs with daily token limits?**

A: Enable daily token budget in `config/concurrency_config.yaml`:

```yaml
daily_token_limit:
  enabled: true
  daily_tokens: 9000000  # Adjust to your budget
```

Processing automatically pauses when the limit is reached and resumes the next day.

**Q: What's the difference between reasoning effort levels (low/medium/high)?**

A: For reasoning models (GPT-5, o-series, Claude 4.5, Gemini 3):

-   **Low**: Fastest, cheapest, good for straightforward documents
-   **Medium**: Balanced quality and cost (recommended default)
-   **High**: Best quality for complex documents, slower and more expensive

For most historical documents, low or medium effort is sufficient.

**Q: How do I customize the output format?**

A: Use different schemas in `schemas/` directory:

-   `markdown_transcription_schema.json`: Formatted text with headings, LaTeX equations
-   `plain_text_transcription_schema.json`: Simple plain text
-   `plain_text_transcription_with_markers_schema.json`: Plain text with page markers

Specify with `--schema` flag or select in interactive mode.

### Processing Questions

**Q: What happens if some pages fail to transcribe?**

A: Failed pages are marked in the output with:

-   `[transcription error: page_name]`: API or processing error
-   `[No transcribable text]`: Model determined page has no readable text
-   `[Transcription not possible]`: Model couldn't process the page

Use the repair tool to retry failed pages:

```bash
python main/repair_transcriptions.py --transcription ./results/document_transcription.txt --errors-only
```

**Q: How do I process documents with mixed content (PDFs, images, ebooks)?**

A: Use auto mode:

```bash
python main/unified_transcriber.py --auto --input ./mixed_folder --output ./results
```

Auto mode automatically detects file types and selects the best processing method for each.

**Q: Can I process password-protected PDFs?**

A: No, ChronoTranscriber does not support encrypted or password-protected PDFs. You must decrypt them first using external tools.

**Q: How do I preserve page numbers in the output?**

A: Use a schema with page markers:

```bash
python main/unified_transcriber.py --type pdfs --method gpt --schema plain_text_transcription_with_markers_schema --input ./docs --output ./results
```

Page numbers will appear as `<page_number>5</page_number>` in the output.

### Batch Processing Questions

**Q: How do I check if my batch jobs are complete?**

A: Run the batch checker:

```bash
python main/check_batches.py
```

This shows status of all batches and automatically downloads completed results.

**Q: Can I cancel a batch job?**

A: Yes, use the cancel tool:

```bash
python main/cancel_batches.py
```

This lists all active batches and allows you to cancel them. Note that you're charged for any processing that occurred before cancellation.

**Q: Where are batch results stored?**

A: Results are saved to the output directory you specified when submitting the batch. Temporary tracking files (`*_temporary.jsonl`) are kept in the same location until results are downloaded.

**Q: What if batch processing fails?**

A: ChronoTranscriber automatically falls back to synchronous processing if batch submission fails. Check logs for details. Common causes:

-   Provider doesn't support batch processing (OpenRouter)
-   API key lacks batch access permissions
-   Network connectivity issues

### Technical Questions

**Q: What image formats are supported?**

A: PNG, JPEG, JPG, WEBP, BMP, TIFF. Images are automatically preprocessed based on the selected provider.

**Q: What's the maximum PDF size?**

A: No hard limit, but very large PDFs (1000+ pages) should use batch processing. Individual page images are limited by provider constraints (typically 20MB per image after preprocessing).

**Q: Can I run multiple transcription jobs simultaneously?**

A: Yes, but be mindful of:

-   API rate limits (configure in `concurrency_config.yaml`)
-   Daily token budgets
-   System memory for image preprocessing

Each job runs independently and can be monitored separately.

**Q: How do I integrate ChronoTranscriber into my existing pipeline?**

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

All scripts return proper exit codes (0 for success, 1 for errors).

## Troubleshooting

### Common Issues

#### API key not found

Solution: Ensure `OPENAI_API_KEY` environment variable is set:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"

# Linux/macOS
export OPENAI_API_KEY=your_api_key_here
```

#### Tesseract not found

Solution: Install Tesseract and configure path in `config/image_processing_config.yaml`:

```yaml
tesseract_image_processing:
  ocr:
    tesseract_cmd: 'C:Program Files (x86)Tesseract-OCRtesseract.exe'
```

#### Batch job missing from tracking file

Solution: Run `check_batches.py` which automatically repairs missing batch IDs from debug artifacts:

```bash
python main/check_batches.py
```

#### Model not supported

Solution: Verify your model name matches the provider's expected format:

-   OpenAI: `gpt-5-mini`, `o3`, `gpt-4o`
-   Anthropic: `claude-sonnet-4-5-20250929`, `claude-3-5-sonnet-20241022`
-   Google: `gemini-3-pro`, `gemini-2.5-pro`
-   OpenRouter: `openai/gpt-5.1`, `anthropic/claude-sonnet-4-5`

Check `modules/llm/providers/` for supported models per provider.

#### Provider API key not found

Solution: Set the environment variable for your chosen provider:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_key"      # OpenAI
$env:ANTHROPIC_API_KEY="your_key"   # Anthropic
$env:GOOGLE_API_KEY="your_key"      # Google
$env:OPENROUTER_API_KEY="your_key"  # OpenRouter

# Linux/macOS
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
```

#### Images not processing correctly

Solution:

1.  Check image format (PNG and JPEG are supported)
2.  Verify image preprocessing settings in `config/image_processing_config.yaml`
3.  Try adjusting `llm_detail` parameter (high vs low vs auto)
4.  Check logs in the configured `logs_dir` for detailed error messages

#### Temporary files not being cleaned up

Solution: Temporary JSONL files are only deleted when:

-   The final transcription file was successfully written
-   All batches in that job completed successfully
-   `retain_temporary_jsonl` is set to `false`

If batches fail or are still in progress, temporary files are retained for recovery.

#### Page order is incorrect in output

Solution: ChronoTranscriber uses a multi-level ordering strategy. If pages are out of order:

1.  Check that page numbers are correctly embedded in filenames
2.  Verify that the JSON schema includes page number extraction
3.  Review the `custom_id` format in batch requests
4.  Check logs for ordering strategy decisions

### Debug Artifacts

ChronoTranscriber creates several debug artifacts to help troubleshoot issues:

-   `<job>_batch_submission_debug.json`: Contains batch IDs, image count, chunk size, and submission timestamp
-   `<job>_temporary.jsonl`: Tracks batch requests and responses with full metadata
-   Log files: Detailed execution logs in the configured `logs_dir` with timestamps and stack traces

### Getting Help

If you encounter issues not covered here:

1.  Check logs: Review detailed error messages in your configured `logs_dir`
2.  Run diagnostics: Execute `python main/check_batches.py` to verify API configuration
3.  Validate configuration: Ensure all YAML files are properly formatted
4.  Verify directories: Confirm all required directories exist with proper permissions
5.  Review requirements: Verify all dependencies are installed correctly
6.  Check model access: Ensure your OpenAI account has access to the selected model

## Contributing

Contributions are welcome! Here's how you can help improve ChronoTranscriber:

### Reporting Issues

When reporting bugs or issues, please include:

-   Description: Clear description of the problem
-   Steps to Reproduce: Detailed steps to reproduce the issue
-   Expected Behavior: What you expected to happen
-   Actual Behavior: What actually happened
-   Environment: OS, Python version, relevant package versions
-   Configuration: Relevant sections from your config files (remove sensitive information)
-   Logs: Relevant log excerpts showing the error

### Suggesting Features

Feature suggestions are appreciated. Please provide:

-   Use Case: Describe the problem or need
-   Proposed Solution: Your idea for addressing it
-   Alternatives: Other approaches you've considered
-   Impact: Who would benefit and how

### Code Contributions

If you'd like to contribute code:

1.  Fork the repository and create a feature branch
2.  Follow the existing code style and architecture patterns
3.  Add tests for new functionality where applicable
4.  Update documentation including this README and inline comments
5.  Test thoroughly with both Tesseract and OpenAI backends
6.  Submit a pull request with a clear description of your changes

### Development Guidelines

-   Modularity: Keep functions focused and modules organized
-   Error Handling: Use try-except blocks with informative error messages
-   Logging: Use the logger for debugging information
-   Configuration: Use YAML configuration files rather than hardcoding values
-   User Experience: Provide clear prompts and feedback in CLI interactions
-   Documentation: Update docstrings and README for any interface changes

### Areas for Contribution

Potential areas where contributions would be valuable:

-   Additional OCR backends: Support for other OCR engines or APIs
-   Enhanced preprocessing: Additional image enhancement algorithms
-   Output formats: Support for different output formats (JSON, XML, etc.)
-   Testing: Unit tests and integration tests
-   Documentation: Tutorials, examples, and use case documentation
-   Performance optimization: Improved concurrent processing or caching
-   Error recovery: Enhanced error handling and recovery mechanisms

## Development

### Recent Updates

**December 2025: Dependency Updates and Maintenance**

All dependencies updated to latest stable versions:

-   LangChain ecosystem: Updated to langchain 1.2.0, langchain-core 1.2.5, with enhanced provider integrations
-   OpenAI SDK: Updated to 2.14.0 with latest API features
-   LangChain providers: langchain-openai 1.1.6, langchain-anthropic 1.3.0, langchain-google-genai 4.1.2
-   Image processing: scikit-image 0.26.0, PyMuPDF 1.26.7 with improved performance
-   Scientific computing: numpy 2.2.6 (maintained for opencv compatibility), networkx 3.6.1
-   Infrastructure: Updated async libraries, improved websocket support, enhanced UUID utilities
-   All 24 outdated packages updated while maintaining full backward compatibility
-   Core functionality verified: All modules import successfully, CLI and interactive modes operational

**November 2025: Version 3.0 - Multi-Provider LangChain Integration**

ChronoTranscriber now supports multiple AI providers through LangChain:

-   OpenAI, Anthropic, Google, and OpenRouter providers with unified interface
-   Automatic capability detection and parameter filtering for reasoning models
-   EPUB support for native text extraction from ebooks
-   Auto mode for intelligent method selection based on document type
-   Cost analysis utilities for tracking API usage and spend
-   Daily token budget management with automatic enforcement
-   Additional context injection for domain-specific transcription guidance
-   Swiss address book schema for specialized historical document extraction
-   System diagnostics module for health checks and troubleshooting

See [RELEASE_NOTES_v3.0.md](RELEASE_NOTES_v3.0.md) for full details.

**October 2025: Version 2.0 - Dual-Mode Operation**

Major release introducing dual-mode operation:

-   Interactive mode with navigation support and visual prompts
-   CLI mode for automation and scripting
-   Critical Windows Unicode encoding fix
-   Batch API service tier compatibility fix

See [RELEASE_NOTES_v2.0.md](RELEASE_NOTES_v2.0.md) for full details.

### Development Guidelines

-   Code follows PEP 8 style guidelines
-   New features include appropriate tests
-   Documentation is updated for API changes
-   Commit messages are clear and descriptive

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.