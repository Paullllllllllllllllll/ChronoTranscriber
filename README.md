# ChronoTranscriber

A Python-based tool designed for researchers and archivists to transcribe historical documents from PDFs, EPUB ebooks, or image folders using either local OCR (Tesseract) or modern vision-language models via OpenAI's API. ChronoTranscriber provides structured JSON outputs, scalable batch processing, and robust error recovery for large-scale document digitization projects.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Batch Processing](#batch-processing)
- [Utilities](#utilities)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Overview

ChronoTranscriber is designed for researchers and archivists who need to (cheaply and comfortably) transcribe historical documents at scale. The tool supports multiple processing backends, provides fine-grained control over image preprocessing, and ensures reproducible results through structured JSON outputs. Also works well (or even better) with non-historical documents (academic papers, books, ebooks, etc.).
Meant to be used in conjunction with [ChronoMiner](https://github.com/Paullllllllllllllllll/ChronoMiner) and [ChronoDownloader](https://github.com/Paullllllllllllllllll/ChronoDownloader) for a full historical document retrieval, transcription and data extraction pipeline.

### Execution Modes

ChronoTranscriber supports two execution modes to accommodate different workflows and user preferences:
- **Interactive Workflow:** Launch `python main/unified_transcriber.py` with no arguments or keep `general.interactive_mode: true` in `config/paths_config.yaml` to step through a guided UI powered by `modules/ui/workflows.py`. The prompts surface available models, schemas, and preprocessing options while letting you review every choice before execution. Navigation features allow you to go back to previous steps (press 'b') or quit at any time (press 'q').
- **CLI / Automation Mode:** Provide command-line arguments (see `python main/unified_transcriber.py --help`) or set `general.interactive_mode: false` to run unattended jobs. Argument parsing is handled by `modules/core/cli_args.py`, giving you scriptable control for cron jobs, CI pipelines, and large batch workflows.

### Document Processing

- PDF Transcription: Process PDF documents with automatic text extraction fallback or page-to-image rendering
- EPUB Extraction: Natively extract structured text chapters from EPUB ebooks without OCR
  - Supports EPUB 2.0 and EPUB 3.0
  - Automatically detects and extracts text from EPUB chapters
  - Preserves EPUB chapter structure and metadata
- Image Folder Transcription: Process directories containing scanned page images (PNG, JPEG)
- Auto Mode Discovery: Scan a mixed directory and automatically choose native, Tesseract, or GPT transcription per file
- Multi-Page Support: Preserve page ordering and handle documents with hundreds or thousands of pages
- Preprocessing Pipeline: Configurable image enhancement including grayscale conversion, transparency handling, deskewing, denoising, and binarization

### OCR Backends

ChronoTranscriber exposes two primary OCR backends that can be selected from the interactive prompts, via CLI flags, or by setting defaults in `config/model_config.yaml`. You can mix and match backends per document type, and auto mode will dynamically choose the best fit based on the source file and your configuration.

#### Tesseract (Local)
- Fully offline processing with no external API calls
- Configurable engine modes and page segmentation
- Advanced preprocessing pipeline for improved accuracy

#### OpenAI Vision-Language Models
- Supported Models: GPT-4o, GPT-4.1, GPT-5, GPT-5-mini, o1, o3, and o-series models
- Processing Modes: 
  - Synchronous: Real-time streaming responses
  - Asynchronous: Batch processing for large jobs
- Model-Specific Features:
  - GPT-5 series: Reasoning effort and text verbosity controls
  - o-series (o1, o3): Reasoning capabilities with automatic feature detection
  - Classic models: Temperature, top_p, frequency_penalty, and presence_penalty controls
- Automatic Capability Detection: The system validates model capabilities and adjusts parameters accordingly
- Minimal User Prompts: Requests send only the text "The image:" alongside the vision payload so that all instructions remain in the system prompt

### Structured Outputs

- JSON Schema Enforcement: Models return structured responses conforming to predefined schemas
- Multiple Schema Support: Choose between markdown formatting or plain text output
- Custom Schemas: Create your own schemas for specialized transcription needs
- Fallback Parsing: Graceful handling of malformed responses with automatic fallback to raw text

### Reliability and Retry Control

- Multi-tier Retry Strategy: Automatic exponential backoff for transient API errors (429, 5xx, network timeouts) with jitter to avoid synchronized retries
- Transcription-aware Retries: Optional, per-condition retries when the model returns `no_transcribable_text` or `transcription_not_possible`, configurable in `config/concurrency_config.yaml`
- Retry Hints: Honors server-provided `Retry-After` headers and logs each attempt for clear observability
- Safe Defaults: Sensible baseline retry settings ship out of the box, with the ability to disable or tighten retries per project requirements

### Batch Processing

- Scalable Submission: Submit large document sets as OpenAI Batch jobs
- Smart Chunking: Automatic request splitting with 150 MB chunk size limit (below the 180 MB API limit)
- Data URL Encoding: Images are base64-encoded and embedded directly in requests (no external hosting required)
- Metadata Tracking: Each request includes image name, page number, and order index for reliable reconstruction
- Debug Artifacts: Submission metadata saved for batch tracking and repair operations

## System Requirements

### Software Dependencies

- Python: 3.8 or higher
- Tesseract OCR (optional): Required only if using local OCR backend
  - Windows: Install from official installer and configure path in `image_processing_config.yaml`
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

### API Requirements

- OpenAI API Key: Required for using vision-language models and batch processing
  - Set as environment variable: `OPENAI_API_KEY=your_key_here`
  - Ensure your account has access to the Responses API and Batch API

### Python Packages

All Python dependencies are listed in `requirements.txt`. Key packages include:

- `openai>=1.57.4`: OpenAI SDK for API interactions
- `PyMuPDF>=1.25.1`: PDF processing
- `pillow>=11.0.0`: Image manipulation
- `pytesseract>=0.3.13`: Tesseract OCR wrapper
- `opencv-python>=4.10.0`: Advanced image processing
- `scikit-image>=0.25.0`: Scientific image processing
- `pyyaml>=6.0.2`: Configuration file parsing
- `aiohttp>=3.11.10`: Asynchronous HTTP requests
- `tqdm>=4.67.1`: Progress bars

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/ChronoTranscriber.git
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

### Install Tesseract (Optional)

If you plan to use local OCR, install Tesseract:

- Windows: Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

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

## Configuration

ChronoTranscriber uses four YAML configuration files located in the `config/` directory. Each file controls a specific aspect of the pipeline.

### 1. Model Configuration (`model_config.yaml`)

Controls which model to use and its behavioral parameters.

```yaml
transcription_model:
  name: gpt-5-mini  # Options: gpt-4o, gpt-4.1, gpt-5, gpt-5-mini, o1, o3
  max_output_tokens: 128000
  
  # GPT-5 and o-series only
  reasoning:
    effort: medium  # Options: low, medium, high
  
  # GPT-5 only
  text:
    verbosity: medium  # Options: low, medium, high
  
  # Classic models (GPT-4o, GPT-4.1) only
  temperature: 0.01
  top_p: 1.0
  frequency_penalty: 0.01
  presence_penalty: 0.01
```

Key Parameters:

- `name`: Model identifier
- `max_output_tokens`: Maximum tokens the model can generate per request
- `reasoning.effort`: Controls reasoning depth for GPT-5 and o-series models (low, medium, high)
- `text.verbosity`: Controls response verbosity for GPT-5 models (low, medium, high)
- `temperature`: Controls randomness (0.0-2.0)
- `top_p`: Nucleus sampling probability (0.0-1.0)
- `frequency_penalty`: Penalizes token repetition (-2.0 to 2.0)
- `presence_penalty`: Penalizes repeated topics (-2.0 to 2.0)

### 2. Paths Configuration (`paths_config.yaml`)

Defines input/output directories, operation mode, and path resolution behavior.

```yaml
general:
  interactive_mode: true  # Toggle between interactive prompts (true) or CLI mode (false)
  retain_temporary_jsonl: true
  input_paths_is_output_path: false
  logs_dir: './logs'
  keep_preprocessed_images: true
  # Auto mode: PDF processing settings
  auto_mode_pdf_use_ocr_for_scanned: true
  auto_mode_pdf_use_ocr_for_searchable: false
  auto_mode_pdf_ocr_method: 'tesseract'
  # Auto mode: Image processing settings
  auto_mode_image_ocr_method: 'tesseract'

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
  Auto:
    input: './input/auto'
    output: './output/auto'
```

Key Parameters:

- `interactive_mode`: Controls operation mode (true for interactive prompts, false for CLI mode)
- `retain_temporary_jsonl`: Keep temporary JSONL files after batch processing completes
- `input_paths_is_output_path`: Write outputs to the same directory as inputs (auto mode honors this when all files share a parent)
- `logs_dir`: Directory for log files
- `keep_preprocessed_images`: Retain preprocessed images after transcription
- `auto_mode_pdf_use_ocr_for_scanned`: Force OCR for scanned/non-searchable PDFs (if false, attempts native extraction)
- `auto_mode_pdf_use_ocr_for_searchable`: Force OCR even for searchable PDFs, bypassing native text extraction
- `auto_mode_pdf_ocr_method`: OCR method to use when OCR is forced for PDFs (`tesseract` or `gpt`)
- `auto_mode_image_ocr_method`: OCR method for images (`tesseract` or `gpt`; automatically falls back if unavailable)
- `transcription_prompt_path` (optional): Custom system prompt file path
- `transcription_schema_path` (optional): Custom JSON schema file path

### 3. Image Processing Configuration (`image_processing_config.yaml`)

Controls image preprocessing for both API and Tesseract backends.

#### API Image Processing

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

- `target_dpi`: DPI for rendering PDF pages to images
- `grayscale_conversion`: Convert images to grayscale to reduce noise
- `handle_transparency`: Flatten transparent images onto white background
- `llm_detail`: Controls image fidelity sent to API (high: better accuracy, higher token usage; low: faster and cheaper; auto: let model decide)
- `jpeg_quality`: JPEG compression quality (1-100) for processed images
- `resize_profile`: Image resizing strategy before API submission
- `low_max_side_px`: Maximum side length in pixels for low-detail resizing
- `high_target_box`: Target dimensions [width, height] for high-detail resizing

#### Tesseract Image Processing

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
    output_format: png  # Options: png, tiff
    preserve_resolution: true
    embed_dpi_metadata: true
```

Preprocessing Pipeline:

1. Flatten Alpha: Remove transparency by compositing onto white background
2. Grayscale Conversion: Convert to grayscale for consistent processing
3. Invert Colors: Ensure dark text on light background (auto/always/never)
4. Deskew: Detect and correct page rotation
5. Denoise: Remove noise using median or bilateral filtering
6. Binarization: Convert to black and white using Sauvola, adaptive, or Otsu thresholding
7. Morphology: Apply morphological operations to enhance text (optional)
8. Border Addition: Add white border to prevent text cropping at edges

### 4. Concurrency Configuration (`concurrency_config.yaml`)

Controls parallel processing and retry behavior.

```yaml
concurrency:
  transcription:
    concurrency_limit: 250
    delay_between_tasks: 0.005
    service_tier: flex  # Options: auto, default, flex, priority
    batch_chunk_size: 50
    
    retry:
      attempts: 10
      wait_min_seconds: 4
      wait_max_seconds: 60
      jitter_max_seconds: 1
      
      transcription_failures:
        no_transcribable_text_retries: 1
        transcription_not_possible_retries: 3
        wait_min_seconds: 2
        wait_max_seconds: 30
        jitter_max_seconds: 1
  
  image_processing:
    concurrency_limit: 24
    delay_between_tasks: 0.0005
```

Key Parameters:

- `concurrency_limit`: Maximum number of concurrent tasks
- `delay_between_tasks`: Delay in seconds between starting tasks
- `service_tier`: OpenAI service tier for rate limiting and processing speed (auto, default, flex, priority)
  - Note: Service tiers apply only to synchronous API calls; batch processing automatically omits this parameter
- `batch_chunk_size`: Number of requests per batch part file (affects chunking)
- Retry settings: Exponential backoff configuration for transient API failures
  - `attempts`: Maximum retry attempts per request
  - `wait_min_seconds`: Minimum wait time before retry
  - `wait_max_seconds`: Maximum wait time before retry
  - `jitter_max_seconds`: Random jitter to prevent synchronized retries
- `transcription_failures`: Optional retries applied when the model responds with `no_transcribable_text` or `transcription_not_possible`
  - Individual counters let you disable or tighten retries per condition (set to 0 to accept the first response)
  - Independent wait controls mirror the primary retry strategy

### Additional Context Guidance

ChronoTranscriber includes an optional domain context file at `additional_context/additional_context.txt`. When content exists, the CLI prompts you to opt in before each processing run. Accepting loads the file into the system prompt at the `{{ADDITIONAL_CONTEXT}}` marker. If you decline or the file is missing, the pipeline inserts the literal word "Empty" so the prompt receives an explicit signal that no context is provided. Context updates do not require code changes; edit the file and rerun the workflow to apply the new guidance.

### Custom Transcription Schemas

ChronoTranscriber supports custom JSON schemas for controlling transcription output format.

#### Included Schemas

- `markdown_transcription_schema.json` (default): Produces markdown-formatted transcriptions with LaTeX equations, headings, and formatting
- `plain_text_transcription_schema.json`: Produces plain text transcriptions without formatting

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

Creating Custom Schemas:

1. Place your schema file in the `schemas/` directory
2. Follow the structure above with your custom field descriptions
3. The `name` field will appear in the interactive schema selection menu
4. Field descriptions instruct the model on how to format its output

## Usage

### Main Transcription Workflow

The primary entry point is `main/unified_transcriber.py`, which provides a workflow for transcribing documents.

#### Interactive Mode

Run the transcriber without arguments:

```bash
python main/unified_transcriber.py
```

The interactive interface guides you through:

1. Select Processing Type (PDF or Images)
2. Select OCR Backend (Tesseract or GPT)
3. Select Processing Mode (Synchronous or Batch, GPT only)
4. Select JSON Schema (GPT only)
5. Configure Additional Context (GPT only)
6. Select Files/Folders
7. Review and Confirm

#### CLI Mode

Process documents with command-line arguments:

```bash
# Process a single PDF with GPT in batch mode
python main/unified_transcriber.py --type pdf --backend gpt --mode batch --schema markdown --input document.pdf

# Process image folder with Tesseract
python main/unified_transcriber.py --type images --backend tesseract --input ./scans/

# View all options
python main/unified_transcriber.py --help
```

#### Providing Additional Context

- Edit: Update `additional_context/additional_context.txt` with project-specific transcription notes
- Skip: Leave the file blank or choose not to load context; the prompt will receive the literal word "Empty"
- Refresh: Modify the file between runs to iterate on instructions without editing code

#### Retry Behavior and Observability

ChronoTranscriber applies a layered retry strategy automatically during GPT-based transcription:

- General API errors (429, 5xx, network timeouts) retry with exponential backoff and jitter
- Transcription outcomes flagged as `no_transcribable_text` or `transcription_not_possible` can trigger additional retries when enabled
- Logs emitted in the console and `logs_dir` include attempt counts and wait durations
- Tuning: Increase `wait_min_seconds`/`wait_max_seconds` for aggressive rate limits, or set individual retry counters to 0 to accept the first model response

### Output Files

Transcription outputs are saved to the configured output directories:

- PDFs: `file_paths.PDFs.output` (or input directory if `input_paths_is_output_path: true`)
- Images: `file_paths.Images.output` (or input directory if `input_paths_is_output_path: true`)

Output Naming Convention:

- `<original_name>_transcription.txt`: Final transcription text file
- `<original_name>_temporary.jsonl`: Temporary batch tracking file (deleted after successful completion unless `retain_temporary_jsonl: true`)
- `<original_name>_batch_submission_debug.json`: Batch metadata for tracking and repair

## Batch Processing

Batch processing allows you to submit hundreds or thousands of pages for asynchronous transcription via OpenAI's Batch API.

### How Batch Processing Works

1. Image Encoding: Images are base64-encoded as data URLs
2. Request Chunking: Requests are split into chunks (≤150 MB per chunk)
3. Metadata Tagging: Each request includes custom_id, image name, page number, and order index
4. Batch Submission: Chunks are submitted as separate batch jobs
5. Debug Artifact: Submission metadata is saved in `<job>_batch_submission_debug.json`

### Monitoring Batch Jobs

Use `check_batches.py` to monitor batch job status and download completed results.

```bash
python main/check_batches.py
```

What It Does:

- Scans all configured directories for temporary JSONL files
- Checks status of all batch jobs (in_progress, completed, failed, etc.)
- Repairs missing batch IDs using debug artifacts
- Downloads results when all batches in a job are complete
- Merges outputs using multi-level ordering strategy
- Diagnoses API and model errors with helpful messages
- Cleans up temporary files after successful completion (optional)

Multi-Level Ordering Strategy:

1. Explicit order info from request metadata
2. Custom ID index from batch responses
3. Embedded page number from structured outputs
4. Page number parsed from filename
5. Stable fallback index

### Cancelling Batch Jobs

Use `cancel_batches.py` to cancel pending or in-progress batch jobs.

```bash
python main/cancel_batches.py
```

What It Does:

- Lists all batch jobs in your account (with pagination)
- Displays summary by status (completed, in_progress, failed, etc.)
- Identifies terminal batches (completed, expired, cancelled, failed)
- Cancels all non-terminal batches
- Shows detailed summary of cancellation results

## Utilities

### Token Cost Analysis

ChronoMiner bundles a lightweight analytics utility that inspects preserved temporary `.jsonl` responses and produces detailed cost estimates for every processed file. The workflow is implemented in `main/cost_analysis.py`, which orchestrates helper logic contained in `modules/operations/cost_analysis.py` and formatted output helpers in `modules/ui/cost_display.py`.

#### When to Run It

- Preserve temporary `.jsonl` files (`retain_temporary_jsonl: true` in `config/paths_config.yaml`).
- After processing is complete, run the analysis to quantify spend across synchronous and batch jobs.
- Use the report to validate budgeting assumptions or to decide whether to switch schemas or models.

#### Execution Modes

- **Interactive UI:** `python -m main.cost_analysis`
  - Mirrors the standard UI look and feel.
  - Automatically locates `.jsonl` files based on schema path configuration.
  - Displays aggregated token totals, per-file summaries, and optional CSV export prompts.
- **CLI Mode:** `python -m main.cost_analysis --save-csv --output path/to/report.csv`
  - Suitable for automation or scheduled reporting.
  - Flags:
    - `--save-csv`: Persist results to CSV (defaults to the folder that contains the first `.jsonl`).
    - `--output`: Override the target CSV path.
    - `--quiet`: Suppress console breakdown and emit only essential status messages.

#### Output Features

- Aggregated totals for uncached input tokens, cached tokens, output tokens, reasoning tokens, and overall totals.
- Dual pricing: standard per-million-token rates and an automatic 50% discount column that models batched/flex billing tiers.
- Model normalization: date-stamped variants (e.g., `gpt-5-mini-2025-08-07`) are mapped to their parent pricing profile before calculations.
- CSV export includes a per-file ledger plus a consolidated summary row that mirrors the on-screen totals.

#### Supported Pricing Profiles (USD per 1M tokens)

| Model | Input | Cached Input | Output |
| --- | --- | --- | --- |
| gpt-5 | 1.25 | 0.125 | 10.00 |
| gpt-5-mini | 0.25 | 0.025 | 2.00 |
| gpt-5-nano | 0.05 | 0.005 | 0.40 |
| gpt-5-chat-latest | 1.25 | 0.125 | 10.00 |
| gpt-5-codex | 1.25 | 0.125 | 10.00 |
| gpt-4.1 | 2.00 | 0.50 | 8.00 |
| gpt-4.1-mini | 0.40 | 0.10 | 1.60 |
| gpt-4.1-nano | 0.10 | 0.025 | 0.40 |
| gpt-4o | 2.50 | 1.25 | 10.00 |
| gpt-4o-2024-05-13 | 5.00 | - | 15.00 |
| gpt-4o-mini | 0.15 | 0.075 | 0.60 |
| gpt-4o-realtime-preview | 5.00 | 2.50 | 20.00 |
| gpt-4o-mini-realtime-preview | 0.60 | 0.30 | 2.40 |
| gpt-4o-audio-preview | 2.50 | - | 10.00 |
| gpt-4o-mini-audio-preview | 0.15 | - | 0.60 |
| gpt-4o-search-preview | 2.50 | - | 10.00 |
| gpt-4o-mini-search-preview | 0.15 | - | 0.60 |
| gpt-audio | 2.50 | 0.00 | 10.00 |
| o1 | 15.00 | 7.50 | 60.00 |
| o1-pro | 150.00 | - | 600.00 |
| o1-mini | 1.10 | 0.55 | 4.40 |
| o3 | 2.00 | 0.50 | 8.00 |
| o3-pro | 20.00 | 0.00 | 80.00 |
| o3-mini | 1.10 | 0.55 | 4.40 |
| o3-deep-research | 10.00 | 2.50 | 40.00 |
| o4-mini | 1.10 | 0.275 | 4.40 |
| o4-mini-deep-research | 2.00 | 0.50 | 8.00 |
| codex-mini-latest | 1.50 | 0.375 | 6.00 |
| computer-use-preview | 3.00 | - | 12.00 |
| gpt-image-1 | 5.00 | 1.25 | - |

> See: https://platform.openai.com/docs/pricing for more information.
> **Note:** Cached input pricing is denoted with `-` wherever OpenAI has not published a discounted tier. The analytics tool automatically treats missing values as zero in the CSV export.

### Daily Token Limit

ChronoTranscriber can automatically track daily OpenAI Responses usage and pause processing when a configurable token budget is exhausted. The feature is disabled by default and is controlled via `config/concurrency_config.yaml`:

```yaml
daily_token_limit:
  enabled: true          # Set to true to activate enforcement
  daily_tokens: 9000000  # Maximum tokens allowed per calendar day
```

#### How It Works

- **Automatic tracking:** Every successful Responses API call reports `usage.total_tokens`, which are summed in `modules/token_tracker.py` with thread-safe persistence.
- **Midnight reset:** The tracker stores state in `.chronotranscriber_token_state.json` and resets at local midnight without manual intervention.
- **Pre-flight checks:** `WorkflowManager.process_selected_items()` verifies the remaining budget before each GPT-driven document. When the limit is exhausted, the workflow waits until the next reset or until the operator cancels with `Ctrl+C`.
- **Live telemetry:** Running totals are emitted to the log and console at the start of processing, after each item, and once the session completes, helping teams monitor burn-in-place.

#### Recommended Usage

1. Enable the block above and adjust `daily_tokens` to match your allocation.
2. Keep `.chronotranscriber_token_state.json` under version-control ignore lists so local usage does not pollute repositories.
3. Delete the state file or edit the JSON manually if you need to reset counts ahead of the daily rollover.

### API Diagnostics

Built into `check_batches.py`, the diagnostics tool verifies your API configuration.

```bash
python main/check_batches.py
```

Diagnostic Checks:

1. API Key Presence: Verifies `OPENAI_API_KEY` environment variable is set
2. Model Listing: Attempts to list available models
3. Batch API Access: Verifies access to the Batch API endpoint

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
├── main/                      # CLI entry points
│   ├── cancel_batches.py
│   ├── check_batches.py
│   ├── repair_transcriptions.py
│   └── unified_transcriber.py
├── modules/                   # Core application modules
│   ├── config/               # Configuration loading
│   ├── core/                 # Core utilities and workflow
│   ├── infra/                # Infrastructure (logging, concurrency)
│   ├── io/                   # File I/O and path utilities
│   ├── llm/                  # LLM interaction and batch processing
│   ├── operations/           # High-level operations (batch check, repair)
│   ├── processing/           # Image and PDF processing
│   └── ui/                   # User interface and prompts
├── schemas/                   # JSON schemas for structured outputs
│   ├── markdown_transcription_schema.json
│   └── plain_text_transcription_schema.json
├── system_prompt/             # System prompt templates
│   └── system_prompt.txt
├── LICENSE
├── README.md
└── requirements.txt
```

### Module Structure

ChronoTranscriber follows a modular architecture with clear separation of concerns:

- `modules/config/`: Configuration loading and validation
- `modules/core/`: Core utilities including console printing, workflow management, path handling, and shared functions
- `modules/infra/`: Infrastructure layer providing logging, concurrency control, and async task management
- `modules/io/`: File I/O operations including path validation, directory scanning, and output management
- `modules/llm/`: LLM interaction layer including OpenAI SDK utilities, batch processing, model validation, and structured output parsing
- `modules/operations/`: High-level operation orchestration (batch checking, repair workflows)
- `modules/processing/`: Document processing including PDF rendering, image preprocessing, and text formatting
- `modules/ui/`: User interface components including interactive prompts and status displays

### Windows Path Length Handling

ChronoTranscriber implements robust handling for Windows MAX_PATH (260 character) limitations:

- Safe Directory Naming: Long document names are automatically truncated with content-based hashes for directory structures
- Preserved File Names: Output files retain their original proper names without hash suffixes
- Automatic Path Resolution: The system uses extended-length path syntax when needed on Windows 10 1607+
- Hybrid Strategy: Directories use hash-based safe names (e.g., `Long_document_name-a3f8d9e2/`) while files preserve original names (e.g., `Long document name_transcription.txt`)

This approach ensures reliable processing of documents with long filenames while maintaining human-readable output files. The implementation follows production-ready patterns similar to npm and Git for handling filesystem limitations.

### Operations Layer

ChronoTranscriber separates orchestration logic from CLI entry points to improve testability and maintainability:

- High-level operations live in `modules/operations/` (e.g., batch checking, repair workflows)
- CLI entry points in `main/` are thin wrappers that delegate to operations modules
- This design pattern allows operations to be reused, tested independently, and invoked programmatically

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
    tesseract_cmd: 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
```

#### Batch job missing from tracking file

Solution: Run `check_batches.py` which automatically repairs missing batch IDs from debug artifacts:

```bash
python main/check_batches.py
```

#### Model not supported

Solution: Check `modules/llm/model_capabilities.py` for supported models. Ensure your OpenAI account has access to the selected model.

#### Images not processing correctly

Solution:

1. Check image format (PNG and JPEG are supported)
2. Verify image preprocessing settings in `config/image_processing_config.yaml`
3. Try adjusting `llm_detail` parameter (high vs low vs auto)
4. Check logs in the configured `logs_dir` for detailed error messages

#### Temporary files not being cleaned up

Solution: Temporary JSONL files are only deleted when:

- The final transcription file was successfully written
- All batches in that job completed successfully
- `retain_temporary_jsonl` is set to `false`

If batches fail or are still in progress, temporary files are retained for recovery.

#### Page order is incorrect in output

Solution: ChronoTranscriber uses a multi-level ordering strategy. If pages are out of order:

1. Check that page numbers are correctly embedded in filenames
2. Verify that the JSON schema includes page number extraction
3. Review the `custom_id` format in batch requests
4. Check logs for ordering strategy decisions

### Debug Artifacts

ChronoTranscriber creates several debug artifacts to help troubleshoot issues:

- `<job>_batch_submission_debug.json`: Contains batch IDs, image count, chunk size, and submission timestamp
- `<job>_temporary.jsonl`: Tracks batch requests and responses with full metadata
- Log files: Detailed execution logs in the configured `logs_dir` with timestamps and stack traces

### Getting Help

If you encounter issues not covered here:

1. Check logs: Review detailed error messages in your configured `logs_dir`
2. Run diagnostics: Execute `python main/check_batches.py` to verify API configuration
3. Validate configuration: Ensure all YAML files are properly formatted
4. Verify directories: Confirm all required directories exist with proper permissions
5. Review requirements: Verify all dependencies are installed correctly
6. Check model access: Ensure your OpenAI account has access to the selected model

## Contributing

Contributions are welcome! Here's how you can help improve ChronoTranscriber:

### Reporting Issues

When reporting bugs or issues, please include:

- Description: Clear description of the problem
- Steps to Reproduce: Detailed steps to reproduce the issue
- Expected Behavior: What you expected to happen
- Actual Behavior: What actually happened
- Environment: OS, Python version, relevant package versions
- Configuration: Relevant sections from your config files (remove sensitive information)
- Logs: Relevant log excerpts showing the error

### Suggesting Features

Feature suggestions are appreciated. Please provide:

- Use Case: Describe the problem or need
- Proposed Solution: Your idea for addressing it
- Alternatives: Other approaches you've considered
- Impact: Who would benefit and how

### Code Contributions

If you'd like to contribute code:

1. Fork the repository and create a feature branch
2. Follow the existing code style and architecture patterns
3. Add tests for new functionality where applicable
4. Update documentation including this README and inline comments
5. Test thoroughly with both Tesseract and OpenAI backends
6. Submit a pull request with a clear description of your changes

### Development Guidelines

- Modularity: Keep functions focused and modules organized
- Error Handling: Use try-except blocks with informative error messages
- Logging: Use the logger for debugging information
- Configuration: Use YAML configuration files rather than hardcoding values
- User Experience: Provide clear prompts and feedback in CLI interactions
- Documentation: Update docstrings and README for any interface changes

### Areas for Contribution

Potential areas where contributions would be valuable:

- Additional OCR backends: Support for other OCR engines or APIs
- Enhanced preprocessing: Additional image enhancement algorithms
- Output formats: Support for different output formats (JSON, XML, etc.)
- Testing: Unit tests and integration tests
- Documentation: Tutorials, examples, and use case documentation
- Performance optimization: Improved concurrent processing or caching
- Error recovery: Enhanced error handling and recovery mechanisms

## License

## Development

### Recent Refactor (2025-10-13)

The codebase underwent a comprehensive refactor to improve organization, reduce duplication, and enhance maintainability. Key improvements include:

- **Centralized configuration service**: Singleton `ConfigService` eliminates redundant config loading
- **Modular utilities**: Extracted token guards, Tesseract utilities, JSONL operations, and progress tracking
- **Better testing support**: New test fixtures and diagnostic utilities
- **Improved documentation**: Comprehensive refactor summary and migration guide

See [REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md) for detailed information about the refactor and migration guide for developers.

### Contributing

Contributions are welcome! Please ensure:

- Code follows PEP 8 style guidelines
- New features include appropriate tests
- Documentation is updated for API changes
- Commit messages are clear and descriptive

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
