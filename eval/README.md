# ChronoTranscriber Evaluation Framework

This directory contains the evaluation framework for measuring transcription quality across multiple LLM providers and models.

## Overview

The evaluation computes **Character Error Rate (CER)** and **Word Error Rate (WER)** by comparing model outputs against manually corrected ground truth transcriptions.

### Page-Level Evaluation

Metrics are computed **page-by-page** using the temporary JSONL files produced by the transcriber. This approach:
- **Eliminates formatting penalties** from whitespace differences in final TXT output
- **Enables accurate per-page error attribution** for debugging
- **Isolates transcription quality** from post-processing effects

## Models Evaluated

| Provider | Model | `model_id` | Reasoning Level |
|----------|-------|-----------|-----------------|
| Local | Tesseract OCR | `tesseract` | None (baseline) |
| OpenAI | GPT-5.2 | `gpt-5.2` | Medium |
| OpenAI | GPT-5 Mini | `gpt-5-mini` | Medium |
| Google | Gemini 3 Pro | `gemini-3-pro` | Medium |
| Google | Gemini 3 Flash | `gemini-3-flash-preview` | None |
| Anthropic | Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | Medium |
| Anthropic | Claude Haiku 4.5 | `claude-haiku-4-5` | Medium |

## Dataset Categories

1. **Address Books** (`address_books/`) - Swiss address book pages from Basel 1900
2. **Bibliography** (`bibliography/`) - European culinary bibliographies
3. **Military Records** (`military_records/`) - Brazilian military enlistment cards

## Directory Structure

```
eval/
├── README.md                    # This file
├── eval_config.yaml             # Configuration for models and paths
├── metrics.py                   # CER/WER computation functions
├── jsonl_eval.py                # JSONL parsing for page-level evaluation
├── transcription_eval.ipynb     # Main evaluation notebook
├── schemas/
│   └── eval_transcription_schema.json
├── test_data/
│   ├── input/                   # Source documents
│   │   ├── address_books/       # JPEG images (31 pages, processed as one source)
│   │   ├── bibliography/        # PDF file(s)
│   │   └── military_records/    # PDF files
│   ├── output/                  # Model transcriptions (JSONL per source) — populate via Step 1
│   │   └── {category}/
│   │       └── {model_name}/
│   │           └── {source}-{hash}/
│   │               └── {source}.jsonl
│   └── ground_truth/            # Manually corrected ground truth
│       └── {category}/
│           └── {source}.jsonl   # Flat JSONL, one file per source
└── reports/                     # Generated evaluation reports (auto-created)
```

**Ground truth sources**:

| Category | Source | Pages |
|----------|--------|-------|
| `address_books` | `address_books.jsonl` | 31 |
| `bibliography` | `Whitaker_1913_English_Cookery_Books_to_the_Year_1850.jsonl` | 187 |
| `military_records` | `Antonio Franco.jsonl` | 2 |
| `military_records` | `Carlos Schimidt.jsonl` | 2 |
| `military_records` | `Elza Elias.jsonl` | 2 |

## Workflow

### Step 1: Run Model Transcriptions

Run transcriptions for each model and category. The
`--output` directory name must match the `name` field in `eval_config.yaml`.

**address_books** (images folder — pass the folder as `--input`):
```bash
# Tesseract baseline
python main/unified_transcriber.py --input eval/test_data/input/address_books \
    --output eval/test_data/output/address_books/tesseract \
    --type images --method tesseract

# GPT-5.2 (medium reasoning)
python main/unified_transcriber.py --input eval/test_data/input/address_books \
    --output eval/test_data/output/address_books/gpt_5.2_medium \
    --type images --method gpt --model gpt-5.2
```

**bibliography** (single PDF):
```bash
python main/unified_transcriber.py \
    --input eval/test_data/input/bibliography/Whitaker_1913_English_Cookery_Books_to_the_Year_1850.pdf \
    --output eval/test_data/output/bibliography/gpt_5.2_medium \
    --type pdf --method gpt --model gpt-5.2
```

**military_records** (each PDF separately, or the whole folder):
```bash
python main/unified_transcriber.py --input eval/test_data/input/military_records \
    --output eval/test_data/output/military_records/gpt_5.2_medium \
    --type pdf --method gpt --model gpt-5.2
```

Repeat for every model in `eval_config.yaml`. The transcriber produces JSONL files
with per-page transcriptions that will be used for evaluation.

### Step 2: Run Evaluation

Open and run `transcription_eval.ipynb` in Jupyter:

```bash
cd eval
jupyter notebook transcription_eval.ipynb
```

The notebook will:
- Discover available outputs and ground truth JSONL files
- Compute CER/WER page-by-page for each model/category combination
- Generate summary tables and per-page rankings
- Export results to `reports/` in JSON, CSV, and Markdown formats

## Metrics

### Character Error Rate (CER)

CER measures the edit distance at the character level:

```
CER = (Substitutions + Deletions + Insertions) / Reference_Length
```

Lower is better. A CER of 0.05 means 5% of characters differ from ground truth.

### Word Error Rate (WER)

WER measures the edit distance at the word level:

```
WER = (Substitutions + Deletions + Insertions) / Reference_Word_Count
```

Lower is better. WER is typically higher than CER since a single character error can affect an entire word.

### Content vs Formatting Metrics

The evaluation differentiates between:

1. **Overall Metrics**: Full text including all formatting elements
2. **Content-Only Metrics**: Text with formatting stripped (pure text extraction quality)
3. **Formatting Metrics**: Separate metrics for:
   - **Page Markers**: `<page_number>X</page_number>` tags
   - **Markdown Elements**: Bold, italic, headings, footnotes, LaTeX equations

This allows analyzing whether errors stem from text recognition vs. formatting interpretation.

## Output Files

After running the evaluation:

| File | Description |
|------|-------------|
| `eval_results_*.json` | Full metrics with all details |
| `eval_results_*.csv` | Tabular format for spreadsheets |
| `eval_results_*.md` | Markdown summary for documentation |
| `eval_chart_*.png` | Visualization (if matplotlib available) |

## Configuration

Edit `eval_config.yaml` to:
- Add or remove models from evaluation
- Change dataset paths
- Adjust normalization settings
- Configure runtime parameters

## Dependencies

The evaluation uses standard Python libraries plus:
- `pyyaml` - Configuration loading
- `matplotlib` (optional) - Visualization

Install with:
```bash
pip install pyyaml matplotlib
```
