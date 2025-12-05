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

| Provider | Model | Reasoning Level |
|----------|-------|-----------------|
| OpenAI | GPT-5.1 | Medium |
| OpenAI | GPT-5 Mini | Medium |
| Google | Gemini 3.0 Pro | Medium |
| Google | Gemini 2.5 Flash | None |
| Anthropic | Claude Sonnet 4.5 | Medium |
| Anthropic | Claude Haiku 4.5 | Medium |

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
│   │   ├── address_books/       # JPEG images
│   │   ├── bibliography/        # PDF files
│   │   └── military_records/    # PDF files
│   ├── output/                  # Model transcriptions (JSONL per source)
│   │   └── {category}/
│   │       └── {model_name}/
│   │           └── {source}/
│   │               └── {source}.jsonl
│   └── ground_truth/            # Manually corrected transcriptions (JSONL)
│       └── {category}/
│           └── {source}.jsonl
└── reports/                     # Generated evaluation reports
```

## Workflow

### Step 1: Run Model Transcriptions

For each model, run transcriptions and save to the appropriate output directory:

```bash
# Example for GPT-5.1
python main/unified_transcriber.py --input eval/test_data/input/address_books \
    --output eval/test_data/output/address_books/gpt_5.1_medium \
    --type images --method gpt
```

Repeat for each model and category combination. The transcriber produces JSONL files
with per-page transcriptions that will be used for evaluation.

### Step 2: Create Ground Truth

Use the helper script to extract transcriptions for manual correction:

1. **Extract** transcriptions to editable text format:
   ```bash
   python main/prepare_ground_truth.py --extract --input eval/test_data/output/address_books/gpt_5.1_medium
   ```
   This creates `*_editable.txt` files with page markers like `=== page 001 ===`.

2. **Edit** the generated text files to correct transcription errors:
   - Each page is marked with `=== page NNN ===`
   - Correct OCR/transcription errors directly in the text
   - Use `[NO TRANSCRIBABLE TEXT]` for blank pages
   - Use `[TRANSCRIPTION NOT POSSIBLE]` for illegible pages

3. **Apply** corrections to create ground truth JSONL:
   ```bash
   python main/prepare_ground_truth.py --apply --input eval/test_data/output/address_books/gpt_5.1_medium
   ```
   This creates JSONL files in `test_data/ground_truth/{category}/`.

4. **Check** ground truth status:
   ```bash
   python main/prepare_ground_truth.py --status
   ```

### Step 3: Run Evaluation

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
