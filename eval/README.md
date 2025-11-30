# ChronoTranscriber Evaluation Framework

This directory contains the evaluation framework for measuring transcription quality across multiple LLM providers and models.

## Overview

The evaluation computes **Character Error Rate (CER)** and **Word Error Rate (WER)** by comparing model outputs against manually corrected ground truth transcriptions.

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
├── transcription_eval.ipynb     # Main evaluation notebook
├── schemas/
│   └── eval_transcription_schema.json
├── test_data/
│   ├── input/                   # Source documents
│   │   ├── address_books/       # 31 JPEG images
│   │   ├── bibliography/        # PDF files
│   │   └── military_records/    # PDF files
│   ├── output/                  # Model transcriptions
│   │   └── {category}/
│   │       └── {model_name}/    # e.g., gpt_5_mini_medium/
│   └── ground_truth/            # Manually corrected transcriptions
│       ├── address_books/
│       ├── bibliography/
│       └── military_records/
└── reports/                     # Generated evaluation reports
```

## Workflow

### Step 1: Create Ground Truth

1. Run a transcription pass using a high-quality model (e.g., GPT-5.1):
   ```bash
   python main/unified_transcriber.py --input eval/test_data/input/address_books \
       --output eval/test_data/output/address_books/ground_truth_draft \
       --type images --method gpt
   ```

2. Manually review and correct the transcriptions

3. Save corrected files to `test_data/ground_truth/{category}/{source_name}.txt`

### Step 2: Run Model Transcriptions

For each model, run transcriptions and save to the appropriate output directory:

```bash
# Example for GPT-5 Mini
python main/unified_transcriber.py --input eval/test_data/input/address_books \
    --output eval/test_data/output/address_books/gpt_5_mini_medium \
    --type images --method gpt --model gpt-5-mini
```

Repeat for each model and category combination.

### Step 3: Run Evaluation

Open and run `transcription_eval.ipynb` in Jupyter:

```bash
cd eval
jupyter notebook transcription_eval.ipynb
```

The notebook will:
- Discover available outputs and ground truth
- Compute CER/WER for each model/category combination
- Generate summary tables and rankings
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
