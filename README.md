## Chrono Transcriber

A pipeline to transcribe historical documents (PDFs or image folders) using either a local OCR engine (Tesseract) or modern vision-language models via the OpenAI **Responses API** with JSON-schema structured outputs. It includes scalable Batch submission, recoverable processing, and careful page ordering for long multi-page sources.


## Core Functions

1. **Transcribe PDFs** into plaintext, with native text extraction fallback and/or page-to-image rendering.  
2. **Transcribe image folders** (PNG/JPEG) representing page scans.  
3. **Two OCR backends**:  
   - **Tesseract** (local).  
   - **OpenAI Responses** (vision models), including GPT-4o, GPT-4.1, **GPT-5** / **GPT-5-mini**, and **o-series** (o1/o3) with automatic capability checks.  
4. **Batch mode** for VLM OCR at scale, with direct data-URL image embedding (no external hosting), chunking, and order-preserving merge.  
5. **Structured outputs** using a JSON Schema injected into the system prompt, ensuring predictable extraction and robust downstream parsing.

---

## Configuration

The project reads three YAML files from `config/`:

- **`model_config.yaml`** — choose model and per-family controls.  
  - `name`: `gpt-5-mini` (default), or `gpt-5`, `o3`, `o1`, `gpt-4o`, `gpt-4.1`.  
  - `max_output_tokens`, `service_tier`.  
  - GPT-5-only: `reasoning.effort`, `text.verbosity`.  
  - Sampler controls (`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`) are only applied to **non-reasoning** models (e.g., GPT-4o/4.1).

- **`image_processing_config.yaml`** — pre-processing and VLM detail.  
  - Page rendering settings when rasterising PDFs.  
  - `llm_detail: auto|high|low` (controls Requests’ `detail` for images).  
  - If `low`, images are resized to a configured max side length prior to encoding as `data:` URLs to improve token efficiency.

- **`paths_config.yaml`** — filesystem conventions.  
  - `file_paths` for `PDFs` and `Images` (separate `input`/`output`).  
  - `allow_relative_paths` with `base_directory`.  
  - `input_paths_is_output_path` to co-locate results.  
  - `keep_preprocessed_images`, `retain_temporary_jsonl`, `logs_dir`.

---

## Setup

1. Install Python dependencies and ensure Tesseract is available if you plan to use the local backend.  
2. Set `OPENAI_API_KEY` in your environment if you will use the VLM backend and/or Batch.  
3. Edit the three config files under `config/` to reflect your paths, model choice, and preprocessing preferences.  
4. (Optional) Adjust the JSON schema under `schemas/` if you want to change the structured output contract (e.g., to include per-page markers).

---

## Usage

### A. Transcribing PDFs

1. Place PDFs under the configured `file_paths.PDFs.input`.  
2. Run the main entry point and select **PDF** mode.  
3. Choose OCR backend:  
   - **Tesseract** for purely local processing.  
   - **OpenAI Responses** for VLM OCR (streaming single requests) or **Batch** for large jobs.  
4. Outputs are written to `file_paths.PDFs.output` (or to input directories if `input_paths_is_output_path: true`).

### B. Transcribing Image Folders

1. Place folders of page images under `file_paths.Images.input`.  
2. Run the main entry point and select **Image Folder** mode.  
3. Select backend and (optionally) Batch.  
4. Outputs are written to `file_paths.Images.output` (or co-located).

> **Supported formats:** PNG and JPEG are packaged directly into Requests/Batch via base64-encoded `data:` URLs.

---

## Batch Mode (Responses API)

Chrono Transcriber can submit large image sets as OpenAI **Batches** against the **Responses** endpoint.

- **How it works**  
  - Images are base64-encoded as `data:` URLs and paired with a **system prompt** containing your JSON Schema (injected automatically).  
  - Requests are chunked, size-aware (≤150 MB per part to allow headroom under the 180 MB limit), and tagged with stable `custom_id`s and per-image metadata (image name, page number, order index).  
  - After submission, the tool writes a local **`batch_submission_debug.json`** (count of batches and their IDs) to support “lost batch id” recovery.

- **Monitoring & finalisation**  
  - Run `check_batches.py` to poll status for all temp JSONL files, diagnose failures, and download results when **all** batches of a source are complete.  
  - The checker merges model outputs using a **multi-level ordering** strategy: explicit order info → custom-id index → embedded page number → number parsed from filename → stable fallback index.  
  - Temporary JSONL files are deleted **only if** the final text was written and all batches in that JSONL completed successfully (configurable via `retain_temporary_jsonl`).

- **Cancelling jobs**  
  - Use `cancel_batches.py` to list batches, skip terminal ones (`completed|expired|cancelled|failed`), and cancel the rest with a clear summary.

---

## Image Preprocessing

- PDF pages can be rasterised to images with your chosen DPI/layout settings.  
- Optional grayscale/alpha handling and safe JPEG export for uniform downstream behaviour.  
- **VLM detail control**: `llm_detail` influences how much visual information is fed to the model (`auto|high|low`). For `low`, images are automatically resized to limit tokens and cost.

---

## Structured Outputs

- The pipeline injects a **JSON Schema** into the system prompt so the model returns typed responses (e.g., fields like `transcription`, flags for “no transcribable text”, etc.).  
- Downstream parsing first attempts structured JSON; if absent/invalid, it falls back to raw text content. This ensures the merge step never blocks.

---

## Model Selection & Capabilities

- Choose from `gpt-4o`, `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `o1`, `o3`.  
- The tool **gates** features by model family (images, structured outputs, reasoning controls, sampler controls). GPT-5 exposes `reasoning.effort` and `text.verbosity`; sampler controls are applied only to classic non-reasoning models.

---

## Relative Paths

- Enable `allow_relative_paths: true` and set `base_directory` to work with relative inputs/outputs across the entire pipeline.  
- If you want results written next to inputs, set `input_paths_is_output_path: true`. Path validation guards against common misconfigurations.

---

## Utilities

- **`check_batches.py`**: scans for temp JSONL files, repairs missing batch IDs using `batch_submission_debug.json`, diagnoses API/model issues, downloads results, and merges them in the right order before (optionally) cleaning up.  
- **`cancel_batches.py`**: lists batches, shows a summary, and cancels all non-terminal jobs; terminal ones are skipped with a report.

---

## Troubleshooting

- Run the **API diagnostics** built into `check_batches.py` to verify API key presence, model listing, and Batch API access if anything looks off.  
- If a temp `.jsonl` appears to be missing some `batch_tracking` lines, the checker can pull IDs from `batch_submission_debug.json` and repair the file before proceeding.

---

## Notes

- Temporary files are only removed when it is safe to do so (successful final write and all batches complete), or retained if `retain_temporary_jsonl: true`.  
- Supported image formats: PNG and JPEG; images are embedded in requests as `data:` URLs.

---

### License

Private; do not distribute without permission.
