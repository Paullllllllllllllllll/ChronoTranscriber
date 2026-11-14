---
name: chronotranscriber
description: >-
  Use when the user wants PDFs, scans, ebooks or recordings transcribed, or pages
  or documents classified with a decision model, resumes, repairs, drops or
  batch-checks a ChronoTranscriber run, or reads its outputs
  (chronotranscriber.sqlite, the rendered .txt or .md).
---

# ChronoTranscriber

The tool runs from a clone of the repository set up as in the README, Installation;
the schemas, classifiers and context files it names live in that clone. The README
is the documentation (Command Reference, Exit Codes and JSON Summary, Output
Contract, Store, Batch Processing, Resume); `chronotranscriber <command> --help`
gives each command's flags, defaults and exit codes. Run every command through uv
with absolute paths, since `uv --directory` resolves relative paths against the
clone:

```bash
uv --directory <clone> run chronotranscriber <command> --input <absolute path> [options] --json
```

## Running a job

1. **Check access.** Done when the key variable of the model's provider is set
   (README, Providers and Response Formats) and you know whether the job runs
   synchronously or as a batch. `--batch` runs on OpenAI, Anthropic and Google;
   OpenRouter and custom endpoints have no batch API (exit 2). Audio has no batch
   mode, and every `--method
   audio-api` run is billed per audio minute, so it needs the user's consent;
   `--method whisper` runs locally without a key.
2. **Dry run.** Add `--dry-run --json` to the intended command. Done when it exits 0,
   `to_process` lists exactly the intended items with the expected `state`, `refused`,
   `blocked` and `unstored` are empty or explained, and every value you set shows
   `"source": "flag"` in `spec`. A folder with several items needs `--all`,
   `--select EXPR` or `--files LIST`; items in subfolders also need `--recursive`,
   without which the run exits 2 with "No items found". `--recursive` reads every
   level in every mode, `--select N` counts in the natural order of the paths below
   the input, and `--files` takes those paths. The classify commands list
   `to_classify` in place of `to_process` (README, Exit Codes and JSON Summary).
3. **Sample** anything beyond one short item: transcribe `--pages 5` (the first five
   pages), read `usage.total.total_tokens` and project the job; the full run resumes
   after the sampled pages. Done when the user has seen the projection for a job
   beyond a few items.
4. **Run.** A job that finishes within the turn runs directly; a longer one runs as a
   background process whose `--json` line you read when it ends. A batch job is two
   steps: `transcribe ... --batch`, then `batch check --input <output folder>
   --wait`.
5. **Verify.** Done when the exit code is 0, `items_failed` and `items_blocked` are 0
   and every path in `outputs` exists; for a batch job, when `batch check --wait`
   exits 0; for a classify command, when the exit code is 0 and `items_failed`,
   `items_refused` and `units_failed` are 0. Exit 1 means an item failed, was
   refused or is leased by another run, or a batch unit failed. Rerun the same
   command, which resumes and retries failed pages, or follow
   [REFERENCE.md](REFERENCE.md), Troubleshooting. Exit 3 means the provider stopped
   serving the run: nothing failed, the unsent pages stay pending, and the same
   command resumes them once the provider serves again.

## Conventions

- The `defaults:` block of the settings file sets the model, effort, service tier
  and concurrency; pass those flags only on the user's request.
- A folder of many short items takes `--open-items N` (8, for example) on
  `transcribe` with a model method and on the classify commands: the run then keeps
  up to N items open under the one `--concurrency` limit, where otherwise a two-page
  item uses two request slots and leaves the rest idle. It has no effect with
  `--batch`, with a local method or on a single item.
- New documents take the model family's image preset, so leave the image flags unset;
  bilevel scans are the exception and take `--image-format png`. A resumed document
  whose stored pages used other image settings is `refused` (exit 1): pass the stored
  values the refusal names, or `--force` to start it over.
- `--schema` takes a path, or a name found in the `--schemas-dir` folders, the
  settings' `schemas_dirs` and then `schemas/` (README, Sources and Methods); a name
  in two of the user folders is an error (exit 2) that names both files. A resume or
  `repair` that would leave done pages of another schema beside its new ones is
  `refused` (exit 1, both hashes named): pass the earlier schema, or `--force` to
  transcribe the document again (README, Resume).
- Context is a topic file `{stem}_transcr_context.txt` beside the input (folder forms:
  README, Sources and Methods). Above 4,000 characters the run warns. An item without
  a sidecar gets the settings' fallback context, if one is set.
- `--decisions shadow`, on the user's request, has a decision model (README, Decision
  pre-pass) record each page's type before transcription, in `page_types` and the
  `page_type` column of `page_text`. `apply` skips a no-text page only under a
  threshold that `decisions.thresholds` validates for the exact request; without one
  it skips nothing. A skipped page has the outcome `skipped_no_text`. A failed
  pre-pass only warns and transcribes the pages untyped (stderr says so), and a rerun
  leaves done pages alone. After a shadow run, count the pages whose `page_type` in
  `page_text` is null and report them; `--force` deletes a document's earlier page
  types before the new pass.
- `classify pages` and `classify documents --classifier NAME|PATH` ask a decision
  model a classifier file's questions about page images, or about a document from
  its first pages, without transcribing; the shipped classifiers are in
  `classifiers/` (README, Classifiers), a project's own are passed by path or found
  in the `--classifiers-dir` folders and the settings' `classifiers_dirs`. Answers
  only record: read them in the `classifications` view. A page answer joins
  `page_text` on document and ordinal once the pages are transcribed, so pass the
  same `--output` and `--layout` as the transcription to match the document names;
  a document answer concerns the whole document.
- `chronotranscriber.sqlite` in each output root holds every unit and provider
  response. Remove documents from it with `drop` (dry run first), and read it
  read-only through the views in README, Store.
