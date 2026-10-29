# ChronoTranscriber v1.1.3

ChronoTranscriber transcribes PDFs, folders of page images, EPUB and MOBI ebooks
and audio recordings. Scanned pages go to a vision-language model or to local
Tesseract OCR, born-digital files give up their text layer, and recordings go to a
speech-to-text API or a local Whisper model. Model calls run on OpenAI, Anthropic,
Google and OpenRouter models and on OpenAI-compatible endpoints that you name in
the settings. The commands serve scripts and agents; `chronotranscriber ui` walks
you through a run in the terminal. Please report bugs on the [GitHub issue
tracker][issues].

[issues]: https://github.com/Paullllllllllllllllll/ChronoTranscriber/issues

## Overview

An item is one PDF, image folder, EPUB, MOBI or audio file. A run discovers the
items of its input, routes each to a method and transcribes the item's units: the
pages of a PDF, the images of a folder, the sections of an ebook or the chunks of
a recording. Each unit is committed to the database `chronotranscriber.sqlite` of
the output folder as soon as it is done, together with every provider response as
received. The `txt` and `md` files are rendered from the database after each item,
so an interrupted run resumes from it and `render` rewrites the files without
model calls. For `book.pdf` and `--output out`, a run writes
`out/book-<hash>/book.txt` and `out/chronotranscriber.sqlite`.

## Installation

ChronoTranscriber needs Python 3.13 or later and
[uv](https://docs.astral.sh/uv/). Install it from a clone, since the settings
file, the schemas and the default context are read from the repository.

```bash
git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
cd ChronoTranscriber
uv sync --all-extras          # or uv sync, plus --extra audio or dev
mkdir config
cp chronotranscriber/resources/settings.example.yaml config/settings.yaml
export OPENAI_API_KEY="sk-..."   # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

The extras are `audio` (faster-whisper for `--method whisper`) and `dev` (pytest,
ruff, mypy). Two system tools are optional:
[Tesseract](https://github.com/tesseract-ocr/tesseract) for `--method tesseract`
and for PDFs without a text layer under `--method native`, and
[FFmpeg](https://ffmpeg.org/) (Windows: `winget install Gyan.FFmpeg`) to cut
recordings and to decode JPEG 2000 images that Pillow cannot read. Both are
found on `PATH` unless `tools:` in the settings names them. The settings name the
API key variable of each provider (`api_keys:`); the defaults are
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` and `OPENROUTER_API_KEY`.

## Quick Start

```bash
uv run chronotranscriber transcribe --input book.pdf --output out
uv run chronotranscriber transcribe --input scans --all --output out --formats txt,md
uv run chronotranscriber transcribe --input scans --select 1-3 --output out
uv run chronotranscriber transcribe --input pages --type images --method tesseract
uv run chronotranscriber transcribe --input talk.mp3 --type audio --method whisper
uv run chronotranscriber transcribe --input scans --all --dry-run --json
uv run chronotranscriber ui
```

Bare `chronotranscriber` prints the commands, `<command> --help` the options of
one. Relative paths resolve against the current directory; with
`uv --directory <clone> run ...` that is the clone, so pass absolute paths.

## Command Reference

The tables list every option of each command with its `--help` wording and
default. Options without a stated default are off or unset until given. The last
column says whether the settings `defaults:` block can set the option; the key is
the flag without dashes, with underscores for hyphens (`reasoning_effort`). A list
option given twice keeps the values of both.

### transcribe

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | PDF, EPUB, MOBI or audio file, image folder, or a folder of these | | no |
| `--type {auto,pdf,images,epub,mobi,audio}` | input type; auto takes the PDFs, image folders, EPUB and MOBI files of the input; audio needs --type audio | auto | yes |
| `--files LIST` | comma-separated names of files or folders inside the input folder; only for type audio, epub, images, mobi, pdf | | no |
| `--select EXPR` | select items by number (1,3), range (1-5) or name search | | yes |
| `--all` | process every discovered item | | yes |
| `--recursive`, `--no-recursive` | search the subfolders of the input folder; only for type audio, epub, images, mobi, pdf | off | yes |
| `--pages RANGE` | pages, images or audio chunks: 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8 | all | no |
| `--method {llm,tesseract,native,whisper,audio-api}` | transcription method; in auto mode it serves scanned PDFs and image folders; EPUB and MOBI take native and audio takes whisper unless --method names another | llm | yes |
| `--searchable-pdfs {ocr,text}` | PDFs with a text layer in auto mode: ocr transcribes them with the method, text takes the text layer; only for type auto | ocr | yes |
| `--output DIR` | output folder | beside the input | yes |
| `--beside-input` | write outputs beside each input | | no |
| `--layout {hash,mirror}` | output folder layout: hash (one hash-suffixed folder per item) or mirror (the input folder tree) | hash | yes |
| `--formats LIST` | comma-separated output formats; from txt, md | txt | yes |
| `--schema NAME\|PATH` | transcription schema: a name found in schemas/ or a path | markdown_transcription_schema | yes |
| `--prompt PATH\|default` | system prompt file, or default (the prompt of the response format) | default | yes |
| `--context TEXT\|PATH\|auto\|none` | transcription context: topics as text or a path to a text file (overrides sidecars), auto (the item's sidecar) or none (no context) | auto | yes |
| `--fallback-context TEXT\|PATH\|none` | transcription context for items without a sidecar under --context auto: topics as text, a path to a text file, or none | none | yes |
| `--context-image PATH\|auto\|none` | image sent with every page: an image file, auto (the item's sidecar image) or none | auto | yes |
| `--provider {openai,anthropic,google,openrouter,custom}` | model provider | inferred from the model name | yes |
| `--model NAME` | model name; a provider: prefix (anthropic:NAME) selects the provider | gpt-6-luna; gpt-transcribe for --method audio-api, gemini-3.6-flash with --provider google | yes |
| `--endpoint NAME` | named custom endpoint from the settings (implies provider custom) | none | yes |
| `--reasoning-effort {none,minimal,low,medium,high,xhigh,max}` | reasoning effort | xhigh | yes |
| `--verbosity {low,medium,high}` | output verbosity | medium | yes |
| `--max-output-tokens N` | maximum output tokens | the model's output limit | yes |
| `--temperature T` | sampling temperature, 0.0 to 2.0 | 1.0 | yes |
| `--top-p P` | nucleus sampling, 0.0 to 1.0; sent only when supported | 1.0 | yes |
| `--service-tier {auto,default,flex,priority}` | OpenAI service tier; only for provider openai | default | yes |
| `--response-format {schema,json,text}` | schema (API-enforced), json (prompted, validated) or text (plain) | schema when the model supports it, else json | yes |
| `--decisions {off,shadow,apply}` | decision-model pre-pass of --method llm: shadow records each page's type; apply also skips pages without text of the book | off | yes |
| `--decision-model ID` | decision model id; only for decisions apply, shadow | typesafe/jev-1.14 | yes |
| `--decision-provider {openai,openrouter}` | decision API dialect; only for decisions apply, shadow | from the model id | yes |
| `--decision-endpoint URL` | decision endpoint URL; only for decisions apply, shadow | the provider's | yes |
| `--dpi native\|N` | render resolution: native or a DPI value | the provider's recommended value | yes |
| `--image-detail {low,high,auto,original}` | image detail sent to the model | the provider's recommended value | yes |
| `--image-format {jpeg,png}` | page image format | the provider's recommended value | yes |
| `--jpeg-quality N` | JPEG quality, 1 to 100 | the provider's recommended value | yes |
| `--grayscale`, `--no-grayscale` | convert page images to grayscale | the provider's recommended value | yes |
| `--ocr-config STR` | Tesseract configuration string; only for method tesseract | --oem 3 --psm 6 | yes |
| `--ocr-preprocess {default,sauvola,none}` | Tesseract image preprocessing preset; only for method tesseract | default | yes |
| `--whisper-model SIZE` | faster-whisper model size; only for method whisper | large-v3 | yes |
| `--language CODE\|auto` | spoken language as an ISO 639-1 code, or auto; only for method audio-api, whisper | auto | yes |
| `--audio-chunk-seconds N` | length of the audio chunks sent per request, in seconds; only for method audio-api, whisper | 600 | yes |
| `--batch`, `--no-batch` | submit the pages to the provider's batch API; only for method llm | off | yes |
| `--sync-fallback`, `--no-sync-fallback` | transcribe synchronously when a batch submission fails; only for method llm | off | yes |
| `--retry-errors`, `--no-retry-errors` | on resume, transcribe pages again whose earlier result is an error; --no-retry-errors keeps those results | on | yes |
| `--force` | transcribe items again from the start: every stored unit of the item is reset, also outside --pages; stored responses are kept | | no |
| `--concurrency N` | parallel requests | 16 | yes |
| `--keep-working-files`, `--no-keep-working-files` | keep Tesseract's preprocessed page images and the audio chunks in the item's working folder | off | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

`--input` is required, and `--beside-input` overrides an `output` settings
default. An input with several items needs one of `--all`, `--select` and
`--files`; without one the run exits 2 and names the three. `--select` numbers
the items from 1 by path, with numbers in names compared by value (`p2` before
`p10`), and takes numbers and ranges (`1-3,5`) or a text that the item names
contain (`ledger`); an out-of-range number warns, and a selection that matches
nothing exits 2. `--files` names items exactly and needs a type other than
`auto`. An option whose condition does not hold (`--ocr-config` with
`--method llm`) has no effect, and the run warns on stderr. The type and method
must form a valid pair:

| `--type` | Methods |
| --- | --- |
| `auto` | `llm` or `tesseract` for scanned PDFs and image folders; EPUB and MOBI take `native` |
| `pdf` | `llm`, `tesseract`, `native` |
| `images` | `llm`, `tesseract` |
| `epub`, `mobi` | `native` |
| `audio` | `whisper`, `audio-api` |

EPUB, MOBI and audio switch to their first method unless `--method` names another
as a flag (a settings default does not count); for the other types an invalid
pair is a usage error. Auto mode takes the PDFs, image folders, EPUB and MOBI
files directly inside the input folder, and with `--searchable-pdfs text` a PDF
with a text layer takes the native method. Audio always needs `--type audio`.

### repair

`repair` also takes the schema, prompt, context, model and image options of
`transcribe`, with their defaults and settings defaults; its `--model` default
is the transcription model.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | a transcription file or the input it was transcribed from, a chronotranscriber.sqlite, or a folder whose databases at or below it are repaired | the current directory | no |
| `--errors-only` | repair transcription errors; the default when no other cause is given | | no |
| `--not-possible` | repair pages marked transcription not possible | | no |
| `--no-text` | repair pages marked no transcribable text | | no |
| `--all-failures` | repair every failure cause | | no |
| `--skipped` | transcribe pages the decision pre-pass skipped as without text | | no |
| `--indices LIST` | 0-based page indices (order in the item) among the selected failures of one document: 0,5,12 or 3-7 | every selected failure | no |
| `--model NAME` | model name; a provider: prefix (anthropic:NAME) selects the provider | gpt-6-luna | yes |
| `--batch`, `--no-batch` | submit the repairs to the provider's batch API | off | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### batch check

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input DIR` | folder searched for batch submissions, subfolders included | the current directory | no |
| `--batch-ids IDS` | comma-separated batch IDs to check | every batch | no |
| `--adopt BATCH_ID` | record this provider batch for the unresolved submission it belongs to, then finalize it | | no |
| `--abandon SUBMISSION_ID` | give up an unresolved or submitted submission: its pages return to pending and its uploaded files are deleted; a submitted batch is not cancelled | | no |
| `--formats LIST` | comma-separated output formats; from txt, md | the formats of the run that submitted the batch | no |
| `--wait` | check again every --poll-minutes until no batch work is pending, or until --wait-hours have passed | | no |
| `--poll-minutes N` | minutes between two checks with --wait | 15 | no |
| `--wait-hours N` | hours after which --wait starts no further check; a running check finishes | 50 | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### batch cancel

One of `--batch-ids`, `--all` and `--input` is required.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--batch-ids IDS` | comma-separated batch IDs to cancel | | no |
| `--all` | cancel every unfinished batch of every provider | | no |
| `--input DIR` | cancel the open batch submissions recorded in the databases below this folder | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### render

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | folder or chronotranscriber.sqlite to render | . | no |
| `--formats LIST` | comma-separated formats to render; from txt, md | the formats each document was rendered in | no |
| `--force` | replace files changed outside the tool | | no |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |

### drop

Exactly one of `--select`, `--files` and `--all` is required. `--select` takes
the expressions of `transcribe --select` and numbers the stored documents from 1
in name order; a `--files` entry names a document or the file name of its input,
and an entry that names two documents is refused.

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | output folder holding chronotranscriber.sqlite, or the database itself | | no |
| `--select EXPR` | select stored documents by number (1,3), range (1-5) or name search | | no |
| `--files LIST` | comma-separated names of stored documents or of their input files | | no |
| `--all` | drop every document of the store | | no |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |

### ui

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

## Exit Codes and JSON Summary

| Code | Meaning |
| --- | --- |
| 0 | every item completed or, under `--batch`, was submitted, or nothing was left to do; a dry run that planned, also without API keys |
| 1 | an item failed (refused items included), another run leases an item (the final message names each blocked item with its reason and a lease with its expiry; an item waiting for a batch is named but does not change the exit code), a repaired page stayed unrepaired, a batch failed or waits unresolved, `batch check --wait` reached `--wait-hours` with work pending, a cancellation failed, a file could not be rendered, `drop` refused a document, or an unexpected error |
| 2 | usage or configuration error: invalid flags or settings, an unset API key variable (except in a dry run), a provider without a batch API under `--batch`, no items in the input, several items without `--all`, `--select` or `--files`, two items that would share one output file (`book.pdf` and `book.epub` beside the input), a `--select` that matches nothing, `repair --indices` with several documents, a `drop` selection that names no stored document, a missing input path or store, a database that is not a ChronoTranscriber store, newer than the tool or unreadable |
| 130 | interrupted (Ctrl+C, or Ctrl+Break on Windows) |

An interrupted `transcribe` run releases its leases, so a rerun right away
resumes with the units still to do. A run that is killed instead keeps its
leases until they expire (10 minutes); until then a rerun reports its items as
blocked and exits 1. A lease taken while a run plans or processes its items also
counts, including for a complete item whose files would be rendered again.

`<command> --help` lists the exit codes of each command. Progress, warnings and
errors go to stderr, in color only on a terminal and when `NO_COLOR` is unset or
empty. With `--json`, stdout carries exactly one JSON line on every exit, errors
and interrupts included; `ui` writes none. Every line starts with `dry_run`, true
whenever `--dry-run` was given. Every field listed for a command is present on
every exit path, zero where nothing ran, unless stated otherwise.

| Command | Fields |
| --- | --- |
| `transcribe` | `items_total`, `items_complete`, `items_failed`, `items_skipped`, `items_blocked`, `items_submitted`, `outputs`, `usage` |
| `repair` | `command` (`repair`), `repaired`, `failed`, `usage` |
| `batch check` | `command` (`check_batches`), `finalized`, `pending`, `failed`, `submitted`, `complete`, `failed_pages`, `outputs`, `usage`, `unresolved`; the booleans `adopted` with `--adopt` and `abandoned` with `--abandon`; with `--wait`, `polls` (checks finished), `waited_seconds` and `stopped` (`settled`, `timeout`, `interrupted`, `error`; null before the first check), and the counts are each submission's state in the store at the end; the line of a parse error carries only the first ten |
| `batch cancel` | `command` (`cancel_batches`), `cancelled`, `failed` |
| `render` | `command` (`render`), `databases`, `documents`, `written`, `unchanged`, `kept`, `stale`, `skipped`, `outputs` |
| `drop` | `command` (`drop`), `dropped`, `refused`, `deleted_files`, `kept_files` |

For `transcribe`, `items_total` counts the items processed (on a plan error, the
items the run would have processed), `items_skipped` the complete, unstored and
blocked items and those of which `--pages` selects no unit (each named in a
warning), `items_blocked` the blocked items among them (those another run
leases exit 1; those waiting for a batch do not), and `items_submitted` the
items whose pages went to a batch under `--batch`, which `items_complete` leaves
out. An item whose selected pages all succeeded while pages outside `--pages`
remain pending or failed counts neither complete nor failed; stderr reports it as
left partial by `--pages`. `outputs` lists the absolute paths of the files
written, the databases last. `usage` holds the tokens of each role that made
calls (`transcription`, `audio`) and their sum under `total`, which is always
present, each as `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`
(input read from the prompt cache) and `reasoning_tokens`; after an interrupt or
an unexpected error it holds the tokens spent until then. A dry run that plans
successfully adds `to_process` (each item's `name`, `type`, `method` and resume
`state`), `skipped` (names of complete items), `unstored`, `blocked` and `refused`
(`name` and `reason`), and `spec`, which maps every field of the effective run
specification, such as `model.model`, to its `value` and `source` (`flag`,
`settings`, `default` or `resolved`, described under Settings File). A dry run
plans without the API key variables and names an unset one in a warning on
stderr.

For `batch check`, `finalized`, `pending`, `failed` and `unresolved` count
submissions; `submitted` counts the submissions the check opened (always 0, since
a transcription has one batch phase), `complete` the documents it rendered with
every page done, `failed_pages` the pages its finalized submissions stored as
failed, `outputs` the files it rendered and `usage` the tokens of the finalized
results under `transcription`. A check that finalizes a failed page exits 1.

For `repair`, `usage` holds the tokens of the repair calls, or of the repair
batches it finalized, under `repair`, and their sum under `total`, in the shape
of the `transcribe` summary's `usage`.

For `render`, `kept` counts files changed outside the tool and left alone,
`stale` files that could not be replaced, `skipped` documents another run holds,
and `outputs` lists the files written. For `drop`, `dropped` names the documents
deleted, `refused` lists each document left in place with its `doc_id`, `name`
and `reason`, `deleted_files` the rendered files deleted and `kept_files` each
file left in place with its `path` and `reason`; a dry run lists what a drop
would do.

When the environment variable `CHRONO_HEARTBEAT_FILE` names a file, `transcribe`
and `batch check --wait` also write their progress there as one JSON object for a
job launcher: `units_done` of `units_total` `unit_label`, `eta_seconds` and
`updated_at` (ISO 8601, UTC). A run counts the current item's pages, images or
audio chunks and names the item's position (`pages processed, item 3/12`); its
ETA, from the unit rate, is given only during the last item and is null
otherwise. A wait counts the submissions finished of those it has seen
(`batches`) before each sleep. The file is replaced at most every 15 seconds, and
a write that fails is skipped without affecting the run.

## Guided UI

`chronotranscriber ui` asks first for the task: a new transcription, a repair,
batch check or cancel, render files again or drop documents from the store. A
new transcription then asks for the input (an input with nothing to process is
asked again), the type and method (valid pairs only) and, for a folder with
several items, `Which items?`: all of them, by number or range, by name search,
or from a checklist. Then come the output location and formats, the model, the
response format with the prompt and the context (`Automatic (sidecar)`, with a
choice for items without a sidecar, `Topics as text`, `A text file` or `None`),
page images, the decision-model pre-pass (its model, API and endpoint only when
it is on), Tesseract or audio options, pages, and the run options with batch
or sync; a batch run asks `If a batch submission fails:` (stop with an error, or
transcribe synchronously as `--sync-fallback` does). `Parallel requests:` is asked
only where calls are synchronous, so not for a batch run without the sync fallback.
A step appears only for the methods that use it. Every list offers Back, and a
text or path prompt takes `<` as Back. Settings defaults are preselected, and
answers act like flags over them. The review screen marks values from settings
defaults with `(s)`, names on its `access` line how the model calls authenticate
(the provider's API key variable), previews the plan without API calls and prints
the equivalent command, which spells out every value that differs from the code
default. It offers Run, Edit a step, Copy command and quit, and Quit; Run is
withheld while the plan fails, and, except for a dry run, while an API key
variable is unset, which the review names. The UI needs an interactive terminal
on stdin and stdout, else it exits with code 2; otherwise its exit codes are
those of the chosen task's command.

## Settings File

`--settings FILE` reads the named file, which must exist. Otherwise
ChronoTranscriber reads `config/settings.yaml` in the repository (gitignored),
else the packaged `chronotranscriber/resources/settings.example.yaml`, and then
logs one line asking you to copy it. Every key is optional; unknown keys are an
error that names the file and the key. The example file:

```yaml
# ChronoTranscriber settings: machine values and optional run defaults.
#
# Copy this file to config/settings.yaml in the repository (gitignored) and
# edit the copy, or pass another file with --settings FILE. Every key is
# optional; a missing key keeps the value shown here. Unknown keys are errors.
# Relative paths resolve against the current directory.

# Environment-variable NAMES that hold each provider's API key (never the key).
api_keys:
  openai: OPENAI_API_KEY
  anthropic: ANTHROPIC_API_KEY
  google: GOOGLE_API_KEY
  openrouter: OPENROUTER_API_KEY

# Named OpenAI-compatible endpoints, selected with --endpoint NAME.
endpoints: {}
#  example:
#    base_url: https://your-endpoint.example.com/v1/
#    api_key_env: YOUR_ENDPOINT_API_KEY
#    supports_vision: true   # the model accepts images
#    supports_schema: false  # the API enforces JSON schemas

# HTTP timeouts in seconds. request is the read timeout; 900 s suits the
# OpenAI flex tier, which may queue requests.
timeouts:
  request: 900
  connect: 10
  write: 30
  pool: 30

# Client-side rate limits per provider as [max_requests, window_seconds].
rate_limits:
  - [120, 1]
  - [15000, 60]
  - [15000, 3600]

# Retries of failed API calls (429, 5xx, timeouts, connection errors).
retry:
  max_attempts: 8         # every call, the first included
  timeout_attempts: 3     # calls that end in a timeout (billed, no usage)
  validation_attempts: 1  # calls that return unparseable structured output
  min_input_tokens: 0     # retry image requests reporting fewer input tokens
  # Wall-clock limit of one page across all attempts: auto (request timeout
  # times timeout_attempts plus 300 s), off, or seconds.
  page_timeout: auto

# Parallelism besides --concurrency (the LLM page requests).
concurrency:
  image_processing: 4     # page rendering and preprocessing for Tesseract
  audio: 4                # remote audio chunk requests
  delay_between_tasks: 0  # seconds each worker pauses before its next request

# Executables; '' finds them on PATH. Windows: winget install Gyan.FFmpeg.
tools:
  tesseract_cmd: ''
  ffmpeg_cmd: ''
  ffprobe_cmd: ''

# Local faster-whisper runtime (optional dependency: the audio extra).
whisper:
  device: auto        # auto | cpu | cuda
  compute_type: auto  # auto | int8 | int8_float16 | float16 | float32
  model_path: ''      # a local model folder; overrides --whisper-model
  download_root: ''   # cache folder of downloaded models; '' uses the default
  beam_size: 5
  vad_filter: true    # drop non-speech segments

# How long recordings are cut before they are sent (see --audio-chunk-seconds).
audio_chunking:
  max_request_bytes: 0  # 0 takes the provider limit (OpenAI 25 MB, Gemini 20 MB)
  overlap_seconds: 0    # audio repeated at each chunk boundary
  chunk_format: mp3     # mp3 | wav16 | copy
  sample_rate: 16000
  mono: true
  apply_to_local: false  # also chunk local whisper runs

# Extra fields of OpenAI audio-api requests (--language sets the language).
audio_api:
  prompt: ''     # vocabulary and style hint
  languages: []  # expected languages, gpt-transcribe only
  keywords: []   # literal terms expected in the audio, gpt-transcribe only

# Long-side limit in pixels of OpenAI pages sent at original detail; null
# keeps the model's default (3000 for gpt-6-luna, none for other models).
images:
  original_max_side_px: null

# Provider prompt caching of the static prompt (instructions, schema, context).
prompt_caching:
  enabled: true
  anthropic:
    ttl: 5m  # 5m or 1h
  openai:
    prompt_cache_retention: 24h  # documented models only; GPT-5.6+ get ttl 30m; null: none
  openrouter:
    anthropic_cache_control: true

# Decision-model pre-pass of --decisions shadow|apply (page types, skips).
decisions:
  image_max_px: 768       # longer side of the page images sent
  pages_per_request: 20   # consecutive pages per request
  profile: general        # page types: general, or culinary (adds recipes, menus, carving)
  timeout_s: 120
  thresholds: {}          # request fingerprint: no-text probability at which apply skips

# Folder of the tool's state files; '' uses ~/.chronotranscriber.
state_dir: ''

# Log folder, or a file ending in .log; '' writes logs/application.log in the
# repository.
logs_dir: ''

# Run defaults keyed by option name; flags override them.
defaults:
  # Context for items without a _transcr_context sidecar: the repository's
  # context file (relative to the current directory), topics as text, or none.
  fallback_context: context/transcr_context.txt
  # model: gpt-6-luna
  # reasoning_effort: xhigh
  # service_tier: default
  # formats: [md]  # txt, md; chronotranscriber.sqlite is always written
  # layout: hash
  # concurrency: 16
  # batch: false
  # retry_errors: true         # transcribe error pages again on a rerun
  # keep_working_files: false  # keep Tesseract images and audio chunks
```

A failed call waits 2 s, doubled per retry, plus up to 1 s of jitter, at most
120 s, a server's `Retry-After` included. In the `defaults:` block, options that
only the command line sets and invalid values are errors. Precedence is flag (or
wizard answer), then settings default, then code default; `--dry-run --json`
prints the source of each value, and the database records the effective spec of
every run. In the dry run's spec of the LLM method an unset output limit, response
format and image option shows the value the requests use, with source `resolved`
(the preset of new documents; a resumed document keeps the image settings it was
started with), and a provider other than OpenAI shows no service tier.

## Providers and Response Formats

The provider comes from `--provider`, else a `provider:` prefix on the model name
(`anthropic:claude-sonnet-5-5`), else the provider rules of the shared capability
table, so that a model runs on the provider its capabilities are resolved for:
Google's `models/` prefix, `gemini` and `gemma` are Google, a `vendor/model` name is
OpenRouter, `claude` is Anthropic, the GPT, o-series and `whisper` names are OpenAI,
and bare `llama`, `mistral`, `mixtral`, `qwen` and `deepseek` names are OpenRouter;
any other name runs on OpenAI. A provider set at a higher precedence level than a
model of another provider is an error. `--endpoint NAME` selects an endpoint from
`endpoints:` and implies provider `custom`; the endpoint declares whether its
model takes images (`supports_vision`) and whether its API enforces JSON schemas
(`supports_schema`).

`--response-format schema` sends the selected schema with the request, `json` puts
it into the prompt and validates the answer's required keys with retries, and
`text` uses the plain-text prompt; all three work on every provider. Unset, a
model with structured output (or an endpoint with `supports_schema`) uses
`schema`, any other `json`; a requested `schema` on a model without structured
output falls back to `json` and logs a warning. Under `schema` the answer is
checked only for being a JSON object; under `json` an answer that holds no JSON
object fails the page. OpenAI receives the schema as a strict `text.format`,
custom endpoints as a strict `response_format`, Anthropic as a `json_schema`
structured output with union types reduced to one type, and Google as
`response_schema`, with or without a thinking configuration. OpenRouter receives
a `response_format` without `strict`, so enforcement depends on OpenRouter and the
routed model.

The family rules in `chronotranscriber/common/capability_table.py`, shared with
AutoExcerpter and ChronoMiner, feed the resolver in
`chronotranscriber/common/capabilities.py`; `chronotranscriber/llm/capabilities.py`
adds the audio flags. The records of `common/capability_facts.py` each add, drop
or re-match a rule of the table, or set one field to the value the vendor's
documentation states (read 13.10.2026), and name their source. The first rule
matching the lowercased model name sets image and audio input, structured output,
reasoning and the documented effort levels, verbosity, sampling, the Flex tier
and output limits. An unmatched name, or one the vendor does not document
(`gpt-5.3`, `glm-flash-latest`), gets a conservative profile without image input
or structured output, so an LLM run with it stops at planning. Without
`--max-output-tokens`, OpenAI, Anthropic and Google models get their family's
limit, OpenRouter and custom endpoints 128,000 tokens.

Synchronous calls and batch bodies send the same OpenAI options. The service
tier reaches OpenAI only, and `flex` only the models OpenAI lists for Flex
processing (GPT-6, GPT-5.6, 5.5, 5.4, 5.2, 5.1 and 5 with the listed mini, nano
and pro variants, o3, o4-mini); other models get no tier, with one warning per
run. An effort a model does not document becomes the next level it does, or its
highest, with one warning per run: for example `none` and `minimal` become `low`
on gpt-6-astra and gpt-6.1-sol, and Gemini `minimal` becomes `low` on Gemini 3.1
Pro and 3.7/3.8 Flash. The verbosity reaches the GPT-5 and GPT-6 reasoning
models only; the chat-latest models are not reasoning models. With prompt
caching on, `prompt_cache_retention` goes to the models documented for extended
retention and GPT-5.6 and later get `prompt_cache_options` with a `30m` TTL, in
synchronous and batch requests alike; other models get neither. Unset image
options take the preset of the model's family (OpenRouter models that of the
family their name names, Claude, Gemini or OpenAI; other OpenRouter models the
custom preset):

| Family | `--dpi` | `--image-detail` | Format | JPEG quality | Grayscale |
| --- | --- | --- | --- | --- | --- |
| openai | native | original | jpeg | 95 | no |
| anthropic | native | auto | jpeg | 95 | no |
| google | 300 | high | jpeg | 95 | no |
| custom | 150 | high | jpeg | 95 | no |

`native` renders each PDF page at the density of its scan. Payloads are sized to
the model's documented limits; `original` detail without a model cap keeps the
rendered size. At `original` detail, gpt-6-luna pages are limited to 3,000 px on
the long side, since larger scans led it to invent text; `images.original_max_side_px`
in the settings sets this limit for every OpenAI model.

## Sources and Methods

| Method | Sources | Engine |
| --- | --- | --- |
| `llm` | PDF pages, folder images | a vision model, with context text and image per item |
| `tesseract` | PDF pages, folder images | local Tesseract after the `--ocr-preprocess` preset |
| `native` | PDF text layer, EPUB, MOBI | text extraction; a PDF without a text layer goes to Tesseract |
| `whisper` | audio | local faster-whisper (the `audio` extra) |
| `audio-api` | audio | OpenAI's transcription endpoint or a Gemini model |

Image folders hold JPEG, PNG, TIFF, BMP, GIF, WebP, JPEG 2000 (`.jp2`, `.j2k`),
PBM, PGM or PPM files in natural sort order; MOBI covers `.mobi`, `.azw`, `.azw3`
and `.kfx`. `--schema` names a schema in `schemas/`
(`markdown_transcription_schema`, `plain_text_transcription_schema`,
`plain_text_transcription_with_markers_schema`) or a path to your own with the
fields `transcription`, `no_transcribable_text` and `transcription_not_possible`.
The default schema prompt states no transcription rules of its own and points the
model to the schema's field descriptions, so a schema of your own carries its
rules there.

A `--context` value that names a `.txt` file that does not exist is an error.
With `--context auto`, an item's context is `{stem}_transcr_context.txt` beside a
file, else `{folder}_transcr_context.txt` beside its folder (an image folder `X/`
takes `X_transcr_context.txt` beside it), else `--fallback-context`. The packaged
settings set `fallback_context` to the repository's `context/transcr_context.txt`,
relative to the current directory; a fallback file that is missing gives no
context and a warning, an empty one gives none. A settings file of your own that
should keep that file as the fallback sets `fallback_context` in its `defaults:`
block. The context image follows the same sidecar lookup with
`_transcr_context_image.{ext}`, else takes `context/transcr_context_image.{ext}`
in the repository, and is sent before every page.

### Decision pre-pass

With `--method llm` and `--decisions shadow` or `apply`, a decision model gives
every page still to transcribe a page type before transcription: binding, blank,
scan_target, library_mark, handwritten, half_title, title_page, illustration,
preface_or_dedication, privilege_or_approval, contents_or_index, section_title,
body_text, table_or_chart, errata, advertisement, colophon or other. The settings
key `decisions.profile: culinary` adds recipe_text, menu_or_bill_of_fare and
carving_or_table_service. Pages go out `decisions.pages_per_request` at a time,
downscaled to `decisions.image_max_px`, and the types land in the store's
`page_types` table and the `page_type` column of `page_text`. The decision model
is OpenAI's `/v1/decisions` (a model id without `/`) or OpenRouter's decision API
(a model id with `/`, the default `typesafe/jev-1.14`); it uses the provider's key
variable from the settings.

`shadow` only records. `apply` also skips the pages whose probability of binding,
blank or scan_target reaches a threshold, marks them done with the outcome
`skipped_no_text`, and transcribes the rest. A threshold holds only for the
request settings it was validated on: `decisions.thresholds` maps a request
fingerprint (the hash of tool, provider, model, endpoint, taxonomy version,
profile and image settings) to the threshold, and `apply` without an entry for
the run's fingerprint records only and names the fingerprint in a warning.
`repair --skipped` transcribes skipped pages.

## Audio

Audio files are MP3, WAV, M4A, MP4, MPGA, MPEG, WebM, FLAC, OGG, AAC or AIFF;
OpenAI accepts `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav` and `webm`, Gemini
`wav`, `mp3`, `aiff`, `aac`, `ogg` and `flac`. `--method audio-api` runs on
provider `openai` (default model `gpt-transcribe`) or `google` (default
`gemini-3.6-flash`). A recording above the request limit (OpenAI 25 MB, Gemini
20 MB) or longer than `--audio-chunk-seconds` is cut by FFmpeg into chunks,
re-encoded as `audio_chunking.chunk_format`; each chunk is one unit. Local Whisper
reads whole files unless `audio_chunking.apply_to_local` is set or `--pages`
selects chunks. Output is plain text without timestamps or speakers, and audio
has no batch mode. Chunk indices are positional, so a run with other chunk
settings warns and transcribes the recording again; stored responses are kept.

## Output Contract

For an input `book.pdf` or an image folder `book/`:

```text
--layout hash (default)   <output>/book-<hash>/book.txt, book.md
--layout mirror           <output>/<input folder, relative>/book.txt, book.md
--beside-input            book.txt, book.md beside the input
```

Under `--layout hash`, an item found in a folder input is named after its path
relative to that folder, extension included: `in/book.pdf` gives
`book.pdf-<hash>/book.txt`, `in/sub/book.pdf` gives `sub/book.pdf-<hash>/book.txt`
and an image folder `in/book/` gives `book-<hash>/book.txt`. `<hash>` is 8 hex
digits of a SHA-256 of the item's path relative to the input folder (of its name
for a single-file input); a name that would exceed the Windows path limit is
shortened and gets such a suffix. `--formats` picks `txt` and `md`;
`chronotranscriber.sqlite` is always written into the output root (the output
folder, or the input's folder beside the input). Tesseract images and audio
chunks stay in the item's working folder only with `--keep-working-files`.

Each done or failed unit becomes one block, in item order: its text as the
provider or engine returned it, without postprocessing, or for a failed unit its
image or chunk name and the placeholder
(`page_0002_pre_processed.jpg: [transcription error]`). `txt` joins the blocks
with a newline, `md` with a blank line, and every file ends with one newline.
Units a `--pages` run did not select are left out. Page numbers read from headers
and footers appear as `<page_number>9</page_number>` when the schema asks for
them, as the default schema does; the default schema also writes `[illegible]`
for an unreadable word or passage and `[?]` for the uncertain part of a word. The
sidecars `{stem}_transcr_context.*` are inputs and are never written.

## Store

Every output root holds one `chronotranscriber.sqlite`, written by every run: the
runs with tool version and effective spec, the prompt, schema and context texts,
the documents and their units, every provider attempt verbatim, the batch
submissions and the registry of rendered files. Readers outside the tool use the
documented views, which stay stable; the tables behind them may change.

| View | One row per | Columns |
| --- | --- | --- |
| `page_text` | unit, by document and position | `document`, `kind`, `page_index`, `image_name`, `order_index`, `text`, `status`, `model`, `usage`, `outcome`, `error`, `provider`, `run_id`, `page_type` |
| `document_status` | document | `document`, `kind`, `input_path`, `status`, `units_total`, `units_done`, `units_failed`, `units_pending`, `last_run_id`, `updated_at` |
| `page_attempts` | provider attempt, in order | `document`, `kind`, `order_index`, `response_id`, `run_id`, `provider`, `model`, `attempt`, `submission_id`, `custom_id`, `raw`, `raw_kind`, `usage`, `error`, `http_status`, `created_at` |

A document is named by its rendered path relative to the output root, without
suffix (`book-1a2b3c4d/book`). In `page_text`, `kind` is `page`, `chunk` or `text`
(native text), `page_index` the 0-based PDF page of a page unit, `status`
`pending`, `done` or `failed`, `outcome` `ok`, `no_text`, `not_possible`,
`skipped_no_text` (skipped by `--decisions apply`) or `failed`, `page_type` the
type the decision pre-pass gave a page unit (null without one), and `usage` JSON
token counts (`input_tokens`, `output_tokens`,
`total_tokens`, `cached_tokens`, `reasoning_tokens`, in synchronous and batch
rows alike). The `error` columns keep the first 2,000 characters of an error
text; `raw` in `page_attempts` holds the response body as received. Open the
database read-only and close it after the query, so a running transcription never
waits on the reader:

```python
import sqlite3
from pathlib import Path

database = Path("out") / "chronotranscriber.sqlite"
connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
try:
    rows = connection.execute(
        "SELECT order_index, image_name, status, text FROM page_text "
        "WHERE document = ? ORDER BY order_index",
        ("book-1a2b3c4d/book",),
    ).fetchall()
finally:
    connection.close()
```

If a read fails with "attempt to write a readonly database" after an interrupted
run, run any `chronotranscriber` command that reads the folder (for example
`render --input <folder> --dry-run`) to roll back the journal.

`render` rewrites the files of every document in the databases at or below
`--input` without model calls, in the formats each document was rendered in
unless `--formats` names others. A file that matches neither its recorded hash
nor the new rendering was edited and is kept unless `--force` is given; a file
that cannot be replaced is marked stale while its units stay stored (exit 1).

A `chronotranscriber.sqlite` that holds tables but no schema version is not a
ChronoTranscriber store; every command refuses it with exit 2. A newer or
unreadable database also exits 2. None of them is changed or recreated.

`drop` deletes the selected documents of one store with their units, responses,
notes and render records, so the next `transcribe` run processes them from
scratch; run and submission rows stay. A rendered file is deleted while it
matches a hash recorded for it; a file changed outside the tool, missing,
unreadable or still recorded for another document stays and is named; while it
stays at the item's output path, the next run skips the item as `unstored`, so
move the file away to transcribe the item again. A document that another run
leases, or whose pages a submitting, submitted or unresolved batch submission
holds, is refused and stays (exit 1). `--dry-run` lists what a drop would delete
and changes nothing.

## Batch Processing

`--batch` submits the pages of LLM items to the batch API of OpenAI, Anthropic or
Google; OpenRouter, custom endpoints and audio have none. The run counts such
items as submitted, not complete. Requests are split by the provider's count and
byte limits, and each part becomes one submission in the database, committed
before the upload. With `--sync-fallback`, an item whose submission fails is
transcribed synchronously.

`batch check` finds the databases below `--input`, finalizes every finished batch
into the store, renders the affected documents in the formats of the run that
submitted the batch (or in `--formats`) and deletes the batch's files at the
provider; failed requests become placeholders for a rerun or `repair`. A refusal,
a content-filter stop and an answer cut at the output-token limit, in every
response format, count as failed requests, in synchronous runs as in batches.
A submission whose create got no confirmed response, or that a killed process
left without a provider batch ID, is unresolved, and its pages are never
resubmitted automatically. OpenAI and Google batches name their submission and
are adopted when exactly one matches. Anthropic batches do not: `batch check`
lists the candidates created around the submission, and `--adopt BATCH_ID` or
`--abandon SUBMISSION_ID` settles it. `--abandon` also gives up a submitted batch
whose results cannot be read (for example one the provider cannot find); the
batch is not cancelled, and a run submits its pages again.

`batch check` makes one pass unless `--wait` is given. With `--wait` it checks
again every `--poll-minutes` (15) until no submission is pending, then exits 0,
or 1 when a submission failed or is unresolved. A submission stays pending while
its batch runs and while another run holds one of its documents. No check starts
after `--wait-hours` (50); a check that is running finishes, and the command exits
1 with `stopped` `timeout`. The limit is soft: OpenAI and Anthropic expire a batch
after 24 hours and Google after 48, a check finalizes an expired batch as failed,
and the wait then ends on its own; the limit bounds a wait whose checks cannot
read the batch statuses. Between checks it holds no lease, so other runs may
work the same folder. A check that fails with a transient error (such as a lost
connection, a timeout, a rate limit or a server error) is reported and repeated
at the next interval; any other error ends the wait with `stopped` `error`.
`--adopt` and `--abandon` apply to the first check only. The counts at the end
are read from the store for every submission the checks saw, so a submission
another run finalized during the wait counts as finalized. `transcribe --batch`
followed by `batch check --input OUTPUT --wait` runs a batch job from submission
to rendered files.

`batch cancel` cancels batches by ID (the provider is inferred from the ID),
every unfinished batch of each provider whose key is set (`--all`), or the open
submissions below a folder (`--input`), whose pages return to pending.

## Repair

`repair --input PATH` takes a rendered file or the input it was transcribed from
and finds its document in the nearest `chronotranscriber.sqlite` at or above it;
a file no database records exits 2. A `chronotranscriber.sqlite` names every
document it holds, and a folder (by default the current one) every document of
the databases at or below it. For each document it selects the failed pages of
the chosen causes, narrowed by `--indices` (one document only), skips pages an
open or unresolved batch holds, renders them with the image settings they were
first rendered with and transcribes them synchronously or with `--batch`. The
stored responses stay, and the documents' files are rendered again. A dry run
plans without the API key variables.

## Resume

Resume is the default and reads only the store; a dry run shows each item's state.

| State | Meaning | Action |
| --- | --- | --- |
| `none` | no stored units | transcribe from the start |
| `partial` | units are pending or failed (failed ones not under `--no-retry-errors`) | transcribe those units |
| `complete` | no pending or failed units (failed units count as done under `--no-retry-errors`) | skip; missing or stale files are rendered again without calls |
| `unstored` | a rendered file exists without store rows, such as a hand-edited file `drop` kept | skip and list; `--force` redoes it |
| `blocked` | another live run holds the item, or its pages wait for a batch | skip; a held item makes the run exit 1 |
| `refused` | units remain, but the stored pages used other image settings | not processed, counted as failed; `--force` starts it over |

Besides the reset its option row describes, `--force` deletes the units another
method stored (for example the `text` units of `native`), and with `--pages` it
leaves the document partial. A resume with another model proceeds with a warning,
and each unit records its model. `repair` transcribes failed units again without
a rerun.

## Architecture

```text
chronotranscriber/
  __main__.py   entry point: dispatch the subcommand
  spec.py       run specifications and the option tables
  settings.py   typed settings: machine values and the defaults block
  run.py        core: discover, plan, run
  events.py     run events reported to the front ends
  ext.py        extension seam with pass-through defaults
  providers.py  provider names and inference from model names
  cli/          argparse from the option tables, reporter, --json, exit codes
  ui/           questionary wizard, review screen, rich progress
  common/       vendored modules shared with the sibling tools
  llm/          client factory, model calls, capabilities, prompts
  sources/      PDF, image folder, EPUB, MOBI and audio sources, auto mode
  methods/      llm, tesseract, native, whisper and audio-api transcribers
  batch/        submit, check and wait, finalize, cancel and repair
  images/       Tesseract preprocessing
  output/       store views, resume classification, renderers, drop, paths
  resources/    prompts and the settings example
schemas/        public transcription schemas
context/        default transcription context
```

The core imports nothing from `cli/` or `ui/`; it never prints, prompts or exits
and reports through run events. `common/` holds tool-agnostic modules (option
tables, the shared option rows, wizard framework, settings loader, the shared
settings blocks, item and page selection, structured calls, retry, rate
limiting, images, image presets, model capability table, provider request
builders and options, capture, SQLite writer, store, batch backends and batch
reconciliation, interrupt handling, the heartbeat file) vendored byte-identically
across the sibling tools: `common/MANIFEST.json` pins each file by its SHA-256,
and `tests/common/test_manifest.py` fails when a copy drifts. Each tool passes
its own values to the shared presets and request options: ChronoTranscriber's
image presets are `IMAGE_PRESETS` in `spec.py`, its request rules `RULES` in
`llm/options.py`. The option rows every tool declares (model, images, output
location, run control, batch check and cancel, render) come from
`common/standard_options.py` with ChronoTranscriber's defaults and help texts,
and `settings.py` reads the shared settings blocks through
`common/settings_schema.py`; `tools`, `whisper`, `audio_chunking` and `audio_api`
stay its own. `ext.py` holds the extension hooks (extra arguments and the argv
words that let a command copied from the ui reproduce them, settings blocks,
wizard steps, a startup hook, consent to batch submission that may give the
reason for a refusal, the key variables plan-time checks require, the access
line of the review screen, client options, an attempt context around every model
call); its defaults pass everything through.

## Development

```bash
uv sync --all-extras
uv run pytest                        # hermetic suite
uv run ruff check .
uv run ruff format --check .
uv run mypy chronotranscriber        # strict
```

The suite removes provider keys from the environment, points the home folder at a
temporary directory, blocks non-loopback connections and fails any test that
writes outside its temporary directory. Model calls go to a scripted fake chat
model behind the real client; batch services, Tesseract, Whisper and the audio
API have fakes. After an intended change, rewrite the golden files with
`CT_UPDATE_GOLDEN=1` and review the diff; `tests/README.md` describes the layout.

The tables of the Command Reference mirror the option tables in
`chronotranscriber/spec.py`; `tests/cli/test_readme_options.py` fails when the two
differ. A new option needs an option-table entry and a wizard step
that sets it; `tests/ui/test_parity.py` enforces the step.

Evaluation data for a subset of historical sources (ground-truth transcriptions and
benchmark runs) is available from the author upon request.

## Versioning

Every release is described in [CHANGELOG.md](CHANGELOG.md).

## License

MIT License. Copyright (c) 2025 Paul Goetz. See [LICENSE](LICENSE) for details.
