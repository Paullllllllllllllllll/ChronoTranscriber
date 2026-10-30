# Changelog

- **v1.2.2** (30 October 2026) -- A rendered file no longer stays current after
  a later run changes the units it was rendered from. Each write that creates,
  writes, resets or deletes a document's units, and the abandonment of a batch
  submission that holds them, marks every rendered file of the document stale
  in the same transaction, and rendering clears the flag of each file it
  records. Earlier, a format rendered by a partial run (`--pages 1 --formats
  txt`) stayed current after a run in another format completed the document,
  so a later `--formats txt` run kept the incomplete file. The change is in the
  shared `common/store.py`, which ChronoMiner and AutoExcerpter vendor.
- **v1.2.1** (30 October 2026) -- A resumed document no longer mixes two
  versions of its input. Every document now records the identity of its input,
  a file's SHA-256 or, for an image folder, which recorded none before, the
  SHA-256 of its images in page order with their file names, and a
  transcription run refuses an item whose stored identity differs, partial and
  complete alike, until `--force` transcribes it from scratch; `repair` refuses
  such a document too, and so does a run whose input changes between planning
  and the item's turn or cannot be hashed. Earlier, a resume recorded the new
  file's hash and transcribed only the pending pages from it. Planning now reads
  every input once. A document that records no identity, such as every image
  folder transcribed so far, is trusted and records its identity on the next run
  that writes it, with a note. The page order of an image folder breaks ties of
  the natural order by file name (`page_01.png` before `page_1.png`), so it no
  longer depends on how the file system lists the folder; this reorders only
  folders whose names tie. Tesseract no longer files a folder image named like
  a PDF page (`page_1.png`) under the page number in its name, which could
  store a page at another image's index; it takes its place in the page order
  and records its file name as source index. A resume or repair refuses a
  folder document, with or without an identity, whose stored pages no longer
  fit its images (an image added or removed, or a page labeled with another
  image than the one at its index) until `--force`, and the classify commands
  refuse it, so that no answer joins the page of another image. A run and
  `repair` check the input and the stored identity again under the document's
  lease and
  record an identity only there. A forced replacement writes the new unit
  plan, the image settings and the new identity, drops the old pre-pass page
  types and the units outside the plan, and marks every rendered file stale,
  all in one transaction that an open or unresolved batch submission holding
  any of the document's units refuses. The classify
  commands record the same identity; folder answers recorded by v1.2.0 count
  as answers about another input, so classify those folders again.
- **v1.2.0** (30 October 2026) -- `classify pages` and `classify documents`
  ask a decision model the questions of a classifier file about the pages of
  PDFs and image folders, or about each item from its first pages, and record
  the answers without transcribing or gating anything. The commands select
  items as `transcribe` does, take the name of the stored document of the same
  input path, join `page_text` on the page's ordinal, reuse the stored answers
  of an unchanged request unless `--force` is given, flag the answers of a
  changed input (a file's SHA-256 or an image folder's digest) stale, refuse an
  item whose stored document records another input file, and report the asked, reused and
  failed units with their `classify` usage in the JSON summary. Five public
  classifiers ship in `classifiers/` (`page_kind`, `script_and_hand`,
  `language`, `layout`, `document_type`). The answers land in the new
  `classifications` view of the store, whose component registers beside the
  tool's own; that component steps to version 3, so earlier releases refuse an
  upgraded store. The vendored common package gains the shared classification
  core: classifier files (`classifiers.py`), request planning and the decision
  runner (`decision_runner.py`), the `classifications` store component
  (`classify_store.py`), `images.decision_jpeg`, `store.write_unleased` and the
  shared classify options. The page-type pre-pass of `--decisions` runs on the
  shared runner and sends the same requests as before; as a deliberate change,
  it now counts the billed usage of a malformed answer, which the earlier pass
  dropped.
- **v1.1.3** (29 October 2026) -- The Pillow requirement is raised to
  13.0, the first release that fixes the FlashPix, TIFF-writer and
  multi-frame loader advisories, so that no install resolves a vulnerable
  Pillow. The lock already held 13.0.0.
- **v1.1.2** (29 October 2026) -- The guided UI asks for the decision-model
  pre-pass of an LLM transcription after the page images: off, shadow or
  apply, and with the pass on, the decision model, its API and its endpoint.
  The printed command carries the answers as `--decisions`,
  `--decision-model`, `--decision-provider` and `--decision-endpoint`.
- **v1.1.1** (29 October 2026) -- The decision pre-pass keeps the retry budgets of
  the transcription calls: timed-out attempts stop at `retry.timeout_attempts`,
  malformed answers count against `retry.validation_attempts`, and the vendored
  `decision_client.py` raises `DecisionStatusError`, whose HTTP status the retry
  ladder reads. A pre-pass that fails after its retries leaves every page to
  transcription with a warning instead of stopping the document.
- **v1.1.0** (29 October 2026) -- `--decisions shadow|apply` adds a decision-model
  pre-pass to `--method llm`: OpenAI's `/v1/decisions` or OpenRouter's decision
  API (default `typesafe/jev-1.14`) gives every page still to transcribe one of
  18 general page types, or 21 under `decisions.profile: culinary`, read in
  groups of `decisions.pages_per_request` downscaled pages. The types land in
  the new `page_types` table and the `page_type` column of `page_text` (store
  component version 2). `apply` skips pages that are likely binding, blank or
  scan target, with the outcome `skipped_no_text`, but only where
  `decisions.thresholds` holds a threshold for the run's request fingerprint;
  `repair --skipped` transcribes such pages. The vendored common package gains
  `decisions.py` (request bodies and typed answers for both dialects),
  `decision_client.py`, `page_types.py` and the shared decision options and
  settings block.
- **v1.0.5** (27 October 2026) -- The lockfile moves `filelock` from 3.32.2
  to 4.2.4 and `setuptools` from 83.0.0 to 84.2.1, which closes the open
  Dependabot advisories (one high and one low for `filelock`, fixed in 4.2.1;
  one low for `setuptools`, fixed in 84.1.0). Both are transitive: `filelock`
  comes through `huggingface-hub` and `setuptools` through `ctranslate2`, both
  under the `audio` extra. The code imports neither, and the test suite,
  ruff and mypy pass unchanged.
- **v1.0.4** (27 October 2026) -- The capability table knows `gpt-6.1-terra`:
  it takes text and image input and structured outputs like `gpt-6.1-sol`,
  accepts the reasoning efforts none to max, and shares the GPT-6 handling of
  verbosity, Flex and prompt caching. Before, the model resolved as text-only
  and image transcription refused it. The vendored `chronotranscriber/common/`
  modules now meet the Google docstring rules as well, so ruff no longer
  exempts them. The Ctrl+Break test hands its child process a settings file in
  the test's temporary folder, so the child no longer reads
  `config/settings.yaml` or writes to the repository's `logs/`.
- **v1.0.3** (27 October 2026) -- Documentation patch: ruff now checks
  docstrings with the pydocstyle rules (`D`) under the Google convention, and
  the package's docstrings follow it, with one-sentence summaries and new
  docstrings for the protocol members of the run events, the batch submitter,
  the transcription methods and the submitted document. The vendored
  `chronotranscriber/common/` modules are exempt, since they stay
  byte-identical with the other tools. Behavior is unchanged.
- **v1.0.2** (26 October 2026) -- Security patch: the lockfile moves Pillow to
  13.0.0, which closes the Pillow 12.3.0 advisories, among them the TIFF decoder
  reads GHSA-c6hw-5grf-ff96 and GHSA-q45j-58j2-45rg and the MIC frame-count loop
  GHSA-22hg-r758-rrw9, which scanned page images can reach. Dependency versions
  only; no code changes.
- **v1.0.1** (24 October 2026) -- Security patch: the lockfile moves certifi to
  2026.10.23, which closes GHSA-6q48-6c7w-9gw2, and urllib3 to 2.8.0, which closes
  GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g. Dependency versions
  only; no code changes.
- **v1.0.0** (19 October 2026) -- First release. ChronoTranscriber is one
  `chronotranscriber` package whose core in `run.py` has three entry points,
  discover, plan and run; the command line (`cli/`) and the guided terminal UI
  (`ui/`) both call them, and the core reports through run events instead of
  printing. The commands `transcribe`, `repair`, `batch check`, `batch cancel`,
  `render`, `drop` and `ui` are each built from an option table in `spec.py`, from
  which the argparse parser and the wizard steps follow; every command offers
  `--dry-run`, a one-line `--json` summary and documented exit codes.
  `settings.py` reads one settings file of machine values (API key variables,
  endpoints, timeouts, rate limits, retries) with an optional `defaults:` block
  keyed by option names; a flag wins over a settings default, and a settings
  default over the code default. Six sources (PDF, image folder, EPUB, MOBI, audio,
  and an auto mode for mixed folders) meet five methods: `llm` (a vision model on
  OpenAI, Anthropic, Google, OpenRouter or a named OpenAI-compatible endpoint, with
  the response formats `schema`, `json` and `text`), `tesseract`, `native` (the
  text layer), `whisper` (faster-whisper) and `audio-api`; the default model is
  gpt-6-luna at `xhigh` effort. Every output folder holds a
  `chronotranscriber.sqlite` store that commits each unit as it finishes, together
  with every provider response verbatim, the runs with their effective specs and
  the batch submissions; its views `page_text`, `document_status` and
  `page_attempts` are the read contract. Leases keep two runs from working one
  document, resume reads only the store, `render` writes the `txt` and `md` files
  from it without model calls, `drop` deletes documents from it, and `repair`
  transcribes failed pages again. The batch layer submits, checks (once or with
  `--wait`), finalizes and cancels through the batch APIs of OpenAI, Anthropic and
  Google, and records each submission before its upload, so that an unconfirmed
  submission can be adopted or abandoned. The logic shared with AutoExcerpter and
  ChronoMiner lives in `common/`, vendored byte-identically and pinned by hash in
  `common/MANIFEST.json`: the store and its SQLite writer, the option tables and
  standard option rows, the settings schema and loader, the wizard framework, the
  provider request builders and request options, the model capability table with
  its sourced vendor facts, the image presets, retry, rate limits, structured
  calls, response capture, item selection, the batch backends with batch
  reconciliation, interrupt handling and the heartbeat file. ChronoTranscriber
  passes its own image presets and request rules to them. The extension seam
  `ext.py` defines the hooks through which a local access layer extends the tool:
  extra arguments (`register_args`, `extra_argv`), settings blocks
  (`settings_blocks`), wizard steps (`wizard_steps`), a startup hook (`startup`),
  consent to batch submission (`batch_allowed`), the key variables and the access
  line of the review screen (`key_variables`, `describe_access`), client options
  (`client_options`) and an attempt context around every model call (`attempt`).
  In this release every hook passes everything through.
