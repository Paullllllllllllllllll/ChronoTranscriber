# Changelog

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
