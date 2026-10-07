# Changelog

- **v4.4.2** (7 October 2026) -- The shared ledger (module version 2.1.6)
  moves `gpt-6-luna` from the small to the large default pool, where the
  other GPT-6 models already sit.
- **v4.4.1** (30 September 2026) -- A page that cannot be rendered or
  encoded, including one over `max_image_bytes`, is now recorded as a
  `[transcription error]` placeholder instead of vanishing from the output,
  so the item counts as failed and repair can re-render the page. In batch
  mode such a page gets metadata but no request, and finalization writes
  its placeholder; a document with no renderable page fails submission.
- **v4.4.0** (30 September 2026) -- Native scan resolution: `target_dpi:
  native` renders each PDF page at the density of its scan image, and payloads
  are sized to each model's documented limits (OpenAI `original` patch caps,
  Anthropic edge and visual-token budgets). Optional lossless PNG payloads with
  a base64 size guard, one log line per downscaled page, and per-page image
  provenance. Resume rejects changed image settings through a preprocessing
  fingerprint, and repair re-renders with the recorded settings, raw source
  images included. The example config is standardized and now ships native
  JPEG at quality 95 for OpenAI and Anthropic. Numeric runs also respect the
  model caps, so newer OpenAI models keep larger pages and Anthropic pages are
  reduced before submission. To roll back, set `target_dpi: 300`,
  `jpeg_quality: 100` (custom stays 90), API `original_max_side_px: 6000` and
  `original_max_pixels: 10240000`, Anthropic `high_max_side_px: 2576`, and the
  previous detail and profile values; the model caps stay in effect.
- **v4.3.1** (30 September 2026) -- The shared ledger (module version 2.1.5)
  adds `gpt-6-astra` to the large default pool; before, its usage was
  recorded without a pool and escaped the per-key pool caps.
- **v4.3.0** (23 September 2026) -- Register `gpt-6-astra` with the same
  profile as the other GPT-6 models: pinned to the Responses route for
  image detail `original`, 1.05M context and 128k output.
- **v4.2.0** (23 September 2026) -- Register `gpt-6-sol`, `gpt-6-luna`
  and `claude-opus-5-5`. Both GPT-6 models are pinned to the Responses
  route, the only route that accepts image detail `original`. Opus 5.5
  joins the Anthropic adaptive-thinking families; without that entry it
  would receive a `budget_tokens` thinking block and fail with HTTP 400.
  The shared ledger (module version 2.1.4) adds `gpt-6-sol` to the large
  default pool and `gpt-6-luna` to the small one.
- **v4.1.0** (22 September 2026) -- A new `--service-tier
  {auto,default,flex,priority}` flag on `transcribe.py` overrides the
  configured OpenAI service tier for one run, without editing
  `concurrency_config.yaml`; it reaches both synchronous calls and batch
  submissions, where `flex` still maps to `auto`.
- **v4.0.1** (21 September 2026) -- Security patch for the notebook
  dependency stack: jupyter-server moves from 2.20.0 to 2.21.1, which
  fixes CVE-2026-86049, where a request that failed with a 500 error
  logged its `Referer` header unredacted and could expose an
  authentication token carried in a URL. Only locally run evaluation
  notebook servers were affected.
- **v4.0.0** (20 September 2026) -- Breaking rename of the three entry
  points that carried transitional names: `main/unified_transcriber.py`
  is now `main/transcribe.py`, `main/repair_transcriptions.py` is now
  `main/repair.py`, and `main/postprocess_transcriptions.py` is now
  `main/postprocess.py`; the old file names are removed outright, with
  no compatibility shims, so callers, scripts, and documentation must
  switch to the new paths. The `command` field of the `--json` summaries
  follows the new names (`repair`, `postprocess`). Also re-synced the vendored shared token
  ledger module to upstream version 2.1.3 and updated its pinned content
  hash accordingly.
- **v3.2.0** (14 September 2026) -- Register `claude-opus-5`,
  `gemini-3.7-flash`, and `gemini-3.6-flash` in the capability registry
  so they no longer fall through to the bare provider defaults; add
  `claude-opus-5` to the Anthropic adaptive-thinking families; accept
  `max` as a reasoning-effort level in the CLI and map it in the
  OpenRouter and Google providers. Security patch: the locked `tornado`
  moves from 6.5.7 to 6.5.8 (CVE-2026-82397, GHSA-mpf4-983q-p7j4).
- **v3.1.1** (15 August 2026) -- Security patch for a transitive
  dependency. The locked `cryptography` build moves from 49.0.0 to
  50.0.0, which closes CVE-2026-69247; the package reaches the project
  indirectly through `google-auth`, so no direct requirement changed and
  the public API is untouched. The full test suite passes unchanged on
  the new resolution.
- **v3.1.0** (13 August 2026) -- Request-stall hardening. A configurable
  per-page wall-clock watchdog (`page_timeout`, default `auto`) now bounds
  one page across all retry attempts, so a single stalled request can no
  longer park a whole run for hours; request timeouts get their own smaller
  retry budget (`retry.timeout_attempts`, default 3) because each timed-out
  attempt is billed server-side without a usage payload; the OpenAI-family
  clients (chat and audio) receive per-phase HTTP timeouts
  (`connect_timeout`/`write_timeout`/`pool_timeout`, defaults 10/30/30 s)
  instead of a scalar that silently set the connect timeout to the full
  read budget; retry log lines now name the page being retried instead of
  `<unknown>`; and the progress display reports every increment within the
  final interval window plus a 45 s report-rate floor, so end-of-run stalls
  are visible. All new config keys default safely when absent.

- **v3.0.1** (4 August 2026) -- Security patch from the weekly sweep:
  JupyterLab moves to 4.6.2, closing two high-severity advisories, and the
  ignore rules now cover `backup/` and every `.env*` spelling so neither a
  local archive nor a credentials file can be staged by accident. No
  runtime change.

- **v3.0.0** (2 August 2026) -- Audio transcription. Speech recordings become a
    first-class processing type: `--type audio` with `--method audio-api` sends
    each recording to the OpenAI audio API (`gpt-transcribe` by default, also
    `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, and `whisper-1`) or to Google
    Gemini, while `--method whisper` transcribes offline through a local
    faster-whisper model shipped as the new optional `audio` extra. A sixth
    config file, `config/audio_config.yaml` (with a tracked
    `audio_config.example.yaml`), selects the remote venue and holds the
    per-venue request parameters, chunk planning, local Whisper runtime, ffmpeg
    executable paths, and a speech-tuned postprocessing profile. Recordings
    above a venue's per-request limit are cut into deterministic ffmpeg chunks
    and each chunk is written as its own JSONL record, so resume,
    `--retry-errors`, and the evaluation tooling behave as they do for pages;
    short files are sent whole and never invoke ffmpeg. Output is plain text
    without timestamps or diarization, audio runs synchronously (no provider
    offers a batch API for it), and `repair_transcriptions.py` does not cover
    audio. Major bump for the new modality, config file, and CLI surface; the
    image and document paths are unchanged.
- **v2.4.0** (19 July 2026) -- Dependency and documentation release from the
    third maintenance sweep. Raise all direct dependency floors to current
    stable releases (langchain stack, openai 2.46, anthropic 0.117,
    google-genai 2.12, PyMuPDF 1.28, pillow 12.3, numpy 2.5) and regenerate
    the lockfile; hold opencv-python below the freshly released 5.0 major
    until it stabilizes. Verify the google-genai inline-batch request shape
    and langchain-openai disabled_params semantics against the upgraded SDKs;
    make the deep-nesting JSON parse test robust on Python 3.14. Documentation:
    describe per-provider batch chunk limits and id-shape cancel/status
    routing in the README, refresh test-suite counts, move eval docs to the
    uv workflow, and correct a stale rate-limiter comment.
- **v2.3.0** (19 July 2026) -- Batch-backend and provider-correctness release
    from the second maintenance sweep. Batch: Google inline submission now
    builds SDK-valid requests (it previously failed validation before any
    network call); batch-id recovery runs on partial tracking loss and
    restores provider and correlation metadata; items resubmitted after
    finalization stay reachable via their new batch ids; duplicate
    submissions can no longer double every page of the output; repaired
    pages are merged back into the temp JSONL so regeneration keeps repairs;
    interactive repair discovery covers all output roots, honors co-located
    outputs, and skips backup/cleaned sidecars; cancellation routes
    Anthropic and Google ids to the right provider; providers without a
    batch API are no longer offered batch and fall under the explicit
    sync-fallback policy. Providers: OpenRouter sampler parameters and
    context-image details respect model capabilities; Anthropic thinking
    budgets are clamped to max_tokens; Gemini 2.x no longer receives
    thinking_level; context images now reach Anthropic, Google, and
    OpenRouter; truncation is detected across all three response shapes;
    null transcriptions no longer become the literal "None". Transcription
    core: live progress reporting during synchronous runs; image-folder
    Tesseract keeps absolute page order across resumed subsets; dropped
    preprocessing pages fail loudly; Ctrl+C interrupts budget waits; EPUB
    chapter filtering no longer discards real chapters; the console prints
    each warning once. CLI argument definitions are unchanged.

- **v2.2.0** (19 July 2026) -- Correctness and interactive-mode release from a
    full maintenance sweep. Output integrity: final txt/md/json outputs are
    written atomically; a crash-truncated JSONL line no longer swallows the
    next resume run's first record; failed output regeneration now fails the
    item and preserves the temp JSONL instead of deleting the only copy of
    paid transcriptions; failed streaming JSONL appends surface as page
    failures; Tesseract page order is derived from absolute page numbers
    across resumed subsets; same-stem EPUBs/MOBIs in different subdirectories
    no longer collide. Batch: finalized temp JSONLs carry a marker so re-check
    runs are idempotent and no longer revert repairs; non-OpenAI results are
    reconciled for missing pages and cancelled/expired batches count as
    failed; Gemini 429/5xx errors are now retried (cause-chain status
    classification); repair never blanks a placeholder with an empty
    re-transcription; batch fallback parsing keeps positional custom_id
    alignment; plain text containing an embedded code fence is no longer
    reduced to the fence content. Interactive mode: cancellation now requires
    confirmation; resume mode is chosen before auto-mode filtering and the
    filter targets the real auto output root; the configured output format is
    honored; MOBI processing and a retry-errors resume option join the
    wizard; back-navigation dead loops and value corruption in the
    postprocess wizard are fixed; completion output is a comprehensive
    overview (real counts, skipped items, failed submissions, token usage,
    absolute output paths, batch job IDs); prompts exit 130 on Ctrl+C. CLI
    argument definitions are unchanged.

- **v2.1.0** (17 July 2026) -- Performance and robustness release; API
    behavior and LLM payloads for in-cap pages are unchanged. The
    hallucination-loop detector drops from ~99 to ~8 ms per page via
    prefix-sum window checks, unblocking the async event loop; Unicode
    normalization and spacing cleanup in postprocessing run via cached
    translation tables and precompiled regexes (2.4x on large documents);
    batch repair replaces O(n squared) target scans with dict lookups (up
    to ~1,000x on large failure sets); batch request building hoists
    run-invariant prompt, schema, and context preparation out of the
    per-page loop (3.7x); capability detection is cached; PDF rasterization
    uses zero-copy pixel buffers and skips no-op EXIF transposes; resume
    re-parses each temp JSONL once instead of up to four times. A new
    `render_strategy` option (`direct`, the default, vs. the legacy
    `supersample`) derives the render DPI from the active provider resize
    profile (2.25x on the Anthropic profile, 6.2x on box-fit; the shipped
    OpenAI `original` profile stays byte-identical for A4 pages at 300
    DPI). The Anthropic `high_max_side_px` default rises from 1568 to 2576
    for the Claude high-resolution tier; Tesseract preprocessing defaults
    to Otsu binarization (Sauvola remains available) and estimates deskew
    angles on a downsampled copy (~4x, angle deviation 0.0 degrees in
    testing); JSON salvage of malformed responses is bounded and degrades
    to the transcription-error fallback instead of crashing on pathological
    inputs.
- **v2.0.5** (17 July 2026) -- Documentation reconciliation release; no code
    changes. Correct the package count (ten packages under `modules/`, not
    nine) and the bundled schema count (three, not four); document the
    previously missing `--model-verbosity`, `--max-output-tokens`, and
    `--output-mode` flags in the CLI reference; refresh the test-count claim
    (roughly 1,500 tests, api-marked live tests deselected by default).
    Rewrite the stale `tests/README.md` (its file tree listed 12 of 71 test
    files, one nonexistent, and a misregistered marker set) as a lean,
    accurate guide. Fix `eval/README.md`: invalid `--type pdf` example
    commands (the CLI accepts `pdfs`), the Gemini 3 Pro model id
    (`gemini-3-pro-preview`), and the undocumented `latex_tables/` and PDF
    chart outputs. Repair eight stale "Used by:" module paths in the shipped
    `*.example.yaml` config templates that still referenced the pre-refactor
    `modules/processing/`, `modules/operations/`, `modules/core/workflow.py`,
    and `modules/llm/batch/` layout.
- **v2.0.4** (16 July 2026) -- Adopt the fully typed shared-ledger test
    (vendored byte-identically across the ChronoTools repos; the ledger module
    itself is unchanged at v2.1.1): Any-typed dynamic module handle,
    `pytest.MonkeyPatch` annotations, and a covariant frozen-datetime
    override, keeping the three vendored copies in sync.
- **v2.0.3** (16 July 2026) -- Shared token-ledger module updated to
    v2.1.1 (vendored byte-identically from ChronoMiner): the lock-free
    reads `read_combined` and `read_breakdown` now degrade gracefully
    (return None) when the ledger file contains valid-but-non-dict JSON
    such as `null` or `[]`, instead of raising `AttributeError` in
    violation of the module's never-crash contract; the vendored test
    suite gains the matching regression test and pins the new module
    hash.

- **v2.0.2** (16 July 2026) -- Three robustness fixes and a repo hygiene fix.
    Tesseract folder preprocessing no longer clobbers source images that share
    a stem across extensions (scan_001.png + scan_001.tif previously mapped to
    one output file, silently dropping a page): colliding stems now get
    extension-inclusive preprocessed names, mirroring the CT-9 GPT-path fix,
    while collision-free files keep their legacy names so existing resume
    artifacts still match. All transcription outputs (txt, md, json) and the
    repair write-back/backup are now written with LF line endings on every
    platform instead of CRLF on Windows; readers use universal newlines, so
    old CRLF files resume and repair unchanged. The OpenAI batch result parser
    now concatenates every output_text part across all message items in order
    (previously it kept only the first part per message and let a later
    message overwrite earlier content). Finally, a UTF-8 BOM at the start of
    .gitattributes -- which made Git parse the line-1 comment as a pattern and
    warn "policy: is not a valid attribute name" on every checkout -- is
    stripped.
- **v2.0.1** (16 July 2026) -- Fix the OpenAI batch backend silently dropping
    `llm_detail: original` (the v2.0.0 recommended default): batch request
    bodies only accepted low/high and fell back to the API's default detail,
    degrading batch transcriptions relative to the synchronous path. The
    backend now validates `original` against the model's
    `supports_image_detail_original` capability, matching the sync provider
    and the legacy request builder. Regression tests cover capable
    (gpt-5.6-luna) and incapable (gpt-4o) models.
- **v2.0.0** (16 July 2026) -- New recommended standard configuration for
    best processing results, shipped as the bundled example defaults: OpenAI
    gpt-5.6-luna at reasoning effort high, original (full-resolution) image
    detail, flex service tier with a 900 s request timeout, and transcription
    concurrency of 16 tuned for OpenAI API tier 3. Major bump because the
    public defaults change processing behavior for fresh clones.
- **v1.24.1** (16 July 2026) -- Deselect the live API smoke tests from the
    default pytest run: `pytest` now applies `-m 'not api'` via `addopts`, so
    the 19 api-marked tests in `tests/integration/test_live_api.py` no longer
    fire real LLM calls (and burn tokens) whenever API keys are present in the
    environment. Run them explicitly with `pytest -m api`. Also clears
    formatter drift in one test file.
- **v1.24.0** (12 July 2026) -- Per-key token accounting and definable daily
    pools, fixing the guard that could block a free local endpoint on paid
    OpenAI usage. The shared cross-tool ledger moves to schema v2: every API
    call is stamped with its provider, the NAME of the env var that served
    it (key values are never stored), and a pool label, recorded per
    (tool, provider, key env, pool) alongside the per-tool totals. Budget
    enforcement becomes two-tier: per-(key, pool) daily caps as the primary
    gate -- pools definable per provider in
    `daily_token_limit.per_key_pool_caps` (bare-int cap or `{cap, models}`
    mapping), with built-in defaults mirroring OpenAI's complimentary daily
    token program -- and the combined `daily_tokens` cap as a secondary
    guard, scoped by the new `scope` knob. Under the default `scope:
    pooled`, calls whose model belongs to no pool (e.g. a self-hosted local
    endpoint) are counted but never blocked. The wait loop names the
    exhausted key, reports other keys' remaining pools, refreshes pool
    config mid-wait, and warns that a key remap takes effect on the next
    run (the provider is built once per run). v1 ledgers are adopted in
    place with un-attributable usage kept under an "unattributed" row.
    Vendored `shared_ledger.py` v2.1.0.

- **v1.23.1** (9 July 2026) -- Fix the repair pipeline to report truthfully.
    Previously `repair_transcriptions` announced `[SUCCESS]` and exited 0 even
    when no placeholder was actually repaired, silently dropping unresolved
    targets; a live incident left a `[transcription error]` marker in place
    under a SUCCESS banner. The repair outcome is now tied to actual file
    content: a new `_count_unrepaired_lines` helper counts remaining
    placeholders, an empty target set returns the full failure count with a
    WARNING instead of a false zero, and `[SUCCESS]` is emitted only when no
    line is left unrepaired, otherwise `[WARN] ... left N of M line(s)
    unrepaired`. The CLI now exits 1 whenever any line remains unrepaired and
    the `--json` payload reflects the true counts, establishing a firm
    exit-code contract for downstream automation. Regression tests cover the
    single-line filename-prefix marker, unresolved targets, a re-failed
    transcription, and the exit-code contract.
- **v1.23.0** (9 July 2026) -- Extend the model-capability registry with the
    current GPT-5.6 family (sol, terra, luna, and the bare alias), GPT-5.5 Pro,
    the Claude 5 generation (Fable 5, Sonnet 5) plus Opus 4.8, and the newly
    GA Gemini 3.5 Flash and Gemini 3.1 Flash-Lite. Each entry mirrors the
    established per-provider profile: GPT-5.5 Pro is registered Responses-only
    with a 1.05M context and image_detail low/high/auto; the new Claude models
    carry 1M context, 128k output, adaptive-thinking-only reasoning (top_p and
    budget_tokens disabled), and their family names are added to the Anthropic
    provider's adaptive-thinking set so they emit an adaptive thinking block
    rather than a rejected budget_tokens block; the GA Gemini entries expose
    thinking_level support at 1,048,576 context. The GA Gemini 3.1 Flash-Lite
    entry is ordered after its preview sibling so preview ids still resolve to
    the preview profile, and the Gemini 2.5 family is annotated with its
    2026-10-16 shutdown date. All 1,454 unit and capability tests pass.

- **v1.22.0** (7 July 2026) -- Adopt shared token ledger 1.2.0 and fix two
    infra defects found in ChronoMiner's audit. The vendored
    `modules/infra/shared_ledger.py` is re-copied: `_merge` now coerces a
    non-numeric stored tool value to 0 and catches `ValueError`/`TypeError`
    alongside `OSError`, so a corrupt ledger degrades to standalone mode
    instead of crashing the call path (never-crash contract). The rate
    limiter's error backoff now imposes a real, bounded admission delay after
    429s: the multiplier previously scaled a zero wait when no window was
    saturated (a silent no-op), and the new penalty is a deadline from wait
    start rather than a perpetual floor, so admission always resumes. The
    token budget's atexit hook delegates to `flush()` in shared mode, so the
    last unsynced ledger delta is pushed at exit instead of being dropped
    (and the private state file is no longer written while the ledger is the
    active persistence). All 1,454 tests pass.

- **v1.21.0** (6 July 2026) -- Multi-agent bug hunt and fix batch across the
    batch, transcribe, and UI layers. Batch result extraction now reads the
    schema's actual `transcription` key via `extract_transcribed_text()`
    (previously a nonexistent `transcribed_text` lookup left raw JSON blobs
    as page text in Anthropic/Google batch outputs). Repair no longer
    hardwires OpenAI: sync repair resolves the configured provider's key
    through the factory, and batch repair refuses cleanly for non-OpenAI
    providers. Batch checking constructs the OpenAI client lazily so
    Anthropic/Google-only setups can finalize their batches. OpenAI batch
    fallback parsing attaches per-line `custom_id`s (no more duplicated
    error placeholders and scrambled order) and fires per batch instead of
    once globally. The streaming pipeline no longer finalizes output on a
    budget-exhausting pass, closing the window where an interruption left a
    truncated file that resume marked complete. ResumeChecker derives
    md/json output names exactly as the writer does, fixing perpetual
    re-transcription of long-stem documents. Folder-image virtual names now
    include the source extension so same-stem files no longer collide in
    JSONL dedup (legacy names still resume; repair resolves both forms).
    Recursive postprocessing mirrors the input tree instead of flattening
    it, auto-mode's resume pre-filter checks the actual Auto output root,
    the interactive wizard's API-key gate honors custom endpoints and
    auto-detects the provider from the model name, and the OpenRouter
    reasoning budget is clamped by the answer reserve. Register `gpt-5.5`
    in the model registry (Responses-native, 1.05M context, 128K output,
    original image detail) including the OpenRouter passthrough.
- **v1.20.0** (6 July 2026) -- Correctness hardening from the production
    budget-exhaustion incident plus a full-repository bug hunt. Fix the bug
    where daily-token-budget exhaustion mid-document left a truncated final
    txt that looked complete while the run exited 0: the mid-document wait
    is now reservation-aware (`would_block_next_page()`), with a fast-fail
    when the per-page estimate exceeds the entire daily budget and a
    post-countdown re-check; on give-up the partial output is withheld, the
    resume JSONL is protected from transient cleanup, and the item raises
    `BudgetExhaustedError` so the run exits 1 with truthful `--json`
    counters. A page failure on the exhausting pass now folds into the same
    withhold flow instead of bypassing it. Also fix: the image-folder path
    swallowing `RuntimeError` (silent exit 0 on broken folders); the sync
    repair loop's reservation-blind wait; auto mode dropping
    `output_format`, context paths, and other settings and checking
    item-level resume against the wrong directories; the `ResumeChecker`'s
    degenerate "." key for single-file inputs; EPUB/MOBI page ranges
    resolved against a `2**31` sentinel (OOM on open spans, silently empty
    output on `last:N`); Anthropic thinking configs failing every page with
    an unretryable 400 when temperature/top_p/top_k were set; auto mode
    gating LLM availability on `OPENAI_API_KEY` alone instead of the
    configured provider; `get_provider()` clobbering explicit CLI
    `--max-output-tokens`/temperature during provider auto-detection; and
    the postprocess CLI missing modern `{stem}.txt` outputs in directory
    scans while exiting 0 despite failures. Extensive regression tests
    accompany every fix.
- **v1.19.1** (5 July 2026) -- Harden retries and the test suite from a live
    failure hunt. Fix a retry gap in `modules/llm/providers/base.py`:
    SDK-wrapped connection failures (`openai.APIConnectionError` raised from
    `httpx.ConnectError`, and the Anthropic SDK's identical wrapping) escaped
    `_should_retry` because only the top-level exception type was inspected;
    a bounded cause-chain walk now classifies them as retryable, so a
    transient TCP failure no longer fails a page permanently on the first
    attempt (regression-tested). Make three unit tests hermetic against the
    machine's live `api_keys_config.yaml` provider remap via a shared
    `no_api_key_remap` fixture, and fix the live-API fixture's stale Google
    model id (`gemini-3-flash` to `gemini-3-flash-preview`), which 404'd on
    every call.
- **v1.19.0** (5 July 2026) -- Fix the daily token budget's reset boundary.
    Both the private per-tool tracker (`modules/infra/token_budget.py`) and
    the vendored shared cross-tool ledger (`modules/infra/shared_ledger.py`,
    bumped to 1.1.0) now roll the budget day over at 00:01 UTC -- one minute
    after OpenAI's 00:00 UTC free-tier reset -- instead of local midnight, so
    the tool never frees its budget before OpenAI's own counter has actually
    reset. The one-minute buffer is a deliberate safety margin against clock
    skew. `get_reset_time()` now returns a timezone-aware UTC datetime;
    user-facing wait messages show the local wall-clock time alongside an
    explicit "(00:01 UTC)" anchor for clarity. Updated docs and config
    comments accordingly. The shared ledger module was re-vendored (module and
    its test) in sync with ChronoMiner and AutoExcerpter to keep all three
    tools on one combined budget day. All tests pass.

- **v1.18.0** (3 July 2026) -- Honest exit codes and full `--json`
    coverage from a live cross-provider bug hunt. Propagate page-level
    transcription failures to the item status, the JSON summary, and the
    exit code: a run whose output contains any `[transcription error]`
    placeholder now exits 1 instead of reporting full success (partial
    output and the resume JSONL are still written first, so resume and
    `--retry-errors` keep working). Name the hash-suffixed output
    directory after the input file when `--input` is a single file
    (previously such runs wrote to a hidden `.-<hash>` directory).
    Classify LangChain `OutputParserException` as validation-retryable so
    flaky-JSON models are retried within the `validation_attempts`
    budget. Implement real one-line `--json` summaries on the four entry
    points where the flag was a documented no-op (`check_batches`,
    `cancel_batches`, `repair_transcriptions`,
    `postprocess_transcriptions`). Suppress the noisy upstream Pydantic
    serializer warnings from the OpenAI SDK with a narrowly scoped
    warnings filter.

- **v1.17.0** (3 July 2026) -- Optional shared cross-tool token budget.
    Add the vendored `modules/infra/shared_ledger.py` (locked delta merges
    into per-tool fields, atomic per-process temp writes, local-midnight
    rollover, degrade-to-standalone) and wire it into the daily token
    tracker behind the new opt-in `shared_token_budget` config block: when
    enabled, `daily_token_limit.daily_tokens` is enforced against the
    COMBINED usage of ChronoTranscriber, ChronoMiner, and AutoExcerpter via
    one ledger at `~/.chronopipeline/token_ledger.json`, with seed-once
    adoption of legacy same-day counts, delta syncs riding the debounced
    background writer, forced refreshes near the cap and while waiting at
    the limit, and per-tool breakdown in the usage stats. Default behavior
    (feature off) is unchanged. Verified live: concurrent tools share one
    limit with zero lost updates.

- **v1.16.0** (3 July 2026) -- Concurrency and token-budget hardening.
    Gate the synchronous repair path on the daily token budget with the same
    drain/wait/re-pass behavior as the main pipeline; move token-state
    persistence to a debounced background writer with per-process-unique
    temp files and race-tolerant retries (no more disk I/O or sleeps on the
    event loop); make the tenacity loop the single retry authority (SDK
    retries disabled, status-code-first classification, HTTP `Retry-After`
    honored, default 8 attempts with a 120 s cap); count Anthropic
    prompt-cache creation and read tokens at full weight in the daily budget
    and recover token usage from failed attempts; add a per-provider
    multi-window rate limiter with adaptive backoff
    (`modules/infra/rate_limit.py`, `concurrency.rate_limits`); run
    Tesseract OCR off the event loop; replace eager task creation with
    bounded lazy submission and make streaming failures cancel producer and
    workers cleanly; re-read `daily_token_limit.daily_tokens` during the
    wait-at-limit loop; implement real provider client teardown; document
    `image_processing.concurrency_limit` and fix stale module references in
    the example configs.

- **v1.15.0** (2 July 2026) -- Hardening release closing the silent-page-loss
    and batch-integrity defects found in a full production audit. OpenAI batch
    error files are now always parsed and reconciled against the submitted
    custom_id map, so failed pages surface as explicit `[transcription error]`
    placeholders instead of vanishing from final outputs; expired and cancelled
    batches are treated as terminal instead of polling forever; batch repair
    correlates results by request index rather than position. The Tesseract
    pipeline gains the absolute-page-order and regenerate-from-JSONL resume
    semantics the GPT streaming path received in v1.7.0, image folders sort
    naturally (`page_2` before `page_10`) via one shared key, and temp JSONLs
    carry a resume-format version that refuses incompatible pre-fix artifacts.
    Oversized batch submissions are split into provider-limited parts, and the
    silent full-price synchronous fallback is now opt-in via `--sync-fallback`.
    Content-quality validators no longer flag the schema's own `![Image: ...]`
    markers, run inside the retry loop, and default to the conservative example
    thresholds. All entry points adopt an agent-friendly CLI contract: exit
    codes 0/1/2/130, a `--json` run summary, `--dry-run`, `--interactive`/
    `--non-interactive` overrides, a non-TTY guard, and a `--retry-errors`
    resume mode. Token-budget state moves to a user-level directory
    (configurable via `general.state_dir`) with one-time legacy adoption; EXIF
    orientation, palette-PNG transparency, embedded DPI after downscaling, and
    EPUB spine ordering are fixed; JSON artifacts write `ensure_ascii=False`;
    the ruff backlog is cleared.

- **v1.14.0** (28 June 2026) -- Ship scrubbed `*.example.yaml` config templates
    with conservative OpenAI defaults and a real->example loader fallback, so a
    fresh clone runs with clear guidance instead of crashing on missing config.
    Each of the five config files now has a tracked `<name>.example.yaml` sibling
    in `config/`; the real `*.yaml` files remain gitignored. The loader tries the
    real file first, falls back to the example with a one-line INFO notice if it is
    absent, and raises a clear error only when neither file exists. The
    `api_keys_config` loader retains its non-raising behavior (returns `{}` when
    both files are absent). The `.gitignore` pattern is updated from `/config/` to
    `/config/*` plus `!/config/*.example.yaml` so examples are tracked while real
    configs stay private.

- **v1.13.0** (28 June 2026) -- Add optional `api_keys_config.yaml` for
    per-provider API-key environment-variable remapping. Each provider can be
    pointed at a custom env var name (for example `openai: OPENAI_API_KEY_2`) to
    swap keys between runs by editing one file; a missing file or omitted
    provider entry falls back to the existing default env var name, so behavior
    is unchanged for current setups. The remap is honored uniformly across the
    sync pipeline, the wizard validation gate, repair, diagnostics, and the
    batch backends, so it applies in batch mode too.

- **v1.12.0** (24 June 2026) -- The daily token limit is now enforced at the
    page level, not just between files. When the limit is enabled, the
    synchronous (GPT) streaming pipeline reserves a self-calibrating estimate
    of per-page token usage before each page, so concurrent workers cannot
    collectively overshoot; once the budget is exhausted mid-file it drains
    in-flight pages, waits for the daily reset, and re-streams the still-pending
    pages from the JSONL resume record. Configured concurrency and per-task
    delay are unchanged when budget is plentiful. Batch mode is now fully exempt
    from token limiting (it is pre-priced and submitted whole). Two optional
    `daily_token_limit` settings tune the estimate (`chunk_estimate_seed`,
    `estimate_smoothing`). All 1287 tests pass.

- **v1.11.0** (21 June 2026) -- Adopted mypy 2.x for static type checking and made
    `mypy .` runnable. Raised the dev pin to `mypy>=2.1`; fixed the config so the
    `__init__.py`-less `main/` no longer resolves twice (`explicit_package_bases`,
    `namespace_packages`, `mypy_path`, and an `exclude` scoping checks to source).
    Added one missing return annotation and three targeted `arg-type` ignores for
    the langchain `HumanMessage` content stub. The source type-checks clean under
    mypy 2.1.0 and all 1,279 tests pass.

- **v1.10.0** (21 June 2026) -- Adopted the google-genai 2.x SDK major.
    Raised the runtime pin from `google-genai>=1.73` to `google-genai>=2` and
    refreshed the lockfile (`google-genai` 1.73.1 -> 2.9.0;
    `langchain-google-genai` unchanged). The Google batch backend imports clean
    and all 1,279 tests pass. Live Google batch API calls are not exercised by
    the test suite; validate a real Google run before relying on it.

- **v1.9.0** (20 June 2026) -- Consolidated six within-module duplication clusters
    behind new private helpers, leaving every public interface and runtime behavior
    unchanged. In `modules/llm/providers/base.py` the three retry-config loaders
    and the content-quality config getter now share a single `_load_retry_config`
    helper. `main/cancel_batches.py` extracts the repeated batch id/status
    normalization into `_extract_batch_id_and_status`.
    `modules/batch/requests.py` folds the two identical submit-and-cleanup blocks
    into `_submit_and_cleanup_batch_file`. `modules/images/pipeline.py` shares its
    longest-side downscale logic via `_cap_longest_side`.
    `modules/batch/backends/google_backend.py` routes both JSONL and inline result
    branches through `_apply_json_content`. `modules/llm/transcriber.py`
    centralizes the common provider transcribe keyword arguments in
    `_transcribe_kwargs`. The empty confirmed dead-code list left nothing to remove.

- **v1.8.0** (20 June 2026) -- Refreshed dependencies under the conservative,
    majors-gated policy. Added `httpx>=0.28` as an explicit runtime dependency,
    since `modules/llm/providers/base.py` imports it directly for the connection
    and timeout exceptions in its retry logic while it was previously only
    transitive. Upgraded the LangChain stack (`langchain-core` to 1.4.8,
    `langchain-openai` to 1.3.2, `langchain-anthropic` to 1.4.6,
    `langchain-google-genai` to 4.2.5), the direct SDKs `openai` (2.43.0) and
    `anthropic` (0.111.0), plus `deskew` (1.6.1), `numpy` (2.4.6), and `lxml`
    (6.1.1) on the runtime side. In the dev and eval groups, raised `ruff`
    (0.15.18), `pytest` (9.1.1), `pytest-asyncio` (1.4.0), `coverage` (7.14.1),
    the type stubs for aiofiles and PyYAML, `pandas` (3.0.3), and `matplotlib`
    (3.11.0). Held two major bumps: `google-genai` stays on 1.73.1 (2.9.0
    withheld) and `mypy` stays on 1.20.2 (2.1.0 withheld), as each had no
    within-major release available. No dependencies were removed, since the deptry
    unused flags are all package-versus-module name-mapping false positives.

- **v1.7.0** (10 June 2026) -- Introduced a streaming in-memory image pipeline for
    all GPT paths (synchronous and batch, PDFs and image folders): pages are
    rendered, preprocessed, and base64-encoded fully in memory, and the
    `preprocessed_images/` folder is no longer written for GPT runs (Tesseract is
    unchanged), with peak memory being one raw page plus the payloads in flight.
    Page-level resume and page-range slicing are now applied before any rendering,
    so resuming a mostly-complete PDF no longer re-renders every page; virtual
    image names keep the historical `*_pre_processed.jpg` pattern so old partial
    JSONLs resume cleanly. Reproducibility provenance was added: each transcription
    record carries `image_provenance` (SHA-256 of the sent JPEG bytes, dimensions,
    byte size, effective DPI) plus `source_file`/`page_index`, and each run writes
    a `file_provenance` record (source SHA-256, PyMuPDF/Pillow versions,
    image-config snapshot). Repair gains an in-memory re-render fallback: when no
    preprocessed image exists on disk, failed pages are re-rendered from the
    recorded source PDF page or source image and repaired from base64 (sync and
    batch modes). Final output for streaming runs is regenerated from the complete
    JSONL so pages completed in earlier resumed runs are included; `order_index` is
    now the absolute page index, fixing page renumbering under page ranges and
    resume. `max_pixels_per_page` default was lowered from 150,000,000 to
    24,000,000, bounding the worst-case raw page to roughly 72 MB RGB while
    staying 2.3x above the 10.24 MP `original_max_pixels` send cap. The now-dead
    disk pipeline was removed: `PDFProcessor.process_images` / `extract_images`,
    `ImageProcessor.process_image` / `process_and_save_images` /
    `process_images_multiprocessing`; `keep_preprocessed_images` now affects only
    Tesseract folders.

- **v1.6.0** (30 May 2026) -- Correctness fixes from a full code review: the
    completion summary now reports real success/failure counts,
    `process_selected_items` returns a `ProcessingSummary`, failed items are no
    longer also counted as processed, and interactive mode stops hardcoding zero
    failures. Both PDF extraction paths now raise when the per-page failure rate
    exceeds the existing image threshold instead of silently returning a short,
    possibly page-misaligned image list. The interactive resume preview now passes
    `output_format` and the resolved output directories to `ResumeChecker` so skip
    counts are accurate for `md`/`json` output rather than always assuming `.txt`.
    The Anthropic batch backend now uses the capability registry
    (`detect_capabilities`) to decide whether to send `temperature`, replacing
    brittle model-name substring matching that drifted from the sync providers.
    Failures are no longer silently swallowed: provider `_invoke_llm` handlers log
    full tracebacks, the JSONL diagnostic-context builder logs instead of passing,
    the JPEG draft fast-path and Tesseract checks use narrowed exceptions, and the
    token-state retry backoff is honored instead of skipped inside a running event
    loop. `parse_indices` rejects negative/open-ended tokens with a clear message
    instead of a confusing range-parse error. Dead-code and duplication cleanup:
    removed the unused `ru_*` aliases and pass-through wrappers in batch repair,
    consolidated the three per-backend image-encoding helpers onto the shared
    `modules.images.encoding` functions, extracted the duplicated EPUB and MOBI
    text-normalization helper into `modules.documents._text`, and removed a dead
    `media_resolution` expression in the Google provider.

- **v1.5.0** (21 May 2026) -- Added configurable `user_instruction` and
    `context_image_instruction` keys under `transcription_model` in
    `model_config.yaml`; when set to an empty string the text block is omitted
    entirely from the user message, sending image-only input, which is required for
    models like `churro-3B` (Qwen2.5-VL fine-tune) that expect no accompanying
    text. Both sync and batch paths respect the new keys across all five providers
    (OpenAI, Anthropic, Google, OpenRouter, Custom) and all four batch backends.
    Fixed pre-existing test failures in `TestCacheTokenExtraction` where the mock
    provider's content quality validator received a MagicMock config instead of a
    dict, triggering false hallucination-loop detection on short test content.

- **v1.4.0** (20 May 2026) -- Added `--output-mode {hash,mirror}` CLI flag: mirror
    mode replicates the input directory hierarchy under the output root, preserving
    edition/page structure for downstream consumers. Fixed a hash collision in
    non-colocated output mode: the directory hash now incorporates the full
    relative path from the input root instead of just the leaf folder name,
    preventing overwrites when multiple editions share page numbers. The resume
    checker supports both mirror mode and relative-path-aware hash lookups.

- **v1.3.1** (19 May 2026) -- Dependency refresh from an environment-wide CVE audit:
    bumped `langchain-core` 1.3.2 -> 1.4.0 (RCE on deserialization); `langsmith`
    0.7.36 -> 0.8.5 (unsafe deserialization; full fix to 1.0.x deferred pending
    upstream constraint relaxation); `pillow` 12.1.1 -> 12.2.0 (FITS GZIP
    decompression bomb); `jupyterlab` 4.5.6 -> 4.5.7 and `notebook` 7.5.5 ->
    7.5.6 (one-click command execution chain); `jupyter-server` 2.17.0 -> 2.18.2
    (persistent cookie secret); `urllib3` 2.6.3 -> 2.7.0 (audit-surface
    consolidation); `deskew` downgraded 1.6.0 -> 1.5.3 as a side effect of
    relaxing the `pillow<12.2` peer constraint. Fixed
    `tests/integration/test_live_api.py` scripted-input drift introduced by
    v1.3.0: inserted one response for the new `configure_additional_context_image`
    prompt so the `GPT_PDF_RESPONSES` sequence matches the post-v1.3.0 workflow.

- **v1.3.0** (5 May 2026) -- Added context image support: a reference image (title
    page, table of contents, column headers) can now be included alongside each
    page image to improve transcription quality, using the same hierarchical
    resolution as text context (`{name}_transcr_context_image.{ext}` convention).
    Added a `--context-image` CLI flag to override the context image path and an
    interactive wizard prompt for context image selection. Supported on the OpenAI
    provider (sync and batch paths); other providers accept the parameter for
    interface compatibility.

- **v1.2.1** (5 May 2026) -- Fixed a circular import cycle between
    `modules.documents`, `modules.images`, `modules.ui`, and `modules.transcribe`
    that prevented startup: `WorkflowUI` is now lazily imported in
    `modules.ui.__init__` and the `AutoSelector` import in `config_builder.py` is
    deferred to function scope. Fixed test-induced directory pollution:
    `WorkflowManager` integration tests now provide `tmp_path`-based `file_paths`
    instead of relying on relative-path defaults that created `epubs_out`,
    `images_out`, `mobis_out`, `pdfs_out` in the project root.

- **v1.2.0** (4 May 2026) -- Applied ruff linter and formatter across entire
    codebase.

- **v1.1.0** (4 May 2026) -- Version bump consolidating post-baseline development.

- **v1.0.1** (25 April 2026) -- Migrated to `pyproject.toml` and updated
    dependencies; fixed test artifacts polluting the project root (initial pass).

- **v1.0.0** (25 April 2026) -- Repository baseline: squashed history into single
    commit.
