# ChronoTranscriber Test Suite

## Layout

```text
tests/
├── conftest.py            # hermetic harness and shared fixtures
├── recording_events.py    # run events recorded for assertions
├── test_hermetic.py       # tests of the harness itself
├── test_logging_setup.py  # logging is configured at run time, not on import
├── matrix_golden.py       # golden JSON of the request and image matrices
├── fixture_paths.py       # placeholder root of fixture paths
├── audio/                 # audio sources, chunking, Whisper and the audio API
├── batch/                 # batch submit, check, finalize, reconcile, cancel
│                          # and repair on the store, over fake batch APIs
├── cli/                   # the console script: help, contract, warnings
├── common/                # chronotranscriber/common/: hash check, SQLite,
│                          # store, capture and the batch backends
├── core/                  # plan and run, spec, settings, events, extension
│                          # seam, store runs and the package structure
├── images/                # page sources, Tesseract, preprocessing, image settings
├── llm/                   # model clients, options, capabilities and attempts
├── methods/               # the unit pipelines and every method on the store
├── output/                # item paths, resume classification and rendering
├── sources/               # PDF, EPUB, MOBI and auto-mode scanning
├── ui/                    # the wizard: steps, sessions, flows, parity with the CLI
└── characterization/      # end-to-end runs compared with golden files
    ├── runner.py          # run_cli(), the only entry into the tool
    ├── fakes*.py          # fake chat model, batch service, Tesseract, audio
    ├── seams.py           # replaces external objects wherever the tool binds them
    ├── inputs*.py         # generated PDFs, image folders, EPUB, MOBI, audio
    ├── golden.py          # snapshot normalization and golden comparison
    ├── test_*.py          # one file per flow (see below)
    ├── context/           # the default context of every characterization run
    └── golden/<group>/<case>/
```

Test files follow `test_<module>.py` naming; find the tests for a module
by its name. `llm/test_request_matrix.py` and
`images/test_image_presets_matrix.py` compare the requests and image settings
of a matrix of providers, models and options with their golden JSON
(`CT_UPDATE_GOLDEN=1` rewrites it).

## Hermetic Harness

`tests/conftest.py` isolates every test from the machine it runs on.
At session start it removes every environment variable whose name
contains `API_KEY`, points `HOME`, `USERPROFILE`, `APPDATA`,
`LOCALAPPDATA`, `HOMEDRIVE`, `HOMEPATH` and the `XDG_*` variables at a
session temp directory.

The autouse `_hermetic` fixture then gives each test its own copy of
that setup under `tmp_path`:

- the home variables and `tempfile.tempdir`;
- the default settings file (`config/settings.yaml`), so a real settings
  file in the repository is never read;
- the default logs directory;
- empty shared rate limiters.

The same fixture blocks connections to any non-loopback address,
refuses to start a ChronoTranscriber entry point (the `chronotranscriber`
program or package, run with `-m` or as a script) as a subprocess, and
removes the log handlers a test installed. A write guard records
every file a test opens for writing; a write outside the pytest base
temp directory fails the test. The `write_guard` fixture lets a test
consume an expected write, as the golden update does.

## Running Tests

Run from the repository root. Pass `--basetemp` when two runs share
the worktree, so their temp directories do not collide.

```powershell
# Full suite
uv run python -m pytest

# One directory, file or test
uv run python -m pytest tests/methods/
uv run python -m pytest tests/output/test_render.py -v
uv run python -m pytest tests/characterization/test_run_control.py -q `
    -p no:cacheprovider --basetemp "$env:TEMP\ct-run-control"

# Coverage, with the data file kept outside the repository
$env:COVERAGE_FILE = "$env:TEMP\chronotranscriber.coverage"
uv run python -m pytest -p no:cacheprovider --basetemp "$env:TEMP\ct-cov" `
    --cov=chronotranscriber --cov-report=term
```

Test dependencies are part of the dev extra: `uv sync --extra dev`. No test
calls a real provider; the harness removes every API key from the
environment and blocks non-loopback connections.

## Characterization Tests

The tests in `tests/characterization/` run the real CLI code paths in
process and compare everything a run produces with golden files.

- `run_cli(command, *, llm, root, settings=None, env=None, **options)` in
  `runner.py` is the only function that knows how the tool is invoked.
  `command` is a subcommand (`transcribe`, `repair`, `batch check`,
  `batch cancel`) and `options` are option names of its table in
  `chronotranscriber/spec.py`; `build_argv` turns them into arguments, and
  an option the command lacks raises. `settings` is written as the run's
  `settings.yaml`; `--json` and `--settings` are always passed. The returned
  `RunResult` holds the exit code, the JSON summary line, the console
  output (`stderr`), the files the run created and the requests the fake
  chat model received; `RunResult.output(item)` gives an item's output path.
- The autouse `default_context` fixture makes `characterization/context/`
  the project context folder, which holds the default context, so the
  goldens do not depend on the repository's `context/`; the
  `project_context` fixture sets another folder.
- The autouse `fake_llm` fixture replaces the provider classes with a
  scripted fake chat model, so the real provider, retry and parsing code
  runs without a network. Pages are matched to scripts by a gray-level
  marker drawn into each generated page. Batch backends, Tesseract,
  Whisper and the audio API run through the other fakes.
- Each case runs twice in fresh roots, and both snapshots must match
  before the second is compared with `golden/<group>/<case>/`. A
  snapshot holds every created file (paths, datetimes, durations,
  versions and hashes normalized to placeholders; `chronotranscriber.sqlite`
  as a text dump of its rows), `summary.json` (exit code and JSON line) and
  `requests.json` (client constructions and request digests).
- To accept a deliberate change, rerun the affected test file with
  `CT_UPDATE_GOLDEN=1`, which rewrites its golden cases, and review the
  diff.

| File | Flow |
| ------ | ------ |
| `test_smoke.py` | one case per command; the options the cases pass parse |
| `test_sources.py` | PDF, image folder, EPUB, MOBI, auto mode; local methods |
| `test_llm_formats.py` | LLM method: providers, custom endpoint, response formats, context |
| `test_response_formats.py` | `--response-format` json and text on every provider |
| `test_batch.py` | batch submit and finalize through a fake batch service |
| `test_sqlite_batch.py` | the store of batch-submitted items through submit, check and repair |
| `test_cancel.py` | `batch cancel` by batch ID and of every unfinished batch |
| `test_audio.py` | audio API and Whisper, chunking |
| `test_audio_overlap.py` | chunk overlap of a recording above the request limit |
| `test_repair.py` | repair and `--retry-errors` |
| `test_run_control.py` | resume, dry run, exit codes, selection, layouts, pages |
| `test_resume_sources.py` | resume of image folder, EPUB, MOBI, audio, auto mode |
| `test_sqlite_output.py` | the SQLite database across reruns, items and lost rows |
| `test_working_files.py` | `--keep-working-files` for Tesseract images and audio chunks |
| `test_golden_mode.py` | the rendered-only golden comparison of `CT_GOLDEN_RENDERED_ONLY=1` |

## Markers

Registered in `pyproject.toml`; `--strict-markers` is enforced and no marker
is deselected by default.

| Marker | Description |
| -------- | ------------- |
| `unit` | Fast unit tests with mocked dependencies |
| `integration` | Tests that use real files; all characterization tests |

Async tests need no marker: `asyncio_mode = "auto"` is set.

## Fixtures

Shared fixtures live in `tests/conftest.py`: `write_guard` from the
harness, the `make_spec`, `make_settings` and `make_job` builders, the
mock audio configuration, temp directories and sample files, mock API
clients and batch handles, and the environment fixtures
`mock_env_no_api_keys` and `mock_env_with_openai_key`.
The characterization fixtures (`default_context`, `fake_llm`, `fake_audio`,
`golden`, `case_root`, `project_context`) live in
`tests/characterization/conftest.py`. Consult both files for the
authoritative list.

## Writing New Tests

- Name files `test_<module>.py`, functions `test_*`; mark each test
  `unit` or `integration` (markers are strict).
- Write only under `tmp_path`; the write guard fails any other write.
- Mock external services; no test calls a real LLM endpoint.
- Run the tool in characterization tests only through `run_cli`.
