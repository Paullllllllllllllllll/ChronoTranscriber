# ChronoTranscriber Test Suite

Test suite for the ChronoTranscriber document transcription application:
roughly 1,700 tests across 76 files.

## Layout

```
tests/
├── conftest.py       # Shared fixtures and pytest configuration
├── unit/             # 73 files: fast, isolated tests covering all
│                     # modules/ packages (providers, batch backends,
│                     # config, documents, images, infra, postprocess,
│                     # transcribe pipeline, UI) and the main/ CLIs
└── integration/      # 3 files: workflow, API interaction, and live
    │                 # API smoke tests (marker: api)
    └── fixtures/     # Sample assets (PDFs)
```

Test files follow `test_<module>.py` naming; find the tests for a module
by its name rather than via a maintained inventory.

## Running Tests

```bash
# Full suite (api-marked tests are deselected by default)
uv run python -m pytest

# Unit or integration only
uv run python -m pytest tests/unit/
uv run python -m pytest tests/integration/

# Single file / class / test
uv run python -m pytest tests/unit/test_postprocess.py -v
uv run python -m pytest tests/unit/test_postprocess.py::TestNormalizeSpacing -v

# Live API smoke tests (require real API keys; fire real LLM calls)
uv run python -m pytest -m api

# Coverage
uv run python -m pytest --cov=modules --cov-report=term-missing
```

Test dependencies are part of the dev extra: `uv sync --extra dev`.

## Markers

Registered in `pyproject.toml` (`--strict-markers` is enforced;
`addopts` applies `-m 'not api'` so live-API tests never run by
default):

| Marker | Description |
|--------|-------------|
| `unit` | Fast unit tests with mocked dependencies |
| `integration` | Tests that may use real files |
| `slow` | Long-running tests |
| `api` | Live API tests requiring real keys (opt-in via `-m api`) |
| `windows` | Windows-specific tests |

Async tests need no marker: `asyncio_mode = "auto"` is set.

## Fixtures

Shared fixtures live in `conftest.py`, grouped as:

- **Configuration** -- `mock_paths_config`, `mock_model_config`,
  `mock_concurrency_config`, `mock_image_processing_config`,
  `mock_config_service`
- **Filesystem** -- `temp_dir`, `temp_input_dir`, `temp_output_dir`,
  `sample_image_folder`, `sample_pdf_file`, `sample_text_file`,
  `sample_jsonl_file`
- **Mock API** -- `mock_openai_response`, `mock_openai_client`,
  `mock_langchain_llm`, `mock_batch_handle`, `mock_batch_status`
- **Environment** -- `mock_env_no_api_keys`, `mock_env_with_openai_key`,
  `no_api_key_remap` (isolates tests from a live `api_keys_config.yaml`)
- **User configuration** -- `sample_user_config`,
  `sample_user_config_gpt`, `sample_transcription_schema`

Consult `conftest.py` for the authoritative list and signatures.

## Writing New Tests

- Name files `test_<module>.py`, functions `test_*`; mark each test
  `unit` or `integration` (markers are strict).
- Mock external services in unit tests; anything that calls a real LLM
  endpoint must carry the `api` marker.
- Use the `tmp_path`/`temp_dir` fixtures instead of writing into the
  project tree.
