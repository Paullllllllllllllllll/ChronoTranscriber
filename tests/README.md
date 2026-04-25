# ChronoTranscriber Test Suite

Comprehensive test suite for the ChronoTranscriber document transcription application.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_cli_args.py     # CLI argument parsing
│   ├── test_postprocess.py  # Text post-processing
│   ├── test_text_processing.py  # Transcription text extraction
│   ├── test_auto_selector.py    # Auto mode file detection
│   ├── test_config_service.py   # Configuration management
│   ├── test_ui_prompts.py       # UI utilities
│   ├── test_user_config.py      # UserConfiguration dataclass
│   ├── test_schema_utils.py     # Schema discovery and loading
│   ├── test_model_capabilities.py  # Model capability detection
│   ├── test_batch_backends.py   # Batch processing backends
│   ├── test_jsonl_utils.py      # JSONL file utilities
│   └── test_token_tracker.py    # Token usage tracking
└── integration/             # Integration tests (may use real files)
    ├── test_workflow_integration.py  # Workflow manager tests
    └── test_api_integration.py       # API interaction tests
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov
```

### Run All Tests

```bash
# From project root
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=modules --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Run tests by marker
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests
pytest -m "not api"      # Skip tests requiring API keys
```

### Run Specific Test Files

```bash
# Single file
pytest tests/unit/test_postprocess.py -v

# Single test class
pytest tests/unit/test_postprocess.py::TestNormalizeSpacing -v

# Single test function
pytest tests/unit/test_postprocess.py::TestNormalizeSpacing::test_expands_tabs -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast unit tests with mocked dependencies |
| `@pytest.mark.integration` | Tests that may use real files |
| `@pytest.mark.slow` | Tests that take longer to run |
| `@pytest.mark.api` | Tests requiring API keys |
| `@pytest.mark.asyncio` | Async tests |
| `@pytest.mark.windows` | Windows-specific tests |

## Fixtures

Common fixtures available in all tests (from `conftest.py`):

### Configuration Fixtures
- `mock_paths_config` - Mock paths configuration dictionary
- `mock_model_config` - Mock model configuration dictionary
- `mock_concurrency_config` - Mock concurrency configuration dictionary
- `mock_image_processing_config` - Mock image processing configuration
- `mock_config_service` - Mock ConfigService with all configs

### File System Fixtures
- `temp_dir` - Temporary directory (auto-cleaned)
- `temp_input_dir` - Temporary input directory
- `temp_output_dir` - Temporary output directory
- `sample_image_folder` - Folder with placeholder images
- `sample_pdf_file` - Placeholder PDF file
- `sample_text_file` - Sample transcription text file
- `sample_jsonl_file` - Sample JSONL tracking file

### Mock API Fixtures
- `mock_openai_response` - Mock OpenAI API response
- `mock_openai_client` - Mock OpenAI client
- `mock_langchain_llm` - Mock LangChain LLM

### Environment Fixtures
- `mock_env_no_api_keys` - Environment with no API keys
- `mock_env_with_openai_key` - Environment with OpenAI key set

### User Configuration Fixtures
- `sample_user_config` - Basic UserConfiguration
- `sample_user_config_gpt` - UserConfiguration for GPT testing

## Writing New Tests

### Unit Test Template

```python
"""Unit tests for modules/example/module.py."""

import pytest
from modules.example.module import function_to_test


class TestFunctionName:
    """Tests for function_name."""
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic expected behavior."""
        result = function_to_test("input")
        assert result == "expected_output"
    
    @pytest.mark.unit
    def test_edge_case(self):
        """Test edge case handling."""
        result = function_to_test("")
        assert result == ""
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_to_test(None)
```

### Integration Test Template

```python
"""Integration tests for feature X."""

import pytest
from pathlib import Path


class TestFeatureIntegration:
    """Integration tests for feature X."""
    
    @pytest.mark.integration
    def test_full_workflow(self, temp_dir):
        """Test complete workflow from input to output."""
        # Setup
        input_file = temp_dir / "input.txt"
        input_file.write_text("test content")
        
        # Execute
        result = process_feature(input_file)
        
        # Verify
        assert result.success
        assert (temp_dir / "output.txt").exists()
```

### Async Test Template

```python
"""Async tests for API interactions."""

import pytest
from unittest.mock import AsyncMock, patch


class TestAsyncFeature:
    """Tests for async feature."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_operation(self, mock_langchain_llm):
        """Test async API operation."""
        with patch('modules.llm.transcriber.get_llm_provider') as mock:
            mock.return_value.get_llm.return_value = mock_langchain_llm
            
            result = await async_function()
            
            assert result is not None
```

## Test Coverage

Generate coverage reports:

```bash
# HTML report (opens in browser)
pytest --cov=modules --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=modules --cov-report=term-missing

# XML report (for CI)
pytest --cov=modules --cov-report=xml
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Run tests with XML output for CI
pytest --junitxml=test-results.xml --cov=modules --cov-report=xml

# Exit with error code on any failure
pytest --strict-markers -x
```

## Troubleshooting

### Tests Not Found
Ensure test files match the pattern `test_*.py` and functions match `test_*`.

### Import Errors
The project root should be in PYTHONPATH. The conftest.py adds it automatically.

### Async Test Issues
Ensure `pytest-asyncio` is installed and tests are marked with `@pytest.mark.asyncio`.

### Fixture Not Found
Check that fixtures are defined in `conftest.py` or the same test file.
