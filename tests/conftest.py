"""Pytest configuration and shared fixtures for ChronoTranscriber tests.

This module provides reusable fixtures for:
- Configuration management
- Temporary file/directory creation
- Mock API clients
- Sample test data
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_paths_config() -> Dict[str, Any]:
    """Provide a mock paths configuration dictionary."""
    return {
        "general": {
            "interactive_mode": False,
            "retain_temporary_jsonl": True,
            "input_paths_is_output_path": False,
            "logs_dir": "logs",
            "keep_preprocessed_images": True,
            "auto_mode_pdf_use_ocr_for_scanned": True,
            "auto_mode_pdf_use_ocr_for_searchable": False,
            "auto_mode_pdf_ocr_method": "tesseract",
            "auto_mode_image_ocr_method": "tesseract",
        },
        "file_paths": {
            "PDFs": {"input": "pdfs_in", "output": "pdfs_out"},
            "Images": {"input": "images_in", "output": "images_out"},
            "EPUBs": {"input": "epubs_in", "output": "epubs_out"},
            "MOBIs": {"input": "mobis_in", "output": "mobis_out"},
            "Auto": {"input": "auto_in", "output": "auto_out"},
        },
    }


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Provide a mock model configuration dictionary."""
    return {
        "transcription_model": {
            "provider": "openai",
            "name": "gpt-4o",
            "max_output_tokens": 4096,
            "temperature": 0.1,
        },
        "reasoning": {
            "effort": "medium",
        },
    }


@pytest.fixture
def mock_concurrency_config() -> Dict[str, Any]:
    """Provide a mock concurrency configuration dictionary."""
    return {
        "concurrency": {
            "transcription": {
                "concurrency_limit": 10,
                "delay_between_tasks": 0.1,
                "service_tier": "auto",
            },
            "retry": {
                "max_attempts": 3,
                "wait_min_seconds": 1,
                "wait_max_seconds": 10,
            },
        },
    }


@pytest.fixture
def mock_image_processing_config() -> Dict[str, Any]:
    """Provide a mock image processing configuration dictionary."""
    return {
        "tesseract_image_processing": {
            "ocr": {
                "tesseract_config": "--oem 3 --psm 6",
                "lang": "eng",
            },
            "preprocessing": {
                "resize_factor": 2.0,
                "denoise": True,
            },
        },
        "postprocessing": {
            "enabled": False,
            "merge_hyphenation": False,
            "collapse_internal_spaces": True,
            "max_blank_lines": 2,
            "tab_size": 4,
            "wrap_lines": False,
            "auto_wrap": False,
            "wrap_width": None,
        },
    }


@pytest.fixture
def mock_config_service(
    mock_paths_config,
    mock_model_config,
    mock_concurrency_config,
    mock_image_processing_config,
):
    """Provide a mock ConfigService with all configurations."""
    mock_service = MagicMock()
    mock_service.get_paths_config.return_value = mock_paths_config
    mock_service.get_model_config.return_value = mock_model_config
    mock_service.get_concurrency_config.return_value = mock_concurrency_config
    mock_service.get_image_processing_config.return_value = mock_image_processing_config
    return mock_service


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after test."""
    temp_path = Path(tempfile.mkdtemp(prefix="chronotest_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_input_dir(temp_dir: Path) -> Path:
    """Create a temporary input directory structure."""
    input_dir = temp_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory structure."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_image_folder(temp_input_dir: Path) -> Path:
    """Create a sample image folder with placeholder images."""
    img_folder = temp_input_dir / "test_images"
    img_folder.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder image files (empty files with image extensions)
    for i in range(3):
        (img_folder / f"page_{i+1:03d}.png").write_bytes(b"")
    
    return img_folder


@pytest.fixture
def sample_pdf_file(temp_input_dir: Path) -> Path:
    """Create a placeholder PDF file for testing."""
    pdf_path = temp_input_dir / "test_document.pdf"
    # Create a minimal PDF-like file (not valid but useful for path testing)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return pdf_path


@pytest.fixture
def sample_text_file(temp_input_dir: Path) -> Path:
    """Create a sample text file for testing."""
    txt_path = temp_input_dir / "sample_transcription.txt"
    txt_path.write_text(
        "Page 1:\nThis is sample transcription text.\n\n"
        "Page 2:\n[transcription error: page_002.png]\n\n"
        "Page 3:\n[No transcribable text]\n",
        encoding="utf-8"
    )
    return txt_path


@pytest.fixture
def sample_jsonl_file(temp_input_dir: Path) -> Path:
    """Create a sample JSONL file for testing."""
    jsonl_path = temp_input_dir / "batch_results.jsonl"
    records = [
        {"image_name": "page_001.png", "text_chunk": "First page text", "order_index": 0},
        {"image_name": "page_002.png", "text_chunk": "Second page text", "order_index": 1},
        {"batch_tracking": {"batch_id": "batch_test123", "provider": "openai"}},
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return jsonl_path


# =============================================================================
# Mock API Client Fixtures
# =============================================================================

@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Provide a mock OpenAI API response."""
    return {
        "id": "resp_test123",
        "object": "response",
        "created": 1234567890,
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "transcription": "This is the transcribed text.",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        })
                    }
                ]
            }
        ],
        "output_text": json.dumps({
            "transcription": "This is the transcribed text.",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }),
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    }


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Provide a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_openai_response
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "transcription": "Test transcription",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        })))]
    )
    return mock_client


@pytest.fixture
def mock_langchain_llm():
    """Provide a mock LangChain LLM."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "transcription": "LangChain transcribed text",
        "no_transcribable_text": False,
        "transcription_not_possible": False,
    })
    mock_response.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    mock_llm.invoke = MagicMock(return_value=mock_response)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


# =============================================================================
# User Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_user_config():
    """Create a sample UserConfiguration for testing."""
    from modules.ui.core import UserConfiguration
    
    config = UserConfiguration()
    config.processing_type = "images"
    config.transcription_method = "tesseract"
    config.use_batch_processing = False
    config.selected_items = []
    config.process_all = False
    return config


@pytest.fixture
def sample_user_config_gpt(sample_user_config):
    """Create a sample UserConfiguration for GPT testing."""
    sample_user_config.transcription_method = "gpt"
    sample_user_config.selected_schema_name = "markdown_transcription_schema"
    sample_user_config.selected_schema_path = PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json"
    return sample_user_config


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def mock_env_no_api_keys():
    """Mock environment with no API keys set."""
    with patch.dict(os.environ, {}, clear=True):
        # Explicitly remove API keys if present
        env_copy = os.environ.copy()
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]:
            env_copy.pop(key, None)
        with patch.dict(os.environ, env_copy, clear=True):
            yield


@pytest.fixture
def mock_env_with_openai_key():
    """Mock environment with OpenAI API key set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}):
        yield


# =============================================================================
# Schema Fixtures
# =============================================================================

@pytest.fixture
def sample_transcription_schema() -> Dict[str, Any]:
    """Provide a sample transcription schema."""
    return {
        "type": "object",
        "properties": {
            "transcription": {
                "type": "string",
                "description": "The transcribed text from the image"
            },
            "no_transcribable_text": {
                "type": "boolean",
                "description": "True if no text could be found"
            },
            "transcription_not_possible": {
                "type": "boolean",
                "description": "True if transcription was not possible"
            },
        },
        "required": ["transcription", "no_transcribable_text", "transcription_not_possible"],
        "additionalProperties": False,
    }


# =============================================================================
# Batch Processing Fixtures
# =============================================================================

@pytest.fixture
def mock_batch_handle():
    """Provide a mock batch handle."""
    from modules.llm.batch.backends.base import BatchHandle
    return BatchHandle(
        batch_id="batch_test_12345",
        provider="openai",
        metadata={"submitted_at": "2024-01-01T00:00:00Z"},
    )


@pytest.fixture
def mock_batch_status():
    """Provide a mock batch status."""
    from modules.llm.batch.backends.base import BatchStatus, BatchStatusInfo
    return BatchStatusInfo(
        status=BatchStatus.COMPLETED,
        total_requests=10,
        completed_requests=10,
        failed_requests=0,
    )


# =============================================================================
# Async Test Helpers
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "api: Tests requiring API keys")
    config.addinivalue_line("markers", "windows: Windows-specific tests")
