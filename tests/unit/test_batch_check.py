from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modules.operations.batch.check as batch_check


@pytest.mark.unit
def test_process_all_batches_skips_non_batched_jsonl(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    # Ensure the batch package __init__ is imported (coverage for modules/operations/batch/__init__.py)
    import modules.operations.batch as _batch_pkg  # noqa: F401

    root = temp_dir / "root"
    root.mkdir()

    # A JSONL with no batch markers should be treated as non-batched and skipped.
    (root / "not_batched.jsonl").write_text(json.dumps({"hello": "world"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(batch_check, "print_info", MagicMock())
    monkeypatch.setattr(batch_check, "print_warning", MagicMock())
    monkeypatch.setattr(batch_check, "print_error", MagicMock())
    monkeypatch.setattr(batch_check, "print_success", MagicMock())

    monkeypatch.setattr(batch_check, "display_batch_summary", MagicMock())
    monkeypatch.setattr(batch_check, "list_all_batches", MagicMock(return_value=[]))

    process_non_openai = MagicMock()
    monkeypatch.setattr(batch_check, "_process_non_openai_batch", process_non_openai)

    client = MagicMock()

    batch_check.process_all_batches(
        root_folder=root,
        processing_settings={"retain_temporary_jsonl": True},
        client=client,
        postprocessing_config=None,
    )

    process_non_openai.assert_not_called()


@pytest.mark.unit
def test_process_all_batches_routes_non_openai_provider(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    root = temp_dir / "root"
    root.mkdir()

    # Minimal batched JSONL with provider set to a non-OpenAI provider.
    lines = [
        {"batch_session": {"status": "completed", "provider": "anthropic"}},
        {"batch_tracking": {"batch_id": "batch_123", "provider": "anthropic"}},
    ]
    (root / "job.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(batch_check, "print_info", MagicMock())
    monkeypatch.setattr(batch_check, "print_warning", MagicMock())
    monkeypatch.setattr(batch_check, "print_error", MagicMock())
    monkeypatch.setattr(batch_check, "print_success", MagicMock())

    monkeypatch.setattr(batch_check, "display_batch_summary", MagicMock())
    monkeypatch.setattr(batch_check, "list_all_batches", MagicMock(return_value=[]))

    monkeypatch.setattr(batch_check, "supports_batch", MagicMock(return_value=True))

    process_non_openai = MagicMock()
    monkeypatch.setattr(batch_check, "_process_non_openai_batch", process_non_openai)

    client = MagicMock()

    batch_check.process_all_batches(
        root_folder=root,
        processing_settings={"retain_temporary_jsonl": True},
        client=client,
        postprocessing_config=None,
    )

    process_non_openai.assert_called_once()
    kwargs = process_non_openai.call_args.kwargs
    assert kwargs["batch_provider"] == "anthropic"
    assert kwargs["batch_ids"] == {"batch_123"}
