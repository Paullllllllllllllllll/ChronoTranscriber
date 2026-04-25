from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modules.batch.check as batch_check
import modules.batch.status as batch_status


@pytest.mark.unit
def test_process_all_batches_skips_non_batched_jsonl(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    # Ensure the batch package __init__ is imported (coverage for modules/operations/batch/__init__.py)
    import modules.batch as _batch_pkg  # noqa: F401

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


class TestStatusSubmodule:
    """Tests for functions from the status submodule."""

    @pytest.mark.unit
    def test_load_config_returns_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_config returns (scan_dirs, processing_settings, postprocessing_config)."""
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = {
            "general": {"retain_temporary_jsonl": True},
            "file_paths": {},
        }
        mock_service.get_image_processing_config.return_value = {
            "postprocessing": {"enabled": True}
        }
        monkeypatch.setattr(batch_status, "get_config_service", lambda: mock_service)
        monkeypatch.setattr(batch_status, "collect_scan_directories", lambda pc: [])

        result = batch_check.load_config()
        assert isinstance(result, tuple)
        assert len(result) == 3
        scan_dirs, settings, pp_config = result
        assert isinstance(scan_dirs, list)
        assert isinstance(settings, dict)
        assert isinstance(pp_config, dict)

    @pytest.mark.unit
    def test_load_config_passes_paths_config_to_collector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_config forwards the full paths_config dict to collect_scan_directories."""
        paths_cfg = {
            "general": {"retain_temporary_jsonl": False},
            "file_paths": {"input": "/some/path"},
        }
        mock_service = MagicMock()
        mock_service.get_paths_config.return_value = paths_cfg
        mock_service.get_image_processing_config.return_value = {"postprocessing": {}}

        captured = {}

        def fake_collect(pc: dict) -> list:
            captured["pc"] = pc
            return [Path("/fake")]

        monkeypatch.setattr(batch_status, "get_config_service", lambda: mock_service)
        monkeypatch.setattr(batch_status, "collect_scan_directories", fake_collect)

        scan_dirs, settings, pp_config = batch_check.load_config()
        assert captured["pc"] is paths_cfg
        assert scan_dirs == [Path("/fake")]
        assert settings == {"retain_temporary_jsonl": False}
        assert pp_config == {}

    @pytest.mark.unit
    def test_parse_temp_file_metadata_empty_file(self, temp_dir: Path) -> None:
        """_parse_temp_file_metadata returns default metadata for an empty file."""
        empty_file = temp_dir / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")

        meta = batch_check._parse_temp_file_metadata(empty_file)
        assert meta["batch_ids"] == set()
        assert meta["batch_provider"] == "openai"  # default
        assert meta["has_batch_session"] is False
        assert meta["has_batch_request"] is False
        assert meta["has_batch_metadata"] is False
        assert meta["image_metadata_count"] == 0
        assert meta["batch_request_count"] == 0

    @pytest.mark.unit
    def test_parse_temp_file_metadata_with_tracking(self, temp_dir: Path) -> None:
        """_parse_temp_file_metadata extracts batch_ids and provider from tracking records."""
        lines = [
            json.dumps({"batch_session": {"status": "completed", "provider": "google"}}),
            json.dumps({"batch_tracking": {"batch_id": "bid_1", "provider": "google"}}),
            json.dumps({"batch_tracking": {"batch_id": "bid_2", "provider": "google"}}),
            json.dumps({"image_metadata": {"custom_id": "req-1"}}),
            json.dumps({"batch_request": {"model": "gemini-2.0"}}),
        ]
        temp_file = temp_dir / "tracked.jsonl"
        temp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = batch_check._parse_temp_file_metadata(temp_file)
        assert meta["batch_ids"] == {"bid_1", "bid_2"}
        assert meta["batch_provider"] == "google"
        assert meta["has_batch_session"] is True
        assert meta["has_batch_request"] is True
        assert meta["has_batch_metadata"] is True
        assert meta["image_metadata_count"] == 1
        assert meta["batch_request_count"] == 1
        assert "completed" in meta["batch_session_statuses"]

    @pytest.mark.unit
    def test_parse_temp_file_metadata_skips_malformed_json(self, temp_dir: Path) -> None:
        """_parse_temp_file_metadata silently skips lines that are not valid JSON."""
        lines = [
            "not valid json at all",
            json.dumps({"batch_tracking": {"batch_id": "good_id"}}),
            "{truncated",
        ]
        temp_file = temp_dir / "mixed.jsonl"
        temp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = batch_check._parse_temp_file_metadata(temp_file)
        assert meta["batch_ids"] == {"good_id"}
        assert len(meta["batch_tracking_records"]) == 1
