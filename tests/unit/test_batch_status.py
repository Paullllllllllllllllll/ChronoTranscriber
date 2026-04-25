"""Unit tests for modules.batch.status.

Exercises the JSONL metadata parser, the debug-artifact recovery fallback,
the ``load_config`` wrapper, and the ``diagnose_api_issues`` diagnostic
printer. The OpenAI SDK is never contacted: the config service and the
``OpenAI`` client are stubbed via ``monkeypatch``.

These tests deliberately duplicate only the *status*-specific surface; tests
for the higher-level orchestration live in ``test_batch_check.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modules.batch.status as batch_status


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:

    @pytest.mark.unit
    def test_returns_three_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = MagicMock()
        service.get_paths_config.return_value = {
            "general": {"retain_temporary_jsonl": True},
            "file_paths": {},
        }
        service.get_image_processing_config.return_value = {
            "postprocessing": {"enabled": True},
        }
        monkeypatch.setattr(batch_status, "get_config_service", lambda: service)
        monkeypatch.setattr(batch_status, "collect_scan_directories", lambda pc: [])

        result = batch_status.load_config()
        assert isinstance(result, tuple)
        assert len(result) == 3

        scan_dirs, settings, pp_config = result
        assert scan_dirs == []
        assert settings == {"retain_temporary_jsonl": True}
        assert pp_config == {"enabled": True}

    @pytest.mark.unit
    def test_forwards_paths_config_to_collector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths_cfg = {
            "general": {"retain_temporary_jsonl": False},
            "file_paths": {"input": "/some/path"},
        }
        service = MagicMock()
        service.get_paths_config.return_value = paths_cfg
        service.get_image_processing_config.return_value = {"postprocessing": {}}

        captured: dict = {}

        def fake_collect(pc: dict) -> list:
            captured["pc"] = pc
            return [Path("/fake")]

        monkeypatch.setattr(batch_status, "get_config_service", lambda: service)
        monkeypatch.setattr(batch_status, "collect_scan_directories", fake_collect)

        scan_dirs, settings, pp_config = batch_status.load_config()
        assert captured["pc"] is paths_cfg
        assert scan_dirs == [Path("/fake")]
        assert settings == {"retain_temporary_jsonl": False}
        assert pp_config == {}


# ---------------------------------------------------------------------------
# _parse_temp_file_metadata
# ---------------------------------------------------------------------------

class TestParseTempFileMetadata:

    @pytest.mark.unit
    def test_batch_session_record_is_parsed(self, tmp_path: Path) -> None:
        """Only the batch_session record contributes status/provider info."""
        lines = [
            json.dumps({"batch_session": {"status": "in_progress", "provider": "openai"}}),
            json.dumps({"image_metadata": {"custom_id": "req-1"}}),
            json.dumps({"image_metadata": {"custom_id": "req-2"}}),
        ]
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert meta["has_batch_session"] is True
        assert meta["batch_provider"] == "openai"
        assert "in_progress" in meta["batch_session_statuses"]
        assert meta["image_metadata_count"] == 2
        assert meta["batch_ids"] == set()

    @pytest.mark.unit
    def test_batch_tracking_records_extract_ids_and_provider(
        self, tmp_path: Path
    ) -> None:
        lines = [
            json.dumps(
                {"batch_tracking": {"batch_id": "bid_1", "provider": "google"}}
            ),
            json.dumps(
                {"batch_tracking": {"batch_id": "bid_2", "provider": "google"}}
            ),
            json.dumps({"batch_request": {"model": "gemini"}}),
        ]
        jsonl = tmp_path / "track.jsonl"
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert meta["batch_ids"] == {"bid_1", "bid_2"}
        assert meta["batch_provider"] == "google"
        assert meta["has_batch_request"] is True
        assert meta["batch_request_count"] == 1

    @pytest.mark.unit
    def test_empty_file_yields_defaults(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("", encoding="utf-8")

        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert meta["batch_ids"] == set()
        assert meta["batch_provider"] == "openai"  # default fallback
        assert meta["has_batch_session"] is False
        assert meta["has_batch_request"] is False
        assert meta["has_batch_metadata"] is False
        assert meta["image_metadata_count"] == 0
        assert meta["batch_request_count"] == 0

    @pytest.mark.unit
    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        lines = [
            "not valid json",
            json.dumps({"batch_tracking": {"batch_id": "good_id"}}),
            "{truncated",
        ]
        jsonl = tmp_path / "mixed.jsonl"
        jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert meta["batch_ids"] == {"good_id"}
        assert len(meta["batch_tracking_records"]) == 1


# ---------------------------------------------------------------------------
# _recover_batch_ids
# ---------------------------------------------------------------------------

class TestRecoverBatchIds:

    @pytest.mark.unit
    def test_returns_existing_ids_when_nonempty(self, tmp_path: Path) -> None:
        """If batch_ids is already non-empty, recovery is a no-op."""
        jsonl = tmp_path / "job_transcription.jsonl"
        jsonl.write_text("", encoding="utf-8")
        existing = {"batch_already_known"}

        out = batch_status._recover_batch_ids(
            temp_file=jsonl,
            identifier="job",
            batch_ids=existing,
            processing_settings={},
        )
        assert out == {"batch_already_known"}

    @pytest.mark.unit
    def test_returns_empty_when_no_debug_artifact(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job_transcription.jsonl"
        jsonl.write_text("", encoding="utf-8")

        out = batch_status._recover_batch_ids(
            temp_file=jsonl,
            identifier="job",
            batch_ids=set(),
            processing_settings={},
        )
        assert out == set()

    @pytest.mark.unit
    def test_recovers_ids_from_debug_artifact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        jsonl = tmp_path / "job_transcription.jsonl"
        jsonl.write_text("", encoding="utf-8")

        # _recover_batch_ids looks for <identifier>_batch_submission_debug.json
        # next to the temp file.  The identifier passed in is "job".
        debug = tmp_path / "job_batch_submission_debug.json"
        debug.write_text(
            json.dumps({"batch_ids": ["recovered_1", "recovered_2"]}),
            encoding="utf-8",
        )

        # Silence the info printer so the test output stays clean
        monkeypatch.setattr(batch_status, "print_info", MagicMock())

        out = batch_status._recover_batch_ids(
            temp_file=jsonl,
            identifier="job",
            batch_ids=set(),
            processing_settings={"persist_recovered_batch_ids": False},
        )
        assert out == {"recovered_1", "recovered_2"}


# ---------------------------------------------------------------------------
# diagnose_api_issues
# ---------------------------------------------------------------------------

class TestDiagnoseAPIIssues:

    @pytest.mark.unit
    def test_runs_without_raising_when_no_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no OPENAI_API_KEY and a failing client, no exception propagates."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Make OpenAI() raise so both list operations hit the error branch
        class BoomClient:
            def __init__(self, *a, **kw) -> None:
                raise RuntimeError("no key")

        monkeypatch.setattr(batch_status, "OpenAI", BoomClient)
        monkeypatch.setattr(batch_status, "print_info", MagicMock())
        monkeypatch.setattr(batch_status, "print_error", MagicMock())

        # Should neither raise nor return anything other than None
        assert batch_status.diagnose_api_issues() is None

    @pytest.mark.unit
    def test_runs_with_key_and_working_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        client = MagicMock()
        client.models.list.return_value = MagicMock(
            data=[MagicMock(), MagicMock()]
        )
        client.batches.list.return_value = MagicMock()
        monkeypatch.setattr(batch_status, "OpenAI", lambda: client)
        monkeypatch.setattr(batch_status, "print_info", MagicMock())
        monkeypatch.setattr(batch_status, "print_error", MagicMock())

        assert batch_status.diagnose_api_issues() is None
