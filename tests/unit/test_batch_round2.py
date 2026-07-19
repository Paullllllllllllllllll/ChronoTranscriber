"""Round-2 regression tests for the batch layer.

Covers duplicate custom_id dedupe, post-finalize resubmission metadata,
debug-artifact recovery with provider/metadata, the Google inline request
shape, the manager's redundant-batch-submission guard, and repair merging
into the temp JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import modules.batch.status as batch_status
from modules.batch.backends import BatchRequest
from modules.batch.backends.google_backend import GoogleBatchBackend
from modules.batch.jsonl import read_jsonl_records
from modules.batch.repair import Job, _merge_repairs_into_temp_jsonl
from modules.batch.results import _dedupe_transcriptions_by_custom_id
from modules.transcribe.manager import WorkflowManager

# ---------------------------------------------------------------------------
# _dedupe_transcriptions_by_custom_id
# ---------------------------------------------------------------------------


class TestDedupeTranscriptions:
    @pytest.mark.unit
    def test_last_wins_for_duplicates(self) -> None:
        entries = [
            {"custom_id": "req-1", "transcription": "old"},
            {"custom_id": "req-2", "transcription": "two"},
            {"custom_id": "req-1", "transcription": "new"},
        ]
        out = _dedupe_transcriptions_by_custom_id(entries)
        assert len(out) == 2
        assert out[0]["transcription"] == "new"
        assert out[1]["transcription"] == "two"

    @pytest.mark.unit
    def test_error_never_replaces_success(self) -> None:
        entries = [
            {"custom_id": "req-1", "transcription": "good"},
            {
                "custom_id": "req-1",
                "transcription": "[transcription error: expired]",
                "error": True,
            },
        ]
        out = _dedupe_transcriptions_by_custom_id(entries)
        assert len(out) == 1
        assert out[0]["transcription"] == "good"

    @pytest.mark.unit
    def test_success_replaces_error(self) -> None:
        entries = [
            {
                "custom_id": "req-1",
                "transcription": "[transcription error: expired]",
                "error": True,
            },
            {"custom_id": "req-1", "transcription": "good"},
        ]
        out = _dedupe_transcriptions_by_custom_id(entries)
        assert len(out) == 1
        assert out[0]["transcription"] == "good"

    @pytest.mark.unit
    def test_entries_without_custom_id_are_kept(self) -> None:
        entries = [
            {"custom_id": "", "transcription": "a"},
            {"transcription": "b"},
            {"custom_id": "req-1", "transcription": "c"},
        ]
        out = _dedupe_transcriptions_by_custom_id(entries)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# _parse_temp_file_metadata: post-finalize resubmission tracking
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestPostFinalizeMetadata:
    @pytest.mark.unit
    def test_finalized_without_resubmission(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"batch_session": {"status": "submitting", "provider": "openai"}},
                {"batch_tracking": {"batch_id": "batch_A", "provider": "openai"}},
                {"batch_session": {"status": "finalized"}},
            ],
        )
        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert "finalized" in meta["batch_session_statuses"]
        assert meta["batch_ids"] == {"batch_A"}
        assert meta["post_finalize_batch_ids"] == set()
        assert meta["post_finalize_tracking_records"] == []

    @pytest.mark.unit
    def test_resubmission_after_finalize_is_surfaced(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"batch_tracking": {"batch_id": "batch_A", "provider": "openai"}},
                {"batch_session": {"status": "finalized"}},
                {"batch_session": {"status": "submitting", "provider": "openai"}},
                {"batch_tracking": {"batch_id": "batch_B", "provider": "openai"}},
            ],
        )
        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert meta["batch_ids"] == {"batch_A", "batch_B"}
        assert meta["post_finalize_batch_ids"] == {"batch_B"}
        assert [t["batch_id"] for t in meta["post_finalize_tracking_records"]] == [
            "batch_B"
        ]

    @pytest.mark.unit
    def test_second_finalize_clears_post_finalize_state(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"batch_tracking": {"batch_id": "batch_A"}},
                {"batch_session": {"status": "finalized"}},
                {"batch_tracking": {"batch_id": "batch_B"}},
                {"batch_session": {"status": "finalized"}},
            ],
        )
        meta = batch_status._parse_temp_file_metadata(jsonl)
        assert meta["post_finalize_batch_ids"] == set()


# ---------------------------------------------------------------------------
# _recover_batch_ids: partial recovery and provider/metadata persistence
# ---------------------------------------------------------------------------


class TestRecoverBatchIdsRound2:
    @pytest.mark.unit
    def test_partial_loss_recovers_missing_part(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost tracking write for one part is recovered even when other
        parts were recorded (the old early-return skipped recovery)."""
        jsonl = tmp_path / "job_transcription.jsonl"
        jsonl.write_text("", encoding="utf-8")
        debug = tmp_path / "job_batch_submission_debug.json"
        debug.write_text(
            json.dumps(
                {
                    "provider": "google",
                    "batch_ids": ["batches/1", "batches/2"],
                    "parts": [
                        {
                            "batch_id": "batches/2",
                            "provider": "google",
                            "metadata": {"custom_id_map": {"req-1": 0}},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(batch_status, "print_info", MagicMock())

        tracking_records: list[dict] = [{"batch_id": "batches/1"}]
        out = batch_status._recover_batch_ids(
            temp_file=jsonl,
            identifier="job",
            batch_ids={"batches/1"},
            processing_settings={"persist_recovered_batch_ids": True},
            tracking_records=tracking_records,
        )
        assert out == {"batches/1", "batches/2"}
        recovered = [t for t in tracking_records if t.get("batch_id") == "batches/2"]
        assert recovered and recovered[0]["provider"] == "google"
        assert recovered[0]["metadata"] == {"custom_id_map": {"req-1": 0}}
        # Persisted records carry provider + metadata for future runs
        persisted = read_jsonl_records(jsonl)
        assert any(
            r.get("batch_tracking", {}).get("batch_id") == "batches/2"
            and r["batch_tracking"].get("provider") == "google"
            and r["batch_tracking"].get("metadata")
            for r in persisted
        )

    @pytest.mark.unit
    def test_no_artifact_leaves_ids_unchanged(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job_transcription.jsonl"
        jsonl.write_text("", encoding="utf-8")
        out = batch_status._recover_batch_ids(
            temp_file=jsonl,
            identifier="job",
            batch_ids={"batch_known"},
            processing_settings={},
        )
        assert out == {"batch_known"}


# ---------------------------------------------------------------------------
# Google inline batch submission shape
# ---------------------------------------------------------------------------


class TestGoogleInlineSubmission:
    @pytest.mark.unit
    def test_inline_requests_use_sdk_config_shape(self) -> None:
        """Inline requests must be InlinedRequest-shaped: system prompt and
        generation parameters folded into ``config`` (the REST-style
        top-level keys are rejected by SDK validation before any network
        call)."""
        backend = GoogleBatchBackend()
        client = MagicMock()
        client.batches.create.return_value = SimpleNamespace(name="batches/xyz")
        requests = [
            BatchRequest(
                custom_id=f"req-{i + 1}",
                image_base64="aGk=",
                mime_type="image/png",
                order_index=i,
                image_info={"image_name": f"p{i}.png"},
            )
            for i in range(2)
        ]
        with patch.object(backend, "_get_client", return_value=client):
            handle = backend.submit_batch(
                requests,
                {
                    "name": "gemini-3-flash-preview",
                    "max_output_tokens": 100,
                    "temperature": 0,
                },
                system_prompt="Transcribe the page.",
            )

        assert handle.batch_id == "batches/xyz"
        assert client.batches.create.call_count == 1
        src = client.batches.create.call_args.kwargs["src"]
        assert len(src) == 2
        for entry in src:
            assert set(entry.keys()) == {"contents", "config"}
            assert "system_instruction" not in entry
            assert "generation_config" not in entry
            assert entry["config"]["system_instruction"]
            assert entry["config"]["max_output_tokens"] == 100
        # The exact shape must validate against the installed SDK offline.
        from google.genai import types

        types.BatchJobSource(inlined_requests=src)


# ---------------------------------------------------------------------------
# Manager guard against redundant batch resubmission
# ---------------------------------------------------------------------------


class TestSkipRedundantBatchSubmission:
    def _call(self, path: Path) -> bool:
        return WorkflowManager._skip_redundant_batch_submission(
            SimpleNamespace(), path, "item"
        )

    @pytest.mark.unit
    def test_pending_batch_blocks_resubmission(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"batch_session": {"status": "submitting", "provider": "openai"}},
                {"batch_tracking": {"batch_id": "batch_A", "provider": "openai"}},
            ],
        )
        assert self._call(jsonl) is True

    @pytest.mark.unit
    def test_finalized_jsonl_blocks_resubmission(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"batch_tracking": {"batch_id": "batch_A"}},
                {"batch_session": {"status": "finalized"}},
            ],
        )
        assert self._call(jsonl) is True

    @pytest.mark.unit
    def test_resubmitted_pending_batch_blocks_again(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"batch_tracking": {"batch_id": "batch_A"}},
                {"batch_session": {"status": "finalized"}},
                {"batch_tracking": {"batch_id": "batch_B"}},
            ],
        )
        assert self._call(jsonl) is True

    @pytest.mark.unit
    def test_fresh_jsonl_allows_submission(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "job.jsonl"
        _write_jsonl(
            jsonl, [{"image_name": "p1.png", "order_index": 0, "text_chunk": "text"}]
        )
        assert self._call(jsonl) is False

    @pytest.mark.unit
    def test_missing_file_allows_submission(self, tmp_path: Path) -> None:
        assert self._call(tmp_path / "absent.jsonl") is False


# ---------------------------------------------------------------------------
# Repair merge into temp JSONL
# ---------------------------------------------------------------------------


class TestMergeRepairsIntoTempJsonl:
    def _job(self, tmp_path: Path, jsonl: Path | None) -> Job:
        return Job(
            parent_folder=tmp_path,
            identifier="doc",
            final_txt_path=tmp_path / "doc.txt",
            temp_jsonl_path=jsonl,
            kind="PDF",
        )

    @pytest.mark.unit
    def test_repaired_text_shadows_placeholder(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "doc.jsonl"
        _write_jsonl(
            jsonl,
            [
                {
                    "image_name": "p1.png",
                    "order_index": 0,
                    "text_chunk": "[transcription error: boom]",
                }
            ],
        )
        _merge_repairs_into_temp_jsonl(
            self._job(tmp_path, jsonl), [("p1.png", 0, "repaired text")]
        )
        from modules.batch.jsonl import extract_transcription_records

        records = extract_transcription_records(
            read_jsonl_records(jsonl), deduplicate=True
        )
        assert len(records) == 1
        assert records[0]["text_chunk"] == "repaired text"

    @pytest.mark.unit
    def test_placeholder_and_invalid_entries_not_merged(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "doc.jsonl"
        _write_jsonl(jsonl, [])
        _merge_repairs_into_temp_jsonl(
            self._job(tmp_path, jsonl),
            [
                ("p1.png", 0, "[transcription error: again]"),
                ("p2.png", None, "text without order"),
                ("", 1, "text without name"),
                ("p3.png", 1, "   "),
            ],
        )
        assert read_jsonl_records(jsonl) == []

    @pytest.mark.unit
    def test_missing_jsonl_is_noop(self, tmp_path: Path) -> None:
        _merge_repairs_into_temp_jsonl(
            self._job(tmp_path, None), [("p1.png", 0, "text")]
        )
        _merge_repairs_into_temp_jsonl(
            self._job(tmp_path, tmp_path / "absent.jsonl"), [("p1.png", 0, "text")]
        )
