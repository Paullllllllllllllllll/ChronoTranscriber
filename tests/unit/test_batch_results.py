"""Unit tests for modules.batch.results.

Exercises the internal sorting helper ``_sort_transcriptions`` and the
finalisation helper ``_finalize_batch_output``. The output writer
(``write_transcription_output``) is mocked so the tests do not touch any
real post-processing pipeline.

``_sort_transcriptions`` mutates its input list in-place via the
multi-level priority strategy (order_info > batch_order > req-<n> >
fallback_index); ``_finalize_batch_output`` builds page dicts in sorted
order and routes them to the writer, respecting the optional cleanup of
the temp JSONL file.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modules.batch.results as batch_results

# ---------------------------------------------------------------------------
# _sort_transcriptions
# ---------------------------------------------------------------------------


class TestSortTranscriptions:
    @pytest.mark.unit
    def test_sorts_by_order_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_results, "print_info", MagicMock())

        entries = [
            {"custom_id": "c", "transcription": "C", "order_info": 2},
            {"custom_id": "a", "transcription": "A", "order_info": 0},
            {"custom_id": "b", "transcription": "B", "order_info": 1},
        ]
        batch_results._sort_transcriptions(entries, {}, tmp_path / "job.jsonl")
        assert [e["custom_id"] for e in entries] == ["a", "b", "c"]

    @pytest.mark.unit
    def test_falls_back_to_batch_order_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_results, "print_info", MagicMock())

        entries = [
            {"custom_id": "img_3", "transcription": "Third"},
            {"custom_id": "img_1", "transcription": "First"},
            {"custom_id": "img_2", "transcription": "Second"},
        ]
        batch_order = {"img_1": 0, "img_2": 1, "img_3": 2}

        batch_results._sort_transcriptions(entries, batch_order, tmp_path / "job.jsonl")
        assert [e["custom_id"] for e in entries] == ["img_1", "img_2", "img_3"]

    @pytest.mark.unit
    def test_req_prefix_extracted_when_no_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without explicit order info or mapping, req-<n> is parsed."""
        monkeypatch.setattr(batch_results, "print_info", MagicMock())

        entries = [
            {"custom_id": "req-3", "transcription": "C"},
            {"custom_id": "req-1", "transcription": "A"},
            {"custom_id": "req-2", "transcription": "B"},
        ]
        batch_results._sort_transcriptions(entries, {}, tmp_path / "job.jsonl")
        assert [e["custom_id"] for e in entries] == ["req-1", "req-2", "req-3"]

    @pytest.mark.unit
    def test_order_info_beats_batch_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """order_info (tier 0) should outrank batch_order (tier 1)."""
        monkeypatch.setattr(batch_results, "print_info", MagicMock())

        entries = [
            {"custom_id": "x", "transcription": "x", "order_info": 0},
            {"custom_id": "y", "transcription": "y"},
        ]
        # batch_order says y should be first, but x has order_info=0 (tier 0)
        batch_order = {"y": 0, "x": 5}
        batch_results._sort_transcriptions(entries, batch_order, tmp_path / "job.jsonl")
        assert [e["custom_id"] for e in entries] == ["x", "y"]


# ---------------------------------------------------------------------------
# _finalize_batch_output
# ---------------------------------------------------------------------------


@pytest.fixture
def silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence all UI printers for a clean test run."""
    for name in (
        "print_info",
        "print_warning",
        "print_error",
        "print_success",
    ):
        monkeypatch.setattr(batch_results, name, MagicMock())
    # Summary displays should also be silenced
    monkeypatch.setattr(batch_results, "display_page_error_summary", MagicMock())
    monkeypatch.setattr(
        batch_results, "display_transcription_not_possible_summary", MagicMock()
    )


class TestFinalizeBatchOutput:
    @pytest.mark.unit
    def test_writes_output_and_keeps_temp_file_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        silence_ui: None,
    ) -> None:
        temp_file = tmp_path / "job_transcription.jsonl"
        temp_file.write_text("", encoding="utf-8")
        expected_out = tmp_path / "job.txt"

        writer = MagicMock(return_value=expected_out)
        monkeypatch.setattr(batch_results, "write_transcription_output", writer)

        all_tx = [
            {
                "transcription": "Page A",
                "image_info": {"page_number": 1, "image_name": "p1.jpg"},
            },
            {
                "transcription": "Page B",
                "image_info": {"page_number": 2, "image_name": "p2.jpg"},
            },
        ]

        batch_results._finalize_batch_output(
            all_transcriptions=all_tx,
            temp_file=temp_file,
            identifier="job",
            processing_settings={"retain_temporary_jsonl": True},
            postprocessing_config=None,
            output_format="txt",
        )

        writer.assert_called_once()
        call_kwargs = writer.call_args
        passed_pages, passed_path = call_kwargs.args[0], call_kwargs.args[1]
        assert (
            passed_path == expected_out.parent / f"{'job'}.txt"
            or passed_path == expected_out
        )
        assert [p["text"] for p in passed_pages] == ["Page A", "Page B"]
        assert [p["page_number"] for p in passed_pages] == [1, 2]
        # Temp file should be retained because retain_temporary_jsonl=True
        assert temp_file.exists()

    @pytest.mark.unit
    def test_deletes_temp_file_when_not_retained(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        silence_ui: None,
    ) -> None:
        temp_file = tmp_path / "job_transcription.jsonl"
        temp_file.write_text("", encoding="utf-8")

        writer = MagicMock(return_value=tmp_path / "job.txt")
        monkeypatch.setattr(batch_results, "write_transcription_output", writer)

        all_tx = [
            {
                "transcription": "only page",
                "image_info": {"page_number": 1, "image_name": "p1.jpg"},
            },
        ]

        batch_results._finalize_batch_output(
            all_transcriptions=all_tx,
            temp_file=temp_file,
            identifier="job",
            processing_settings={"retain_temporary_jsonl": False},
            postprocessing_config=None,
            output_format="txt",
        )

        assert not temp_file.exists()

    @pytest.mark.unit
    def test_keeps_temp_file_when_writer_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        silence_ui: None,
    ) -> None:
        """If writing fails, the temp file is kept even with retain=False."""
        temp_file = tmp_path / "job_transcription.jsonl"
        temp_file.write_text("", encoding="utf-8")

        writer = MagicMock(side_effect=OSError("disk full"))
        monkeypatch.setattr(batch_results, "write_transcription_output", writer)

        batch_results._finalize_batch_output(
            all_transcriptions=[
                {
                    "transcription": "x",
                    "image_info": {"page_number": 1, "image_name": "p1.jpg"},
                }
            ],
            temp_file=temp_file,
            identifier="job",
            processing_settings={"retain_temporary_jsonl": False},
            postprocessing_config=None,
            output_format="txt",
        )

        # Failure path: temp file stays because processing_success=False
        assert temp_file.exists()

    @pytest.mark.unit
    def test_handles_empty_transcription_list(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        silence_ui: None,
    ) -> None:
        """An empty transcription list still invokes the writer with empty pages."""
        temp_file = tmp_path / "job_transcription.jsonl"
        temp_file.write_text("", encoding="utf-8")

        writer = MagicMock(return_value=tmp_path / "job.txt")
        monkeypatch.setattr(batch_results, "write_transcription_output", writer)

        batch_results._finalize_batch_output(
            all_transcriptions=[],
            temp_file=temp_file,
            identifier="job",
            processing_settings={"retain_temporary_jsonl": True},
            postprocessing_config=None,
            output_format="txt",
        )

        writer.assert_called_once()
        passed_pages = writer.call_args.args[0]
        assert passed_pages == []

    @pytest.mark.unit
    def test_forwards_output_format_and_postprocessing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        silence_ui: None,
    ) -> None:
        temp_file = tmp_path / "job_transcription.jsonl"
        temp_file.write_text("", encoding="utf-8")

        writer = MagicMock(return_value=tmp_path / "job.md")
        monkeypatch.setattr(batch_results, "write_transcription_output", writer)

        pp_cfg = {"enabled": True, "merge_hyphenation": False}

        batch_results._finalize_batch_output(
            all_transcriptions=[
                {
                    "transcription": "page",
                    "image_info": {"page_number": 1, "image_name": "p.jpg"},
                }
            ],
            temp_file=temp_file,
            identifier="job",
            processing_settings={"retain_temporary_jsonl": True},
            postprocessing_config=pp_cfg,
            output_format="md",
        )

        writer.assert_called_once()
        kwargs = writer.call_args.kwargs
        assert kwargs["output_format"] == "md"
        assert kwargs["postprocess"] is True
        assert kwargs["postprocessing_config"] == pp_cfg


# ---------------------------------------------------------------------------
# _download_and_parse_openai_results — error-file + reconciliation (B1)
# ---------------------------------------------------------------------------


class _FakeContent:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeFiles:
    def __init__(self, mapping: dict[str, bytes]) -> None:
        self._mapping = mapping

    def content(self, file_id: str) -> _FakeContent:
        return _FakeContent(self._mapping[file_id])


class _FakeBatches:
    def retrieve(self, batch_id: str) -> object:  # pragma: no cover - not used
        raise AssertionError("retrieve should not be called when output present")


class _FakeClient:
    def __init__(self, mapping: dict[str, bytes]) -> None:
        self.files = _FakeFiles(mapping)
        self.batches = _FakeBatches()


class TestOpenAIErrorFileReconciliation:
    @pytest.mark.unit
    def test_error_file_and_missing_pages_yield_placeholders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("print_info", "print_warning", "print_error", "print_success"):
            monkeypatch.setattr(batch_results, name, MagicMock())
        monkeypatch.setattr(
            batch_results, "print_transcription_item_error", MagicMock()
        )

        out_content = (
            json.dumps(
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 200,
                        "body": {"transcription": "Page one"},
                    },
                }
            )
            + "\n"
        ).encode("utf-8")
        err_content = (
            json.dumps(
                {
                    "custom_id": "req-2",
                    "response": {
                        "status_code": 400,
                        "body": {
                            "error": {"message": "bad image", "code": "invalid_request"}
                        },
                    },
                }
            )
            + "\n"
        ).encode("utf-8")

        client = _FakeClient({"out1": out_content, "err1": err_content})
        batch_dict = {
            "batch_x": {
                "id": "batch_x",
                "output_file_id": "out1",
                "error_file_id": "err1",
            }
        }
        custom_id_map = {
            "req-1": {"image_name": "p1.jpg", "page_number": 1, "order_index": 0},
            "req-2": {"image_name": "p2.jpg", "page_number": 2, "order_index": 1},
            "req-3": {"image_name": "p3.jpg", "page_number": 3, "order_index": 2},
        }
        batch_order = {"req-1": 0, "req-2": 1, "req-3": 2}

        entries, all_completed = batch_results._download_and_parse_openai_results(
            {"batch_x"},
            batch_dict,
            client,  # type: ignore[arg-type]
            custom_id_map,
            batch_order,
            tmp_path / "job.jsonl",
        )

        assert all_completed is True
        by_id = {e.get("custom_id"): e for e in entries}
        # Successful page survives.
        assert by_id["req-1"]["transcription"] == "Page one"
        # Error-file page emits a placeholder instead of vanishing.
        assert by_id["req-2"].get("error") is True
        assert "[transcription error" in by_id["req-2"]["transcription"]
        # Page missing from BOTH output and error files is reconciled.
        assert by_id["req-3"].get("error") is True
        assert "[transcription error" in by_id["req-3"]["transcription"]
