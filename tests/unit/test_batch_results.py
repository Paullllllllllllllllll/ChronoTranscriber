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

        batch_results._sort_transcriptions(
            entries, batch_order, tmp_path / "job.jsonl"
        )
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
        batch_results._sort_transcriptions(
            entries, batch_order, tmp_path / "job.jsonl"
        )
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
        assert passed_path == expected_out.parent / f"{'job'}.txt" or passed_path == expected_out
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
