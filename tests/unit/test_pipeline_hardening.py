"""Tests for transcription-pipeline hardening.

Covers two hardening fixes in modules.transcribe.pipeline:

* Output-write failures propagate as ``OutputWriteError`` (so the item counts
  failed) instead of being swallowed while the temp JSONL is cleaned up.
* Tesseract page images derive their absolute order index from the filename,
  so a resumed run over a different page subset does not scramble page order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.transcribe import pipeline as pipeline_mod
from modules.transcribe.pipeline import (
    OutputWriteError,
    _absolute_order_index,
    run_transcription_pipeline,
    write_output_from_jsonl,
)

# ---------------------------------------------------------------------------
# Absolute order index (Tesseract page ordering across resumed subsets)
# ---------------------------------------------------------------------------


class TestAbsoluteOrderIndex:
    @pytest.mark.unit
    def test_tesseract_names_map_to_absolute_page(self) -> None:
        """page_NNNN_tess_preprocessed maps to page-1, regardless of list order."""
        files = [
            Path("page_0005_tess_preprocessed.png"),
            Path("page_0001_tess_preprocessed.png"),
            Path("page_0006_tess_preprocessed.png"),
        ]
        mapping = _absolute_order_index(files)
        assert mapping["page_0005_tess_preprocessed.png"] == 4
        assert mapping["page_0001_tess_preprocessed.png"] == 0
        assert mapping["page_0006_tess_preprocessed.png"] == 5

    @pytest.mark.unit
    def test_case_insensitive_match(self) -> None:
        mapping = _absolute_order_index([Path("PAGE_0003_TESS_PREPROCESSED.TIF")])
        assert mapping["PAGE_0003_TESS_PREPROCESSED.TIF"] == 2

    @pytest.mark.unit
    def test_non_tesseract_names_fall_back_to_positional(self) -> None:
        """Image-folder sources with no page number keep the positional index."""
        files = [Path("scan_a.jpg"), Path("scan_b.jpg"), Path("scan_c.jpg")]
        mapping = _absolute_order_index(files)
        assert mapping == {"scan_a.jpg": 0, "scan_b.jpg": 1, "scan_c.jpg": 2}


# ---------------------------------------------------------------------------
# Output-write failure propagation
# ---------------------------------------------------------------------------


def _write_resume_jsonl(path: Path, image_name: str) -> None:
    """Write a minimal versioned JSONL with one completed transcription."""
    marker = {"resume_format": {"version": 1}}
    record = {
        "file_name": "source.pdf",
        "image_name": image_name,
        "order_index": 0,
        "method": "gpt",
        "text_chunk": "already transcribed page",
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(marker, ensure_ascii=False) + "\n")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class TestWriteOutputFromJsonl:
    @pytest.mark.unit
    def test_returns_false_when_writer_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jsonl = tmp_path / "s.jsonl"
        _write_resume_jsonl(jsonl, "img1.png")

        def boom(*args, **kwargs):
            raise OSError("simulated disk-full")

        monkeypatch.setattr(pipeline_mod, "write_transcription_output", boom)
        ok = write_output_from_jsonl(jsonl, tmp_path / "out.txt", {})
        assert ok is False


@pytest.mark.asyncio
class TestRunTranscriptionPipelineWriteFailure:
    async def test_all_processed_write_failure_raises_and_preserves_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The all-images-already-processed early return must raise (not swallow)
        a write failure, and leave the temp JSONL intact for the next resume."""
        jsonl = tmp_path / "source.jsonl"
        _write_resume_jsonl(jsonl, "img1.png")
        out_path = tmp_path / "out.txt"

        def boom(*args, **kwargs):
            raise OSError("simulated disk-full during regeneration")

        monkeypatch.setattr(pipeline_mod, "write_transcription_output", boom)

        with pytest.raises(OutputWriteError):
            await run_transcription_pipeline(
                image_files=[tmp_path / "img1.png"],
                method="gpt",
                transcriber=None,
                temp_jsonl_path=jsonl,
                output_txt_path=out_path,
                source_name="source.pdf",
                concurrency_config={},
                image_processing_config={},
                postprocessing_config={},
                resume_mode="skip",
            )

        # The only copy of the transcriptions must survive the failed write.
        assert jsonl.exists()
        assert jsonl.stat().st_size > 0
