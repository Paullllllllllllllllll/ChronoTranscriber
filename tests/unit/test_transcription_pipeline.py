"""Tests for modules.transcribe.pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.transcribe.pipeline import (
    _build_jsonl_record,
    run_transcription_pipeline,
    transcribe_single_image,
    write_output_from_jsonl,
)

# ---------------------------------------------------------------------------
# transcribe_single_image
# ---------------------------------------------------------------------------


class TestTranscribeSingleImage:
    @pytest.mark.asyncio
    async def test_gpt_method_without_transcriber_returns_error(
        self, tmp_path: Path
    ) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"")
        result = await transcribe_single_image(
            img_path=img,
            transcriber=None,
            method="gpt",
            order_index=0,
        )
        assert result[0] == str(img)
        assert "[transcription error" in result[2]

    @pytest.mark.asyncio
    @patch("modules.transcribe.pipeline.perform_ocr")
    async def test_tesseract_method(self, mock_ocr, tmp_path: Path) -> None:
        mock_ocr.return_value = "OCR text"
        img = tmp_path / "test.png"
        img.write_bytes(b"")
        result = await transcribe_single_image(
            img_path=img,
            transcriber=None,
            method="tesseract",
            order_index=5,
        )
        assert result[2] == "OCR text"
        assert result[4] == 5
        mock_ocr.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_method_returns_none_text(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"")
        result = await transcribe_single_image(
            img_path=img,
            transcriber=None,
            method="unknown",
            order_index=0,
        )
        assert result[2] is None

    @pytest.mark.asyncio
    @patch("modules.transcribe.pipeline.transcribe_image_with_llm")
    @patch("modules.transcribe.pipeline.extract_transcribed_text")
    async def test_gpt_method_success(
        self, mock_extract, mock_llm, tmp_path: Path
    ) -> None:
        mock_llm.return_value = {"content": "some text"}
        mock_extract.return_value = "Extracted text"

        img = tmp_path / "page.png"
        img.write_bytes(b"")
        transcriber = MagicMock()

        result = await transcribe_single_image(
            img_path=img,
            transcriber=transcriber,
            method="gpt",
            order_index=2,
        )
        assert result[2] == "Extracted text"
        assert result[3] == {"content": "some text"}
        assert result[4] == 2

    @pytest.mark.asyncio
    @patch("modules.transcribe.pipeline.transcribe_image_with_llm")
    @patch("modules.transcribe.pipeline.extract_transcribed_text")
    async def test_gpt_extraction_error_returns_error_marker(
        self, mock_extract, mock_llm, tmp_path: Path
    ) -> None:
        mock_llm.return_value = {"content": "raw"}
        mock_extract.side_effect = ValueError("parse error")

        img = tmp_path / "page.png"
        img.write_bytes(b"")
        transcriber = MagicMock()

        result = await transcribe_single_image(
            img_path=img,
            transcriber=transcriber,
            method="gpt",
            order_index=0,
        )
        assert "[transcription error" in result[2]

    @pytest.mark.asyncio
    @patch("modules.transcribe.pipeline.perform_ocr")
    async def test_exception_during_ocr_returns_error(
        self, mock_ocr, tmp_path: Path
    ) -> None:
        mock_ocr.side_effect = RuntimeError("OCR crash")

        img = tmp_path / "page.png"
        img.write_bytes(b"")

        result = await transcribe_single_image(
            img_path=img,
            transcriber=None,
            method="tesseract",
            order_index=0,
        )
        assert "[transcription error" in result[2]


# ---------------------------------------------------------------------------
# _build_jsonl_record
# ---------------------------------------------------------------------------


class TestBuildJsonlRecord:
    def test_returns_none_for_none_tuple(self) -> None:
        assert _build_jsonl_record(None, "src", "gpt", False, None) is None

    def test_returns_none_for_short_tuple(self) -> None:
        assert _build_jsonl_record(("a", "b"), "src", "gpt", False, None) is None

    def test_returns_none_when_text_is_none(self) -> None:
        result = _build_jsonl_record(
            ("path", "img.png", None, None, 0), "src", "gpt", False, None
        )
        assert result is None

    def test_folder_uses_folder_name_key(self) -> None:
        result = _build_jsonl_record(
            ("path", "img.png", "text", None, 0), "myfolder", "tesseract", True, None
        )
        assert "folder_name" in result
        assert result["folder_name"] == "myfolder"

    def test_pdf_uses_file_name_key(self) -> None:
        result = _build_jsonl_record(
            ("path", "img.png", "text", None, 0), "myfile.pdf", "tesseract", False, None
        )
        assert "file_name" in result
        assert result["file_name"] == "myfile.pdf"

    def test_includes_basic_fields(self) -> None:
        result = _build_jsonl_record(
            ("/path/to/img.png", "img.png", "Some text", None, 3),
            "source.pdf",
            "tesseract",
            False,
            None,
        )
        assert result["pre_processed_image"] == "/path/to/img.png"
        assert result["image_name"] == "img.png"
        assert result["text_chunk"] == "Some text"
        assert result["order_index"] == 3
        assert result["method"] == "tesseract"
        assert "timestamp" in result

    def test_gpt_includes_request_context(self) -> None:
        mock_transcriber = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.model = "gpt-4o"
        mock_extractor.service_tier = "auto"
        mock_extractor.max_output_tokens = 4096
        mock_extractor.temperature = 0.1
        mock_extractor.top_p = 1.0
        mock_extractor.presence_penalty = 0.0
        mock_extractor.frequency_penalty = 0.0
        mock_extractor.stop = None
        mock_extractor.seed = None
        mock_extractor.reasoning = None
        mock_extractor.text_params = None
        mock_transcriber.extractor = mock_extractor

        raw_resp = {"output": "data"}
        result = _build_jsonl_record(
            ("/path", "img.png", "text", raw_resp, 0),
            "source.pdf",
            "gpt",
            False,
            mock_transcriber,
        )
        assert "request_context" in result
        assert result["request_context"]["model"] == "gpt-4o"
        assert "raw_response" in result

    def test_tesseract_no_request_context(self) -> None:
        result = _build_jsonl_record(
            ("/path", "img.png", "text", None, 0),
            "source.pdf",
            "tesseract",
            False,
            None,
        )
        assert "request_context" not in result
        assert "raw_response" not in result


# ---------------------------------------------------------------------------
# write_output_from_jsonl
# ---------------------------------------------------------------------------


class TestWriteOutputFromJsonl:
    @patch("modules.postprocess.writer.postprocess_transcription")
    @patch("modules.transcribe.pipeline.read_jsonl_records")
    @patch("modules.transcribe.pipeline.extract_transcription_records")
    def test_writes_output_file(
        self, mock_extract, mock_read, mock_pp, tmp_path: Path
    ) -> None:
        mock_read.return_value = [{"dummy": "record"}]
        mock_extract.return_value = [
            {"image_name": "p1.png", "text_chunk": "Page 1 text", "order_index": 0},
            {"image_name": "p2.png", "text_chunk": "Page 2 text", "order_index": 1},
        ]
        mock_pp.return_value = "Processed output"

        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text("{}", encoding="utf-8")
        output_path = tmp_path / "output.txt"

        result = write_output_from_jsonl(jsonl_path, output_path, {})
        assert result is True
        assert output_path.read_text(encoding="utf-8") == "Processed output"

    @patch("modules.transcribe.pipeline.read_jsonl_records")
    @patch("modules.transcribe.pipeline.extract_transcription_records")
    def test_returns_false_when_no_records(
        self, mock_extract, mock_read, tmp_path: Path
    ) -> None:
        mock_read.return_value = []
        mock_extract.return_value = []

        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "output.txt"

        result = write_output_from_jsonl(jsonl_path, output_path, {})
        assert result is False

    @patch("modules.transcribe.pipeline.read_jsonl_records")
    def test_returns_false_on_exception(self, mock_read, tmp_path: Path) -> None:
        mock_read.side_effect = RuntimeError("read error")

        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "output.txt"

        result = write_output_from_jsonl(jsonl_path, output_path, {})
        assert result is False


# ---------------------------------------------------------------------------
# run_transcription_pipeline
# ---------------------------------------------------------------------------


class TestRunTranscriptionPipeline:
    @pytest.mark.asyncio
    @patch("modules.transcribe.pipeline.write_output_from_jsonl")
    @patch("modules.transcribe.pipeline.get_processed_image_names")
    async def test_all_images_already_processed_skips(
        self, mock_processed, mock_write, tmp_path: Path
    ) -> None:
        mock_processed.return_value = {"page1.png", "page2.png"}
        mock_write.return_value = True

        img1 = tmp_path / "page1.png"
        img2 = tmp_path / "page2.png"
        img1.write_bytes(b"")
        img2.write_bytes(b"")

        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "output.txt"

        await run_transcription_pipeline(
            image_files=[img1, img2],
            method="tesseract",
            transcriber=None,
            temp_jsonl_path=jsonl_path,
            output_txt_path=output_path,
            source_name="test",
            concurrency_config={"concurrency": {"transcription": {}}},
            image_processing_config={},
            postprocessing_config={},
        )
        mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_overwrite_mode_clears_jsonl(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("old data\n", encoding="utf-8")

        img = tmp_path / "page.png"
        img.write_bytes(b"")
        output_path = tmp_path / "output.txt"

        # The output is regenerated from the JSONL, so the fake concurrency
        # runner must stream its result through ``on_result`` just as the real
        # runner does; otherwise nothing is recorded to combine.
        async def fake_runner(func, args_list, limit, delay, on_result=None):
            results = [(str(img), "page.png", "text", None, 0)]
            if on_result is not None:
                for r in results:
                    await on_result(r)
            return results

        with (
            patch(
                "modules.transcribe.pipeline.run_concurrent_transcription_tasks",
                side_effect=fake_runner,
            ),
            patch(
                "modules.postprocess.writer.postprocess_transcription",
                return_value="final text",
            ),
        ):
            await run_transcription_pipeline(
                image_files=[img],
                method="tesseract",
                transcriber=None,
                temp_jsonl_path=jsonl_path,
                output_txt_path=output_path,
                source_name="test",
                concurrency_config={"concurrency": {"transcription": {}}},
                image_processing_config={},
                postprocessing_config={},
                resume_mode="overwrite",
            )
        # The old data should have been cleared and the output regenerated
        # from the streamed JSONL records.
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "final text"


# ---------------------------------------------------------------------------
# Tesseract resume regression (B2): a resumed run must keep prior pages and
# preserve absolute page order.
# ---------------------------------------------------------------------------


class TestTesseractResumeRegression:
    @pytest.mark.asyncio
    async def test_resumed_run_preserves_prior_pages_and_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.batch.jsonl import resume_marker_record, write_jsonl_record

        names = ["page1.png", "page2.png", "page3.png"]
        imgs: list[Path] = []
        for n in names:
            p = tmp_path / n
            p.write_bytes(b"")
            imgs.append(p)

        jsonl_path = tmp_path / "temp.jsonl"
        # Prior (run-1) resume artifact that already transcribed pages 0 and 2,
        # carrying their absolute order_index values.
        write_jsonl_record(jsonl_path, resume_marker_record())
        write_jsonl_record(
            jsonl_path,
            {
                "folder_name": "src",
                "image_name": "page1.png",
                "order_index": 0,
                "text_chunk": "T1",
                "method": "tesseract",
            },
        )
        write_jsonl_record(
            jsonl_path,
            {
                "folder_name": "src",
                "image_name": "page3.png",
                "order_index": 2,
                "text_chunk": "T3",
                "method": "tesseract",
            },
        )
        output_path = tmp_path / "out.txt"

        # Only the still-pending page (page2) is OCR'd on this resumed run.
        monkeypatch.setattr(
            "modules.transcribe.pipeline.perform_ocr", lambda path, cfg: "T2"
        )

        captured: dict[str, Any] = {}

        def fake_writer(pages: Any, out: Path, **kw: Any) -> Path:
            captured["pages"] = pages
            Path(out).write_text("done", encoding="utf-8")
            return Path(out)

        monkeypatch.setattr(
            "modules.transcribe.pipeline.write_transcription_output", fake_writer
        )

        await run_transcription_pipeline(
            image_files=imgs,
            method="tesseract",
            transcriber=None,
            temp_jsonl_path=jsonl_path,
            output_txt_path=output_path,
            source_name="src",
            concurrency_config={
                "concurrency": {"transcription": {"concurrency_limit": 2}}
            },
            image_processing_config={},
            postprocessing_config={},
            is_folder=True,
        )

        pages = captured["pages"]
        # All three pages present, in absolute page order (not scrambled, not
        # limited to this run's single page).
        assert [p["text"] for p in pages] == ["T1", "T2", "T3"]
        assert [p["page_number"] for p in pages] == [1, 2, 3]
