"""Tests for modules.core.transcription_pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.core.transcription_pipeline import (
    transcribe_single_image,
    _build_jsonl_record,
    write_output_from_jsonl,
    run_transcription_pipeline,
)


# ---------------------------------------------------------------------------
# transcribe_single_image
# ---------------------------------------------------------------------------

class TestTranscribeSingleImage:
    @pytest.mark.asyncio
    async def test_gpt_method_without_transcriber_returns_error(self, tmp_path):
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
    @patch("modules.core.transcription_pipeline.perform_ocr")
    async def test_tesseract_method(self, mock_ocr, tmp_path):
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
    async def test_unknown_method_returns_none_text(self, tmp_path):
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
    @patch("modules.core.transcription_pipeline.transcribe_image_with_llm")
    @patch("modules.core.transcription_pipeline.extract_transcribed_text")
    async def test_gpt_method_success(self, mock_extract, mock_llm, tmp_path):
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
    @patch("modules.core.transcription_pipeline.transcribe_image_with_llm")
    @patch("modules.core.transcription_pipeline.extract_transcribed_text")
    async def test_gpt_extraction_error_returns_error_marker(
        self, mock_extract, mock_llm, tmp_path
    ):
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
    @patch("modules.core.transcription_pipeline.perform_ocr")
    async def test_exception_during_ocr_returns_error(self, mock_ocr, tmp_path):
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
    def test_returns_none_for_none_tuple(self):
        assert _build_jsonl_record(None, "src", "gpt", False, None) is None

    def test_returns_none_for_short_tuple(self):
        assert _build_jsonl_record(("a", "b"), "src", "gpt", False, None) is None

    def test_returns_none_when_text_is_none(self):
        result = _build_jsonl_record(
            ("path", "img.png", None, None, 0), "src", "gpt", False, None
        )
        assert result is None

    def test_folder_uses_folder_name_key(self):
        result = _build_jsonl_record(
            ("path", "img.png", "text", None, 0), "myfolder", "tesseract", True, None
        )
        assert "folder_name" in result
        assert result["folder_name"] == "myfolder"

    def test_pdf_uses_file_name_key(self):
        result = _build_jsonl_record(
            ("path", "img.png", "text", None, 0), "myfile.pdf", "tesseract", False, None
        )
        assert "file_name" in result
        assert result["file_name"] == "myfile.pdf"

    def test_includes_basic_fields(self):
        result = _build_jsonl_record(
            ("/path/to/img.png", "img.png", "Some text", None, 3),
            "source.pdf", "tesseract", False, None,
        )
        assert result["pre_processed_image"] == "/path/to/img.png"
        assert result["image_name"] == "img.png"
        assert result["text_chunk"] == "Some text"
        assert result["order_index"] == 3
        assert result["method"] == "tesseract"
        assert "timestamp" in result

    def test_gpt_includes_request_context(self):
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
            "source.pdf", "gpt", False, mock_transcriber,
        )
        assert "request_context" in result
        assert result["request_context"]["model"] == "gpt-4o"
        assert "raw_response" in result

    def test_tesseract_no_request_context(self):
        result = _build_jsonl_record(
            ("/path", "img.png", "text", None, 0),
            "source.pdf", "tesseract", False, None,
        )
        assert "request_context" not in result
        assert "raw_response" not in result


# ---------------------------------------------------------------------------
# write_output_from_jsonl
# ---------------------------------------------------------------------------

class TestWriteOutputFromJsonl:
    @patch("modules.core.transcription_pipeline.postprocess_transcription")
    @patch("modules.core.transcription_pipeline.read_jsonl_records")
    @patch("modules.core.transcription_pipeline.extract_transcription_records")
    def test_writes_output_file(self, mock_extract, mock_read, mock_pp, tmp_path):
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

    @patch("modules.core.transcription_pipeline.read_jsonl_records")
    @patch("modules.core.transcription_pipeline.extract_transcription_records")
    def test_returns_false_when_no_records(self, mock_extract, mock_read, tmp_path):
        mock_read.return_value = []
        mock_extract.return_value = []

        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "output.txt"

        result = write_output_from_jsonl(jsonl_path, output_path, {})
        assert result is False

    @patch("modules.core.transcription_pipeline.read_jsonl_records")
    def test_returns_false_on_exception(self, mock_read, tmp_path):
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
    @patch("modules.core.transcription_pipeline.write_output_from_jsonl")
    @patch("modules.core.transcription_pipeline.get_processed_image_names")
    async def test_all_images_already_processed_skips(
        self, mock_processed, mock_write, tmp_path
    ):
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
    async def test_overwrite_mode_clears_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("old data\n", encoding="utf-8")

        img = tmp_path / "page.png"
        img.write_bytes(b"")
        output_path = tmp_path / "output.txt"

        with patch(
            "modules.core.transcription_pipeline.run_concurrent_transcription_tasks",
            new_callable=AsyncMock,
            return_value=[
                (str(img), "page.png", "text", None, 0)
            ],
        ), patch(
            "modules.core.transcription_pipeline.postprocess_transcription",
            return_value="final text",
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
        # The old data should have been cleared
        assert output_path.exists()
