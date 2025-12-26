"""Unit tests for modules/processing/text_processing.py.

Tests text extraction and processing functions for transcription outputs.
"""

from __future__ import annotations

import json
import pytest

from modules.processing.text_processing import (
    detect_transcription_cause,
    format_page_line,
    extract_transcribed_text,
    process_batch_output,
)


class TestDetectTranscriptionCause:
    """Tests for detect_transcription_cause function."""
    
    @pytest.mark.unit
    def test_detects_api_error(self):
        """Test detection of API error placeholders."""
        assert detect_transcription_cause("[transcription error]") == "api_error"
        assert detect_transcription_cause("[transcription error: page.png]") == "api_error"
        assert detect_transcription_cause("page.png: [transcription error]") == "api_error"
    
    @pytest.mark.unit
    def test_detects_no_text(self):
        """Test detection of no-text placeholders."""
        assert detect_transcription_cause("[No transcribable text]") == "no_text"
        assert detect_transcription_cause("[no transcribable text]") == "no_text"
    
    @pytest.mark.unit
    def test_detects_not_possible(self):
        """Test detection of not-possible placeholders."""
        assert detect_transcription_cause("[Transcription not possible]") == "not_possible"
        assert detect_transcription_cause("[transcription not possible]") == "not_possible"
    
    @pytest.mark.unit
    def test_normal_text_is_ok(self):
        """Test that normal text returns 'ok'."""
        assert detect_transcription_cause("This is normal transcription text.") == "ok"
        assert detect_transcription_cause("Hello World") == "ok"
    
    @pytest.mark.unit
    def test_empty_text(self):
        """Test handling of empty text."""
        assert detect_transcription_cause("") == "ok"
        assert detect_transcription_cause(None) == "ok"
    
    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test case insensitivity of detection."""
        assert detect_transcription_cause("[TRANSCRIPTION ERROR]") == "api_error"
        assert detect_transcription_cause("[NO TRANSCRIBABLE TEXT]") == "no_text"


class TestFormatPageLine:
    """Tests for format_page_line function."""
    
    @pytest.mark.unit
    def test_with_image_name(self):
        """Test formatting with image name."""
        result = format_page_line("Some text", page_number=1, image_name="page_001.png")
        # Normal text should just return the text (no header for ok text)
        assert "Some text" in result
    
    @pytest.mark.unit
    def test_with_page_number_fallback(self):
        """Test formatting with page number when no image name."""
        result = format_page_line("[transcription error]", page_number=5, image_name=None)
        assert "Page 5:" in result
        assert "[transcription error]" in result
    
    @pytest.mark.unit
    def test_unknown_image_fallback(self):
        """Test formatting when no identifier available."""
        result = format_page_line("[No transcribable text]", page_number=None, image_name=None)
        assert "[unknown image]:" in result
    
    @pytest.mark.unit
    def test_error_placeholder_inline(self):
        """Test that error placeholders are kept inline with header."""
        result = format_page_line("[transcription error: test.png]", page_number=1, image_name="test.png")
        assert "test.png:" in result
        assert "[transcription error" in result
    
    @pytest.mark.unit
    def test_normal_text_no_header(self):
        """Test that normal text doesn't get header prepended."""
        result = format_page_line("Normal transcription text.", page_number=1, image_name="page.png")
        # For normal text, should return just the text
        assert result == "Normal transcription text."


class TestExtractTranscribedText:
    """Tests for extract_transcribed_text function."""
    
    @pytest.mark.unit
    def test_schema_object_with_transcription(self):
        """Test extraction from schema object with transcription."""
        data = {
            "transcription": "This is the text.",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }
        result = extract_transcribed_text(data)
        assert result == "This is the text."
    
    @pytest.mark.unit
    def test_schema_object_no_transcribable_text(self):
        """Test extraction when no_transcribable_text is True."""
        data = {
            "transcription": "",
            "no_transcribable_text": True,
            "transcription_not_possible": False,
        }
        result = extract_transcribed_text(data)
        assert result == "[No transcribable text]"
    
    @pytest.mark.unit
    def test_schema_object_not_possible(self):
        """Test extraction when transcription_not_possible is True."""
        data = {
            "transcription": "",
            "no_transcribable_text": False,
            "transcription_not_possible": True,
        }
        result = extract_transcribed_text(data)
        assert result == "[Transcription not possible]"
    
    @pytest.mark.unit
    def test_responses_api_output_text(self):
        """Test extraction from Responses API with output_text."""
        data = {
            "output_text": json.dumps({
                "transcription": "API text",
                "no_transcribable_text": False,
                "transcription_not_possible": False,
            })
        }
        result = extract_transcribed_text(data)
        assert result == "API text"
    
    @pytest.mark.unit
    def test_responses_api_output_list(self):
        """Test extraction from Responses API with output list."""
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "transcription": "List text",
                                "no_transcribable_text": False,
                                "transcription_not_possible": False,
                            })
                        }
                    ]
                }
            ]
        }
        result = extract_transcribed_text(data)
        assert result == "List text"
    
    @pytest.mark.unit
    def test_chat_completions_parsed(self):
        """Test extraction from Chat Completions with parsed field."""
        data = {
            "choices": [
                {
                    "message": {
                        "parsed": {
                            "transcription": "Parsed text",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        }
                    }
                }
            ]
        }
        result = extract_transcribed_text(data)
        assert result == "Parsed text"
    
    @pytest.mark.unit
    def test_chat_completions_content(self):
        """Test extraction from Chat Completions content field."""
        data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "transcription": "Content text",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        })
                    }
                }
            ]
        }
        result = extract_transcribed_text(data)
        assert result == "Content text"
    
    @pytest.mark.unit
    def test_unrecognized_format(self):
        """Test handling of unrecognized response format."""
        data = {"unknown_field": "value"}
        result = extract_transcribed_text(data)
        assert "[transcription error]" in result


class TestProcessBatchOutput:
    """Tests for process_batch_output function."""
    
    @pytest.mark.unit
    def test_responses_api_format(self):
        """Test processing Responses API batch output."""
        content = json.dumps({
            "response": {
                "status_code": 200,
                "body": {
                    "output_text": json.dumps({
                        "transcription": "Batch text",
                        "no_transcribable_text": False,
                        "transcription_not_possible": False,
                    })
                }
            }
        })
        result = process_batch_output(content.encode())
        assert len(result) == 1
        assert result[0] == "Batch text"
    
    @pytest.mark.unit
    def test_jsonl_format(self):
        """Test processing JSONL batch output."""
        lines = [
            json.dumps({
                "response": {
                    "body": {
                        "transcription": "Line 1",
                        "no_transcribable_text": False,
                        "transcription_not_possible": False,
                    }
                }
            }),
            json.dumps({
                "response": {
                    "body": {
                        "transcription": "Line 2",
                        "no_transcribable_text": False,
                        "transcription_not_possible": False,
                    }
                }
            }),
        ]
        content = "\n".join(lines)
        result = process_batch_output(content.encode())
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_json_array_format(self):
        """Test processing JSON array batch output."""
        items = [
            {
                "choices": [
                    {"message": {"content": json.dumps({
                        "transcription": "Array item",
                        "no_transcribable_text": False,
                        "transcription_not_possible": False,
                    })}}
                ]
            }
        ]
        content = json.dumps(items)
        result = process_batch_output(content.encode())
        assert len(result) == 1
        assert result[0] == "Array item"
    
    @pytest.mark.unit
    def test_handles_bytes_input(self):
        """Test that bytes input is handled correctly."""
        content = b'{"transcription": "Bytes test", "no_transcribable_text": false, "transcription_not_possible": false}'
        result = process_batch_output(content)
        assert len(result) == 1
    
    @pytest.mark.unit
    def test_handles_string_input(self):
        """Test that string input is handled correctly."""
        content = '{"transcription": "String test", "no_transcribable_text": false, "transcription_not_possible": false}'
        result = process_batch_output(content)
        assert len(result) == 1
    
    @pytest.mark.unit
    def test_invalid_json_line(self):
        """Test handling of invalid JSON lines."""
        content = "not valid json\n" + json.dumps({
            "transcription": "Valid",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        })
        result = process_batch_output(content.encode())
        # Should skip invalid line but process valid one
        assert len(result) >= 1
    
    @pytest.mark.unit
    def test_empty_content(self):
        """Test handling of empty content."""
        result = process_batch_output(b"")
        assert result == []
    
    @pytest.mark.unit
    def test_whitespace_content(self):
        """Test handling of whitespace-only content."""
        result = process_batch_output(b"   \n  \n  ")
        assert result == []
