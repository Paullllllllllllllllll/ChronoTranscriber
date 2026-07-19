"""Unit tests for modules/processing/response_parsing.py.

Tests text extraction and processing functions for transcription outputs.
"""

from __future__ import annotations

import json
import time

import pytest

from modules.llm.response_parsing import (
    _normalize_llm_text,
    _salvage_last_json_object,
    _strip_code_fences,
    _try_parse_json,
    detect_transcription_cause,
    extract_transcribed_text,
    format_page_line,
    process_batch_output,
)


class TestDetectTranscriptionCause:
    """Tests for detect_transcription_cause function."""

    @pytest.mark.unit
    def test_detects_api_error(self) -> None:
        """Test detection of API error placeholders."""
        assert detect_transcription_cause("[transcription error]") == "api_error"
        assert (
            detect_transcription_cause("[transcription error: page.png]") == "api_error"
        )
        assert (
            detect_transcription_cause("page.png: [transcription error]") == "api_error"
        )

    @pytest.mark.unit
    def test_detects_no_text(self) -> None:
        """Test detection of no-text placeholders."""
        assert detect_transcription_cause("[No transcribable text]") == "no_text"
        assert detect_transcription_cause("[no transcribable text]") == "no_text"

    @pytest.mark.unit
    def test_detects_not_possible(self) -> None:
        """Test detection of not-possible placeholders."""
        assert (
            detect_transcription_cause("[Transcription not possible]") == "not_possible"
        )
        assert (
            detect_transcription_cause("[transcription not possible]") == "not_possible"
        )

    @pytest.mark.unit
    def test_normal_text_is_ok(self) -> None:
        """Test that normal text returns 'ok'."""
        assert detect_transcription_cause("This is normal transcription text.") == "ok"
        assert detect_transcription_cause("Hello World") == "ok"

    @pytest.mark.unit
    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        assert detect_transcription_cause("") == "ok"
        assert detect_transcription_cause(None) == "ok"

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Test case insensitivity of detection."""
        assert detect_transcription_cause("[TRANSCRIPTION ERROR]") == "api_error"
        assert detect_transcription_cause("[NO TRANSCRIBABLE TEXT]") == "no_text"


class TestFormatPageLine:
    """Tests for format_page_line function."""

    @pytest.mark.unit
    def test_with_image_name(self) -> None:
        """Test formatting with image name."""
        result = format_page_line("Some text", page_number=1, image_name="page_001.png")
        # Normal text should just return the text (no header for ok text)
        assert "Some text" in result

    @pytest.mark.unit
    def test_with_page_number_fallback(self) -> None:
        """Test formatting with page number when no image name."""
        result = format_page_line(
            "[transcription error]", page_number=5, image_name=None
        )
        assert "Page 5:" in result
        assert "[transcription error]" in result

    @pytest.mark.unit
    def test_unknown_image_fallback(self) -> None:
        """Test formatting when no identifier available."""
        result = format_page_line(
            "[No transcribable text]", page_number=None, image_name=None
        )
        assert "[unknown image]:" in result

    @pytest.mark.unit
    def test_error_placeholder_inline(self) -> None:
        """Test that error placeholders are kept inline with header."""
        result = format_page_line(
            "[transcription error: test.png]", page_number=1, image_name="test.png"
        )
        assert "test.png:" in result
        assert "[transcription error" in result

    @pytest.mark.unit
    def test_normal_text_no_header(self) -> None:
        """Test that normal text doesn't get header prepended."""
        result = format_page_line(
            "Normal transcription text.", page_number=1, image_name="page.png"
        )
        # For normal text, should return just the text
        assert result == "Normal transcription text."


class TestExtractTranscribedText:
    """Tests for extract_transcribed_text function."""

    @pytest.mark.unit
    def test_schema_object_with_transcription(self) -> None:
        """Test extraction from schema object with transcription."""
        data = {
            "transcription": "This is the text.",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }
        result = extract_transcribed_text(data)
        assert result == "This is the text."

    @pytest.mark.unit
    def test_schema_object_no_transcribable_text(self) -> None:
        """Test extraction when no_transcribable_text is True."""
        data = {
            "transcription": "",
            "no_transcribable_text": True,
            "transcription_not_possible": False,
        }
        result = extract_transcribed_text(data)
        assert result == "[No transcribable text]"

    @pytest.mark.unit
    def test_schema_object_not_possible(self) -> None:
        """Test extraction when transcription_not_possible is True."""
        data = {
            "transcription": "",
            "no_transcribable_text": False,
            "transcription_not_possible": True,
        }
        result = extract_transcribed_text(data)
        assert result == "[Transcription not possible]"

    @pytest.mark.unit
    def test_responses_api_output_text(self) -> None:
        """Test extraction from Responses API with output_text."""
        data = {
            "output_text": json.dumps(
                {
                    "transcription": "API text",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                }
            )
        }
        result = extract_transcribed_text(data)
        assert result == "API text"

    @pytest.mark.unit
    def test_responses_api_output_list(self) -> None:
        """Test extraction from Responses API with output list."""
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "transcription": "List text",
                                    "no_transcribable_text": False,
                                    "transcription_not_possible": False,
                                }
                            ),
                        }
                    ],
                }
            ]
        }
        result = extract_transcribed_text(data)
        assert result == "List text"

    @pytest.mark.unit
    def test_chat_completions_parsed(self) -> None:
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
    def test_chat_completions_content(self) -> None:
        """Test extraction from Chat Completions content field."""
        data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "transcription": "Content text",
                                "no_transcribable_text": False,
                                "transcription_not_possible": False,
                            }
                        )
                    }
                }
            ]
        }
        result = extract_transcribed_text(data)
        assert result == "Content text"

    @pytest.mark.unit
    def test_unrecognized_format(self) -> None:
        """Test handling of unrecognized response format."""
        data = {"unknown_field": "value"}
        result = extract_transcribed_text(data)
        assert "[transcription error]" in result

    @pytest.mark.unit
    def test_extract_error_response_returns_placeholder(self) -> None:
        """Error-response dict with empty output_text returns the error placeholder."""
        data = {"output_text": "", "error": "Connection error."}
        result = extract_transcribed_text(data, "page_001.jpg")
        assert result == "[transcription error]"

    @pytest.mark.unit
    def test_extract_error_response_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Error-response dict logs a WARNING that names the actual error."""
        import logging

        data = {"output_text": "", "error": "Connection error."}
        with caplog.at_level(logging.WARNING, logger="modules.llm.response_parsing"):
            extract_transcribed_text(data, "page_001.jpg")
        assert "Connection error." in caplog.text


class TestProcessBatchOutput:
    """Tests for process_batch_output function."""

    @pytest.mark.unit
    def test_responses_api_format(self) -> None:
        """Test processing Responses API batch output."""
        content = json.dumps(
            {
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps(
                            {
                                "transcription": "Batch text",
                                "no_transcribable_text": False,
                                "transcription_not_possible": False,
                            }
                        )
                    },
                }
            }
        )
        result = process_batch_output(content.encode())
        assert len(result) == 1
        assert result[0] == "Batch text"

    @pytest.mark.unit
    def test_jsonl_format(self) -> None:
        """Test processing JSONL batch output."""
        lines = [
            json.dumps(
                {
                    "response": {
                        "body": {
                            "transcription": "Line 1",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        }
                    }
                }
            ),
            json.dumps(
                {
                    "response": {
                        "body": {
                            "transcription": "Line 2",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        }
                    }
                }
            ),
        ]
        content = "\n".join(lines)
        result = process_batch_output(content.encode())
        assert len(result) == 2

    @pytest.mark.unit
    def test_json_array_format(self) -> None:
        """Test processing JSON array batch output."""
        items = [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "transcription": "Array item",
                                    "no_transcribable_text": False,
                                    "transcription_not_possible": False,
                                }
                            )
                        }
                    }
                ]
            }
        ]
        content = json.dumps(items)
        result = process_batch_output(content.encode())
        assert len(result) == 1
        assert result[0] == "Array item"

    @pytest.mark.unit
    def test_empty_transcription_keeps_positional_alignment(self) -> None:
        """An empty extraction must yield a placeholder, not a skipped line.

        Callers correlate the returned list positionally with per-line
        custom_ids; dropping a line would shift every later page onto the
        wrong custom_id.
        """
        lines = [
            json.dumps(
                {
                    "response": {
                        "body": {
                            "transcription": "Page one",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        }
                    }
                }
            ),
            # Empty output_text extracts to "" -> placeholder expected.
            json.dumps({"response": {"body": {"output_text": ""}}}),
            json.dumps(
                {
                    "response": {
                        "body": {
                            "transcription": "Page three",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        }
                    }
                }
            ),
        ]
        result = process_batch_output("\n".join(lines).encode())
        assert len(result) == 3
        assert result[0] == "Page one"
        assert result[1] == "[transcription error]"
        assert result[2] == "Page three"

    @pytest.mark.unit
    def test_handles_bytes_input(self) -> None:
        """Test that bytes input is handled correctly."""
        content = (
            b'{"transcription": "Bytes test", "no_transcribable_text": false,'
            b' "transcription_not_possible": false}'
        )
        result = process_batch_output(content)
        assert len(result) == 1

    @pytest.mark.unit
    def test_handles_string_input(self) -> None:
        """Test that string input is handled correctly."""
        content = (
            '{"transcription": "String test", "no_transcribable_text": false,'
            ' "transcription_not_possible": false}'
        )
        result = process_batch_output(content)
        assert len(result) == 1

    @pytest.mark.unit
    def test_invalid_json_line(self) -> None:
        """Test handling of invalid JSON lines."""
        content = "not valid json\n" + json.dumps(
            {
                "transcription": "Valid",
                "no_transcribable_text": False,
                "transcription_not_possible": False,
            }
        )
        result = process_batch_output(content.encode())
        # Should skip invalid line but process valid one
        assert len(result) >= 1

    @pytest.mark.unit
    def test_empty_content(self) -> None:
        """Test handling of empty content."""
        result = process_batch_output(b"")
        assert result == []

    @pytest.mark.unit
    def test_whitespace_content(self) -> None:
        """Test handling of whitespace-only content."""
        result = process_batch_output(b"   \n  \n  ")
        assert result == []


class TestCheckTranscriptionFlags:
    """Tests for _check_transcription_flags helper."""

    @pytest.mark.unit
    def test_no_transcribable_text(self) -> None:
        from modules.llm.response_parsing import _check_transcription_flags

        assert (
            _check_transcription_flags({"no_transcribable_text": True})
            == "[No transcribable text]"
        )

    @pytest.mark.unit
    def test_transcription_not_possible(self) -> None:
        from modules.llm.response_parsing import _check_transcription_flags

        assert (
            _check_transcription_flags({"transcription_not_possible": True})
            == "[Transcription not possible]"
        )

    @pytest.mark.unit
    def test_no_flags_returns_none(self) -> None:
        from modules.llm.response_parsing import _check_transcription_flags

        assert _check_transcription_flags({"transcription": "hello"}) is None

    @pytest.mark.unit
    def test_both_flags_false_returns_none(self) -> None:
        from modules.llm.response_parsing import _check_transcription_flags

        assert (
            _check_transcription_flags(
                {
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                }
            )
            is None
        )

    @pytest.mark.unit
    def test_no_transcribable_text_takes_precedence(self) -> None:
        from modules.llm.response_parsing import _check_transcription_flags

        result = _check_transcription_flags(
            {
                "no_transcribable_text": True,
                "transcription_not_possible": True,
            }
        )
        assert result == "[No transcribable text]"


class TestSalvageLastJsonObject:
    """Regression tests for _salvage_last_json_object and _try_parse_json.

    Guards against a latent crash: a pathological model response with many
    unbalanced/deeply nested braces used to backtrack quadratically and could
    raise an uncaught RecursionError, killing the process instead of degrading
    to the placeholder fallback.
    """

    @pytest.mark.unit
    def test_salvages_last_of_concatenated_objects(self) -> None:
        """Ordinary behavior is preserved: the last valid object wins."""
        text = '{"transcription": "first"}{"transcription": "second"}'
        obj = _salvage_last_json_object(text)
        assert obj is not None
        assert obj["transcription"] == "second"

    @pytest.mark.unit
    def test_salvages_trailing_object_after_prose(self) -> None:
        text = 'garbage {oops not json} tail {"transcription": "ok"}'
        obj = _salvage_last_json_object(text)
        assert obj is not None
        assert obj["transcription"] == "ok"

    @pytest.mark.unit
    def test_no_closing_brace_returns_none(self) -> None:
        assert _salvage_last_json_object("{" * 5000) is None

    @pytest.mark.unit
    def test_empty_returns_none(self) -> None:
        assert _salvage_last_json_object("") is None

    @pytest.mark.unit
    def test_try_parse_json_deeply_nested_returns_none(self) -> None:
        """A deeply nested valid-JSON string must not raise RecursionError."""
        depth = 6000
        deep = '{"a":' * depth + "1" + "}" * depth
        # Must degrade to None (unparseable at this depth), not crash.
        assert _try_parse_json(deep) is None

    @pytest.mark.unit
    def test_thousands_of_unbalanced_braces_degrade_fast(self) -> None:
        """Thousands of unbalanced braces return the fallback in bounded time."""
        text = '{"a":' * 6000 + "}"
        start = time.perf_counter()
        result = _salvage_last_json_object(text)
        elapsed = time.perf_counter() - start
        assert result is None
        assert elapsed < 5.0

    @pytest.mark.unit
    def test_deeply_nested_balanced_braces_no_crash(self) -> None:
        """Deeply nested balanced braces salvage to None without crashing."""
        depth = 6000
        text = '{"a":' * depth + "1" + "}" * depth
        start = time.perf_counter()
        result = _salvage_last_json_object(text)
        elapsed = time.perf_counter() - start
        # Bounded backtracking never reaches the (over-deep) outermost object.
        assert result is None
        assert elapsed < 5.0

    @pytest.mark.unit
    def test_extract_transcribed_text_pathological_content_no_crash(self) -> None:
        """The public extractor degrades to a placeholder on pathological JSON."""
        data = {"output_text": "{" * 6000}
        result = extract_transcribed_text(data, "page.png")
        # No exception; a string result is returned (fallback path).
        assert isinstance(result, str)


class TestStripCodeFences:
    """Tests for _strip_code_fences helper."""

    @pytest.mark.unit
    def test_json_code_fence(self) -> None:
        text = '```json\n{"transcription": "hello"}\n```'
        assert _strip_code_fences(text) == '{"transcription": "hello"}'

    @pytest.mark.unit
    def test_plain_code_fence(self) -> None:
        text = '```\n{"transcription": "hello"}\n```'
        assert _strip_code_fences(text) == '{"transcription": "hello"}'

    @pytest.mark.unit
    def test_no_fence_returns_unchanged(self) -> None:
        text = '{"transcription": "hello"}'
        assert _strip_code_fences(text) == text

    @pytest.mark.unit
    def test_multiple_fences_returns_last(self) -> None:
        text = '```\nthinking...\n```\n```json\n{"transcription": "real"}\n```'
        assert _strip_code_fences(text) == '{"transcription": "real"}'

    @pytest.mark.unit
    def test_fence_with_leading_prose_passes_through(self) -> None:
        # A response that does not START with a fence must pass through
        # unchanged: stripping would discard the surrounding text. JSON
        # extraction for prose-wrapped fenced JSON is handled downstream by
        # _normalize_llm_text's brace isolation (covered below).
        text = 'Here is the result:\n```json\n{"transcription": "hello"}\n```\nDone!'
        assert _strip_code_fences(text) == text

    @pytest.mark.unit
    def test_plain_text_with_embedded_fence_preserved(self) -> None:
        # Regression: a plain-text transcription legitimately containing a
        # fenced block (e.g. tabular content) must not be reduced to the
        # fence content.
        text = "Chapter I opens here.\n```\ncol1 col2\n```\nProse continues."
        assert _strip_code_fences(text) == text


class TestNormalizeLlmText:
    """Tests for _normalize_llm_text helper."""

    @pytest.mark.unit
    def test_clean_json_unchanged(self) -> None:
        text = '{"transcription": "hello"}'
        assert _normalize_llm_text(text) == text

    @pytest.mark.unit
    def test_code_fenced_json(self) -> None:
        text = '```json\n{"transcription": "hello"}\n```'
        result = _normalize_llm_text(text)
        assert result.startswith("{")
        assert json.loads(result)["transcription"] == "hello"

    @pytest.mark.unit
    def test_prose_wrapped_fenced_json_still_extracted(self) -> None:
        # With _strip_code_fences now requiring a leading fence, JSON wrapped
        # in prose plus a fence is still isolated by the brace-scan fallback.
        text = 'Here is the result:\n```json\n{"transcription": "hello"}\n```\nDone!'
        result = _normalize_llm_text(text)
        assert json.loads(result)["transcription"] == "hello"

    @pytest.mark.unit
    def test_plain_text_with_embedded_fence_not_reduced(self) -> None:
        text = "Chapter I opens here.\n```\ncol1 col2\n```\nProse continues."
        assert _normalize_llm_text(text) == text

    @pytest.mark.unit
    def test_preamble_postamble(self) -> None:
        text = (
            "Here is the transcription:\n"
            '{"transcription": "hello", "no_transcribable_text": false, '
            '"transcription_not_possible": false, "image_analysis": "text"}\n'
            "I hope this helps!"
        )
        result = _normalize_llm_text(text)
        assert result.startswith("{")
        assert json.loads(result)["transcription"] == "hello"

    @pytest.mark.unit
    def test_plain_text_unchanged(self) -> None:
        text = "The text on this page reads: Lorem ipsum"
        assert _normalize_llm_text(text) == text

    @pytest.mark.unit
    def test_empty_string(self) -> None:
        assert _normalize_llm_text("") == ""

    @pytest.mark.unit
    def test_whitespace_stripped(self) -> None:
        assert _normalize_llm_text("   \n\t  ") == ""


class TestExtractCodeFencedAndPreamble:
    """Integration tests: extract_transcribed_text with code-fenced and
    preamble-wrapped LLM responses (custom provider scenarios)."""

    @pytest.mark.unit
    def test_code_fenced_json_output_text(self) -> None:
        """Scenario A: Code-fenced JSON from custom provider via output_text."""
        data = {
            "output_text": (
                '```json\n{"transcription": "fenced text", '
                '"image_analysis": "...", "no_transcribable_text": false, '
                '"transcription_not_possible": false}\n```'
            )
        }
        assert extract_transcribed_text(data, "test.png") == "fenced text"

    @pytest.mark.unit
    def test_json_with_preamble_output_text(self) -> None:
        """Scenario B: JSON with conversational preamble/postamble."""
        data = {
            "output_text": (
                "Here is the transcription:\n"
                '{"transcription": "preamble text", "image_analysis": "...", '
                '"no_transcribable_text": false, "transcription_not_possible": false}\n'
                "I hope this helps!"
            )
        }
        assert extract_transcribed_text(data, "test.png") == "preamble text"

    @pytest.mark.unit
    def test_plain_text_fallback_output_text(self) -> None:
        """Scenario C: Plain text with no JSON at all."""
        data = {
            "output_text": "The text on this page reads:\nLorem ipsum dolor sit amet."
        }
        result = extract_transcribed_text(data, "test.png")
        assert "Lorem ipsum" in result
        assert "[transcription error]" not in result

    @pytest.mark.unit
    def test_code_fenced_no_transcribable_text_flag(self) -> None:
        """Code-fenced JSON with no_transcribable_text=true."""
        data = {
            "output_text": (
                '```json\n{"transcription": null, "image_analysis": "blank page",'
                ' "no_transcribable_text": true,'
                ' "transcription_not_possible": false}\n```'
            )
        }
        assert extract_transcribed_text(data, "test.png") == "[No transcribable text]"

    @pytest.mark.unit
    def test_code_fenced_chat_completions(self) -> None:
        """Code-fenced JSON in Chat Completions content field."""
        data = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '```json\n{"transcription": "chat fenced", '
                            '"image_analysis": "...", "no_transcribable_text": false, '
                            '"transcription_not_possible": false}\n```'
                        )
                    }
                }
            ]
        }
        assert extract_transcribed_text(data, "test.png") == "chat fenced"

    @pytest.mark.unit
    def test_preamble_chat_completions(self) -> None:
        """JSON with preamble in Chat Completions content field."""
        data = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Sure! Here you go:\n"
                            '{"transcription": "preamble chat", '
                            '"image_analysis": "...", "no_transcribable_text": false, '
                            '"transcription_not_possible": false}'
                        )
                    }
                }
            ]
        }
        assert extract_transcribed_text(data, "test.png") == "preamble chat"
