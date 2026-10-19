"""Text extraction from model answers in chronotranscriber.llm.response_parsing.

``extract_transcribed_text`` turns a provider response dict into page text, a
flag placeholder or the error placeholder. Without structured outputs a model
may answer in prose, wrap its JSON in fences, concatenate objects or bend the
schema; each shape yields text or a placeholder, never an exception. The
helpers cover placeholder detection, page lines, fence stripping, JSON
isolation and bounded-time salvage of pathological input.
"""

from __future__ import annotations

import json
import time

import pytest

from chronotranscriber.llm.response_parsing import (
    _check_transcription_flags,
    _salvage_last_json_object,
    _strip_code_fences,
    _try_parse_json,
    detect_transcription_cause,
    extract_transcribed_text,
    format_page_line,
    normalize_llm_text,
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
    def test_with_page_number_fallback(self) -> None:
        """A placeholder without image name is headed by the page number."""
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
    def test_schema_object_null_transcription_returns_empty(self) -> None:
        """A null transcription yields "" rather than the string "None"."""
        data = {
            "transcription": None,
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }
        assert extract_transcribed_text(data) == ""

    @pytest.mark.unit
    def test_responses_output_text_null_transcription_returns_empty(self) -> None:
        """A Responses output_text JSON with a null transcription yields ""."""
        data = {
            "output_text": json.dumps(
                {
                    "transcription": None,
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                }
            )
        }
        assert extract_transcribed_text(data) == ""

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
    @pytest.mark.parametrize("stored", [False, True])
    def test_answer_with_the_retired_image_analysis_field(self, stored: bool) -> None:
        """Answers stored under earlier schemas still carry image_analysis."""
        answer = {
            "image_analysis": "Single column; page number in the footer.",
            "transcription": "Old text",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }
        data = {"output_text": json.dumps(answer)} if stored else answer
        assert extract_transcribed_text(data) == "Old text"

    @pytest.mark.unit
    def test_chat_completions_payload_is_unrecognized(self) -> None:
        """No current path produces Chat Completions payloads; they are errors."""
        data = {
            "choices": [{"message": {"content": json.dumps({"transcription": "x"})}}]
        }
        assert extract_transcribed_text(data) == "[transcription error]"

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
        with caplog.at_level(
            logging.WARNING, logger="chronotranscriber.llm.response_parsing"
        ):
            extract_transcribed_text(data, "page_001.jpg")
        assert "Connection error." in caplog.text


class TestCheckTranscriptionFlags:
    """Tests for _check_transcription_flags helper."""

    @pytest.mark.unit
    def test_no_transcribable_text_takes_precedence(self) -> None:
        result = _check_transcription_flags(
            {
                "no_transcribable_text": True,
                "transcription_not_possible": True,
            }
        )
        assert result == "[No transcribable text]"


class TestSalvageLastJsonObject:
    """_salvage_last_json_object and _try_parse_json on pathological input.

    Many unbalanced or deeply nested braces degrade to None in bounded time
    instead of backtracking quadratically or raising RecursionError.
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
        # Must not crash. On CPython 3.13 the C scanner exhausts its recursion
        # budget at this depth and the helper degrades to None; on 3.14+ the
        # adaptive recursion limit parses the document, so a dict is equally
        # acceptable.
        result = _try_parse_json(deep)
        assert result is None or isinstance(result, dict)

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
    @pytest.mark.parametrize("opening", ["```json", "```"])
    def test_leading_code_fence_stripped(self, opening: str) -> None:
        text = opening + '\n{"transcription": "hello"}\n```'
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
        # normalize_llm_text's brace isolation (covered below).
        text = 'Here is the result:\n```json\n{"transcription": "hello"}\n```\nDone!'
        assert _strip_code_fences(text) == text

    @pytest.mark.unit
    def test_plain_text_with_embedded_fence_preserved(self) -> None:
        # A plain-text transcription that contains a fenced block (e.g.
        # tabular content) is not reduced to the fence content.
        text = "Chapter I opens here.\n```\ncol1 col2\n```\nProse continues."
        assert _strip_code_fences(text) == text

    @pytest.mark.unit
    def test_fully_wrapped_json_with_inner_fence_keeps_inner(self) -> None:
        # A JSON answer fully wrapped in ```json ... ``` whose transcription
        # string contains a fenced block loses only the outermost fence.
        text = (
            "```json\n"
            '{"transcription": "Here is code:\n'
            "```python\n"
            "x = 1\n"
            "```\n"
            'done"}\n'
            "```"
        )
        result = _strip_code_fences(text)
        assert result.startswith("{")
        assert result.endswith("}")
        assert "```python" in result
        assert "x = 1" in result
        assert result != 'done"}'


class TestNormalizeLlmText:
    """Tests for normalize_llm_text helper."""

    @pytest.mark.unit
    def test_clean_json_unchanged(self) -> None:
        text = '{"transcription": "hello"}'
        assert normalize_llm_text(text) == text

    @pytest.mark.unit
    def test_code_fenced_json(self) -> None:
        text = '```json\n{"transcription": "hello"}\n```'
        result = normalize_llm_text(text)
        assert result.startswith("{")
        assert json.loads(result)["transcription"] == "hello"

    @pytest.mark.unit
    def test_prose_wrapped_fenced_json_still_extracted(self) -> None:
        # JSON wrapped in prose plus a fence is isolated by the brace scan.
        text = 'Here is the result:\n```json\n{"transcription": "hello"}\n```\nDone!'
        result = normalize_llm_text(text)
        assert json.loads(result)["transcription"] == "hello"

    @pytest.mark.unit
    def test_plain_text_with_embedded_fence_not_reduced(self) -> None:
        text = "Chapter I opens here.\n```\ncol1 col2\n```\nProse continues."
        assert normalize_llm_text(text) == text

    @pytest.mark.unit
    def test_preamble_postamble(self) -> None:
        text = (
            "Here is the transcription:\n"
            '{"transcription": "hello", "no_transcribable_text": false, '
            '"transcription_not_possible": false, "image_analysis": "text"}\n'
            "I hope this helps!"
        )
        result = normalize_llm_text(text)
        assert result.startswith("{")
        assert json.loads(result)["transcription"] == "hello"

    @pytest.mark.unit
    def test_plain_text_unchanged(self) -> None:
        text = "The text on this page reads: Lorem ipsum"
        assert normalize_llm_text(text) == text

    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["", "   \n\t  "])
    def test_blank_text_normalizes_to_empty(self, text: str) -> None:
        assert normalize_llm_text(text) == ""


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


def _answer(text: str) -> dict[str, str]:
    """Wrap a raw model answer the way the provider response dict carries it."""
    return {"output_text": text}


class TestLooselyShapedAnswers:
    """Answers that bend the schema yield text or a placeholder, not an error."""

    @pytest.mark.unit
    def test_plain_text_where_object_expected(self) -> None:
        text = "Erstes Capitel.\nVon der Zubereitung der Suppen."
        assert extract_transcribed_text(_answer(text), "p.jpg") == text

    @pytest.mark.unit
    def test_list_where_string_expected(self) -> None:
        answer = json.dumps({"transcription": ["Line one", "Line two"]})
        result = extract_transcribed_text(_answer(answer), "p.jpg")
        assert "Line one" in result
        assert "Line two" in result
        assert "[transcription error" not in result

    @pytest.mark.unit
    def test_list_where_string_expected_in_schema_object(self) -> None:
        result = extract_transcribed_text({"transcription": ["Line one", "Line two"]})
        assert "Line one" in result
        assert "Line two" in result

    @pytest.mark.unit
    def test_json_array_where_object_expected(self) -> None:
        answer = '["Line one", "Line two"]'
        assert extract_transcribed_text(_answer(answer), "p.jpg") == answer

    @pytest.mark.unit
    def test_object_without_transcription_key_returns_text(self) -> None:
        answer = json.dumps({"text": "Line one"})
        assert extract_transcribed_text(_answer(answer), "p.jpg") == answer

    @pytest.mark.unit
    def test_salvages_object_after_invalid_fenced_block(self) -> None:
        answer = '```json\n{not json}\n```\n{"transcription": "kept"}'
        assert extract_transcribed_text(_answer(answer), "p.jpg") == "kept"

    @pytest.mark.unit
    def test_truncated_json_returns_text(self) -> None:
        answer = '{"transcription": "cut off mid'
        assert extract_transcribed_text(_answer(answer), "p.jpg") == answer

    @pytest.mark.unit
    def test_transcription_not_possible_flag_in_answer(self) -> None:
        answer = json.dumps({"transcription": None, "transcription_not_possible": True})
        result = extract_transcribed_text(_answer(answer), "p.jpg")
        assert result == "[Transcription not possible]"

    @pytest.mark.unit
    def test_flag_in_fenced_answer_with_prose(self) -> None:
        answer = (
            "The page is blank.\n```json\n"
            '{"transcription": "", "no_transcribable_text": true}\n```'
        )
        result = extract_transcribed_text(_answer(answer), "p.jpg")
        assert result == "[No transcribable text]"
