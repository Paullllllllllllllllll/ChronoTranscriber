"""Prompt templating: schema token or marker replacement and context injection.

``render_prompt_with_schema`` fills ``{{TRANSCRIPTION_SCHEMA}}``, replaces the
JSON after ``The JSON schema:`` or appends the schema. The injection function
fills ``{{ADDITIONAL_CONTEXT}}`` or drops its section when the context is
blank; ``prepare_prompt_with_context`` applies both.
"""

from __future__ import annotations

import pytest

from chronotranscriber.llm.prompt_utils import (
    inject_additional_context,
    prepare_prompt_with_context,
    render_prompt_with_schema,
)


class TestRenderPromptWithSchema:
    """Tests for render_prompt_with_schema function."""

    @pytest.mark.unit
    def test_replaces_token(self) -> None:
        """Test replacement of {{TRANSCRIPTION_SCHEMA}} token."""
        prompt = "Use this schema: {{TRANSCRIPTION_SCHEMA}}"
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}

        result = render_prompt_with_schema(prompt, schema)

        assert "{{TRANSCRIPTION_SCHEMA}}" not in result
        assert '"type": "object"' in result

    @pytest.mark.unit
    def test_replaces_marker_with_json(self) -> None:
        """Test replacement of JSON after 'The JSON schema:' marker."""
        prompt = 'Use this. The JSON schema:\n{"old": "schema"}'
        schema = {"new": "schema"}

        result = render_prompt_with_schema(prompt, schema)

        assert '"new": "schema"' in result
        assert '"old": "schema"' not in result

    @pytest.mark.unit
    def test_marker_block_ends_at_its_own_brace(self) -> None:
        """A closing brace in text after the schema block stays in place."""
        prompt = (
            'The JSON schema:\n{"old": {"nested": 1}}\n'
            "Write sums as {a+b}; never emit a lone }.\n"
            "{{ADDITIONAL_CONTEXT}}"
        )

        result = render_prompt_with_schema(prompt, {"new": "schema"})

        assert result.startswith('The JSON schema:\n{\n  "new": "schema"\n}\n')
        assert result.endswith(
            "Write sums as {a+b}; never emit a lone }.\n{{ADDITIONAL_CONTEXT}}"
        )
        assert '"old"' not in result

    @pytest.mark.unit
    def test_appends_when_no_marker(self) -> None:
        """Test that schema is appended when no token or marker exists."""
        prompt = "Transcribe the text from this image."
        schema = {"type": "object"}

        result = render_prompt_with_schema(prompt, schema)

        assert "The JSON schema:" in result
        assert '"type": "object"' in result

    @pytest.mark.unit
    def test_empty_schema(self) -> None:
        """Test with empty schema."""
        prompt = "{{TRANSCRIPTION_SCHEMA}}"
        schema = {}

        result = render_prompt_with_schema(prompt, schema)

        assert "{}" in result

    @pytest.mark.unit
    def test_marker_at_end(self) -> None:
        """Test marker at end of prompt."""
        prompt = "Instructions here. The JSON schema:"
        schema = {"type": "object"}

        result = render_prompt_with_schema(prompt, schema)

        assert "Instructions here." in result
        assert '"type": "object"' in result


class TestInjectAdditionalContext:
    """Tests for inject_additional_context function."""

    @pytest.mark.unit
    def test_replaces_marker(self) -> None:
        """Test replacement of {{ADDITIONAL_CONTEXT}} marker."""
        prompt = "Consider: {{ADDITIONAL_CONTEXT}}"
        context = "This is historical document from 1850."

        result = inject_additional_context(prompt, context)

        assert "{{ADDITIONAL_CONTEXT}}" not in result
        assert "historical document from 1850" in result

    @pytest.mark.unit
    def test_no_marker_unchanged(self) -> None:
        """Test that prompt without marker is unchanged."""
        prompt = "No marker in this prompt."
        context = "Some context"

        result = inject_additional_context(prompt, context)

        assert result == prompt

    @pytest.mark.unit
    @pytest.mark.parametrize("context", ["", None, "   \n\t  "])
    def test_blank_context_removes_section(self, context: str | None) -> None:
        """A blank or missing context removes the Additional context section."""
        prompt = "Additional context:\n{{ADDITIONAL_CONTEXT}}\n\nOther content."

        result = inject_additional_context(prompt, context)

        assert "{{ADDITIONAL_CONTEXT}}" not in result
        assert "Additional context:" not in result
        assert "Other content." in result

    @pytest.mark.unit
    def test_context_trimmed(self) -> None:
        """Test that context whitespace is trimmed."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        context = "  trimmed content  "

        result = inject_additional_context(prompt, context)

        assert "trimmed content" in result
        assert "  trimmed" not in result

    @pytest.mark.unit
    def test_multiple_markers_all_replaced(self) -> None:
        """Test that multiple markers are all replaced."""
        prompt = "First: {{ADDITIONAL_CONTEXT}} Second: {{ADDITIONAL_CONTEXT}}"
        context = "context value"

        result = inject_additional_context(prompt, context)

        assert result.count("context value") == 2
        assert "{{ADDITIONAL_CONTEXT}}" not in result

    @pytest.mark.unit
    def test_multiline_context(self) -> None:
        """Test with multiline context."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        context = "Line 1\nLine 2\nLine 3"

        result = inject_additional_context(prompt, context)

        assert "Line 1\nLine 2\nLine 3" in result


class TestPreparePromptWithContext:
    """Tests for prepare_prompt_with_context function."""

    @pytest.mark.unit
    def test_combines_schema_and_context(self) -> None:
        """Test that schema and context are both applied."""
        prompt = (
            "{{TRANSCRIPTION_SCHEMA}}\n\nAdditional context:\n{{ADDITIONAL_CONTEXT}}"
        )
        schema = {"type": "object"}
        context = "Historical document from 1850."

        result = prepare_prompt_with_context(prompt, schema, context)

        assert '"type": "object"' in result
        assert "Historical document from 1850." in result

    @pytest.mark.unit
    def test_schema_only(self) -> None:
        """Test with schema but no context."""
        prompt = (
            "{{TRANSCRIPTION_SCHEMA}}\n\nAdditional context:\n{{ADDITIONAL_CONTEXT}}"
        )
        schema = {"type": "object"}

        result = prepare_prompt_with_context(prompt, schema, None)

        assert '"type": "object"' in result
        assert "Additional context:" not in result

    @pytest.mark.unit
    def test_context_only(self) -> None:
        """Test with context but no schema."""
        prompt = "Instructions. {{ADDITIONAL_CONTEXT}}"
        context = "Some context"

        result = prepare_prompt_with_context(prompt, None, context)

        assert "Instructions." in result
        assert "Some context" in result

    @pytest.mark.unit
    def test_neither_schema_nor_context(self) -> None:
        """Test with neither schema nor context."""
        prompt = "Plain prompt with {{ADDITIONAL_CONTEXT}} marker."

        result = prepare_prompt_with_context(prompt, None, None)

        assert "Plain prompt with" in result
        assert "{{ADDITIONAL_CONTEXT}}" not in result
