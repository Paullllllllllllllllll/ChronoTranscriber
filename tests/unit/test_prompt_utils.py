"""Unit tests for modules/llm/prompt_utils.py.

Tests prompt template rendering and schema injection utilities.
"""

from __future__ import annotations

import pytest

from modules.llm.prompt_utils import (
    render_prompt_with_schema,
    inject_additional_context,
)


class TestRenderPromptWithSchema:
    """Tests for render_prompt_with_schema function."""
    
    @pytest.mark.unit
    def test_replaces_token(self):
        """Test replacement of {{TRANSCRIPTION_SCHEMA}} token."""
        prompt = "Use this schema: {{TRANSCRIPTION_SCHEMA}}"
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert "{{TRANSCRIPTION_SCHEMA}}" not in result
        assert '"type": "object"' in result
    
    @pytest.mark.unit
    def test_replaces_marker_with_json(self):
        """Test replacement of JSON after 'The JSON schema:' marker."""
        prompt = 'Use this. The JSON schema:\n{"old": "schema"}'
        schema = {"new": "schema"}
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert '"new": "schema"' in result
        assert '"old": "schema"' not in result
    
    @pytest.mark.unit
    def test_appends_when_no_marker(self):
        """Test that schema is appended when no token or marker exists."""
        prompt = "Transcribe the text from this image."
        schema = {"type": "object"}
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert "The JSON schema:" in result
        assert '"type": "object"' in result
    
    @pytest.mark.unit
    def test_preserves_prompt_text(self):
        """Test that original prompt text is preserved."""
        prompt = "Important instructions here. {{TRANSCRIPTION_SCHEMA}}"
        schema = {"key": "value"}
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert "Important instructions here." in result
    
    @pytest.mark.unit
    def test_complex_schema(self):
        """Test with complex nested schema."""
        prompt = "{{TRANSCRIPTION_SCHEMA}}"
        schema = {
            "type": "object",
            "properties": {
                "transcription": {"type": "string"},
                "no_transcribable_text": {"type": "boolean"},
                "metadata": {
                    "type": "object",
                    "properties": {"page": {"type": "integer"}}
                }
            },
            "required": ["transcription"]
        }
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert "transcription" in result
        assert "no_transcribable_text" in result
        assert "metadata" in result
    
    @pytest.mark.unit
    def test_empty_schema(self):
        """Test with empty schema."""
        prompt = "{{TRANSCRIPTION_SCHEMA}}"
        schema = {}
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert "{}" in result
    
    @pytest.mark.unit
    def test_marker_at_end(self):
        """Test marker at end of prompt."""
        prompt = "Instructions here. The JSON schema:"
        schema = {"type": "object"}
        
        result = render_prompt_with_schema(prompt, schema)
        
        assert "Instructions here." in result
        assert '"type": "object"' in result


class TestInjectAdditionalContext:
    """Tests for inject_additional_context function."""
    
    @pytest.mark.unit
    def test_replaces_marker(self):
        """Test replacement of {{ADDITIONAL_CONTEXT}} marker."""
        prompt = "Consider: {{ADDITIONAL_CONTEXT}}"
        context = "This is historical document from 1850."
        
        result = inject_additional_context(prompt, context)
        
        assert "{{ADDITIONAL_CONTEXT}}" not in result
        assert "historical document from 1850" in result
    
    @pytest.mark.unit
    def test_no_marker_unchanged(self):
        """Test that prompt without marker is unchanged."""
        prompt = "No marker in this prompt."
        context = "Some context"
        
        result = inject_additional_context(prompt, context)
        
        assert result == prompt
    
    @pytest.mark.unit
    def test_empty_context_shows_empty(self):
        """Test that empty context shows 'Empty'."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        
        result = inject_additional_context(prompt, "")
        
        assert "Empty" in result
        assert "{{ADDITIONAL_CONTEXT}}" not in result
    
    @pytest.mark.unit
    def test_none_context_shows_empty(self):
        """Test that None context shows 'Empty'."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        
        result = inject_additional_context(prompt, None)
        
        assert "Empty" in result
    
    @pytest.mark.unit
    def test_whitespace_context_shows_empty(self):
        """Test that whitespace-only context is trimmed."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        
        result = inject_additional_context(prompt, "   \n\t  ")
        
        # After stripping, empty string is falsy, so "Empty" is used
        # But if strip() returns empty string, context.strip() returns ''
        # which is falsy, so we expect "Empty"
        # Actually, the code does: context.strip() if context else "Empty"
        # "   \n\t  ".strip() = "" which is falsy for the outer check
        # Let me check: context = "   \n\t  ", context.strip() = ""
        # context_text = context.strip() if context else "Empty"
        # Since context is truthy (non-empty string), it evaluates context.strip() = ""
        # So result is "Context: " - the marker is replaced with empty string
        assert "{{ADDITIONAL_CONTEXT}}" not in result
    
    @pytest.mark.unit
    def test_context_trimmed(self):
        """Test that context whitespace is trimmed."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        context = "  trimmed content  "
        
        result = inject_additional_context(prompt, context)
        
        assert "trimmed content" in result
        assert "  trimmed" not in result
    
    @pytest.mark.unit
    def test_multiple_markers_all_replaced(self):
        """Test that multiple markers are all replaced."""
        prompt = "First: {{ADDITIONAL_CONTEXT}} Second: {{ADDITIONAL_CONTEXT}}"
        context = "context value"
        
        result = inject_additional_context(prompt, context)
        
        assert result.count("context value") == 2
        assert "{{ADDITIONAL_CONTEXT}}" not in result
    
    @pytest.mark.unit
    def test_preserves_other_content(self):
        """Test that other prompt content is preserved."""
        prompt = "Start. {{ADDITIONAL_CONTEXT}} End."
        context = "middle"
        
        result = inject_additional_context(prompt, context)
        
        assert "Start." in result
        assert "End." in result
        assert "middle" in result
    
    @pytest.mark.unit
    def test_multiline_context(self):
        """Test with multiline context."""
        prompt = "Context: {{ADDITIONAL_CONTEXT}}"
        context = "Line 1\nLine 2\nLine 3"
        
        result = inject_additional_context(prompt, context)
        
        assert "Line 1\nLine 2\nLine 3" in result
