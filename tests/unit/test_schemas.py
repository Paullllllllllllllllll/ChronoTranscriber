"""Unit tests for modules/llm/schemas.py."""

from __future__ import annotations

import pytest
from pathlib import Path


class TestTranscriptionOutput:
    """Tests for TranscriptionOutput Pydantic model."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default values for TranscriptionOutput."""
        from modules.llm.schemas import TranscriptionOutput
        
        output = TranscriptionOutput(image_analysis="Test analysis")
        
        assert output.image_analysis == "Test analysis"
        assert output.transcription is None
        assert output.no_transcribable_text is False
        assert output.transcription_not_possible is False

    @pytest.mark.unit
    def test_full_initialization(self):
        """Test full initialization with all fields."""
        from modules.llm.schemas import TranscriptionOutput
        
        output = TranscriptionOutput(
            image_analysis="Detailed analysis",
            transcription="Transcribed text here",
            no_transcribable_text=False,
            transcription_not_possible=False,
        )
        
        assert output.image_analysis == "Detailed analysis"
        assert output.transcription == "Transcribed text here"
        assert output.no_transcribable_text is False
        assert output.transcription_not_possible is False

    @pytest.mark.unit
    def test_no_text_scenario(self):
        """Test scenario where no transcribable text is found."""
        from modules.llm.schemas import TranscriptionOutput
        
        output = TranscriptionOutput(
            image_analysis="Image appears to be blank",
            transcription=None,
            no_transcribable_text=True,
            transcription_not_possible=False,
        )
        
        assert output.transcription is None
        assert output.no_transcribable_text is True

    @pytest.mark.unit
    def test_not_possible_scenario(self):
        """Test scenario where transcription is not possible."""
        from modules.llm.schemas import TranscriptionOutput
        
        output = TranscriptionOutput(
            image_analysis="Image is corrupted or unreadable",
            transcription=None,
            no_transcribable_text=False,
            transcription_not_possible=True,
        )
        
        assert output.transcription is None
        assert output.transcription_not_possible is True

    @pytest.mark.unit
    def test_to_dict_method(self):
        """Test to_dict method returns proper dictionary."""
        from modules.llm.schemas import TranscriptionOutput
        
        output = TranscriptionOutput(
            image_analysis="Test",
            transcription="Some text",
        )
        
        result = output.to_dict()
        
        assert isinstance(result, dict)
        assert result["image_analysis"] == "Test"
        assert result["transcription"] == "Some text"
        assert result["no_transcribable_text"] is False
        assert result["transcription_not_possible"] is False

    @pytest.mark.unit
    def test_to_json_method(self):
        """Test to_json method returns valid JSON string."""
        from modules.llm.schemas import TranscriptionOutput
        import json
        
        output = TranscriptionOutput(
            image_analysis="Test",
            transcription="Some text",
        )
        
        result = output.to_json()
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["image_analysis"] == "Test"
        assert parsed["transcription"] == "Some text"


class TestLoadSchemaAsPydantic:
    """Tests for load_schema_as_pydantic function."""

    @pytest.mark.unit
    def test_returns_transcription_output_for_markdown_schema(self):
        """Test returns TranscriptionOutput for markdown transcription schema."""
        from modules.llm.schemas import load_schema_as_pydantic, TranscriptionOutput
        
        schema_path = Path("schemas/markdown_transcription_schema.json")
        result = load_schema_as_pydantic(schema_path)
        
        assert result is TranscriptionOutput

    @pytest.mark.unit
    def test_returns_transcription_output_for_transcription_schema(self):
        """Test returns TranscriptionOutput for any transcription schema."""
        from modules.llm.schemas import load_schema_as_pydantic, TranscriptionOutput
        
        schema_path = Path("schemas/plain_text_transcription_schema.json")
        result = load_schema_as_pydantic(schema_path)
        
        assert result is TranscriptionOutput

    @pytest.mark.unit
    def test_returns_transcription_output_for_unknown_schema(self):
        """Test returns TranscriptionOutput for unknown schema (fallback)."""
        from modules.llm.schemas import load_schema_as_pydantic, TranscriptionOutput
        
        schema_path = Path("schemas/unknown_schema.json")
        result = load_schema_as_pydantic(schema_path)
        
        assert result is TranscriptionOutput


class TestJsonSchemaToPydanticCompatible:
    """Tests for json_schema_to_pydantic_compatible function."""

    @pytest.mark.unit
    def test_unwraps_wrapped_schema(self):
        """Test unwrapping schema from {name, strict, schema} format."""
        from modules.llm.schemas import json_schema_to_pydantic_compatible
        
        wrapped_schema = {
            "name": "test_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"field": {"type": "string"}},
            }
        }
        
        result = json_schema_to_pydantic_compatible(wrapped_schema)
        
        assert result == {
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }

    @pytest.mark.unit
    def test_returns_unwrapped_schema_unchanged(self):
        """Test that already unwrapped schema is returned unchanged."""
        from modules.llm.schemas import json_schema_to_pydantic_compatible
        
        raw_schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }
        
        result = json_schema_to_pydantic_compatible(raw_schema)
        
        assert result == raw_schema

    @pytest.mark.unit
    def test_handles_empty_dict(self):
        """Test handling of empty dictionary."""
        from modules.llm.schemas import json_schema_to_pydantic_compatible
        
        result = json_schema_to_pydantic_compatible({})
        
        assert result == {}


class TestTranscriptionOutputFieldDescriptions:
    """Tests for field descriptions in TranscriptionOutput."""

    @pytest.mark.unit
    def test_image_analysis_has_description(self):
        """Test that image_analysis field has description."""
        from modules.llm.schemas import TranscriptionOutput
        
        schema = TranscriptionOutput.model_json_schema()
        
        assert "image_analysis" in schema["properties"]
        assert "description" in schema["properties"]["image_analysis"]

    @pytest.mark.unit
    def test_transcription_has_description(self):
        """Test that transcription field has description."""
        from modules.llm.schemas import TranscriptionOutput
        
        schema = TranscriptionOutput.model_json_schema()
        
        assert "transcription" in schema["properties"]

    @pytest.mark.unit
    def test_no_transcribable_text_has_description(self):
        """Test that no_transcribable_text field has description."""
        from modules.llm.schemas import TranscriptionOutput
        
        schema = TranscriptionOutput.model_json_schema()
        
        assert "no_transcribable_text" in schema["properties"]
        assert "description" in schema["properties"]["no_transcribable_text"]

    @pytest.mark.unit
    def test_transcription_not_possible_has_description(self):
        """Test that transcription_not_possible field has description."""
        from modules.llm.schemas import TranscriptionOutput
        
        schema = TranscriptionOutput.model_json_schema()
        
        assert "transcription_not_possible" in schema["properties"]
        assert "description" in schema["properties"]["transcription_not_possible"]
