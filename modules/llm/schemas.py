"""Pydantic schemas for structured LLM outputs.

LangChain's with_structured_output() works best with Pydantic models,
which provide automatic validation and parsing. This module defines
the transcription schema as a Pydantic model.

Note:
    The JSON schema files in /schemas/ are still supported for backward
    compatibility and batch processing. This module provides Pydantic
    equivalents for LangChain integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field


class TranscriptionOutput(BaseModel):
    """Pydantic model for transcription output.
    
    This model matches the structure of markdown_transcription_schema.json
    and enables automatic validation with LangChain's with_structured_output().
    """
    
    image_analysis: str = Field(
        description=(
            "Return a brief summary of what you can see on the image, "
            "listing all formatting elements and visible text. "
            "Identify the position of page number(s) and footnote(s) on the image precisely."
        )
    )
    
    transcription: Optional[str] = Field(
        default=None,
        description=(
            "- Task: Return a **markdown-converted, verbatim transcription** of the provided image.\n"
            "- Scope — include everything: Transcribe **all** visible text and structural elements: "
            "headers; footers; page numbers; tables; etc.\n"
            "- Markdown formatting: Use markdown formatting for transcribing text, "
            "e.g., `*italic*` or `**bold**`; `# Heading 1`, `## Heading 2`, etc.\n"
            "- Equations: Convert any equations to **LaTeX** and enclose them in `$$ … $$`.\n"
            "- Page numbers — exact format: Write page numbers appearing on headers and footers "
            "(not in the main text or tables) like this example: `<page_number>9<page_number>`.\n"
            "- Footnotes: Format footnotes like in this example: `[^1]: My reference`.\n"
            "- Images or diagrams: Indicate their presence with a brief description in square brackets, "
            "e.g., `[Image: diagram of a cell]`.\n"
            "- Layout preservation: **Preserve all line breaks** exactly as in the image.\n"
            "- Multi-column text: Transcribe column by column (finish column 1 before column 2, etc.), "
            "keep line breaks within columns intact.\n"
            "- Cross-page continuity: This is likely one page of a multi-page document. "
            "**Preserve any sentences that continue from the previous or to the next page exactly as shown.**\n"
            "- Output format: Provide **only** the transcription. If **no transcribable text** is found, output `null`.\n"
            "- No fabrication: **Do not hallucinate.** Keep spelling, punctuation, and wording **verbatim**."
        )
    )
    
    no_transcribable_text: bool = Field(
        default=False,
        description="True if the image contains no transcribable text; otherwise, false."
    )
    
    transcription_not_possible: bool = Field(
        default=False,
        description="True *if and only if* transcription is entirely impossible; otherwise, false."
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()


def load_schema_as_pydantic(schema_path: Path) -> Type[BaseModel]:
    """Load a JSON schema and return the corresponding Pydantic model.
    
    For now, this returns TranscriptionOutput for the standard schema.
    Future versions could dynamically generate Pydantic models from JSON schemas.
    
    Args:
        schema_path: Path to the JSON schema file
    
    Returns:
        Pydantic model class
    """
    # For the standard transcription schema, return our pre-defined model
    if "markdown_transcription" in str(schema_path) or "transcription" in str(schema_path):
        return TranscriptionOutput
    
    # For unknown schemas, return the generic TranscriptionOutput
    # In the future, we could use pydantic's create_model to dynamically generate
    return TranscriptionOutput


def json_schema_to_pydantic_compatible(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON schema to a format compatible with LangChain's with_structured_output.
    
    LangChain expects the schema in a specific format. This function
    unwraps our schema format if needed.
    
    Args:
        schema: JSON schema dict (possibly wrapped in {name, strict, schema})
    
    Returns:
        Unwrapped JSON schema dict
    """
    if isinstance(schema, dict) and "schema" in schema:
        inner = schema["schema"]
        return inner if isinstance(inner, dict) else schema
    return schema
