"""Prompt template rendering utilities.

Provides functions for rendering prompt templates with schema information
for transcription tasks.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def render_prompt_with_schema(prompt_text: str, schema_obj: Dict[str, Any]) -> str:
    """
    Inject a JSON schema into a prompt using one of three strategies:
    - If the token "{{TRANSCRIPTION_SCHEMA}}" exists, replace it with a pretty-printed schema.
    - If the marker "The JSON schema:" exists, replace any parsable JSON object that follows
      the marker with the new schema; if no parsable block is found, append the schema after the marker.
    - If neither token nor marker exists, append a new section "The JSON schema:" followed by the schema.
    """
    try:
        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
    except Exception:
        schema_str = str(schema_obj)

    token = "{{TRANSCRIPTION_SCHEMA}}"
    if token in prompt_text:
        return prompt_text.replace(token, schema_str)

    marker = "The JSON schema:"
    if marker in prompt_text:
        # Find the marker and attempt to locate a JSON object after it to replace
        idx = prompt_text.find(marker)
        # Locate the first opening brace after the marker
        start_brace = prompt_text.find("{", idx)
        if start_brace != -1:
            # Heuristic: find the last closing brace in the document; ensures we capture a block
            end_brace = prompt_text.rfind("}")
            if end_brace != -1 and end_brace > start_brace:
                return (
                    prompt_text[:start_brace]
                    + schema_str
                    + prompt_text[end_brace + 1 :]
                )
        # Fallback: append schema right after the marker
        return prompt_text + "\n" + schema_str

    # No token or marker found: append a new section at the end
    return prompt_text + "\n\nThe JSON schema:\n" + schema_str


def inject_additional_context(prompt_text: str, context: str) -> str:
    """
    Inject additional context into a prompt using the {{ADDITIONAL_CONTEXT}} marker.
    
    If the marker exists, it is replaced with the provided context.
    If the marker does not exist, the prompt is returned unchanged (fail-safe behavior).
    If context is empty or None, the marker is replaced with "Empty".
    
    Parameters
    ----------
    prompt_text : str
        The prompt template text containing the marker
    context : str
        The additional context to inject
        
    Returns
    -------
    str
        The prompt with context injected or marker removed
    """
    marker = "{{ADDITIONAL_CONTEXT}}"
    if marker not in prompt_text:
        return prompt_text
    
    # Replace marker with context (or "Empty" if no context provided)
    context_text = context.strip() if context else "Empty"
    return prompt_text.replace(marker, context_text)
