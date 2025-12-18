from __future__ import annotations

from typing import Any, Dict, Iterable


def validate_transcription_output(output: Dict[str, Any], schema_wrapper: Dict[str, Any]) -> None:
    """
    Validate a transcription output against the schema.
    
    Args:
        output: The output dictionary to validate.
        schema_wrapper: The schema wrapper containing the JSON schema.
        
    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(output, dict):
        raise ValueError(f"Output must be a JSON object (got {type(output).__name__})")

    root_schema = schema_wrapper.get("schema") if isinstance(schema_wrapper, dict) else schema_wrapper
    if not isinstance(root_schema, dict):
        return

    required: Iterable[str] = root_schema.get("required") or []
    if not isinstance(required, list):
        required = []

    for key in required:
        if key not in output:
            raise ValueError(f"Missing required top-level key: {key}")

    properties = root_schema.get("properties")
    additional_properties = root_schema.get("additionalProperties", True)
    if additional_properties is False and isinstance(properties, dict):
        allowed = set(properties.keys())
        extra = sorted(set(output.keys()) - allowed)
        if extra:
            raise ValueError(f"Unexpected top-level key(s): {', '.join(extra)}")

    # Validate transcription-specific fields
    if "transcription" in output:
        val = output.get("transcription")
        if val is not None and not isinstance(val, str):
            raise ValueError(
                f"'transcription' must be a string or null (got {type(val).__name__})"
            )

    if "no_transcribable_text" in output:
        val = output.get("no_transcribable_text")
        if not isinstance(val, bool):
            raise ValueError(
                f"'no_transcribable_text' must be a boolean (got {type(val).__name__})"
            )

    if "transcription_not_possible" in output:
        val = output.get("transcription_not_possible")
        if not isinstance(val, bool):
            raise ValueError(
                f"'transcription_not_possible' must be a boolean (got {type(val).__name__})"
            )
