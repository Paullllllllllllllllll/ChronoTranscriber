"""Language model integration package.

Provides OpenAI API integration, batch processing, schema utilities, and model capabilities.
"""

# Avoid circular imports - use direct imports instead of re-exporting
__all__ = [
    "open_transcriber",
    "transcribe_image_with_openai",
    "OpenAITranscriber",
    "list_schema_options",
    "detect_capabilities",
    "ensure_image_support",
]
