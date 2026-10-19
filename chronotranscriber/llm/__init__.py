"""Language model calls: client factory, structured calls, page transcription.

Supports OpenAI, Anthropic, Google, OpenRouter and named custom endpoints
through LangChain. Prompt inputs live in :mod:`chronotranscriber.llm.prompts`,
response parsing in :mod:`chronotranscriber.llm.response_parsing`.
"""

from chronotranscriber.llm.prompts import list_schema_options
from chronotranscriber.llm.transcription import (
    PageTranscriber,
    transcribe_image_with_llm,
)

__all__ = [
    "PageTranscriber",
    "list_schema_options",
    "transcribe_image_with_llm",
]
