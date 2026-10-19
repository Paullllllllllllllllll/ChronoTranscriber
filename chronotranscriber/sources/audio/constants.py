"""Format tables, provider limits and sentinels of the audio source.

The extension-to-MIME table used for inline audio parts, the per-provider
upload ceilings and container allow-lists, the minimum chunk length and
the placeholder an empty transcript collapses to.
"""

from __future__ import annotations

# The per-provider allow-lists below are subsets of these keys.
SUPPORTED_AUDIO_FORMATS: dict[str, str] = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mpga": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
}

SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(SUPPORTED_AUDIO_FORMATS)

# Hard per-request payload ceilings. OpenAI rejects uploads above 25 MB on
# /v1/audio/transcriptions; Gemini caps an inline audio part at about 20 MB.
OPENAI_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024
GEMINI_INLINE_LIMIT_BYTES = 20 * 1024 * 1024

OPENAI_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
)
GEMINI_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".aiff", ".aac", ".ogg", ".flac"}
)

# Sentinel for "the model returned nothing". It must stay byte-identical to
# the Tesseract placeholder in chronotranscriber/images/tesseract_runtime,
# since chronotranscriber/llm/response_parsing detects both with one regex.
NO_TRANSCRIBABLE_TEXT = "[No transcribable text]"

DEFAULT_MIN_CHUNK_SECONDS = 30
"""The shortest chunk the planner cuts, in seconds."""

__all__ = [
    "DEFAULT_MIN_CHUNK_SECONDS",
    "GEMINI_ALLOWED_EXTENSIONS",
    "GEMINI_INLINE_LIMIT_BYTES",
    "NO_TRANSCRIBABLE_TEXT",
    "OPENAI_ALLOWED_EXTENSIONS",
    "OPENAI_UPLOAD_LIMIT_BYTES",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SUPPORTED_AUDIO_FORMATS",
]
