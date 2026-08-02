"""Format tables, provider limits, and sentinels for the audio pipeline.

Single source of truth for what the audio workflow accepts and what each
remote venue will take: the extension-to-MIME table used when building
inline audio parts, the per-provider upload ceilings and container
allow-lists, and the placeholder an empty transcript collapses to.
"""

from __future__ import annotations

# Extension -> MIME type for every container the audio workflow will open.
# The per-provider allow-lists below are strict subsets of these keys.
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
# /v1/audio/transcriptions; Gemini caps an INLINE audio part at ~20 MB (larger
# recordings need the Files API, which the v1 audio backend does not use).
OPENAI_UPLOAD_LIMIT_BYTES = 25 * 1024 * 1024
GEMINI_INLINE_LIMIT_BYTES = 20 * 1024 * 1024

# Containers each venue documents as accepted.
OPENAI_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
)
GEMINI_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".aiff", ".aac", ".ogg", ".flac"}
)

# Models that take ``languages[]`` / ``keywords[]`` list parameters. These are
# not in the openai SDK's ``transcriptions.create`` signature, so the backend
# sends them via ``extra_body``; every other model uses the scalar ``language``.
OPENAI_PLURAL_PARAM_MODELS: tuple[str, ...] = ("gpt-transcribe",)

# whisper-1 silently truncates its prompt at 224 tokens; the backend trims
# ahead of the call so the discarded tail is visible in the log instead.
WHISPER1_PROMPT_TOKEN_CAP = 224

# Sentinel for "the model returned nothing". Must stay byte-identical to the
# Tesseract placeholder in modules/images/tesseract_runtime.perform_ocr, since
# modules/llm/response_parsing detects both through the same regex.
NO_TRANSCRIBABLE_TEXT = "[No transcribable text]"

# Chunk-planning defaults, overridable through chunking.target_seconds.
DEFAULT_TARGET_CHUNK_SECONDS = 600
DEFAULT_MIN_CHUNK_SECONDS = 30

__all__ = [
    "DEFAULT_MIN_CHUNK_SECONDS",
    "DEFAULT_TARGET_CHUNK_SECONDS",
    "GEMINI_ALLOWED_EXTENSIONS",
    "GEMINI_INLINE_LIMIT_BYTES",
    "NO_TRANSCRIBABLE_TEXT",
    "OPENAI_ALLOWED_EXTENSIONS",
    "OPENAI_PLURAL_PARAM_MODELS",
    "OPENAI_UPLOAD_LIMIT_BYTES",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SUPPORTED_AUDIO_FORMATS",
    "WHISPER1_PROMPT_TOKEN_CAP",
]
