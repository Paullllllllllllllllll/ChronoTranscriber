"""Text post-processing and output writing for transcription results.

Two submodules:
- ``text`` — Unicode normalization, hyphenation, whitespace, line wrapping.
- ``writer`` — txt / md / json output format dispatch.
"""

from modules.postprocess.text import (
    fix_hyphenation,
    normalize_spacing,
    normalize_unicode_text,
    postprocess_file,
    postprocess_text,
    postprocess_transcription,
    wrap_long_lines,
)
from modules.postprocess.writer import (
    resolve_output_path,
    write_transcription_output,
)

__all__ = [
    # Text pipeline
    "postprocess_text",
    "postprocess_file",
    "postprocess_transcription",
    "normalize_unicode_text",
    "fix_hyphenation",
    "normalize_spacing",
    "wrap_long_lines",
    # Output writing
    "write_transcription_output",
    "resolve_output_path",
]
