"""Audio transcription: chunk planning, cutting, and speech-to-text.

Covers the whole path from a recording on disk to transcript text: probing
duration and planning deterministic segments (:mod:`chunker`), cutting them
with ffmpeg and streaming them as payloads (:mod:`ffmpeg_runtime`,
:mod:`audio_stream`), and submitting each to a remote venue
(:mod:`transcriber`, :mod:`backends`) or to a local faster-whisper model
(:mod:`whisper_runtime`). Output-path naming lives in :mod:`paths` and stays
byte-identical to the PDF workflow's, so resume works across both.

Both external runtimes are optional: ffmpeg is looked up at call time and
``faster-whisper`` is imported lazily, so importing this package never fails
because a recording tool is missing.
"""

from modules.audio.audio_stream import (
    AudioChunkPayload,
    audio_chunk_name,
    compute_audio_skip_indices,
    parse_audio_chunk_index,
    stream_audio_chunks,
)
from modules.audio.chunker import (
    ChunkSpec,
    chunk_plan_signature,
    estimate_output_bytes_per_second,
    plan_chunks,
)
from modules.audio.constants import (
    NO_TRANSCRIBABLE_TEXT,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_AUDIO_FORMATS,
)
from modules.audio.ffmpeg_runtime import (
    configure_ffmpeg_executables,
    cut_segment,
    ensure_ffmpeg_available,
    is_ffmpeg_available,
    probe_duration_seconds,
)
from modules.audio.paths import prepare_audio_output
from modules.audio.transcriber import AudioTranscriber, open_audio_transcriber
from modules.audio.whisper_runtime import (
    ensure_faster_whisper_available,
    is_faster_whisper_available,
    transcribe_file,
)

__all__ = [
    "NO_TRANSCRIBABLE_TEXT",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SUPPORTED_AUDIO_FORMATS",
    "AudioChunkPayload",
    "AudioTranscriber",
    "ChunkSpec",
    "audio_chunk_name",
    "chunk_plan_signature",
    "compute_audio_skip_indices",
    "configure_ffmpeg_executables",
    "cut_segment",
    "ensure_faster_whisper_available",
    "ensure_ffmpeg_available",
    "estimate_output_bytes_per_second",
    "is_faster_whisper_available",
    "is_ffmpeg_available",
    "open_audio_transcriber",
    "parse_audio_chunk_index",
    "plan_chunks",
    "prepare_audio_output",
    "probe_duration_seconds",
    "stream_audio_chunks",
    "transcribe_file",
]
