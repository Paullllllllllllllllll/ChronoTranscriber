"""Tests for modules.audio.constants: format tables, limits, and sentinels."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modules.audio.constants import (
    DEFAULT_MIN_CHUNK_SECONDS,
    DEFAULT_TARGET_CHUNK_SECONDS,
    GEMINI_ALLOWED_EXTENSIONS,
    GEMINI_INLINE_LIMIT_BYTES,
    NO_TRANSCRIBABLE_TEXT,
    OPENAI_ALLOWED_EXTENSIONS,
    OPENAI_PLURAL_PARAM_MODELS,
    OPENAI_UPLOAD_LIMIT_BYTES,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_AUDIO_FORMATS,
    WHISPER1_PROMPT_TOKEN_CAP,
)

# ---------------------------------------------------------------------------
# Extension / MIME table coherence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatTable:
    def test_extension_set_matches_format_table_keys(self) -> None:
        assert frozenset(SUPPORTED_AUDIO_FORMATS) == SUPPORTED_AUDIO_EXTENSIONS

    def test_every_extension_is_a_lowercase_dotted_suffix(self) -> None:
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            assert ext.startswith(".")
            assert ext == ext.lower()
            assert len(ext) > 1

    def test_every_mime_type_is_an_audio_type(self) -> None:
        for ext, mime in SUPPORTED_AUDIO_FORMATS.items():
            assert mime.startswith("audio/"), ext

    def test_no_duplicate_extensions(self) -> None:
        assert len(SUPPORTED_AUDIO_FORMATS) == len(SUPPORTED_AUDIO_EXTENSIONS)


# ---------------------------------------------------------------------------
# Provider allow-lists and payload ceilings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderLimits:
    def test_openai_allowlist_is_a_subset_of_the_format_table(self) -> None:
        assert OPENAI_ALLOWED_EXTENSIONS <= SUPPORTED_AUDIO_EXTENSIONS

    def test_gemini_allowlist_is_a_subset_of_the_format_table(self) -> None:
        assert GEMINI_ALLOWED_EXTENSIONS <= SUPPORTED_AUDIO_EXTENSIONS

    def test_allowlists_are_non_empty(self) -> None:
        assert OPENAI_ALLOWED_EXTENSIONS
        assert GEMINI_ALLOWED_EXTENSIONS

    def test_gemini_inline_ceiling_is_below_the_openai_upload_ceiling(self) -> None:
        assert 0 < GEMINI_INLINE_LIMIT_BYTES < OPENAI_UPLOAD_LIMIT_BYTES

    def test_documented_ceilings(self) -> None:
        assert OPENAI_UPLOAD_LIMIT_BYTES == 25 * 1024 * 1024
        assert GEMINI_INLINE_LIMIT_BYTES == 20 * 1024 * 1024

    def test_mp3_and_wav_are_accepted_by_both_venues(self) -> None:
        # The chunker's two convertible formats must be sendable anywhere.
        for ext in (".mp3", ".wav"):
            assert ext in OPENAI_ALLOWED_EXTENSIONS
            assert ext in GEMINI_ALLOWED_EXTENSIONS

    def test_plural_param_models_are_openai_audio_models(self) -> None:
        from modules.config.capabilities import detect_capabilities

        assert OPENAI_PLURAL_PARAM_MODELS
        for model in OPENAI_PLURAL_PARAM_MODELS:
            assert detect_capabilities(model).supports_audio_input is True

    def test_whisper1_prompt_cap(self) -> None:
        assert WHISPER1_PROMPT_TOKEN_CAP == 224


# ---------------------------------------------------------------------------
# Shared no-text sentinel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoTranscribableTextSentinel:
    def test_sentinel_literal(self) -> None:
        assert NO_TRANSCRIBABLE_TEXT == "[No transcribable text]"

    def test_sentinel_is_byte_identical_to_the_tesseract_placeholder(
        self, tmp_path: Path
    ) -> None:
        """The Tesseract path must emit exactly this string for empty OCR."""
        from modules.images.tesseract_runtime import perform_ocr

        img_path = tmp_path / "blank.png"
        img_path.write_bytes(b"")

        fake_image = MagicMock()
        fake_image.__enter__ = MagicMock(return_value=fake_image)
        fake_image.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "modules.images.tesseract_runtime.Image.open", return_value=fake_image
            ),
            patch(
                "modules.images.tesseract_runtime.pytesseract.image_to_string",
                return_value="   \n  ",
            ),
        ):
            result = perform_ocr(img_path, "--oem 3 --psm 6")

        assert result == NO_TRANSCRIBABLE_TEXT

    def test_sentinel_is_classified_as_no_text_not_an_error(self) -> None:
        from modules.llm.response_parsing import detect_transcription_cause

        assert detect_transcription_cause(NO_TRANSCRIBABLE_TEXT) == "no_text"


# ---------------------------------------------------------------------------
# Chunk-planning defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkDefaults:
    def test_default_target_and_minimum(self) -> None:
        assert DEFAULT_TARGET_CHUNK_SECONDS == 600
        assert DEFAULT_MIN_CHUNK_SECONDS == 30
        assert DEFAULT_MIN_CHUNK_SECONDS < DEFAULT_TARGET_CHUNK_SECONDS
