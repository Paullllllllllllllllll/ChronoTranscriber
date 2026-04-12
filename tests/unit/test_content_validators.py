"""Tests for content quality validators."""

import pytest

from modules.processing.content_quality_error import ContentQualityError
from modules.processing.content_validators import (
    detect_excessive_line_repetition,
    detect_hallucination_loop,
    detect_system_prompt_bleed,
    detect_truncation,
    validate_content_quality,
)

# ── Sample texts ─────────────────────────────────────────────────────────────

CLEAN_TRANSCRIPTION = (
    "LYON 69000\n\n"
    "**Paul Bocuse** ⭐⭐⭐ 🍴🍴🍴🍴🍴\n"
    "40 r. de la Plage ☎ 78 42 90 90\n"
    "SC : R 250/350 — carte 300 à 450\n"
    "Spéc. Soupe aux truffes noires.\n\n"
    "**Léon de Lyon** ⭐⭐ 🍴🍴🍴🍴\n"
    "1 r. Pléney ☎ 78 28 11 33\n"
    "SC : R 180/280\n"
)

HALLUCINATION_LOOP = "de la Playa belongs here " * 200

# Realistic Michelin boilerplate that should NOT trigger (entries share format)
MICHELIN_BOILERPLATE = "\n\n".join(
    [
        f"**Restaurant {i}**, {10 + i} rue de la Paix ☎ 01.23.45.67.8{i}\n"
        f"fermé dim soir et lundi – SC : R 45/85 – ☕ 12"
        for i in range(12)
    ]
)

# Whitespace-padded map labels (should NOT trigger under new rules)
WHITESPACE_LOOP = "straat              \n" * 20
NEWLINE_LOOP = "\n" * 100
DOT_LEADER_LOOP = ". . . . . . . . . . . . . . . . . . . . . . .\n" * 10

SINGLE_CHAR_REPEAT = "Normal text\n" + "M" * 100 + "\nMore text"

SHORT_TRUNCATION = "LYON\n**Bocuse**"

SYSTEM_PROMPT_BLEED = (
    '{"type": "object", "properties": {"transcription": {"type": "string"}}, '
    '"additionalProperties": false}\n\nLYON 69000\n**Paul Bocuse** ⭐⭐⭐'
)

MAP_LABEL_LOOP = (
    "R. de la Republique\n" * 25
    + "Pl. Bellecour\n" * 25
)

DEFAULT_CONFIG = {
    "enabled": True,
    "max_repeated_substring_length": 30,
    "max_repeated_substring_count": 15,
    "min_alphanum_in_substring": 15,
    "max_single_char_repeat": 75,
    "min_transcription_length": 50,
    "bleed_check_chars": 200,
    "system_prompt_bleed_patterns": [
        '"type": "object"',
        '"additionalProperties"',
        '"transcription_not_possible"',
    ],
    "max_line_repeat_count": 18,
    "min_line_length_for_repeat": 10,
}


# ── detect_hallucination_loop ────────────────────────────────────────────────

class TestDetectHallucinationLoop:
    def test_clean_text_passes(self):
        assert detect_hallucination_loop(CLEAN_TRANSCRIPTION, DEFAULT_CONFIG) is None

    def test_repeated_substring_detected(self):
        result = detect_hallucination_loop(HALLUCINATION_LOOP, DEFAULT_CONFIG)
        assert result is not None
        assert "repeated" in result.lower()

    def test_single_char_repeat_detected(self):
        result = detect_hallucination_loop(SINGLE_CHAR_REPEAT, DEFAULT_CONFIG)
        assert result is not None
        assert "M" in result

    def test_short_repeat_below_threshold(self):
        text = "abc " * 10  # below 30-char threshold
        assert detect_hallucination_loop(text, DEFAULT_CONFIG) is None

    def test_empty_text_passes(self):
        assert detect_hallucination_loop("", DEFAULT_CONFIG) is None

    def test_none_text_passes(self):
        assert detect_hallucination_loop(None, DEFAULT_CONFIG) is None

    def test_michelin_boilerplate_passes(self):
        """12 entries sharing standard Michelin formatting must not trigger."""
        assert detect_hallucination_loop(MICHELIN_BOILERPLATE, DEFAULT_CONFIG) is None

    def test_whitespace_loop_ignored(self):
        """'straat              ' x20 should not trigger (insufficient alphanum)."""
        assert detect_hallucination_loop(WHITESPACE_LOOP, DEFAULT_CONFIG) is None

    def test_newline_loop_ignored(self):
        """Pure newline runs should not trigger substring check."""
        assert detect_hallucination_loop(NEWLINE_LOOP, DEFAULT_CONFIG) is None

    def test_dot_leader_loop_ignored(self):
        """Table-of-contents dot leaders should not trigger (punctuation-heavy)."""
        assert detect_hallucination_loop(DOT_LEADER_LOOP, DEFAULT_CONFIG) is None


# ── detect_truncation ────────────────────────────────────────────────────────

class TestDetectTruncation:
    def test_clean_text_passes(self):
        assert detect_truncation(
            CLEAN_TRANSCRIPTION, False, False, DEFAULT_CONFIG
        ) is None

    def test_short_text_detected(self):
        result = detect_truncation(SHORT_TRUNCATION, False, False, DEFAULT_CONFIG)
        assert result is not None
        assert "below minimum" in result.lower()

    def test_short_text_ok_when_no_transcribable(self):
        assert detect_truncation(SHORT_TRUNCATION, True, False, DEFAULT_CONFIG) is None

    def test_short_text_ok_when_not_possible(self):
        assert detect_truncation(SHORT_TRUNCATION, False, True, DEFAULT_CONFIG) is None

    def test_none_text_detected(self):
        result = detect_truncation(None, False, False, DEFAULT_CONFIG)
        assert result is not None
        assert "None" in result

    def test_exactly_at_threshold(self):
        text = "x" * 50
        assert detect_truncation(text, False, False, DEFAULT_CONFIG) is None

    def test_one_below_threshold(self):
        text = "x" * 49
        result = detect_truncation(text, False, False, DEFAULT_CONFIG)
        assert result is not None


# ── detect_system_prompt_bleed ───────────────────────────────────────────────

class TestDetectSystemPromptBleed:
    def test_clean_text_passes(self):
        assert detect_system_prompt_bleed(CLEAN_TRANSCRIPTION, DEFAULT_CONFIG) is None

    def test_bleed_detected(self):
        result = detect_system_prompt_bleed(SYSTEM_PROMPT_BLEED, DEFAULT_CONFIG)
        assert result is not None
        assert "bleed" in result.lower()

    def test_pattern_past_check_window_passes(self):
        text = "x" * 250 + '{"type": "object"}'
        assert detect_system_prompt_bleed(text, DEFAULT_CONFIG) is None

    def test_empty_text_passes(self):
        assert detect_system_prompt_bleed("", DEFAULT_CONFIG) is None


# ── detect_excessive_line_repetition ─────────────────────────────────────────

class TestDetectExcessiveLineRepetition:
    def test_clean_text_passes(self):
        assert detect_excessive_line_repetition(
            CLEAN_TRANSCRIPTION, DEFAULT_CONFIG
        ) is None

    def test_map_label_loop_detected(self):
        result = detect_excessive_line_repetition(MAP_LABEL_LOOP, DEFAULT_CONFIG)
        assert result is not None
        assert "repeated" in result.lower()

    def test_short_repeated_lines_ignored(self):
        text = "abc\n" * 50  # 3 chars, below min_line_length_for_repeat
        assert detect_excessive_line_repetition(text, DEFAULT_CONFIG) is None

    def test_below_threshold_passes(self):
        text = "R. de la Republique\n" * 17  # below 18 threshold
        assert detect_excessive_line_repetition(text, DEFAULT_CONFIG) is None

    def test_empty_text_passes(self):
        assert detect_excessive_line_repetition("", DEFAULT_CONFIG) is None


# ── validate_content_quality (orchestrator) ──────────────────────────────────

class TestValidateContentQuality:
    def test_clean_text_passes(self):
        validate_content_quality(
            CLEAN_TRANSCRIPTION, False, False, DEFAULT_CONFIG
        )

    def test_hallucination_raises(self):
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                HALLUCINATION_LOOP, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "hallucination_loop"

    def test_truncation_raises(self):
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                SHORT_TRUNCATION, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "truncation"

    def test_bleed_raises(self):
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                SYSTEM_PROMPT_BLEED, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "system_prompt_bleed"

    def test_line_repetition_raises(self):
        # The repeated line also triggers hallucination_loop since it's >20 chars
        # repeated >5 times. Either detection is acceptable.
        text = "Some preamble text here\n" + "Av. des Champs-Elysees\n" * 20
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(text, False, False, DEFAULT_CONFIG)
        assert exc_info.value.failure_type in (
            "hallucination_loop",
            "excessive_line_repetition",
        )

    def test_disabled_config_skips_all(self):
        config = {**DEFAULT_CONFIG, "enabled": False}
        # Would fail on truncation if enabled
        validate_content_quality(SHORT_TRUNCATION, False, False, config)

    def test_no_transcribable_text_skips(self):
        validate_content_quality(SHORT_TRUNCATION, True, False, DEFAULT_CONFIG)

    def test_transcription_not_possible_skips(self):
        validate_content_quality(SHORT_TRUNCATION, False, True, DEFAULT_CONFIG)

    def test_first_failure_wins(self):
        # Text has both hallucination AND line repetition; hallucination checked first
        text = "M" * 100 + "\n" + ("R. de la Republique\n" * 25)
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(text, False, False, DEFAULT_CONFIG)
        assert exc_info.value.failure_type == "hallucination_loop"
