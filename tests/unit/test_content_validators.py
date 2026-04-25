"""Tests for content quality validators."""

from __future__ import annotations

import pytest

from modules.llm.quality import ContentQualityError
from modules.llm.quality import (
    detect_excessive_line_repetition,
    detect_hallucination_loop,
    detect_invalid_transcription_markers,
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
    "invalid_markers": {
        "enabled": True,
        "icon_description_literals": [
            "!icon:",
            "(icon)",
            "[see image",
            "[figure",
            "[picture",
            "(hotel icon)",
            "(fork and spoon)",
        ],
        "generic_placeholder_patterns": [
            "[icon]",
            "[symbol]",
            "[unknown]",
        ],
        "max_generic_placeholder_count": 3,
        "max_question_mark_ratio": 0.05,
    },
}

# Sample texts for invalid-transcription-marker validator tests.
ICON_DESCRIPTION_DRIFT = (
    "LYON 69000 Rhône\n\n"
    "Le Grand Hôtel, 15 rue de la Paix ☎ 78 42 90 90\n"
    "SC : R 250/350 — carte 300 à 450\n\n"
    "!Icon: Hotel\n"
    "**Paul Bocuse** (3 Michelin stars), 40 r. de la Plage\n"
)

ICON_PARENTHETICAL_DRIFT = (
    "LYON 69000 Rhône\n\n"
    "(hotel icon) Royal Garden, Kensington High St. ☎ 937 8000\n"
    "M 19.75/24.00 st. – 389 rm 107.50/175.00 st.\n\n"
    "(fork and spoon) Halcyon, 81 Holland Park ☎ 727 7288\n"
)

GENERIC_PLACEHOLDER_FLOODING = (
    "LYON 69000 Rhône\n\n"
    "[icon] Paul Bocuse, 40 r. de la Plage ☎ 78 42 90 90\n"
    "SC : R 250/350\n\n"
    "[icon] Léon de Lyon, 1 r. Pléney ☎ 78 28 11 33\n"
    "SC : R 180/280\n\n"
    "[icon] Pierre Orsi, 3 pl. Kléber ☎ 78 89 57 68\n"
    "SC : R 140/180\n\n"
    "[icon] La Mère Brazier, 12 r. Royale ☎ 78 28 15 49\n"
    "SC : R 200/280\n"
)

QUESTION_MARK_CLUSTER = (
    "LY?N 6900? Rh?ne\n\n"
    "?? Gr?nd H?tel, 15 r. de l? P?ix ☎ 7? 42 9? 90\n"
    "SC : R 25?/350 — c?rte 3?0 à ?50\n\n"
    "?? P?ul B?cuse, 40 r. de la Pl?ge ☎ ?8 42 90 9?\n"
    "SC : R ???/???\n\n"
    "?? Lé?n ?e Ly?n, 1 r. Plén?y ☎ ?8 28 11 33\n"
)


# ── detect_hallucination_loop ────────────────────────────────────────────────

class TestDetectHallucinationLoop:
    def test_clean_text_passes(self) -> None:
        assert detect_hallucination_loop(CLEAN_TRANSCRIPTION, DEFAULT_CONFIG) is None

    def test_repeated_substring_detected(self) -> None:
        result = detect_hallucination_loop(HALLUCINATION_LOOP, DEFAULT_CONFIG)
        assert result is not None
        assert "repeated" in result.lower()

    def test_single_char_repeat_detected(self) -> None:
        result = detect_hallucination_loop(SINGLE_CHAR_REPEAT, DEFAULT_CONFIG)
        assert result is not None
        assert "M" in result

    def test_short_repeat_below_threshold(self) -> None:
        text = "abc " * 10  # below 30-char threshold
        assert detect_hallucination_loop(text, DEFAULT_CONFIG) is None

    def test_empty_text_passes(self) -> None:
        assert detect_hallucination_loop("", DEFAULT_CONFIG) is None

    def test_none_text_passes(self) -> None:
        assert detect_hallucination_loop(None, DEFAULT_CONFIG) is None

    def test_michelin_boilerplate_passes(self) -> None:
        """12 entries sharing standard Michelin formatting must not trigger."""
        assert detect_hallucination_loop(MICHELIN_BOILERPLATE, DEFAULT_CONFIG) is None

    def test_whitespace_loop_ignored(self) -> None:
        """'straat              ' x20 should not trigger (insufficient alphanum)."""
        assert detect_hallucination_loop(WHITESPACE_LOOP, DEFAULT_CONFIG) is None

    def test_newline_loop_ignored(self) -> None:
        """Pure newline runs should not trigger substring check."""
        assert detect_hallucination_loop(NEWLINE_LOOP, DEFAULT_CONFIG) is None

    def test_dot_leader_loop_ignored(self) -> None:
        """Table-of-contents dot leaders should not trigger (punctuation-heavy)."""
        assert detect_hallucination_loop(DOT_LEADER_LOOP, DEFAULT_CONFIG) is None


# ── detect_truncation ────────────────────────────────────────────────────────

class TestDetectTruncation:
    def test_clean_text_passes(self) -> None:
        assert detect_truncation(
            CLEAN_TRANSCRIPTION, False, False, DEFAULT_CONFIG
        ) is None

    def test_short_text_detected(self) -> None:
        result = detect_truncation(SHORT_TRUNCATION, False, False, DEFAULT_CONFIG)
        assert result is not None
        assert "below minimum" in result.lower()

    def test_short_text_ok_when_no_transcribable(self) -> None:
        assert detect_truncation(SHORT_TRUNCATION, True, False, DEFAULT_CONFIG) is None

    def test_short_text_ok_when_not_possible(self) -> None:
        assert detect_truncation(SHORT_TRUNCATION, False, True, DEFAULT_CONFIG) is None

    def test_none_text_detected(self) -> None:
        result = detect_truncation(None, False, False, DEFAULT_CONFIG)
        assert result is not None
        assert "None" in result

    def test_exactly_at_threshold(self) -> None:
        text = "x" * 50
        assert detect_truncation(text, False, False, DEFAULT_CONFIG) is None

    def test_one_below_threshold(self) -> None:
        text = "x" * 49
        result = detect_truncation(text, False, False, DEFAULT_CONFIG)
        assert result is not None


# ── detect_system_prompt_bleed ───────────────────────────────────────────────

class TestDetectSystemPromptBleed:
    def test_clean_text_passes(self) -> None:
        assert detect_system_prompt_bleed(CLEAN_TRANSCRIPTION, DEFAULT_CONFIG) is None

    def test_bleed_detected(self) -> None:
        result = detect_system_prompt_bleed(SYSTEM_PROMPT_BLEED, DEFAULT_CONFIG)
        assert result is not None
        assert "bleed" in result.lower()

    def test_pattern_past_check_window_passes(self) -> None:
        text = "x" * 250 + '{"type": "object"}'
        assert detect_system_prompt_bleed(text, DEFAULT_CONFIG) is None

    def test_empty_text_passes(self) -> None:
        assert detect_system_prompt_bleed("", DEFAULT_CONFIG) is None


# ── detect_excessive_line_repetition ─────────────────────────────────────────

class TestDetectExcessiveLineRepetition:
    def test_clean_text_passes(self) -> None:
        assert detect_excessive_line_repetition(
            CLEAN_TRANSCRIPTION, DEFAULT_CONFIG
        ) is None

    def test_map_label_loop_detected(self) -> None:
        result = detect_excessive_line_repetition(MAP_LABEL_LOOP, DEFAULT_CONFIG)
        assert result is not None
        assert "repeated" in result.lower()

    def test_short_repeated_lines_ignored(self) -> None:
        text = "abc\n" * 50  # 3 chars, below min_line_length_for_repeat
        assert detect_excessive_line_repetition(text, DEFAULT_CONFIG) is None

    def test_below_threshold_passes(self) -> None:
        text = "R. de la Republique\n" * 17  # below 18 threshold
        assert detect_excessive_line_repetition(text, DEFAULT_CONFIG) is None

    def test_empty_text_passes(self) -> None:
        assert detect_excessive_line_repetition("", DEFAULT_CONFIG) is None


# ── detect_invalid_transcription_markers ────────────────────────────────────

class TestDetectInvalidTranscriptionMarkers:
    def test_clean_text_passes(self) -> None:
        assert detect_invalid_transcription_markers(
            CLEAN_TRANSCRIPTION, DEFAULT_CONFIG
        ) is None

    def test_icon_description_literal_detected(self) -> None:
        result = detect_invalid_transcription_markers(
            ICON_DESCRIPTION_DRIFT, DEFAULT_CONFIG
        )
        assert result is not None
        assert "icon" in result.lower()

    def test_icon_parenthetical_drift_detected(self) -> None:
        result = detect_invalid_transcription_markers(
            ICON_PARENTHETICAL_DRIFT, DEFAULT_CONFIG
        )
        assert result is not None

    def test_generic_placeholder_flooding_detected(self) -> None:
        result = detect_invalid_transcription_markers(
            GENERIC_PLACEHOLDER_FLOODING, DEFAULT_CONFIG
        )
        assert result is not None
        assert "[icon]" in result
        assert "repeated" in result.lower()

    def test_generic_placeholder_below_threshold_passes(self) -> None:
        text = (
            "**Paul Bocuse** ⭐⭐⭐\n"
            "[icon] Le Grand Hôtel ☎ 78 42 90 90 — SC : R 200/300\n"
            "[icon] Pierre Orsi ☎ 78 89 57 68 — SC : R 140/180\n"
            "**Léon de Lyon** ⭐⭐\n"
        )
        # 2 occurrences, below default threshold of 3 — should pass
        assert detect_invalid_transcription_markers(text, DEFAULT_CONFIG) is None

    def test_question_mark_cluster_detected(self) -> None:
        result = detect_invalid_transcription_markers(
            QUESTION_MARK_CLUSTER, DEFAULT_CONFIG
        )
        assert result is not None
        assert "question-mark" in result.lower() or "?" in result

    def test_single_question_mark_passes(self) -> None:
        text = CLEAN_TRANSCRIPTION + "\n(was this Paul Bocuse or son?)"
        assert detect_invalid_transcription_markers(text, DEFAULT_CONFIG) is None

    def test_disabled_config_skips(self) -> None:
        config = {**DEFAULT_CONFIG, "invalid_markers": {"enabled": False}}
        assert detect_invalid_transcription_markers(
            ICON_DESCRIPTION_DRIFT, config
        ) is None

    def test_empty_text_passes(self) -> None:
        assert detect_invalid_transcription_markers("", DEFAULT_CONFIG) is None

    def test_none_text_passes(self) -> None:
        assert detect_invalid_transcription_markers(None, DEFAULT_CONFIG) is None

    def test_michelin_boilerplate_passes(self) -> None:
        """Realistic Michelin entries must not trigger false positives."""
        assert detect_invalid_transcription_markers(
            MICHELIN_BOILERPLATE, DEFAULT_CONFIG
        ) is None


# ── validate_content_quality (orchestrator) ──────────────────────────────────

class TestValidateContentQuality:
    def test_clean_text_passes(self) -> None:
        validate_content_quality(
            CLEAN_TRANSCRIPTION, False, False, DEFAULT_CONFIG
        )

    def test_hallucination_raises(self) -> None:
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                HALLUCINATION_LOOP, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "hallucination_loop"

    def test_truncation_raises(self) -> None:
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                SHORT_TRUNCATION, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "truncation"

    def test_bleed_raises(self) -> None:
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                SYSTEM_PROMPT_BLEED, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "system_prompt_bleed"

    def test_line_repetition_raises(self) -> None:
        # The repeated line also triggers hallucination_loop since it's >20 chars
        # repeated >5 times. Either detection is acceptable.
        text = "Some preamble text here\n" + "Av. des Champs-Elysees\n" * 20
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(text, False, False, DEFAULT_CONFIG)
        assert exc_info.value.failure_type in (
            "hallucination_loop",
            "excessive_line_repetition",
        )

    def test_disabled_config_skips_all(self) -> None:
        config = {**DEFAULT_CONFIG, "enabled": False}
        # Would fail on truncation if enabled
        validate_content_quality(SHORT_TRUNCATION, False, False, config)

    def test_no_transcribable_text_skips(self) -> None:
        validate_content_quality(SHORT_TRUNCATION, True, False, DEFAULT_CONFIG)

    def test_transcription_not_possible_skips(self) -> None:
        validate_content_quality(SHORT_TRUNCATION, False, True, DEFAULT_CONFIG)

    def test_first_failure_wins(self) -> None:
        # Text has both hallucination AND line repetition; hallucination checked first
        text = "M" * 100 + "\n" + ("R. de la Republique\n" * 25)
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(text, False, False, DEFAULT_CONFIG)
        assert exc_info.value.failure_type == "hallucination_loop"

    def test_invalid_marker_icon_description_raises(self) -> None:
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                ICON_DESCRIPTION_DRIFT, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "invalid_transcription_markers"

    def test_invalid_marker_generic_flooding_raises(self) -> None:
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                GENERIC_PLACEHOLDER_FLOODING, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "invalid_transcription_markers"

    def test_invalid_marker_question_mark_cluster_raises(self) -> None:
        with pytest.raises(ContentQualityError) as exc_info:
            validate_content_quality(
                QUESTION_MARK_CLUSTER, False, False, DEFAULT_CONFIG
            )
        assert exc_info.value.failure_type == "invalid_transcription_markers"
