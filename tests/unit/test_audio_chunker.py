"""Tests for modules.audio.chunker: deterministic chunk planning."""

from __future__ import annotations

import pytest

from modules.audio.chunker import (
    ChunkSpec,
    chunk_plan_signature,
    estimate_output_bytes_per_second,
    plan_chunks,
)
from modules.audio.constants import OPENAI_UPLOAD_LIMIT_BYTES

# Nominal mp3 mono rate used by the planner's own estimator.
_MP3_MONO_BPS = 8000.0


def _plan(**overrides: object) -> list[ChunkSpec]:
    """Plan chunks for a 1-hour recording with sensible defaults."""
    kwargs: dict[str, object] = {
        "duration_seconds": 3600.0,
        "size_bytes": 200 * 1024 * 1024,
        "target_seconds": 600,
        "max_request_bytes": OPENAI_UPLOAD_LIMIT_BYTES,
        "output_bytes_per_second": _MP3_MONO_BPS,
    }
    kwargs.update(overrides)
    return plan_chunks(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Whole-file plans
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWholeFilePlans:
    def test_short_small_recording_is_left_whole(self) -> None:
        specs = _plan(duration_seconds=120.0, size_bytes=1_000_000)
        assert specs == [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]

    def test_recording_exactly_at_both_limits_is_left_whole(self) -> None:
        specs = _plan(
            duration_seconds=600.0,
            size_bytes=OPENAI_UPLOAD_LIMIT_BYTES,
        )
        assert len(specs) == 1
        assert specs[0].duration_seconds is None

    def test_unknown_duration_falls_back_to_a_whole_file_plan(self) -> None:
        specs = _plan(duration_seconds=0.0)
        assert specs == [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]

    def test_negative_duration_falls_back_to_a_whole_file_plan(self) -> None:
        specs = _plan(duration_seconds=-5.0)
        assert len(specs) == 1
        assert specs[0].duration_seconds is None

    def test_non_positive_request_cap_falls_back_to_a_whole_file_plan(self) -> None:
        specs = _plan(max_request_bytes=0)
        assert len(specs) == 1
        assert specs[0].duration_seconds is None

    def test_oversized_recording_longer_than_the_size_bound_is_split(self) -> None:
        # Over the byte cap and long enough that the size-derived ceiling binds.
        specs = _plan(
            duration_seconds=4000.0,
            size_bytes=OPENAI_UPLOAD_LIMIT_BYTES + 1,
            output_bytes_per_second=100_000.0,
        )
        assert len(specs) > 1

    def test_oversized_but_short_recording_is_still_re_encoded(self) -> None:
        # A 4-minute 46 MB WAV: over the byte cap yet shorter than one chunk.
        # The derived single spec must keep a finite duration — ``None`` would
        # read as "send the source untouched" and upload the oversized file.
        specs = _plan(
            duration_seconds=240.0,
            size_bytes=46 * 1024 * 1024,
        )
        assert len(specs) == 1
        assert specs[0].start_seconds == 0.0
        assert specs[0].duration_seconds is not None
        assert specs[0].duration_seconds >= 240.0


# ---------------------------------------------------------------------------
# Boundary math
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoundaryMath:
    def test_one_hour_at_600s_targets_yields_six_chunks(self) -> None:
        specs = _plan()
        assert len(specs) == 6
        assert [s.index for s in specs] == [0, 1, 2, 3, 4, 5]
        assert [s.start_seconds for s in specs] == [
            0.0,
            600.0,
            1200.0,
            1800.0,
            2400.0,
            3000.0,
        ]

    def test_every_chunk_but_the_last_carries_the_effective_duration(self) -> None:
        specs = _plan()
        assert all(s.duration_seconds == 600.0 for s in specs[:-1])

    def test_last_chunk_runs_to_the_end_of_the_recording(self) -> None:
        specs = _plan()
        assert specs[-1].duration_seconds is None

    def test_indices_are_contiguous_and_zero_based(self) -> None:
        specs = _plan(duration_seconds=5000.0)
        assert [s.index for s in specs] == list(range(len(specs)))

    def test_starts_are_strictly_increasing(self) -> None:
        specs = _plan(duration_seconds=5000.0)
        starts = [s.start_seconds for s in specs]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_plan_covers_the_whole_recording(self) -> None:
        specs = _plan()
        assert specs[-1].start_seconds < 3600.0

    def test_determinism_two_identical_calls_match(self) -> None:
        assert _plan() == _plan()

    def test_determinism_holds_for_an_overlapped_plan(self) -> None:
        first = _plan(overlap_seconds=45.0)
        second = _plan(overlap_seconds=45.0)
        assert first == second


# ---------------------------------------------------------------------------
# Size-derived ceiling and the minimum-length floor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSizeBoundAndFloor:
    def test_high_bitrate_binds_the_chunk_length_below_the_target(self) -> None:
        # 100 kB/s fills 90 % of a 25 MB request in 235 s, well under 600 s.
        specs = _plan(output_bytes_per_second=100_000.0)
        expected = int(OPENAI_UPLOAD_LIMIT_BYTES * 0.9 // 100_000)
        assert specs[0].duration_seconds == float(expected)
        assert expected < 600

    def test_size_bound_produces_more_chunks_than_the_target_alone(self) -> None:
        by_target = _plan()
        by_size = _plan(output_bytes_per_second=100_000.0)
        assert len(by_size) > len(by_target)

    def test_min_chunk_seconds_floors_a_pathological_bitrate_estimate(self) -> None:
        specs = _plan(output_bytes_per_second=10_000_000.0, min_chunk_seconds=30)
        assert specs[0].duration_seconds == 30.0

    def test_custom_min_chunk_seconds_is_honored(self) -> None:
        specs = _plan(output_bytes_per_second=10_000_000.0, min_chunk_seconds=120)
        assert specs[0].duration_seconds == 120.0

    def test_zero_output_rate_does_not_divide_by_zero(self) -> None:
        specs = _plan(output_bytes_per_second=0.0)
        assert specs
        assert specs[0].duration_seconds == 600.0


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOverlap:
    def test_overlap_shifts_the_chunk_starts(self) -> None:
        specs = _plan(overlap_seconds=60.0)
        assert [s.start_seconds for s in specs[:4]] == [0.0, 540.0, 1080.0, 1620.0]

    def test_overlap_yields_at_least_as_many_chunks(self) -> None:
        assert len(_plan(overlap_seconds=60.0)) >= len(_plan())

    def test_overlap_does_not_change_the_nominal_chunk_length(self) -> None:
        specs = _plan(overlap_seconds=60.0)
        assert specs[0].duration_seconds == 600.0

    def test_overlap_at_or_beyond_the_chunk_length_is_ignored(self) -> None:
        specs = _plan(overlap_seconds=600.0)
        assert [s.start_seconds for s in specs] == [s.start_seconds for s in _plan()]

    def test_negative_overlap_is_clamped_to_zero(self) -> None:
        assert _plan(overlap_seconds=-30.0) == _plan()


# ---------------------------------------------------------------------------
# chunk_plan_signature
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkPlanSignature:
    def test_empty_plan_signature(self) -> None:
        assert chunk_plan_signature([]) == "0x0s-00000000"

    def test_signature_is_stable_across_calls(self) -> None:
        specs = _plan()
        assert chunk_plan_signature(specs) == chunk_plan_signature(specs)

    def test_signature_is_stable_across_equivalent_plans(self) -> None:
        assert chunk_plan_signature(_plan()) == chunk_plan_signature(_plan())

    def test_signature_encodes_count_and_nominal_length(self) -> None:
        signature = chunk_plan_signature(_plan())
        assert signature.startswith("6x600s-")

    def test_signature_differs_for_a_different_target_length(self) -> None:
        assert chunk_plan_signature(_plan()) != chunk_plan_signature(
            _plan(target_seconds=300)
        )

    def test_signature_differs_for_a_different_overlap(self) -> None:
        assert chunk_plan_signature(_plan()) != chunk_plan_signature(
            _plan(overlap_seconds=60.0)
        )

    def test_signature_differs_for_a_different_duration(self) -> None:
        assert chunk_plan_signature(_plan()) != chunk_plan_signature(
            _plan(duration_seconds=7200.0)
        )

    def test_signature_of_a_whole_file_plan(self) -> None:
        whole = [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]
        signature = chunk_plan_signature(whole)
        assert signature.startswith("1x0s-")
        assert len(signature.split("-")[-1]) == 8


# ---------------------------------------------------------------------------
# estimate_output_bytes_per_second
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateOutputBytesPerSecond:
    def test_mp3_mono(self) -> None:
        assert estimate_output_bytes_per_second("mp3", 16000, True) == 8000.0

    def test_mp3_stereo_doubles_the_rate(self) -> None:
        assert estimate_output_bytes_per_second("mp3", 16000, False) == 16000.0

    def test_mp3_rate_is_independent_of_the_sample_rate(self) -> None:
        assert estimate_output_bytes_per_second(
            "mp3", 48000, True
        ) == estimate_output_bytes_per_second("mp3", 8000, True)

    def test_wav16_is_two_bytes_per_sample_and_always_mono(self) -> None:
        assert estimate_output_bytes_per_second("wav16", 16000, True) == 32000.0
        assert estimate_output_bytes_per_second("wav16", 16000, False) == 32000.0

    def test_wav16_scales_with_the_sample_rate(self) -> None:
        assert estimate_output_bytes_per_second("wav16", 44100, True) == 88200.0

    def test_wav16_never_returns_zero_for_a_degenerate_sample_rate(self) -> None:
        assert estimate_output_bytes_per_second("wav16", 0, True) == 2.0

    def test_copy_uses_the_measured_source_rate(self) -> None:
        assert (
            estimate_output_bytes_per_second(
                "copy", 16000, True, source_bytes_per_second=24000.0
            )
            == 24000.0
        )

    def test_copy_without_a_source_rate_falls_back_to_the_mp3_estimate(self) -> None:
        assert estimate_output_bytes_per_second("copy", 16000, True) == 8000.0

    def test_copy_ignores_a_non_positive_source_rate(self) -> None:
        assert (
            estimate_output_bytes_per_second(
                "copy", 16000, True, source_bytes_per_second=0.0
            )
            == 8000.0
        )

    def test_unknown_format_falls_back_to_the_mp3_estimate(self) -> None:
        assert estimate_output_bytes_per_second("opus", 16000, True) == 8000.0

    def test_empty_format_falls_back_to_mp3(self) -> None:
        assert estimate_output_bytes_per_second("", 16000, True) == 8000.0

    def test_format_matching_is_case_and_whitespace_insensitive(self) -> None:
        assert estimate_output_bytes_per_second(" WAV16 ", 16000, True) == 32000.0

    def test_every_estimate_is_strictly_positive(self) -> None:
        for fmt in ("mp3", "wav16", "copy", "nonsense"):
            assert estimate_output_bytes_per_second(fmt, 1, True) > 0
