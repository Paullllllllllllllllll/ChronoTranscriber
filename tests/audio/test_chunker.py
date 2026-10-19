"""Tests for the deterministic chunk planner of the audio source."""

from __future__ import annotations

import pytest

from chronotranscriber.sources.audio.chunker import (
    ChunkSpec,
    chunk_plan_signature,
    estimate_output_bytes_per_second,
    plan_chunks,
)
from chronotranscriber.sources.audio.constants import (
    GEMINI_ALLOWED_EXTENSIONS,
    OPENAI_ALLOWED_EXTENSIONS,
    OPENAI_UPLOAD_LIMIT_BYTES,
)

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


@pytest.mark.unit
@pytest.mark.parametrize("extension", [".mp3", ".wav"])
def test_the_chunker_output_formats_are_accepted_by_both_providers(
    extension: str,
) -> None:
    # The chunker's two convertible formats must be sendable anywhere.
    assert extension in OPENAI_ALLOWED_EXTENSIONS
    assert extension in GEMINI_ALLOWED_EXTENSIONS


@pytest.mark.unit
class TestWholeFilePlans:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"duration_seconds": 120.0, "size_bytes": 1_000_000},
            {"duration_seconds": 600.0, "size_bytes": OPENAI_UPLOAD_LIMIT_BYTES},
            {"duration_seconds": 0.0},
            {"duration_seconds": -5.0},
            {"max_request_bytes": 0},
        ],
        ids=[
            "short-and-small",
            "exactly-at-both-limits",
            "unknown-duration",
            "negative-duration",
            "non-positive-request-cap",
        ],
    )
    def test_the_recording_is_left_whole(self, overrides: dict[str, object]) -> None:
        specs = _plan(**overrides)
        assert specs == [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]

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

    @pytest.mark.parametrize("overlap", [0.0, 45.0])
    def test_determinism_two_identical_calls_match(self, overlap: float) -> None:
        assert _plan(overlap_seconds=overlap) == _plan(overlap_seconds=overlap)


@pytest.mark.unit
class TestSizeBoundAndFloor:
    def test_high_bitrate_binds_the_chunk_length_below_the_target(self) -> None:
        # 100 kB/s fills 90 % of a 25 MB request in 235 s, well under 600 s.
        specs = _plan(output_bytes_per_second=100_000.0)
        expected = int(OPENAI_UPLOAD_LIMIT_BYTES * 0.9 // 100_000)
        assert specs[0].duration_seconds == float(expected)
        assert expected < 600

    @pytest.mark.parametrize("floor", [30, 120])
    def test_min_chunk_seconds_floors_a_pathological_bitrate_estimate(
        self, floor: int
    ) -> None:
        specs = _plan(output_bytes_per_second=10_000_000.0, min_chunk_seconds=floor)
        assert specs[0].duration_seconds == float(floor)

    def test_zero_output_rate_does_not_divide_by_zero(self) -> None:
        specs = _plan(output_bytes_per_second=0.0)
        assert specs
        assert specs[0].duration_seconds == 600.0


@pytest.mark.unit
class TestOverlap:
    def test_overlap_shifts_the_chunk_starts(self) -> None:
        specs = _plan(overlap_seconds=60.0)
        assert [s.start_seconds for s in specs[:4]] == [0.0, 540.0, 1080.0, 1620.0]

    def test_overlap_does_not_change_the_nominal_chunk_length(self) -> None:
        specs = _plan(overlap_seconds=60.0)
        assert specs[0].duration_seconds == 600.0

    def test_overlap_at_or_beyond_the_chunk_length_is_ignored(self) -> None:
        specs = _plan(overlap_seconds=600.0)
        assert [s.start_seconds for s in specs] == [s.start_seconds for s in _plan()]

    def test_negative_overlap_is_clamped_to_zero(self) -> None:
        assert _plan(overlap_seconds=-30.0) == _plan()


@pytest.mark.unit
class TestChunkPlanSignature:
    def test_empty_plan_signature(self) -> None:
        assert chunk_plan_signature([]) == "0x0s-00000000"

    def test_signature_is_stable_across_calls(self) -> None:
        specs = _plan()
        assert chunk_plan_signature(specs) == chunk_plan_signature(specs)

    def test_signature_encodes_count_and_nominal_length(self) -> None:
        signature = chunk_plan_signature(_plan())
        assert signature.startswith("6x600s-")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"target_seconds": 300},
            {"overlap_seconds": 60.0},
            {"duration_seconds": 7200.0},
        ],
    )
    def test_signature_differs_when_the_plan_differs(
        self, overrides: dict[str, object]
    ) -> None:
        assert chunk_plan_signature(_plan()) != chunk_plan_signature(_plan(**overrides))

    def test_signature_of_a_whole_file_plan(self) -> None:
        whole = [ChunkSpec(index=0, start_seconds=0.0, duration_seconds=None)]
        signature = chunk_plan_signature(whole)
        assert signature.startswith("1x0s-")
        assert len(signature.split("-")[-1]) == 8


@pytest.mark.unit
class TestEstimateOutputBytesPerSecond:
    @pytest.mark.parametrize(
        ("fmt", "sample_rate", "mono", "source_rate", "expected"),
        [
            ("mp3", 16000, True, None, 8000.0),
            ("mp3", 16000, False, None, 16000.0),
            ("mp3", 48000, True, None, 8000.0),
            ("wav16", 16000, True, None, 32000.0),
            ("wav16", 16000, False, None, 32000.0),
            ("wav16", 44100, True, None, 88200.0),
            ("wav16", 0, True, None, 2.0),
            ("copy", 16000, True, 24000.0, 24000.0),
            ("copy", 16000, True, None, 8000.0),
            ("copy", 16000, True, 0.0, 8000.0),
            ("opus", 16000, True, None, 8000.0),
            ("", 16000, True, None, 8000.0),
            (" WAV16 ", 16000, True, None, 32000.0),
        ],
        ids=[
            "mp3-mono",
            "mp3-stereo-doubles",
            "mp3-ignores-sample-rate",
            "wav16-mono",
            "wav16-always-mono",
            "wav16-scales-with-sample-rate",
            "wav16-degenerate-sample-rate",
            "copy-measured-source-rate",
            "copy-without-source-rate",
            "copy-non-positive-source-rate",
            "unknown-format-falls-back-to-mp3",
            "empty-format-falls-back-to-mp3",
            "format-is-case-and-whitespace-insensitive",
        ],
    )
    def test_estimates(
        self,
        fmt: str,
        sample_rate: int,
        mono: bool,
        source_rate: float | None,
        expected: float,
    ) -> None:
        kwargs: dict[str, float] = {}
        if source_rate is not None:
            kwargs["source_bytes_per_second"] = source_rate
        estimate = estimate_output_bytes_per_second(fmt, sample_rate, mono, **kwargs)
        assert estimate == expected
