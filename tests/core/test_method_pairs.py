"""The valid source and method pairs follow the option table."""

from __future__ import annotations

from pathlib import Path

from chronotranscriber.methods import VALID_PAIRS
from chronotranscriber.sources import SOURCE_TYPES
from chronotranscriber.spec import METHODS_BY_TYPE

GOLDEN = Path(__file__).parents[1] / "characterization" / "golden"

RUN_CASES: dict[tuple[str, str], str] = {
    ("pdf", "llm"): "llm/pdf_openai_schema_markdown",
    ("pdf", "tesseract"): "sources/pdf_tesseract",
    ("pdf", "native"): "sources/pdf_native_pages",
    ("images", "llm"): "llm/images_google_schema",
    ("images", "tesseract"): "sources/images_tesseract",
    ("epub", "native"): "sources/epub_native",
    ("mobi", "native"): "sources/mobi_native",
    ("audio", "whisper"): "audio/whisper_single_request",
    ("audio", "audio-api"): "audio/openai_single_request",
    ("auto", "llm"): "sources/auto_mixed_example_config",
    ("auto", "tesseract"): "sources/auto_mixed_local_methods",
}
"""The characterization case that runs each source and method pair end to end."""


def test_the_pairs_cover_every_source_type() -> None:
    assert set(VALID_PAIRS) == set(SOURCE_TYPES)


def test_the_pairs_are_the_option_table_without_auto() -> None:
    expected = {kind: m for kind, m in METHODS_BY_TYPE.items() if kind != "auto"}
    assert dict(VALID_PAIRS) == expected


def test_the_auto_methods_serve_pdfs_and_image_folders() -> None:
    for method in METHODS_BY_TYPE["auto"]:
        assert method in VALID_PAIRS["pdf"]
        assert method in VALID_PAIRS["images"]


def test_every_pair_has_an_end_to_end_run_case() -> None:
    pairs = {(kind, m) for kind, methods in METHODS_BY_TYPE.items() for m in methods}
    unmapped = sorted(pairs - set(RUN_CASES))
    stale = sorted(set(RUN_CASES) - pairs)
    assert not unmapped, f"pairs without a run case: {unmapped}"
    assert not stale, f"run cases for pairs the option table lacks: {stale}"


def test_every_run_case_has_a_golden_directory() -> None:
    missing = sorted(
        case for case in RUN_CASES.values() if not (GOLDEN / case).is_dir()
    )
    assert not missing, f"golden cases missing under {GOLDEN}: {missing}"
