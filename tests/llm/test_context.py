"""Transcription context: text or a file, sidecars, and the fallback context."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronotranscriber.llm.prompts import (
    ContextError,
    fallback_context,
    item_context,
)
from chronotranscriber.spec import CONTEXT_NONE, OPTIONS, resolve_spec


@pytest.fixture
def item(tmp_path: Path) -> Path:
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    return path


def test_context_text_is_sent_as_given(item: Path) -> None:
    (item.parent / "book_transcr_context.txt").write_text("sidecar", encoding="utf-8")

    assert item_context(item, "ledgers, Basel, 1780s") == "ledgers, Basel, 1780s"


def test_a_context_path_is_read(item: Path) -> None:
    context = item.parent / "topics.txt"
    context.write_text("  from a file\n", encoding="utf-8")

    assert item_context(item, str(context)) == "from a file"


def test_a_missing_context_file_is_an_error(item: Path) -> None:
    with pytest.raises(ContextError, match="not found"):
        item_context(item, str(item.parent / "typo.txt"))


def test_the_sidecar_wins_over_the_fallback(item: Path) -> None:
    (item.parent / "book_transcr_context.txt").write_text("sidecar", encoding="utf-8")

    assert item_context(item, None, "fallback topics") == "sidecar"


def test_the_fallback_serves_an_item_without_a_sidecar(item: Path) -> None:
    assert item_context(item, None, "fallback topics") == "fallback topics"
    assert item_context(item, CONTEXT_NONE, "fallback topics") is None


def test_no_fallback_gives_no_context(item: Path) -> None:
    assert item_context(item, None) is None


def test_fallback_context_reads_a_file_and_takes_text(tmp_path: Path) -> None:
    context = tmp_path / "fallback.txt"
    context.write_text("file topics\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    assert fallback_context(str(context)) == ("file topics", None)
    assert fallback_context("plain topics") == ("plain topics", None)
    assert fallback_context(str(empty)) == (None, None)
    assert fallback_context(None) == (None, None)


def test_a_missing_fallback_file_warns(tmp_path: Path) -> None:
    text, warning = fallback_context(str(tmp_path / "context" / "missing.txt"))

    assert text is None
    assert warning is not None and "missing.txt" in warning


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("auto", None),
        ("none", CONTEXT_NONE),
        ("Basel ledgers", "Basel ledgers"),
        ("ctx.txt", "ctx.txt"),
    ],
)
def test_context_values_convert(given: str, expected: str | None) -> None:
    assert OPTIONS["context"].convert(given) == expected


def test_fallback_none_undoes_a_settings_fallback() -> None:
    given = OPTIONS["fallback_context"].convert("none")
    spec = resolve_spec(
        {"input": Path("doc.pdf"), "fallback_context": given},
        {"fallback_context": "context/transcr_context.txt"},
    ).spec

    assert spec.prompt.fallback_context is None
