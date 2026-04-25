"""Smoke tests for the modules.llm public surface."""

from __future__ import annotations

import pytest

import modules.llm as llm


@pytest.mark.unit
def test_llm_exposes_schema_utils() -> None:
    fn = llm.list_schema_options
    assert callable(fn)
    assert fn.__name__ == "list_schema_options"


@pytest.mark.unit
def test_llm_exposes_transcriber_symbols() -> None:
    cls = llm.LangChainTranscriber
    assert getattr(cls, "__name__", "") == "LangChainTranscriber"


@pytest.mark.unit
def test_llm_exposes_provider_abstraction() -> None:
    assert hasattr(llm, "BaseProvider")
    assert hasattr(llm, "TranscriptionResult")
    assert hasattr(llm, "get_provider")


@pytest.mark.unit
def test_llm_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _ = getattr(llm, "definitely_not_a_symbol")
