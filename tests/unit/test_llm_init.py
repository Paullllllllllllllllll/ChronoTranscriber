from __future__ import annotations

import pytest

import modules.llm as llm


@pytest.mark.unit
def test_llm_dunder_getattr_resolves_schema_utils() -> None:
    fn = llm.list_schema_options
    assert callable(fn)
    assert fn.__name__ == "list_schema_options"


@pytest.mark.unit
def test_llm_dunder_getattr_resolves_model_capabilities() -> None:
    fn = llm.ensure_image_support
    assert callable(fn)
    assert fn.__name__ == "ensure_image_support"


@pytest.mark.unit
def test_llm_dunder_getattr_resolves_transcriber_symbols() -> None:
    cls = llm.LangChainTranscriber
    assert getattr(cls, "__name__", "") == "LangChainTranscriber"


@pytest.mark.unit
def test_llm_dunder_getattr_unknown_raises() -> None:
    with pytest.raises(AttributeError):
        _ = getattr(llm, "definitely_not_a_symbol")
