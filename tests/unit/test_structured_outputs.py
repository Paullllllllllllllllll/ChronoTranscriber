from __future__ import annotations

import pytest

from modules.llm.structured_outputs import _unwrap_schema, build_structured_text_format


@pytest.mark.unit
def test_unwrap_schema_wrapper_dict_returns_name_schema_strict() -> None:
    name, schema, strict = _unwrap_schema(
        {
            "name": "MySchema",
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "strict": False,
        }
    )
    assert name == "MySchema"
    assert schema["type"] == "object"
    assert strict is False


@pytest.mark.unit
def test_unwrap_schema_bare_schema_uses_default_name_and_strict() -> None:
    name, schema, strict = _unwrap_schema({"type": "object", "properties": {}})
    assert name == "TranscriptionSchema"
    assert schema["type"] == "object"
    assert strict is True


@pytest.mark.unit
def test_unwrap_schema_invalid_input_returns_empty_schema() -> None:
    name, schema, strict = _unwrap_schema({})
    assert name == "TranscriptionSchema"
    assert schema == {}
    assert strict is True


@pytest.mark.unit
def test_build_structured_text_format_returns_none_for_empty_schema() -> None:
    assert build_structured_text_format({}) is None


@pytest.mark.unit
def test_build_structured_text_format_returns_expected_format() -> None:
    fmt = build_structured_text_format(
        {"name": "X", "schema": {"type": "object", "properties": {}}, "strict": True}
    )
    assert fmt == {
        "type": "json_schema",
        "name": "X",
        "schema": {"type": "object", "properties": {}},
        "strict": True,
    }
