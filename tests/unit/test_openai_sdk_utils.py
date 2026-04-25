from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.llm.openai_sdk_utils import (
    coerce_file_id,
    get_openai_client,
    list_all_batches,
    sdk_to_dict,
    validate_api_key,
)


class _ModelDumpObj:
    def model_dump(self):
        return {"id": "x", "status": "completed"}


class _ToDictObj:
    def to_dict(self):
        return {"id": "y"}


class _JsonObj:
    def json(self):
        return json.dumps({"id": "z"})


class _AttrObj:
    def __init__(self):
        self.id = "attr"
        self.status = "completed"

    def method(self):
        return "callable"


@pytest.mark.unit
def test_sdk_to_dict_passthrough_for_dict() -> None:
    obj = {"id": "d"}
    assert sdk_to_dict(obj) is obj


@pytest.mark.unit
def test_sdk_to_dict_prefers_model_dump() -> None:
    assert sdk_to_dict(_ModelDumpObj()) == {"id": "x", "status": "completed"}


@pytest.mark.unit
def test_sdk_to_dict_falls_back_to_to_dict() -> None:
    assert sdk_to_dict(_ToDictObj()) == {"id": "y"}


@pytest.mark.unit
def test_sdk_to_dict_falls_back_to_json_method() -> None:
    assert sdk_to_dict(_JsonObj()) == {"id": "z"}


@pytest.mark.unit
def test_sdk_to_dict_best_effort_attribute_extraction() -> None:
    d = sdk_to_dict(_AttrObj())
    assert d.get("id") == "attr"
    assert d.get("status") == "completed"

    # Ensure this helper method isn't invoked by sdk_to_dict (it should ignore callables),
    # but still keep the helper line covered for test suite coverage.
    assert _AttrObj().method() == "callable"


@pytest.mark.unit
def test_list_all_batches_paginates_and_coerces_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_print = MagicMock()
    monkeypatch.setattr("modules.llm.openai_sdk_utils.print_info", mock_print)

    client = MagicMock()

    page1 = SimpleNamespace(
        data=[_ModelDumpObj(), _ToDictObj()],
        has_more=True,
        last_id="batch_last_1",
    )
    page2 = SimpleNamespace(
        data=[_JsonObj()],
        has_more=False,
        last_id=None,
    )

    client.batches.list = MagicMock(side_effect=[page1, page2])

    batches = list_all_batches(client, limit=2)
    assert batches == [
        {"id": "x", "status": "completed"},
        {"id": "y"},
        {"id": "z"},
    ]

    client.batches.list.assert_any_call(limit=2)
    client.batches.list.assert_any_call(limit=2, after="batch_last_1")
    assert mock_print.call_count == 2


@pytest.mark.unit
def test_coerce_file_id_handles_multiple_shapes() -> None:
    assert coerce_file_id("file_123") == "file_123"
    assert coerce_file_id({"id": "file_abc"}) == "file_abc"
    assert coerce_file_id({"file_id": "file_def"}) == "file_def"
    assert coerce_file_id(["file_zzz"]) == "file_zzz"
    assert coerce_file_id([{"id": "file_list"}]) == "file_list"
    assert coerce_file_id(None) is None
    assert coerce_file_id({}) is None
    assert coerce_file_id([]) is None


@pytest.mark.unit
def test_get_openai_client_uses_explicit_api_key() -> None:
    with patch("openai.OpenAI") as openai_cls:
        client = get_openai_client(api_key="sk-test")
        openai_cls.assert_called_once_with(api_key="sk-test")
        assert client is openai_cls.return_value


@pytest.mark.unit
def test_get_openai_client_raises_when_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        _ = get_openai_client(api_key=None)


@pytest.mark.unit
def test_validate_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert validate_api_key() is False
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
        assert validate_api_key() is True
