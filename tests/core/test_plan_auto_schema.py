"""Schema and context selection of auto-mode runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.llm.prompts import SCHEMAS_DIR
from chronotranscriber.run import ConfigurationError, RunPlan, plan
from tests.conftest import make_settings, make_spec


@pytest.fixture
def auto_input(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a folder holding one image folder; the OpenAI key is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    folder = temp_dir / "test_folder"
    folder.mkdir()
    (folder / "page_001.png").write_bytes(b"")
    return temp_dir


def _plan(auto_input: Path, **flags: Any) -> RunPlan:
    return plan(make_spec(auto_input, type="auto", **flags), make_settings())


@pytest.mark.unit
def test_a_schema_name_is_looked_up(auto_input: Path) -> None:
    run_plan = _plan(
        auto_input,
        output=auto_input / "output",
        schema="plain_text_transcription_schema",
    )

    assert run_plan.auto
    assert (
        run_plan.schema_path
        == (SCHEMAS_DIR / "plain_text_transcription_schema.json").resolve()
    )


@pytest.mark.unit
def test_a_schema_path_is_accepted(auto_input: Path) -> None:
    schema = SCHEMAS_DIR / "markdown_transcription_schema.json"
    assert _plan(auto_input, schema=str(schema)).schema_path == schema.resolve()


@pytest.mark.unit
def test_an_unknown_schema_is_an_error(auto_input: Path) -> None:
    with pytest.raises(ConfigurationError, match="'nonexistent_schema' not found"):
        _plan(auto_input, schema="nonexistent_schema")


@pytest.mark.unit
def test_the_default_schema_applies(auto_input: Path) -> None:
    run_plan = _plan(auto_input)

    assert run_plan.schema_path is not None
    assert run_plan.schema_path.name == "markdown_transcription_schema.json"


@pytest.mark.unit
def test_a_context_file_is_used(auto_input: Path) -> None:
    context_file = auto_input / "context.txt"
    context_file.write_text("Test context", encoding="utf-8")

    (item,) = _plan(auto_input, context=context_file).to_process

    assert item.method == "llm"
    assert item.context == "Test context"


@pytest.mark.unit
def test_context_none_turns_the_lookup_off(auto_input: Path) -> None:
    (auto_input / "test_folder_transcr_context.txt").write_text(
        "Folder context", encoding="utf-8"
    )

    (item,) = _plan(auto_input, context="none").to_process
    (looked_up,) = _plan(auto_input).to_process

    assert item.context is None
    assert looked_up.context == "Folder context"
