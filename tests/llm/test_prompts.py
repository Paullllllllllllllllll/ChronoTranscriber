"""Schema lookup of chronotranscriber.llm.prompts.

``list_schema_options`` lists the valid schemas in ``schemas/``; every
returned path exists and the default markdown schema is among them.
"""

from __future__ import annotations

import pytest

from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH, list_schema_options


@pytest.mark.unit
def test_schema_paths_exist() -> None:
    """Every listed schema path points at an existing file."""
    options = list_schema_options()
    for name, path in options:
        assert path.exists(), f"Schema {name} path does not exist: {path}"


@pytest.mark.unit
def test_default_schema_is_listed() -> None:
    """The default markdown schema passes validation and is offered."""
    paths = [path for _, path in list_schema_options()]
    assert DEFAULT_SCHEMA_PATH in paths
