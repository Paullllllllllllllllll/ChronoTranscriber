"""Replace external objects by identity wherever the tool binds them.

The fakes name what they replace by the object's public home (such as
``langchain_openai.ChatOpenAI``) or, for a tool function, by its name alone,
never by the tool module that imports it. ``replace`` swaps the object in its
public home, in the module that defines it and in every tool module that
binds it under any name; the test's ``MonkeyPatch`` restores all of them.
Every tool module is imported first, so no module imported later during a
test can keep a stand-in bound after the test.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
import sys
from types import ModuleType

import pytest

TOOL_PACKAGE = "chronotranscriber"


@functools.cache
def load_tool_modules() -> tuple[str, ...]:
    """Import every module of the tool package once; return their names.

    ``__main__`` modules and modules whose optional dependencies are missing
    are skipped.
    """
    package = importlib.import_module(TOOL_PACKAGE)
    names = [TOOL_PACKAGE]
    for info in pkgutil.walk_packages(
        package.__path__, prefix=f"{TOOL_PACKAGE}.", onerror=lambda _name: None
    ):
        if info.name.rsplit(".", 1)[-1] == "__main__":
            continue
        try:
            importlib.import_module(info.name)
        except ImportError:
            continue
        names.append(info.name)
    return tuple(names)


def tool_modules() -> list[ModuleType]:
    """Return every loaded module of the tool package."""
    load_tool_modules()
    prefix = f"{TOOL_PACKAGE}."
    return [
        module
        for name, module in list(sys.modules.items())
        if module is not None and (name == TOOL_PACKAGE or name.startswith(prefix))
    ]


def tool_definition(name: str) -> tuple[str, str]:
    """Return ``(module, name)`` of the one tool module that defines *name*.

    Raises ``LookupError`` when no tool module or more than one defines it.
    """
    found = sorted(
        module.__name__
        for module in tool_modules()
        if getattr(vars(module).get(name), "__module__", None) == module.__name__
    )
    if len(found) != 1:
        raise LookupError(f"{name!r} is defined in {len(found)} tool modules: {found}")
    return found[0], name


def replace(
    monkeypatch: pytest.MonkeyPatch, module_name: str, name: str, stand_in: object
) -> object:
    """Replace the object ``module_name.name`` holds wherever it is bound.

    The object is swapped in *module_name*, in its defining module and in
    every tool module attribute that holds it. Returns the replaced object.
    """
    load_tool_modules()
    module = importlib.import_module(module_name)
    target = getattr(module, name)
    monkeypatch.setattr(module, name, stand_in)
    defining = sys.modules.get(str(getattr(target, "__module__", "")))
    if defining is not None and vars(defining).get(name) is target:
        monkeypatch.setattr(defining, name, stand_in)
    for tool_module in tool_modules():
        for attribute, value in list(vars(tool_module).items()):
            if value is target:
                monkeypatch.setattr(tool_module, attribute, stand_in)
    return target


def replace_tool_function(
    monkeypatch: pytest.MonkeyPatch, name: str, stand_in: object
) -> object:
    """Replace the tool function *name* wherever the tool binds it."""
    module_name, _ = tool_definition(name)
    return replace(monkeypatch, module_name, name, stand_in)


__all__ = [
    "TOOL_PACKAGE",
    "load_tool_modules",
    "replace",
    "replace_tool_function",
    "tool_definition",
    "tool_modules",
]
