"""Prompt inputs of LLM transcription: system prompt, schema and context.

The system prompt comes from ``--prompt`` or the prompt of the response
format; the schema from ``--schema``, a name found in ``schemas/`` or a
path. Context text is resolved per item through
:mod:`chronotranscriber.common.context`: ``--context`` (text, or a path to
a text file), else the item's ``_transcr_context`` sidecar, else
``--fallback-context``. The context image takes ``--context-image``, else
the item's ``_transcr_context_image`` sidecar, else the project's default
image in ``context/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from chronotranscriber.common.context import (
    TEXT_EXTENSIONS,
    ContextError,
    read_context_file,
    resolve_context,
    resolve_context_image,
)
from chronotranscriber.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronotranscriber.spec import CONTEXT_NONE, ContextChoice, ImageChoice

__all__ = [
    "CONTEXT_DIR",
    "CONTEXT_SUFFIX",
    "DEFAULT_SCHEMA_PATH",
    "IMAGE_CONTEXT_SUFFIX",
    "PLAIN_PROMPT_PATH",
    "SCHEMAS_DIR",
    "SCHEMA_PROMPT_PATH",
    "ContextError",
    "fallback_context",
    "item_context",
    "item_context_image",
    "list_schema_options",
    "resolve_schema",
    "system_prompt_path",
]

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _PACKAGE_DIR.parent
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
PROMPTS_DIR = _PACKAGE_DIR / "resources" / "prompts"
SCHEMA_PROMPT_PATH = PROMPTS_DIR / "transcription_prompt_schema.txt"
PLAIN_PROMPT_PATH = PROMPTS_DIR / "transcription_prompt_plain.txt"
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "markdown_transcription_schema.json"

CONTEXT_DIR = PROJECT_ROOT / "context"
"""The project context folder with the default context image."""

CONTEXT_SUFFIX = "_transcr_context"
IMAGE_CONTEXT_SUFFIX = "_transcr_context_image"
_DEFAULT_NAME = "transcr_context"
_IMAGE_EXTENSIONS = tuple(sorted(SUPPORTED_IMAGE_EXTENSIONS))

REQUIRED_PROPERTIES = frozenset(
    {"transcription", "no_transcribable_text", "transcription_not_possible"}
)


def system_prompt_path(prompt: Path | None) -> Path | None:
    """Return the ``--prompt`` file resolved against the current directory.

    None selects the prompt of the response format.

    Raises:
        FileNotFoundError: The file does not exist.
    """
    if prompt is None:
        return None
    resolved = prompt.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"System prompt missing: {resolved}")
    return resolved


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load schema JSON %s: %s", path, exc)
        return None
    return result if isinstance(result, dict) else None


def _bare_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """Return the schema of a ``{name, strict, schema}`` wrapper, or *obj*."""
    inner = obj.get("schema")
    return inner if isinstance(inner, dict) else obj


def _schema_name(obj: dict[str, Any], fallback: str) -> str:
    name = obj.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def _has_required_structure(bare: dict[str, Any]) -> bool:
    properties = bare.get("properties")
    return (
        bare.get("type") == "object"
        and isinstance(properties, dict)
        and set(properties) >= REQUIRED_PROPERTIES
    )


def list_schema_options() -> list[tuple[str, Path]]:
    """Return ``(name, path)`` of every valid schema in ``schemas/``, by name.

    Files that do not parse or lack the transcription fields are skipped
    with a warning.
    """
    folder = SCHEMAS_DIR.resolve()
    if not folder.exists():
        logger.warning("Schemas directory does not exist: %s", folder)
        return []
    options: list[tuple[str, Path]] = []
    for path in sorted(folder.glob("*.json")):
        obj = _load_json(path)
        if not obj:
            continue
        if not _has_required_structure(_bare_schema(obj)):
            logger.warning("Ignoring schema missing required structure: %s", path.name)
            continue
        options.append((_schema_name(obj, path.stem), path))
    options.sort(key=lambda option: option[0].lower())
    return options


def resolve_schema(value: str) -> tuple[str, Path]:
    """Return the name and file of a schema given by name or by path.

    A name is looked up among :func:`list_schema_options`; a relative path
    resolves against the current directory.

    Raises:
        ValueError: Neither a known schema name nor an existing file.
    """
    options = dict(list_schema_options())
    if value in options:
        return value, options[value]
    candidate = Path(value).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
        return _schema_name(_load_json(resolved) or {}, resolved.stem), resolved
    raise ValueError(f"Schema '{value}' not found. Available: {list(options)}")


def _names_file(value: str) -> bool:
    """Whether *value* reads as a context file name rather than topics."""
    text = value.strip()
    return "\n" not in text and Path(text).suffix.lower() in TEXT_EXTENSIONS


def _is_file(path: Path) -> bool:
    """Whether *path* is a file; text that is no valid path is none."""
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


def _text_or_file(value: str | Path) -> str | Path:
    """Return *value* as a path when it names an existing file, else as text.

    A :class:`~pathlib.Path` is always a file.

    Raises:
        ContextError: *value* names a text file that does not exist.
    """
    candidate = Path(str(value).strip()).expanduser()
    if _is_file(candidate):
        return candidate.resolve()
    if isinstance(value, Path) or _names_file(value):
        raise ContextError(f"Context file not found: {candidate}")
    return value


def fallback_context(value: str | None) -> tuple[str | None, str | None]:
    """Return the ``--fallback-context`` text and a warning, if any.

    A path to an existing file is read; an empty file gives no fallback. A
    text file name that does not exist gives none either, with a warning, so
    the relative path of the packaged settings does not stop a run started
    outside the repository. Anything else is context text.
    """
    if value is None or not value.strip():
        return None, None
    candidate = Path(value.strip()).expanduser()
    if _is_file(candidate):
        return read_context_file(candidate.resolve()), None
    if _names_file(value):
        return None, (
            f"Fallback context file not found: {candidate.resolve()}; items "
            "without a sidecar get no context."
        )
    return value, None


def _default_image() -> Path | None:
    for ext in _IMAGE_EXTENSIONS:
        candidate = CONTEXT_DIR / f"{_DEFAULT_NAME}_image{ext}"
        if candidate.is_file():
            return candidate
    return None


def item_context(
    item: Path, choice: ContextChoice, fallback: str | None = None
) -> str | None:
    """Return the context text of *item* for the ``--context`` *choice*.

    Text is sent as given and a path to a text file is read; ``none`` sends
    no context; the automatic lookup takes the item's sidecar (beside a
    file, or beside its folder; beside an image folder), else *fallback*,
    the text :func:`fallback_context` returned.

    Raises:
        ContextError: The given file is missing, empty or unreadable.
    """
    if choice == CONTEXT_NONE:
        return None
    explicit = _text_or_file(choice) if choice is not None else None
    resolved = resolve_context(
        item,
        CONTEXT_SUFFIX,
        explicit=explicit,
        default=fallback if explicit is None else None,
        use_sidecars=explicit is None,
        folder_sidecar_for_directories=False,
    )
    return resolved.text if resolved is not None else None


def item_context_image(item: Path, choice: ImageChoice) -> Path | None:
    """Return the context image of *item* for the ``--context-image`` *choice*.

    A given image is used as is; ``none`` sends no image; the automatic
    lookup takes the item's sidecar image, else the project's default image.

    Raises:
        ContextError: The given image is missing or not a supported image.
    """
    if choice == CONTEXT_NONE:
        return None
    explicit = choice.expanduser().resolve() if isinstance(choice, Path) else None
    resolved = resolve_context_image(
        item,
        IMAGE_CONTEXT_SUFFIX,
        explicit=explicit,
        default=None if explicit is not None else _default_image(),
        use_sidecars=explicit is None,
        folder_sidecar_for_directories=False,
        extensions=_IMAGE_EXTENSIONS,
    )
    return resolved[1] if resolved is not None else None
