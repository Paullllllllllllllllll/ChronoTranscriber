"""JSON schema loading and validation utilities.

Provides functions for loading and validating JSON schemas used for
structured outputs in transcription tasks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.config.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

REQUIRED_PROPERTIES = {
    "image_analysis",
    "transcription",
    "no_transcribable_text",
    "transcription_not_possible",
}


def _schemas_dir() -> Path:
    return (PROJECT_ROOT / "schemas").resolve()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
            return result if isinstance(result, dict) else None
    except Exception as e:
        logger.warning("Failed to load schema JSON %s: %s", path, e)
        return None


def _extract_bare_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Accept wrapper form { name, strict, schema: {...} } or bare {...}
    if isinstance(obj, dict) and "schema" in obj and isinstance(obj["schema"], dict):
        return obj["schema"]
    return obj


def _extract_schema_name(obj: Dict[str, Any], fallback: str) -> str:
    if isinstance(obj, dict):
        name = obj.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return fallback


def _has_required_structure(bare: Dict[str, Any]) -> bool:
    try:
        if not isinstance(bare, dict):
            return False
        if bare.get("type") != "object":
            return False
        props = bare.get("properties")
        if not isinstance(props, dict):
            return False
        missing = REQUIRED_PROPERTIES - set(props.keys())
        return not missing
    except Exception:
        return False


def list_schema_options() -> List[Tuple[str, Path]]:
    """
    Discover valid transcription schemas under schemas/.

    Returns a list of (schema_name, file_path) tuples sorted by schema_name.
    Invalid or malformed schema files are ignored with a warning.
    """
    dir_path = _schemas_dir()
    if not dir_path.exists():
        logger.warning("Schemas directory does not exist: %s", dir_path)
        return []

    results: List[Tuple[str, Path]] = []
    for p in sorted(dir_path.glob("*.json")):
        obj = _load_json(p)
        if not obj:
            continue
        bare = _extract_bare_schema(obj)
        if not _has_required_structure(bare):
            logger.warning("Ignoring schema missing required structure: %s", p.name)
            continue
        name = _extract_schema_name(obj, p.stem)
        results.append((name, p))

    # Sort by schema name for stable UI
    results.sort(key=lambda t: t[0].lower())
    return results


def find_schema_path_by_name(name: str) -> Optional[Path]:
    for n, p in list_schema_options():
        if n == name:
            return p
    return None
