"""Shared plain-text normalization helpers for document extractors."""

from __future__ import annotations


def normalize_text(value: str) -> str:
    """Collapse extraneous whitespace while preserving single blank lines."""
    if not value:
        return ""

    lines = [line.strip() for line in value.splitlines()]
    normalized_lines: list[str] = []

    for line in lines:
        if line:
            normalized_lines.append(line)
        elif normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")

    # Remove trailing blanks
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)
