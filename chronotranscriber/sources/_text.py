"""Shared plain-text normalization helpers for document extractors."""

from __future__ import annotations

from collections.abc import Sequence


def metadata_header(title: str | None, authors: Sequence[str]) -> str:
    """Return the ``# Title:`` and ``# Author(s):`` lines of an ebook, or ""."""
    lines: list[str] = []
    if title:
        lines.append(f"# Title: {title.strip()}")
    author_line = ", ".join(author.strip() for author in authors if author.strip())
    if author_line:
        lines.append(f"# Author(s): {author_line}")
    return "\n".join(lines)


def ebook_plain_text(
    title: str | None, authors: Sequence[str], sections: Sequence[str]
) -> str:
    """Return the metadata header and the normalized sections as plain text.

    The header and every non-empty section are separated by a blank line;
    a non-empty result ends with one newline.
    """
    blocks = [metadata_header(title, authors)]
    blocks.extend(normalize_text(section) for section in sections)
    rendered = "\n\n".join(block for block in blocks if block)
    return rendered + "\n" if rendered else ""


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

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)
