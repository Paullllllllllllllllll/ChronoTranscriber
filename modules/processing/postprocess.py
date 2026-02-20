"""Post-processing utilities for OCR/transcription output cleanup.

This module provides text normalization and cleanup functions for transcription
outputs. It is designed to be conservative, removing clearly spurious artifacts
while preserving semantic content (including Markdown formatting and emojis).

Processing stages:
1. Unicode normalization - NFC normalization, removal of control characters
2. Hyphenation merging - Optional merging of line-break hyphenated words
3. Whitespace normalization - Tab expansion, space collapsing, blank line limits
4. Line wrapping - Optional word-based wrapping with smart detection

Based on postprocess_chronominer_ocr.py from MiscellaneousHelperScripts.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# High-plane icon glyphs sometimes seen in OCR outputs (e.g., Michelin guides),
# used as bullets or check marks. Map these to a standard bullet character.
AEGEAN_ICON_CODEPOINTS = {
    0x10101,  # AEGEAN WORD SEPARATOR DOT
    0x10102,  # AEGEAN CHECK MARK
    0x10103,
    0x10104,
    0x10105,
}

# Pattern for detecting line-break hyphenation: word-hyphen-newline-word
# Requires at least 3 chars before hyphen and 2 after for safety
_HYPHEN_PATTERN = re.compile(r"(\w{3,})-\n(\w{2,})")


def normalize_unicode_text(text: str) -> str:
    """
    Unicode normalization and removal of clearly spurious characters.

    Steps:
    - NFC normalization so that accents use composed characters.
    - Map some high-plane icon-like glyphs used for bullets/card logos
      to a generic bullet character.
    - Drop soft hyphens, zero-width spaces and BOMs.
    - Remove all remaining control/format/surrogate/unassigned chars
      except newline and tab.
    
    Args:
        text: Input text to normalize.
        
    Returns:
        Normalized text with spurious characters removed.
    """
    # Normalize accents to composed form (NFC)
    text = unicodedata.normalize("NFC", text)

    # Map rare icon glyphs to a simple bullet
    translation = {cp: "•" for cp in AEGEAN_ICON_CODEPOINTS}
    text = text.translate(translation)

    # Explicitly drop some frequent layout artifacts
    drop_chars = {
        "\u00AD",  # SOFT HYPHEN - invisible hyphenation hint
        "\u200B",  # ZERO WIDTH SPACE - invisible separator
        "\ufeff",  # BOM - byte order mark
    }
    for ch in drop_chars:
        text = text.replace(ch, "")

    # Remove other control / format / unassigned chars safely
    out_chars: list[str] = []
    for ch in text:
        # Preserve newlines and tabs
        if ch in ("\n", "\t"):
            out_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        # Skip all other control/format/surrogate/unassigned chars (category starts with 'C')
        if cat.startswith("C"):
            continue
        out_chars.append(ch)
    
    return "".join(out_chars)


def fix_hyphenation(text: str) -> str:
    """
    Merge words split across lines with a hyphen at the end of the line.

    Example:
        "politi-\\nche" -> "politiche"

    To avoid damaging genuine hyphenated compounds such as "Jean-Baptiste",
    merging is restricted to cases where both sides look like lower-case
    word fragments.
    
    Args:
        text: Input text with potential line-break hyphenations.
        
    Returns:
        Text with hyphenated line breaks merged where appropriate.
    """

    def repl(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        # Only merge if both parts end/start with lowercase letters
        # This preserves proper names like "Jean-Baptiste"
        if (
            left[-1].isalpha()
            and right[0].isalpha()
            and left[-1].islower()
            and right[0].islower()
        ):
            return left + right
        # Fall back to original spelling (do not merge)
        return left + "-\n" + right

    return _HYPHEN_PATTERN.sub(repl, text)


def normalize_spacing(
    text: str,
    collapse_internal: bool = True,
    max_blank_lines: int = 2,
    tab_size: int = 4,
) -> str:
    """
    Whitespace normalization.

    - Tabs are expanded to a fixed number of spaces.
    - Trailing spaces on each line are stripped.
    - Internal runs of spaces between non-space characters are capped
      at two (to keep some room for tables and leader dots).
    - Runs of blank lines are limited.
    
    Args:
        text: Input text to normalize.
        collapse_internal: If True, collapse runs of 3+ internal spaces to 2.
        max_blank_lines: Maximum number of consecutive blank lines to keep.
        tab_size: Number of spaces to expand each tab into.
        
    Returns:
        Text with normalized whitespace.
    """
    # Expand tabs to spaces
    text = text.expandtabs(tab_size)

    lines = text.splitlines()
    new_lines: list[str] = []
    blank_run = 0

    for line in lines:
        # Strip trailing spaces but keep leading indentation
        line = line.rstrip(" ")

        if collapse_internal:
            # Collapse runs of 2+ spaces between non-space chars to exactly 2
            line = re.sub(r"(?<=\S) {2,}(?=\S)", "  ", line)

        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_blank_lines:
                new_lines.append("")
            # Extra blank lines are skipped
        else:
            blank_run = 0
            new_lines.append(line)

    # Trailing newline at EOF for POSIX-friendliness
    return "\n".join(new_lines) + "\n"


def should_wrap_line(line: str) -> bool:
    """
    Heuristic filter for lines that should NOT be wrapped.

    We avoid wrapping:
    - Empty or whitespace-only lines
    - Markdown headings (# ...)
    - Page-number markers (<page_number> ...)
    - Image annotations (![Image: ...] or legacy [Image: ...])
    - Markdown-style table rows (| ... |)
    - Lines with failure placeholders ([transcription error], etc.)
    
    Args:
        line: Line to check.
        
    Returns:
        True if the line should be wrapped, False if it should be left alone.
    """
    if not line.strip():
        return False

    stripped = line.lstrip()

    # Markdown heading
    if stripped.startswith("#"):
        return False

    # Page number markers (ChronoTranscriber format)
    if stripped.startswith("<page_number>"):
        return False

    # Image description blocks from the OCR pipeline (Markdown format ![...] and legacy [...])
    if stripped.startswith("!["):
        return False
    if stripped.startswith("[") and "Image:" in stripped and stripped.endswith("]"):
        return False

    # Failure placeholders - don't wrap these
    lower = stripped.lower()
    if "[transcription error" in lower:
        return False
    if "[no transcribable text" in lower:
        return False
    if "[transcription not possible" in lower:
        return False

    # Simple heuristic for Markdown table rows
    if stripped.startswith("|") and stripped.endswith("|"):
        return False

    # Image name headers (e.g., "page_001.jpg: [transcription...]")
    if ":" in stripped and stripped.index(":") < 50:
        after_colon = stripped[stripped.index(":") + 1:].strip()
        if after_colon.startswith("[") and after_colon.endswith("]"):
            return False

    return True


def compute_auto_wrap_width(text: str) -> int:
    """
    Compute an automatic wrapping width based on the average line length
    of text blocks.

    A block is defined as a maximal sequence of non-empty lines.
    For each block we compute the mean line length; the global wrap
    width is then the mean of these block means, rounded to the nearest
    integer.

    Only blocks with at least three non-empty lines are considered,
    to avoid headings and isolated lines distorting the estimate.
    
    Args:
        text: Input text to analyze.
        
    Returns:
        Computed wrap width (minimum 20, default 80 if no valid blocks).
    """
    lines = text.splitlines()

    blocks: list[list[int]] = []
    current_block: list[int] = []

    for line in lines:
        if line.strip():
            current_block.append(len(line))
        else:
            if current_block:
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)

    block_means: list[float] = []
    for block in blocks:
        if len(block) < 3:
            continue
        block_means.append(sum(block) / float(len(block)))

    if not block_means:
        # Fallback to a conservative default if the text is too sparse
        return 80

    avg = sum(block_means) / float(len(block_means))
    return max(20, int(round(avg)))


def wrap_long_lines(text: str, width: int) -> str:
    """
    Wrap lines longer than `width` characters, using a simple word-wrap
    algorithm that preserves leading indentation and skips structured
    lines deemed unsuitable for wrapping.

    Wrapping is performed after whitespace normalization so that the
    line lengths reflect the final spacing.
    
    Args:
        text: Input text to wrap.
        width: Target line width in characters.
        
    Returns:
        Text with long lines wrapped.
    """
    if width <= 0:
        return text

    lines = text.splitlines()
    wrapped_lines: list[str] = []

    for line in lines:
        # Do not wrap structural or short lines
        if len(line) <= width or not should_wrap_line(line):
            wrapped_lines.append(line)
            continue

        # Preserve leading spaces as indentation
        indent_len = len(line) - len(line.lstrip(" "))
        indent = line[:indent_len]
        content = line[indent_len:].strip()

        if not content:
            wrapped_lines.append(line)
            continue

        max_content_width = max(1, width - indent_len)

        while len(content) > max_content_width:
            # Try to break at the last space before the limit
            break_pos = content.rfind(" ", 0, max_content_width + 1)
            if break_pos <= 0:
                # No space found; hard break
                break_pos = max_content_width
            segment = content[:break_pos].rstrip()
            wrapped_lines.append(indent + segment)
            content = content[break_pos:].lstrip()

        wrapped_lines.append(indent + content)

    return "\n".join(wrapped_lines) + "\n"


def postprocess_text(
    text: str,
    merge_hyphenation: bool = False,
    collapse_internal_spaces: bool = True,
    max_blank_lines: int = 2,
    tab_size: int = 4,
    wrap_lines: bool = False,
    wrap_width: Optional[int] = None,
    auto_wrap: bool = False,
) -> str:
    """
    Run the full post-processing pipeline on a text string.

    Args:
        text: Input text.
        merge_hyphenation: Merge words split by hyphen + newline when they
            look like lower-case word fragments.
        collapse_internal_spaces: If True, collapse sequences of three or more
            internal spaces between non-space characters to two spaces.
        max_blank_lines: Maximum number of consecutive blank lines to keep.
        tab_size: Number of spaces used to expand each tab character.
        wrap_lines: If True, perform line wrapping after spacing normalization.
        wrap_width: Explicit target width for wrapping. If None and wrap_lines
            is True and auto_wrap is False, a default of 80 is used.
        auto_wrap: If True and wrap_lines is True, compute wrap_width
            automatically from text blocks.
            
    Returns:
        Post-processed text.
    """
    # 1. Unicode normalization and control-character cleanup
    text = normalize_unicode_text(text)

    # 2. Optional hyphenation merging
    if merge_hyphenation:
        text = fix_hyphenation(text)

    # 3. Whitespace normalization
    text = normalize_spacing(
        text,
        collapse_internal=collapse_internal_spaces,
        max_blank_lines=max_blank_lines,
        tab_size=tab_size,
    )

    # 4. Optional line wrapping
    if wrap_lines:
        if auto_wrap:
            width = compute_auto_wrap_width(text)
        else:
            width = wrap_width if wrap_width is not None else 80
        text = wrap_long_lines(text, width)

    return text


def postprocess_transcription(
    text: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Apply post-processing to transcription text using configuration settings.
    
    This is the main entry point for the post-processing pipeline, designed
    to be called from the transcription workflow.
    
    Args:
        text: Transcription text to process.
        config: Post-processing configuration dictionary. If None, uses defaults.
            Expected keys match image_processing_config.yaml postprocessing section:
            - enabled: bool (master toggle, default True when called directly)
            - merge_hyphenation: bool
            - collapse_internal_spaces: bool
            - max_blank_lines: int
            - tab_size: int
            - wrap_lines: bool
            - auto_wrap: bool
            - wrap_width: int or None
            
    Returns:
        Post-processed text, or original text if processing is disabled.
    """
    if config is None:
        config = {}
    
    # Check if post-processing is enabled (default True when called directly)
    if not config.get("enabled", True):
        return text
    
    # Extract settings with defaults
    merge_hyphenation = config.get("merge_hyphenation", False)
    collapse_internal_spaces = config.get("collapse_internal_spaces", True)
    max_blank_lines = config.get("max_blank_lines", 2)
    tab_size = config.get("tab_size", 4)
    wrap_lines = config.get("wrap_lines", False)
    auto_wrap = config.get("auto_wrap", False)
    wrap_width = config.get("wrap_width")
    
    # Convert wrap_width to int or None
    if wrap_width is not None:
        try:
            wrap_width = int(wrap_width)
        except (ValueError, TypeError):
            wrap_width = None
    
    return postprocess_text(
        text,
        merge_hyphenation=merge_hyphenation,
        collapse_internal_spaces=collapse_internal_spaces,
        max_blank_lines=max_blank_lines,
        tab_size=tab_size,
        wrap_lines=wrap_lines,
        wrap_width=wrap_width,
        auto_wrap=auto_wrap,
    )


def postprocess_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    in_place: bool = False,
) -> Path:
    """
    Post-process a transcription file.
    
    Args:
        input_path: Path to the input file.
        output_path: Path to write output. If None and not in_place, returns
            processed text without writing.
        config: Post-processing configuration dictionary.
        in_place: If True, overwrite the input file.
        
    Returns:
        Path to the output file (input_path if in_place, output_path otherwise).
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If neither output_path nor in_place is specified.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None and not in_place:
        raise ValueError("Either output_path or in_place=True must be specified")
    
    # Read input
    text = input_path.read_text(encoding="utf-8", errors="replace")
    
    # Process
    processed = postprocess_transcription(text, config)
    
    # Write output
    target_path = input_path if in_place else output_path
    target_path.write_text(processed, encoding="utf-8", newline="\n")
    
    logger.info(f"Post-processed {'in place' if in_place else 'to'}: {target_path}")
    
    return target_path
