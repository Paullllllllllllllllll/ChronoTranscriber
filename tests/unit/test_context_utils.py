"""Unit tests for modules/llm/context_utils.py.

Tests the 3-level hierarchy:
1. File-specific:   {input_stem}_transcr_context.txt   next to the input file
2. Folder-specific: {parent_folder}_transcr_context.txt next to the input's parent folder
3. General fallback: context/transcr_context.txt        in the project context directory

Suffix: transcr_context
"""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# _resolve_context (generic engine)
# ---------------------------------------------------------------------------

class TestResolveContext:
    """Tests for _resolve_context hierarchical resolution."""

    @pytest.mark.unit
    def test_file_specific_found(self, temp_dir):
        """File-specific context is returned when present."""
        from modules.llm.context_utils import _resolve_context

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy pdf")
        ctx = temp_dir / "document_transcr_context.txt"
        ctx.write_text("File-specific context")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=temp_dir)
        assert content == "File-specific context"
        assert path == ctx

    @pytest.mark.unit
    def test_folder_specific_found(self, temp_dir):
        """Folder-specific context is returned when no file-specific exists."""
        from modules.llm.context_utils import _resolve_context

        subfolder = temp_dir / "my_archive"
        subfolder.mkdir()
        pdf = subfolder / "document.pdf"
        pdf.write_text("dummy pdf")
        ctx = temp_dir / "my_archive_transcr_context.txt"
        ctx.write_text("Folder-specific context")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=temp_dir)
        assert content == "Folder-specific context"
        assert path == ctx

    @pytest.mark.unit
    def test_general_fallback(self, temp_dir):
        """General fallback is returned when no file/folder context exists."""
        from modules.llm.context_utils import _resolve_context

        ctx_dir = temp_dir / "context"
        ctx_dir.mkdir()
        general = ctx_dir / "transcr_context.txt"
        general.write_text("General context")

        deep = temp_dir / "a" / "b"
        deep.mkdir(parents=True)
        pdf = deep / "document.pdf"
        pdf.write_text("dummy")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=ctx_dir)
        assert content == "General context"
        assert path == general

    @pytest.mark.unit
    def test_file_wins_over_folder(self, temp_dir):
        """File-specific context takes precedence over folder-specific."""
        from modules.llm.context_utils import _resolve_context

        subfolder = temp_dir / "archive"
        subfolder.mkdir()
        pdf = subfolder / "doc.pdf"
        pdf.write_text("dummy")

        file_ctx = subfolder / "doc_transcr_context.txt"
        file_ctx.write_text("file wins")
        folder_ctx = temp_dir / "archive_transcr_context.txt"
        folder_ctx.write_text("folder loses")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=temp_dir)
        assert content == "file wins"
        assert path == file_ctx

    @pytest.mark.unit
    def test_folder_wins_over_general(self, temp_dir):
        """Folder-specific context takes precedence over general fallback."""
        from modules.llm.context_utils import _resolve_context

        ctx_dir = temp_dir / "context"
        ctx_dir.mkdir()
        general = ctx_dir / "transcr_context.txt"
        general.write_text("general ctx")

        subfolder = temp_dir / "archive"
        subfolder.mkdir()
        pdf = subfolder / "doc.pdf"
        pdf.write_text("dummy")
        folder_ctx = temp_dir / "archive_transcr_context.txt"
        folder_ctx.write_text("folder wins")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=ctx_dir)
        assert content == "folder wins"
        assert path == folder_ctx

    @pytest.mark.unit
    def test_returns_none_when_no_context(self, temp_dir):
        """Returns (None, None) when no context exists anywhere."""
        from modules.llm.context_utils import _resolve_context

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy pdf")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=temp_dir)
        assert content is None
        assert path is None

    @pytest.mark.unit
    def test_empty_file_skipped(self, temp_dir):
        """Empty context file is skipped."""
        from modules.llm.context_utils import _resolve_context

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy")
        ctx = temp_dir / "document_transcr_context.txt"
        ctx.write_text("")

        content, path = _resolve_context("transcr_context", input_path=pdf, context_dir=temp_dir)
        assert content is None
        assert path is None

    @pytest.mark.unit
    def test_directory_input_folder_specific(self, temp_dir):
        """For directory inputs, folder-specific context lives next to the folder."""
        from modules.llm.context_utils import _resolve_context

        folder = temp_dir / "my_images"
        folder.mkdir()
        ctx = temp_dir / "my_images_transcr_context.txt"
        ctx.write_text("Folder context")

        content, path = _resolve_context("transcr_context", input_path=folder, context_dir=temp_dir)
        assert content == "Folder context"
        assert path == ctx

    @pytest.mark.unit
    def test_no_input_path_uses_general(self, temp_dir):
        """General fallback is used when no input_path is provided."""
        from modules.llm.context_utils import _resolve_context

        ctx_dir = temp_dir / "context"
        ctx_dir.mkdir()
        general = ctx_dir / "transcr_context.txt"
        general.write_text("fallback only")

        content, path = _resolve_context("transcr_context", input_path=None, context_dir=ctx_dir)
        assert content == "fallback only"
        assert path == general


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

class TestResolveContextForFile:
    """Tests for resolve_context_for_file function."""

    @pytest.mark.unit
    def test_finds_file_specific_context(self, temp_dir):
        """File-specific context uses _transcr_context suffix."""
        from modules.llm.context_utils import resolve_context_for_file

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy pdf")
        ctx = temp_dir / "document_transcr_context.txt"
        ctx.write_text("File-specific context")

        content, path = resolve_context_for_file(pdf, context_dir=temp_dir)
        assert content == "File-specific context"
        assert path == ctx

    @pytest.mark.unit
    def test_finds_folder_specific_context(self, temp_dir):
        """Folder-specific context uses _transcr_context suffix."""
        from modules.llm.context_utils import resolve_context_for_file

        subfolder = temp_dir / "my_archive"
        subfolder.mkdir()
        pdf = subfolder / "document.pdf"
        pdf.write_text("dummy pdf")
        ctx = temp_dir / "my_archive_transcr_context.txt"
        ctx.write_text("Folder-specific context")

        content, path = resolve_context_for_file(pdf, context_dir=temp_dir)
        assert content == "Folder-specific context"
        assert path == ctx

    @pytest.mark.unit
    def test_returns_none_when_no_context(self, temp_dir):
        """Returns None when no context found."""
        from modules.llm.context_utils import resolve_context_for_file

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy pdf")

        content, path = resolve_context_for_file(pdf, context_dir=temp_dir)
        assert content is None
        assert path is None


class TestResolveContextForFolder:
    """Tests for resolve_context_for_folder function."""

    @pytest.mark.unit
    def test_finds_folder_specific_context(self, temp_dir):
        """Folder-specific context in parent directory."""
        from modules.llm.context_utils import resolve_context_for_folder

        folder = temp_dir / "my_images"
        folder.mkdir()
        ctx = temp_dir / "my_images_transcr_context.txt"
        ctx.write_text("Folder context")

        content, path = resolve_context_for_folder(folder, context_dir=temp_dir)
        assert content == "Folder context"
        assert path == ctx

    @pytest.mark.unit
    def test_returns_none_when_no_context(self, temp_dir):
        """Returns None when no context found."""
        from modules.llm.context_utils import resolve_context_for_folder

        folder = temp_dir / "my_images"
        folder.mkdir()

        content, path = resolve_context_for_folder(folder, context_dir=temp_dir)
        assert content is None
        assert path is None


class TestResolveContextForImage:
    """Tests for resolve_context_for_image function."""

    @pytest.mark.unit
    def test_finds_image_specific_context(self, temp_dir):
        """Image-specific context uses _transcr_context suffix."""
        from modules.llm.context_utils import resolve_context_for_image

        img = temp_dir / "page_001.png"
        img.write_bytes(b"")
        ctx = temp_dir / "page_001_transcr_context.txt"
        ctx.write_text("Image-specific context")

        content, path = resolve_context_for_image(img, context_dir=temp_dir)
        assert content == "Image-specific context"
        assert path == ctx

    @pytest.mark.unit
    def test_finds_folder_context_for_image(self, temp_dir):
        """Folder-specific context for image uses _transcr_context suffix."""
        from modules.llm.context_utils import resolve_context_for_image

        folder = temp_dir / "scans"
        folder.mkdir()
        img = folder / "page_001.png"
        img.write_bytes(b"")
        ctx = temp_dir / "scans_transcr_context.txt"
        ctx.write_text("Folder context for images")

        content, path = resolve_context_for_image(img, context_dir=temp_dir)
        assert content == "Folder context for images"
        assert path == ctx


# ---------------------------------------------------------------------------
# _read_and_validate_context
# ---------------------------------------------------------------------------

class TestReadAndValidateContext:
    """Tests for _read_and_validate_context function."""

    @pytest.mark.unit
    def test_reads_valid_context(self, temp_dir):
        """Valid file returns stripped content."""
        from modules.llm.context_utils import _read_and_validate_context

        ctx = temp_dir / "context.txt"
        ctx.write_text("Valid context content")
        assert _read_and_validate_context(ctx) == "Valid context content"

    @pytest.mark.unit
    def test_returns_none_for_empty_file(self, temp_dir):
        """Empty file returns None."""
        from modules.llm.context_utils import _read_and_validate_context

        ctx = temp_dir / "empty.txt"
        ctx.write_text("")
        assert _read_and_validate_context(ctx) is None

    @pytest.mark.unit
    def test_returns_none_for_whitespace_only(self, temp_dir):
        """Whitespace-only file returns None."""
        from modules.llm.context_utils import _read_and_validate_context

        ctx = temp_dir / "whitespace.txt"
        ctx.write_text("   \n\t  \n  ")
        assert _read_and_validate_context(ctx) is None

    @pytest.mark.unit
    def test_strips_whitespace(self, temp_dir):
        """Content is stripped of leading/trailing whitespace."""
        from modules.llm.context_utils import _read_and_validate_context

        ctx = temp_dir / "context.txt"
        ctx.write_text("  trimmed content  \n")
        assert _read_and_validate_context(ctx) == "trimmed content"


# ---------------------------------------------------------------------------
# load_context_from_path
# ---------------------------------------------------------------------------

class TestLoadContextFromPath:
    """Tests for load_context_from_path function."""

    @pytest.mark.unit
    def test_loads_existing_context(self, temp_dir):
        """Loads context from existing path."""
        from modules.llm.context_utils import load_context_from_path

        ctx = temp_dir / "context.txt"
        ctx.write_text("Context content")
        assert load_context_from_path(ctx) == "Context content"

    @pytest.mark.unit
    def test_returns_none_for_none_path(self):
        """Returns None when path is None."""
        from modules.llm.context_utils import load_context_from_path
        assert load_context_from_path(None) is None

    @pytest.mark.unit
    def test_returns_none_for_nonexistent_path(self, temp_dir):
        """Returns None for nonexistent path."""
        from modules.llm.context_utils import load_context_from_path
        assert load_context_from_path(temp_dir / "nonexistent.txt") is None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestContextSizeThreshold:
    """Tests for context size threshold handling."""

    @pytest.mark.unit
    def test_default_threshold_constant(self):
        """Default threshold constant value."""
        from modules.llm.context_utils import DEFAULT_CONTEXT_SIZE_THRESHOLD
        assert DEFAULT_CONTEXT_SIZE_THRESHOLD == 4000
