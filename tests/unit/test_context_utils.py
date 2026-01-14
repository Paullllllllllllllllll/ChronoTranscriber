"""Unit tests for modules/llm/context_utils.py."""

from __future__ import annotations

import pytest
from pathlib import Path


class TestResolveContextForFile:
    """Tests for resolve_context_for_file function."""

    @pytest.mark.unit
    def test_finds_file_specific_context(self, temp_dir):
        """Test finding file-specific context (same name with .txt extension)."""
        from modules.llm.context_utils import resolve_context_for_file
        
        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_text("dummy pdf")
        context_file = temp_dir / "document.txt"
        context_file.write_text("File-specific context")
        
        content, path = resolve_context_for_file(pdf_file)
        
        assert content == "File-specific context"
        assert path == context_file

    @pytest.mark.unit
    def test_finds_folder_specific_context(self, temp_dir):
        """Test finding folder-specific context."""
        from modules.llm.context_utils import resolve_context_for_file
        
        subfolder = temp_dir / "my_archive"
        subfolder.mkdir()
        pdf_file = subfolder / "document.pdf"
        pdf_file.write_text("dummy pdf")
        context_file = temp_dir / "my_archive.txt"
        context_file.write_text("Folder-specific context")
        
        content, path = resolve_context_for_file(pdf_file)
        
        assert content == "Folder-specific context"
        assert path == context_file

    @pytest.mark.unit
    def test_returns_none_when_no_context(self, temp_dir):
        """Test returns None when no context found."""
        from modules.llm.context_utils import resolve_context_for_file
        
        pdf_file = temp_dir / "document.pdf"
        pdf_file.write_text("dummy pdf")
        
        content, path = resolve_context_for_file(pdf_file, global_context_path=temp_dir / "nonexistent.txt")
        
        assert content is None
        assert path is None

    @pytest.mark.unit
    def test_skips_same_file(self, temp_dir):
        """Test that a .txt file doesn't match itself."""
        from modules.llm.context_utils import resolve_context_for_file
        
        txt_file = temp_dir / "document.txt"
        txt_file.write_text("Some text")
        
        content, path = resolve_context_for_file(txt_file, global_context_path=temp_dir / "nonexistent.txt")
        
        assert content is None
        assert path is None


class TestResolveContextForFolder:
    """Tests for resolve_context_for_folder function."""

    @pytest.mark.unit
    def test_finds_folder_specific_context(self, temp_dir):
        """Test finding folder-specific context in parent directory."""
        from modules.llm.context_utils import resolve_context_for_folder
        
        folder = temp_dir / "my_images"
        folder.mkdir()
        context_file = temp_dir / "my_images.txt"
        context_file.write_text("Folder context")
        
        content, path = resolve_context_for_folder(folder)
        
        assert content == "Folder context"
        assert path == context_file

    @pytest.mark.unit
    def test_finds_in_folder_context(self, temp_dir):
        """Test finding context.txt inside folder."""
        from modules.llm.context_utils import resolve_context_for_folder
        
        folder = temp_dir / "my_images"
        folder.mkdir()
        context_file = folder / "context.txt"
        context_file.write_text("In-folder context")
        
        content, path = resolve_context_for_folder(folder)
        
        assert content == "In-folder context"
        assert path == context_file

    @pytest.mark.unit
    def test_folder_specific_takes_precedence(self, temp_dir):
        """Test that folder-specific context takes precedence over in-folder."""
        from modules.llm.context_utils import resolve_context_for_folder
        
        folder = temp_dir / "my_images"
        folder.mkdir()
        
        folder_specific = temp_dir / "my_images.txt"
        folder_specific.write_text("Folder-specific context")
        
        in_folder = folder / "context.txt"
        in_folder.write_text("In-folder context")
        
        content, path = resolve_context_for_folder(folder)
        
        assert content == "Folder-specific context"
        assert path == folder_specific


class TestResolveContextForImage:
    """Tests for resolve_context_for_image function."""

    @pytest.mark.unit
    def test_finds_image_specific_context(self, temp_dir):
        """Test finding image-specific context."""
        from modules.llm.context_utils import resolve_context_for_image
        
        image_file = temp_dir / "page_001.png"
        image_file.write_bytes(b"")
        context_file = temp_dir / "page_001.txt"
        context_file.write_text("Image-specific context")
        
        content, path = resolve_context_for_image(image_file)
        
        assert content == "Image-specific context"
        assert path == context_file

    @pytest.mark.unit
    def test_finds_folder_context_for_image(self, temp_dir):
        """Test finding folder context for image."""
        from modules.llm.context_utils import resolve_context_for_image
        
        folder = temp_dir / "scans"
        folder.mkdir()
        image_file = folder / "page_001.png"
        image_file.write_bytes(b"")
        context_file = temp_dir / "scans.txt"
        context_file.write_text("Folder context for images")
        
        content, path = resolve_context_for_image(image_file)
        
        assert content == "Folder context for images"
        assert path == context_file


class TestReadAndValidateContext:
    """Tests for _read_and_validate_context function."""

    @pytest.mark.unit
    def test_reads_valid_context(self, temp_dir):
        """Test reading valid context file."""
        from modules.llm.context_utils import _read_and_validate_context
        
        context_file = temp_dir / "context.txt"
        context_file.write_text("Valid context content")
        
        result = _read_and_validate_context(context_file)
        
        assert result == "Valid context content"

    @pytest.mark.unit
    def test_returns_none_for_empty_file(self, temp_dir):
        """Test returns None for empty context file."""
        from modules.llm.context_utils import _read_and_validate_context
        
        context_file = temp_dir / "empty.txt"
        context_file.write_text("")
        
        result = _read_and_validate_context(context_file)
        
        assert result is None

    @pytest.mark.unit
    def test_returns_none_for_whitespace_only(self, temp_dir):
        """Test returns None for whitespace-only file."""
        from modules.llm.context_utils import _read_and_validate_context
        
        context_file = temp_dir / "whitespace.txt"
        context_file.write_text("   \n\t  \n  ")
        
        result = _read_and_validate_context(context_file)
        
        assert result is None

    @pytest.mark.unit
    def test_strips_whitespace(self, temp_dir):
        """Test that content is stripped of leading/trailing whitespace."""
        from modules.llm.context_utils import _read_and_validate_context
        
        context_file = temp_dir / "context.txt"
        context_file.write_text("  trimmed content  \n")
        
        result = _read_and_validate_context(context_file)
        
        assert result == "trimmed content"


class TestLoadContextFromPath:
    """Tests for load_context_from_path function."""

    @pytest.mark.unit
    def test_loads_existing_context(self, temp_dir):
        """Test loading context from existing path."""
        from modules.llm.context_utils import load_context_from_path
        
        context_file = temp_dir / "context.txt"
        context_file.write_text("Context content")
        
        result = load_context_from_path(context_file)
        
        assert result == "Context content"

    @pytest.mark.unit
    def test_returns_none_for_none_path(self):
        """Test returns None when path is None."""
        from modules.llm.context_utils import load_context_from_path
        
        result = load_context_from_path(None)
        
        assert result is None

    @pytest.mark.unit
    def test_returns_none_for_nonexistent_path(self, temp_dir):
        """Test returns None for nonexistent path."""
        from modules.llm.context_utils import load_context_from_path
        
        result = load_context_from_path(temp_dir / "nonexistent.txt")
        
        assert result is None


class TestContextSizeThreshold:
    """Tests for context size threshold handling."""

    @pytest.mark.unit
    def test_default_threshold_constant(self):
        """Test default threshold constant value."""
        from modules.llm.context_utils import DEFAULT_CONTEXT_SIZE_THRESHOLD
        
        assert DEFAULT_CONTEXT_SIZE_THRESHOLD == 4000
