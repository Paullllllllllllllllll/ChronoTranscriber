"""Unit tests for context image resolution in modules/config/context.py.

Tests the 3-level hierarchy for image context:
1. File-specific:   {input_stem}_transcr_context_image.{ext}
2. Folder-specific: {parent_folder}_transcr_context_image.{ext}
3. General fallback: context/transcr_context_image.{ext}

Extension priority follows sorted(SUPPORTED_IMAGE_EXTENSIONS).
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestResolveImageContextFileLevel:
    """File-level context image resolution."""

    @pytest.mark.unit
    def test_file_specific_png(self, temp_dir: Path) -> None:
        """File-specific context image (.png) is resolved."""
        from modules.config.context import resolve_context_image_for_file

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy", encoding="utf-8")
        ctx = temp_dir / "document_transcr_context_image.png"
        ctx.write_bytes(b"\x89PNG")

        result = resolve_context_image_for_file(pdf, context_dir=temp_dir)
        assert result == ctx

    @pytest.mark.unit
    def test_file_specific_jpg(self, temp_dir: Path) -> None:
        """File-specific context image (.jpg) is resolved."""
        from modules.config.context import resolve_context_image_for_file

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy", encoding="utf-8")
        ctx = temp_dir / "document_transcr_context_image.jpg"
        ctx.write_bytes(b"\xff\xd8\xff")

        result = resolve_context_image_for_file(pdf, context_dir=temp_dir)
        assert result == ctx

    @pytest.mark.unit
    def test_folder_specific_when_no_file_specific(self, temp_dir: Path) -> None:
        """Folder-specific context image is used as fallback."""
        from modules.config.context import resolve_context_image_for_file

        subfolder = temp_dir / "my_archive"
        subfolder.mkdir()
        pdf = subfolder / "document.pdf"
        pdf.write_text("dummy", encoding="utf-8")
        ctx = temp_dir / "my_archive_transcr_context_image.png"
        ctx.write_bytes(b"\x89PNG")

        result = resolve_context_image_for_file(pdf, context_dir=temp_dir)
        assert result == ctx

    @pytest.mark.unit
    def test_general_fallback(self, temp_dir: Path) -> None:
        """General fallback context image is used when no specific exists."""
        from modules.config.context import resolve_context_image_for_file

        subfolder = temp_dir / "my_archive"
        subfolder.mkdir()
        pdf = subfolder / "document.pdf"
        pdf.write_text("dummy", encoding="utf-8")

        ctx_dir = temp_dir / "context"
        ctx_dir.mkdir()
        ctx = ctx_dir / "transcr_context_image.png"
        ctx.write_bytes(b"\x89PNG")

        result = resolve_context_image_for_file(pdf, context_dir=ctx_dir)
        assert result == ctx

    @pytest.mark.unit
    def test_file_specific_wins_over_folder(self, temp_dir: Path) -> None:
        """File-specific context image takes precedence over folder."""
        from modules.config.context import resolve_context_image_for_file

        subfolder = temp_dir / "archive"
        subfolder.mkdir()
        pdf = subfolder / "page.pdf"
        pdf.write_text("dummy", encoding="utf-8")

        file_ctx = subfolder / "page_transcr_context_image.png"
        file_ctx.write_bytes(b"\x89PNG-file")
        folder_ctx = temp_dir / "archive_transcr_context_image.png"
        folder_ctx.write_bytes(b"\x89PNG-folder")

        result = resolve_context_image_for_file(pdf, context_dir=temp_dir)
        assert result == file_ctx

    @pytest.mark.unit
    def test_no_context_image_returns_none(self, temp_dir: Path) -> None:
        """Returns None when no context image exists at any level."""
        from modules.config.context import resolve_context_image_for_file

        pdf = temp_dir / "document.pdf"
        pdf.write_text("dummy", encoding="utf-8")

        result = resolve_context_image_for_file(pdf, context_dir=temp_dir)
        assert result is None


class TestResolveImageContextFolderLevel:
    """Folder-level context image resolution."""

    @pytest.mark.unit
    def test_folder_context_image(self, temp_dir: Path) -> None:
        """Folder-specific context image is resolved for folders."""
        from modules.config.context import (
            resolve_context_image_for_folder,
        )

        folder = temp_dir / "scanned_pages"
        folder.mkdir()
        ctx = temp_dir / "scanned_pages_transcr_context_image.png"
        ctx.write_bytes(b"\x89PNG")

        result = resolve_context_image_for_folder(folder, context_dir=temp_dir)
        assert result == ctx

    @pytest.mark.unit
    def test_folder_fallback_to_general(self, temp_dir: Path) -> None:
        """General fallback is used for folder when no specific exists."""
        from modules.config.context import (
            resolve_context_image_for_folder,
        )

        folder = temp_dir / "scanned_pages"
        folder.mkdir()

        ctx_dir = temp_dir / "context"
        ctx_dir.mkdir()
        ctx = ctx_dir / "transcr_context_image.jpg"
        ctx.write_bytes(b"\xff\xd8\xff")

        result = resolve_context_image_for_folder(folder, context_dir=ctx_dir)
        assert result == ctx

    @pytest.mark.unit
    def test_folder_no_match_returns_none(self, temp_dir: Path) -> None:
        """Returns None when no context image exists for folder."""
        from modules.config.context import (
            resolve_context_image_for_folder,
        )

        folder = temp_dir / "scanned_pages"
        folder.mkdir()

        result = resolve_context_image_for_folder(folder, context_dir=temp_dir)
        assert result is None


class TestResolveImageContextImageLevel:
    """Image-level context resolution (individual image files)."""

    @pytest.mark.unit
    def test_image_specific_context(self, temp_dir: Path) -> None:
        """Context image for a specific image file is resolved."""
        from modules.config.context import (
            resolve_context_image_for_image,
        )

        img = temp_dir / "page_001.png"
        img.write_bytes(b"\x89PNG")
        ctx = temp_dir / "page_001_transcr_context_image.png"
        ctx.write_bytes(b"\x89PNG-ctx")

        result = resolve_context_image_for_image(img, context_dir=temp_dir)
        assert result == ctx


class TestExtensionPriority:
    """Sorted extension order for deterministic tiebreaking."""

    @pytest.mark.unit
    def test_sorted_extension_order(self, temp_dir: Path) -> None:
        """When multiple extensions exist, the sorted-first wins."""
        from modules.config.context import resolve_context_image_for_file

        pdf = temp_dir / "doc.pdf"
        pdf.write_text("dummy", encoding="utf-8")

        png = temp_dir / "doc_transcr_context_image.png"
        png.write_bytes(b"\x89PNG")
        jpg = temp_dir / "doc_transcr_context_image.jpg"
        jpg.write_bytes(b"\xff\xd8\xff")

        result = resolve_context_image_for_file(pdf, context_dir=temp_dir)
        assert result is not None
        # .jpg sorts before .png
        assert result.suffix == ".jpg"


class TestLoadContextImageFromPath:
    """Tests for load_context_image_from_path."""

    @pytest.mark.unit
    def test_valid_path(self, temp_dir: Path) -> None:
        """Valid image path is returned as-is."""
        from modules.config.context import load_context_image_from_path

        img = temp_dir / "context.png"
        img.write_bytes(b"\x89PNG")

        result = load_context_image_from_path(img)
        assert result == img

    @pytest.mark.unit
    def test_none_input(self) -> None:
        """None input returns None."""
        from modules.config.context import load_context_image_from_path

        assert load_context_image_from_path(None) is None

    @pytest.mark.unit
    def test_nonexistent_path(self, temp_dir: Path) -> None:
        """Non-existent path returns None."""
        from modules.config.context import load_context_image_from_path

        result = load_context_image_from_path(temp_dir / "does_not_exist.png")
        assert result is None

    @pytest.mark.unit
    def test_unsupported_extension(self, temp_dir: Path) -> None:
        """Unsupported extension returns None."""
        from modules.config.context import load_context_image_from_path

        txt = temp_dir / "context.txt"
        txt.write_text("not an image", encoding="utf-8")

        result = load_context_image_from_path(txt)
        assert result is None

    @pytest.mark.unit
    def test_supported_extensions(self, temp_dir: Path) -> None:
        """All common supported extensions are accepted."""
        from modules.config.context import load_context_image_from_path

        for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            img = temp_dir / f"ctx{ext}"
            img.write_bytes(b"\x00")
            result = load_context_image_from_path(img)
            assert result == img, f"Expected {ext} to be accepted"
