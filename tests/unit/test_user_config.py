"""Unit tests for modules/ui/core.py UserConfiguration.

Tests the UserConfiguration dataclass and its methods.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from modules.ui.core import UserConfiguration


class TestUserConfiguration:
    """Tests for UserConfiguration dataclass."""
    
    @pytest.mark.unit
    def test_default_initialization(self):
        """Test UserConfiguration with default values."""
        config = UserConfiguration()
        
        assert config.processing_type is None
        assert config.transcription_method is None
        assert config.use_batch_processing is False
        assert config.selected_items == []
        assert config.process_all is False
        assert config.selected_schema_name is None
        assert config.selected_schema_path is None
        assert config.additional_context_path is None
        assert config.auto_decisions is None
        assert config.auto_selector is None
        assert config.resume_mode == "skip"
        assert config.page_range is None
    
    @pytest.mark.unit
    def test_custom_initialization(self, temp_dir):
        """Test UserConfiguration with custom values."""
        items = [temp_dir / "file1.pdf", temp_dir / "file2.pdf"]
        
        config = UserConfiguration(
            processing_type="pdfs",
            transcription_method="gpt",
            use_batch_processing=True,
            selected_items=items,
            process_all=True,
            selected_schema_name="custom_schema",
        )
        
        assert config.processing_type == "pdfs"
        assert config.transcription_method == "gpt"
        assert config.use_batch_processing is True
        assert config.selected_items == items
        assert config.process_all is True
        assert config.selected_schema_name == "custom_schema"
    
    @pytest.mark.unit
    def test_selected_items_defaults_to_empty_list(self):
        """Test that selected_items defaults to empty list, not None."""
        config = UserConfiguration()
        assert config.selected_items == []
        assert config.selected_items is not None
    
    @pytest.mark.unit
    def test_str_representation_images(self):
        """Test string representation for image processing."""
        config = UserConfiguration(
            processing_type="images",
            transcription_method="tesseract",
            selected_items=[Path("img1"), Path("img2")],
        )
        
        str_repr = str(config)
        
        assert "images" in str_repr
        assert "Tesseract" in str_repr or "tesseract" in str_repr
        assert "2" in str_repr  # Number of selected items
    
    @pytest.mark.unit
    def test_str_representation_pdfs_gpt(self):
        """Test string representation for PDF with GPT."""
        config = UserConfiguration(
            processing_type="pdfs",
            transcription_method="gpt",
            use_batch_processing=True,
            selected_schema_name="markdown_schema",
            selected_items=[Path("doc.pdf")],
        )
        
        str_repr = str(config)
        
        assert "pdfs" in str_repr
        assert "GPT" in str_repr or "gpt" in str_repr.lower()
        assert "batch" in str_repr.lower()
        assert "markdown_schema" in str_repr
    
    @pytest.mark.unit
    def test_str_representation_auto_mode(self):
        """Test string representation for auto mode."""
        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[1, 2, 3],  # Simplified for testing
        )
        
        str_repr = str(config)
        
        assert "auto" in str_repr.lower()
        assert "3" in str_repr  # Decision count
    
    @pytest.mark.unit
    def test_str_representation_native(self):
        """Test string representation for native extraction."""
        config = UserConfiguration(
            processing_type="pdfs",
            transcription_method="native",
            selected_items=[Path("doc.pdf")],
        )
        
        str_repr = str(config)
        
        assert "Native" in str_repr or "native" in str_repr.lower()
    
    @pytest.mark.unit
    def test_mutable_selected_items(self):
        """Test that selected_items can be modified."""
        config = UserConfiguration()
        
        config.selected_items.append(Path("new_file.pdf"))
        
        assert len(config.selected_items) == 1
        assert config.selected_items[0] == Path("new_file.pdf")
    
    @pytest.mark.unit
    def test_process_all_flag(self):
        """Test process_all flag behavior."""
        config = UserConfiguration(process_all=True)
        
        assert config.process_all is True
        
        config.process_all = False
        assert config.process_all is False

    @pytest.mark.unit
    def test_page_range_field(self):
        """Test page_range field on UserConfiguration."""
        from modules.core.page_range import parse_page_range

        config = UserConfiguration()
        assert config.page_range is None

        pr = parse_page_range("first:5")
        config.page_range = pr
        assert config.page_range is pr
        assert config.page_range.first_n == 5

    @pytest.mark.unit
    def test_str_representation_with_page_range(self):
        """Test string representation includes page range when set."""
        from modules.core.page_range import parse_page_range

        config = UserConfiguration(
            processing_type="pdfs",
            transcription_method="native",
            selected_items=[Path("doc.pdf")],
        )
        config.page_range = parse_page_range("first:5")

        str_repr = str(config)
        assert "first 5" in str_repr

    @pytest.mark.unit
    def test_str_representation_auto_with_page_range(self):
        """Test string representation for auto mode with page range."""
        from modules.core.page_range import parse_page_range

        config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[1, 2],
        )
        config.page_range = parse_page_range("last:3")

        str_repr = str(config)
        assert "last 3" in str_repr
