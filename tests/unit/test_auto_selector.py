"""Unit tests for modules/core/auto_selector.py.

Tests automatic file detection and transcription method selection.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.core.auto_selector import AutoSelector, FileDecision


class TestFileDecision:
    """Tests for FileDecision dataclass."""
    
    @pytest.mark.unit
    def test_creation(self, temp_dir):
        """Test FileDecision can be created with all fields."""
        decision = FileDecision(
            file_path=temp_dir / "test.pdf",
            file_type="pdf",
            method="native",
            reason="PDF contains searchable text"
        )
        assert decision.file_path == temp_dir / "test.pdf"
        assert decision.file_type == "pdf"
        assert decision.method == "native"
        assert decision.reason == "PDF contains searchable text"


class TestAutoSelectorInit:
    """Tests for AutoSelector initialization."""
    
    @pytest.mark.unit
    def test_default_config(self, mock_paths_config, mock_env_no_api_keys):
        """Test initialization with default config values."""
        selector = AutoSelector(mock_paths_config)
        
        assert selector.pdf_use_ocr_for_scanned is True
        assert selector.pdf_use_ocr_for_searchable is False
        assert selector.pdf_ocr_method == "tesseract"
        assert selector.image_ocr_method == "tesseract"
    
    @pytest.mark.unit
    def test_custom_config(self, mock_env_no_api_keys):
        """Test initialization with custom config values."""
        config = {
            "general": {
                "auto_mode_pdf_use_ocr_for_scanned": False,
                "auto_mode_pdf_use_ocr_for_searchable": True,
                "auto_mode_pdf_ocr_method": "gpt",
                "auto_mode_image_ocr_method": "gpt",
            }
        }
        selector = AutoSelector(config)
        
        assert selector.pdf_use_ocr_for_scanned is False
        assert selector.pdf_use_ocr_for_searchable is True
        assert selector.pdf_ocr_method == "gpt"
        assert selector.image_ocr_method == "gpt"
    
    @pytest.mark.unit
    def test_gpt_available_with_key(self, mock_paths_config, mock_env_with_openai_key):
        """Test GPT availability when API key is set."""
        selector = AutoSelector(mock_paths_config)
        assert selector.gpt_available is True
    
    @pytest.mark.unit
    def test_gpt_not_available_without_key(self, mock_paths_config, mock_env_no_api_keys):
        """Test GPT not available when no API key."""
        selector = AutoSelector(mock_paths_config)
        assert selector.gpt_available is False


class TestAutoSelectorScanDirectory:
    """Tests for AutoSelector.scan_directory method."""
    
    @pytest.mark.unit
    def test_empty_directory(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test scanning empty directory."""
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert pdfs == []
        assert images == []
        assert epubs == []
        assert mobis == []
    
    @pytest.mark.unit
    def test_nonexistent_directory(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test scanning nonexistent directory."""
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir / "nonexistent")
        
        assert pdfs == []
        assert images == []
        assert epubs == []
        assert mobis == []
    
    @pytest.mark.unit
    def test_finds_pdf_files(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test finding PDF files."""
        (temp_dir / "doc1.pdf").write_bytes(b"%PDF")
        (temp_dir / "doc2.pdf").write_bytes(b"%PDF")
        
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert len(pdfs) == 2
        assert all(p.suffix == ".pdf" for p in pdfs)
    
    @pytest.mark.unit
    def test_finds_image_files(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test finding image files."""
        (temp_dir / "image1.png").write_bytes(b"")
        (temp_dir / "image2.jpg").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert len(images) == 2
    
    @pytest.mark.unit
    def test_finds_image_folders(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test finding folders containing images."""
        img_folder = temp_dir / "images"
        img_folder.mkdir()
        (img_folder / "page1.png").write_bytes(b"")
        (img_folder / "page2.png").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert len(images) == 1
        assert images[0].is_dir()
    
    @pytest.mark.unit
    def test_finds_epub_files(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test finding EPUB files."""
        (temp_dir / "book.epub").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert len(epubs) == 1
        assert epubs[0].suffix == ".epub"
    
    @pytest.mark.unit
    def test_finds_mobi_files(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test finding MOBI/Kindle files."""
        (temp_dir / "book.mobi").write_bytes(b"")
        (temp_dir / "kindle.azw3").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert len(mobis) == 2


class TestAutoSelectorDecidePdfMethod:
    """Tests for AutoSelector.decide_pdf_method method."""
    
    @pytest.mark.unit
    def test_native_for_searchable_pdf(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test native method for searchable PDF when OCR not forced."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        
        selector = AutoSelector(mock_paths_config)
        
        with patch.object(selector, 'pdf_use_ocr_for_searchable', False):
            with patch('modules.core.auto_selector.PDFProcessor') as mock_processor:
                mock_processor.return_value.is_native_pdf.return_value = True
                method, reason = selector.decide_pdf_method(pdf_path)
        
        assert method == "native"
        assert "searchable" in reason.lower()
    
    @pytest.mark.unit
    def test_tesseract_for_scanned_pdf(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test Tesseract method for scanned PDF."""
        pdf_path = temp_dir / "scanned.pdf"
        pdf_path.write_bytes(b"%PDF")
        
        selector = AutoSelector(mock_paths_config)
        
        with patch('modules.core.auto_selector.PDFProcessor') as mock_processor:
            mock_processor.return_value.is_native_pdf.return_value = False
            method, reason = selector.decide_pdf_method(pdf_path)
        
        assert method == "tesseract"
    
    @pytest.mark.unit
    def test_gpt_fallback_to_tesseract(self, temp_dir, mock_env_no_api_keys):
        """Test GPT falls back to Tesseract when no API key."""
        config = {
            "general": {
                "auto_mode_pdf_use_ocr_for_scanned": True,
                "auto_mode_pdf_ocr_method": "gpt",
            }
        }
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        
        selector = AutoSelector(config)
        
        with patch('modules.core.auto_selector.PDFProcessor') as mock_processor:
            mock_processor.return_value.is_native_pdf.return_value = False
            method, reason = selector.decide_pdf_method(pdf_path)
        
        assert method == "tesseract"
        assert "fallback" in reason.lower() or "unavailable" in reason.lower()


class TestAutoSelectorDecideImageMethod:
    """Tests for AutoSelector.decide_image_method method."""
    
    @pytest.mark.unit
    def test_tesseract_method(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test Tesseract method for images."""
        img_path = temp_dir / "image.png"
        img_path.write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        method, reason = selector.decide_image_method(img_path)
        
        assert method == "tesseract"
    
    @pytest.mark.unit
    def test_gpt_method_with_key(self, temp_dir, mock_env_with_openai_key):
        """Test GPT method when API key available."""
        config = {
            "general": {
                "auto_mode_image_ocr_method": "gpt",
            }
        }
        img_path = temp_dir / "image.png"
        img_path.write_bytes(b"")
        
        selector = AutoSelector(config)
        method, reason = selector.decide_image_method(img_path)
        
        assert method == "gpt"
    
    @pytest.mark.unit
    def test_gpt_fallback_without_key(self, temp_dir, mock_env_no_api_keys):
        """Test GPT falls back when no API key."""
        config = {
            "general": {
                "auto_mode_image_ocr_method": "gpt",
            }
        }
        img_path = temp_dir / "image.png"
        img_path.write_bytes(b"")
        
        selector = AutoSelector(config)
        method, reason = selector.decide_image_method(img_path)
        
        assert method == "tesseract"
        assert "fallback" in reason.lower()


class TestAutoSelectorDecideEpubMethod:
    """Tests for AutoSelector.decide_epub_method method."""
    
    @pytest.mark.unit
    def test_always_native(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test EPUB always uses native extraction."""
        epub_path = temp_dir / "book.epub"
        epub_path.write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        method, reason = selector.decide_epub_method(epub_path)
        
        assert method == "native"
        assert "native" in reason.lower()


class TestAutoSelectorDecideMobiMethod:
    """Tests for AutoSelector.decide_mobi_method method."""
    
    @pytest.mark.unit
    def test_always_native(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test MOBI always uses native extraction."""
        mobi_path = temp_dir / "book.mobi"
        mobi_path.write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        method, reason = selector.decide_mobi_method(mobi_path)
        
        assert method == "native"
        assert "native" in reason.lower()


class TestAutoSelectorCreateDecisions:
    """Tests for AutoSelector.create_decisions method."""
    
    @pytest.mark.unit
    def test_creates_decisions_for_all_files(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test creating decisions for mixed file types."""
        # Create test files
        (temp_dir / "doc.pdf").write_bytes(b"%PDF")
        (temp_dir / "image.png").write_bytes(b"")
        (temp_dir / "book.epub").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        
        with patch('modules.core.auto_selector.PDFProcessor') as mock_processor:
            mock_processor.return_value.is_native_pdf.return_value = True
            decisions = selector.create_decisions(temp_dir)
        
        assert len(decisions) == 3
        
        # Verify decision types
        file_types = {d.file_type for d in decisions}
        assert "pdf" in file_types
        assert "image" in file_types
        assert "epub" in file_types
    
    @pytest.mark.unit
    def test_empty_directory_no_decisions(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test no decisions for empty directory."""
        selector = AutoSelector(mock_paths_config)
        decisions = selector.create_decisions(temp_dir)
        
        assert decisions == []
    
    @pytest.mark.unit
    def test_image_folder_decision(self, mock_paths_config, temp_dir, mock_env_no_api_keys):
        """Test decision for image folder."""
        img_folder = temp_dir / "images"
        img_folder.mkdir()
        (img_folder / "page1.png").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        decisions = selector.create_decisions(temp_dir)
        
        assert len(decisions) == 1
        assert decisions[0].file_type == "image_folder"
        assert decisions[0].file_path.is_dir()
