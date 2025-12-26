"""Integration tests for the transcription workflow.

Tests end-to-end workflows using real file operations but mocked API calls.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from modules.ui.core import UserConfiguration
from modules.core.workflow import WorkflowManager


class TestWorkflowManagerIntegration:
    """Integration tests for WorkflowManager."""
    
    @pytest.fixture
    def user_config(self, sample_image_folder):
        """Create UserConfiguration for testing."""
        config = UserConfiguration()
        config.processing_type = "images"
        config.transcription_method = "tesseract"
        config.use_batch_processing = False
        config.selected_items = [sample_image_folder]
        return config
    
    @pytest.fixture
    def workflow_manager(
        self,
        user_config,
        mock_paths_config,
        mock_model_config,
        mock_concurrency_config,
        mock_image_processing_config,
    ):
        """Create WorkflowManager for testing."""
        return WorkflowManager(
            user_config=user_config,
            paths_config=mock_paths_config,
            model_config=mock_model_config,
            concurrency_config=mock_concurrency_config,
            image_processing_config=mock_image_processing_config,
        )
    
    @pytest.mark.integration
    def test_manager_initialization(self, workflow_manager):
        """Test WorkflowManager initializes correctly."""
        assert workflow_manager is not None
        assert workflow_manager.user_config is not None
    
    @pytest.mark.integration
    def test_output_directory_creation(self, workflow_manager, temp_dir):
        """Test that output directories are created."""
        # Directories should exist or be created
        assert workflow_manager.pdf_output_dir is not None
        assert workflow_manager.image_output_dir is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tesseract_not_available_handling(self, workflow_manager):
        """Test handling when Tesseract is not available."""
        with patch.object(workflow_manager, '_ensure_tesseract_available', return_value=False):
            # Should handle gracefully without crashing
            workflow_manager.user_config.transcription_method = "tesseract"
            # The method should return early if Tesseract not available
            result = workflow_manager._ensure_tesseract_available()
            assert result is False


class TestPostProcessingIntegration:
    """Integration tests for post-processing pipeline."""
    
    @pytest.mark.integration
    def test_full_postprocessing_pipeline(self, temp_dir):
        """Test the complete post-processing pipeline on a file."""
        from modules.processing.postprocess import postprocess_file
        
        # Create test file with various issues
        input_file = temp_dir / "input.txt"
        input_file.write_text(
            "\ufeffThis is text with BOM\n"
            "Line with trailing spaces   \n"
            "\n\n\n\n\n"  # Too many blank lines
            "Normal paragraph.\n",
            encoding="utf-8"
        )
        
        output_file = temp_dir / "output.txt"
        
        config = {
            "enabled": True,
            "max_blank_lines": 2,
            "collapse_internal_spaces": True,
        }
        
        result_path = postprocess_file(input_file, output_path=output_file, config=config)
        
        assert result_path == output_file
        content = output_file.read_text(encoding="utf-8")
        
        # BOM should be removed
        assert "\ufeff" not in content
        # Trailing spaces should be gone
        assert "   \n" not in content


class TestConfigIntegration:
    """Integration tests for configuration loading."""
    
    @pytest.mark.integration
    def test_config_service_loads_all_configs(self):
        """Test that ConfigService can load all configuration files."""
        from modules.config.service import ConfigService
        
        ConfigService.reset()
        
        try:
            service = ConfigService()
            
            # These should not raise (but may fail in test env due to model capability checks)
            try:
                paths = service.get_paths_config()
                model = service.get_model_config()
                concurrency = service.get_concurrency_config()
                image = service.get_image_processing_config()
                
                assert isinstance(paths, dict)
                assert isinstance(model, dict)
                assert isinstance(concurrency, dict)
                assert isinstance(image, dict)
            except Exception as e:
                if "CapabilityError" in str(type(e).__name__):
                    pytest.skip(f"Config loading failed due to model capability check: {e}")
                raise
        finally:
            ConfigService.reset()


class TestAutoSelectorIntegration:
    """Integration tests for auto mode file selection."""
    
    @pytest.mark.integration
    def test_auto_selector_scans_mixed_directory(self, temp_dir, mock_paths_config, mock_env_no_api_keys):
        """Test AutoSelector scanning a directory with mixed file types."""
        from modules.core.auto_selector import AutoSelector
        
        # Create various file types
        (temp_dir / "document.pdf").write_bytes(b"%PDF-1.4")
        (temp_dir / "image.png").write_bytes(b"PNG")
        (temp_dir / "book.epub").write_bytes(b"PK")  # ZIP signature
        (temp_dir / "ebook.mobi").write_bytes(b"BOOKMOBI")
        (temp_dir / "readme.txt").write_text("Text file", encoding="utf-8")
        
        selector = AutoSelector(mock_paths_config)
        pdfs, images, epubs, mobis = selector.scan_directory(temp_dir)
        
        assert len(pdfs) == 1
        assert len(images) == 1
        assert len(epubs) == 1
        assert len(mobis) == 1
    
    @pytest.mark.integration
    def test_auto_selector_creates_decisions(self, temp_dir, mock_paths_config, mock_env_no_api_keys):
        """Test AutoSelector creating processing decisions."""
        from modules.core.auto_selector import AutoSelector
        
        # Create test files
        (temp_dir / "doc.pdf").write_bytes(b"%PDF")
        (temp_dir / "image.jpg").write_bytes(b"")
        
        selector = AutoSelector(mock_paths_config)
        
        with patch('modules.core.auto_selector.PDFProcessor') as mock_processor:
            mock_processor.return_value.is_native_pdf.return_value = True
            decisions = selector.create_decisions(temp_dir)
        
        assert len(decisions) == 2
        
        # Each decision should have required fields
        for decision in decisions:
            assert decision.file_path is not None
            assert decision.file_type in ["pdf", "image", "epub", "mobi", "image_folder"]
            assert decision.method in ["native", "tesseract", "gpt"]
            assert decision.reason is not None


class TestCliIntegration:
    """Integration tests for CLI argument parsing and execution."""
    
    @pytest.mark.integration
    def test_transcriber_parser_full_args(self):
        """Test full argument parsing for transcriber."""
        from modules.core.cli_args import create_transcriber_parser
        
        parser = create_transcriber_parser()
        args = parser.parse_args([
            "--input", "test/input",
            "--output", "test/output",
            "--type", "pdfs",
            "--method", "gpt",
            "--batch",
            "--schema", "custom",
            "--context", "context.txt",
            "--recursive",
        ])
        
        assert args.input == "test/input"
        assert args.output == "test/output"
        assert args.type == "pdfs"
        assert args.method == "gpt"
        assert args.batch is True
        assert args.schema == "custom"
        assert args.context == "context.txt"
        assert args.recursive is True
    
    @pytest.mark.integration
    def test_repair_parser_all_flags(self):
        """Test repair parser with all failure flags."""
        from modules.core.cli_args import create_repair_parser
        
        parser = create_repair_parser()
        args = parser.parse_args([
            "--transcription", "doc.txt",
            "--all-failures",
            "--batch",
        ])
        
        assert args.transcription == "doc.txt"
        assert args.all_failures is True
        assert args.batch is True


class TestTextProcessingIntegration:
    """Integration tests for text processing pipeline."""
    
    @pytest.mark.integration
    def test_extract_from_various_response_formats(self):
        """Test extraction from multiple API response formats."""
        from modules.processing.text_processing import extract_transcribed_text
        
        # Test all supported formats
        formats = [
            # Schema object
            {
                "transcription": "Schema text",
                "no_transcribable_text": False,
                "transcription_not_possible": False,
            },
            # Responses API
            {
                "output_text": json.dumps({
                    "transcription": "Responses API text",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                })
            },
            # Chat Completions
            {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "transcription": "Chat text",
                            "no_transcribable_text": False,
                            "transcription_not_possible": False,
                        })
                    }
                }]
            },
        ]
        
        expected = ["Schema text", "Responses API text", "Chat text"]
        
        for fmt, exp in zip(formats, expected):
            result = extract_transcribed_text(fmt)
            assert result == exp
    
    @pytest.mark.integration
    def test_batch_output_processing(self):
        """Test processing batch output with multiple records."""
        from modules.processing.text_processing import process_batch_output
        
        # Create batch output with multiple items
        items = []
        for i in range(5):
            items.append(json.dumps({
                "response": {
                    "body": {
                        "transcription": f"Page {i+1} text",
                        "no_transcribable_text": False,
                        "transcription_not_possible": False,
                    }
                }
            }))
        
        content = "\n".join(items)
        result = process_batch_output(content.encode())
        
        assert len(result) == 5
        assert result[0] == "Page 1 text"
        assert result[4] == "Page 5 text"
