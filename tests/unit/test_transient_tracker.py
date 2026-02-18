"""Unit tests for TransientFileTracker in modules/core/workflow.py."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestTransientFileTracker:
    """Tests for TransientFileTracker class."""

    @pytest.mark.unit
    def test_register_jsonl(self, tmp_path: Path):
        """Test registering a JSONL file for tracking."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.touch()
        
        tracker.register_jsonl(jsonl_path, "gpt")
        
        assert len(tracker._jsonl_files) == 1
        assert tracker._jsonl_files[0] == (jsonl_path, "gpt")

    @pytest.mark.unit
    def test_register_preprocessed_folder(self, tmp_path: Path):
        """Test registering a preprocessed folder for tracking."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        folder_path = tmp_path / "preprocessed_images"
        folder_path.mkdir()
        
        tracker.register_preprocessed_folder(folder_path, "test.pdf")
        
        assert len(tracker._preprocessed_folders) == 1
        assert tracker._preprocessed_folders[0] == (folder_path, "test.pdf")

    @pytest.mark.unit
    def test_mark_jsonl_complete_removes_from_tracking(self, tmp_path: Path):
        """Test that marking JSONL complete removes it from tracking."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.touch()
        
        tracker.register_jsonl(jsonl_path, "gpt")
        assert len(tracker._jsonl_files) == 1
        
        tracker.mark_jsonl_complete(jsonl_path)
        assert len(tracker._jsonl_files) == 0

    @pytest.mark.unit
    def test_mark_preprocessed_complete_removes_from_tracking(self, tmp_path: Path):
        """Test that marking preprocessed folder complete removes it from tracking."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        folder_path = tmp_path / "preprocessed_images"
        folder_path.mkdir()
        
        tracker.register_preprocessed_folder(folder_path, "test.pdf")
        assert len(tracker._preprocessed_folders) == 1
        
        tracker.mark_preprocessed_complete(folder_path)
        assert len(tracker._preprocessed_folders) == 0

    @pytest.mark.unit
    def test_cleanup_pending_deletes_jsonl_when_not_retained(self, tmp_path: Path):
        """Test that cleanup_pending deletes JSONL when retain_temporary_jsonl is False."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"retain_temporary_jsonl": False}, use_batch_processing=False)
        
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"test": "data"}')
        
        tracker.register_jsonl(jsonl_path, "tesseract")
        assert jsonl_path.exists()
        
        tracker.cleanup_pending()
        
        assert not jsonl_path.exists()
        assert len(tracker._jsonl_files) == 0

    @pytest.mark.unit
    def test_cleanup_pending_preserves_jsonl_when_retained(self, tmp_path: Path):
        """Test that cleanup_pending preserves JSONL when retain_temporary_jsonl is True."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"retain_temporary_jsonl": True}, use_batch_processing=False)
        
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"test": "data"}')
        
        tracker.register_jsonl(jsonl_path, "tesseract")
        
        tracker.cleanup_pending()
        
        assert jsonl_path.exists()
        assert len(tracker._jsonl_files) == 0

    @pytest.mark.unit
    def test_cleanup_pending_preserves_gpt_batch_jsonl(self, tmp_path: Path):
        """Test that cleanup_pending preserves JSONL for GPT batch mode."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"retain_temporary_jsonl": False}, use_batch_processing=True)
        
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"test": "data"}')
        
        tracker.register_jsonl(jsonl_path, "gpt")
        
        tracker.cleanup_pending()
        
        # JSONL should be preserved for batch tracking
        assert jsonl_path.exists()

    @pytest.mark.unit
    def test_cleanup_pending_deletes_preprocessed_folder_when_not_kept(self, tmp_path: Path):
        """Test that cleanup_pending deletes preprocessed folder when keep_preprocessed_images is False."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"keep_preprocessed_images": False}, use_batch_processing=False)
        
        folder_path = tmp_path / "preprocessed_images"
        folder_path.mkdir()
        (folder_path / "image1.png").write_bytes(b"fake image")
        
        tracker.register_preprocessed_folder(folder_path, "test.pdf")
        assert folder_path.exists()
        
        tracker.cleanup_pending()
        
        assert not folder_path.exists()
        assert len(tracker._preprocessed_folders) == 0

    @pytest.mark.unit
    def test_cleanup_pending_preserves_preprocessed_folder_when_kept(self, tmp_path: Path):
        """Test that cleanup_pending preserves preprocessed folder when keep_preprocessed_images is True."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"keep_preprocessed_images": True}, use_batch_processing=False)
        
        folder_path = tmp_path / "preprocessed_images"
        folder_path.mkdir()
        (folder_path / "image1.png").write_bytes(b"fake image")
        
        tracker.register_preprocessed_folder(folder_path, "test.pdf")
        
        tracker.cleanup_pending()
        
        assert folder_path.exists()
        assert len(tracker._preprocessed_folders) == 0

    @pytest.mark.unit
    def test_clear_removes_all_tracking_without_cleanup(self, tmp_path: Path):
        """Test that clear() removes tracking without deleting files."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"retain_temporary_jsonl": False, "keep_preprocessed_images": False})
        
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"test": "data"}')
        folder_path = tmp_path / "preprocessed_images"
        folder_path.mkdir()
        
        tracker.register_jsonl(jsonl_path, "tesseract")
        tracker.register_preprocessed_folder(folder_path, "test.pdf")
        
        tracker.clear()
        
        # Files should still exist
        assert jsonl_path.exists()
        assert folder_path.exists()
        # But tracking should be cleared
        assert len(tracker._jsonl_files) == 0
        assert len(tracker._preprocessed_folders) == 0

    @pytest.mark.unit
    def test_multiple_files_tracked(self, tmp_path: Path):
        """Test tracking multiple files simultaneously."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        
        jsonl1 = tmp_path / "test1.jsonl"
        jsonl2 = tmp_path / "test2.jsonl"
        folder1 = tmp_path / "preprocessed1"
        folder2 = tmp_path / "preprocessed2"
        
        jsonl1.touch()
        jsonl2.touch()
        folder1.mkdir()
        folder2.mkdir()
        
        tracker.register_jsonl(jsonl1, "gpt")
        tracker.register_jsonl(jsonl2, "tesseract")
        tracker.register_preprocessed_folder(folder1, "test1.pdf")
        tracker.register_preprocessed_folder(folder2, "test2.pdf")
        
        assert len(tracker._jsonl_files) == 2
        assert len(tracker._preprocessed_folders) == 2
        
        # Mark one of each complete
        tracker.mark_jsonl_complete(jsonl1)
        tracker.mark_preprocessed_complete(folder1)
        
        assert len(tracker._jsonl_files) == 1
        assert len(tracker._preprocessed_folders) == 1
        assert tracker._jsonl_files[0] == (jsonl2, "tesseract")
        assert tracker._preprocessed_folders[0] == (folder2, "test2.pdf")

    @pytest.mark.unit
    def test_cleanup_handles_missing_files_gracefully(self, tmp_path: Path):
        """Test that cleanup handles already-deleted files gracefully."""
        from modules.core.workflow import TransientFileTracker
        
        tracker = TransientFileTracker()
        tracker.configure({"retain_temporary_jsonl": False, "keep_preprocessed_images": False})
        
        jsonl_path = tmp_path / "test.jsonl"
        folder_path = tmp_path / "preprocessed_images"
        
        # Register files that don't exist
        tracker.register_jsonl(jsonl_path, "tesseract")
        tracker.register_preprocessed_folder(folder_path, "test.pdf")
        
        # Should not raise any errors
        tracker.cleanup_pending()
        
        assert len(tracker._jsonl_files) == 0
        assert len(tracker._preprocessed_folders) == 0


class TestTransientTrackerIntegration:
    """Integration tests for TransientFileTracker with WorkflowManager."""

    @pytest.mark.unit
    def test_workflow_manager_initializes_tracker(self):
        """Test that WorkflowManager initializes TransientFileTracker."""
        from modules.core.workflow import WorkflowManager, TransientFileTracker
        from modules.ui.core import UserConfiguration
        
        user_config = UserConfiguration()
        user_config.transcription_method = "gpt"
        user_config.use_batch_processing = False
        
        with patch('modules.core.workflow.configure_tesseract_executable'):
            wm = WorkflowManager(
                user_config=user_config,
                paths_config={"general": {}},
                model_config={},
                concurrency_config={},
                image_processing_config={}
            )
        
        assert hasattr(wm, '_transient_tracker')
        assert isinstance(wm._transient_tracker, TransientFileTracker)

    @pytest.mark.unit
    def test_tracker_configured_with_processing_settings(self):
        """Test that tracker is configured with processing settings."""
        from modules.core.workflow import WorkflowManager
        from modules.ui.core import UserConfiguration
        
        user_config = UserConfiguration()
        user_config.transcription_method = "gpt"
        user_config.use_batch_processing = True
        user_config.resume_mode = "overwrite"  # Avoid JSONL retention override
        
        processing_settings = {
            "retain_temporary_jsonl": False,
            "keep_preprocessed_images": False
        }
        
        with patch('modules.core.workflow.configure_tesseract_executable'):
            wm = WorkflowManager(
                user_config=user_config,
                paths_config={"general": processing_settings},
                model_config={},
                concurrency_config={},
                image_processing_config={}
            )
        
        assert wm._transient_tracker._use_batch_processing is True
        assert wm._transient_tracker._processing_settings.get("retain_temporary_jsonl") is False
        assert wm._transient_tracker._processing_settings.get("keep_preprocessed_images") is False


# =============================================================================
# CT-5: KeyboardInterrupt / asyncio.CancelledError cleanup regression tests
#
# Before the fix, process_selected_items() only had `except Exception` guarding
# individual items.  A KeyboardInterrupt or asyncio.CancelledError propagated
# directly to the `finally` block with interrupted=False, so clear() was called
# instead of cleanup_pending(), leaving transient files on disk.
# =============================================================================

class TestKeyboardInterruptCleanup:
    """CT-5 regression: process_selected_items calls cleanup_pending on BaseException."""

    def _make_workflow_manager(self):
        """Build a minimal WorkflowManager with a replaceable tracker."""
        from modules.core.workflow import WorkflowManager
        from modules.ui.core import UserConfiguration

        user_config = UserConfiguration()
        user_config.transcription_method = "tesseract"
        user_config.use_batch_processing = False
        user_config.processing_type = "images"
        user_config.resume_mode = "overwrite"
        user_config.selected_items = [Path("dummy_folder")]

        with patch("modules.core.workflow.configure_tesseract_executable"):
            wm = WorkflowManager(
                user_config=user_config,
                paths_config={"general": {
                    "retain_temporary_jsonl": False,
                    "keep_preprocessed_images": False,
                }},
                model_config={},
                concurrency_config={},
                image_processing_config={},
            )
        return wm

    @pytest.mark.unit
    def test_keyboard_interrupt_calls_cleanup_pending(self):
        """KeyboardInterrupt during processing triggers cleanup_pending(), not clear()."""
        import asyncio
        from unittest.mock import MagicMock, patch

        wm = self._make_workflow_manager()
        mock_tracker = MagicMock()
        wm._transient_tracker = mock_tracker

        async def _raise(*args, **kwargs):
            raise KeyboardInterrupt

        with patch.object(wm, "process_single_image_folder", side_effect=_raise):
            with pytest.raises(KeyboardInterrupt):
                asyncio.run(wm.process_selected_items(transcriber=None))

        mock_tracker.cleanup_pending.assert_called_once()
        mock_tracker.clear.assert_not_called()

    @pytest.mark.unit
    def test_cancelled_error_calls_cleanup_pending(self):
        """asyncio.CancelledError during processing triggers cleanup_pending(), not clear()."""
        import asyncio
        from unittest.mock import MagicMock, patch

        wm = self._make_workflow_manager()
        mock_tracker = MagicMock()
        wm._transient_tracker = mock_tracker

        async def _raise(*args, **kwargs):
            raise asyncio.CancelledError

        with patch.object(wm, "process_single_image_folder", side_effect=_raise):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(wm.process_selected_items(transcriber=None))

        mock_tracker.cleanup_pending.assert_called_once()
        mock_tracker.clear.assert_not_called()

    @pytest.mark.unit
    def test_successful_run_calls_clear_not_cleanup_pending(self):
        """Successful processing calls clear(), not cleanup_pending()."""
        import asyncio
        from unittest.mock import MagicMock, patch

        wm = self._make_workflow_manager()
        mock_tracker = MagicMock()
        wm._transient_tracker = mock_tracker

        async def _succeed(*args, **kwargs):
            return None

        with patch.object(wm, "process_single_image_folder", side_effect=_succeed):
            asyncio.run(wm.process_selected_items(transcriber=None))

        mock_tracker.clear.assert_called_once()
        mock_tracker.cleanup_pending.assert_not_called()

    @pytest.mark.unit
    def test_item_exception_calls_cleanup_pending(self):
        """A per-item Exception (caught internally) also triggers cleanup_pending()."""
        import asyncio
        from unittest.mock import MagicMock, patch

        wm = self._make_workflow_manager()
        mock_tracker = MagicMock()
        wm._transient_tracker = mock_tracker

        async def _raise_runtime(*args, **kwargs):
            raise RuntimeError("item failed")

        with patch.object(wm, "process_single_image_folder", side_effect=_raise_runtime):
            # RuntimeError is caught per-item; it should not propagate
            asyncio.run(wm.process_selected_items(transcriber=None))

        mock_tracker.cleanup_pending.assert_called_once()
        mock_tracker.clear.assert_not_called()

    @pytest.mark.unit
    def test_keyboard_interrupt_propagates_after_cleanup(self):
        """KeyboardInterrupt is re-raised after cleanup_pending() is called."""
        import asyncio
        from unittest.mock import MagicMock, patch

        wm = self._make_workflow_manager()
        wm._transient_tracker = MagicMock()

        async def _raise(*args, **kwargs):
            raise KeyboardInterrupt

        with patch.object(wm, "process_single_image_folder", side_effect=_raise):
            with pytest.raises(KeyboardInterrupt):
                asyncio.run(wm.process_selected_items(transcriber=None))

    @pytest.mark.unit
    def test_cancelled_error_propagates_after_cleanup(self):
        """asyncio.CancelledError is re-raised after cleanup_pending() is called."""
        import asyncio
        from unittest.mock import MagicMock, patch

        wm = self._make_workflow_manager()
        wm._transient_tracker = MagicMock()

        async def _raise(*args, **kwargs):
            raise asyncio.CancelledError

        with patch.object(wm, "process_single_image_folder", side_effect=_raise):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(wm.process_selected_items(transcriber=None))
