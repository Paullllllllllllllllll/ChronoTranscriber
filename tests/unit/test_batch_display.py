from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modules.ui.batch_display as bd


@pytest.mark.unit
def test_format_page_image_handles_missing_page_number() -> None:
    assert bd._format_page_image(None, "img.png") == "(img.png)"


@pytest.mark.unit
def test_format_page_image_includes_page_when_present() -> None:
    assert bd._format_page_image(3, "img.png") == "page 3 (img.png)"


@pytest.mark.unit
def test_display_batch_processing_progress_completed(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    monkeypatch.setattr(bd, "ui_print", MagicMock())
    monkeypatch.setattr(bd, "print_success", MagicMock())
    monkeypatch.setattr(bd, "print_warning", MagicMock())
    monkeypatch.setattr(bd, "print_info", MagicMock())

    temp_file = temp_dir / "x.jsonl"
    temp_file.write_text("", encoding="utf-8")

    bd.display_batch_processing_progress(
        temp_file=temp_file,
        batch_ids=["b1", "b2"],
        completed_count=2,
        missing_count=0,
    )

    bd.print_success.assert_called_once()


@pytest.mark.unit
def test_print_transcription_item_error_includes_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bd, "print_error", MagicMock())

    bd.print_transcription_item_error(
        image_name="img.png",
        page_number=5,
        status_code=400,
        err_code="bad_request",
        err_message="oops",
    )

    msg = bd.print_error.call_args[0][0]
    assert "page 5" in msg
    assert "status=400" in msg
    assert "code=bad_request" in msg
    assert "message=oops" in msg


@pytest.mark.unit
def test_display_page_error_summary_formats_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bd, "print_warning", MagicMock())
    monkeypatch.setattr(bd, "ui_print", MagicMock())

    bd.display_page_error_summary(
        [
            {
                "custom_id": "req-1",
                "image_info": {"image_name": "img.png", "page_number": 2},
                "error_details": {"status_code": 500, "code": "server", "message": "fail"},
            }
        ]
    )

    bd.print_warning.assert_called_once()
    # Ensure at least one formatted ui_print call happened
    assert bd.ui_print.call_count >= 1


@pytest.mark.unit
def test_display_batch_summary_groups_by_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bd, "ui_print", MagicMock())
    monkeypatch.setattr(bd, "print_info", MagicMock())
    monkeypatch.setattr(bd, "print_separator", MagicMock())

    bd.display_batch_summary(
        [
            {"id": "b1", "status": "completed"},
            {"id": "b2", "status": "failed"},
            {"id": "b3", "status": "in_progress"},
        ]
    )

    # Should render header lines and at least one status group summary
    assert bd.ui_print.call_count > 0
