"""Auto mode: the items of a folder and the method of each."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chronotranscriber.sources import SourceItem
from chronotranscriber.sources.auto import decide, file_item, scan


@pytest.mark.unit
class TestScan:
    def test_an_empty_folder_has_no_items(self, temp_dir: Path) -> None:
        found = scan(temp_dir)
        assert found.items == ()
        assert found.loose_images == 0

    def test_items_come_in_type_order(self, temp_dir: Path) -> None:
        (temp_dir / "book.epub").write_bytes(b"")
        (temp_dir / "doc.pdf").write_bytes(b"%PDF")
        (temp_dir / "kindle.azw3").write_bytes(b"")
        scans = temp_dir / "scans"
        scans.mkdir()
        (scans / "page1.png").write_bytes(b"")

        types = [item.type for item in scan(temp_dir).items]

        assert types == ["pdf", "images", "epub", "mobi"]

    def test_loose_images_are_counted_not_returned(self, temp_dir: Path) -> None:
        (temp_dir / "image1.png").write_bytes(b"")
        (temp_dir / "image2.jpg").write_bytes(b"")

        found = scan(temp_dir)

        assert found.items == ()
        assert found.loose_images == 2

    def test_folders_without_images_and_audio_files_are_ignored(
        self, temp_dir: Path
    ) -> None:
        (temp_dir / "empty").mkdir()
        (temp_dir / "talk.mp3").write_bytes(b"")
        assert scan(temp_dir).items == ()

    def test_a_single_file_is_an_item_when_auto_mode_takes_its_type(
        self, temp_dir: Path
    ) -> None:
        assert file_item(temp_dir / "doc.pdf") == SourceItem(
            temp_dir / "doc.pdf", "pdf"
        )
        assert file_item(temp_dir / "talk.mp3") is None


@pytest.mark.unit
class TestDecide:
    @pytest.mark.parametrize("kind", ["epub", "mobi"])
    def test_ebooks_take_their_text(self, temp_dir: Path, kind: str) -> None:
        assert decide(SourceItem(temp_dir / f"b.{kind}", kind), "llm", "ocr") == (
            "native"
        )

    @pytest.mark.parametrize(
        ("searchable", "mode", "expected"),
        [
            (True, "text", "native"),
            (True, "ocr", "tesseract"),
            (False, "text", "tesseract"),
        ],
    )
    def test_a_pdf_takes_the_method_unless_its_text_is_wanted(
        self, temp_dir: Path, searchable: bool, mode: str, expected: str
    ) -> None:
        pdf = SourceItem(temp_dir / "doc.pdf", "pdf")
        with patch("chronotranscriber.sources.auto.PDFProcessor") as processor:
            processor.return_value.is_native_pdf.return_value = searchable
            assert decide(pdf, "tesseract", mode) == expected

    def test_image_folders_take_the_method(self, temp_dir: Path) -> None:
        folder = SourceItem(temp_dir / "scans", "images")
        assert decide(folder, "llm", "text") == "llm"
