"""Discovery: Mabillon's working folders below an output root are never inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronotranscriber.sources import SourceItem, discover_items
from chronotranscriber.sources.auto import scan


def _mabillon_folder(root: Path) -> None:
    """Write a Mabillon working folder with page images and a PDF inside."""
    source = root / ".mabillon" / "book" / "work" / "source"
    source.mkdir(parents=True)
    (root / ".mabillon" / "page1.png").write_bytes(b"")
    (source / "p0001.png").write_bytes(b"")
    (source / "copy.pdf").write_bytes(b"%PDF")


@pytest.mark.unit
class TestMabillonFolders:
    def test_auto_mode_skips_them(self, temp_dir: Path) -> None:
        _mabillon_folder(temp_dir)
        (temp_dir / "doc.pdf").write_bytes(b"%PDF")

        assert scan(temp_dir).items == (SourceItem(temp_dir / "doc.pdf", "pdf"),)

    def test_recursive_runs_skip_them(self, temp_dir: Path) -> None:
        _mabillon_folder(temp_dir)
        (temp_dir / "doc.pdf").write_bytes(b"%PDF")
        scans = temp_dir / "scans"
        scans.mkdir()
        (scans / "page1.png").write_bytes(b"")

        pdfs = discover_items(temp_dir, "pdf", recursive=True)
        folders = discover_items(temp_dir, "images", recursive=True)

        assert pdfs == [SourceItem(temp_dir / "doc.pdf", "pdf")]
        assert folders == [SourceItem(scans, "images")]
