"""Discovery: the items of an input folder, their order, names and outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

import chronotranscriber.run as core
from chronotranscriber.llm.prompts import CONTEXT_SUFFIX, IMAGE_CONTEXT_SUFFIX
from chronotranscriber.methods.recording import CHUNK_FOLDER
from chronotranscriber.methods.tesseract import PREPROCESSED_FOLDER
from chronotranscriber.output.paths import create_safe_directory_name
from chronotranscriber.sources import (
    SIDECAR_SUFFIXES,
    WORKING_FOLDERS,
    SourceItem,
    discover_items,
    find_items,
)
from chronotranscriber.sources.auto import scan
from tests.conftest import make_settings, make_spec

_TREE = (
    "p10.pdf",
    "Ledger.pdf",
    "p2.pdf",
    "b.PDF",
    "a.epub",
    "k.azw3",
    "talk.mp3",
    "z.flac",
    "loose.png",
    "notes.txt",
    "p2.txt",
    "p2.md",
    "chronotranscriber.sqlite",
    "vol_10/x.pdf",
    "vol_2/x.pdf",
    "vol_2/deep/y.pdf",
    "vol_2/.mabillon/q.pdf",
    "scans_10/1.png",
    "scans_2/1.png",
    "scans_2/2.png",
    "scans_2/inner/3.png",
    "p2-0b399eab/preprocessed_images_tesseract/page_0001.jpg",
    "talk-8e219476/audio_chunks/chunk_000.wav",
    ".mabillon/q.pdf",
    ".mabillon/book/source/p0001.png",
    "p2_transcr_context.txt",
    "scans_2_transcr_context_image.png",
    "vol_2/x_transcr_context_image.png",
)
"""An input tree: mixed types, names whose natural and plain orders differ,
Mabillon working folders, rendered files, a store, kept working folders and
context sidecars."""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    for name in _TREE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "empty").mkdir()
    return root


def _found(root: Path, kind: str, **options: object) -> list[str]:
    items = discover_items(root, kind, **options)  # type: ignore[arg-type]
    assert all(item.type == kind for item in items)
    return [item.path.relative_to(root).as_posix() for item in items]


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


@pytest.mark.unit
class TestItemsAndOrder:
    def test_a_folder_gives_its_own_files_of_the_type(self, tree: Path) -> None:
        assert _found(tree, "pdf") == ["b.PDF", "Ledger.pdf", "p2.pdf", "p10.pdf"]
        assert _found(tree, "epub") == ["a.epub"]
        assert _found(tree, "mobi") == ["k.azw3"]

    def test_audio_files_come_in_natural_order(self, tree: Path) -> None:
        assert _found(tree, "audio") == ["talk.mp3", "z.flac"]

    def test_recursive_takes_every_level_in_natural_path_order(
        self, tree: Path
    ) -> None:
        assert _found(tree, "pdf", recursive=True) == [
            "b.PDF",
            "Ledger.pdf",
            "p2.pdf",
            "p10.pdf",
            "vol_2/deep/y.pdf",
            "vol_2/x.pdf",
            "vol_10/x.pdf",
        ]

    def test_recursive_search_skips_kept_working_folders(self, tree: Path) -> None:
        assert _found(tree, "audio", recursive=True) == ["talk.mp3", "z.flac"]

    def test_a_skipped_working_folder_is_named_in_a_warning(self, tree: Path) -> None:
        found = find_items(tree, "audio", recursive=True)
        spec = make_spec(tree, type="audio", method="whisper", recursive=True)

        _, warnings = core.find_input(spec.input, tree)

        assert sorted(found.working) == [
            "p2-0b399eab/preprocessed_images_tesseract",
            "talk-8e219476/audio_chunks",
        ]
        (warning,) = warnings
        assert "and 1 more" in warning
        assert "--keep-working-files" in warning
        assert find_items(tree, "audio").working == ()

    def test_an_image_run_takes_the_folder_itself(self, tree: Path) -> None:
        assert discover_items(tree, "images") == [SourceItem(tree, "images")]

    def test_recursive_images_take_the_image_folders_at_every_level(
        self, tree: Path
    ) -> None:
        found = find_items(tree, "images", recursive=True)

        assert [item.path.relative_to(tree).as_posix() for item in found.items] == [
            "scans_2",
            "scans_10",
        ]
        assert found.nested == ((tree / "scans_2" / "inner", tree / "scans_2"),)

    def test_the_output_folder_inside_the_input_is_skipped(self, tree: Path) -> None:
        for name in ("Out/stray.pdf", "Out/scan/1.png", "Out/deep/s.pdf"):
            (tree / name).parent.mkdir(parents=True, exist_ok=True)
            (tree / name).write_bytes(b"x")

        pdfs = find_items(tree, "pdf", recursive=True, output_dir=tree / "out")
        folders = scan(tree, recursive=True, output_dir=tree / "Out")

        assert all("Out" not in item.path.parts for item in pdfs.items)
        assert all("Out" not in item.path.parts for item in folders.items)
        assert "Out/stray.pdf" in _found(tree, "pdf", recursive=True)

    def test_the_skip_list_names_the_tools_own_folders_and_sidecars(self) -> None:
        assert {PREPROCESSED_FOLDER, CHUNK_FOLDER} == WORKING_FOLDERS
        assert {CONTEXT_SUFFIX, IMAGE_CONTEXT_SUFFIX} == set(SIDECAR_SUFFIXES)

    def test_recursive_auto_mode_takes_the_items_at_every_level(
        self, tree: Path
    ) -> None:
        found = scan(tree, recursive=True)

        assert [
            (item.path.relative_to(tree).as_posix(), item.type) for item in found.items
        ] == [
            ("a.epub", "epub"),
            ("b.PDF", "pdf"),
            ("k.azw3", "mobi"),
            ("Ledger.pdf", "pdf"),
            ("p2.pdf", "pdf"),
            ("p10.pdf", "pdf"),
            ("scans_2", "images"),
            ("scans_10", "images"),
            ("vol_2/deep/y.pdf", "pdf"),
            ("vol_2/x.pdf", "pdf"),
            ("vol_10/x.pdf", "pdf"),
        ]
        assert found.loose_images == 1
        assert found.nested == ((tree / "scans_2" / "inner", tree / "scans_2"),)

    def test_a_nested_image_folder_is_named_in_a_warning(
        self, tree: Path, tmp_path: Path
    ) -> None:
        spec = make_spec(
            tree, type="images", recursive=True, output=tmp_path / "out", dry_run=True
        )

        warnings = core.plan(spec, make_settings()).warnings

        assert any(
            "'scans_2/inner'" in warning and "'scans_2'" in warning
            for warning in warnings
        )

    def test_files_name_items_taken_in_natural_order(self, tree: Path) -> None:
        assert _found(tree, "pdf", files=["vol_2/x.pdf", "p2.pdf"]) == [
            "p2.pdf",
            "vol_2/x.pdf",
        ]
        assert _found(tree, "images", files=["scans_2/inner", "scans_2"]) == [
            "scans_2",
            "scans_2/inner",
        ]
        assert _found(tree, "pdf", files=["vol_2\\deep\\y.pdf"]) == ["vol_2/deep/y.pdf"]

    def test_auto_mode_takes_the_direct_items_in_natural_order(
        self, tree: Path
    ) -> None:
        found = scan(tree)

        assert [
            (item.path.relative_to(tree).as_posix(), item.type) for item in found.items
        ] == [
            ("a.epub", "epub"),
            ("b.PDF", "pdf"),
            ("k.azw3", "mobi"),
            ("Ledger.pdf", "pdf"),
            ("p2.pdf", "pdf"),
            ("p10.pdf", "pdf"),
            ("scans_2", "images"),
            ("scans_10", "images"),
        ]
        assert found.loose_images == 1

    def test_a_linked_folder_is_not_entered(self, tree: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "l.pdf").write_bytes(b"x")
        (outside / "1.png").write_bytes(b"x")
        try:
            (tree / "link").symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory links need a privilege on this system")

        assert "link/l.pdf" not in _found(tree, "pdf", recursive=True)
        assert all(item.path.name != "link" for item in scan(tree).items)

    def test_a_junction_is_not_entered(self, tree: Path, tmp_path: Path) -> None:
        winapi = pytest.importorskip("_winapi")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "l.pdf").write_bytes(b"x")
        (outside / "1.png").write_bytes(b"x")
        winapi.CreateJunction(str(outside), str(tree / "junction"))

        assert "junction/l.pdf" not in _found(tree, "pdf", recursive=True)
        assert all(item.path.name != "junction" for item in scan(tree).items)

    def test_processing_keeps_the_discovery_order(
        self, tree: Path, tmp_path: Path
    ) -> None:
        spec = make_spec(
            tree, type="pdf", recursive=True, output=tmp_path / "out", dry_run=True
        )
        run_plan = core.plan(spec, make_settings())

        assert [item.path for item in run_plan.to_process] == [
            item.path for item in discover_items(tree, "pdf", recursive=True)
        ]

    def test_select_numbers_and_searches_the_paths_below_the_input(
        self, tree: Path, tmp_path: Path
    ) -> None:
        def chosen(expression: str) -> list[str]:
            spec = make_spec(
                tree,
                type="pdf",
                recursive=True,
                select=expression,
                output=tmp_path / "out",
                dry_run=True,
            )
            run_plan = core.plan(spec, make_settings())
            return [
                item.path.relative_to(tree).as_posix() for item in run_plan.to_process
            ]

        assert chosen("4-5") == ["p10.pdf", "vol_2/deep/y.pdf"]
        assert chosen("vol_2") == ["vol_2/deep/y.pdf", "vol_2/x.pdf"]


def _location(
    tree: Path, relative: str, kind: str, layout: str, output: Path | None
) -> tuple[str, str]:
    root, name, path = core.item_location(
        SourceItem(tree / relative, kind),
        output_dir=output,
        layout=layout,
        input_root=tree,
    )
    return name, path.relative_to(root).as_posix()


def _hashed(relative: str) -> str:
    return Path(create_safe_directory_name(str(Path(relative)))).as_posix()


@pytest.mark.unit
class TestNamesAndOutputs:
    def test_the_found_items_keep_their_names_and_outputs(
        self, tree: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        items = [
            *discover_items(tree, "pdf", recursive=True),
            *discover_items(tree, "images", files=["scans_2"]),
        ]
        names = {
            item.path.relative_to(tree).as_posix(): [
                _location(tree, item.path.relative_to(tree).as_posix(), item.type, *how)
                for how in (("hash", out), ("mirror", out), ("hash", None))
            ]
            for item in items
        }

        assert names["vol_2/deep/y.pdf"] == [
            (
                f"{_hashed('vol_2/deep/y.pdf')}/y",
                f"{_hashed('vol_2/deep/y.pdf')}/y.txt",
            ),
            ("vol_2/deep/y", "vol_2/deep/y.txt"),
            ("y", "y.txt"),
        ]
        assert names["p2.pdf"] == [
            (f"{_hashed('p2.pdf')}/p2", f"{_hashed('p2.pdf')}/p2.txt"),
            ("p2", "p2.txt"),
            ("p2", "p2.txt"),
        ]
        assert names["scans_2"] == [
            (f"{_hashed('scans_2')}/scans_2", f"{_hashed('scans_2')}/scans_2.txt"),
            ("scans_2/scans_2", "scans_2/scans_2.txt"),
            ("scans_2", "scans_2.txt"),
        ]
        assert set(names) == {
            "b.PDF",
            "Ledger.pdf",
            "p10.pdf",
            "p2.pdf",
            "vol_10/x.pdf",
            "vol_2/x.pdf",
            "vol_2/deep/y.pdf",
            "scans_2",
        }
