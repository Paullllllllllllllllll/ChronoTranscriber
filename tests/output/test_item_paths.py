"""Item output paths: beside the input, hash folders and the mirror layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronotranscriber.output.paths import (
    create_safe_directory_name,
    item_paths,
    prepare_item_paths,
    relative_key,
)


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "in" / "sub" / "doc.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF")
    return path


@pytest.mark.unit
def test_the_hash_layout_keys_the_folder_by_the_relative_path(
    tmp_path: Path, pdf: Path
) -> None:
    out = tmp_path / "out"
    paths = item_paths(pdf, output_dir=out, layout="hash", input_root=tmp_path / "in")

    assert paths.folder == out / create_safe_directory_name(str(Path("sub/doc.pdf")))
    assert paths.output.name == "doc.txt"
    assert paths.output.parent == paths.folder


@pytest.mark.unit
def test_a_single_file_input_keys_the_folder_by_its_stem(
    tmp_path: Path, pdf: Path
) -> None:
    paths = item_paths(pdf, output_dir=tmp_path / "out", layout="hash", input_root=pdf)
    assert paths.folder.name == create_safe_directory_name("doc")


@pytest.mark.unit
def test_beside_the_input_the_output_sits_next_to_it(
    pdf: Path,
) -> None:
    paths = item_paths(pdf, output_dir=None, layout="hash", input_root=pdf.parent)

    assert paths.output == pdf.parent / "doc.txt"
    assert paths.folder == pdf.parent / create_safe_directory_name("doc")


@pytest.mark.unit
def test_the_mirror_layout_replicates_the_input_tree(tmp_path: Path, pdf: Path) -> None:
    out = tmp_path / "out"
    paths = item_paths(
        pdf,
        output_dir=out,
        layout="mirror",
        input_root=tmp_path / "in",
        output_format="md",
    )

    assert paths.folder == out / "sub"
    assert paths.output.name == "doc.md"


@pytest.mark.unit
def test_an_image_folder_mirrors_itself(tmp_path: Path) -> None:
    folder = tmp_path / "in" / "vol1"
    folder.mkdir(parents=True)
    out = tmp_path / "out"

    paths = item_paths(
        folder, output_dir=out, layout="mirror", input_root=folder.parent
    )

    assert paths.folder == out / "vol1"
    assert paths.output.name == "vol1.txt"


@pytest.mark.unit
def test_without_mirroring_the_hash_layout_applies(tmp_path: Path, pdf: Path) -> None:
    out = tmp_path / "out"
    paths = item_paths(
        pdf, output_dir=out, layout="mirror", input_root=pdf, mirror=False
    )
    assert paths.folder.parent == out
    assert paths.folder.name == create_safe_directory_name("doc")


@pytest.mark.unit
def test_prepare_creates_only_the_folder_of_the_output(
    tmp_path: Path, pdf: Path
) -> None:
    paths = prepare_item_paths(
        pdf, output_dir=tmp_path / "out", layout="hash", input_root=pdf
    )
    assert paths.output.parent.is_dir()
    assert list(paths.output.parent.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["hash", "mirror"])
def test_prepare_writes_no_file_in_any_layout(
    tmp_path: Path, pdf: Path, layout: str
) -> None:
    out = tmp_path / "out"
    prepare_item_paths(pdf, output_dir=out, layout=layout, input_root=pdf.parent)
    assert [path for path in out.rglob("*") if path.is_file()] == []


@pytest.mark.unit
def test_beside_the_input_prepare_creates_no_working_folder(pdf: Path) -> None:
    before = sorted(pdf.parent.iterdir())
    paths = prepare_item_paths(
        pdf, output_dir=None, layout="hash", input_root=pdf.parent
    )
    assert sorted(pdf.parent.iterdir()) == before
    assert not paths.folder.exists()


@pytest.mark.unit
class TestSingleFileInputKey:
    def test_relative_key_is_none_when_item_equals_input_root(
        self, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        assert relative_key(pdf, pdf) is None

    def test_relative_key_for_directory_input_is_the_relative_path(
        self, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        assert relative_key(pdf, tmp_path) == "book.pdf"
