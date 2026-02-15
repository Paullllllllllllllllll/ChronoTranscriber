"""Unit tests for modules.core.path_config.PathConfig."""

from __future__ import annotations

import pytest
from pathlib import Path


class TestPathConfig:
    """Tests for PathConfig dataclass and factory."""

    @pytest.fixture
    def sample_paths_config(self) -> dict:
        return {
            "general": {
                "input_paths_is_output_path": False,
            },
            "file_paths": {
                "PDFs": {"input": "pdfs_in", "output": "pdfs_out"},
                "Images": {"input": "images_in", "output": "images_out"},
                "EPUBs": {"input": "epubs_in", "output": "epubs_out"},
                "MOBIs": {"input": "mobis_in", "output": "mobis_out"},
                "Auto": {"input": "auto_in", "output": "auto_out"},
            },
        }

    @pytest.mark.unit
    def test_from_paths_config_basic(self, sample_paths_config):
        from modules.core.path_config import PathConfig

        pc = PathConfig.from_paths_config(sample_paths_config)
        assert pc.pdf_input_dir == Path("pdfs_in")
        assert pc.pdf_output_dir == Path("pdfs_out")
        assert pc.image_input_dir == Path("images_in")
        assert pc.image_output_dir == Path("images_out")
        assert pc.epub_input_dir == Path("epubs_in")
        assert pc.epub_output_dir == Path("epubs_out")
        assert pc.mobi_input_dir == Path("mobis_in")
        assert pc.mobi_output_dir == Path("mobis_out")
        assert pc.auto_input_dir == Path("auto_in")
        assert pc.auto_output_dir == Path("auto_out")
        assert pc.use_input_as_output is False

    @pytest.mark.unit
    def test_from_paths_config_use_input_as_output(self, sample_paths_config):
        from modules.core.path_config import PathConfig

        sample_paths_config["general"]["input_paths_is_output_path"] = True
        pc = PathConfig.from_paths_config(sample_paths_config)
        assert pc.use_input_as_output is True

    @pytest.mark.unit
    def test_from_paths_config_defaults_on_empty(self):
        from modules.core.path_config import PathConfig

        pc = PathConfig.from_paths_config({})
        assert pc.pdf_input_dir == Path("pdfs_in")
        assert pc.pdf_output_dir == Path("pdfs_out")
        assert pc.use_input_as_output is False

    @pytest.mark.unit
    def test_base_dirs_for_type(self, sample_paths_config):
        from modules.core.path_config import PathConfig

        pc = PathConfig.from_paths_config(sample_paths_config)
        assert pc.base_dirs_for_type("pdfs") == (Path("pdfs_in"), Path("pdfs_out"))
        assert pc.base_dirs_for_type("images") == (Path("images_in"), Path("images_out"))
        assert pc.base_dirs_for_type("epubs") == (Path("epubs_in"), Path("epubs_out"))
        assert pc.base_dirs_for_type("mobis") == (Path("mobis_in"), Path("mobis_out"))
        assert pc.base_dirs_for_type("auto") == (Path("auto_in"), Path("auto_out"))

    @pytest.mark.unit
    def test_base_dirs_for_type_invalid_raises(self, sample_paths_config):
        from modules.core.path_config import PathConfig

        pc = PathConfig.from_paths_config(sample_paths_config)
        with pytest.raises(ValueError, match="Unknown processing type"):
            pc.base_dirs_for_type("invalid")

    @pytest.mark.unit
    def test_ensure_output_dirs_creates_directories(self, tmp_path):
        from modules.core.path_config import PathConfig

        pc = PathConfig(
            pdf_output_dir=tmp_path / "pdfs",
            image_output_dir=tmp_path / "images",
            epub_output_dir=tmp_path / "epubs",
            mobi_output_dir=tmp_path / "mobis",
            use_input_as_output=False,
        )
        pc.ensure_output_dirs()
        assert (tmp_path / "pdfs").is_dir()
        assert (tmp_path / "images").is_dir()
        assert (tmp_path / "epubs").is_dir()
        assert (tmp_path / "mobis").is_dir()

    @pytest.mark.unit
    def test_ensure_output_dirs_skips_when_input_as_output(self, tmp_path):
        from modules.core.path_config import PathConfig

        pc = PathConfig(
            pdf_output_dir=tmp_path / "pdfs",
            image_output_dir=tmp_path / "images",
            epub_output_dir=tmp_path / "epubs",
            mobi_output_dir=tmp_path / "mobis",
            use_input_as_output=True,
        )
        pc.ensure_output_dirs()
        assert not (tmp_path / "pdfs").exists()

    @pytest.mark.unit
    def test_ensure_input_dirs_creates_directories(self, tmp_path):
        from modules.core.path_config import PathConfig

        pc = PathConfig(
            pdf_input_dir=tmp_path / "pdfs_in",
            image_input_dir=tmp_path / "images_in",
            epub_input_dir=tmp_path / "epubs_in",
            auto_input_dir=tmp_path / "auto_in",
            auto_output_dir=tmp_path / "auto_out",
        )
        pc.ensure_input_dirs()
        assert (tmp_path / "pdfs_in").is_dir()
        assert (tmp_path / "images_in").is_dir()
        assert (tmp_path / "epubs_in").is_dir()
        assert (tmp_path / "auto_in").is_dir()
        assert (tmp_path / "auto_out").is_dir()
