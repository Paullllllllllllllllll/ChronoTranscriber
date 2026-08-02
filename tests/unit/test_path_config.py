"""Unit tests for modules.infra.paths.PathConfig."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestPathConfig:
    """Tests for PathConfig dataclass and factory."""

    @pytest.fixture
    def sample_paths_config(self, tmp_path: Path) -> dict:
        return {
            "general": {
                "input_paths_is_output_path": False,
            },
            "file_paths": {
                "PDFs": {
                    "input": str(tmp_path / "pdfs_in"),
                    "output": str(tmp_path / "pdfs_out"),
                },
                "Images": {
                    "input": str(tmp_path / "images_in"),
                    "output": str(tmp_path / "images_out"),
                },
                "EPUBs": {
                    "input": str(tmp_path / "epubs_in"),
                    "output": str(tmp_path / "epubs_out"),
                },
                "MOBIs": {
                    "input": str(tmp_path / "mobis_in"),
                    "output": str(tmp_path / "mobis_out"),
                },
                "Auto": {
                    "input": str(tmp_path / "auto_in"),
                    "output": str(tmp_path / "auto_out"),
                },
            },
        }

    @pytest.mark.unit
    def test_from_paths_config_basic(
        self,
        sample_paths_config: dict,
        tmp_path: Path,
    ) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config(sample_paths_config)
        assert pc.pdf_input_dir == tmp_path / "pdfs_in"
        assert pc.pdf_output_dir == tmp_path / "pdfs_out"
        assert pc.image_input_dir == tmp_path / "images_in"
        assert pc.image_output_dir == tmp_path / "images_out"
        assert pc.epub_input_dir == tmp_path / "epubs_in"
        assert pc.epub_output_dir == tmp_path / "epubs_out"
        assert pc.mobi_input_dir == tmp_path / "mobis_in"
        assert pc.mobi_output_dir == tmp_path / "mobis_out"
        assert pc.auto_input_dir == tmp_path / "auto_in"
        assert pc.auto_output_dir == tmp_path / "auto_out"
        assert pc.use_input_as_output is False

    @pytest.mark.unit
    def test_from_paths_config_use_input_as_output(
        self,
        sample_paths_config: dict,
    ) -> None:
        from modules.infra.paths import PathConfig

        sample_paths_config["general"]["input_paths_is_output_path"] = True
        pc = PathConfig.from_paths_config(sample_paths_config)
        assert pc.use_input_as_output is True

    @pytest.mark.unit
    def test_from_paths_config_defaults_on_empty(self) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config({})
        assert pc.pdf_input_dir == Path("pdfs_in")
        assert pc.pdf_output_dir == Path("pdfs_out")
        assert pc.use_input_as_output is False

    @pytest.mark.unit
    def test_base_dirs_for_type(
        self,
        sample_paths_config: dict,
        tmp_path: Path,
    ) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config(sample_paths_config)
        assert pc.base_dirs_for_type("pdfs") == (
            tmp_path / "pdfs_in",
            tmp_path / "pdfs_out",
        )
        assert pc.base_dirs_for_type("images") == (
            tmp_path / "images_in",
            tmp_path / "images_out",
        )
        assert pc.base_dirs_for_type("epubs") == (
            tmp_path / "epubs_in",
            tmp_path / "epubs_out",
        )
        assert pc.base_dirs_for_type("mobis") == (
            tmp_path / "mobis_in",
            tmp_path / "mobis_out",
        )
        assert pc.base_dirs_for_type("auto") == (
            tmp_path / "auto_in",
            tmp_path / "auto_out",
        )

    @pytest.mark.unit
    def test_base_dirs_for_type_invalid_raises(self, sample_paths_config: dict) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config(sample_paths_config)
        with pytest.raises(ValueError, match="Unknown processing type"):
            pc.base_dirs_for_type("invalid")

    @pytest.mark.unit
    def test_ensure_output_dirs_creates_directories(self, tmp_path: Path) -> None:
        from modules.infra.paths import PathConfig

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
    def test_ensure_output_dirs_skips_when_input_as_output(
        self, tmp_path: Path
    ) -> None:
        from modules.infra.paths import PathConfig

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
    def test_ensure_input_dirs_creates_directories(self, tmp_path: Path) -> None:
        from modules.infra.paths import PathConfig

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


class TestPathConfigAudioBackCompat:
    """A pre-audio paths_config (no ``Audio`` section) must still load."""

    @pytest.fixture
    def legacy_paths_config(self, tmp_path: Path) -> dict:
        """A paths_config written before the audio workflow existed."""
        return {
            "general": {"input_paths_is_output_path": False},
            "file_paths": {
                "PDFs": {
                    "input": str(tmp_path / "pdfs_in"),
                    "output": str(tmp_path / "pdfs_out"),
                },
                "Images": {
                    "input": str(tmp_path / "images_in"),
                    "output": str(tmp_path / "images_out"),
                },
                "EPUBs": {
                    "input": str(tmp_path / "epubs_in"),
                    "output": str(tmp_path / "epubs_out"),
                },
                "MOBIs": {
                    "input": str(tmp_path / "mobis_in"),
                    "output": str(tmp_path / "mobis_out"),
                },
            },
        }

    @pytest.mark.unit
    def test_from_paths_config_succeeds_without_an_audio_section(
        self, legacy_paths_config: dict
    ) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config(legacy_paths_config)
        assert pc is not None

    @pytest.mark.unit
    def test_audio_dirs_fall_back_to_the_documented_defaults(
        self, legacy_paths_config: dict
    ) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config(legacy_paths_config)
        assert pc.audio_input_dir == Path("audio_in")
        assert pc.audio_output_dir == Path("audio_out")

    @pytest.mark.unit
    def test_configured_audio_dirs_are_used_when_present(self, tmp_path: Path) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig.from_paths_config(
            {
                "general": {},
                "file_paths": {
                    "Audio": {
                        "input": str(tmp_path / "recordings"),
                        "output": str(tmp_path / "transcripts"),
                    }
                },
            }
        )
        assert pc.audio_input_dir == tmp_path / "recordings"
        assert pc.audio_output_dir == tmp_path / "transcripts"

    @pytest.mark.unit
    def test_base_dirs_for_type_supports_audio(self, tmp_path: Path) -> None:
        from modules.infra.paths import PathConfig

        pc = PathConfig(
            audio_input_dir=tmp_path / "ai",
            audio_output_dir=tmp_path / "ao",
        )
        assert pc.base_dirs_for_type("audio") == (tmp_path / "ai", tmp_path / "ao")

    @pytest.mark.unit
    def test_ensure_output_dirs_still_creates_exactly_four_directories(
        self, tmp_path: Path
    ) -> None:
        from modules.infra.paths import PathConfig

        root = tmp_path / "outputs"
        pc = PathConfig(
            pdf_output_dir=root / "pdfs",
            image_output_dir=root / "images",
            epub_output_dir=root / "epubs",
            mobi_output_dir=root / "mobis",
            audio_output_dir=root / "audio",
            auto_output_dir=root / "auto",
            use_input_as_output=False,
        )
        pc.ensure_output_dirs()
        assert sorted(p.name for p in root.iterdir()) == [
            "epubs",
            "images",
            "mobis",
            "pdfs",
        ]

    @pytest.mark.unit
    def test_ensure_output_dirs_does_not_materialize_the_audio_dir(
        self, tmp_path: Path
    ) -> None:
        # The audio workflow creates its output dir lazily, so an image or PDF
        # run must never leave an empty audio_out/ behind.
        from modules.infra.paths import PathConfig

        pc = PathConfig(
            pdf_output_dir=tmp_path / "pdfs",
            image_output_dir=tmp_path / "images",
            epub_output_dir=tmp_path / "epubs",
            mobi_output_dir=tmp_path / "mobis",
            audio_output_dir=tmp_path / "audio_out",
            use_input_as_output=False,
        )
        pc.ensure_output_dirs()
        assert not (tmp_path / "audio_out").exists()
