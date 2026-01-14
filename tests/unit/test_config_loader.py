"""Unit tests for modules/config/config_loader.py."""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile


class TestIsRepoRoot:
    """Tests for _is_repo_root function."""

    @pytest.mark.unit
    def test_detects_config_and_schemas_dirs(self, temp_dir):
        """Test detection when both config/ and schemas/ exist."""
        from modules.config.config_loader import _is_repo_root
        
        (temp_dir / "config").mkdir()
        (temp_dir / "schemas").mkdir()
        
        assert _is_repo_root(temp_dir) is True

    @pytest.mark.unit
    def test_detects_readme_and_requirements(self, temp_dir):
        """Test detection when README.md and requirements.txt exist."""
        from modules.config.config_loader import _is_repo_root
        
        (temp_dir / "README.md").write_text("# Test")
        (temp_dir / "requirements.txt").write_text("pytest")
        
        assert _is_repo_root(temp_dir) is True

    @pytest.mark.unit
    def test_returns_false_for_empty_dir(self, temp_dir):
        """Test returns False for empty directory."""
        from modules.config.config_loader import _is_repo_root
        
        assert _is_repo_root(temp_dir) is False

    @pytest.mark.unit
    def test_returns_false_for_partial_markers(self, temp_dir):
        """Test returns False when only some markers exist."""
        from modules.config.config_loader import _is_repo_root
        
        (temp_dir / "config").mkdir()
        # No schemas dir
        
        assert _is_repo_root(temp_dir) is False


class TestExpandPathStr:
    """Tests for _expand_path_str function."""

    @pytest.mark.unit
    def test_expands_user_home(self):
        """Test expansion of ~ to user home directory."""
        from modules.config.config_loader import _expand_path_str
        
        result = _expand_path_str("~/test")
        assert "~" not in str(result)
        assert "test" in str(result)

    @pytest.mark.unit
    def test_expands_env_vars(self):
        """Test expansion of environment variables."""
        from modules.config.config_loader import _expand_path_str
        
        with patch.dict(os.environ, {"TEST_VAR": "myvalue"}):
            result = _expand_path_str("$TEST_VAR/path")
            assert "myvalue" in str(result)

    @pytest.mark.unit
    def test_returns_path_object(self):
        """Test that function returns a Path object."""
        from modules.config.config_loader import _expand_path_str
        
        result = _expand_path_str("/some/path")
        assert isinstance(result, Path)


class TestProjectRoot:
    """Tests for PROJECT_ROOT constant."""

    @pytest.mark.unit
    def test_project_root_exists(self):
        """Test that PROJECT_ROOT points to an existing directory."""
        from modules.config.config_loader import PROJECT_ROOT
        
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    @pytest.mark.unit
    def test_project_root_has_config_dir(self):
        """Test that PROJECT_ROOT contains config directory."""
        from modules.config.config_loader import PROJECT_ROOT
        
        assert (PROJECT_ROOT / "config").exists()

    @pytest.mark.unit
    def test_project_root_has_schemas_dir(self):
        """Test that PROJECT_ROOT contains schemas directory."""
        from modules.config.config_loader import PROJECT_ROOT
        
        assert (PROJECT_ROOT / "schemas").exists()


class TestConfigDir:
    """Tests for CONFIG_DIR constant."""

    @pytest.mark.unit
    def test_config_dir_exists(self):
        """Test that CONFIG_DIR points to existing directory."""
        from modules.config.config_loader import CONFIG_DIR
        
        assert CONFIG_DIR.exists()
        assert CONFIG_DIR.is_dir()

    @pytest.mark.unit
    def test_config_dir_contains_yaml_files(self):
        """Test that CONFIG_DIR contains YAML config files."""
        from modules.config.config_loader import CONFIG_DIR
        
        yaml_files = list(CONFIG_DIR.glob("*.yaml"))
        assert len(yaml_files) > 0


class TestTranscriptionModelDataclass:
    """Tests for _TranscriptionModel dataclass."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values for _TranscriptionModel."""
        from modules.config.config_loader import _TranscriptionModel
        
        model = _TranscriptionModel(name="gpt-4o")
        
        assert model.name == "gpt-4o"
        assert model.expects_image_inputs is True
        assert model.max_output_tokens is None
        assert model.temperature is None

    @pytest.mark.unit
    def test_custom_values(self):
        """Test custom values for _TranscriptionModel."""
        from modules.config.config_loader import _TranscriptionModel
        
        model = _TranscriptionModel(
            name="custom-model",
            expects_image_inputs=False,
            max_output_tokens=4096,
            temperature=0.5,
        )
        
        assert model.name == "custom-model"
        assert model.expects_image_inputs is False
        assert model.max_output_tokens == 4096
        assert model.temperature == 0.5


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    @pytest.mark.unit
    def test_initialization_default_path(self):
        """Test ConfigLoader initialization with default path."""
        from modules.config.config_loader import ConfigLoader, DEFAULT_CONFIG_PATH
        
        loader = ConfigLoader()
        assert loader.config_path == DEFAULT_CONFIG_PATH

    @pytest.mark.unit
    def test_initialization_custom_path(self, temp_dir):
        """Test ConfigLoader initialization with custom path."""
        from modules.config.config_loader import ConfigLoader
        
        custom_path = temp_dir / "custom_config.yaml"
        loader = ConfigLoader(config_path=custom_path)
        assert loader.config_path == custom_path

    @pytest.mark.unit
    def test_load_yaml_file_success(self, temp_dir):
        """Test successful YAML file loading."""
        from modules.config.config_loader import ConfigLoader
        
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  item: 123")
        
        result = ConfigLoader._load_yaml_file(yaml_file)
        
        assert result == {"key": "value", "nested": {"item": 123}}

    @pytest.mark.unit
    def test_load_yaml_file_missing(self, temp_dir):
        """Test loading missing YAML file raises error."""
        from modules.config.config_loader import ConfigLoader
        
        missing_file = temp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            ConfigLoader._load_yaml_file(missing_file)

    @pytest.mark.unit
    def test_load_yaml_file_invalid_syntax(self, temp_dir):
        """Test loading YAML with invalid syntax raises error."""
        from modules.config.config_loader import ConfigLoader
        
        yaml_file = temp_dir / "invalid.yaml"
        # Use truly invalid YAML with a tab character in indentation which triggers ScannerError
        yaml_file.write_text("key:\n\t- invalid tab indent")
        
        with pytest.raises(ValueError) as exc_info:
            ConfigLoader._load_yaml_file(yaml_file)
        
        assert "YAML parsing error" in str(exc_info.value)

    @pytest.mark.unit
    def test_load_yaml_file_empty(self, temp_dir):
        """Test loading empty YAML file returns empty dict."""
        from modules.config.config_loader import ConfigLoader
        
        yaml_file = temp_dir / "empty.yaml"
        yaml_file.write_text("")
        
        result = ConfigLoader._load_yaml_file(yaml_file)
        assert result == {}

    @pytest.mark.unit
    def test_get_model_config_returns_copy(self):
        """Test that get_model_config returns a copy."""
        from modules.config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        loader._raw = {"key": "value"}
        
        config1 = loader.get_model_config()
        config2 = loader.get_model_config()
        
        assert config1 == config2
        config1["key"] = "modified"
        assert config2["key"] == "value"  # Original unchanged

    @pytest.mark.unit
    def test_get_paths_config_caches_result(self):
        """Test that get_paths_config caches the result."""
        from modules.config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        # First call loads and caches
        config1 = loader.get_paths_config()
        # Second call returns cached
        config2 = loader.get_paths_config()
        
        assert config1 == config2

    @pytest.mark.unit
    def test_to_abs_with_absolute_path(self):
        """Test _to_abs with absolute path."""
        from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
        
        abs_path = "/absolute/path"
        result = ConfigLoader._to_abs(abs_path, PROJECT_ROOT)
        
        # Should contain the absolute path (resolved)
        assert "absolute" in result

    @pytest.mark.unit
    def test_to_abs_with_relative_path(self):
        """Test _to_abs with relative path."""
        from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
        
        result = ConfigLoader._to_abs("relative/path", PROJECT_ROOT)
        
        assert str(PROJECT_ROOT) in result
        assert "relative" in result

    @pytest.mark.unit
    def test_to_abs_with_none(self):
        """Test _to_abs with None returns None."""
        from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
        
        result = ConfigLoader._to_abs(None, PROJECT_ROOT)
        assert result is None

    @pytest.mark.unit
    def test_to_abs_with_empty_string(self):
        """Test _to_abs with empty string returns empty string."""
        from modules.config.config_loader import ConfigLoader, PROJECT_ROOT
        
        result = ConfigLoader._to_abs("", PROJECT_ROOT)
        assert result == ""


class TestNormalizePathsConfig:
    """Tests for _normalize_paths_config method."""

    @pytest.mark.unit
    def test_normalizes_general_paths(self):
        """Test normalization of general section paths."""
        from modules.config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        config = {
            "general": {
                "logs_dir": "./logs",
            }
        }
        
        result = loader._normalize_paths_config(config)
        
        # Should be absolute path
        assert Path(result["general"]["logs_dir"]).is_absolute()

    @pytest.mark.unit
    def test_normalizes_file_paths_sections(self):
        """Test normalization of file_paths sections."""
        from modules.config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        config = {
            "file_paths": {
                "PDFs": {"input": "./pdfs_in", "output": "./pdfs_out"},
                "Images": {"input": "./images_in", "output": "./images_out"},
            }
        }
        
        result = loader._normalize_paths_config(config)
        
        # All paths should be absolute
        assert Path(result["file_paths"]["PDFs"]["input"]).is_absolute()
        assert Path(result["file_paths"]["PDFs"]["output"]).is_absolute()
        assert Path(result["file_paths"]["Images"]["input"]).is_absolute()
        assert Path(result["file_paths"]["Images"]["output"]).is_absolute()

    @pytest.mark.unit
    def test_handles_empty_config(self):
        """Test handling of empty config."""
        from modules.config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        result = loader._normalize_paths_config({})
        
        assert result == {"general": {}, "file_paths": {}}

    @pytest.mark.unit
    def test_preserves_non_path_values(self):
        """Test that non-path values are preserved."""
        from modules.config.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        config = {
            "general": {
                "interactive_mode": True,
                "retain_temporary_jsonl": False,
            }
        }
        
        result = loader._normalize_paths_config(config)
        
        assert result["general"]["interactive_mode"] is True
        assert result["general"]["retain_temporary_jsonl"] is False
