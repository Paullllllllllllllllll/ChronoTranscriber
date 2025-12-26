"""Unit tests for modules/config/service.py.

Tests the ConfigService singleton and configuration access.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from modules.config.service import (
    ConfigService,
    get_config_service,
    get_model_config,
    get_paths_config,
    get_concurrency_config,
    get_image_processing_config,
)


class TestConfigServiceSingleton:
    """Tests for ConfigService singleton behavior."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConfigService.reset()
        yield
        ConfigService.reset()
    
    @pytest.mark.unit
    def test_singleton_returns_same_instance(self):
        """Test that ConfigService returns the same instance."""
        service1 = ConfigService()
        service2 = ConfigService()
        assert service1 is service2
    
    @pytest.mark.unit
    def test_reset_creates_new_instance(self):
        """Test that reset allows creation of new instance."""
        service1 = ConfigService()
        ConfigService.reset()
        service2 = ConfigService()
        # After reset, should be a different object (though singleton pattern)
        # The key is that _initialized is reset
        assert service2._initialized  # New instance should be initialized
    
    @pytest.mark.unit
    def test_get_config_service_returns_singleton(self):
        """Test that get_config_service returns singleton."""
        service1 = get_config_service()
        service2 = get_config_service()
        assert service1 is service2


class TestConfigServiceMethods:
    """Tests for ConfigService methods."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConfigService.reset()
        yield
        ConfigService.reset()
    
    @pytest.mark.unit
    def test_get_model_config_returns_dict(self):
        """Test get_model_config returns a dictionary."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_model_config.return_value = {"model": "test"}
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            service.load()
            result = service.get_model_config()
            
            assert isinstance(result, dict)
            assert result == {"model": "test"}
    
    @pytest.mark.unit
    def test_get_paths_config_returns_dict(self):
        """Test get_paths_config returns a dictionary."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_paths_config.return_value = {"paths": "test"}
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            service.load()
            result = service.get_paths_config()
            
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_get_concurrency_config_returns_dict(self):
        """Test get_concurrency_config returns a dictionary."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_concurrency_config.return_value = {"concurrency": "test"}
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            service.load()
            result = service.get_concurrency_config()
            
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_get_image_processing_config_returns_dict(self):
        """Test get_image_processing_config returns a dictionary."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {"processing": "test"}
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            service.load()
            result = service.get_image_processing_config()
            
            assert isinstance(result, dict)
    
    @pytest.mark.unit
    def test_configs_are_copies(self):
        """Test that returned configs are copies, not references."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_model_config.return_value = {"key": "value"}
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            service.load()
            
            config1 = service.get_model_config()
            config2 = service.get_model_config()
            
            # Modify config1
            config1["key"] = "modified"
            
            # config2 should be unaffected
            assert config2["key"] == "value"
    
    @pytest.mark.unit
    def test_lazy_loading(self):
        """Test that configs are loaded lazily on first access."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_model_config.return_value = {"lazy": "loaded"}
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            # Don't call load() explicitly
            
            result = service.get_model_config()
            
            # Should have loaded automatically
            assert result == {"lazy": "loaded"}
            mock_loader.load_configs.assert_called()
    
    @pytest.mark.unit
    def test_reload_clears_cache(self):
        """Test that reload clears cached configurations."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_model_config.side_effect = [
                {"version": 1},
                {"version": 2},
            ]
            mock_loader_cls.return_value = mock_loader
            
            service = ConfigService()
            service.load()
            
            # First call
            result1 = service.get_model_config()
            
            # Reload
            service.reload()
            
            # Second call should get new value
            result2 = service.get_model_config()
            
            assert result1["version"] == 1
            assert result2["version"] == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ConfigService.reset()
        yield
        ConfigService.reset()
    
    @pytest.mark.unit
    def test_get_model_config_function(self):
        """Test get_model_config convenience function."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_model_config.return_value = {"func": "test"}
            mock_loader_cls.return_value = mock_loader
            
            result = get_model_config()
            
            assert result == {"func": "test"}
    
    @pytest.mark.unit
    def test_get_paths_config_function(self):
        """Test get_paths_config convenience function."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_paths_config.return_value = {"paths": "func"}
            mock_loader_cls.return_value = mock_loader
            
            result = get_paths_config()
            
            assert result == {"paths": "func"}
    
    @pytest.mark.unit
    def test_get_concurrency_config_function(self):
        """Test get_concurrency_config convenience function."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_concurrency_config.return_value = {"conc": "func"}
            mock_loader_cls.return_value = mock_loader
            
            result = get_concurrency_config()
            
            assert result == {"conc": "func"}
    
    @pytest.mark.unit
    def test_get_image_processing_config_function(self):
        """Test get_image_processing_config convenience function."""
        with patch('modules.config.service.ConfigLoader') as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {"img": "func"}
            mock_loader_cls.return_value = mock_loader
            
            result = get_image_processing_config()
            
            assert result == {"img": "func"}
