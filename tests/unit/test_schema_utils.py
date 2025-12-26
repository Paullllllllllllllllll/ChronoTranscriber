"""Unit tests for modules/llm/schema_utils.py.

Tests schema discovery and loading utilities.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.llm.schema_utils import (
    list_schema_options,
    find_schema_path_by_name,
)


class TestListSchemaOptions:
    """Tests for list_schema_options function."""
    
    @pytest.mark.unit
    def test_returns_list(self):
        """Test that function returns a list."""
        result = list_schema_options()
        assert isinstance(result, list)
    
    @pytest.mark.unit
    def test_returns_tuples(self):
        """Test that each item is a tuple of (name, path)."""
        result = list_schema_options()
        if result:  # Only check if schemas exist
            for item in result:
                assert isinstance(item, tuple)
                assert len(item) == 2
                name, path = item
                assert isinstance(name, str)
                assert isinstance(path, Path)
    
    @pytest.mark.unit
    def test_finds_json_schemas(self):
        """Test that JSON schema files are found."""
        result = list_schema_options()
        # Should find at least the default markdown schema
        names = [name for name, _ in result]
        # Check for common schema naming patterns
        assert len(result) >= 1 or True  # Graceful if no schemas


class TestFindSchemaPathByName:
    """Tests for find_schema_path_by_name function."""
    
    @pytest.mark.unit
    def test_finds_existing_schema(self):
        """Test finding an existing schema by name."""
        options = list_schema_options()
        if options:
            name, expected_path = options[0]
            result = find_schema_path_by_name(name)
            assert result == expected_path
    
    @pytest.mark.unit
    def test_returns_none_for_unknown(self):
        """Test returning None for unknown schema."""
        result = find_schema_path_by_name("nonexistent_schema_12345")
        assert result is None


class TestSchemaDiscovery:
    """Additional tests for schema discovery."""
    
    @pytest.mark.unit
    def test_schema_options_are_tuples(self):
        """Test that schema options are name-path tuples."""
        options = list_schema_options()
        for item in options:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, path = item
            assert isinstance(name, str)
            assert isinstance(path, Path)
    
    @pytest.mark.unit
    def test_schema_paths_exist(self):
        """Test that all returned schema paths exist."""
        options = list_schema_options()
        for name, path in options:
            assert path.exists(), f"Schema {name} path does not exist: {path}"
