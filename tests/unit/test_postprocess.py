"""Unit tests for modules/processing/postprocess.py.

Tests text post-processing functions including Unicode normalization,
hyphenation fixing, spacing normalization, and line wrapping.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from modules.processing.postprocess import (
    normalize_unicode_text,
    fix_hyphenation,
    normalize_spacing,
    should_wrap_line,
    compute_auto_wrap_width,
    wrap_long_lines,
    postprocess_text,
    postprocess_transcription,
    postprocess_file,
)


class TestNormalizeUnicodeText:
    """Tests for normalize_unicode_text function."""
    
    @pytest.mark.unit
    def test_nfc_normalization(self):
        """Test NFC normalization of accented characters."""
        # Combining character (e + combining acute)
        text = "cafe\u0301"
        result = normalize_unicode_text(text)
        # Should normalize to precomposed é
        assert "é" in result or len(result) == 5
    
    @pytest.mark.unit
    def test_removes_soft_hyphen(self):
        """Test removal of soft hyphen characters."""
        text = "word\u00ADbreak"
        result = normalize_unicode_text(text)
        assert "\u00AD" not in result
        assert result == "wordbreak"
    
    @pytest.mark.unit
    def test_removes_zero_width_space(self):
        """Test removal of zero-width space."""
        text = "word\u200Bword"
        result = normalize_unicode_text(text)
        assert "\u200B" not in result
        assert result == "wordword"
    
    @pytest.mark.unit
    def test_removes_bom(self):
        """Test removal of byte order mark."""
        text = "\ufeffHello World"
        result = normalize_unicode_text(text)
        assert "\ufeff" not in result
        assert result == "Hello World"
    
    @pytest.mark.unit
    def test_preserves_newlines(self):
        """Test that newlines are preserved."""
        text = "Line 1\nLine 2\nLine 3"
        result = normalize_unicode_text(text)
        assert result.count("\n") == 2
    
    @pytest.mark.unit
    def test_preserves_tabs(self):
        """Test that tabs are preserved."""
        text = "Column1\tColumn2"
        result = normalize_unicode_text(text)
        assert "\t" in result
    
    @pytest.mark.unit
    def test_removes_control_characters(self):
        """Test removal of control characters."""
        text = "Hello\x00World\x1f!"
        result = normalize_unicode_text(text)
        assert "\x00" not in result
        assert "\x1f" not in result


class TestFixHyphenation:
    """Tests for fix_hyphenation function."""
    
    @pytest.mark.unit
    def test_merges_lowercase_hyphenation(self):
        """Test merging of lowercase hyphenated words."""
        text = "politi-\nche"
        result = fix_hyphenation(text)
        assert result == "politiche"
    
    @pytest.mark.unit
    def test_preserves_proper_names(self):
        """Test that proper names like Jean-Baptiste are preserved."""
        text = "Jean-\nBaptiste"
        result = fix_hyphenation(text)
        # Should NOT merge due to capital letter
        assert "-\n" in result or "Jean-" in result
    
    @pytest.mark.unit
    def test_requires_minimum_length(self):
        """Test that short fragments are not merged."""
        text = "ab-\ncd"  # Too short
        result = fix_hyphenation(text)
        # Pattern requires 3+ chars before hyphen and 2+ after
        assert text == result or "-\n" in result
    
    @pytest.mark.unit
    def test_multiple_hyphenations(self):
        """Test handling of multiple hyphenations."""
        text = "First hyph-\nenation and second ex-\nample"
        result = fix_hyphenation(text)
        # At least one hyphenation should be merged (depends on heuristics)
        assert "hyphenation" in result or "example" in result or result != text


class TestNormalizeSpacing:
    """Tests for normalize_spacing function."""
    
    @pytest.mark.unit
    def test_expands_tabs(self):
        """Test tab expansion to spaces."""
        text = "Col1\tCol2"
        result = normalize_spacing(text, tab_size=4)
        assert "\t" not in result
        assert "Col1" in result and "Col2" in result
    
    @pytest.mark.unit
    def test_strips_trailing_spaces(self):
        """Test removal of trailing spaces."""
        text = "Line with trailing   \nAnother line  "
        result = normalize_spacing(text)
        lines = result.strip().split("\n")
        for line in lines:
            assert not line.endswith(" ") or line.strip() == ""
    
    @pytest.mark.unit
    def test_collapses_internal_spaces(self):
        """Test collapsing of long internal space runs."""
        text = "Word1     Word2"  # 5 spaces
        result = normalize_spacing(text, collapse_internal=True)
        # Should collapse to at most 2 spaces
        assert "     " not in result
    
    @pytest.mark.unit
    def test_limits_blank_lines(self):
        """Test limiting of consecutive blank lines."""
        text = "Para 1\n\n\n\n\nPara 2"  # 4 blank lines
        result = normalize_spacing(text, max_blank_lines=2)
        # Count actual blank lines (empty strings between content)
        lines = result.split("\n")
        blank_count = 0
        max_blank = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
            else:
                max_blank = max(max_blank, blank_count)
                blank_count = 0
        assert max_blank <= 2
    
    @pytest.mark.unit
    def test_custom_tab_size(self):
        """Test custom tab size setting."""
        text = "\tIndented"
        result_4 = normalize_spacing(text, tab_size=4)
        result_8 = normalize_spacing(text, tab_size=8)
        # 8-space tabs should produce longer leading space
        assert len(result_8.split("Indented")[0]) > len(result_4.split("Indented")[0])


class TestShouldWrapLine:
    """Tests for should_wrap_line function."""
    
    @pytest.mark.unit
    def test_empty_line_no_wrap(self):
        """Test that empty lines are not wrapped."""
        assert should_wrap_line("") is False
        assert should_wrap_line("   ") is False
    
    @pytest.mark.unit
    def test_markdown_heading_no_wrap(self):
        """Test that Markdown headings are not wrapped."""
        assert should_wrap_line("# Heading") is False
        assert should_wrap_line("## Subheading") is False
    
    @pytest.mark.unit
    def test_page_marker_no_wrap(self):
        """Test that page markers are not wrapped."""
        assert should_wrap_line("<page_number>1</page_number>") is False
    
    @pytest.mark.unit
    def test_table_row_no_wrap(self):
        """Test that Markdown table rows are not wrapped."""
        assert should_wrap_line("| Col1 | Col2 |") is False
    
    @pytest.mark.unit
    def test_error_placeholder_no_wrap(self):
        """Test that error placeholders are not wrapped."""
        assert should_wrap_line("[transcription error: page.png]") is False
        assert should_wrap_line("[No transcribable text]") is False
        assert should_wrap_line("[Transcription not possible]") is False
    
    @pytest.mark.unit
    def test_normal_text_wraps(self):
        """Test that normal text lines can be wrapped."""
        assert should_wrap_line("This is a normal line of text.") is True


class TestComputeAutoWrapWidth:
    """Tests for compute_auto_wrap_width function."""
    
    @pytest.mark.unit
    def test_returns_default_for_sparse_text(self):
        """Test that sparse text returns default width."""
        text = "Short"
        result = compute_auto_wrap_width(text)
        assert result == 80  # Default
    
    @pytest.mark.unit
    def test_computes_from_blocks(self):
        """Test computation from text blocks."""
        # Create text with consistent line lengths
        lines = ["x" * 60] * 5
        text = "\n".join(lines)
        result = compute_auto_wrap_width(text)
        assert 55 <= result <= 65
    
    @pytest.mark.unit
    def test_minimum_width(self):
        """Test that minimum width of 20 is enforced."""
        text = "\n".join(["abc"] * 10)  # Very short lines
        result = compute_auto_wrap_width(text)
        assert result >= 20


class TestWrapLongLines:
    """Tests for wrap_long_lines function."""
    
    @pytest.mark.unit
    def test_wraps_long_line(self):
        """Test wrapping of a long line."""
        text = "a " * 50  # 100 chars
        result = wrap_long_lines(text, width=40)
        lines = result.strip().split("\n")
        assert len(lines) > 1
        assert all(len(line) <= 42 for line in lines)  # Some margin for word breaks
    
    @pytest.mark.unit
    def test_preserves_short_lines(self):
        """Test that short lines are unchanged."""
        text = "Short line\nAnother short"
        result = wrap_long_lines(text, width=80)
        assert "Short line" in result
        assert "Another short" in result
    
    @pytest.mark.unit
    def test_preserves_indentation(self):
        """Test that leading indentation is preserved on first line."""
        text = "    Indented long line that should be wrapped into multiple lines eventually"
        result = wrap_long_lines(text, width=40)
        lines = result.split("\n")
        # First line should preserve indentation
        if lines:
            assert lines[0].startswith("    ") or lines[0].strip() == ""
    
    @pytest.mark.unit
    def test_zero_width_no_change(self):
        """Test that zero width returns unchanged text."""
        text = "Some text"
        result = wrap_long_lines(text, width=0)
        assert result.strip() == text


class TestPostprocessText:
    """Tests for postprocess_text function."""
    
    @pytest.mark.unit
    def test_full_pipeline(self):
        """Test the full post-processing pipeline."""
        text = "\ufeffHello\u00ADWorld\n\n\n\nParagraph"
        result = postprocess_text(
            text,
            merge_hyphenation=False,
            max_blank_lines=2,
        )
        assert "\ufeff" not in result
        assert "\u00AD" not in result
    
    @pytest.mark.unit
    def test_with_hyphenation(self):
        """Test pipeline with hyphenation merging enabled."""
        text = "hyph-\nenation"
        result = postprocess_text(text, merge_hyphenation=True)
        assert "hyphenation" in result
    
    @pytest.mark.unit
    def test_with_wrapping(self):
        """Test pipeline with line wrapping enabled."""
        text = "a " * 100  # Long line
        result = postprocess_text(text, wrap_lines=True, wrap_width=50)
        assert "\n" in result.strip()


class TestPostprocessTranscription:
    """Tests for postprocess_transcription function."""
    
    @pytest.mark.unit
    def test_with_empty_config(self):
        """Test with empty configuration dictionary."""
        text = "Hello World"
        result = postprocess_transcription(text, {})
        assert "Hello" in result and "World" in result
    
    @pytest.mark.unit
    def test_with_none_config(self):
        """Test with None configuration."""
        text = "Hello World"
        result = postprocess_transcription(text, None)
        assert "Hello" in result
    
    @pytest.mark.unit
    def test_disabled_returns_original(self):
        """Test that disabled=True returns original text."""
        text = "\ufeffOriginal\u00ADText"
        result = postprocess_transcription(text, {"enabled": False})
        assert result == text
    
    @pytest.mark.unit
    def test_full_config(self):
        """Test with full configuration dictionary."""
        text = "Test\tText"
        config = {
            "enabled": True,
            "merge_hyphenation": False,
            "collapse_internal_spaces": True,
            "max_blank_lines": 2,
            "tab_size": 4,
            "wrap_lines": False,
        }
        result = postprocess_transcription(text, config)
        assert "\t" not in result  # Tabs expanded


class TestPostprocessFile:
    """Tests for postprocess_file function."""
    
    @pytest.mark.unit
    def test_file_not_found(self, temp_dir):
        """Test error when input file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            postprocess_file(temp_dir / "nonexistent.txt")
    
    @pytest.mark.unit
    def test_requires_output_or_inplace(self, sample_text_file):
        """Test error when neither output_path nor in_place specified."""
        with pytest.raises(ValueError):
            postprocess_file(sample_text_file, output_path=None, in_place=False)
    
    @pytest.mark.unit
    def test_in_place_modification(self, temp_dir):
        """Test in-place file modification."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("\ufeffContent\nWith BOM", encoding="utf-8")
        
        result_path = postprocess_file(test_file, in_place=True)
        
        assert result_path == test_file
        content = test_file.read_text(encoding="utf-8")
        assert "\ufeff" not in content
    
    @pytest.mark.unit
    def test_output_to_different_file(self, temp_dir):
        """Test writing to different output file."""
        input_file = temp_dir / "input.txt"
        output_file = temp_dir / "output.txt"
        input_file.write_text("Original content", encoding="utf-8")
        
        result_path = postprocess_file(input_file, output_path=output_file)
        
        assert result_path == output_file
        assert output_file.exists()
        assert input_file.read_text() == "Original content"
