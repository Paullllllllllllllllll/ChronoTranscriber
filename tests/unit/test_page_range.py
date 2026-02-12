"""Unit tests for modules/core/page_range.py."""

from __future__ import annotations

import pytest

from modules.core.page_range import PageRange, parse_page_range


class TestParsePageRange:
    """Tests for parse_page_range()."""

    @pytest.mark.unit
    def test_bare_integer_first_n(self):
        pr = parse_page_range("5")
        assert pr.first_n == 5
        assert pr.last_n is None
        assert pr.spans == ()

    @pytest.mark.unit
    def test_explicit_first_n(self):
        pr = parse_page_range("first:10")
        assert pr.first_n == 10

    @pytest.mark.unit
    def test_explicit_first_n_with_spaces(self):
        pr = parse_page_range("first : 3")
        assert pr.first_n == 3

    @pytest.mark.unit
    def test_last_n(self):
        pr = parse_page_range("last:7")
        assert pr.last_n == 7
        assert pr.first_n is None

    @pytest.mark.unit
    def test_last_n_case_insensitive(self):
        pr = parse_page_range("LAST:2")
        assert pr.last_n == 2

    @pytest.mark.unit
    def test_single_page(self):
        pr = parse_page_range("3")
        # bare integer → first 3 pages
        assert pr.first_n == 3

    @pytest.mark.unit
    def test_range_both_sides(self):
        pr = parse_page_range("3-7")
        assert pr.first_n is None
        assert pr.last_n is None
        assert pr.spans == ((2, 6),)  # 0-indexed

    @pytest.mark.unit
    def test_range_open_end(self):
        pr = parse_page_range("3-")
        assert pr.spans == ((2, 2**31),)

    @pytest.mark.unit
    def test_range_open_start(self):
        pr = parse_page_range("-7")
        assert pr.spans == ((0, 6),)

    @pytest.mark.unit
    def test_compound_range(self):
        pr = parse_page_range("1,3,5-8")
        # After merge: (0,0), (2,2), (4,7)
        assert pr.spans == ((0, 0), (2, 2), (4, 7))

    @pytest.mark.unit
    def test_compound_with_overlap(self):
        pr = parse_page_range("1-5,3-8")
        # Merged: (0,7)
        assert pr.spans == ((0, 7),)

    @pytest.mark.unit
    def test_compound_adjacent(self):
        pr = parse_page_range("1-3,4-6")
        # Adjacent spans merge: (0,5)
        assert pr.spans == ((0, 5),)

    @pytest.mark.unit
    def test_raw_preserved(self):
        pr = parse_page_range("first:5")
        assert pr.raw == "first:5"

    @pytest.mark.unit
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_page_range("")

    @pytest.mark.unit
    def test_zero_page_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_page_range("0")

    @pytest.mark.unit
    def test_first_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_page_range("first:0")

    @pytest.mark.unit
    def test_last_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_page_range("last:0")

    @pytest.mark.unit
    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="must not exceed"):
            parse_page_range("7-3")

    @pytest.mark.unit
    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError):
            parse_page_range("abc")

    @pytest.mark.unit
    def test_bare_dash_raises(self):
        with pytest.raises(ValueError, match="both sides are empty"):
            parse_page_range("-")


class TestPageRangeResolve:
    """Tests for PageRange.resolve()."""

    @pytest.mark.unit
    def test_first_n_within_bounds(self):
        pr = parse_page_range("5")
        assert pr.resolve(20) == [0, 1, 2, 3, 4]

    @pytest.mark.unit
    def test_first_n_exceeds_total(self):
        pr = parse_page_range("50")
        assert pr.resolve(10) == list(range(10))

    @pytest.mark.unit
    def test_last_n_within_bounds(self):
        pr = parse_page_range("last:3")
        assert pr.resolve(10) == [7, 8, 9]

    @pytest.mark.unit
    def test_last_n_exceeds_total(self):
        pr = parse_page_range("last:50")
        assert pr.resolve(5) == [0, 1, 2, 3, 4]

    @pytest.mark.unit
    def test_range_within_bounds(self):
        pr = parse_page_range("3-7")
        assert pr.resolve(20) == [2, 3, 4, 5, 6]

    @pytest.mark.unit
    def test_range_clamped(self):
        pr = parse_page_range("3-100")
        result = pr.resolve(10)
        assert result == [2, 3, 4, 5, 6, 7, 8, 9]

    @pytest.mark.unit
    def test_open_end_range(self):
        pr = parse_page_range("8-")
        assert pr.resolve(10) == [7, 8, 9]

    @pytest.mark.unit
    def test_open_start_range(self):
        pr = parse_page_range("-3")
        assert pr.resolve(10) == [0, 1, 2]

    @pytest.mark.unit
    def test_compound_resolve(self):
        pr = parse_page_range("1,3,5-8")
        assert pr.resolve(20) == [0, 2, 4, 5, 6, 7]

    @pytest.mark.unit
    def test_zero_total(self):
        pr = parse_page_range("5")
        assert pr.resolve(0) == []

    @pytest.mark.unit
    def test_negative_total(self):
        pr = parse_page_range("5")
        assert pr.resolve(-1) == []

    @pytest.mark.unit
    def test_single_page_range(self):
        """A compound single page like '3' in compound syntax."""
        pr = parse_page_range("3-3")
        assert pr.resolve(10) == [2]


class TestPageRangeDescribe:
    """Tests for PageRange.describe()."""

    @pytest.mark.unit
    def test_first_n_describe(self):
        pr = parse_page_range("first:5")
        assert pr.describe() == "first 5 page(s)"

    @pytest.mark.unit
    def test_last_n_describe(self):
        pr = parse_page_range("last:3")
        assert pr.describe() == "last 3 page(s)"

    @pytest.mark.unit
    def test_range_describe(self):
        pr = parse_page_range("3-7")
        assert pr.describe() == "pages 3-7"

    @pytest.mark.unit
    def test_compound_describe(self):
        pr = parse_page_range("1,3,5-8")
        assert pr.describe() == "pages 1,3,5-8"

    @pytest.mark.unit
    def test_single_page_describe(self):
        pr = parse_page_range("3-3")
        assert pr.describe() == "pages 3"


class TestPageRangeIsEmptySpec:
    """Tests for PageRange.is_empty_spec()."""

    @pytest.mark.unit
    def test_first_n_not_empty(self):
        pr = parse_page_range("5")
        assert pr.is_empty_spec() is False

    @pytest.mark.unit
    def test_default_page_range_empty(self):
        pr = PageRange()
        assert pr.is_empty_spec() is True
