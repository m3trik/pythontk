#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk StrUtils.

Comprehensive edge case coverage for:
- set_case
- get_mangled_name
- get_text_between_delimiters
- get_matching_hierarchy_items
- split_at_delimiter
- insert / rreplace
- truncate
- get_trailing_integers
- find_str / find_str_and_format
- format_suffix

Run with:
    python -m pytest test_str.py -v
    python test_str.py
"""
import unittest

from pythontk import StrUtils

from conftest import BaseTestCase


class StrTest(BaseTestCase):
    """String utilities test class with comprehensive edge case coverage."""

    # -------------------------------------------------------------------------
    # set_case Tests
    # -------------------------------------------------------------------------

    def test_set_case_basic(self):
        """Test set_case converts strings to various cases."""
        self.assertEqual(StrUtils.set_case("xxx", "upper"), "XXX")
        self.assertEqual(StrUtils.set_case("XXX", "lower"), "xxx")
        self.assertEqual(StrUtils.set_case("xxx", "capitalize"), "Xxx")
        self.assertEqual(StrUtils.set_case("xxX", "swapcase"), "XXx")
        self.assertEqual(StrUtils.set_case("xxx XXX", "title"), "Xxx Xxx")
        self.assertEqual(StrUtils.set_case("xXx", "pascal"), "XXx")
        self.assertEqual(StrUtils.set_case("xXx", "camel"), "xXx")

    def test_set_case_list_input(self):
        """Test set_case with list input."""
        self.assertEqual(StrUtils.set_case(["xXx"], "camel"), ["xXx"])
        self.assertEqual(StrUtils.set_case(["abc", "def"], "upper"), ["ABC", "DEF"])

    def test_set_case_none_and_empty(self):
        """Test set_case with None and empty string."""
        self.assertEqual(StrUtils.set_case(None, "camel"), "")
        self.assertEqual(StrUtils.set_case("", "camel"), "")
        self.assertEqual(StrUtils.set_case("", "upper"), "")

    def test_set_case_unicode(self):
        """Test set_case with unicode characters."""
        self.assertEqual(StrUtils.set_case("ñoño", "upper"), "ÑOÑO")
        self.assertEqual(StrUtils.set_case("CAFÉ", "lower"), "café")
        self.assertEqual(StrUtils.set_case("über", "capitalize"), "Über")

    def test_set_case_single_char(self):
        """Test set_case with single character."""
        self.assertEqual(StrUtils.set_case("a", "upper"), "A")
        self.assertEqual(StrUtils.set_case("Z", "lower"), "z")

    def test_set_case_numbers_and_symbols(self):
        """Test set_case with numbers and symbols."""
        self.assertEqual(StrUtils.set_case("123abc", "upper"), "123ABC")
        self.assertEqual(StrUtils.set_case("!@#ABC", "lower"), "!@#abc")

    def test_set_case_whitespace(self):
        """Test set_case with whitespace strings."""
        self.assertEqual(StrUtils.set_case("   ", "upper"), "   ")
        self.assertEqual(StrUtils.set_case("\t\n", "lower"), "\t\n")

    # -------------------------------------------------------------------------
    # get_mangled_name Tests
    # -------------------------------------------------------------------------

    def test_get_mangled_name_with_class_string(self):
        """Test get_mangled_name with class name as string."""
        self.assertEqual(
            StrUtils.get_mangled_name("DummyClass", "__my_attribute"),
            "_DummyClass__my_attribute",
        )

    def test_get_mangled_name_with_class(self):
        """Test get_mangled_name with class object."""

        class DummyClass:
            pass

        self.assertEqual(
            StrUtils.get_mangled_name(DummyClass, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

    def test_get_mangled_name_with_instance(self):
        """Test get_mangled_name with class instance."""

        class DummyClass:
            pass

        dummy_instance = DummyClass()
        self.assertEqual(
            StrUtils.get_mangled_name(dummy_instance, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

    def test_get_mangled_name_invalid_attr_type(self):
        """Test get_mangled_name raises TypeError for non-string attribute."""
        with self.assertRaises(TypeError):
            StrUtils.get_mangled_name("DummyClass", 123)

    def test_get_mangled_name_invalid_attr_prefix(self):
        """Test get_mangled_name raises ValueError for non-dunder attribute."""
        with self.assertRaises(ValueError):
            StrUtils.get_mangled_name("DummyClass", "my_attribute")

    def test_get_mangled_name_single_underscore(self):
        """Test get_mangled_name with single underscore prefix."""
        with self.assertRaises(ValueError):
            StrUtils.get_mangled_name("MyClass", "_single")

    # -------------------------------------------------------------------------
    # get_text_between_delimiters Tests
    # -------------------------------------------------------------------------

    def test_get_text_between_delimiters_basic(self):
        """Test get_text_between_delimiters extracts text between markers."""
        input_string = (
            "Here is the <!-- start -->first match<!-- end --> and "
            "here is the <!-- start -->second match<!-- end -->"
        )
        result = StrUtils.get_text_between_delimiters(
            input_string, "<!-- start -->", "<!-- end -->", as_string=True
        )
        self.assertEqual(result, "first match second match")

    def test_get_text_between_delimiters_no_match(self):
        """Test get_text_between_delimiters with no matches."""
        result = StrUtils.get_text_between_delimiters(
            "no delimiters here", "[start]", "[end]", as_string=True
        )
        self.assertEqual(result, "")

    def test_get_text_between_delimiters_empty_string(self):
        """Test get_text_between_delimiters with empty input."""
        result = StrUtils.get_text_between_delimiters(
            "", "[start]", "[end]", as_string=True
        )
        self.assertEqual(result, "")

    def test_get_text_between_delimiters_nested(self):
        """Test get_text_between_delimiters with adjacent delimiters."""
        input_string = "(a)(b)(c)"
        result = list(StrUtils.get_text_between_delimiters(input_string, "(", ")"))
        self.assertEqual(result, ["a", "b", "c"])

    def test_get_text_between_delimiters_empty_content(self):
        """Test get_text_between_delimiters with empty content between delimiters."""
        result = list(
            StrUtils.get_text_between_delimiters("[start][end]", "[start]", "[end]")
        )
        self.assertEqual(result, [""])

    # -------------------------------------------------------------------------
    # get_matching_hierarchy_items Tests
    # -------------------------------------------------------------------------

    def test_get_matching_hierarchy_items_upstream(self):
        """Test get_matching_hierarchy_items finds upstream items."""
        hierarchy_items = [
            "polygons|mesh#submenu",
            "polygons|submenu",
            "polygons",
            "polygons|mesh",
            "polygons|other",
            "polygons|mesh|other",
            "other",
        ]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, upstream=True
            ),
            ["polygons"],
        )

    def test_get_matching_hierarchy_items_downstream(self):
        """Test get_matching_hierarchy_items finds downstream items."""
        hierarchy_items = [
            "polygons|mesh#submenu",
            "polygons|submenu",
            "polygons",
            "polygons|mesh",
            "polygons|other",
            "polygons|mesh|other",
            "other",
        ]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, downstream=True, delimiters=["|", "#"]
            ),
            ["polygons|mesh|other", "polygons|mesh#submenu"],
        )

    def test_get_matching_hierarchy_items_reversed(self):
        """Test get_matching_hierarchy_items with reverse option."""
        hierarchy_items = [
            "polygons|mesh#submenu",
            "polygons|mesh|other",
        ]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items,
                target,
                downstream=True,
                delimiters=["|", "#"],
                reverse=True,
            ),
            ["polygons|mesh#submenu", "polygons|mesh|other"],
        )

    def test_get_matching_hierarchy_items_exact(self):
        """Test get_matching_hierarchy_items with exact option."""
        hierarchy_items = ["polygons", "polygons|mesh"]
        target = "polygons|mesh"
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, upstream=True, exact=True
            ),
            ["polygons", "polygons|mesh"],
        )

    def test_get_matching_hierarchy_items_empty_list(self):
        """Test get_matching_hierarchy_items with empty list."""
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items([], "target", upstream=True), []
        )

    # -------------------------------------------------------------------------
    # split_delimited_string Tests
    # -------------------------------------------------------------------------

    def test_split_delimited_string_basic(self):
        """Test split_delimited_string splits strings correctly."""
        # Test list output (default)
        self.assertEqual(
            StrUtils.split_delimited_string("str|ing"),
            ["str", "ing"],
        )
        # Test tuple output (occurrence specified)
        self.assertEqual(
            StrUtils.split_delimited_string("str|ing", occurrence=-1),
            ("str", "ing"),
        )
        # Test list input (vectorized)
        self.assertEqual(
            StrUtils.split_delimited_string(["str|ing", "string"], occurrence=-1),
            [("str", "ing"), ("string", "")],
        )

    def test_split_delimited_string_with_occurrence(self):
        """Test split_delimited_string with specific occurrence."""
        self.assertEqual(
            StrUtils.split_delimited_string("aCHARScCHARSd", "CHARS", occurrence=0),
            ("", "a"),
        )

    def test_split_delimited_string_empty_string(self):
        """Test split_delimited_string with empty string."""
        # List mode
        self.assertEqual(StrUtils.split_delimited_string(""), [])
        # Tuple mode
        self.assertEqual(StrUtils.split_delimited_string("", occurrence=-1), ("", ""))

    def test_split_delimited_string_no_delimiter(self):
        """Test split_delimited_string when delimiter not found."""
        # List mode
        self.assertEqual(StrUtils.split_delimited_string("hello", "|"), ["hello"])
        # Tuple mode
        self.assertEqual(
            StrUtils.split_delimited_string("hello", "|", occurrence=-1), ("hello", "")
        )

    def test_split_delimited_string_delimiter_at_start(self):
        """Test split_delimited_string with delimiter at start."""
        self.assertEqual(
            StrUtils.split_delimited_string("|hello", occurrence=-1), ("", "hello")
        )

    def test_split_delimited_string_delimiter_at_end(self):
        """Test split_delimited_string with delimiter at end."""
        self.assertEqual(
            StrUtils.split_delimited_string("hello|", occurrence=-1), ("hello", "")
        )

    def test_split_delimited_string_multiple_delimiters(self):
        """Test split_delimited_string with multiple occurrences."""
        # Default splits all
        self.assertEqual(StrUtils.split_delimited_string("a|b|c"), ["a", "b", "c"])
        # Split at last
        self.assertEqual(
            StrUtils.split_delimited_string("a|b|c", occurrence=-1), ("a|b", "c")
        )

    # -------------------------------------------------------------------------
    # insert Tests
    # -------------------------------------------------------------------------

    def test_insert_basic(self):
        """Test insert adds substrings at specified positions."""
        self.assertEqual(
            StrUtils.insert("ins into str", "substr ", " "),
            "ins substr into str",
        )

    def test_insert_from_end(self):
        """Test insert from end of string."""
        self.assertEqual(
            StrUtils.insert("ins into str", " end of", " ", -1, True),
            "ins into end of str",
        )

    def test_insert_no_delimiter(self):
        """Test insert when delimiter not found."""
        self.assertEqual(
            StrUtils.insert("ins into str", "insert this", "atCharsThatDontExist"),
            "ins into str",
        )

    def test_insert_at_index(self):
        """Test insert at numeric index."""
        self.assertEqual(StrUtils.insert("ins into str", 666, 0), "666ins into str")

    def test_insert_empty_string(self):
        """Test insert into empty string."""
        self.assertEqual(StrUtils.insert("", "text", 0), "text")

    def test_insert_empty_substring(self):
        """Test insert empty substring."""
        self.assertEqual(StrUtils.insert("hello", "", " "), "hello")

    # -------------------------------------------------------------------------
    # rreplace Tests
    # -------------------------------------------------------------------------

    def test_rreplace_all_occurrences(self):
        """Test rreplace replaces all from right side."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22), "aa22cc22")

    def test_rreplace_limited(self):
        """Test rreplace with count limit."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 1), "aabbcc22")
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 3), "aa22cc22")

    def test_rreplace_zero_count(self):
        """Test rreplace with zero count."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 0), "aabbccbb")

    def test_rreplace_not_found(self):
        """Test rreplace when pattern not found."""
        self.assertEqual(StrUtils.rreplace("hello", "xyz", "abc"), "hello")

    def test_rreplace_empty_string(self):
        """Test rreplace on empty string."""
        self.assertEqual(StrUtils.rreplace("", "a", "b"), "")

    # -------------------------------------------------------------------------
    # truncate Tests
    # -------------------------------------------------------------------------

    def test_truncate_start(self):
        """Test truncate from start (default)."""
        self.assertEqual(StrUtils.truncate("12345678", 4), "..5678")

    def test_truncate_end(self):
        """Test truncate from end."""
        self.assertEqual(StrUtils.truncate("12345678", 4, "end"), "1234..")

    def test_truncate_custom_indicator(self):
        """Test truncate with custom indicator."""
        self.assertEqual(StrUtils.truncate("12345678", 4, "end", "--"), "1234--")

    def test_truncate_middle(self):
        """Test truncate from middle."""
        self.assertEqual(StrUtils.truncate("12345678", 6, "middle"), "12..78")

    def test_truncate_none_input(self):
        """Test truncate with None input."""
        self.assertIsNone(StrUtils.truncate(None, 4))

    def test_truncate_no_truncation_needed(self):
        """Test truncate when string is shorter than limit."""
        self.assertEqual(StrUtils.truncate("hi", 10), "hi")

    def test_truncate_empty_string(self):
        """Test truncate with empty string."""
        self.assertEqual(StrUtils.truncate("", 10), "")

    def test_truncate_exact_length(self):
        """Test truncate when string equals limit."""
        self.assertEqual(StrUtils.truncate("1234", 4), "1234")

    def test_truncate_unicode(self):
        """Test truncate with unicode characters."""
        result = StrUtils.truncate("αβγδεζηθ", 4)
        self.assertEqual(len(result), 6)  # 4 chars + 2 for ..

    def test_truncate_very_long_string(self):
        """Test truncate with very long string."""
        long_str = "a" * 1000
        result = StrUtils.truncate(long_str, 10)
        self.assertEqual(len(result), 12)  # 10 chars + 2 for ..

    # -------------------------------------------------------------------------
    # get_trailing_integers Tests
    # -------------------------------------------------------------------------

    def test_get_trailing_integers_basic(self):
        """Test get_trailing_integers extracts numbers from end of string."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1"), 1)

    def test_get_trailing_integers_as_string(self):
        """Test get_trailing_integers returning string."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1", 0, True), "1")

    def test_get_trailing_integers_increment(self):
        """Test get_trailing_integers with increment."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1", 1), 2)

    def test_get_trailing_integers_none_input(self):
        """Test get_trailing_integers with None input."""
        self.assertIsNone(StrUtils.get_trailing_integers(None))

    def test_get_trailing_integers_no_numbers(self):
        """Test get_trailing_integers with no trailing numbers."""
        result = StrUtils.get_trailing_integers("Cube")
        self.assertIsNone(result)

    def test_get_trailing_integers_multi_digit(self):
        """Test get_trailing_integers with multi-digit number."""
        self.assertEqual(StrUtils.get_trailing_integers("object123"), 123)

    def test_get_trailing_integers_zero(self):
        """Test get_trailing_integers with trailing zero."""
        self.assertEqual(StrUtils.get_trailing_integers("item0"), 0)

    def test_get_trailing_integers_leading_zeros(self):
        """Test get_trailing_integers with leading zeros - zeros are NOT preserved."""
        # Note: The implementation uses int() internally, so leading zeros are lost
        self.assertEqual(StrUtils.get_trailing_integers("item007", 0, True), "7")

    def test_get_trailing_integers_empty_string(self):
        """Test get_trailing_integers with empty string returns empty string."""
        result = StrUtils.get_trailing_integers("")
        self.assertEqual(result, "")

    # -------------------------------------------------------------------------
    # find_str Tests
    # -------------------------------------------------------------------------

    def test_find_str_wildcard(self):
        """Test find_str with wildcard patterns."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
            "keepColorBorder",
            "keepColorBorderWeight",
        ]
        expected = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorderWeight",
            "keepColorBorderWeight",
        ]
        self.assertEqual(StrUtils.find_str("*Weight*", lst), expected)

    def test_find_str_regex(self):
        """Test find_str with regex patterns."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
        ]
        expected = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorderWeight",
        ]
        self.assertEqual(
            StrUtils.find_str("Weight$|Weights$", lst, regex=True), expected
        )

    def test_find_str_case_insensitive(self):
        """Test find_str with case insensitive matching."""
        lst = ["Weight", "WEIGHT", "weight"]
        result = StrUtils.find_str("*weight*", lst, False, True)
        self.assertEqual(len(result), 3)

    def test_find_str_multiple_patterns(self):
        """Test find_str with multiple patterns."""
        lst = ["invertVertexWeights", "keepBorderWeight"]
        self.assertEqual(
            StrUtils.find_str("*Weights|*Weight", lst),
            ["invertVertexWeights", "keepBorderWeight"],
        )

    def test_find_str_empty_list(self):
        """Test find_str with empty list."""
        self.assertEqual(StrUtils.find_str("*test*", []), [])

    def test_find_str_no_matches(self):
        """Test find_str when nothing matches."""
        self.assertEqual(StrUtils.find_str("xyz", ["abc", "def"]), [])

    def test_find_str_all_match(self):
        """Test find_str when all match."""
        self.assertEqual(StrUtils.find_str("*", ["a", "b", "c"]), ["a", "b", "c"])

    # -------------------------------------------------------------------------
    # find_str_and_format Tests
    # -------------------------------------------------------------------------

    def test_find_str_and_format_remove_pattern(self):
        """Test find_str_and_format to remove matched pattern."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "", "*Weights"), ["invertVertex"]
        )

    def test_find_str_and_format_replace(self):
        """Test find_str_and_format with replacement."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "new name", "*Weights"), ["new name"]
        )

    def test_find_str_and_format_insert(self):
        """Test find_str_and_format with insert pattern."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*insert*", "*Weights"),
            ["invertVertexinsert"],
        )

    def test_find_str_and_format_suffix(self):
        """Test find_str_and_format adding suffix."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*_suffix", "*Weights"),
            ["invertVertex_suffix"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "**_suffix", "*Weights"),
            ["invertVertexWeights_suffix"],
        )

    def test_find_str_and_format_prefix(self):
        """Test find_str_and_format adding prefix."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "prefix_*", "*Weights"),
            ["prefix_Weights"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "prefix_**", "*Weights"),
            ["prefix_invertVertexWeights"],
        )

    def test_find_str_and_format_with_original(self):
        """Test find_str_and_format with return_originals."""
        lst = ["invertVertexWeights"]
        self.assertEqual(
            StrUtils.find_str_and_format(
                lst, "new name", "*weights", False, True, True
            ),
            [("invertVertexWeights", "new name")],
        )

    def test_find_str_and_format_empty_list(self):
        """Test find_str_and_format with empty list."""
        self.assertEqual(StrUtils.find_str_and_format([], "new", "*old*"), [])

    # -------------------------------------------------------------------------
    # format_suffix Tests
    # -------------------------------------------------------------------------

    def test_format_suffix_basic(self):
        """Test format_suffix adds suffixes correctly."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", "Cube1"),
            "p001_suffix",
        )

    def test_format_suffix_list_strip(self):
        """Test format_suffix with list of strings to strip."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", ["Cu", "be1"]),
            "p001_suffix",
        )

    def test_format_suffix_strip_trailing(self):
        """Test format_suffix with strip_trailing option."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", "", True),
            "p001Cube_suffix",
        )

    def test_format_suffix_strip_chars(self):
        """Test format_suffix with strip_chars option."""
        self.assertEqual(
            StrUtils.format_suffix("pCube_GEO1", "_suffix", "", True, True),
            "pCube_suffix",
        )

    def test_format_suffix_empty_suffix(self):
        """Test format_suffix with empty suffix."""
        self.assertEqual(
            StrUtils.format_suffix("test", "", ""),
            "test",
        )

    def test_format_suffix_no_strip(self):
        """Test format_suffix with nothing to strip."""
        self.assertEqual(
            StrUtils.format_suffix("hello", "_end", "xyz"),
            "hello_end",
        )


if __name__ == "__main__":
    unittest.main(exit=False)
