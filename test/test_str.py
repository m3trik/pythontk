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

    def test_format_suffix_strip_trailing_ints_preserves_underscore_separated(self):
        """Verify strip_trailing_ints only strips digits directly appended to the
        name (e.g. CHECKLIST01 -> CHECKLIST) and preserves intentional underscore-
        separated numbering (e.g. CHECKLIST_01 stays CHECKLIST_01).

        Bug: regex r'\\d+$' stripped digits regardless of preceding underscore.
        Fixed: 2026-03-02
        """
        # Digits directly appended — should be stripped
        self.assertEqual(
            StrUtils.format_suffix("CHECKLIST01", "", "", strip_trailing_ints=True),
            "CHECKLIST",
        )
        # Underscore-separated digits — should be preserved
        self.assertEqual(
            StrUtils.format_suffix("CHECKLIST_01", "", "", strip_trailing_ints=True),
            "CHECKLIST_01",
        )
        # Single digit directly appended
        self.assertEqual(
            StrUtils.format_suffix("pCube1", "", "", strip_trailing_ints=True),
            "pCube",
        )
        # Single digit after underscore — preserved
        self.assertEqual(
            StrUtils.format_suffix("pCube_1", "", "", strip_trailing_ints=True),
            "pCube_1",
        )

    # -------------------------------------------------------------------------
    # alpha_sequence Tests
    # -------------------------------------------------------------------------

    def test_alpha_sequence_single_letter(self):
        """Indices 0-25 produce A-Z."""
        self.assertEqual(StrUtils.alpha_sequence(0), "A")
        self.assertEqual(StrUtils.alpha_sequence(1), "B")
        self.assertEqual(StrUtils.alpha_sequence(25), "Z")

    def test_alpha_sequence_double_letter(self):
        """Indices 26+ wrap to AA, AB, ..., AZ, BA."""
        self.assertEqual(StrUtils.alpha_sequence(26), "AA")
        self.assertEqual(StrUtils.alpha_sequence(27), "AB")
        self.assertEqual(StrUtils.alpha_sequence(51), "AZ")
        self.assertEqual(StrUtils.alpha_sequence(52), "BA")

    def test_alpha_sequence_triple_letter(self):
        """Index 702 wraps to AAA (after ZZ at 701)."""
        self.assertEqual(StrUtils.alpha_sequence(701), "ZZ")
        self.assertEqual(StrUtils.alpha_sequence(702), "AAA")

    def test_alpha_sequence_negative_raises(self):
        """Negative indices raise ValueError."""
        with self.assertRaises(ValueError):
            StrUtils.alpha_sequence(-1)

    # -------------------------------------------------------------------------
    # sequential_suffixes Tests
    # -------------------------------------------------------------------------

    def test_sequential_suffixes_letters_under_threshold(self):
        """Counts at or below the switch threshold return uppercase letters."""
        self.assertEqual(StrUtils.sequential_suffixes(0), [])
        self.assertEqual(StrUtils.sequential_suffixes(3), ["A", "B", "C"])
        self.assertEqual(
            StrUtils.sequential_suffixes(26),
            [chr(ord("A") + i) for i in range(26)],
        )

    def test_sequential_suffixes_numeric_above_threshold(self):
        """Counts above the threshold fall back to zero-padded numerics."""
        out = StrUtils.sequential_suffixes(27)
        self.assertEqual(len(out), 27)
        self.assertEqual(out[0], "01")
        self.assertEqual(out[26], "27")
        # Padding widens to match the count's digit length.
        out_120 = StrUtils.sequential_suffixes(120)
        self.assertEqual(out_120[0], "001")
        self.assertEqual(out_120[-1], "120")

    def test_sequential_suffixes_lowercase(self):
        """``lowercase=True`` returns lowercase letters in the letter scheme."""
        self.assertEqual(StrUtils.sequential_suffixes(3, lowercase=True), ["a", "b", "c"])
        # Lowercase is irrelevant once we're in numeric mode.
        out = StrUtils.sequential_suffixes(40, lowercase=True)
        self.assertTrue(all(s.isdigit() for s in out))

    def test_sequential_suffixes_custom_switch_at(self):
        """``switch_at`` lets callers force the numeric branch earlier."""
        self.assertEqual(StrUtils.sequential_suffixes(5, switch_at=3)[:3], ["01", "02", "03"])

    # -------------------------------------------------------------------------
    # resolve_name_collisions Tests
    # -------------------------------------------------------------------------

    def test_resolve_collisions_alpha_basic(self):
        """Three colliding mats get _A, _B, _C; lone wood3 just strips to wood."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1", "mat2", "wood3"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
        )
        self.assertEqual(
            result,
            {"mat": "mat_A", "mat1": "mat_B", "mat2": "mat_C", "wood3": "wood"},
        )

    def test_resolve_collisions_single_member_strips_to_base(self):
        """A non-colliding name still strips to base regardless of suffix scheme."""
        result = StrUtils.resolve_name_collisions(
            ["mat3"], strip_trailing_ints=True, collision_suffix="alpha"
        )
        self.assertEqual(result, {"mat3": "mat"})

    def test_resolve_collisions_no_change_omitted(self):
        """A name already at its target base does not appear in the result."""
        result = StrUtils.resolve_name_collisions(
            ["mat"], strip_trailing_ints=True, collision_suffix="alpha"
        )
        self.assertEqual(result, {})

    def test_resolve_collisions_none_keeps_originals(self):
        """collision_suffix=None leaves multi-member groups unchanged."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1"], strip_trailing_ints=True, collision_suffix=None
        )
        self.assertEqual(result, {})

    def test_resolve_collisions_none_still_strips_singletons(self):
        """Even with collision_suffix=None, single-member groups strip to base."""
        result = StrUtils.resolve_name_collisions(
            ["wood3"], strip_trailing_ints=True, collision_suffix=None
        )
        self.assertEqual(result, {"wood3": "wood"})

    def test_resolve_collisions_numeric(self):
        """Numeric scheme zero-pads to width = max(2, len(str(count)))."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1", "mat2"],
            strip_trailing_ints=True,
            collision_suffix="numeric",
        )
        self.assertEqual(
            result, {"mat": "mat_01", "mat1": "mat_02", "mat2": "mat_03"}
        )

    def test_resolve_collisions_numeric_pads_for_large_groups(self):
        """100+ members -> 3-digit padding."""
        names = [f"x{i}" if i else "x" for i in range(100)]
        result = StrUtils.resolve_name_collisions(
            names, strip_trailing_ints=True, collision_suffix="numeric"
        )
        self.assertEqual(result["x"], "x_001")
        self.assertEqual(result["x99"], "x_100")

    def test_resolve_collisions_callable_scheme(self):
        """Custom callable suffix(i, count) is honored."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix=lambda i, count: f"v{i}",
        )
        self.assertEqual(result, {"mat": "mat_v0", "mat1": "mat_v1"})

    def test_resolve_collisions_preserves_input_order(self):
        """Within a group, suffixes are assigned in input order."""
        result = StrUtils.resolve_name_collisions(
            ["mat2", "mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
        )
        self.assertEqual(
            result, {"mat2": "mat_A", "mat": "mat_B", "mat1": "mat_C"}
        )

    def test_resolve_collisions_empty_base_skipped(self):
        """Names that strip to empty are omitted from grouping."""
        result = StrUtils.resolve_name_collisions(
            ["123", "mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
        )
        self.assertNotIn("123", result)
        self.assertEqual(result, {"mat": "mat_A", "mat1": "mat_B"})

    def test_resolve_collisions_custom_separator(self):
        """suffix_separator is used between base and suffix."""
        result = StrUtils.resolve_name_collisions(
            ["mat", "mat1"],
            strip_trailing_ints=True,
            collision_suffix="alpha",
            suffix_separator="-",
        )
        self.assertEqual(result, {"mat": "mat-A", "mat1": "mat-B"})

    def test_resolve_collisions_alpha_27_members(self):
        """Group of 27 wraps from Z to AA."""
        names = [f"x{i}" if i else "x" for i in range(27)]
        result = StrUtils.resolve_name_collisions(
            names, strip_trailing_ints=True, collision_suffix="alpha"
        )
        self.assertEqual(result["x"], "x_A")
        self.assertEqual(result["x25"], "x_Z")
        self.assertEqual(result["x26"], "x_AA")

    # -------------------------------------------------------------------------
    # replace_placeholders Tests
    # -------------------------------------------------------------------------

    def test_replace_placeholders_basic(self):
        self.assertEqual(
            StrUtils.replace_placeholders("{a}_{b}", a="x", b="y"), "x_y"
        )

    def test_replace_placeholders_format_spec(self):
        self.assertEqual(
            StrUtils.replace_placeholders("v{n:03d}", n=5), "v005"
        )

    def test_replace_placeholders_missing_preserves_placeholder(self):
        self.assertEqual(
            StrUtils.replace_placeholders("{a}_{b}", a="x"), "x_{b}"
        )

    def test_replace_placeholders_missing_preserves_format_spec(self):
        # The bug fixed in SafeFormatter.format_field: unresolved {n:03d}
        # used to collapse to {n}, losing padding for a second pass.
        self.assertEqual(
            StrUtils.replace_placeholders("{stem}_v{n:03d}", stem="shot"),
            "shot_v{n:03d}",
        )
        self.assertEqual(
            StrUtils.replace_placeholders("{user}_{stem}_v{n:03d}", user="maya"),
            "maya_{stem}_v{n:03d}",
        )

    def test_replace_placeholders_two_stage_substitution(self):
        # Stage 1 leaves {n:03d} intact; stage 2 applies the spec.
        stage1 = StrUtils.replace_placeholders("{user}_{stem}_v{n:03d}", user="m")
        self.assertEqual(stage1.format(stem="shot", n=7), "m_shot_v007")

    def test_replace_placeholders_other_format_specs_preserved(self):
        self.assertEqual(
            StrUtils.replace_placeholders("{key:>10}"), "{key:>10}"
        )
        self.assertEqual(
            StrUtils.replace_placeholders("{key:.4f}"), "{key:.4f}"
        )


if __name__ == "__main__":
    unittest.main(exit=False)
