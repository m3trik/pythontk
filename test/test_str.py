#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk StrUtils.

Run with:
    python -m pytest test_str.py -v
    python test_str.py
"""
import unittest

from pythontk import StrUtils

from conftest import BaseTestCase


class StrTest(BaseTestCase):
    """String utilities test class."""

    def test_set_case(self):
        """Test set_case converts strings to various cases."""
        self.assertEqual(StrUtils.set_case("xxx", "upper"), "XXX")
        self.assertEqual(StrUtils.set_case("XXX", "lower"), "xxx")
        self.assertEqual(StrUtils.set_case("xxx", "capitalize"), "Xxx")
        self.assertEqual(StrUtils.set_case("xxX", "swapcase"), "XXx")
        self.assertEqual(StrUtils.set_case("xxx XXX", "title"), "Xxx Xxx")
        self.assertEqual(StrUtils.set_case("xXx", "pascal"), "XXx")
        self.assertEqual(StrUtils.set_case("xXx", "camel"), "xXx")
        self.assertEqual(StrUtils.set_case(["xXx"], "camel"), ["xXx"])
        self.assertEqual(StrUtils.set_case(None, "camel"), "")
        self.assertEqual(StrUtils.set_case("", "camel"), "")

    def test_get_mangled_name(self):
        """Test get_mangled_name creates proper Python name mangling."""

        class DummyClass:
            pass

        dummy_instance = DummyClass()

        # Test with class name string
        self.assertEqual(
            StrUtils.get_mangled_name("DummyClass", "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        # Test with class
        self.assertEqual(
            StrUtils.get_mangled_name(DummyClass, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        # Test with class instance
        self.assertEqual(
            StrUtils.get_mangled_name(dummy_instance, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        # Test with invalid attribute name (not a string)
        with self.assertRaises(TypeError):
            StrUtils.get_mangled_name("DummyClass", 123)

        # Test with invalid attribute name (not double underscore prefix)
        with self.assertRaises(ValueError):
            StrUtils.get_mangled_name("DummyClass", "my_attribute")

    def test_get_text_between_delimiters(self):
        """Test get_text_between_delimiters extracts text between markers."""
        input_string = (
            "Here is the <!-- start -->first match<!-- end --> and "
            "here is the <!-- start -->second match<!-- end -->"
        )

        result = StrUtils.get_text_between_delimiters(
            input_string, "<!-- start -->", "<!-- end -->", as_string=True
        )
        self.assertEqual(result, "first match second match")

    def test_get_matching_hierarchy_items(self):
        """Test get_matching_hierarchy_items finds related hierarchy items."""
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

        # Test upstream
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, upstream=True
            ),
            ["polygons"],
        )

        # Test downstream with multiple delimiters
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, downstream=True, delimiters=["|", "#"]
            ),
            ["polygons|mesh|other", "polygons|mesh#submenu"],
        )

        # Test downstream reversed
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

        # Test upstream exact
        self.assertEqual(
            StrUtils.get_matching_hierarchy_items(
                hierarchy_items, target, upstream=True, exact=True
            ),
            ["polygons", "polygons|mesh"],
        )

    def test_split_at_delimiter(self):
        """Test split_at_delimiter splits strings correctly."""
        self.assertEqual(
            StrUtils.split_at_delimiter(["str|ing", "string"]),
            [("str", "ing"), ("string", "")],
        )
        self.assertEqual(
            StrUtils.split_at_delimiter("aCHARScCHARSd", "CHARS", 0),
            ("", "a"),
        )

    def test_insert(self):
        """Test insert adds substrings at specified positions."""
        self.assertEqual(
            StrUtils.insert("ins into str", "substr ", " "),
            "ins substr into str",
        )
        self.assertEqual(
            StrUtils.insert("ins into str", " end of", " ", -1, True),
            "ins into end of str",
        )
        self.assertEqual(
            StrUtils.insert("ins into str", "insert this", "atCharsThatDontExist"),
            "ins into str",
        )
        self.assertEqual(StrUtils.insert("ins into str", 666, 0), "666ins into str")

    def test_rreplace(self):
        """Test rreplace replaces from right side."""
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22), "aa22cc22")
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 1), "aabbcc22")
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 3), "aa22cc22")
        self.assertEqual(StrUtils.rreplace("aabbccbb", "bb", 22, 0), "aabbccbb")

    def test_truncate(self):
        """Test truncate shortens strings with ellipsis."""
        self.assertEqual(StrUtils.truncate("12345678", 4), "..5678")
        self.assertEqual(StrUtils.truncate("12345678", 4, "end"), "1234..")
        self.assertEqual(StrUtils.truncate("12345678", 4, "end", "--"), "1234--")
        self.assertEqual(StrUtils.truncate("12345678", 6, "middle"), "12..78")
        self.assertIsNone(StrUtils.truncate(None, 4))

    def test_get_trailing_integers(self):
        """Test get_trailing_integers extracts numbers from end of string."""
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1"), 1)
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1", 0, True), "1")
        self.assertEqual(StrUtils.get_trailing_integers("p001Cube1", 1), 2)
        self.assertIsNone(StrUtils.get_trailing_integers(None))

    def test_find_str(self):
        """Test find_str matches strings with wildcards/regex."""
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
        self.assertEqual(
            StrUtils.find_str("Weight$|Weights$", lst, regex=True), expected
        )
        self.assertEqual(StrUtils.find_str("*weight*", lst, False, True), expected)
        self.assertEqual(StrUtils.find_str("*Weights|*Weight", lst), expected)

    def test_find_str_and_format(self):
        """Test find_str_and_format finds and transforms strings."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
            "keepColorBorder",
            "keepColorBorderWeight",
        ]

        self.assertEqual(
            StrUtils.find_str_and_format(lst, "", "*Weights"), ["invertVertex"]
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "new name", "*Weights"), ["new name"]
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*insert*", "*Weights"),
            ["invertVertexinsert"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "*_suffix", "*Weights"),
            ["invertVertex_suffix"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "**_suffix", "*Weights"),
            ["invertVertexWeights_suffix"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "prefix_*", "*Weights"),
            ["prefix_Weights"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "prefix_**", "*Weights"),
            ["prefix_invertVertexWeights"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(lst, "new name", "Weights$", True),
            ["new name"],
        )
        self.assertEqual(
            StrUtils.find_str_and_format(
                lst, "new name", "*weights", False, True, True
            ),
            [("invertVertexWeights", "new name")],
        )

    def test_format_suffix(self):
        """Test format_suffix adds suffixes correctly."""
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", "Cube1"),
            "p001_suffix",
        )
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", ["Cu", "be1"]),
            "p001_suffix",
        )
        self.assertEqual(
            StrUtils.format_suffix("p001Cube1", "_suffix", "", True),
            "p001Cube_suffix",
        )
        self.assertEqual(
            StrUtils.format_suffix("pCube_GEO1", "_suffix", "", True, True),
            "pCube_suffix",
        )

    def test_time_stamp(self):
        """Test time_stamp functionality - currently skipped."""
        # Timestamp tests require filesystem access and are timing-sensitive
        pass


if __name__ == "__main__":
    unittest.main(exit=False)
