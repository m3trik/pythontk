#!/usr/bin/python
# coding=utf-8
"""
DEPRECATED: This file has been replaced by modular test files.

Please use the individual test modules instead:
    - test_core.py    : CoreUtils tests
    - test_str.py     : StrUtils tests
    - test_iter.py    : IterUtils tests
    - test_file.py    : FileUtils tests
    - test_img.py     : ImgUtils tests
    - test_math.py    : MathUtils tests

To run all tests:
    python run_all_tests.py -v

This file is kept for reference only.
"""
"""
Unit tests for pythontk package.

Run with:
    python -m unittest ptk_test
    python -m pytest ptk_test.py -v
"""
import os
import re
import unittest
from pathlib import Path

from pythontk import (
    CoreUtils,
    FileUtils,
    ImgUtils,
    IterUtils,
    MathUtils,
    StrUtils,
)


# =============================================================================
# Test Utilities & Base Classes
# =============================================================================


class TestPaths:
    """Centralized test path management."""

    BASE_DIR = Path(__file__).parent
    TEST_FILES_DIR = BASE_DIR / "test_files"
    IMGTK_TEST_DIR = TEST_FILES_DIR / "imgtk_test"

    @classmethod
    def get(cls, *parts: str) -> str:
        """Get absolute path to a test file."""
        return str(cls.TEST_FILES_DIR.joinpath(*parts))

    @classmethod
    def get_imgtk(cls, filename: str) -> str:
        """Get path to an image test file."""
        return str(cls.IMGTK_TEST_DIR / filename)


class BaseTestCase(unittest.TestCase):
    """Base test case with common utilities and assertions."""

    # Class-level reference to utility classes (no inheritance mixing)
    core = CoreUtils
    file = FileUtils
    img = ImgUtils
    iter = IterUtils
    math = MathUtils
    str = StrUtils

    @staticmethod
    def replace_mem_address(obj: object) -> str:
        """Normalize memory addresses in string representations for comparison.

        Parameters:
            obj: Object to convert and normalize.

        Returns:
            String with memory addresses replaced by '0x00000000000'.

        Example:
            >>> replace_mem_address("<Widget at 0x1ebe2677e80>")
            "<Widget at 0x00000000000>"
        """
        return re.sub(r"0x[a-fA-F\d]+", "0x00000000000", str(obj))

    def assertImageMode(self, image, expected_mode: str, msg: str = None):
        """Assert that a PIL Image has the expected mode."""
        self.assertEqual(image.mode, expected_mode, msg)

    def assertImageSize(self, image, expected_size: tuple, msg: str = None):
        """Assert that a PIL Image has the expected size."""
        self.assertEqual(image.size, expected_size, msg)

    def assertPathExists(self, path: str, msg: str = None):
        """Assert that a file or directory exists."""
        self.assertTrue(os.path.exists(path), msg or f"Path does not exist: {path}")


# =============================================================================
# Core Utils Tests
# =============================================================================


class CoreTest(BaseTestCase):
    """CoreUtils test class."""

    def test_imports(self):
        """Test that package imports work correctly."""
        import types
        import pythontk as ptk
        from pythontk import IterUtils
        from pythontk import make_iterable

        self.assertIsInstance(ptk, types.ModuleType)
        self.assertIsInstance(IterUtils, type)
        self.assertIsInstance(make_iterable, types.FunctionType)

    def test_cached_property(self):
        """Test the `cached_property` decorator caches results after first access."""

        class MyClass:
            def __init__(self):
                self._counter = 0

            @CoreUtils.cached_property
            def counter(self):
                """Property that increments counter on each access."""
                self._counter += 1
                return self._counter

        my_instance = MyClass()

        # Property not yet computed
        self.assertEqual(my_instance._counter, 0)

        # First access computes and caches
        self.assertEqual(my_instance.counter, 1)
        self.assertEqual(my_instance._counter, 1)

        # Subsequent accesses return cached value
        self.assertEqual(my_instance.counter, 1)
        self.assertEqual(my_instance._counter, 1)
        self.assertEqual(my_instance.counter, 1)
        self.assertEqual(my_instance._counter, 1)

    # -------------------------------------------------------------------------
    # Listify decorator tests
    # -------------------------------------------------------------------------

    def test_listify_standalone_function_with_threading(self):
        """Test listify with threading on a standalone function."""

        @CoreUtils.listify(threading=True)
        def to_str(n):
            return str(n)

        self.assertEqual(to_str([0, 1]), ["0", "1"])

    def test_listify_function_with_arg_name(self):
        """Test listify with explicit arg_name parameter."""

        @CoreUtils.listify(arg_name="n")
        def to_str_arg_name(n):
            return str(n)

        self.assertEqual(to_str_arg_name([0, 1]), ["0", "1"])

    def test_listify_function_with_arg_name_and_threading(self):
        """Test listify with both arg_name and threading."""

        @CoreUtils.listify(arg_name="n", threading=True)
        def to_str_arg_name_threaded(n):
            return str(n)

        self.assertEqual(to_str_arg_name_threaded([0, 1]), ["0", "1"])

    def test_listify_method_within_class(self):
        """Test listify on an instance method."""

        class TestClass:
            @CoreUtils.listify
            def to_str(self, n, x=None):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str([0, 1]), ["0", "1"])

    def test_listify_static_method_within_class(self):
        """Test listify on a static method."""

        class TestClass:
            @staticmethod
            @CoreUtils.listify
            def to_str_staticmethod(n):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str_staticmethod([0, 1]), ["0", "1"])

    def test_listify_class_method_within_class(self):
        """Test listify on a class method."""

        class TestClass:
            @classmethod
            @CoreUtils.listify
            def to_str_classmethod(cls, n):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str_classmethod([0, 1]), ["0", "1"])

    def test_listify_method_within_class_with_threading(self):
        """Test listify with threading on an instance method."""

        class TestClass:
            @CoreUtils.listify(threading=True)
            def to_str_threading(self, n):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str_threading([0, 1]), ["0", "1"])

    def test_listify_method_within_class_with_arg_name_and_threading(self):
        """Test listify with arg_name and threading on an instance method."""

        class TestClass:
            @CoreUtils.listify(arg_name="n", threading=True)
            def to_str_arg_name(self, n):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str_arg_name([0, 1]), ["0", "1"])

    def test_listify_method_within_class_with_none(self):
        """Test listify handles None input correctly."""

        class TestClass:
            @CoreUtils.listify
            def to_str(self, n, x=None):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str(None), "None")

    def test_listify_method_with_overlapping_args_and_kwargs(self):
        """Test listify with various arg/kwarg combinations."""

        class TestClass:
            @CoreUtils.listify(arg_name="n")
            def to_str(self, n, x=None):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str([0, 1], x=2), ["0", "1"])
        self.assertEqual(test_obj.to_str(n=[0, 1], x=2), ["0", "1"])
        self.assertEqual(test_obj.to_str([0, 1], x=2, n=[0, 1]), ["0", "1"])

    def test_listify_method_with_keyword_arg_conflict(self):
        """Test listify with multiple decorated methods."""

        class TestClass:
            @CoreUtils.listify
            def to_str(self, n, x=None):
                return str(n)

            @CoreUtils.listify(arg_name="n")
            def to_str_arg_name(self, n, x=None):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str([0, 1], x=2), ["0", "1"])
        self.assertEqual(test_obj.to_str(n=[0, 1], x=2), ["0", "1"])
        self.assertEqual(test_obj.to_str_arg_name([0, 1], x=2), ["0", "1"])

    def test_listify_method_within_class_with_valid_none(self):
        """Test listify handles None values in list correctly."""

        class TestClass:
            @CoreUtils.listify(arg_name="n")
            def to_str(self, n=None):
                return str(n) if n is not None else "None"

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str(None), "None")
        self.assertEqual(test_obj.to_str([None, 1]), ["None", "1"])

    # -------------------------------------------------------------------------
    # CoreUtils method tests
    # -------------------------------------------------------------------------

    def test_format_return(self):
        """Test format_return normalizes return values."""
        self.assertEqual(CoreUtils.format_return([""]), "")
        self.assertEqual(CoreUtils.format_return([""], [""]), [""])
        self.assertEqual(CoreUtils.format_return(["", ""]), ["", ""])
        self.assertEqual(CoreUtils.format_return([], ""), None)

    def test_set_attributes(self):
        """Test set_attributes sets object attributes."""
        obj = type("TestObj", (), {})()
        result = CoreUtils.set_attributes(obj, attr="value")
        self.assertIsNone(result)
        self.assertEqual(obj.attr, "value")

    def test_get_attributes(self):
        """Test get_attributes retrieves object attributes."""
        obj = type("TestObj", (), {})()
        obj._subtest = None  # Set as instance attribute
        self.assertEqual(CoreUtils.get_attributes(obj, "_subtest"), {"_subtest": None})

    def test_cycle(self):
        """Test cycle iterates through list cyclically."""
        self.assertEqual(CoreUtils.cycle([0, 1], "ID"), 0)
        self.assertEqual(CoreUtils.cycle([0, 1], "ID"), 1)
        self.assertEqual(CoreUtils.cycle([0, 1], "ID"), 0)

    def test_are_similar(self):
        """Test are_similar compares values within tolerance."""
        self.assertTrue(CoreUtils.are_similar(1, 10, 9))
        self.assertFalse(CoreUtils.are_similar(1, 10, 8))

    def test_randomize(self):
        """Test randomize returns subset of input list."""
        result = CoreUtils.randomize(list(range(10)), 1.0)
        for item in result:
            self.assertIn(item, range(10))

        result = CoreUtils.randomize(list(range(10)), 0.5)
        for item in result:
            self.assertIn(item, range(10))


# =============================================================================
# String Utils Tests
# =============================================================================


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


# =============================================================================
# Iterator Utils Tests
# =============================================================================


class IterTest(BaseTestCase):
    """Iterator utilities test class."""

    def test_make_iterable(self):
        """Test make_iterable wraps non-iterables appropriately."""

        class ExampleClass:
            pass

        class ExampleClassWithAttr:
            __apimfn__ = True

        example_instance = ExampleClass()
        example_instance_with_attr = ExampleClassWithAttr()

        # Custom objects become tuples
        self.assertEqual(IterUtils.make_iterable(example_instance), (example_instance,))
        self.assertEqual(
            IterUtils.make_iterable(example_instance_with_attr),
            (example_instance_with_attr,),
        )

        # Scalars become tuples
        self.assertEqual(IterUtils.make_iterable("foo"), ("foo",))
        self.assertEqual(IterUtils.make_iterable(1), (1,))
        self.assertEqual(IterUtils.make_iterable(""), ("",))

        # Collections stay as-is
        self.assertEqual(IterUtils.make_iterable(["foo", "bar"]), ["foo", "bar"])
        self.assertEqual(IterUtils.make_iterable(("foo", "bar")), ("foo", "bar"))
        self.assertEqual(IterUtils.make_iterable({"foo": "bar"}), {"foo": "bar"})
        self.assertEqual(IterUtils.make_iterable(range(3)), range(3))
        self.assertEqual(IterUtils.make_iterable({1, 2, 3}), {1, 2, 3})

        # Iterators are converted to lists
        self.assertEqual(
            IterUtils.make_iterable(map(str, range(3))),
            list(map(str, range(3))),
        )
        self.assertEqual(
            IterUtils.make_iterable(filter(lambda x: x % 2 == 0, range(3))),
            list(filter(lambda x: x % 2 == 0, range(3))),
        )
        self.assertEqual(
            IterUtils.make_iterable(zip(["a", "b", "c"], range(3))),
            list(zip(["a", "b", "c"], range(3))),
        )

    def test_nested_depth(self):
        """Test nested_depth calculates nesting level correctly."""
        self.assertEqual(IterUtils.nested_depth([[1, 2], [3, 4]]), 1)
        self.assertEqual(IterUtils.nested_depth([1, 2, 3, 4]), 0)

    def test_flatten(self):
        """Test flatten unnests nested lists."""
        self.assertEqual(list(IterUtils.flatten([[1, 2], [3, 4]])), [1, 2, 3, 4])

    def test_collapse_integer_sequence(self):
        """Test collapse_integer_sequence creates range notation."""
        lst = [19, 22, 23, 24, 25, 26]

        self.assertEqual(IterUtils.collapse_integer_sequence(lst), "19, 22-6")
        self.assertEqual(IterUtils.collapse_integer_sequence(lst, 1), "19, ...")
        self.assertEqual(
            IterUtils.collapse_integer_sequence(lst, None, False, False),
            ["19", "22..26"],
        )

    def test_bit_array_to_list(self):
        """Test bit_array_to_list converts bit flags to indices."""
        flags = bytes.fromhex("beef")
        bits = [flags[i // 8] & 1 << i % 8 != 0 for i in range(len(flags) * 8)]

        self.assertEqual(
            IterUtils.bit_array_to_list(bits),
            [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16],
        )

    def test_indices(self):
        """Test indices finds all occurrences of value."""
        self.assertEqual(tuple(IterUtils.indices([0, 1, 2, 2, 3], 2)), (2, 3))
        self.assertEqual(tuple(IterUtils.indices([0, 1, 2, 2, 3], 4)), ())

    def test_rindex(self):
        """Test rindex finds last occurrence of value."""
        self.assertEqual(IterUtils.rindex([0, 1, 2, 2, 3], 2), 3)
        self.assertEqual(IterUtils.rindex([0, 1, 2, 2, 3], 4), -1)

    def test_remove_duplicates(self):
        """Test remove_duplicates removes duplicate values."""
        self.assertEqual(IterUtils.remove_duplicates([0, 1, 2, 3, 2]), [0, 1, 2, 3])
        self.assertEqual(
            IterUtils.remove_duplicates([0, 1, 2, 3, 2], False), [0, 1, 3, 2]
        )

    def test_filter_list(self):
        """Test filter_list includes/excludes items by pattern."""
        self.assertEqual(IterUtils.filter_list([0, 1, 2, 3, 2], [1, 2, 3], 2), [1, 3])
        self.assertEqual(
            IterUtils.filter_list(
                [0, 1, "file.txt", "file.jpg"], ["*file*", 0], "*.txt"
            ),
            [0, "file.jpg"],
        )

        # Test with map_func
        self.assertEqual(
            IterUtils.filter_list(
                ["apple", "banana", "cherry"], "*a*", "*n*", map_func=lambda x: x[::-1]
            ),
            ["apple"],
        )

        # Test with map_func and check_unmapped=True
        self.assertEqual(
            IterUtils.filter_list(
                ["apple", "banana", "cherry"],
                "*e*",
                "*n*",
                map_func=lambda x: x[::-1],
                check_unmapped=True,
            ),
            ["apple", "cherry"],
        )

        # Test with check_unmapped=True
        self.assertEqual(
            IterUtils.filter_list([1, 2, 3, 4, 5], [2, 3], 4, check_unmapped=True),
            [2, 3],
        )

        # Test with object inputs and check_unmapped=True
        class MyObject:
            def __init__(self, name):
                self.name = name

        obj1 = MyObject("object1")
        obj2 = MyObject("object2")
        obj3 = MyObject("object3")
        self.assertEqual(
            IterUtils.filter_list(
                [obj1, obj2, obj3],
                obj2,
                obj3,
                map_func=lambda x: x.name,
                check_unmapped=True,
            ),
            [obj2],
        )

        # Test with nested tuples and removal of empty tuples
        self.assertEqual(
            IterUtils.filter_list(
                [
                    ("bevel", "path/to/bevel.py"),
                    ("other_file", "path/to/other_file.py"),
                ],
                inc=["bevel"],
                exc=["path/to/bevel.py"],
            ),
            [("bevel",)],
        )

    def test_filter_dict(self):
        """Test filter_dict filters dictionary by keys/values."""
        dct = {1: "1", "two": 2, 3: "three"}

        self.assertEqual(
            IterUtils.filter_dict(dct, exc="*t*", values=True),
            {1: "1", "two": 2},
        )
        self.assertEqual(
            IterUtils.filter_dict(dct, exc="t*", keys=True),
            {1: "1", 3: "three"},
        )
        self.assertEqual(
            IterUtils.filter_dict(dct, exc=1, keys=True),
            {"two": 2, 3: "three"},
        )

    def test_split_list(self):
        """Test split_list divides lists in various ways."""
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]

        self.assertEqual(IterUtils.split_list(lA, "2parts"), [[1, 2, 3, 5], [7, 8, 9]])
        self.assertEqual(
            IterUtils.split_list(lB, "2parts"), [[1, "2", 3, 5], ["7", 8, 9]]
        )
        self.assertEqual(
            IterUtils.split_list(lA, "2parts+"), [[1, 2, 3], [5, 7, 8], [9]]
        )
        self.assertEqual(
            IterUtils.split_list(lB, "2parts+"), [[1, "2", 3], [5, "7", 8], [9]]
        )
        self.assertEqual(
            IterUtils.split_list(lA, "2chunks"),
            [[1, 2], [3, 5], [7, 8], [9]],
        )
        self.assertEqual(
            IterUtils.split_list(lB, "2chunks"),
            [[1, "2"], [3, 5], ["7", 8], [9]],
        )
        self.assertEqual(
            IterUtils.split_list(lA, "contiguous"),
            [[1, 2, 3], [5], [7, 8, 9]],
        )
        self.assertEqual(
            IterUtils.split_list(lB, "contiguous"),
            [[1, "2", 3], [5], ["7", 8, 9]],
        )
        self.assertEqual(IterUtils.split_list(lA, "range"), [[1, 3], [5], [7, 9]])
        self.assertEqual(IterUtils.split_list(lB, "range"), [[1, 3], [5], ["7", 9]])


# =============================================================================
# File Utils Tests
# =============================================================================


class FileTest(BaseTestCase):
    """File utilities test class."""

    @classmethod
    def setUpClass(cls):
        """Set up test paths used across file tests."""
        cls.test_base_path = Path(__file__).parent
        cls.test_files_path = cls.test_base_path / "test_files"
        cls.file1_path = cls.test_files_path / "file1.txt"
        cls.file2_path = cls.test_files_path / "file2.txt"

    def test_format_path(self):
        """Test format_path normalizes and parses paths."""
        # Test basic path normalization
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3"), "X:/n/dir1/dir3")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3", "path"), "X:/n/dir1/dir3"
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "path"),
            "X:/n/dir1/dir3/.vscode",
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "path"),
            "X:/n/dir1/dir3/.vscode",
        )

        # Test UNC path
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "path"),
            r"\\192.168.1.240/nas/lost+found",
        )

        # Test environment variable expansion
        self.assertEqual(
            FileUtils.format_path(r"%programfiles%", "path"), "C:/Program Files"
        )

        # Test directory extraction
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "dir"), "dir3")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "dir"), ".vscode"
        )
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "dir"),
            ".vscode",
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "dir"),
            "lost+found",
        )
        self.assertEqual(
            FileUtils.format_path(r"%programfiles%", "dir"), "Program Files"
        )

        # Test file extraction
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "file"), "")
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "file"), "")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "file"),
            "tasks.json",
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "file"),
            "file.ext",
        )
        self.assertEqual(FileUtils.format_path(r"%programfiles%", "file"), "")

        # Test name extraction
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "name"), "")
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "name"), "")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "name"), "tasks"
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "name"),
            "file",
        )
        self.assertEqual(FileUtils.format_path(r"%programfiles%", "name"), "")

        # Test extension extraction
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3", "ext"), "")
        self.assertEqual(FileUtils.format_path(r"X:\n/dir1/dir3/.vscode", "ext"), "")
        self.assertEqual(
            FileUtils.format_path(r"X:\n/dir1/dir3/.vscode/tasks.json", "ext"), "json"
        )
        self.assertEqual(
            FileUtils.format_path(r"\\192.168.1.240\nas/lost+found/file.ext", "ext"),
            "ext",
        )
        self.assertEqual(FileUtils.format_path(r"%programfiles%", "ext"), "")

        # Test edge cases
        self.assertEqual(FileUtils.format_path(r"programfiles", "name"), "programfiles")
        self.assertEqual(FileUtils.format_path(r"programfiles", "path"), "programfiles")
        self.assertEqual(
            FileUtils.format_path(r"programfiles/", "path"), "programfiles"
        )

        # Test list input
        self.assertEqual(
            FileUtils.format_path(
                [r"X:\n/dir1/dir3", r"X:\n/dir1/dir3/.vscode"], "dir"
            ),
            ["dir3", ".vscode"],
        )

    def test_is_valid(self):
        """Test is_valid checks file/directory existence."""
        self.assertTrue(FileUtils.is_valid(str(self.file1_path), "file"))
        self.assertTrue(FileUtils.is_valid(str(self.test_files_path), "dir"))

    def test_write_to_file(self):
        """Test write_to_file writes content correctly."""
        result = FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')
        self.assertIsNone(result)

    def test_get_file_contents(self):
        """Test get_file_contents reads file content."""
        # Ensure file has expected content
        FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')

        content = FileUtils.get_file_contents(str(self.file1_path), as_list=True)
        self.assertEqual(content, ['__version__ = "0.9.0"'])

    def test_create_directory(self):
        """Test create_dir creates directories."""
        sub_dir = str(self.test_files_path / "sub-directory")
        result = FileUtils.create_dir(sub_dir)
        self.assertIsNone(result)
        self.assertTrue(os.path.isdir(sub_dir))

    def test_get_file_info(self):
        """Test get_file_info extracts file metadata."""
        files = [str(self.file1_path), str(self.file2_path)]

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "filename", "filepath"]),
            [
                ("file1.txt", "file1", str(self.file1_path)),
                ("file2.txt", "file2", str(self.file2_path)),
            ],
        )

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "filetype"]),
            [("file1.txt", ".txt"), ("file2.txt", ".txt")],
        )

        self.assertEqual(
            FileUtils.get_file_info(files, ["filename", "filetype"]),
            [("file1", ".txt"), ("file2", ".txt")],
        )

        self.assertEqual(
            FileUtils.get_file_info(files, ["file", "size"]),
            [
                ("file1.txt", os.path.getsize(str(self.file1_path))),
                ("file2.txt", os.path.getsize(str(self.file2_path))),
            ],
        )

    def test_get_directory_contents(self):
        """Test get_dir_contents lists directory contents."""
        path = str(self.test_files_path)
        base_path = str(self.test_base_path)

        imgtk_test_dirpath = os.path.join(base_path, "test_files\\imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files\\sub-directory")

        with self.subTest("Test returned dirpaths"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "dirpath"),
                [imgtk_test_dirpath, sub_directory_dirpath],
            )

        with self.subTest("Test returned filenames recursively"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "filename", recursive=True),
                [
                    "file1",
                    "file2",
                    "test",
                    "im_Base_color",
                    "im_h",
                    "im_Height",
                    "im_Metallic",
                    "im_Mixed_AO",
                    "im_n",
                    "im_Normal_DirectX",
                    "im_Normal_OpenGL",
                    "im_Roughness",
                ],
            )

        with self.subTest("Test returned file and dir"):
            self.assertEqual(
                sorted(FileUtils.get_dir_contents(path, ["file", "dir"])),
                sorted(
                    [
                        "imgtk_test",
                        "sub-directory",
                        "file1.txt",
                        "file2.txt",
                        "test.json",
                    ]
                ),
            )

        with self.subTest("Test with exc_dirs"):
            self.assertEqual(
                sorted(
                    FileUtils.get_dir_contents(path, ["file", "dir"], exc_dirs=["sub*"])
                ),
                sorted(["imgtk_test", "file1.txt", "file2.txt", "test.json"]),
            )

        with self.subTest("Test with inc_files"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "filename", inc_files="*.txt"),
                ["file1", "file2"],
            )

        with self.subTest("Test returned file with inc_files"):
            self.assertEqual(
                FileUtils.get_dir_contents(path, "file", inc_files="*.txt"),
                ["file1.txt", "file2.txt"],
            )

        with self.subTest("Test returned dirpath and dir"):
            self.assertEqual(
                sorted(FileUtils.get_dir_contents(path, ["dirpath", "dir"])),
                [
                    imgtk_test_dirpath,
                    sub_directory_dirpath,
                    "imgtk_test",
                    "sub-directory",
                ],
            )

        with self.subTest("Test group_by_type functionality"):
            result = FileUtils.get_dir_contents(
                path, ["dirpath", "file"], group_by_type=True
            )
            self.assertIsInstance(result, dict)
            self.assertIn("dirpath", result)
            self.assertIn("file", result)
            self.assertEqual(
                sorted(result["dirpath"]),
                sorted([imgtk_test_dirpath, sub_directory_dirpath]),
            )
            self.assertEqual(
                sorted(result["file"]),
                sorted(["file1.txt", "file2.txt", "test.json"]),
            )

    def test_get_object_path(self):
        """Test get_object_path extracts path from various objects."""
        path = str(self.test_base_path)

        # Test with __file__ variable
        self.assertEqual(FileUtils.get_object_path(__file__), path)
        self.assertEqual(
            FileUtils.get_object_path(__file__, inc_filename=True),
            os.path.abspath(__file__),
        )

        # Test with a module
        import pythontk

        self.assertEqual(
            FileUtils.get_object_path(pythontk), os.path.dirname(pythontk.__file__)
        )

        # Test with a class
        class TestClass:
            pass

        self.assertEqual(FileUtils.get_object_path(TestClass), path)

        # Test with a callable object (function)
        def test_function():
            pass

        self.assertEqual(FileUtils.get_object_path(test_function), path)

        # Test with None
        self.assertEqual(FileUtils.get_object_path(None), "")

    def test_get_file(self):
        """Test get_file opens file handle."""
        file_handle = FileUtils.get_file(str(self.file1_path))
        self.assertIn("TextIOWrapper", str(type(file_handle)))
        file_handle.close()

    def test_get_classes_from_dir(self):
        """Test get_classes_from_path discovers classes in Python files."""
        path = str(self.test_base_path)

        def fp(name):
            return os.path.join(path, name)

        # Note: Class names may change as we refactor - this test may need updating
        result = FileUtils.get_classes_from_path(path, "classname")
        self.assertIn("BaseTestCase", result)
        self.assertIn("CoreTest", result)
        self.assertIn("StrTest", result)

    def test_update_version(self):
        """Test PackageManager version management."""
        from pythontk.core_utils import PackageManager

        # Reset to known version first
        FileUtils.write_to_file(str(self.file1_path), '__version__ = "0.9.0"')

        # Test increment
        result = PackageManager.update_version(str(self.file1_path), "increment")
        self.assertEqual(str(result), "0.9.1")

        # Test decrement
        result = PackageManager.update_version(str(self.file1_path), "decrement")
        self.assertEqual(str(result), "0.9.0")

    def test_json(self):
        """Test JSON file operations."""
        json_path = str(self.test_files_path / "test.json")

        # Set JSON file
        FileUtils.set_json_file(json_path)
        self.assertEqual(FileUtils.get_json_file(), json_path)

        # Set/get JSON value
        FileUtils.set_json("key", "value")
        self.assertEqual(FileUtils.get_json("key"), "value")


# =============================================================================
# Image Utils Tests
# =============================================================================


class ImgTest(BaseTestCase):
    """Image utilities test class."""

    # Class-level test images
    im_h = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
    im_n = ImgUtils.create_image("RGB", (1024, 1024), (127, 127, 255))

    def test_create_image(self):
        """Test create_image creates images with correct properties."""
        img = ImgUtils.create_image("RGB", (1024, 1024), (0, 0, 0))
        self.assertEqual(img, self.im_h)

    def test_resize_image(self):
        """Test resize_image changes image dimensions."""
        resized = ImgUtils.resize_image(self.im_h, 32, 32)
        self.assertEqual(resized.size, (32, 32))

    def test_save_image_file(self):
        """Test save_image writes image to disk."""
        result_h = ImgUtils.save_image(self.im_h, "test_files/imgtk_test/im_h.png")
        result_n = ImgUtils.save_image(self.im_n, "test_files/imgtk_test/im_n.png")
        self.assertIsNone(result_h)
        self.assertIsNone(result_n)

    def test_get_images(self):
        """Test get_images finds images by pattern."""
        images = ImgUtils.get_images("test_files/imgtk_test/", "*Normal*")
        self.assertEqual(
            list(images.keys()),
            [
                "test_files/imgtk_test/im_Normal_DirectX.png",
                "test_files/imgtk_test/im_Normal_OpenGL.png",
            ],
        )

    def test_resolve_map_type(self):
        """Test resolve_map_type identifies texture types from filename."""
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_h.png"),
            "Height",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_h.png", key=False),
            "_H",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_n.png"),
            "Normal",
        )
        self.assertEqual(
            ImgUtils.resolve_map_type("test_files/imgtk_test/im_n.png", key=False),
            "_N",
        )

    def test_filter_images_by_type(self):
        """Test filter_images_by_type filters by texture type."""
        files = FileUtils.get_dir_contents("test_files/imgtk_test")
        self.assertEqual(
            ImgUtils.filter_images_by_type(files, "Height"),
            ["im_h.png", "im_Height.png"],
        )

    def test_sort_images_by_type(self):
        """Test sort_images_by_type groups images by texture type."""
        self.assertEqual(
            ImgUtils.sort_images_by_type(
                [("im_h.png", "<im_h>"), ("im_n.png", "<im_n>")]
            ),
            {
                "Height": [("im_h.png", "<im_h>")],
                "Normal": [("im_n.png", "<im_n>")],
            },
        )
        self.assertEqual(
            ImgUtils.sort_images_by_type({"im_h.png": "<im_h>", "im_n.png": "<im_n>"}),
            {
                "Height": [("im_h.png", "<im_h>")],
                "Normal": [("im_n.png", "<im_n>")],
            },
        )

    def test_contains_map_types(self):
        """Test contains_map_types checks for texture types."""
        self.assertTrue(ImgUtils.contains_map_types([("im_h.png", "<im_h>")], "Height"))
        self.assertTrue(
            ImgUtils.contains_map_types(
                {"im_h.png": "<im_h>", "im_n.png": "<im_n>"}, "Height"
            )
        )
        self.assertTrue(
            ImgUtils.contains_map_types({"Height": [("im_h.png", "<im_h>")]}, "Height")
        )
        self.assertTrue(
            ImgUtils.contains_map_types(
                {"Height": [("im_h.png", "<im_h>")]}, ["Height", "Normal"]
            )
        )

    def test_is_normal_map(self):
        """Test is_normal_map identifies normal maps."""
        self.assertFalse(ImgUtils.is_normal_map("im_h.png"))
        self.assertTrue(ImgUtils.is_normal_map("im_n.png"))

    def test_invert_channels(self):
        """Test invert_channels inverts specified color channels."""
        result = ImgUtils.invert_channels(self.im_n, "g")
        channel = result.getchannel("G")
        self.assertEqual(channel.mode, "L")

    def test_create_dx_from_gl(self):
        """Test create_dx_from_gl converts OpenGL to DirectX normal maps."""
        dx_path = ImgUtils.create_dx_from_gl(
            "test_files/imgtk_test/im_Normal_OpenGL.png"
        )
        expected = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "test_files",
                "imgtk_test",
                "im_Normal_DirectX.png",
            )
        )
        self.assertEqual(os.path.normpath(dx_path), os.path.normpath(expected))

    def test_create_gl_from_dx(self):
        """Test create_gl_from_dx converts DirectX to OpenGL normal maps."""
        gl_path = ImgUtils.create_gl_from_dx(
            "test_files/imgtk_test/im_Normal_DirectX.png"
        )
        expected = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "test_files",
                "imgtk_test",
                "im_Normal_OpenGL.png",
            )
        )
        self.assertEqual(os.path.normpath(gl_path), os.path.normpath(expected))

    def test_create_mask(self):
        """Test create_mask generates image masks."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")

        mask1 = ImgUtils.create_mask("test_files/imgtk_test/im_Base_color.png", bg)
        self.assertEqual(mask1.mode, "L")

        mask2 = ImgUtils.create_mask(
            "test_files/imgtk_test/im_Base_color.png",
            "test_files/imgtk_test/im_Base_color.png",
        )
        self.assertEqual(mask2.mode, "L")

    def test_fill_masked_area(self):
        """Test fill_masked_area fills masked regions with color."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        mask = ImgUtils.create_mask("test_files/imgtk_test/im_Base_color.png", bg)

        result = ImgUtils.fill_masked_area(
            "test_files/imgtk_test/im_Base_color.png", (0, 255, 0), mask
        )
        self.assertEqual(result.mode, "RGB")

    def test_fill(self):
        """Test fill fills image with color."""
        result = ImgUtils.fill(self.im_h, (127, 127, 127))
        self.assertEqual(result.mode, "RGB")

    def test_get_background(self):
        """Test get_background determines background color."""
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_Height.png", "I"), 32767
        )
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_Height.png", "L"), 255
        )
        self.assertEqual(
            ImgUtils.get_background("test_files/imgtk_test/im_n.png", "RGB"),
            (127, 127, 255),
        )

    def test_replace_color(self):
        """Test replace_color substitutes colors in image."""
        bg = ImgUtils.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        result = ImgUtils.replace_color(
            "test_files/imgtk_test/im_Base_color.png", bg, (255, 0, 0)
        )
        self.assertEqual(result.mode, "RGBA")

    def test_set_contrast(self):
        """Test set_contrast adjusts image contrast."""
        result = ImgUtils.set_contrast("test_files/imgtk_test/im_Mixed_AO.png", 255)
        self.assertEqual(result.mode, "L")

    def test_convert_rgb_to_gray(self):
        """Test convert_rgb_to_gray converts to grayscale array."""
        result = ImgUtils.convert_rgb_to_gray(self.im_h)
        self.assertEqual(str(type(result)), "<class 'numpy.ndarray'>")

    def test_convert_rgb_to_hsv(self):
        """Test convert_rgb_to_hsv converts to HSV color space."""
        result = ImgUtils.convert_rgb_to_hsv(self.im_h)
        self.assertEqual(result.mode, "HSV")

    def test_convert_i_to_l(self):
        """Test convert_i_to_l converts I mode to L mode."""
        im_i = ImgUtils.create_image("I", (32, 32))
        result = ImgUtils.convert_i_to_l(im_i)
        self.assertEqual(result.mode, "L")

    def test_are_identical(self):
        """Test are_identical compares images."""
        self.assertFalse(ImgUtils.are_identical(self.im_h, self.im_n))
        self.assertTrue(ImgUtils.are_identical(self.im_h, self.im_h))


# =============================================================================
# Math Utils Tests
# =============================================================================


class MathTest(BaseTestCase):
    """Math utilities test class."""

    def test_get_vector_from_two_points(self):
        """Test get_vector_from_two_points calculates direction vector."""
        self.assertEqual(
            MathUtils.get_vector_from_two_points((1, 2, 3), (1, 1, -1)),
            (0, -1, -4),
        )

    def test_clamp(self):
        """Test clamp restricts values to range."""
        self.assertEqual(
            MathUtils.clamp(range(10), 3, 7),
            [3, 3, 3, 3, 4, 5, 6, 7, 7, 7],
        )

    def test_normalize(self):
        """Test normalize creates unit vectors."""
        self.assertEqual(
            MathUtils.normalize((2, 3, 4)),
            (0.3713906763541037, 0.5570860145311556, 0.7427813527082074),
        )
        self.assertEqual(
            MathUtils.normalize((2, 3)),
            (0.5547001962252291, 0.8320502943378437),
        )
        self.assertEqual(
            MathUtils.normalize((2, 3, 4), 2),
            (0.7427813527082074, 1.1141720290623112, 1.4855627054164149),
        )

    def test_get_magnitude(self):
        """Test get_magnitude calculates vector length."""
        self.assertEqual(MathUtils.get_magnitude((2, 3, 4)), 5.385164807134504)
        self.assertEqual(MathUtils.get_magnitude((2, 3)), 3.605551275463989)

    def test_dot_product(self):
        """Test dot_product calculates scalar product."""
        self.assertEqual(MathUtils.dot_product((1, 2, 3), (1, 1, -1)), 0)
        self.assertEqual(MathUtils.dot_product((1, 2), (1, 1)), 3)
        self.assertEqual(MathUtils.dot_product((1, 2, 3), (1, 1, -1), True), 0)

    def test_cross_product(self):
        """Test cross_product calculates vector product."""
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1)),
            (-5, 4, -1),
        )
        self.assertEqual(
            MathUtils.cross_product((3, 1, 1), (1, 4, 2), (1, 3, 4)),
            (7, 4, 2),
        )
        self.assertEqual(
            MathUtils.cross_product((1, 2, 3), (1, 1, -1), None, 1),
            (-0.7715167498104595, 0.6172133998483676, -0.1543033499620919),
        )

    def test_move_point_relative(self):
        """Test move_point_relative translates points."""
        self.assertEqual(
            MathUtils.move_point_relative((0, 5, 0), (0, 5, 0)),
            (0, 10, 0),
        )
        self.assertEqual(
            MathUtils.move_point_relative((0, 5, 0), 5, (0, 1, 0)),
            (0, 10, 0),
        )

    def test_move_point_relative_along_vector(self):
        """Test move_point_relative_along_vector moves points along vector."""
        self.assertEqual(
            MathUtils.move_point_relative_along_vector(
                (0, 0, 0), (0, 10, 0), (0, 1, 0), 5
            ),
            (0.0, 5.0, 0.0),
        )
        self.assertEqual(
            MathUtils.move_point_relative_along_vector(
                (0, 0, 0), (0, 10, 0), (0, 1, 0), 5, False
            ),
            (0.0, -5.0, 0.0),
        )

    def test_distance_between_points(self):
        """Test distance_between_points calculates Euclidean distance."""
        self.assertEqual(
            MathUtils.distance_between_points((0, 10, 0), (0, 5, 0)),
            5.0,
        )

    def test_get_center_of_two_points(self):
        """Test get_center_of_two_points finds midpoint."""
        self.assertEqual(
            MathUtils.get_center_of_two_points((0, 10, 0), (0, 5, 0)),
            (0.0, 7.5, 0.0),
        )

    def test_get_angle_from_two_vectors(self):
        """Test get_angle_from_two_vectors calculates angle between vectors."""
        self.assertEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1)),
            1.5707963267948966,
        )
        self.assertEqual(
            MathUtils.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1), True),
            90,
        )

    def test_get_angle_from_three_points(self):
        """Test get_angle_from_three_points calculates angle at vertex."""
        self.assertEqual(
            MathUtils.get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3)),
            0.7904487543360762,
        )
        self.assertEqual(
            MathUtils.get_angle_from_three_points(
                (1, 1, 1), (-1, 2, 3), (1, 4, -3), True
            ),
            45.29,
        )

    def test_get_two_sides_of_asa_triangle(self):
        """Test get_two_sides_of_asa_triangle solves ASA triangle."""
        self.assertEqual(
            MathUtils.get_two_sides_of_asa_triangle(60, 60, 100),
            (100.00015320566493, 100.00015320566493),
        )

    def test_xyz_rotation(self):
        """Test xyz_rotation applies rotations."""
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0)),
            (3.589792907376932e-09, 1.9999999964102069, 3.589792907376932e-09),
        )
        self.assertEqual(
            MathUtils.xyz_rotation(2, (0, 1, 0), [], True),
            (0.0, 114.59, 0.0),
        )

    def test_lerp(self):
        """Test lerp performs linear interpolation."""
        self.assertEqual(MathUtils.lerp(0, 10, 0.5), 5.0)
        self.assertEqual(MathUtils.lerp(-10, 10, 0.5), 0.0)
        self.assertEqual(MathUtils.lerp(0, 10, 0), 0)
        self.assertEqual(MathUtils.lerp(0, 10, 1), 10)


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(exit=False)
