#!/usr/bin/python
# coding=utf-8
import os
import unittest
import inspect
from pythontk import Misc, File, Img, Iter, Math, Str


class Main(unittest.TestCase):
    """Main test class."""

    def perform_test(self, *cases):
        """Execute the test cases."""
        for case in cases:
            if isinstance(case, dict):
                for expression, expected_result in case.items():
                    method_name = str(expression).split("(")[0]
                    self._test_case(expression, method_name, expected_result)
            elif isinstance(case, tuple) and len(case) == 2:
                expression, expected_result = case
                if isinstance(expression, str):
                    method_name = str(expression).split("(")[0]
                else:
                    method_name = expression.__class__.__name__
                self._test_case(expression, method_name, expected_result)

    def _test_case(self, expression, method_name, expected_result):
        try:
            if isinstance(expression, str):
                path = os.path.abspath(inspect.getfile(eval(method_name)))
            else:
                path = os.path.abspath(inspect.getfile(expression.__class__))
        except (TypeError, IOError):
            path = ""

        if isinstance(expression, str):
            result = eval(expression)
        else:
            result = expression

        self.assertEqual(
            result,
            expected_result,
            f"\n\n# Error: {path}\n#\t{method_name}\n#\tExpected {type(expected_result)}: {expected_result}\n#\tReturned {type(result)}: {result}",
        )

    @staticmethod
    def replace_mem_address(obj):
        """Replace memory addresses in a string representation of an object with a fixed format of '0x00000000000'.

        Parameters:
                obj (object): The input object. The function first converts this object to a string using the `str` function.

        Returns:
                (str) The string representation of the object with all memory addresses replaced.

        Example:
                >>> replace_mem_address("<class 'str'> <PySide2.QtWidgets.QWidget(0x1ebe2677e80, name='MayaWindow') at 0x000001EBE6D48500>")
                "<class 'str'> <PySide2.QtWidgets.QWidget(0x00000000000, name='MayaWindow') at 0x00000000000>"
        """
        import re

        return re.sub(r"0x[a-fA-F\d]+", "0x00000000000", str(obj))


class MiscTest(Main, Misc):
    """Misc test class."""

    def test_imports(self):
        """Test imports."""

        import types
        import pythontk as ptk
        from pythontk import Iter
        from pythontk import make_iterable

        self.assertIsInstance(ptk, types.ModuleType)
        self.assertIsInstance(Iter, type)
        self.assertIsInstance(make_iterable, types.FunctionType)

    def test_cached_property(self):
        """Test the `cached_property` decorator."""

        class MyClass:
            def __init__(self):
                self._counter = 0

            @Misc.cached_property
            def counter(self):
                """A property that increments the counter by one each time it's accessed."""
                self._counter += 1
                return self._counter

        my_instance = MyClass()

        # At this point, the property should not be computed yet, so the counter should still be zero.
        self.assertEqual(my_instance._counter, 0)

        # The first time we access the property, it should compute the result and increment the counter.
        self.assertEqual(my_instance.counter, 1)
        self.assertEqual(my_instance._counter, 1)

        # Subsequent accesses should not recompute the property, so the counter should stay at one.
        self.assertEqual(my_instance.counter, 1)
        self.assertEqual(my_instance._counter, 1)
        self.assertEqual(my_instance.counter, 1)
        self.assertEqual(my_instance._counter, 1)

    def test_listify(self):
        # 1. Standalone function with threading
        @Misc.listify(threading=True)
        def to_str(n):
            return str(n)

        # 2. Function with arg_name specified
        @Misc.listify(arg_name="n")
        def to_str_arg_name(n):
            return str(n)

        # 3. Function with arg_name specified and threading
        @Misc.listify(arg_name="n", threading=True)
        def to_str_arg_name_threaded(n):
            return str(n)

        # 4. Method within a class with threading
        class TestClass:
            @Misc.listify
            def to_str(self, n, x=None):
                return str(n)

            @staticmethod
            @Misc.listify
            def to_str_staticmethod(n):
                return str(n)

            @classmethod
            @Misc.listify
            def to_str_classmethod(cls, n):
                return str(n)

            @Misc.listify(threading=True)
            def to_str_threading(self, n):
                return str(n)

            @Misc.listify(arg_name="n", threading=True)
            def to_str_arg_name(self, n):
                return str(n)

        self.assertEqual(to_str([0, 1]), ["0", "1"])
        self.assertEqual(to_str_arg_name([0, 1]), ["0", "1"])
        self.assertEqual(to_str_arg_name_threaded([0, 1]), ["0", "1"])
        test_obj = TestClass()
        self.assertEqual(test_obj.to_str([0, 1]), ["0", "1"])
        self.assertEqual(test_obj.to_str_staticmethod([0, 1]), ["0", "1"])
        self.assertEqual(test_obj.to_str_classmethod([0, 1]), ["0", "1"])
        self.assertEqual(test_obj.to_str_threading([0, 1]), ["0", "1"])
        self.assertEqual(test_obj.to_str_arg_name([0, 1]), ["0", "1"])
        self.assertEqual(test_obj.to_str(None), "None")

    def test_formatReturn(self):
        """Test format_return method."""
        self.assertEqual(self.format_return([""]), "")
        self.assertEqual(self.format_return([""], [""]), [""])
        self.assertEqual(self.format_return(["", ""]), ["", ""])

    def test_setAttributes(self):
        """Test set_attributes method."""
        self.perform_test(
            {
                "self.set_attributes(self, attr='value')": None,
            }
        )

    def test_getAttributes(self):
        """Test get_attributes method."""
        self.perform_test(
            {
                "self.get_attributes(self, '_subtest')": {"_subtest": None},
            }
        )

    def test_cycle(self):
        """Test cycle method."""
        self.perform_test(
            {
                "self.cycle([0,1], 'ID')": 0,
                "self.cycle([0,1], 'ID')": 1,
                "self.cycle([0,1], 'ID')": 0,
            }
        )

    def test_areSimilar(self):
        """Test are_similar method."""
        self.perform_test(
            {
                "self.are_similar(1, 10, 9)": True,
                "self.are_similar(1, 10, 8)": False,
            }
        )

    def test_randomize(self):
        """Test randomize method."""
        print("\nrandomize: skipped")
        self.perform_test(
            {
                # "self.randomize(range(10), 1.0)": [],
                # "self.randomize(range(10), 0.5)": [],
            }
        )


class StrTest(Main, Str):
    """String test class."""

    def test_setCase(self):
        """Test set_case method."""
        self.assertEqual(self.set_case("xxx", "upper"), "XXX")
        self.assertEqual(self.set_case("XXX", "lower"), "xxx")
        self.assertEqual(self.set_case("xxx", "capitalize"), "Xxx")
        self.assertEqual(self.set_case("xxX", "swapcase"), "XXx")
        self.assertEqual(self.set_case("xxx XXX", "title"), "Xxx Xxx")
        self.assertEqual(self.set_case("xXx", "pascal"), "XXx")
        self.assertEqual(self.set_case("xXx", "camel"), "xXx")
        self.assertEqual(self.set_case(["xXx"], "camel"), ["xXx"])
        self.assertEqual(self.set_case(None, "camel"), "")
        self.assertEqual(self.set_case("", "camel"), "")

    def test_getMangledName(self):
        """ """

        class DummyClass:
            ...

        dummy_instance = DummyClass()

        self.assertEqual(  # Test with class name
            self.get_mangled_name("DummyClass", "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        self.assertEqual(  # Test with class
            self.get_mangled_name(DummyClass, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        self.assertEqual(  # Test with class instance
            self.get_mangled_name(dummy_instance, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        # Test with invalid attribute name (not a string)
        with self.assertRaises(TypeError):
            self.get_mangled_name("DummyClass", 123)

        # Test with invalid attribute name (does not start with double underscore)
        with self.assertRaises(ValueError):
            self.get_mangled_name("DummyClass", "my_attribute")

    def test_getTextBetweenDelimiters(self):
        """Test get_text_between_delimiters method."""
        input_string = "Here is the <!-- start -->first match<!-- end --> and here is the <!-- start -->second match<!-- end -->"

        self.perform_test(
            {
                f"self.get_text_between_delimiters('{input_string}', '<!-- start -->', '<!-- end -->', as_string=True)": "first match second match",
            }
        )

    def test_getMatchingHierarchyItems(self):
        """Test get_matching_hierarchy_items method."""
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

        self.perform_test(
            (
                self.get_matching_hierarchy_items(
                    hierarchy_items, target, upstream=True
                ),
                ["polygons"],
            ),
            (
                self.get_matching_hierarchy_items(
                    hierarchy_items, target, downstream=True, delimiters=["|", "#"]
                ),
                ["polygons|mesh|other", "polygons|mesh#submenu"],
            ),
            (
                self.get_matching_hierarchy_items(
                    hierarchy_items,
                    target,
                    downstream=True,
                    delimiters=["|", "#"],
                    reverse=True,
                ),
                ["polygons|mesh#submenu", "polygons|mesh|other"],
            ),
            (
                self.get_matching_hierarchy_items(
                    hierarchy_items, target, upstream=True, exact=True
                ),
                [
                    "polygons",
                    "polygons|mesh",
                ],
            ),
        )

    def test_splitAtChars(self):
        """Test split_at_chars method."""
        self.perform_test(
            {
                "self.split_at_chars(['str|ing', 'string'])": [
                    ("str", "ing"),
                    ("string", ""),
                ],
                "self.split_at_chars('aCHARScCHARSd', 'CHARS', 0)": ("", "a"),
            }
        )

    def test_insert(self):
        """Test insert method."""
        self.perform_test(
            {
                "self.insert('ins into str', 'substr ', ' ')": "ins substr into str",
                "self.insert('ins into str', ' end of', ' ', -1, True)": "ins into end of str",
                "self.insert('ins into str', 'insert this', 'atCharsThatDontExist')": "ins into str",
                "self.insert('ins into str', 666, 0)": "666ins into str",
            }
        )

    def test_rreplace(self):
        """Test rreplace method."""
        self.perform_test(
            {
                "self.rreplace('aabbccbb', 'bb', 22)": "aa22cc22",
                "self.rreplace('aabbccbb', 'bb', 22, 1)": "aabbcc22",
                "self.rreplace('aabbccbb', 'bb', 22, 3)": "aa22cc22",
                "self.rreplace('aabbccbb', 'bb', 22, 0)": "aabbccbb",
            }
        )

    def test_truncate(self):
        """Test truncate method."""
        self.perform_test(
            {
                "self.truncate('12345678', 4)": "..5678",
                "self.truncate('12345678', 4, False)": "1234..",
                "self.truncate('12345678', 4, False, '--')": "1234--",
                "self.truncate(None, 4)": None,
            }
        )

    def test_getTrailingIntegers(self):
        """Test get_trailing_integers method."""
        self.perform_test(
            {
                "self.get_trailing_integers('p001Cube1')": 1,
                "self.get_trailing_integers('p001Cube1', 0, True)": "1",
                "self.get_trailing_integers('p001Cube1', 1)": 2,
                "self.get_trailing_integers(None)": None,
            }
        )

    def test_findStr(self):
        """Test find_str method."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
            "keepColorBorder",
            "keepColorBorderWeight",
        ]
        rtn = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorderWeight",
            "keepColorBorderWeight",
        ]

        self.perform_test(
            {
                f"self.find_str('*Weight*', {lst})": rtn,
                f"self.find_str('Weight$|Weights$', {lst}, regex=True)": rtn,
                f"self.find_str('*weight*', {lst}, False, True)": rtn,
                f"self.find_str('*Weights|*Weight', {lst})": rtn,
            }
        )

    def test_findStrAndFormat(self):
        """Test find_str_and_format method."""
        lst = [
            "invertVertexWeights",
            "keepCreaseEdgeWeight",
            "keepBorder",
            "keepBorderWeight",
            "keepColorBorder",
            "keepColorBorderWeight",
        ]

        self.perform_test(
            {
                f"self.find_str_and_format({lst}, '', '*Weights')": ["invertVertex"],
                f"self.find_str_and_format({lst}, 'new name', '*Weights')": [
                    "new name"
                ],
                f"self.find_str_and_format({lst}, '*insert*', '*Weights')": [
                    "invertVertexinsert"
                ],
                f"self.find_str_and_format({lst}, '*_suffix', '*Weights')": [
                    "invertVertex_suffix"
                ],
                f"self.find_str_and_format({lst}, '**_suffix', '*Weights')": [
                    "invertVertexWeights_suffix"
                ],
                f"self.find_str_and_format({lst}, 'prefix_*', '*Weights')": [
                    "prefix_Weights"
                ],
                f"self.find_str_and_format({lst}, 'prefix_**', '*Weights')": [
                    "prefix_invertVertexWeights"
                ],
                f"self.find_str_and_format({lst}, 'new name', 'Weights$', True)": [
                    "new name"
                ],
                f"self.find_str_and_format({lst}, 'new name', '*weights', False, True, True)": [
                    ("invertVertexWeights", "new name")
                ],
            }
        )

    def test_formatSuffix(self):
        """Test format_suffix method."""
        self.perform_test(
            {
                "self.format_suffix('p001Cube1', '_suffix', 'Cube1')": "p00_suffix",
                "self.format_suffix('p001Cube1', '_suffix', ['Cu', 'be1'])": "p00_suffix",
                "self.format_suffix('p001Cube1', '_suffix', '', True)": "p001Cube_suffix",
                "self.format_suffix('pCube_GEO1', '_suffix', '', True, True)": "pCube_suffix",
            }
        )


class IterTest(Main, Iter):
    """ """


def test_make_iterable(self):
    # Test an object that isn't a string, list, tuple, set, dict, range, map, filter, or zip
    class ExampleClass:
        ...

    example_instance = ExampleClass()
    self.assertEqual(Iter.make_iterable(example_instance), (example_instance,))
    self.assertEqual(Iter.make_iterable("x"), ("x",))
    self.assertEqual(Iter.make_iterable(1), (1,))
    self.assertEqual(Iter.make_iterable(""), ("",))
    self.assertEqual(Iter.make_iterable(["x", "y"]), ["x", "y"])
    self.assertEqual(Iter.make_iterable(("x", "y")), ("x", "y"))
    self.assertEqual(Iter.make_iterable({"x": "y"}), {"x": "y"})
    self.assertEqual(Iter.make_iterable(range(3)), range(3))
    self.assertEqual(Iter.make_iterable({1, 2, 3}), {1, 2, 3})
    # Note: Map, filter, and zip objects are evaluated once and can't be used again,
    # so we convert them to lists first
    self.assertEqual(Iter.make_iterable(map(str, range(3))), list(map(str, range(3))))
    self.assertEqual(
        Iter.make_iterable(filter(lambda x: x % 2 == 0, range(3))),
        list(filter(lambda x: x % 2 == 0, range(3))),
    )
    self.assertEqual(
        Iter.make_iterable(zip(["a", "b", "c"], range(3))),
        list(zip(["a", "b", "c"], range(3))),
    )

    def test_nestedDepth(self):
        """ """
        self.perform_test(
            {
                "self.nested_depth([[1, 2], [3, 4]])": 1,
                "self.nested_depth([1, 2, 3, 4])": 0,
            }
        )

    def test_flatten(self):
        """ """
        self.perform_test(
            {
                "list(self.flatten([[1, 2], [3, 4]]))": [1, 2, 3, 4],
            }
        )

    def test_collapseList(self):
        """ """
        lst = [19, 22, 23, 24, 25, 26]

        self.perform_test(
            {
                f"self.collapse_integer_sequence({lst})": "19, 22-6",
                f"self.collapse_integer_sequence({lst}, 1)": "19, ...",
                f"self.collapse_integer_sequence({lst}, None, False, False)": [
                    "19",
                    "22..26",
                ],
            }
        )

    def test_bitArrayToList(self):
        """ """
        flags = bytes.fromhex("beef")
        bits = [flags[i // 8] & 1 << i % 8 != 0 for i in range(len(flags) * 8)]

        self.perform_test(
            {
                f"self.bit_array_to_list({bits})": [
                    2,
                    3,
                    4,
                    5,
                    6,
                    8,
                    9,
                    10,
                    11,
                    12,
                    14,
                    15,
                    16,
                ],
            }
        )

    def test_indices(self):
        """ """
        self.perform_test(
            {
                "tuple(self.indices([0, 1, 2, 2, 3], 2))": (2, 3),
                "tuple(self.indices([0, 1, 2, 2, 3], 4))": (),
            }
        )

    def test_rindex(self):
        """ """
        self.perform_test(
            {
                "self.rindex([0, 1, 2, 2, 3], 2)": 3,
                "self.rindex([0, 1, 2, 2, 3], 4)": -1,
            }
        )

    def test_removeDuplicates(self):
        """ """
        self.perform_test(
            {
                "self.remove_duplicates([0, 1, 2, 3, 2])": [0, 1, 2, 3],
                "self.remove_duplicates([0, 1, 2, 3, 2], False)": [0, 1, 3, 2],
            }
        )

    def test_filterWithMappedValues(self):
        """ """
        original_list = ["1", "2", "3", "4", "5", "6"]

        def keep_even_numbers(lst):
            return [x for x in lst if x % 2 == 0]

        self.perform_test(
            [
                (
                    self.filter_mapped_values(
                        original_list, keep_even_numbers, lambda x: int(x)
                    ),
                    ["2", "4", "6"],
                ),
            ]
        )

    def test_filterDict(self):
        """ """
        dct = {1: "1", "two": 2, 3: "three"}

        self.perform_test(
            {
                f"self.filter_dict({dct}, exc='*t*', values=True)": {1: "1", "two": 2},
                f"self.filter_dict({dct}, exc='t*', keys=True)": {1: "1", 3: "three"},
                f"self.filter_dict({dct}, exc=1, keys=True)": {"two": 2, 3: "three"},
            }
        )

    def test_filterList(self):
        """ """
        self.perform_test(
            {
                "self.filter_list([0, 1, 2, 3, 2], [1, 2, 3], 2)": [1, 3],
                "self.filter_list([0, 1, 'file.txt', 'file.jpg'], ['*file*', 0], '*.txt')": [
                    0,
                    "file.jpg",
                ],
            }
        )

    def test_splitList(self):
        """ """
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]

        self.perform_test(
            {
                f"self.split_list({lA}, '2parts')": [[1, 2, 3, 5], [7, 8, 9]],
                f"self.split_list({lB}, '2parts')": [[1, "2", 3, 5], ["7", 8, 9]],
                f"self.split_list({lA}, '2parts+')": [[1, 2, 3], [5, 7, 8], [9]],
                f"self.split_list({lB}, '2parts+')": [[1, "2", 3], [5, "7", 8], [9]],
                f"self.split_list({lA}, '2chunks')": [[1, 2], [3, 5], [7, 8], [9]],
                f"self.split_list({lB}, '2chunks')": [[1, "2"], [3, 5], ["7", 8], [9]],
                f"self.split_list({lA}, 'contiguous')": [[1, 2, 3], [5], [7, 8, 9]],
                f"self.split_list({lB}, 'contiguous')": [[1, "2", 3], [5], ["7", 8, 9]],
                f"self.split_list({lA}, 'range')": [[1, 3], [5], [7, 9]],
                f"self.split_list({lB}, 'range')": [[1, 3], [5], ["7", 9]],
            }
        )


class FileTest(Main, File):
    """ """

    def test_formatPath(self):
        """ """
        p1 = r"X:\n/dir1/dir3"
        p2 = r"X:\n/dir1/dir3/.vscode"
        p3 = r"X:\n/dir1/dir3/.vscode/tasks.json"
        p4 = r"\\192.168.1.240\nas/lost+found/file.ext"
        p5 = r"%programfiles%"
        p6 = r"programfiles"
        p7 = r"programfiles/"

        self.perform_test(
            {
                f"self.format_path(r'{p1}')": "X:/n/dir1/dir3",
                f"self.format_path(r'{p1}', 'path')": "X:/n/dir1/dir3",
                f"self.format_path(r'{p2}', 'path')": "X:/n/dir1/dir3/.vscode",
                f"self.format_path(r'{p3}', 'path')": "X:/n/dir1/dir3/.vscode",
                f"self.format_path(r'{p4}', 'path')": r"\\192.168.1.240/nas/lost+found",
                f"self.format_path(r'{p5}', 'path')": "C:/Program Files",
                f"self.format_path(r'{p1}', 'dir')": "dir3",
                f"self.format_path(r'{p2}', 'dir')": ".vscode",
                f"self.format_path(r'{p3}', 'dir')": ".vscode",
                f"self.format_path(r'{p4}', 'dir')": "lost+found",
                f"self.format_path(r'{p5}', 'dir')": "Program Files",
                f"self.format_path(r'{p1}', 'file')": "",
                f"self.format_path(r'{p2}', 'file')": "",
                f"self.format_path(r'{p3}', 'file')": "tasks.json",
                f"self.format_path(r'{p4}', 'file')": "file.ext",
                f"self.format_path(r'{p5}', 'file')": "",
                f"self.format_path(r'{p1}', 'name')": "",
                f"self.format_path(r'{p2}', 'name')": "",
                f"self.format_path(r'{p3}', 'name')": "tasks",
                f"self.format_path(r'{p4}', 'name')": "file",
                f"self.format_path(r'{p5}', 'name')": "",
                f"self.format_path(r'{p1}', 'ext')": "",
                f"self.format_path(r'{p2}', 'ext')": "",
                f"self.format_path(r'{p3}', 'ext')": "json",
                f"self.format_path(r'{p4}', 'ext')": "ext",
                f"self.format_path(r'{p5}', 'ext')": "",
                f"self.format_path(r'{p6}', 'filename')": "programfiles",
                f"self.format_path(r'{p6}', 'path')": "programfiles",
                f"self.format_path(r'{p7}', 'path')": "programfiles",
                f"self.format_path({[p1, p2]}, 'dir')": ["dir3", ".vscode"],
            }
        )

    def test_timeStamp(self):
        """ """
        paths = [
            r"%ProgramFiles%",
            r"C:/",
        ]

        print("\ntimestamp: skipped")
        self.perform_test(
            {
                # "self.time_stamp({})".format(paths): [],
                # "self.time_stamp({}, False, '%m-%d-%Y  %H:%M', True)".format(paths): [],
                # "self.time_stamp({}, True)".format(paths): [],
            }
        )

    def test_isValid(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"self.is_valid(r'{file}')": "file",
                f"self.is_valid(r'{path}')": "dir",
            }
        )

    def test_writeToFile(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"self.write_to_file(r'{file}', '__version__ = \"0.9.0\"')": None,
            }
        )

    def test_getFileContents(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"self.get_file_contents(r'{file}', as_list=True)": '__version__ = "0.9.0"',
                f"self.get_file_contents(r'{file}', as_list=True)": [
                    '__version__ = "0.9.0"'
                ],
            }
        )

    def test_createDirectory(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"

        self.perform_test(
            {
                f"self.create_dir(r'{path}'+'/sub-directory')": None,
            }
        )

    def test_getFileInfo(self):
        """ """
        base_path = os.path.dirname(__file__)
        relative_path = "test_files"
        path = os.path.join(base_path, relative_path)

        file1_path = os.path.join(path, "file1.txt")
        file2_path = os.path.join(path, "file2.txt")

        files = [file1_path, file2_path]

        self.assertEqual(
            self.get_file_info(files, "file|filename|filepath"),
            [
                ("file1.txt", "file1", file1_path),
                ("file2.txt", "file2", file2_path),
            ],
        )

        self.assertEqual(
            self.get_file_info(files, "file|filetype"),
            [
                ("file1.txt", ".txt"),
                ("file2.txt", ".txt"),
            ],
        )

        self.assertEqual(
            self.get_file_info(files, "filename|filetype"),
            [
                ("file1", ".txt"),
                ("file2", ".txt"),
            ],
        )

        self.assertEqual(
            self.get_file_info(files, "file|size"),
            [
                ("file1.txt", os.path.getsize(file1_path)),
                ("file2.txt", os.path.getsize(file2_path)),
            ],
        )

    def test_getDirectoryContents(self):
        """ """
        base_path = os.path.dirname(__file__)
        relative_path = "test_files"
        path = os.path.join(base_path, relative_path)

        test_files_dirpath = os.path.join(base_path, "test_files")
        imgtk_test_dirpath = os.path.join(base_path, "test_files\\imgtk_test")
        sub_directory_dirpath = os.path.join(base_path, "test_files\\sub-directory")

        self.assertEqual(
            self.get_dir_contents(path, "dirpaths"),
            [
                imgtk_test_dirpath,
                sub_directory_dirpath,
            ],
        )
        self.assertEqual(
            self.get_dir_contents(path, "filenames", recursive=True),
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
        self.assertEqual(
            self.get_dir_contents(path, "files|dirs"),
            ["imgtk_test", "sub-directory", "file1.txt", "file2.txt", "test.json"],
        )
        self.assertEqual(
            self.get_dir_contents(path, "files|dirs", exc_dirs=["sub*"]),
            ["imgtk_test", "file1.txt", "file2.txt", "test.json"],
        )
        self.assertEqual(
            self.get_dir_contents(path, "filenames", inc_files="*.txt"),
            ["file1", "file2"],
        )
        self.assertEqual(
            self.get_dir_contents(path, "files", inc_files="*.txt"),
            ["file1.txt", "file2.txt"],
        )
        self.assertEqual(
            sorted(self.get_dir_contents(path, "dirpath|dir")),
            [
                imgtk_test_dirpath,
                sub_directory_dirpath,
                "imgtk_test",
                "sub-directory",
            ],
        )

    def test_getFilepath(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__))

        self.perform_test(
            {
                "self.get_filepath(__file__)": path,
                "self.get_filepath(__file__, True)": __file__,
            }
        )

    def test_getFile(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"str(self.get_file(r'{file}'))": r"<_io.TextIOWrapper name='O:\\Cloud\\Code\\_scripts\\pythontk\\test/test_files/file1.txt' mode='a+' encoding='cp1252'>",
            }
        )

    def test_updateVersion(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"str(self.update_version(r'{file}', 'increment'))": r"0.9.1",
                f"str(self.update_version(r'{file}', 'decrement'))": r"0.9.0",
            }
        )

    def test_json(self):
        """ """
        p = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        path = "/".join(p.split("\\")).rstrip("/")
        file = path + "/test.json"

        self.perform_test(
            {
                f"self.set_json_file(r'{file}')": None,
                "self.get_json_file()": file,
                "self.set_json('key', 'value')": None,
                "self.get_json('key')": "value",
            }
        )


class ImgTest(Main, Img):
    """ """

    im_h = Img.create_image("RGB", (1024, 1024), (0, 0, 0))
    im_n = Img.create_image("RGB", (1024, 1024), (127, 127, 255))

    def test_createImage(self):
        """ """
        self.perform_test(
            {
                "self.create_image('RGB', (1024, 1024), (0, 0, 0))": self.im_h,
            }
        )

    def test_resizeImage(self):
        """ """
        self.perform_test(
            {
                "self.resize_image(self.im_h, 32, 32).size": (32, 32),
            }
        )

    def test_saveImageFile(self):
        """ """
        self.perform_test(
            {
                "self.save_image(self.im_h, 'test_files/imgtk_test/im_h.png')": None,
                "self.save_image(self.im_n, 'test_files/imgtk_test/im_n.png')": None,
            }
        )

    def test_getImages(self):
        """ """
        # print (\n'test_getImages:', self.get_images('test_files/imgtk_test/'))
        self.perform_test(
            {
                "list(self.get_images('test_files/imgtk_test/', '*Normal*').keys())": [
                    "test_files/imgtk_test/im_Normal_DirectX.png",
                    "test_files/imgtk_test/im_Normal_OpenGL.png",
                ],
            }
        )

    def test_getImageFiles(self):
        """ """
        print("\ngetImageFiles: skipped")
        self.perform_test(
            {
                # "self.get_image_files('*.png|*.jpg')": '',
            }
        )

    def test_getImageDirectory(self):
        """ """
        print("\ngetImageDirectory: skipped")
        self.perform_test(
            {
                # "self.get_image_dir()": '',
            }
        )

    def test_getImageTypeFromFilename(self):
        """ """
        self.perform_test(
            {
                "self.get_image_type_from_filename('test_files/imgtk_test/im_h.png')": "Height",
                "self.get_image_type_from_filename('test_files/imgtk_test/im_h.png', key=False)": "_H",
                "self.get_image_type_from_filename('test_files/imgtk_test/im_n.png')": "Normal",
                "self.get_image_type_from_filename('test_files/imgtk_test/im_n.png', key=False)": "_N",
            }
        )

    def test_filterImagesByType(self):
        """ """
        self.perform_test(
            {
                "self.filter_images_by_type(File.get_dir_contents('test_files/imgtk_test'), 'Height')": [
                    "im_h.png",
                    "im_Height.png",
                ],
            }
        )

    def test_sortImagesByType(self):
        """ """
        self.perform_test(
            {
                "self.sort_images_by_type([('im_h.png', '<im_h>'), ('im_n.png', '<im_n>')])": {
                    "Height": [("im_h.png", "<im_h>")],
                    "Normal": [("im_n.png", "<im_n>")],
                },
                "self.sort_images_by_type({'im_h.png':'<im_h>', 'im_n.png':'<im_n>'})": {
                    "Height": [("im_h.png", "<im_h>")],
                    "Normal": [("im_n.png", "<im_n>")],
                },
            }
        )

    def test_containsMapTypes(self):
        """ """
        self.perform_test(
            {
                "self.contains_map_types([('im_h.png', '<im_h>')], 'Height')": True,
                "self.contains_map_types({'im_h.png':'<im_h>', 'im_n.png':'<im_n>'}, 'Height')": True,
                "self.contains_map_types({'Height': [('im_h.png', '<im_h>')]}, 'Height')": True,
                "self.contains_map_types({'Height': [('im_h.png', '<im_h>')]}, 'Height|Normal')": True,
                "self.contains_map_types({'Height': [('im_h.png', '<im_h>')]}, ['Height', 'Normal'])": True,
            }
        )

    def test_isNormalMap(self):
        """ """
        self.perform_test(
            {
                "self.is_normal_map('im_h.png')": False,
                "self.is_normal_map('im_n.png')": True,
            }
        )

    def test_invertChannels(self):
        """ """
        self.perform_test(
            {
                "str(self.invert_channels(self.im_n, 'g').getchannel('G')).split('size')[0]": "<PIL.Image.Image image mode=L ",
            }
        )

    def test_createDXFromGL(self):
        """ """
        self.perform_test(
            {
                "self.create_dx_from_gl('test_files/imgtk_test/im_Normal_OpenGL.png')": "test_files/imgtk_test/im_Normal_DirectX.png",
            }
        )

    def test_createGLFromDX(self):
        """ """
        self.perform_test(
            {
                "self.create_gl_from_dx('test_files/imgtk_test/im_Normal_DirectX.png')": "test_files/imgtk_test/im_Normal_OpenGL.png",
            }
        )

    def test_createMask(self):
        """ """
        bg = self.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        # self.create_mask('test_files/imgtk_test/im_Base_color.png', self.bg).show()
        self.perform_test(
            {
                f"str(self.create_mask('test_files/imgtk_test/im_Base_color.png', {bg})).split('size')[0]": "<PIL.Image.Image image mode=L ",
                "str(self.create_mask('test_files/imgtk_test/im_Base_color.png', 'test_files/imgtk_test/im_Base_color.png')).split('size')[0]": "<PIL.Image.Image image mode=L ",
            }
        )

    def test_fillMaskedArea(self):
        """ """
        bg = self.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        self.mask_fillMaskedArea = self.create_mask(
            "test_files/imgtk_test/im_Base_color.png", bg
        )
        # self.fill_masked_area('test_files/imgtk_test/im_Base_color.png', (0, 255, 0), self.mask).show()
        self.perform_test(
            {
                "str(self.fill_masked_area('test_files/imgtk_test/im_Base_color.png', (0, 255, 0), self.mask_fillMaskedArea)).split('size')[0]": "<PIL.Image.Image image mode=RGB ",
            }
        )

    def test_fill(self):
        """ """
        # self.fill(self.im_h, (255, 0, 0)).show()
        self.perform_test(
            {
                "str(self.fill(self.im_h, (127, 127, 127))).split('size')[0]": "<PIL.Image.Image image mode=RGB ",
            }
        )

    def test_getBackground(self):
        """ """
        self.perform_test(
            {
                "self.get_background('test_files/imgtk_test/im_Height.png', 'I')": 32767,
                "self.get_background('test_files/imgtk_test/im_Height.png', 'L')": 255,
                "self.get_background('test_files/imgtk_test/im_n.png', 'RGB')": (
                    127,
                    127,
                    255,
                ),
            }
        )

    def test_replaceColor(self):
        """ """
        bg = self.get_background("test_files/imgtk_test/im_Base_color.png", "RGB")
        # self.replace_color('test_files/imgtk_test/im_Base_color.png', self.bg, (255, 0, 0)).show()
        self.perform_test(
            {
                f"str(self.replace_color('test_files/imgtk_test/im_Base_color.png', {bg}, (255, 0, 0))).split('size')[0]": "<PIL.Image.Image image mode=RGBA ",
            }
        )

    def test_setContrast(self):
        """ """
        # self.set_contrast('test_files/imgtk_test/im_Mixed_AO.png', 255).show()
        self.perform_test(
            {
                "str(self.set_contrast('test_files/imgtk_test/im_Mixed_AO.png', 255)).split('size')[0]": "<PIL.Image.Image image mode=L ",
            }
        )

    def test_convert_rgb_to_gray(self):
        """ """
        # print (\n'test_convert_rgb_to_gray:', self.convert_rgb_to_gray(self.im_h))
        self.perform_test(
            {
                "str(type(self.convert_rgb_to_gray(self.im_h)))": "<class 'numpy.ndarray'>",
            }
        )

    def test_convert_RGB_to_HSV(self):
        """ """
        self.perform_test(
            {
                "str(self.convert_rgb_to_hsv(self.im_h)).split('size')[0]": "<PIL.Image.Image image mode=HSV ",
            }
        )

    def test_convert_I_to_L(self):
        """ """
        self.im_convert_I_to_L = self.create_image("I", (32, 32))
        # im = self.convert_i_to_l(self.im)
        self.perform_test(
            {
                "self.convert_i_to_l(self.im_convert_I_to_L).mode": "L",
            }
        )

    def test_areIdentical(self):
        """ """
        self.perform_test(
            {
                "self.are_identical(self.im_h, self.im_n)": False,
                "self.are_identical(self.im_h, self.im_h)": True,
            }
        )


class MathTest(Main, Math):
    """ """

    def test_getVectorFromTwoPoints(self):
        """ """
        self.perform_test(
            {
                "self.get_vector_from_two_points((1, 2, 3), (1, 1, -1))": (0, -1, -4),
            }
        )

    def test_clamp(self):
        """ """
        self.perform_test(
            {
                "self.clamp(range(10), 3, 7)": [3, 3, 3, 3, 4, 5, 6, 7, 7, 7],
            }
        )

    def test_normalize(self):
        """ """
        self.perform_test(
            {
                "self.normalize((2, 3, 4))": (
                    0.3713906763541037,
                    0.5570860145311556,
                    0.7427813527082074,
                ),
                "self.normalize((2, 3))": (0.5547001962252291, 0.8320502943378437),
                "self.normalize((2, 3, 4), 2)": (
                    0.7427813527082074,
                    1.1141720290623112,
                    1.4855627054164149,
                ),
            }
        )

    def test_getMagnitude(self):
        """ """
        self.perform_test(
            {
                "self.get_magnitude((2, 3, 4))": 5.385164807134504,
                "self.get_magnitude((2, 3))": 3.605551275463989,
            }
        )

    def test_dotProduct(self):
        """ """
        self.perform_test(
            {
                "self.dot_product((1, 2, 3), (1, 1, -1))": 0,
                "self.dot_product((1, 2), (1, 1))": 3,
                "self.dot_product((1, 2, 3), (1, 1, -1), True)": 0,
            }
        )

    def test_crossProduct(self):
        """ """
        self.perform_test(
            {
                "self.cross_product((1, 2, 3), (1, 1, -1))": (-5, 4, -1),
                "self.cross_product((3, 1, 1), (1, 4, 2), (1, 3, 4))": (7, 4, 2),
                "self.cross_product((1, 2, 3), (1, 1, -1), None, 1)": (
                    -0.7715167498104595,
                    0.6172133998483676,
                    -0.1543033499620919,
                ),
            }
        )

    def test_movePointRelative(self):
        """ """
        self.perform_test(
            {
                "self.move_point_relative((0, 5, 0), (0, 5, 0))": (0, 10, 0),
                "self.move_point_relative((0, 5, 0), 5, (0, 1, 0))": (0, 10, 0),
            }
        )

    def test_movePointAlongVectorRelativeToPoint(self):
        """ """
        self.perform_test(
            {
                "self.move_point_relative_along_vector((0, 0, 0), (0, 10, 0), (0, 1, 0), 5)": (
                    0.0,
                    5.0,
                    0.0,
                ),
                "self.move_point_relative_along_vector((0, 0, 0), (0, 10, 0), (0, 1, 0), 5, False)": (
                    0.0,
                    -5.0,
                    0.0,
                ),
            }
        )

    def test_getDistanceBetweenTwoPoints(self):
        """ """
        self.perform_test(
            {
                "self.get_distance((0, 10, 0), (0, 5, 0))": 5.0,
            }
        )

    def test_getCenterPointBetweenTwoPoints(self):
        """ """
        self.perform_test(
            {
                "self.get_center_of_two_points((0, 10, 0), (0, 5, 0))": (
                    0.0,
                    7.5,
                    0.0,
                ),
            }
        )

    def test_getAngleFrom2Vectors(self):
        """ """
        self.perform_test(
            {
                "self.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1))": 1.5707963267948966,
                "self.get_angle_from_two_vectors((1, 2, 3), (1, 1, -1), True)": 90,
            }
        )

    def test_getAngleFrom3Points(self):
        """ """
        self.perform_test(
            {
                "self.get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3))": 0.7904487543360762,
                "self.get_angle_from_three_points((1, 1, 1), (-1, 2, 3), (1, 4, -3), True)": 45.29,
            }
        )

    def test_getTwoSidesOfASATriangle(self):
        """ """
        self.perform_test(
            {
                "self.get_two_sides_of_asa_triangle(60, 60, 100)": (
                    100.00015320566493,
                    100.00015320566493,
                ),
            }
        )

    def test_xyzRotation(self):
        """ """
        self.perform_test(
            {
                "self.xyz_rotation(2, (0, 1, 0))": (
                    3.589792907376932e-09,
                    1.9999999964102069,
                    3.589792907376932e-09,
                ),
                "self.xyz_rotation(2, (0, 1, 0), [], True)": (0.0, 114.59, 0.0),
            }
        )


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(exit=False)
    # print(self.are_similar)

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------

"""
def test_(self):
        '''
        '''
        self.perform_test({
            "<class>.()": ,
        })
"""

# --------------------------------------------------------------------------------------------
# deprecated:
# --------------------------------------------------------------------------------------------
