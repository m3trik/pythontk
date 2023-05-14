#!/usr/bin/python
# coding=utf-8
import os
import unittest
import inspect

from pythontk import Core, File, Img, Iter, Math, Str


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
            f"\n\n# Error: {path}\n#\tCall: {method_name}({', '.join(map(str, function_args)) if 'function_args' in locals() else ''})\n#\tExpected {type(expected_result)}: {expected_result}\n#\tReturned {type(result)}: {result}",
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


class CoreTest(Main, Core):
    """Core test class."""

    def test_imports(self):
        """Test imports."""

        import types
        import pythontk as ptk
        from pythontk import Iter
        from pythontk import makeList

        self.assertIsInstance(ptk, types.ModuleType)
        self.assertIsInstance(Iter, type)
        self.assertIsInstance(makeList, types.FunctionType)

    def test_listify(self):
        """ """

        @Core.listify
        def to_string(n):
            return str(n)

        self.assertEqual(
            to_string(range(10)),
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        )

    def test_formatReturn(self):
        """Test formatReturn method."""
        self.assertEqual(self.formatReturn([""]), "")
        self.assertEqual(self.formatReturn([""], [""]), [""])
        self.assertEqual(self.formatReturn(["", ""]), ["", ""])

    def test_setAttributes(self):
        """Test setAttributes method."""
        self.perform_test(
            {
                "self.setAttributes(self, attr='value')": None,
            }
        )

    def test_getAttributes(self):
        """Test getAttributes method."""
        self.perform_test(
            {
                "self.getAttributes(self, '_subtest')": {"_subtest": None},
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
        """Test areSimilar method."""
        self.perform_test(
            {
                "self.areSimilar(1, 10, 9)": True,
                "self.areSimilar(1, 10, 8)": False,
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
        """Test setCase method."""
        self.perform_test(
            {
                "self.setCase('xxx', 'upper')": "XXX",
                "self.setCase('XXX', 'lower')": "xxx",
                "self.setCase('xxx', 'capitalize')": "Xxx",
                "self.setCase('xxX', 'swapcase')": "XXx",
                "self.setCase('xxx XXX', 'title')": "Xxx Xxx",
                "self.setCase('xXx', 'pascal')": "XXx",
                "self.setCase('xXx', 'camel')": "xXx",
                "self.setCase(['xXx'], 'camel')": ["xXx"],
                "self.setCase(None, 'camel')": "",
                "self.setCase('', 'camel')": "",
            }
        )

    def test_getMangledName(self):
        """ """

        class DummyClass:
            ...

        dummy_instance = DummyClass()

        self.assertEqual(  # Test with class name
            self.getMangledName("DummyClass", "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        self.assertEqual(  # Test with class
            self.getMangledName(DummyClass, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        self.assertEqual(  # Test with class instance
            self.getMangledName(dummy_instance, "__my_attribute"),
            "_DummyClass__my_attribute",
        )

        # Test with invalid attribute name (not a string)
        with self.assertRaises(TypeError):
            self.getMangledName("DummyClass", 123)

        # Test with invalid attribute name (does not start with double underscore)
        with self.assertRaises(ValueError):
            self.getMangledName("DummyClass", "my_attribute")

    def test_getTextBetweenDelimiters(self):
        """Test getTextBetweenDelimiters method."""
        input_string = "Here is the <!-- start -->first match<!-- end --> and here is the <!-- start -->second match<!-- end -->"

        self.perform_test(
            {
                f"self.getTextBetweenDelimiters('{input_string}', '<!-- start -->', '<!-- end -->', as_string=True)": "first match second match",
            }
        )

    def test_getMatchingHierarchyItems(self):
        """Test getMatchingHierarchyItems method."""
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
                self.getMatchingHierarchyItems(hierarchy_items, target, upstream=True),
                ["polygons"],
            ),
            (
                self.getMatchingHierarchyItems(
                    hierarchy_items, target, downstream=True, delimiters=["|", "#"]
                ),
                ["polygons|mesh|other", "polygons|mesh#submenu"],
            ),
            (
                self.getMatchingHierarchyItems(
                    hierarchy_items,
                    target,
                    downstream=True,
                    delimiters=["|", "#"],
                    reverse=True,
                ),
                ["polygons|mesh#submenu", "polygons|mesh|other"],
            ),
            (
                self.getMatchingHierarchyItems(
                    hierarchy_items, target, upstream=True, exact=True
                ),
                [
                    "polygons",
                    "polygons|mesh",
                ],
            ),
        )

    def test_splitAtChars(self):
        """Test splitAtChars method."""
        self.perform_test(
            {
                "self.splitAtChars(['str|ing', 'string'])": [
                    ("str", "ing"),
                    ("string", ""),
                ],
                "self.splitAtChars('aCHARScCHARSd', 'CHARS', 0)": ("", "a"),
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
        """Test getTrailingIntegers method."""
        self.perform_test(
            {
                "self.getTrailingIntegers('p001Cube1')": 1,
                "self.getTrailingIntegers('p001Cube1', 0, True)": "1",
                "self.getTrailingIntegers('p001Cube1', 1)": 2,
                "self.getTrailingIntegers(None)": None,
            }
        )

    def test_findStr(self):
        """Test findStr method."""
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
                f"self.findStr('*Weight*', {lst})": rtn,
                f"self.findStr('Weight$|Weights$', {lst}, regEx=True)": rtn,
                f"self.findStr('*weight*', {lst}, False, True)": rtn,
                f"self.findStr('*Weights|*Weight', {lst})": rtn,
            }
        )

    def test_findStrAndFormat(self):
        """Test findStrAndFormat method."""
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
                f"self.findStrAndFormat({lst}, '', '*Weights')": ["invertVertex"],
                f"self.findStrAndFormat({lst}, 'new name', '*Weights')": ["new name"],
                f"self.findStrAndFormat({lst}, '*insert*', '*Weights')": [
                    "invertVertexinsert"
                ],
                f"self.findStrAndFormat({lst}, '*_suffix', '*Weights')": [
                    "invertVertex_suffix"
                ],
                f"self.findStrAndFormat({lst}, '**_suffix', '*Weights')": [
                    "invertVertexWeights_suffix"
                ],
                f"self.findStrAndFormat({lst}, 'prefix_*', '*Weights')": [
                    "prefix_Weights"
                ],
                f"self.findStrAndFormat({lst}, 'prefix_**', '*Weights')": [
                    "prefix_invertVertexWeights"
                ],
                f"self.findStrAndFormat({lst}, 'new name', 'Weights$', True)": [
                    "new name"
                ],
                f"self.findStrAndFormat({lst}, 'new name', '*weights', False, True, True)": [
                    ("invertVertexWeights", "new name")
                ],
            }
        )

    def test_formatSuffix(self):
        """Test formatSuffix method."""
        self.perform_test(
            {
                "self.formatSuffix('p001Cube1', '_suffix', 'Cube1')": "p00_suffix",
                "self.formatSuffix('p001Cube1', '_suffix', ['Cu', 'be1'])": "p00_suffix",
                "self.formatSuffix('p001Cube1', '_suffix', '', True)": "p001Cube_suffix",
                "self.formatSuffix('pCube_GEO1', '_suffix', '', True, True)": "pCube_suffix",
            }
        )


class IterTest(Main, Iter):
    """ """

    def test_makeList(self):
        """ """
        self.perform_test(
            {
                "self.makeList('x')": ["x"],
                "self.makeList(1)": [1],
                "self.makeList('')": [""],
                "self.makeList({'x':'y'})": ["x"],
            }
        )

    def test_nestedDepth(self):
        """ """
        self.perform_test(
            {
                "self.nestedDepth([[1, 2], [3, 4]])": 1,
                "self.nestedDepth([1, 2, 3, 4])": 0,
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
                f"self.collapseList({lst})": "19, 22-6",
                f"self.collapseList({lst}, 1)": "19, ...",
                f"self.collapseList({lst}, None, False, False)": ["19", "22..26"],
            }
        )

    def test_bitArrayToList(self):
        """ """
        flags = bytes.fromhex("beef")
        bits = [flags[i // 8] & 1 << i % 8 != 0 for i in range(len(flags) * 8)]

        self.perform_test(
            {
                f"self.bitArrayToList({bits})": [
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
                "self.removeDuplicates([0, 1, 2, 3, 2])": [0, 1, 2, 3],
                "self.removeDuplicates([0, 1, 2, 3, 2], False)": [0, 1, 3, 2],
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
                    self.filterWithMappedValues(
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
                f"self.filterDict({dct}, exc='*t*', values=True)": {1: "1", "two": 2},
                f"self.filterDict({dct}, exc='t*', keys=True)": {1: "1", 3: "three"},
                f"self.filterDict({dct}, exc=1, keys=True)": {"two": 2, 3: "three"},
            }
        )

    def test_filterList(self):
        """ """
        self.perform_test(
            {
                "self.filterList([0, 1, 2, 3, 2], [1, 2, 3], 2)": [1, 3],
                "self.filterList([0, 1, 'file.txt', 'file.jpg'], ['*file*', 0], '*.txt')": [
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
                f"self.splitList({lA}, '2parts')": [[1, 2, 3, 5], [7, 8, 9]],
                f"self.splitList({lB}, '2parts')": [[1, "2", 3, 5], ["7", 8, 9]],
                f"self.splitList({lA}, '2parts+')": [[1, 2, 3], [5, 7, 8], [9]],
                f"self.splitList({lB}, '2parts+')": [[1, "2", 3], [5, "7", 8], [9]],
                f"self.splitList({lA}, '2chunks')": [[1, 2], [3, 5], [7, 8], [9]],
                f"self.splitList({lB}, '2chunks')": [[1, "2"], [3, 5], ["7", 8], [9]],
                f"self.splitList({lA}, 'contiguous')": [[1, 2, 3], [5], [7, 8, 9]],
                f"self.splitList({lB}, 'contiguous')": [[1, "2", 3], [5], ["7", 8, 9]],
                f"self.splitList({lA}, 'range')": [[1, 3], [5], [7, 9]],
                f"self.splitList({lB}, 'range')": [[1, 3], [5], ["7", 9]],
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
                f"self.formatPath(r'{p1}')": "X:/n/dir1/dir3",
                f"self.formatPath(r'{p1}', 'path')": "X:/n/dir1/dir3",
                f"self.formatPath(r'{p2}', 'path')": "X:/n/dir1/dir3/.vscode",
                f"self.formatPath(r'{p3}', 'path')": "X:/n/dir1/dir3/.vscode",
                f"self.formatPath(r'{p4}', 'path')": r"\\192.168.1.240/nas/lost+found",
                f"self.formatPath(r'{p5}', 'path')": "C:/Program Files",
                f"self.formatPath(r'{p1}', 'dir')": "dir3",
                f"self.formatPath(r'{p2}', 'dir')": ".vscode",
                f"self.formatPath(r'{p3}', 'dir')": ".vscode",
                f"self.formatPath(r'{p4}', 'dir')": "lost+found",
                f"self.formatPath(r'{p5}', 'dir')": "Program Files",
                f"self.formatPath(r'{p1}', 'file')": "",
                f"self.formatPath(r'{p2}', 'file')": "",
                f"self.formatPath(r'{p3}', 'file')": "tasks.json",
                f"self.formatPath(r'{p4}', 'file')": "file.ext",
                f"self.formatPath(r'{p5}', 'file')": "",
                f"self.formatPath(r'{p1}', 'name')": "",
                f"self.formatPath(r'{p2}', 'name')": "",
                f"self.formatPath(r'{p3}', 'name')": "tasks",
                f"self.formatPath(r'{p4}', 'name')": "file",
                f"self.formatPath(r'{p5}', 'name')": "",
                f"self.formatPath(r'{p1}', 'ext')": "",
                f"self.formatPath(r'{p2}', 'ext')": "",
                f"self.formatPath(r'{p3}', 'ext')": "json",
                f"self.formatPath(r'{p4}', 'ext')": "ext",
                f"self.formatPath(r'{p5}', 'ext')": "",
                f"self.formatPath(r'{p6}', 'filename')": "programfiles",
                f"self.formatPath(r'{p6}', 'path')": "programfiles",
                f"self.formatPath(r'{p7}', 'path')": "programfiles",
                f"self.formatPath({[p1, p2]}, 'dir')": ["dir3", ".vscode"],
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
                # "self.timeStamp({})".format(paths): [],
                # "self.timeStamp({}, False, '%m-%d-%Y  %H:%M', True)".format(paths): [],
                # "self.timeStamp({}, True)".format(paths): [],
            }
        )

    def test_isValid(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"self.isValid(r'{file}')": "file",
                f"self.isValid(r'{path}')": "dir",
            }
        )

    def test_writeToFile(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"self.writeToFile(r'{file}', '__version__ = \"0.9.0\"')": None,
            }
        )

    def test_getFileContents(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"self.getFileContents(r'{file}', asList=True)": '__version__ = "0.9.0"',
                f"self.getFileContents(r'{file}', asList=True)": [
                    '__version__ = "0.9.0"'
                ],
            }
        )

    def test_createDirectory(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"

        self.perform_test(
            {
                f"self.createDir(r'{path}'+'/sub-directory')": None,
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
            self.getFileInfo(files, "file|filename|filepath"),
            [
                ("file1.txt", "file1", file1_path),
                ("file2.txt", "file2", file2_path),
            ],
        )

        self.assertEqual(
            self.getFileInfo(files, "file|filetype"),
            [
                ("file1.txt", ".txt"),
                ("file2.txt", ".txt"),
            ],
        )

        self.assertEqual(
            self.getFileInfo(files, "filename|filetype"),
            [
                ("file1", ".txt"),
                ("file2", ".txt"),
            ],
        )

        self.assertEqual(
            self.getFileInfo(files, "file|size"),
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
            self.getDirContents(path, "dirpaths"),
            [
                imgtk_test_dirpath,
                sub_directory_dirpath,
            ],
        )
        self.assertEqual(
            self.getDirContents(path, "filenames", recursive=True),
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
            self.getDirContents(path, "files|dirs"),
            ["imgtk_test", "sub-directory", "file1.txt", "file2.txt", "test.json"],
        )
        self.assertEqual(
            self.getDirContents(path, "files|dirs", excDirs=["sub*"]),
            ["imgtk_test", "file1.txt", "file2.txt", "test.json"],
        )
        self.assertEqual(
            self.getDirContents(path, "filenames", incFiles="*.txt"),
            ["file1", "file2"],
        )
        self.assertEqual(
            self.getDirContents(path, "files", incFiles="*.txt"),
            ["file1.txt", "file2.txt"],
        )
        self.assertEqual(
            sorted(self.getDirContents(path, "dirpath|dir")),
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
                "self.getFilepath(__file__)": path,
                "self.getFilepath(__file__, True)": __file__,
            }
        )

    def test_getFile(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"str(self.getFile(r'{file}'))": r"<_io.TextIOWrapper name='O:\\Cloud\\Code\\_scripts\\pythontk\\test/test_files/file1.txt' mode='a+' encoding='cp1252'>",
            }
        )

    def test_updateVersion(self):
        """ """
        path = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        file = path + "/file1.txt"

        self.perform_test(
            {
                f"str(self.updateVersion(r'{file}', 'increment'))": r"0.9.1",
                f"str(self.updateVersion(r'{file}', 'decrement'))": r"0.9.0",
            }
        )

    def test_json(self):
        """ """
        p = os.path.abspath(os.path.dirname(__file__)) + "/test_files"
        path = "/".join(p.split("\\")).rstrip("/")
        file = path + "/test.json"

        self.perform_test(
            {
                f"self.setJsonFile(r'{file}')": None,
                "self.getJsonFile()": file,
                "self.setJson('key', 'value')": None,
                "self.getJson('key')": "value",
            }
        )


class ImgTest(Main, Img):
    """ """

    im_h = Img.createImage("RGB", (1024, 1024), (0, 0, 0))
    im_n = Img.createImage("RGB", (1024, 1024), (127, 127, 255))

    def test_createImage(self):
        """ """
        self.perform_test(
            {
                "self.createImage('RGB', (1024, 1024), (0, 0, 0))": self.im_h,
            }
        )

    def test_resizeImage(self):
        """ """
        self.perform_test(
            {
                "self.resizeImage(self.im_h, 32, 32).size": (32, 32),
            }
        )

    def test_saveImageFile(self):
        """ """
        self.perform_test(
            {
                "self.saveImageFile(self.im_h, 'test_files/imgtk_test/im_h.png')": None,
                "self.saveImageFile(self.im_n, 'test_files/imgtk_test/im_n.png')": None,
            }
        )

    def test_getImages(self):
        """ """
        # print (\n'test_getImages:', self.getImages('test_files/imgtk_test/'))
        self.perform_test(
            {
                "list(self.getImages('test_files/imgtk_test/', '*Normal*').keys())": [
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
                # "self.getImageFiles('*.png|*.jpg')": '',
            }
        )

    def test_getImageDirectory(self):
        """ """
        print("\ngetImageDirectory: skipped")
        self.perform_test(
            {
                # "self.getImageDirectory()": '',
            }
        )

    def test_getImageTypeFromFilename(self):
        """ """
        self.perform_test(
            {
                "self.getImageTypeFromFilename('test_files/imgtk_test/im_h.png')": "Height",
                "self.getImageTypeFromFilename('test_files/imgtk_test/im_h.png', key=False)": "_H",
                "self.getImageTypeFromFilename('test_files/imgtk_test/im_n.png')": "Normal",
                "self.getImageTypeFromFilename('test_files/imgtk_test/im_n.png', key=False)": "_N",
            }
        )

    def test_filterImagesByType(self):
        """ """
        self.perform_test(
            {
                "self.filterImagesByType(File.getDirContents('test_files/imgtk_test'), 'Height')": [
                    "im_h.png",
                    "im_Height.png",
                ],
            }
        )

    def test_sortImagesByType(self):
        """ """
        self.perform_test(
            {
                "self.sortImagesByType([('im_h.png', '<im_h>'), ('im_n.png', '<im_n>')])": {
                    "Height": [("im_h.png", "<im_h>")],
                    "Normal": [("im_n.png", "<im_n>")],
                },
                "self.sortImagesByType({'im_h.png':'<im_h>', 'im_n.png':'<im_n>'})": {
                    "Height": [("im_h.png", "<im_h>")],
                    "Normal": [("im_n.png", "<im_n>")],
                },
            }
        )

    def test_containsMapTypes(self):
        """ """
        self.perform_test(
            {
                "self.containsMapTypes([('im_h.png', '<im_h>')], 'Height')": True,
                "self.containsMapTypes({'im_h.png':'<im_h>', 'im_n.png':'<im_n>'}, 'Height')": True,
                "self.containsMapTypes({'Height': [('im_h.png', '<im_h>')]}, 'Height')": True,
                "self.containsMapTypes({'Height': [('im_h.png', '<im_h>')]}, 'Height|Normal')": True,
                "self.containsMapTypes({'Height': [('im_h.png', '<im_h>')]}, ['Height', 'Normal'])": True,
            }
        )

    def test_isNormalMap(self):
        """ """
        self.perform_test(
            {
                "self.isNormalMap('im_h.png')": False,
                "self.isNormalMap('im_n.png')": True,
            }
        )

    def test_invertChannels(self):
        """ """
        self.perform_test(
            {
                "str(self.invertChannels(self.im_n, 'g').getchannel('G')).split('size')[0]": "<PIL.Image.Image image mode=L ",
            }
        )

    def test_createDXFromGL(self):
        """ """
        self.perform_test(
            {
                "self.createDXFromGL('test_files/imgtk_test/im_Normal_OpenGL.png')": "test_files/imgtk_test/im_Normal_DirectX.png",
            }
        )

    def test_createGLFromDX(self):
        """ """
        self.perform_test(
            {
                "self.createGLFromDX('test_files/imgtk_test/im_Normal_DirectX.png')": "test_files/imgtk_test/im_Normal_OpenGL.png",
            }
        )

    def test_createMask(self):
        """ """
        bg = self.getBackground("test_files/imgtk_test/im_Base_color.png", "RGB")
        # self.createMask('test_files/imgtk_test/im_Base_color.png', self.bg).show()
        self.perform_test(
            {
                f"str(self.createMask('test_files/imgtk_test/im_Base_color.png', {bg})).split('size')[0]": "<PIL.Image.Image image mode=L ",
                "str(self.createMask('test_files/imgtk_test/im_Base_color.png', 'test_files/imgtk_test/im_Base_color.png')).split('size')[0]": "<PIL.Image.Image image mode=L ",
            }
        )

    def test_fillMaskedArea(self):
        """ """
        bg = self.getBackground("test_files/imgtk_test/im_Base_color.png", "RGB")
        self.mask_fillMaskedArea = self.createMask(
            "test_files/imgtk_test/im_Base_color.png", bg
        )
        # self.fillMaskedArea('test_files/imgtk_test/im_Base_color.png', (0, 255, 0), self.mask).show()
        self.perform_test(
            {
                "str(self.fillMaskedArea('test_files/imgtk_test/im_Base_color.png', (0, 255, 0), self.mask_fillMaskedArea)).split('size')[0]": "<PIL.Image.Image image mode=RGB ",
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
                "self.getBackground('test_files/imgtk_test/im_Height.png', 'I')": 32767,
                "self.getBackground('test_files/imgtk_test/im_Height.png', 'L')": 255,
                "self.getBackground('test_files/imgtk_test/im_n.png', 'RGB')": (
                    127,
                    127,
                    255,
                ),
            }
        )

    def test_replaceColor(self):
        """ """
        bg = self.getBackground("test_files/imgtk_test/im_Base_color.png", "RGB")
        # self.replaceColor('test_files/imgtk_test/im_Base_color.png', self.bg, (255, 0, 0)).show()
        self.perform_test(
            {
                f"str(self.replaceColor('test_files/imgtk_test/im_Base_color.png', {bg}, (255, 0, 0))).split('size')[0]": "<PIL.Image.Image image mode=RGBA ",
            }
        )

    def test_setContrast(self):
        """ """
        # self.setContrast('test_files/imgtk_test/im_Mixed_AO.png', 255).show()
        self.perform_test(
            {
                "str(self.setContrast('test_files/imgtk_test/im_Mixed_AO.png', 255)).split('size')[0]": "<PIL.Image.Image image mode=L ",
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
                "str(self.convert_RGB_to_HSV(self.im_h)).split('size')[0]": "<PIL.Image.Image image mode=HSV ",
            }
        )

    def test_convert_I_to_L(self):
        """ """
        self.im_convert_I_to_L = self.createImage("I", (32, 32))
        # im = self.convert_I_to_L(self.im)
        self.perform_test(
            {
                "self.convert_I_to_L(self.im_convert_I_to_L).mode": "L",
            }
        )

    def test_areIdentical(self):
        """ """
        self.perform_test(
            {
                "self.areIdentical(self.im_h, self.im_n)": False,
                "self.areIdentical(self.im_h, self.im_h)": True,
            }
        )


class MathTest(Main, Math):
    """ """

    def test_getVectorFromTwoPoints(self):
        """ """
        self.perform_test(
            {
                "self.getVectorFromTwoPoints((1, 2, 3), (1, 1, -1))": (0, -1, -4),
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
                "self.getMagnitude((2, 3, 4))": 5.385164807134504,
                "self.getMagnitude((2, 3))": 3.605551275463989,
            }
        )

    def test_dotProduct(self):
        """ """
        self.perform_test(
            {
                "self.dotProduct((1, 2, 3), (1, 1, -1))": 0,
                "self.dotProduct((1, 2), (1, 1))": 3,
                "self.dotProduct((1, 2, 3), (1, 1, -1), True)": 0,
            }
        )

    def test_crossProduct(self):
        """ """
        self.perform_test(
            {
                "self.crossProduct((1, 2, 3), (1, 1, -1))": (-5, 4, -1),
                "self.crossProduct((3, 1, 1), (1, 4, 2), (1, 3, 4))": (7, 4, 2),
                "self.crossProduct((1, 2, 3), (1, 1, -1), None, 1)": (
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
                "self.movePointRelative((0, 5, 0), (0, 5, 0))": (0, 10, 0),
                "self.movePointRelative((0, 5, 0), 5, (0, 1, 0))": (0, 10, 0),
            }
        )

    def test_movePointAlongVectorRelativeToPoint(self):
        """ """
        self.perform_test(
            {
                "self.movePointAlongVectorRelativeToPoint((0, 0, 0), (0, 10, 0), (0, 1, 0), 5)": (
                    0.0,
                    5.0,
                    0.0,
                ),
                "self.movePointAlongVectorRelativeToPoint((0, 0, 0), (0, 10, 0), (0, 1, 0), 5, False)": (
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
                "self.getDistBetweenTwoPoints((0, 10, 0), (0, 5, 0))": 5.0,
            }
        )

    def test_getCenterPointBetweenTwoPoints(self):
        """ """
        self.perform_test(
            {
                "self.getCenterPointBetweenTwoPoints((0, 10, 0), (0, 5, 0))": (
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
                "self.getAngleFrom2Vectors((1, 2, 3), (1, 1, -1))": 1.5707963267948966,
                "self.getAngleFrom2Vectors((1, 2, 3), (1, 1, -1), True)": 90,
            }
        )

    def test_getAngleFrom3Points(self):
        """ """
        self.perform_test(
            {
                "self.getAngleFrom3Points((1, 1, 1), (-1, 2, 3), (1, 4, -3))": 0.7904487543360762,
                "self.getAngleFrom3Points((1, 1, 1), (-1, 2, 3), (1, 4, -3), True)": 45.29,
            }
        )

    def test_getTwoSidesOfASATriangle(self):
        """ """
        self.perform_test(
            {
                "self.getTwoSidesOfASATriangle(60, 60, 100)": (
                    100.00015320566493,
                    100.00015320566493,
                ),
            }
        )

    def test_xyzRotation(self):
        """ """
        self.perform_test(
            {
                "self.xyzRotation(2, (0, 1, 0))": (
                    3.589792907376932e-09,
                    1.9999999964102069,
                    3.589792907376932e-09,
                ),
                "self.xyzRotation(2, (0, 1, 0), [], True)": (0.0, 114.59, 0.0),
            }
        )


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(exit=False)
    # print(self.areSimilar)

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
