#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk IterUtils.

Run with:
    python -m pytest test_iter.py -v
    python test_iter.py
"""
import unittest

from pythontk import IterUtils

from conftest import BaseTestCase


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


if __name__ == "__main__":
    unittest.main(exit=False)
