#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk IterUtils.

Comprehensive edge case coverage for:
- make_iterable
- nested_depth
- flatten
- collapse_integer_sequence
- bit_array_to_list
- indices / rindex
- remove_duplicates
- filter_list / filter_dict
- split_list

Run with:
    python -m pytest test_iter.py -v
    python test_iter.py
"""
import unittest

from pythontk import IterUtils

from conftest import BaseTestCase


class IterTest(BaseTestCase):
    """Iterator utilities test class with comprehensive edge case coverage."""

    # -------------------------------------------------------------------------
    # make_iterable Tests
    # -------------------------------------------------------------------------

    def test_make_iterable_custom_objects(self):
        """Test make_iterable wraps custom objects in tuples."""

        class ExampleClass:
            pass

        class ExampleClassWithAttr:
            __apimfn__ = True

        example_instance = ExampleClass()
        example_instance_with_attr = ExampleClassWithAttr()

        self.assertEqual(IterUtils.make_iterable(example_instance), (example_instance,))
        self.assertEqual(
            IterUtils.make_iterable(example_instance_with_attr),
            (example_instance_with_attr,),
        )

    def test_make_iterable_scalars(self):
        """Test make_iterable wraps scalars in tuples."""
        self.assertEqual(IterUtils.make_iterable("foo"), ("foo",))
        self.assertEqual(IterUtils.make_iterable(1), (1,))
        self.assertEqual(IterUtils.make_iterable(""), ("",))
        self.assertEqual(IterUtils.make_iterable(3.14), (3.14,))
        self.assertEqual(IterUtils.make_iterable(True), (True,))

    def test_make_iterable_none(self):
        """Test make_iterable handles None - returns empty tuple."""
        result = IterUtils.make_iterable(None)
        self.assertEqual(result, ())

    def test_make_iterable_bytes(self):
        """Test make_iterable handles bytes as scalar."""
        result = IterUtils.make_iterable(b"hello")
        self.assertEqual(result, (b"hello",))

    def test_make_iterable_collections_unchanged(self):
        """Test make_iterable leaves collections as-is."""
        self.assertEqual(IterUtils.make_iterable(["foo", "bar"]), ["foo", "bar"])
        self.assertEqual(IterUtils.make_iterable(("foo", "bar")), ("foo", "bar"))
        self.assertEqual(IterUtils.make_iterable({"foo": "bar"}), {"foo": "bar"})
        self.assertEqual(IterUtils.make_iterable(range(3)), range(3))
        self.assertEqual(IterUtils.make_iterable({1, 2, 3}), {1, 2, 3})

    def test_make_iterable_empty_collections(self):
        """Test make_iterable handles empty collections."""
        self.assertEqual(IterUtils.make_iterable([]), [])
        self.assertEqual(IterUtils.make_iterable(()), ())
        self.assertEqual(IterUtils.make_iterable({}), {})
        self.assertEqual(IterUtils.make_iterable(set()), set())

    def test_make_iterable_single_element_collections(self):
        """Test make_iterable handles single element collections."""
        self.assertEqual(IterUtils.make_iterable([42]), [42])
        self.assertEqual(IterUtils.make_iterable((42,)), (42,))
        self.assertEqual(IterUtils.make_iterable({42}), {42})

    def test_make_iterable_iterators(self):
        """Test make_iterable converts iterators to lists."""
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

    def test_make_iterable_generators(self):
        """Test make_iterable returns generator as-is (not converted to list)."""
        from types import GeneratorType

        def gen():
            yield 1
            yield 2
            yield 3

        result = IterUtils.make_iterable(gen())
        self.assertIsInstance(result, GeneratorType)
        # Can still consume the generator
        self.assertEqual(list(result), [1, 2, 3])

    def test_make_iterable_generators_snapshot(self):
        """Test make_iterable with snapshot=True converts generator to list."""

        def gen():
            yield 1
            yield 2
            yield 3

        result = IterUtils.make_iterable(gen(), snapshot=True)
        self.assertEqual(result, [1, 2, 3])

    def test_make_iterable_nested_list(self):
        """Test make_iterable with nested structures."""
        nested = [[1, 2], [3, 4]]
        result = IterUtils.make_iterable(nested)
        self.assertEqual(result, [[1, 2], [3, 4]])

    # -------------------------------------------------------------------------
    # nested_depth Tests
    # -------------------------------------------------------------------------

    def test_nested_depth_basic(self):
        """Test nested_depth calculates nesting level correctly."""
        self.assertEqual(IterUtils.nested_depth([[1, 2], [3, 4]]), 1)
        self.assertEqual(IterUtils.nested_depth([1, 2, 3, 4]), 0)

    def test_nested_depth_deeply_nested(self):
        """Test nested_depth with deeply nested structures."""
        self.assertEqual(IterUtils.nested_depth([[[1]]]), 2)
        self.assertEqual(IterUtils.nested_depth([[[[1]]]]), 3)
        self.assertEqual(IterUtils.nested_depth([[[[[1]]]]]), 4)

    def test_nested_depth_empty_list(self):
        """Test nested_depth with empty list."""
        self.assertEqual(IterUtils.nested_depth([]), 0)

    def test_nested_depth_mixed_nesting(self):
        """Test nested_depth with mixed nesting levels."""
        # Should return max depth
        mixed = [[1, 2], [[3, 4]]]
        result = IterUtils.nested_depth(mixed)
        self.assertGreaterEqual(result, 1)

    def test_nested_depth_single_element(self):
        """Test nested_depth with single element."""
        self.assertEqual(IterUtils.nested_depth([42]), 0)
        self.assertEqual(IterUtils.nested_depth([[42]]), 1)

    # -------------------------------------------------------------------------
    # flatten Tests
    # -------------------------------------------------------------------------

    def test_flatten_basic(self):
        """Test flatten unnests nested lists."""
        self.assertEqual(list(IterUtils.flatten([[1, 2], [3, 4]])), [1, 2, 3, 4])

    def test_flatten_deeply_nested(self):
        """Test flatten with deeply nested structures."""
        nested = [[[1, 2]], [[3, 4]]]
        result = list(IterUtils.flatten(nested))
        # Should flatten to some degree
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_flatten_empty_list(self):
        """Test flatten with empty list."""
        self.assertEqual(list(IterUtils.flatten([])), [])

    def test_flatten_empty_sublists(self):
        """Test flatten with empty sublists."""
        self.assertEqual(list(IterUtils.flatten([[], [1], []])), [1])

    def test_flatten_single_level(self):
        """Test flatten with already flat list."""
        self.assertEqual(list(IterUtils.flatten([1, 2, 3])), [1, 2, 3])

    def test_flatten_mixed_types(self):
        """Test flatten with mixed types."""
        result = list(IterUtils.flatten([[1, "a"], [2.5, None]]))
        self.assertEqual(result, [1, "a", 2.5, None])

    def test_flatten_preserves_strings(self):
        """Test flatten doesn't break apart strings."""
        result = list(IterUtils.flatten([["hello", "world"]]))
        self.assertIn("hello", result)
        self.assertIn("world", result)

    # -------------------------------------------------------------------------
    # collapse_integer_sequence Tests
    # -------------------------------------------------------------------------

    def test_collapse_integer_sequence_basic(self):
        """Test collapse_integer_sequence creates range notation."""
        lst = [19, 22, 23, 24, 25, 26]
        self.assertEqual(IterUtils.collapse_integer_sequence(lst), "19, 22-6")

    def test_collapse_integer_sequence_limit(self):
        """Test collapse_integer_sequence with limit parameter."""
        lst = [19, 22, 23, 24, 25, 26]
        self.assertEqual(IterUtils.collapse_integer_sequence(lst, 1), "19, ...")

    def test_collapse_integer_sequence_no_string(self):
        """Test collapse_integer_sequence returning list."""
        lst = [19, 22, 23, 24, 25, 26]
        self.assertEqual(
            IterUtils.collapse_integer_sequence(lst, None, False, False),
            ["19", "22..26"],
        )

    def test_collapse_integer_sequence_empty(self):
        """Test collapse_integer_sequence with empty list."""
        result = IterUtils.collapse_integer_sequence([])
        self.assertEqual(result, "")

    def test_collapse_integer_sequence_single(self):
        """Test collapse_integer_sequence with single element."""
        result = IterUtils.collapse_integer_sequence([5])
        self.assertEqual(result, "5")

    def test_collapse_integer_sequence_no_ranges(self):
        """Test collapse_integer_sequence with non-contiguous sequence."""
        result = IterUtils.collapse_integer_sequence([1, 3, 5, 7])
        self.assertIn("1", result)
        self.assertIn("3", result)

    def test_collapse_integer_sequence_all_contiguous(self):
        """Test collapse_integer_sequence with all contiguous."""
        result = IterUtils.collapse_integer_sequence([1, 2, 3, 4, 5])
        self.assertIn("1-5", result)

    # -------------------------------------------------------------------------
    # bit_array_to_list Tests
    # -------------------------------------------------------------------------

    def test_bit_array_to_list_basic(self):
        """Test bit_array_to_list converts bit flags to indices."""
        flags = bytes.fromhex("beef")
        bits = [flags[i // 8] & 1 << i % 8 != 0 for i in range(len(flags) * 8)]
        self.assertEqual(
            IterUtils.bit_array_to_list(bits),
            [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16],
        )

    def test_bit_array_to_list_empty(self):
        """Test bit_array_to_list with empty array returns None."""
        self.assertIsNone(IterUtils.bit_array_to_list([]))

    def test_bit_array_to_list_all_false(self):
        """Test bit_array_to_list with all False bits."""
        self.assertEqual(IterUtils.bit_array_to_list([False, False, False]), [])

    def test_bit_array_to_list_all_true(self):
        """Test bit_array_to_list with all True bits - uses 1-based indexing."""
        self.assertEqual(IterUtils.bit_array_to_list([True, True, True]), [1, 2, 3])

    def test_bit_array_to_list_single_bit(self):
        """Test bit_array_to_list with single True bit - uses 1-based indexing."""
        self.assertEqual(IterUtils.bit_array_to_list([False, False, True]), [3])

    # -------------------------------------------------------------------------
    # indices / rindex Tests
    # -------------------------------------------------------------------------

    def test_indices_basic(self):
        """Test indices finds all occurrences of value."""
        self.assertEqual(tuple(IterUtils.indices([0, 1, 2, 2, 3], 2)), (2, 3))

    def test_indices_not_found(self):
        """Test indices returns empty when not found."""
        self.assertEqual(tuple(IterUtils.indices([0, 1, 2, 2, 3], 4)), ())

    def test_indices_empty_list(self):
        """Test indices with empty list."""
        self.assertEqual(tuple(IterUtils.indices([], 1)), ())

    def test_indices_single_occurrence(self):
        """Test indices with single occurrence."""
        self.assertEqual(tuple(IterUtils.indices([1, 2, 3], 2)), (1,))

    def test_indices_all_same(self):
        """Test indices when all elements are same."""
        self.assertEqual(tuple(IterUtils.indices([5, 5, 5, 5], 5)), (0, 1, 2, 3))

    def test_indices_with_none(self):
        """Test indices finding None values."""
        self.assertEqual(tuple(IterUtils.indices([None, 1, None], None)), (0, 2))

    def test_rindex_basic(self):
        """Test rindex finds last occurrence of value."""
        self.assertEqual(IterUtils.rindex([0, 1, 2, 2, 3], 2), 3)

    def test_rindex_not_found(self):
        """Test rindex returns -1 when not found."""
        self.assertEqual(IterUtils.rindex([0, 1, 2, 2, 3], 4), -1)

    def test_rindex_single_occurrence(self):
        """Test rindex with single occurrence."""
        self.assertEqual(IterUtils.rindex([1, 2, 3], 2), 1)

    def test_rindex_first_element(self):
        """Test rindex when target is first element only."""
        self.assertEqual(IterUtils.rindex([5, 1, 2, 3], 5), 0)

    def test_rindex_empty_list(self):
        """Test rindex with empty list."""
        self.assertEqual(IterUtils.rindex([], 1), -1)

    # -------------------------------------------------------------------------
    # remove_duplicates Tests
    # -------------------------------------------------------------------------

    def test_remove_duplicates_basic(self):
        """Test remove_duplicates removes duplicate values."""
        self.assertEqual(IterUtils.remove_duplicates([0, 1, 2, 3, 2]), [0, 1, 2, 3])

    def test_remove_duplicates_keep_last(self):
        """Test remove_duplicates with keep_first=False."""
        self.assertEqual(
            IterUtils.remove_duplicates([0, 1, 2, 3, 2], False), [0, 1, 3, 2]
        )

    def test_remove_duplicates_empty_list(self):
        """Test remove_duplicates with empty list."""
        self.assertEqual(IterUtils.remove_duplicates([]), [])

    def test_remove_duplicates_no_duplicates(self):
        """Test remove_duplicates when no duplicates exist."""
        self.assertEqual(IterUtils.remove_duplicates([1, 2, 3]), [1, 2, 3])

    def test_remove_duplicates_all_same(self):
        """Test remove_duplicates when all elements are same."""
        self.assertEqual(IterUtils.remove_duplicates([5, 5, 5, 5]), [5])

    def test_remove_duplicates_single_element(self):
        """Test remove_duplicates with single element."""
        self.assertEqual(IterUtils.remove_duplicates([42]), [42])

    def test_remove_duplicates_with_none(self):
        """Test remove_duplicates with None values."""
        self.assertEqual(IterUtils.remove_duplicates([None, 1, None, 2]), [None, 1, 2])

    def test_remove_duplicates_preserves_order(self):
        """Test remove_duplicates preserves order."""
        self.assertEqual(
            IterUtils.remove_duplicates([3, 1, 4, 1, 5, 9, 2, 6, 5]),
            [3, 1, 4, 5, 9, 2, 6],
        )

    def test_remove_duplicates_with_strings(self):
        """Test remove_duplicates with strings."""
        self.assertEqual(
            IterUtils.remove_duplicates(["a", "b", "a", "c"]), ["a", "b", "c"]
        )

    # -------------------------------------------------------------------------
    # filter_list Tests
    # -------------------------------------------------------------------------

    def test_filter_list_basic(self):
        """Test filter_list includes/excludes items by pattern."""
        self.assertEqual(IterUtils.filter_list([0, 1, 2, 3, 2], [1, 2, 3], 2), [1, 3])

    def test_filter_list_with_patterns(self):
        """Test filter_list with glob patterns."""
        self.assertEqual(
            IterUtils.filter_list(
                [0, 1, "file.txt", "file.jpg"], ["*file*", 0], "*.txt"
            ),
            [0, "file.jpg"],
        )

    def test_filter_list_with_map_func(self):
        """Test filter_list with map_func."""
        self.assertEqual(
            IterUtils.filter_list(
                ["apple", "banana", "cherry"], "*a*", "*n*", map_func=lambda x: x[::-1]
            ),
            ["apple"],
        )

    def test_filter_list_with_check_unmapped(self):
        """Test filter_list with check_unmapped=True."""
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

    def test_filter_list_empty_list(self):
        """Test filter_list with empty list."""
        self.assertEqual(IterUtils.filter_list([], [1, 2], 3), [])

    def test_filter_list_no_matches(self):
        """Test filter_list when nothing matches."""
        self.assertEqual(IterUtils.filter_list([1, 2, 3], [10, 20], None), [])

    def test_filter_list_all_excluded(self):
        """Test filter_list when all items are excluded."""
        self.assertEqual(IterUtils.filter_list([1, 2, 3], [1, 2, 3], [1, 2, 3]), [])

    def test_filter_list_with_objects(self):
        """Test filter_list with object inputs."""

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

    def test_filter_list_nested_tuples(self):
        """Test filter_list with nested tuples."""
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

    def test_filter_list_single_element(self):
        """Test filter_list with single element list."""
        self.assertEqual(IterUtils.filter_list([42], [42], None), [42])
        self.assertEqual(IterUtils.filter_list([42], None, [42]), [])

    # -------------------------------------------------------------------------
    # filter_dict Tests
    # -------------------------------------------------------------------------

    def test_filter_dict_basic(self):
        """Test filter_dict filters dictionary by keys/values."""
        dct = {1: "1", "two": 2, 3: "three"}
        self.assertEqual(
            IterUtils.filter_dict(dct, exc="*t*", values=True),
            {1: "1", "two": 2},
        )

    def test_filter_dict_by_keys(self):
        """Test filter_dict filtering by keys."""
        dct = {1: "1", "two": 2, 3: "three"}
        self.assertEqual(
            IterUtils.filter_dict(dct, exc="t*", keys=True),
            {1: "1", 3: "three"},
        )

    def test_filter_dict_exact_match(self):
        """Test filter_dict with exact match."""
        dct = {1: "1", "two": 2, 3: "three"}
        self.assertEqual(
            IterUtils.filter_dict(dct, exc=1, keys=True),
            {"two": 2, 3: "three"},
        )

    def test_filter_dict_empty(self):
        """Test filter_dict with empty dictionary."""
        self.assertEqual(IterUtils.filter_dict({}, exc="*", keys=True), {})

    def test_filter_dict_no_matches(self):
        """Test filter_dict when no matches for exclusion."""
        dct = {"a": 1, "b": 2}
        self.assertEqual(IterUtils.filter_dict(dct, exc="*z*", keys=True), dct)

    def test_filter_dict_all_excluded(self):
        """Test filter_dict when everything is excluded."""
        dct = {"a": 1, "b": 2}
        result = IterUtils.filter_dict(dct, exc="*", keys=True)
        self.assertEqual(result, {})

    # -------------------------------------------------------------------------
    # split_list Tests
    # -------------------------------------------------------------------------

    def test_split_list_2parts(self):
        """Test split_list with 2parts mode."""
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]
        self.assertEqual(IterUtils.split_list(lA, "2parts"), [[1, 2, 3, 5], [7, 8, 9]])
        self.assertEqual(
            IterUtils.split_list(lB, "2parts"), [[1, "2", 3, 5], ["7", 8, 9]]
        )

    def test_split_list_2parts_plus(self):
        """Test split_list with 2parts+ mode."""
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]
        self.assertEqual(
            IterUtils.split_list(lA, "2parts+"), [[1, 2, 3], [5, 7, 8], [9]]
        )
        self.assertEqual(
            IterUtils.split_list(lB, "2parts+"), [[1, "2", 3], [5, "7", 8], [9]]
        )

    def test_split_list_2chunks(self):
        """Test split_list with 2chunks mode."""
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]
        self.assertEqual(
            IterUtils.split_list(lA, "2chunks"),
            [[1, 2], [3, 5], [7, 8], [9]],
        )
        self.assertEqual(
            IterUtils.split_list(lB, "2chunks"),
            [[1, "2"], [3, 5], ["7", 8], [9]],
        )

    def test_split_list_contiguous(self):
        """Test split_list with contiguous mode."""
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]
        self.assertEqual(
            IterUtils.split_list(lA, "contiguous"),
            [[1, 2, 3], [5], [7, 8, 9]],
        )
        self.assertEqual(
            IterUtils.split_list(lB, "contiguous"),
            [[1, "2", 3], [5], ["7", 8, 9]],
        )

    def test_split_list_range(self):
        """Test split_list with range mode."""
        lA = [1, 2, 3, 5, 7, 8, 9]
        lB = [1, "2", 3, 5, "7", 8, 9]
        self.assertEqual(IterUtils.split_list(lA, "range"), [[1, 3], [5], [7, 9]])
        self.assertEqual(IterUtils.split_list(lB, "range"), [[1, 3], [5], ["7", 9]])

    def test_split_list_empty(self):
        """Test split_list with empty list raises ValueError (division by zero in range)."""
        with self.assertRaises(ValueError):
            IterUtils.split_list([], "2parts")

    def test_split_list_single_element(self):
        """Test split_list with single element."""
        self.assertEqual(IterUtils.split_list([42], "2parts"), [[42]])

    def test_split_list_two_elements(self):
        """Test split_list with two elements."""
        result = IterUtils.split_list([1, 2], "2parts")
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main(exit=False)
