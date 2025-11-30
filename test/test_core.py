#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk CoreUtils.

Comprehensive edge case coverage for:
- cached_property decorator
- listify decorator
- format_return
- set_attributes / get_attributes
- has_attribute
- cycle
- are_similar
- randomize

Run with:
    python -m pytest test_core.py -v
    python test_core.py
"""
import unittest
import types

from pythontk import CoreUtils

from conftest import BaseTestCase


class CoreTest(BaseTestCase):
    """CoreUtils test class with comprehensive edge case coverage."""

    # -------------------------------------------------------------------------
    # Import Tests
    # -------------------------------------------------------------------------

    def test_imports(self):
        """Test that package imports work correctly."""
        import pythontk as ptk
        from pythontk import IterUtils
        from pythontk import make_iterable

        self.assertIsInstance(ptk, types.ModuleType)
        self.assertIsInstance(IterUtils, type)
        self.assertIsInstance(make_iterable, types.FunctionType)

    # -------------------------------------------------------------------------
    # cached_property Tests
    # -------------------------------------------------------------------------

    def test_cached_property_basic(self):
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

    def test_cached_property_returns_none(self):
        """Test cached_property correctly caches None values."""

        class MyClass:
            def __init__(self):
                self.call_count = 0

            @CoreUtils.cached_property
            def nullable(self):
                self.call_count += 1
                return None

        obj = MyClass()
        result1 = obj.nullable
        result2 = obj.nullable

        self.assertIsNone(result1)
        self.assertIsNone(result2)
        self.assertEqual(obj.call_count, 1)  # Called only once

    def test_cached_property_multiple_instances(self):
        """Test that cached_property is per-instance."""

        class MyClass:
            def __init__(self, value):
                self.value = value

            @CoreUtils.cached_property
            def computed(self):
                return self.value * 2

        obj1 = MyClass(5)
        obj2 = MyClass(10)

        self.assertEqual(obj1.computed, 10)
        self.assertEqual(obj2.computed, 20)

    def test_cached_property_with_exception(self):
        """Test cached_property behavior when computation raises exception."""

        class MyClass:
            def __init__(self):
                self.should_fail = True

            @CoreUtils.cached_property
            def risky(self):
                if self.should_fail:
                    raise ValueError("Intentional error")
                return "success"

        obj = MyClass()

        with self.assertRaises(ValueError):
            _ = obj.risky

        # After fixing the condition, should work
        obj.should_fail = False
        # Note: The cache will NOT be set if exception was raised
        # So we can now get the result
        self.assertEqual(obj.risky, "success")

    # -------------------------------------------------------------------------
    # Listify Decorator Tests
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

    def test_listify_with_empty_list(self):
        """Test listify handles empty lists correctly."""

        @CoreUtils.listify
        def double(n):
            return n * 2

        self.assertEqual(double([]), [])

    def test_listify_with_single_item(self):
        """Test listify handles single items (non-list) correctly."""

        @CoreUtils.listify
        def double(n):
            return n * 2

        # Single item should return single result, not list
        self.assertEqual(double(5), 10)

    def test_listify_with_none(self):
        """Test listify handles None input correctly."""

        class TestClass:
            @CoreUtils.listify
            def to_str(self, n, x=None):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str(None), "None")

    def test_listify_with_string_input(self):
        """Test listify treats strings as single items, not iterables."""

        @CoreUtils.listify
        def process(s):
            return s.upper()

        # String should be treated as single item
        self.assertEqual(process("hello"), "HELLO")
        # List of strings should be processed individually
        self.assertEqual(process(["hello", "world"]), ["HELLO", "WORLD"])

    def test_listify_with_bytes_input(self):
        """Test listify treats bytes as single items."""

        @CoreUtils.listify
        def process(b):
            return len(b)

        self.assertEqual(process(b"hello"), 5)
        self.assertEqual(process([b"hello", b"world"]), [5, 5])

    def test_listify_preserves_return_types(self):
        """Test listify returns list when input is list, single when input is single."""

        @CoreUtils.listify
        def identity(x):
            return x

        # Single input returns single output
        self.assertEqual(identity(42), 42)
        self.assertEqual(identity("test"), "test")

        # List input returns list output
        self.assertEqual(identity([1, 2, 3]), [1, 2, 3])
        self.assertEqual(identity((1, 2)), [1, 2])  # Tuple becomes list

    def test_listify_method_with_overlapping_args_and_kwargs(self):
        """Test listify with various arg/kwarg combinations."""

        class TestClass:
            @CoreUtils.listify(arg_name="n")
            def to_str(self, n, x=None):
                return str(n)

        test_obj = TestClass()
        self.assertEqual(test_obj.to_str([0, 1], x=2), ["0", "1"])
        self.assertEqual(test_obj.to_str(n=[0, 1], x=2), ["0", "1"])

    def test_listify_with_generator_input(self):
        """Test listify handles generators correctly."""

        @CoreUtils.listify
        def double(n):
            return n * 2

        gen = (x for x in [1, 2, 3])
        result = double(gen)
        self.assertEqual(result, [2, 4, 6])

    def test_listify_with_set_input(self):
        """Test listify handles sets correctly."""

        @CoreUtils.listify
        def double(n):
            return n * 2

        result = double({1, 2, 3})
        self.assertEqual(sorted(result), [2, 4, 6])

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
    # format_return Tests
    # -------------------------------------------------------------------------

    def test_format_return_single_element(self):
        """Test format_return with single element list."""
        self.assertEqual(CoreUtils.format_return([""]), "")
        self.assertEqual(CoreUtils.format_return(["hello"]), "hello")
        self.assertEqual(CoreUtils.format_return([42]), 42)

    def test_format_return_with_original_multi(self):
        """Test format_return preserves list when original was multi-element."""
        self.assertEqual(CoreUtils.format_return([""], [""]), [""])
        self.assertEqual(CoreUtils.format_return(["a"], ["a", "b"]), ["a"])

    def test_format_return_multiple_elements(self):
        """Test format_return with multiple elements."""
        self.assertEqual(CoreUtils.format_return(["", ""]), ["", ""])
        self.assertEqual(CoreUtils.format_return([1, 2, 3]), [1, 2, 3])

    def test_format_return_empty_list(self):
        """Test format_return with empty list."""
        self.assertIsNone(CoreUtils.format_return([], ""))
        self.assertIsNone(CoreUtils.format_return([]))

    def test_format_return_with_none_elements(self):
        """Test format_return with None in list."""
        self.assertIsNone(CoreUtils.format_return([None]))
        self.assertEqual(CoreUtils.format_return([None, None]), [None, None])

    def test_format_return_preserves_original_tuple(self):
        """Test format_return with tuple as original."""
        result = CoreUtils.format_return(["a"], ("x", "y"))
        self.assertEqual(result, ["a"])

    def test_format_return_preserves_original_set(self):
        """Test format_return with set as original."""
        result = CoreUtils.format_return(["a"], {"x", "y"})
        self.assertEqual(result, ["a"])

    def test_format_return_preserves_original_dict(self):
        """Test format_return with dict as original."""
        result = CoreUtils.format_return(["a"], {"x": 1})
        self.assertEqual(result, ["a"])

    def test_format_return_preserves_original_range(self):
        """Test format_return with range as original."""
        result = CoreUtils.format_return(["a"], range(5))
        self.assertEqual(result, ["a"])

    # -------------------------------------------------------------------------
    # set_attributes / get_attributes Tests
    # -------------------------------------------------------------------------

    def test_set_attributes_basic(self):
        """Test set_attributes sets object attributes."""
        obj = type("TestObj", (), {})()
        result = CoreUtils.set_attributes(obj, attr="value")
        self.assertIsNone(result)
        self.assertEqual(obj.attr, "value")

    def test_set_attributes_multiple(self):
        """Test set_attributes with multiple attributes."""
        obj = type("TestObj", (), {})()
        CoreUtils.set_attributes(obj, a=1, b=2, c=3)
        self.assertEqual(obj.a, 1)
        self.assertEqual(obj.b, 2)
        self.assertEqual(obj.c, 3)

    def test_set_attributes_skips_falsy_values(self):
        """Test set_attributes skips falsy attribute names and values."""
        obj = type("TestObj", (), {})()
        # Empty string attr name or None value should be skipped
        CoreUtils.set_attributes(obj, valid="ok", empty_val=None)
        self.assertEqual(obj.valid, "ok")
        self.assertFalse(hasattr(obj, "empty_val"))

    def test_get_attributes_basic(self):
        """Test get_attributes retrieves object attributes."""
        obj = type("TestObj", (), {})()
        obj._subtest = None
        self.assertEqual(CoreUtils.get_attributes(obj, "_subtest"), {"_subtest": None})

    def test_get_attributes_with_include(self):
        """Test get_attributes with include filter."""
        # Must use instance attributes (set on obj), not class attributes
        obj = type("TestObj", (), {})()
        obj.a = 1
        obj.b = 2
        obj.c = 3
        result = CoreUtils.get_attributes(obj, inc=["a", "b"])
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_get_attributes_with_exclude(self):
        """Test get_attributes with exclude filter."""
        # Must use instance attributes (set on obj), not class attributes
        obj = type("TestObj", (), {})()
        obj.a = 1
        obj.b = 2
        obj.c = 3
        result = CoreUtils.get_attributes(obj, exc=["c"])
        self.assertNotIn("c", result)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_get_attributes_exclude_takes_precedence(self):
        """Test that exclude takes precedence over include."""
        # Must use instance attributes (set on obj), not class attributes
        obj = type("TestObj", (), {})()
        obj.a = 1
        obj.b = 2
        obj.c = 3
        result = CoreUtils.get_attributes(obj, inc=["a", "b", "c"], exc=["b"])
        self.assertNotIn("b", result)
        self.assertIn("a", result)
        self.assertIn("c", result)

    # -------------------------------------------------------------------------
    # has_attribute Tests
    # -------------------------------------------------------------------------

    def test_has_attribute_exists(self):
        """Test has_attribute returns True for existing attributes."""

        class MyClass:
            static_attr = "value"

        self.assertTrue(CoreUtils.has_attribute(MyClass, "static_attr"))

    def test_has_attribute_not_exists(self):
        """Test has_attribute returns False for non-existing attributes."""

        class MyClass:
            pass

        self.assertFalse(CoreUtils.has_attribute(MyClass, "nonexistent"))

    def test_has_attribute_inherited(self):
        """Test has_attribute finds inherited attributes."""

        class Parent:
            inherited_attr = "parent"

        class Child(Parent):
            pass

        self.assertTrue(CoreUtils.has_attribute(Child, "inherited_attr"))

    def test_has_attribute_doesnt_trigger_getattr(self):
        """Test has_attribute doesn't invoke __getattr__."""

        class MyClass:
            def __getattr__(self, name):
                return "dynamic"

        # __getattr__ would return a value, but has_attribute should return False
        self.assertFalse(CoreUtils.has_attribute(MyClass, "anything"))

    # -------------------------------------------------------------------------
    # cycle Tests
    # -------------------------------------------------------------------------

    def test_cycle_basic(self):
        """Test cycle iterates through list cyclically."""
        self.assertEqual(CoreUtils.cycle([0, 1], "ID"), 0)
        self.assertEqual(CoreUtils.cycle([0, 1], "ID"), 1)
        self.assertEqual(CoreUtils.cycle([0, 1], "ID"), 0)

    def test_cycle_with_different_keys(self):
        """Test cycle maintains separate cycles for different keys."""
        CoreUtils.CYCLEDICT.clear()  # Reset state

        self.assertEqual(CoreUtils.cycle([1, 2, 3], "key1"), 1)
        self.assertEqual(CoreUtils.cycle([10, 20], "key2"), 10)
        self.assertEqual(CoreUtils.cycle([1, 2, 3], "key1"), 2)
        self.assertEqual(CoreUtils.cycle([10, 20], "key2"), 20)

    def test_cycle_query_mode(self):
        """Test cycle query mode returns current value without advancing."""
        CoreUtils.CYCLEDICT.clear()

        CoreUtils.cycle([1, 2, 3], "test")  # Returns 1, advances to 2
        result = CoreUtils.cycle([1, 2, 3], "test", query=True)
        self.assertEqual(result, 1)  # Last returned value

    def test_cycle_single_element(self):
        """Test cycle with single element list."""
        CoreUtils.CYCLEDICT.clear()

        self.assertEqual(CoreUtils.cycle([42], "single"), 42)
        self.assertEqual(CoreUtils.cycle([42], "single"), 42)
        self.assertEqual(CoreUtils.cycle([42], "single"), 42)

    def test_cycle_with_various_types(self):
        """Test cycle with various data types in sequence."""
        CoreUtils.CYCLEDICT.clear()

        sequence = ["a", 1, None, (1, 2)]
        for expected in sequence:
            self.assertEqual(CoreUtils.cycle(sequence, "mixed"), expected)

    # -------------------------------------------------------------------------
    # are_similar Tests
    # -------------------------------------------------------------------------

    def test_are_similar_within_tolerance(self):
        """Test are_similar returns True within tolerance."""
        self.assertTrue(CoreUtils.are_similar(1, 10, 9))
        self.assertTrue(CoreUtils.are_similar(5, 5, 0))
        self.assertTrue(CoreUtils.are_similar(1.5, 2.0, 0.5))

    def test_are_similar_outside_tolerance(self):
        """Test are_similar returns False outside tolerance."""
        self.assertFalse(CoreUtils.are_similar(1, 10, 8))
        self.assertFalse(CoreUtils.are_similar(0, 100, 50))

    def test_are_similar_zero_tolerance(self):
        """Test are_similar with zero tolerance (exact match)."""
        self.assertTrue(CoreUtils.are_similar(5, 5, 0))
        self.assertFalse(CoreUtils.are_similar(5, 5.001, 0))

    def test_are_similar_negative_numbers(self):
        """Test are_similar with negative numbers."""
        self.assertTrue(CoreUtils.are_similar(-5, -3, 2))
        self.assertFalse(CoreUtils.are_similar(-5, 5, 5))

    def test_are_similar_floats(self):
        """Test are_similar with floating point numbers."""
        self.assertTrue(CoreUtils.are_similar(0.1 + 0.2, 0.3, 0.0001))

    def test_are_similar_lists(self):
        """Test are_similar with list inputs."""
        self.assertTrue(CoreUtils.are_similar([1, 2], [1, 2], 0))
        self.assertTrue(CoreUtils.are_similar([1, 2], [2, 3], 1))

    def test_are_similar_tuples(self):
        """Test are_similar with tuple inputs."""
        self.assertTrue(CoreUtils.are_similar((1, 2, 3), (1, 2, 3), 0))

    def test_are_similar_non_numeric(self):
        """Test are_similar with non-numeric values (equality check)."""
        self.assertTrue(CoreUtils.are_similar("abc", "abc", 0))
        self.assertFalse(CoreUtils.are_similar("abc", "def", 0))

    # -------------------------------------------------------------------------
    # randomize Tests
    # -------------------------------------------------------------------------

    def test_randomize_full_ratio(self):
        """Test randomize with full ratio returns all elements."""
        result = CoreUtils.randomize(list(range(10)), 1.0)
        self.assertEqual(len(result), 10)
        for item in result:
            self.assertIn(item, range(10))

    def test_randomize_half_ratio(self):
        """Test randomize with half ratio returns approximately half."""
        result = CoreUtils.randomize(list(range(10)), 0.5)
        self.assertEqual(len(result), 5)
        for item in result:
            self.assertIn(item, range(10))

    def test_randomize_zero_ratio(self):
        """Test randomize with zero ratio returns empty list."""
        result = CoreUtils.randomize(list(range(10)), 0.0)
        self.assertEqual(result, [])

    def test_randomize_above_one_clamped(self):
        """Test randomize clamps ratio above 1.0."""
        result = CoreUtils.randomize(list(range(5)), 2.0)
        self.assertEqual(len(result), 5)

    def test_randomize_empty_list(self):
        """Test randomize with empty list."""
        result = CoreUtils.randomize([], 1.0)
        self.assertEqual(result, [])

    def test_randomize_single_element(self):
        """Test randomize with single element list."""
        result = CoreUtils.randomize([42], 1.0)
        self.assertEqual(result, [42])

    def test_randomize_preserves_uniqueness(self):
        """Test randomize doesn't duplicate elements."""
        original = list(range(10))
        result = CoreUtils.randomize(original, 1.0)
        self.assertEqual(len(result), len(set(result)))

    def test_randomize_is_actually_random(self):
        """Test that randomize produces different orders (statistical)."""
        original = list(range(20))
        results = [tuple(CoreUtils.randomize(original, 1.0)) for _ in range(10)]
        # At least some should be different (very unlikely all same)
        unique_results = set(results)
        self.assertGreater(len(unique_results), 1)


if __name__ == "__main__":
    unittest.main(exit=False)
