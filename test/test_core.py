#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk CoreUtils.

Run with:
    python -m pytest test_core.py -v
    python test_core.py
"""
import unittest

from pythontk import CoreUtils

from conftest import BaseTestCase


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


if __name__ == "__main__":
    unittest.main(exit=False)
