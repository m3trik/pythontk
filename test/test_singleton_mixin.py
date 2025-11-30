#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk SingletonMixin.

Run with:
    python -m pytest test_singleton_mixin.py -v
    python test_singleton_mixin.py
"""
import unittest

from pythontk.core_utils.singleton_mixin import SingletonMixin

from conftest import BaseTestCase


class SingletonMixinTest(BaseTestCase):
    """SingletonMixin test class."""

    def setUp(self):
        """Reset singleton instances before each test."""
        # Create fresh test classes for each test to avoid cross-contamination
        pass

    def tearDown(self):
        """Clean up singleton instances after each test."""
        # Reset any instances that may have been created
        SingletonMixin._instances.clear()

    def test_singleton_returns_same_instance(self):
        """Test that multiple instantiations return the same object."""

        class MySingleton(SingletonMixin):
            pass

        instance1 = MySingleton()
        instance2 = MySingleton()
        self.assertIs(instance1, instance2)

    def test_singleton_with_key_creates_separate_instances(self):
        """Test that different keys create different instances."""

        class KeyedSingleton(SingletonMixin):
            pass

        instance_a = KeyedSingleton(singleton_key="a")
        instance_b = KeyedSingleton(singleton_key="b")
        instance_a2 = KeyedSingleton(singleton_key="a")

        self.assertIsNot(instance_a, instance_b)
        self.assertIs(instance_a, instance_a2)

    def test_singleton_init_called_once(self):
        """Test that __init__ is only called once per singleton."""

        class CountingInit(SingletonMixin):
            init_count = 0

            def __init__(self):
                CountingInit.init_count += 1

        CountingInit()
        CountingInit()
        CountingInit()

        self.assertEqual(CountingInit.init_count, 1)

    def test_singleton_init_with_args(self):
        """Test that __init__ args are used only on first instantiation."""

        class ConfiguredSingleton(SingletonMixin):
            def __init__(self, value=None):
                self.value = value

        instance1 = ConfiguredSingleton(value="first")
        instance2 = ConfiguredSingleton(value="second")

        self.assertEqual(instance1.value, "first")
        self.assertEqual(instance2.value, "first")  # Same instance, first value kept

    def test_reset_instance_allows_new_creation(self):
        """Test that reset_instance allows creating a new instance."""

        class ResettableSingleton(SingletonMixin):
            def __init__(self, value=None):
                self.value = value

        instance1 = ResettableSingleton(value="first")
        ResettableSingleton.reset_instance()
        instance2 = ResettableSingleton(value="second")

        self.assertIsNot(instance1, instance2)
        self.assertEqual(instance1.value, "first")
        self.assertEqual(instance2.value, "second")

    def test_reset_instance_with_key(self):
        """Test that reset_instance works with keyed singletons."""

        class KeyedResettable(SingletonMixin):
            def __init__(self, value=None, **kwargs):
                # Accept **kwargs to handle singleton_key passthrough
                self.value = value

        instance_a = KeyedResettable(singleton_key="a", value="a-first")
        instance_b = KeyedResettable(singleton_key="b", value="b-first")

        KeyedResettable.reset_instance(singleton_key="a")

        instance_a2 = KeyedResettable(singleton_key="a", value="a-second")
        instance_b2 = KeyedResettable(singleton_key="b", value="b-attempt")

        self.assertIsNot(instance_a, instance_a2)
        self.assertIs(instance_b, instance_b2)
        self.assertEqual(instance_a2.value, "a-second")
        self.assertEqual(instance_b2.value, "b-first")  # Not reset, keeps first

    def test_has_instance(self):
        """Test has_instance returns correct boolean."""

        class CheckableSingleton(SingletonMixin):
            pass

        self.assertFalse(CheckableSingleton.has_instance())

        CheckableSingleton()

        self.assertTrue(CheckableSingleton.has_instance())

    def test_has_instance_with_key(self):
        """Test has_instance works with keyed singletons."""

        class KeyedCheckable(SingletonMixin):
            pass

        self.assertFalse(KeyedCheckable.has_instance(singleton_key="mykey"))

        KeyedCheckable(singleton_key="mykey")

        self.assertTrue(KeyedCheckable.has_instance(singleton_key="mykey"))
        self.assertFalse(KeyedCheckable.has_instance(singleton_key="otherkey"))

    def test_instance_class_method(self):
        """Test the instance() class method works like direct instantiation."""

        class MethodSingleton(SingletonMixin):
            pass

        instance1 = MethodSingleton.instance()
        instance2 = MethodSingleton()
        instance3 = MethodSingleton.instance()

        self.assertIs(instance1, instance2)
        self.assertIs(instance2, instance3)

    def test_subclasses_have_separate_instances(self):
        """Test that subclasses maintain their own singleton instances."""

        class BaseSingleton(SingletonMixin):
            pass

        class ChildA(BaseSingleton):
            pass

        class ChildB(BaseSingleton):
            pass

        base_instance = BaseSingleton()
        child_a = ChildA()
        child_b = ChildB()

        self.assertIsNot(base_instance, child_a)
        self.assertIsNot(base_instance, child_b)
        self.assertIsNot(child_a, child_b)

    def test_singleton_with_inheritance_and_init(self):
        """Test that inheritance works correctly with custom __init__."""

        class ParentSingleton(SingletonMixin):
            def __init__(self, name=None):
                self.name = name

        class ChildSingleton(ParentSingleton):
            def __init__(self, name=None, extra=None):
                super().__init__(name)
                self.extra = extra

        child = ChildSingleton(name="child", extra="data")
        child2 = ChildSingleton(name="ignored", extra="ignored")

        self.assertIs(child, child2)
        self.assertEqual(child.name, "child")
        self.assertEqual(child.extra, "data")


if __name__ == "__main__":
    unittest.main(exit=False)
