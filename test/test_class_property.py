#!/usr/bin/python
# coding=utf-8
"""
Unit tests for pythontk ClassProperty.

Run with:
    python -m pytest test_class_property.py -v
    python test_class_property.py
"""
import unittest

from pythontk.core_utils.class_property import ClassProperty

from conftest import BaseTestCase


class ClassPropertyTest(BaseTestCase):
    """ClassProperty test class."""

    def test_class_property_access_on_class(self):
        """Test that ClassProperty can be accessed on the class directly."""

        class MyClass:
            _value = 42

            @ClassProperty
            def value(cls):
                return cls._value

        self.assertEqual(MyClass.value, 42)

    def test_class_property_access_on_instance(self):
        """Test that ClassProperty can also be accessed on an instance."""

        class MyClass:
            _value = 42

            @ClassProperty
            def value(cls):
                return cls._value

        instance = MyClass()
        self.assertEqual(instance.value, 42)

    def test_class_property_reflects_class_changes(self):
        """Test that ClassProperty reflects changes to underlying class attribute."""

        class MyClass:
            _value = 42

            @ClassProperty
            def value(cls):
                return cls._value

        self.assertEqual(MyClass.value, 42)
        MyClass._value = 100
        self.assertEqual(MyClass.value, 100)

    def test_class_property_with_computation(self):
        """Test ClassProperty with computed values."""

        class MyClass:
            _items = [1, 2, 3, 4, 5]

            @ClassProperty
            def count(cls):
                return len(cls._items)

            @ClassProperty
            def total(cls):
                return sum(cls._items)

        self.assertEqual(MyClass.count, 5)
        self.assertEqual(MyClass.total, 15)

        MyClass._items.append(10)
        self.assertEqual(MyClass.count, 6)
        self.assertEqual(MyClass.total, 25)

    def test_class_property_inheritance(self):
        """Test that ClassProperty works with inheritance."""

        class Parent:
            _name = "Parent"

            @ClassProperty
            def name(cls):
                return cls._name

        class Child(Parent):
            _name = "Child"

        self.assertEqual(Parent.name, "Parent")
        self.assertEqual(Child.name, "Child")

    def test_class_property_with_class_method_behavior(self):
        """Test that ClassProperty receives the class as first argument."""

        class MyClass:
            _data = {"key": "value"}

            @ClassProperty
            def keys(cls):
                return list(cls._data.keys())

        self.assertEqual(MyClass.keys, ["key"])

    def test_class_property_different_from_instance_property(self):
        """Test that ClassProperty differs from regular property behavior."""

        class MyClass:
            _class_value = "class"

            @ClassProperty
            def class_prop(cls):
                return cls._class_value

            def __init__(self):
                self._instance_value = "instance"

            @property
            def instance_prop(self):
                return self._instance_value

        # Class property accessible on class
        self.assertEqual(MyClass.class_prop, "class")

        # Instance property requires instance
        instance = MyClass()
        self.assertEqual(instance.instance_prop, "instance")

        # Class property also accessible on instance
        self.assertEqual(instance.class_prop, "class")


if __name__ == "__main__":
    unittest.main(exit=False)
