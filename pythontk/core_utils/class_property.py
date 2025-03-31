# !/usr/bin/python
# coding=utf-8
from typing import Dict, Optional, Any


class ClassProperty:
    """A descriptor for class-level properties (replaces @classmethod @property).

    This decorator allows you to define a property that can be accessed directly on the class,
    rather than on an instance of the class. This is useful for defining properties that are
    related to the class itself rather than to a specific instance.

    Example:
        class MyClass:
            _value = 42

            @ClassProperty
            def value(cls):
                return cls._value

        print(MyClass.value)  # Output: 42
        MyClass._value = 100
    """

    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import unittest

    class TestClassProperty(unittest.TestCase):
        class MyClass:
            _value = 42

            @ClassProperty
            def value(cls):
                return cls._value

        def test_class_property(self):
            self.assertEqual(self.MyClass.value, 42)

        def test_class_property_setter(self):
            self.MyClass._value = 100
            self.assertEqual(self.MyClass.value, 100)

    unittest.main(exit=False)


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
