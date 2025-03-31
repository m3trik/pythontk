# !/usr/bin/python
# coding=utf-8
from typing import Dict, Optional, Any


class SingletonMixin:
    """A mixin class that provides singleton behavior.

    This mixin ensures that only one instance of a class is created. If an instance already exists,
    the existing instance is returned. This is useful for classes that should only have a single
    instance throughout the application, such as configuration managers or resource loaders.

    Example:
        class MyClass(SingletonMixin):
            def __init__(self, value):
                self.value = value

        MyClass.instance(42)  # Creates an instance with value 42
        instance1 = MyClass.instance()  # Returns the same instance
        instance2 = MyClass.instance()
        assert instance1 is instance2  # True, both are the same instance
    """

    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def has_instance(cls):
        return cls in cls._instances

    @classmethod
    def reset_instance(cls):
        if cls in cls._instances:
            del cls._instances[cls]


# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import unittest

    class TestSingletonMixin(unittest.TestCase):
        class TestClass(SingletonMixin):
            pass

        def test_singleton(self):
            instance1 = self.TestClass()
            instance2 = self.TestClass()
            self.assertIs(instance1, instance2)

        def test_reset_instance(self):
            instance1 = self.TestClass()
            self.TestClass.reset_instance()
            instance2 = self.TestClass()
            self.assertIsNot(instance1, instance2)
            self.assertIs(instance1, self.TestClass._instances[self.TestClass])

    unittest.main(exit=False)


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
