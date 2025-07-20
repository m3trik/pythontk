# !/usr/bin/python
# coding=utf-8
from typing import Dict, Optional, Any


class SingletonMixin:
    """Reusable singleton mixin that supports optional key-based instances."""

    _instances: Dict[Any, Any] = {}

    def __init__(self, *args, **kwargs):
        """Prevent object.__init__() from being called with arguments."""
        pass

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        key: Any = kwargs.pop("singleton_key", cls)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    @classmethod
    def instance(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(*args, **kwargs)

    @classmethod
    def has_instance(cls, singleton_key: Optional[Any] = None) -> bool:
        return (
            singleton_key in cls._instances if singleton_key else cls in cls._instances
        )

    @classmethod
    def reset_instance(cls, singleton_key: Optional[Any] = None) -> None:
        cls._instances.pop(singleton_key or cls, None)


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
