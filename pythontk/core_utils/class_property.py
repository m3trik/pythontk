from typing import Dict, Optional, Any


class ClassProperty:
    """A descriptor for class-level properties (replaces @classmethod @property)."""

    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import unittest

    class TestNamespaceHandler(unittest.TestCase):
        def test_class_property(self):
            class TestClass:
                _instance: Optional[Dict[str, Any]] = None

                @ClassProperty
                def instance(cls) -> Dict[str, Any]:
                    if cls._instance is None:
                        cls._instance = {}
                    return cls._instance

            self.assertEqual(TestClass.instance, {})
            TestClass.instance["key"] = "value"
            self.assertEqual(TestClass.instance, {"key": "value"})

    unittest.main(exit=False)


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
