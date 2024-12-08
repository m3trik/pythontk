from typing import Callable, Dict, Optional, Any
from pythontk.core_utils import LoggingMixin


class NamespaceHandler:
    """A NamespaceHandler that manages its own internal dictionary without
    attaching attributes directly to the owner object."""

    def __init__(
        self,
        owner: Any,
        namespace_attr: str,
        resolver: Optional[Callable[[str], Any]] = None,
    ):
        print(f"[NamespaceHandler] Initializing for '{namespace_attr}'")

        # Store reference attributes internally without triggering owner setattr/getattr
        self.__dict__["_namespace"] = {}
        self.__dict__["_resolver"] = resolver
        self.__dict__["_owner"] = owner

        print(f"[NamespaceHandler] Initialized with internal namespace.")

    def __getattr__(self, name: str) -> Any:
        print(f"[NamespaceHandler] __getattr__ called for '{name}'")

        # Access the internal namespace directly to avoid owner interaction
        if name in self._namespace:
            return self._namespace[name]

        if self._resolver:
            resolved_value = self._resolver(name)
            if resolved_value is not None:
                self._namespace[name] = resolved_value
                return resolved_value

        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            # Handle internal attributes directly
            self.__dict__[name] = value
        else:
            # Set value in the internal namespace dictionary
            self._namespace[name] = value

    def __getitem__(self, key: str) -> Any:
        print(f"[NamespaceHandler] __getitem__ called for '{key}'")
        return self._namespace[key]

    def __setitem__(self, key: str, value: Any):
        print(f"[NamespaceHandler] __setitem__ called for '{key}' with value '{value}'")
        self._namespace[key] = value

    def items(self):
        return self._namespace.items()

    def keys(self):
        return self._namespace.keys()

    def values(self):
        return self._namespace.values()


# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import unittest

    class TestNamespaceHandler(unittest.TestCase):
        def test_namespace_handler(self):
            class Owner:
                pass

            owner = Owner()
            resolver = lambda name: f"Resolved {name}"
            handler = NamespaceHandler(owner, "namespace", resolver)
            owner.namespace = {"existing": "value"}

            # Ensure the namespace has been properly initialized with the existing attribute
            handler._namespace = owner.namespace

            self.assertEqual(handler.existing, "value")
            self.assertEqual(handler["existing"], "value")
            self.assertEqual(list(handler.items()), [("existing", "value")])
            self.assertEqual(list(handler.keys()), ["existing"])
            self.assertEqual(list(handler.values()), ["value"])

            # Test resolving a new attribute
            self.assertEqual(handler.new_attr, "Resolved new_attr")

    # Use TextTestRunner to run tests to prevent SystemExit issue.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNamespaceHandler)
    runner = unittest.TextTestRunner()
    runner.run(suite)

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
