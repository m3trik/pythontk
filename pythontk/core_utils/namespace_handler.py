import weakref
from typing import Callable, Optional, Any
from pythontk.core_utils import LoggingMixin


class NamespaceHandler(LoggingMixin):
    """A NamespaceHandler that manages its own internal dictionary without attaching
    attributes directly to the owner object.

    Parameters:
        owner (Any): The owner object that the namespace is attached to.
        identifier (str, optional): An identifier for logging or tracking purposes.
        resolver (Callable[[str], Any], optional): A function that resolves attribute names.
        log_level (str, optional): The logging level for the logger.

    Example:
        def resolver(name: str) -> Any:
            # Custom resolver logic here
            return f"Resolved {name}" if name != "unknown_attr" else None

        handler = NamespaceHandler(owner=my_object, identifier="example", resolver=resolver)
        print(handler.some_attr)  # Calls the resolver if not cached
        handler.some_attr = "New Value"  # Sets the attribute directly
    """

    def __init__(
        self,
        owner: Any,
        identifier: str = None,
        resolver: Optional[Callable[[str], Any]] = None,
        use_weakref: bool = False,
        log_level: str = "WARNING",
    ):
        self.logger.setLevel(log_level)
        self.__dict__["_identifier"] = identifier
        self.__dict__["_resolver"] = resolver
        self.__dict__["_owner"] = owner
        self.__dict__["_use_weakref"] = use_weakref
        if use_weakref:
            self.__dict__["_attributes"] = weakref.WeakValueDictionary()
        else:
            self.__dict__["_attributes"] = {}

    def __contains__(self, key: str) -> bool:
        """Explicit containment check for NamespaceHandler."""
        return key in self.__dict__["_attributes"]

    def __getattr__(self, name: str) -> Any:
        """Handles dynamic attribute resolution with recursion prevention."""
        self.logger.debug(
            f"[{self.__dict__.get('_identifier')}] __getattr__ called for '{name}'"
        )

        # Prevent recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            )

        attributes = self.__dict__.get("_attributes", {})

        # Return if cached
        if name in attributes:
            self.logger.debug(
                f"[{self.__dict__.get('_identifier')}] Returning cached '{name}'"
            )
            return attributes[name]

        # Attempt resolution
        if self.__dict__["_resolver"]:
            self.logger.debug(
                f"[{self.__dict__.get('_identifier')}] Attempting to resolve '{name}' via resolver..."
            )
            resolved_value = self.__dict__["_resolver"](name)
            if resolved_value is not None:
                attributes[name] = resolved_value  # Cache successful resolution
                self.logger.debug(
                    f"[{self.__dict__.get('_identifier')}] Resolved and cached '{name}'"
                )
                return resolved_value

        self.logger.debug(
            f"[{self.__dict__.get('_identifier')}] Attribute '{name}' not found."
        )
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            self.__dict__[name] = value
        else:
            if self.__dict__.get("_use_weakref", False):
                try:
                    self.__dict__["_attributes"][name] = value
                except TypeError:
                    # Not weakref-able; store as strong reference
                    self.logger.debug(
                        f"{type(value)} is not weakref-able, storing strong ref."
                    )
                    # WeakValueDictionary stores refs in .data attribute
                    self.__dict__["_attributes"].data[name] = value
            else:
                self.__dict__["_attributes"][name] = value

    def __getitem__(self, key: str) -> Any:
        try:
            return self.__dict__["_attributes"][key]
        except KeyError:
            available = list(self.keys())
            self.logger.debug(
                f"Namespace '{self.__dict__.get('_identifier')}' has no key '{key}'.\n\tAvailable keys: {available}"
            )
            raise

    def __setitem__(self, key: str, value: Any):
        self.__setattr__(key, value)

    def setdefault(self, name: str, default: Any = None) -> Any:
        """
        Optional helper method to demonstrate setdefault usage.
        """
        return self.__dict__["_attributes"].setdefault(name, default)

    def items(self):
        return self.__dict__["_attributes"].items()

    def keys(self):
        return self.__dict__["_attributes"].keys()

    def values(self):
        return self.__dict__["_attributes"].values()


# --------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import unittest

    class TestNamespaceHandler(unittest.TestCase):
        def setUp(self):
            class Owner:
                pass

            def test_resolver(name):
                return f"Resolved {name}" if name != "unknown_attr" else None

            self.owner = Owner()
            self.handler = NamespaceHandler(
                owner=self.owner, identifier="example_namespace", resolver=test_resolver
            )

        def test_identifier_logging(self):
            """
            Ensures the identifier is stored and used in debug logs.
            """
            self.assertEqual(self.handler._identifier, "example_namespace")

        def test_existing_attribute(self):
            self.handler._attributes["existing"] = "value"
            self.assertEqual(self.handler.existing, "value")
            self.assertEqual(self.handler["existing"], "value")

        def test_resolver_success(self):
            self.assertEqual(self.handler.new_attr, "Resolved new_attr")

        def test_resolver_failure(self):
            with self.assertRaises(AttributeError) as context:
                _ = self.handler.unknown_attr

            self.assertIn(
                "NamespaceHandler object has no attribute 'unknown_attr'",
                str(context.exception),
            )

        def test_set_and_get(self):
            self.handler.dynamic_attr = "Dynamic Value"
            self.assertEqual(self.handler.dynamic_attr, "Dynamic Value")

        def test_dict_access(self):
            self.handler["key"] = "Dict Value"
            self.assertEqual(self.handler["key"], "Dict Value")

        def test_items_keys_values(self):
            self.handler._attributes["a"] = 1
            self.handler._attributes["b"] = 2
            self.assertEqual(list(self.handler.items()), [("a", 1), ("b", 2)])
            self.assertEqual(list(self.handler.keys()), ["a", "b"])
            self.assertEqual(list(self.handler.values()), [1, 2])

    # Run the tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNamespaceHandler)
    runner = unittest.TextTestRunner()
    runner.run(suite)


# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
