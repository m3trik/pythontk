from typing import Callable, Optional, Any
from pythontk.core_utils import LoggingMixin


class NamespaceHandler(LoggingMixin):
    """A NamespaceHandler that manages its own internal dictionary without attaching
    attributes directly to the owner object.

    Parameters:
        owner (Any): The owner object that the namespace is attached to.
        identifier (str, optional): An identifier for logging or tracking purposes.
        resolver (Callable[[str], Any], optional): A function that resolves attribute names.
        log_level (str, optional): The logging level for the logger. Defaults to "DEBUG".
    """

    def __init__(
        self,
        owner: Any,
        identifier: str = None,
        resolver: Optional[Callable[[str], Any]] = None,
        log_level: str = "WARNING",
    ):
        # Set logger level
        self.logger.setLevel(log_level)
        self.logger.debug(f"Initializing for '{identifier}'")

        # Store identifier for logging or tracking
        self.__dict__["_identifier"] = identifier

        # Assign attributes directly to __dict__ to avoid infinite recursion
        self.__dict__["_attributes"] = {}
        self.__dict__["_resolver"] = resolver
        self.__dict__["_owner"] = owner

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
        """
        Handles setting attributes safely without causing recursion.
        """
        if name.startswith("_"):
            self.__dict__[name] = value  # Directly set internal attributes
        else:
            self.__dict__["_attributes"][name] = value  # Store in attributes

    def __getitem__(self, key: str) -> Any:
        try:
            return self.__dict__["_attributes"][key]
        except KeyError:
            available = list(self.keys())
            err = f"Namespace '{self.__dict__.get('_identifier')}' has no key '{key}'. Available keys: {available}"
            self.logger.error(err)
            raise KeyError(err)

    def __setitem__(self, key: str, value: Any):
        self.logger.debug(
            f"[{self.__dict__.get('_identifier')}] __setitem__ called for '{key}' with value '{value}'"
        )
        self.__dict__["_attributes"][key] = value

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
