import logging
from typing import Callable, Dict, Optional, List, Any


class NamespaceHandler:
    """NamespaceHandler dynamically manages access to a namespace dictionary
    within an owner object. It provides attribute-style access to the
    namespace and resolves missing attributes using a resolver function."""

    def __init__(
        self,
        owner: Any,
        namespace_attr: str,
        resolver: Optional[Callable[[str], Any]] = None,
        log_level: str = "WARNING",
    ):
        self._logger = self.init_logger(log_level)
        self._logger.debug(f"Initializing NamespaceHandler for '{namespace_attr}'")

        if hasattr(owner, namespace_attr):
            raise AttributeError(
                f"'{owner.__class__.__name__}' already has an attribute '{namespace_attr}'"
            )
        super().__setattr__("_owner", owner)
        super().__setattr__("_namespace_attr", namespace_attr)
        super().__setattr__("_resolver", resolver)
        super().__setattr__("_namespace", {})
        setattr(owner, namespace_attr, self._namespace)
        self._logger.debug(f"NamespaceHandler initialized: {self._namespace}")

    def init_logger(self, log_level: str) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.NamespaceHandler")
        logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
        return logger

    def __getattr__(self, name: str) -> Any:
        self._logger.debug(f"__getattr__ called for {name}")
        namespace = self._get_namespace()
        if name in namespace:
            value = namespace[name]
            self._logger.debug(f"Found {name} in namespace: {value}")
            return value
        if self._resolver:
            self._logger.debug(f"{name} not found in namespace, calling resolver")
            resolved_value = self._resolver(name)
            self._logger.debug(f"Resolved {name}: {resolved_value}")
            if resolved_value is not None:
                namespace[name] = resolved_value
            return resolved_value
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any):
        if name in ("_owner", "_namespace_attr", "_resolver", "_namespace", "_logger"):
            super().__setattr__(name, value)
        else:
            self._logger.debug(f"__setattr__ called for {name} with value {value}")
            namespace = self._get_namespace()
            namespace[name] = value
            self._logger.debug(f"Set {name} in namespace: {self._namespace}")

    def __getitem__(self, name: str) -> Any:
        self._logger.debug(f"__getitem__ called for {name}")
        namespace = self._get_namespace()
        if name in namespace:
            value = namespace[name]
            self._logger.debug(f"Found {name} in namespace: {value}")
            return value
        if self._resolver:
            self._logger.debug(f"{name} not found in namespace, calling resolver")
            resolved_value = self._resolver(name)
            self._logger.debug(f"Resolved {name}: {resolved_value}")
            if resolved_value is not None:
                namespace[name] = resolved_value
            return resolved_value
        raise KeyError(name)

    def __setitem__(self, name: str, value: Any):
        self._logger.debug(f"__setitem__ called for {name} with value {value}")
        namespace = self._get_namespace()
        namespace[name] = value
        self._logger.debug(f"Set {name} in namespace: {self._namespace}")

    def items(self):
        self._logger.debug("items called")
        return self._namespace.items()

    def keys(self):
        self._logger.debug("keys called")
        return self._namespace.keys()

    def values(self):
        self._logger.debug("values called")
        return self._namespace.values()

    def _get_namespace(self) -> Dict[str, Any]:
        if self._namespace is None:
            try:
                self._namespace = getattr(self._owner, self._namespace_attr)
            except AttributeError:
                self._namespace = {}
                setattr(self._owner, self._namespace_attr, self._namespace)
        return self._namespace


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

        def test_namespace_handler_logging(self):
            class Owner:
                pass

            owner = Owner()
            resolver = lambda name: f"Resolved {name}"
            handler = NamespaceHandler(owner, "namespace", resolver, log_level="DEBUG")
            owner.namespace = {"existing": "value"}

            # Ensure the namespace has been properly initialized with the existing attribute
            handler._namespace = owner.namespace

            with self.assertLogs(f"{__name__}.NamespaceHandler", level="DEBUG") as cm:
                handler.new_attr

            self.assertIn(
                "DEBUG:__main__.NamespaceHandler:__getattr__ called for new_attr",
                cm.output[0],
            )
            self.assertIn(
                "DEBUG:__main__.NamespaceHandler:new_attr not found in namespace, calling resolver",
                cm.output[1],
            )
            self.assertIn(
                "DEBUG:__main__.NamespaceHandler:Resolved new_attr: Resolved new_attr",
                cm.output[2],
            )

    # Use TextTestRunner to run tests to prevent SystemExit issue.
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNamespaceHandler)
    runner = unittest.TextTestRunner()
    runner.run(suite)

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------
