import unittest
from pythontk.core_utils.namespace_handler import NamespaceHandler


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
            "NamespaceHandler has no attribute 'unknown_attr'",
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
        self.assertCountEqual(list(self.handler.items()), [("a", 1), ("b", 2)])
        self.assertCountEqual(list(self.handler.keys()), ["a", "b"])
        self.assertCountEqual(list(self.handler.values()), [1, 2])


if __name__ == "__main__":
    # Run the tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNamespaceHandler)
    runner = unittest.TextTestRunner()
    runner.run(suite)
